from abc import ABC, abstractmethod

from collections import OrderedDict, deque
import copy

import torch
import torch.nn as nn
import torch.nn.functional as f

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

from robomimic.config.config import Config

import robomimic.models.obs_nets as obs_nets
import robomimic.utils.tensor_utils as tensor_utils
import robomimic.utils.torch_utils as torch_utils
import robomimic.utils.obs_utils as obs_utils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

from robomimic_ext.models.transformer import ConditionalTransformerForDiffusion
from robomimic_ext.models.u_net import ConditionalUnet1DForDiffusion
from robomimic_ext.utils.net_utils import replace_bn_with_gn


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if ("unet" in algo_config) and algo_config.unet.enabled:
        return DiffusionUnetPolicy, {}
    elif ("transformer" in algo_config) and algo_config.transformer.enabled:
        return DiffusionTransformerPolicy, {}
    else:
        raise RuntimeError()


class DiffusionPolicyBase(ABC, PolicyAlgo):
    """
    Base for a diffusion policy algorithm.
    """
    def __init__(
        self,
        algo_config: Config,
        obs_config: Config,
        global_config: Config,
        obs_key_shapes: OrderedDict,
        ac_dim: int,
        device: torch.device
    ):
        """
        Policy algorithm that uses a UNet for diffusion.

        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        super().__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

        # check the horizon parameters
        ## example
        ## To = 3, Tp = 3, Ta = 2
        ## => minimum stack_frame = To + Tp - 1 = 5 and check that Ta <= Tp
        ## |-|-|-|-|-| <= this is the minimal stack_frame
        ## |o|o|o|
        ##     |p|p|p|
        ##     |a|a|
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        Ta = self.algo_config.horizon.action_horizon
        assert To + Tp - 1 <= self.global_config.train.frame_stack, f"Frame stack must be at least {To + Tp - 1}"
        assert Ta <= Tp, f"Action horizon must be less than or equal to prediction horizon"

        self.action_queue = None # queue of actions to execute

    def _create_networks(self) -> None:
        """
        Create the networks for the policy algorithm -> @self.nets
        """
        # set up observation encoder
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = obs_utils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = obs_nets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        obs_encoder = replace_bn_with_gn(obs_encoder)  # replace all BatchNorm with GroupNorm to work with EMA
        obs_dim = obs_encoder.output_shape()[0]

        # create noise prediction network
        noise_pred_net = self._create_noise_pred_net(obs_dim)

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().to(self.device)

        # setup noise scheduler
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_start=self.algo_config.ddpm.beta_start,
                beta_end=self.algo_config.ddpm.beta_end,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                variance_type=self.algo_config.ddpm.variance_type,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
            num_train_timesteps = self.algo_config.ddpm.num_train_timesteps
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
            prediction_type = self.algo_config.ddpm.prediction_type
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_start=self.algo_config.ddpm.beta_start,
                beta_end=self.algo_config.ddpm.beta_end,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
            num_train_timesteps = self.algo_config.ddim.num_train_timesteps
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
            prediction_type = self.algo_config.ddim.prediction_type
        else:
            raise RuntimeError()

        # setup EMA
        ema_model = None
        if self.algo_config.ema.enabled:
            ema_model = EMAModel(
                parameters=nets.parameters(),
                inv_gamma=self.algo_config.ema.inv_gamma,
                power=self.algo_config.ema.power,
                min_value=self.algo_config.ema.min_value,
                max_value=self.algo_config.ema.max_value
            )

        # store attributes
        self.nets = nets
        self._shadow_nets = copy.deepcopy(self.nets).eval()
        self._shadow_nets.requires_grad_(False)

        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type

        self.ema_model = ema_model

    @abstractmethod
    def _create_noise_pred_net(self, encoded_obs_dim: int) -> nn.Module:
        """
        Create the noise prediction network.
        :param encoded_obs_dim: dimension of the encoded observation (@self.obs_encoder)
        :return: noise prediction network (nn.Module)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]} # only use first To observations
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present

        # the policy should learn to predict the next Tp actions
        # starting in the current time_step
        start = To - 1
        end = start + Tp
        input_batch["actions"] = batch["actions"][:, start:end, :]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return tensor_utils.to_device(tensor_utils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]

        with torch_utils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyBase, self).train_on_batch(batch, epoch, validate=validate)

            # extract actions
            actions = batch['actions']

            # extract and encode observations
            inputs = {
                'obs': batch["obs"],
                'goal': batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k]), \
                f"Expected {len(self.obs_shapes[k])} dimensions for {k}, got {inputs['obs'][k].ndim - 2}"
            obs_features = tensor_utils.time_distributed(
                inputs,
                self.nets['policy']['obs_encoder'],
                inputs_as_kwargs=True
            )
            assert obs_features.ndim == 3  # [B, To, obs_feature_dim]
            
            # forward diffusion process
            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                low=0,
                high=self.num_train_timesteps,
                size=(B,),
                device=self.device
            ).long()

            # noinspection PyTypeChecker
            noisy_actions = self.noise_scheduler.add_noise(
                original_samples=actions,
                noise=noise,
                timesteps=timesteps,
            )

            # predict the noise residual
            pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, obs_features
            )

            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = noisy_actions
            else:
                raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

            # L2 loss
            loss = f.mse_loss(pred, target)

            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = tensor_utils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = torch_utils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                # update Exponential Moving Average of the model weights
                if self.ema_model is not None:
                    self.ema_model.step(self.nets.parameters())

                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyBase, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queue
        Ta = self.algo_config.horizon.action_horizon
        action_queue = deque(maxlen=Ta)
        self.action_queue = action_queue

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, obs_dim]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, action_dim]
        """
        if len(self.action_queue) == 0:
            # no actions left, run inference and put actions into the queue

            To = self.algo_config.horizon.observation_horizon
            # incoming observation sequence has entries of shape (1, stack_frame, obs_dim)
            # => the last 'stack_frame' observations are included
            # => we only need the last 'To' observations
            # => the last observation is the current observation
            obs_dict = {k: obs_dict[k][:, -To:, :] for k in obs_dict}  # only use last 'To' observations

            # output_shape (1,Ta,action_dim)
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)
            self.action_queue.extend(action_sequence[0]) # add action sequence of length Ta to the queue

        # has action, execute from left to right => output has shape action_dim
        action = self.action_queue.popleft() # one dimensional action tensor

        # => add batch dimension => shape (1, action_dim)
        action = action.unsqueeze(0)
        return action

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim

        # select network
        nets = self.nets
        if self.ema_model is not None:
            self.ema_model.copy_to(parameters=self._shadow_nets.parameters())
            nets = self._shadow_nets

        # extract and encode observations
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }

        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k]), \
                f"Expected {len(self.obs_shapes[k])} dimensions for {k}, got {inputs['obs'][k].ndim - 2}"

        obs_features = tensor_utils.time_distributed(
            inputs,
            nets['policy']['obs_encoder'],
            inputs_as_kwargs=True
        )
        assert obs_features.ndim == 3  # (batch, horizon, dim)
        B = obs_features.shape[0]

        # initialize action from Gaussian noise
        prediction = torch.randn(
            (B, Tp, action_dim), device=self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=prediction,
                timestep=k,
                cond=obs_features
            )

            # inverse diffusion step (remove noise)
            prediction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=prediction
            ).prev_sample

            # only return the next 'Ta' action predictions
            action_seq = prediction[:, :Ta, :] # shape (B, Ta, action_dim)
            return action_seq


    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema_model.state_dict() if self.ema_model is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema_model.load_state_dict(model_dict["ema"])

#####################################################################################
# Explicit implementations of the DiffusionPolicyUNet class
#####################################################################################

class DiffusionUnetPolicy(DiffusionPolicyBase):
    """
    Policy algorithm that uses a conditional UNet for diffusion.
    """

    def _create_noise_pred_net(self, encoded_obs_dim: int) -> nn.Module:
        """
        Create the noise prediction network.
        :param encoded_obs_dim: dimension of the encoded observation (@self.obs_encoder)
        :return: noise prediction network (nn.Module)
        """
        unet = ConditionalUnet1DForDiffusion(
            input_dim=self.ac_dim,
            cond_dim=encoded_obs_dim,
            cond_horizon=self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups
        )
        return unet

class DiffusionTransformerPolicy(DiffusionPolicyBase):
    """
    Policy algorithm that uses a transformer for diffusion.
    """
    def _create_noise_pred_net(self, encoded_obs_dim: int) -> nn.Module:
        """
        Create the noise prediction network.
        :param encoded_obs_dim: dimension of the encoded observation (@self.obs_encoder)
        :return: noise prediction network (nn.Module)
        """
        transformer = ConditionalTransformerForDiffusion(
            input_dim=self.ac_dim,
            cond_dim=encoded_obs_dim,
            cond_horizon=self.algo_config.horizon.observation_horizon,
            num_layers=self.algo_config.transformer.num_layers,
            num_heads=self.algo_config.transformer.num_heads,
            embed_dim=self.algo_config.transformer.embed_dim,
            p_drop_embed=self.algo_config.transformer.p_drop_embed,
            p_drop_attn=self.algo_config.transformer.p_drop_attn,
            causal_attn=self.algo_config.transformer.causal_attn,
            n_cond_layers=self.algo_config.transformer.n_cond_layers
        )
        return transformer

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        super()._create_optimizers()
        # TODO replace 'super()._create_optimizers()' with the following code:
        ## 1) LowDim
        ## 1a) https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/transformer_for_diffusion.py#L197
        ## 1b) https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/transformer_for_diffusion.py#L197
        ## 1c) https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py#L169
        ## 2) With Images:
        ## 2a) https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py#L296
