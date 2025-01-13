# TODO clean up this file, to only predict action

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from src.imitation_learning.models.normalizer import LinearNormalizer
from .policy import PolicyBase
from src.imitation_learning.models.transformer_for_diffusion import TransformerForDiffusion
from src.imitation_learning.models.mask_generator import MaskGenerator

class DiffusionTransformerPolicy(PolicyBase):
    def __init__(
        self,
        model: TransformerForDiffusion, # TODO remove this
        noise_scheduler: DDPMScheduler, # TODO remove this
        horizon,
        obs_dim,
        action_dim,
        n_action_steps=1,
        n_obs_steps=1,
        num_inference_steps=None,
        obs_as_cond=True,
        pred_action_steps_only=True,
        # parameters passed to step
        **kwargs
    ) -> None:
        super().__init__(use_normalizer=True)
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = TransformerForDiffusion(
            obs_dim, action_dim,
            
        )


        self.noise_scheduler = noise_scheduler
        self.mask_generator = MaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config["num_train_timesteps"]
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data, condition_mask,
        cond=None, generator=None,
        # keyword arguments to scheduler.step
        **kwargs
        ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
        result: action
        """

        nobs = self.normalizer(obs_dict)
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.model.device
        dtype = self.model.dtype

        # handle different ways of passing observation
        cond = None
        if self.obs_as_cond:
            cond = nobs[:, :To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # un-normalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        #TODO not compatible with new normalizer
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:, :self.n_obs_steps, :]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config["num_train_timesteps"],
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config["prediction_type"]
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def train_on_batch(self, batch) -> float:
        # TODO
        pass

    def eval_on_batch(self, batch) -> float:
        # TODO
        pass

    def save(self, path):
        # TODO
        pass

    def load(self, path):
        # TODO
        pass