from robomimic.config.base_config import BaseConfig

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `robomimic/algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters # TODO
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.weight_decay = 1e-3  # weight decay
        self.algo.optim_params.policy.betas = (0.9, 0.95)  # betas for Adam
        self.algo.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []  # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep"  # learning rate scheduler ("multistep", "linear", etc)

        # general diffusion parameters # TODO (horizon, obs_as_cond, pred_action_steps_only, etc)

        # Basic Network architecture
        ## UNet parameters
        self.algo.unet.enabled = False
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        ## Transformer parameters
        self.algo.transformer.enabled = True
        self.algo.transformer.num_layers = 8
        self.algo.transformer.num_heads = 4
        self.algo.transformer.embed_dim = 256
        self.algo.transformer.p_drop_embed = 0.0
        self.algo.transformer.p_drop_attn = 0.3
        self.algo.transformer.causal_attn = True
        self.algo.transformer.time_as_condition = True # if false, use BERT like encoder only arch, time as input
        self.algo.n_cond_layers = 0 # >0: use transformer encoder for cond, otherwise use MLP


        # EMA
        self.algo.ema.enabled = True
        self.algo.ema.update_after_step = 0
        self.algo.ema.inv_gamma = 1.0
        self.algo.ema.power = 0.75
        self.algo.ema.min_value = 0.0
        self.algo.ema.max_value = 0.9999

        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_start = 0.0001
        self.algo.ddpm.beta_end = 0.02
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.variance_type = 'fixed_small'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'
        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 100
        self.algo.ddim.beta_start = 0.0001
        self.algo.ddim.beta_end = 0.02
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'