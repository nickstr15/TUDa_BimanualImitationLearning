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
        # specific optimization parameters for the diffusion architecture
        ## noise_pred_net
        self.algo.optim_params.noise_pred_net.optimizer.optimizer_type = "adamw"      # optimizer type ("adam", "adamw")
        self.algo.optim_params.noise_pred_net.optimizer.weight_decay = 1e-3           # L2 regularization strength
        self.algo.optim_params.noise_pred_net.optimizer.learning_rate.initial = 1e-4  # policy learning rate
        self.algo.optim.params.noise_pred_net.optimizer.betas = (0.9, 0.95)           # betas for Adam(W) optimizer
        self.algo.optim.params.noise_pred_net.lr_scheduler.type = "cosine"            # learning rate scheduler ("multistep", "linear", "cosine")
        self.algo.optim.params.noise_pred_net.lr_scheduler.warmup_steps = 1000        # number of warmup steps for learning rate
        self.algo.optim_params.noise_pred_net.lr_scheduler.decay_factor = 0.1         # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.noise_pred_net.lr_scheduler.epoch_schedule = []        # epochs where LR decay occurs (for multistep and linear scheduler)
        ## observation encoder
        self.algo.optim_params.obs_encoder.optimizer.optimizer_type = "adamw"      # optimizer type ("adam", "adamw")
        self.algo.optim_params.obs_encoder.optimizer.weight_decay = 1e-6           # L2 regularization strength
        self.algo.optim_params.obs_encoder.optimizer.learning_rate.initial = 1e-4  # policy learning rate
        self.algo.optim.params.obs_encoder.optimizer.betas = (0.9, 0.95)           # betas for Adam(W) optimizer
        self.algo.optim.params.obs_encoder.lr_scheduler.type = "cosine"            # learning rate scheduler ("multistep", "linear", "cosine")
        self.algo.optim.params.obs_encoder.lr_scheduler.warmup_steps = 1000        # number of warmup steps for learning rate
        self.algo.optim_params.obs_encoder.lr_scheduler.decay_factor = 0.1         # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.obs_encoder.lr_scheduler.epoch_schedule = []        # epochs where LR decay occurs (for multistep and linear scheduler)

        # horizon parameters (To, Tp, Ta)
        ## length of observation sequence used to predict the next action sequence, short: To
        self.algo.horizon.observation_horizon = 2
        ## length of action sequence predicted by the model based on the observation sequence, short Tp
        self.algo.horizon.prediction_horizon = 4
        ## length of action sequence used to interact with the environment, short Ta
        ## must be less than or equal to prediction horizon
        ## these number of actions are executed before new action sequence of length Tp is predicted
        self.algo.horizon.action_horizon = 1
        ## example
        ### To = 3, Tp = 3, Ta = 2
        ### (o) |o|o|o|        # incoming observation sequence
        ### (p)     |p|p|p|    # policy predicts this action sequence
        ### (a)     |a|a|      # these actions are executed before new action sequence is predicted
        ### ! the data of (o) and (p) is needed during training, so the frame_stack (@self.train.frame_stack) parameter
        ###   must be larger or equal to frame_stack >= (To + Tp - 1)
        ### (a) is only used during inference, the actions in (a)
        ###   correspond to the respective predictions in (p)
        ### example with three prediction steps
        ### (t) |1|2|3|4|5|6|7|8|...
        ### ----|-------------------------------------------
        ### (o) |o|o|o|
        ### (p)     |p|p|p|
        ### (a)     |a|a|
        ### ------------|-----------------------------------
        ### (o)         |o|o|o|
        ### (p)             |p|p|p|
        ### (a)             |a|a|
        ### --------------------|---------------------------
        ### (o)                 |o|o|o|
        ### (p)                     |p|p|p|
        ### (a)                     |a|a|
        ### ----------------------------|-------------------


        # Basic Network architecture
        ## UNet parameters
        self.algo.unet.enabled = False
        self.algo.unet.diffusion_step_embed_dim = 256 # dimension of the diffusion step embedding
        self.algo.unet.down_dims = [256, 512, 1024] # list of dimensions for the down-sampling layers
        self.algo.unet.kernel_size = 5 # kernel size for the convolutional layers
        self.algo.unet.n_groups = 8 # number of groups for the group normalization layers
        ## Transformer parameters
        self.algo.transformer.enabled = True
        self.algo.transformer.num_layers = 8 # number of transformer layers
        self.algo.transformer.num_heads = 4 # number of attention heads
        self.algo.transformer.embed_dim = 256 # dimension of the input embedding
        self.algo.transformer.p_drop_embed = 0.0 # dropout probability for the input embedding
        self.algo.transformer.p_drop_attn = 0.3 # dropout probability for the attention layers
        self.algo.transformer.causal_attn = True # if true, use causal attention
        self.algo.transformer.n_cond_layers = 0 # >0: use transformer encoder for cond, otherwise use MLP


        # EMA
        self.algo.ema.enabled = True # enable EMA
        self.algo.ema.update_after_step = 0 # start updating EMA after this step, NOT USED AT THE MOMENT
        self.algo.ema.inv_gamma = 1.0
        self.algo.ema.power = 0.75
        self.algo.ema.min_value = 0.0
        self.algo.ema.max_value = 0.9999

        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True # enable DDPM
        self.algo.ddpm.num_train_timesteps = 100 # number of training timesteps
        self.algo.ddpm.num_inference_timesteps = 100 # number of inference timesteps, should equal num_train_timesteps
        self.algo.ddpm.beta_start = 0.0001
        self.algo.ddpm.beta_end = 0.02
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.variance_type = 'fixed_small'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'
        ## DDIM (faster inference with less inference timesteps)
        self.algo.ddim.enabled = False # enable DDIM
        self.algo.ddim.num_train_timesteps = 100 # number of training timesteps
        self.algo.ddim.num_inference_timesteps = 100 # can be smaller than num_train_timesteps for faster inference
        self.algo.ddim.beta_start = 0.0001
        self.algo.ddim.beta_end = 0.02
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'