import torch
from .model import ModelBase

class MaskGenerator(ModelBase):
    """
    Generates masks for action and observation dimensions over a batch of sequences.

    Parameters:
        action_dim (int): The number of action dimensions.
        obs_dim (int): The number of observation dimensions.
        max_n_obs_steps (int, optional): Maximum number of observation steps. Defaults to 2.
        fix_obs_steps (bool, optional): If True, all sequences have exactly max_n_obs_steps observation steps.
                                        If False, the number of observation steps is sampled randomly.
                                        Defaults to True.
        action_visible (bool, optional): If True, actions can also have visibility determined by observation steps.
                                         Defaults to False.
    """
    def __init__(self,
                 action_dim, obs_dim,
                 max_n_obs_steps=2,
                 fix_obs_steps=True,
                 action_visible=False
                 ):
        super().__init__(None, None) # ignore input_dim and output_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape: torch.Size, seed: int = None) -> torch.Tensor:
        """
        Generate masks for the given sequence shape.

        Parameters:
            shape (torch.Size): A 3D tensor shape (B, T, D), where:
                                - B: Batch size.
                                - T: Sequence length.
                                - D: Feature dimensions (must equal action_dim + obs_dim).
            seed (int, optional): Seed for random number generation. Defaults to None.

        Returns:
            torch.Tensor: A boolean tensor of shape (B, T, D), where True indicates visible dimensions.
        """
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim), \
            f"Feature dimensions (D={D}) must equal action_dim + obs_dim ({self.action_dim + self.obs_dim})."

        # Create a random number generator
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # Initialize a mask tensor
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)

        # Identify action and observation dimensions
        is_action_dim = dim_mask.clone()
        is_action_dim[..., :self.action_dim] = True  # First `action_dim` are action dimensions
        is_obs_dim = ~is_action_dim  # Remaining dimensions are observation dimensions

        # Generate observation step masks
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps + 1,
                size=(B,), generator=rng, device=device)

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim  # Apply only to observation dimensions

        # Start with observation mask
        mask = obs_mask

        # Generate action step masks if actions are visible
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim  # Apply only to action dimensions

            mask = mask | action_mask  # Combine observation and action masks

        return mask