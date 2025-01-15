from typing import Union

import torch
from torch import nn

from robomimic_ext.models.common import ModuleForDiffusion


class ConditionalTransformerForDiffusion(ModuleForDiffusion):
    def __init__(self):
        super().__init__()

        raise NotImplementedError()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Conditional UNet.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim), where B is the batch size,
                                   T is the sequence length, and input_dim is the feature dimension.
            timestep (Union[torch.Tensor, float, int]): Diffusion step. Can be a scalar or a tensor of shape (B,).
            cond (torch.Tensor, optional): Conditioning vector of shape (B, cond_dim). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, input_dim).
        """
        raise NotImplementedError()