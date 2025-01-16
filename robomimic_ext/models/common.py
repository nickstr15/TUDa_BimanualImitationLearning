import math
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    This class generates sinusoidal positional embeddings.
    The embedding consists of alternating sine and cosine functions with exponentially
    decreasing frequencies.
    """

    def __init__(self, dim: int):
        """
        Initialize the sinusoidal positional embedding.

        :param dim: Total dimension of the positional embedding, must be even.
        """
        super().__init__()

        assert dim % 2 == 0, "Dimension of positional embedding must be even."

        self._half_dim = dim // 2
        # Precompute frequencies
        scale = math.log(1000) / (self._half_dim - 1)
        self._frequencies = torch.exp(-scale * torch.arange(self._half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the sinusoidal positional embedding for the given input tensor.

        :param x: Input tensor of shape (batch_size, ). Each value represents a position.
        :return: Positional embedding tensor of shape (batch_size, dim).
        """
        device = x.device
        frequencies = self._frequencies.to(device)  # Move precomputed frequencies to the correct device

        # Combine input with frequencies
        emb = x.unsqueeze(-1) * frequencies.unsqueeze(0)

        # Concatenate sine and cosine embeddings
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ModuleForDiffusion(ABC, nn.Module):
    """
    Module for diffusion models.

    This class defines the base module structure for diffusion models.
    """
    def __init__(self):
        """
        Initialize the diffusion module.
        """
        super().__init__()

    @abstractmethod
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim), where B is the batch size,
                                   T is the sequence length, and input_dim is the feature dimension.
            timestep (Union[torch.Tensor, float, int]): Diffusion step. Can be a scalar or a tensor of shape (B,).
            cond (torch.Tensor, optional): Conditioning vector of shape (B, cond_dim). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, input_dim).
        """
        raise NotImplementedError("Must implement forward method in derived class.")