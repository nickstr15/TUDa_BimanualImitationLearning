import math
import torch
from torch import nn


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

        :param dim: Total dimension of the positional embedding.
        """
        super().__init__()

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

