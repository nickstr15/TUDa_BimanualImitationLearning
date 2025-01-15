from typing import Union

import torch
import torch.nn as nn

from robomimic_ext.models.common import SinusoidalPosEmb

class ConditionalUnet1DForDiffusion(nn.Module):
    """
    A 1D Conditional UNet architecture designed for diffusion models.
    The network is structured with down-sampling,mid-level, and up-sampling modules,
    each incorporating conditional residual blocks to integrate conditioning information.

    Args:
        input_dim (int): Dimensionality of the input features.
        cond_dim (int): Dimensionality of the global conditioning vector, typically obs_horizon * obs_dim.
        diffusion_step_embed_dim (int, optional): Dimensionality of the diffusion step embedding. Default is 256.
        down_dims (list of int, optional): Channel sizes for each level of the UNet during down-sampling. The length of
                                           this list determines the number of levels. Default is [256, 512, 1024].
        kernel_size (int, optional): Kernel size for the convolutional layers. Default is 5.
        n_groups (int, optional): Number of groups for Group Normalization. Default is 8.

    Attributes:
        diffusion_step_encoder (nn.Sequential): Processes the diffusion step input using sinusoidal embeddings and MLPs.
        down_modules (nn.ModuleList): Contains modules for down-sampling, including conditional residual blocks.
        mid_modules (nn.ModuleList): Contains mid-level processing modules with conditional residual blocks.
        up_modules (nn.ModuleList): Contains modules for up-sampling, including conditional residual blocks.
        final_conv (nn.Sequential): Final convolutional layers for reconstructing the output.

    Example:
        model = ConditionalUnet1DForDiffusion(input_dim=64, cond_dim=128)
        sample = torch.randn(16, 128, 64)  # Batch size 16, sequence length 128, input_dim 64
        timestep = 50  # Diffusion step
        cond = torch.randn(16, 128)  # Conditioning vector
        output = model(sample, timestep, cond)
        print(output.shape)  # Output shape: (16, 128, 64)
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

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
        # shape (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension if necessary
        timesteps = timesteps.expand(sample.shape[0])

        # build total condition
        total_cond = self.diffusion_step_encoder(timesteps)
        if cond is not None:
            total_cond = torch.cat((total_cond, cond), dim=-1)

        x = sample

        # downsample and store hidden states
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, total_cond)
            x = resnet2(x, total_cond)
            h.append(x)
            x = downsample(x)

        # mid-level processing
        for mid_module in self.mid_modules:
            x = mid_module(x, total_cond)

        # upsample and concatenate hidden states (residual connections)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, total_cond)
            x = resnet2(x, total_cond)
            x = upsample(x)

        # final convolution
        x = self.final_conv(x)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)
        return x

class ConditionalResidualBlock1D(nn.Module):
    """
    A conditional residual block for 1D input that incorporates
    conditioning information via Feature-wise Linear Modulation (FiLM).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_dim (int): Dimensionality of the conditioning vector.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        n_groups (int, optional): Number of groups for Group Normalization. Default is 8.
        cond_predict_scale (bool, optional): Whether the conditioning vector predicts both scale and bias. Default is False.

    Attributes:
        blocks (nn.ModuleList): A list containing two Conv1dBlock modules for feature processing.
        cond_encoder (nn.Sequential): A sequence of layers that processes the conditioning vector.
        residual_conv (nn.Module): A convolutional layer or identity mapping for the residual connection.
        cond_predict_scale (bool): Whether scale and bias are predicted separately from the conditioning vector.
        out_channels (int): Number of output channels.

    Example:
        cond_block = ConditionalResidualBlock1D(in_channels=32, out_channels=64, cond_dim=128)
        x = torch.randn(1, 32, 128)  # Input tensor of shape (batch_size, in_channels, length)
        cond = torch.randn(1, 128)  # Conditioning vector of shape (batch_size, cond_dim)
        output = cond_block(x, cond)
        print(output.shape)  # Output shape: (1, 64, 128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # Predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )

        # Ensure dimensions are compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConditionalResidualBlock1D.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, horizon).
            cond (torch.Tensor): Conditioning tensor of shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, horizon).
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class Downsample1d(nn.Module):
    """
    A module that performs down-sampling of 1D inputs using a 1D convolutional layer.

    Args:
        dim (int): The number of input and output channels. The input and output dimensions are the same.

    Attributes:
        conv (nn.Conv1d): A 1D convolutional layer with kernel size 3, stride 2, and padding 1.
                          This reduces the spatial resolution of the input by a factor of 2.

    Example:
        downsample = Downsample1d(dim=64)
        input_tensor = torch.randn(1, 64, 128)  # Batch size 1, 64 channels, 128 time steps
        output_tensor = downsample(input_tensor)
        print(output_tensor.shape)  # Output shape: (1, 64, 64)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the down-sampling operation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Down-sampled output tensor of shape (batch_size, channels, length // 2).
        """
        return self.conv(x)

class Upsample1d(nn.Module):
    """
    A module that performs up-sampling of 1D inputs using a 1D transposed convolutional layer.

    Args:
        dim (int): The number of input and output channels. The input and output dimensions are the same.

    Attributes:
        conv (nn.ConvTranspose1d): A 1D transposed convolutional layer with kernel size 4, stride 2, and padding 1.
                                   This increases the spatial resolution of the input by a factor of 2.

    Example:
        upsample = Upsample1d(dim=64)
        input_tensor = torch.randn(1, 64, 64)  # Batch size 1, 64 channels, 64 time steps
        output_tensor = upsample(input_tensor)
        print(output_tensor.shape)  # Output shape: (1, 64, 128)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the up-sampling operation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Up-sampled output tensor of shape (batch_size, channels, length * 2).
        """
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    A 1D convolutional block that performs the following sequence of operations:
    1. 1D Convolution
    2. Group Normalization
    3. Mish Activation

    Args:
        inp_channels (int): Number of input channels for the convolutional layer.
        out_channels (int): Number of output channels for the convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
        n_groups (int, optional): Number of groups for Group Normalization. Default is 8.

    Attributes:
        block (nn.Sequential): Sequential container with the Conv1d, GroupNorm, and Mish layers.

    Example:
        conv_block = Conv1dBlock(inp_channels=32, out_channels=64, kernel_size=3)
        input_tensor = torch.randn(1, 32, 128)  # Batch size 1, 32 channels, 128 time steps
        output_tensor = conv_block(input_tensor)
        print(output_tensor.shape)  # Output shape: (1, 64, 128)
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Conv1dBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inp_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length).
        """
        return self.block(x)


