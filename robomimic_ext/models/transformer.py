from typing import Union

import torch
from torch import nn

from robomimic_ext.models.common import ModuleForDiffusion, SinusoidalPosEmb


class ConditionalTransformerForDiffusion(ModuleForDiffusion):
    """
    Conditional Transformer for diffusion models.

    Adapted version of https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    with two fixed parameters (in the original implementation):
    - time_as_cond = True
    - obs_as_cond = True
    """
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        cond_horizon: int,
        num_layers: int = 8,
        num_heads: int = 4,
        embed_dim: int = 256,
        p_drop_embed: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0
    ):
        super().__init__()

        T_cond = cond_horizon # cond in forward pass has size (B, T_cond, input_dim)
        T_cond_total = 1 + T_cond # time is the first token in cond

        # input embedding
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, T_cond, embed_dim))
        self.drop_embed = nn.Dropout(p_drop_embed)

        # conditional embedding
        self.time_embed = nn.Linear(input_dim, embed_dim)
        self.cond_embed = nn.Linear(cond_dim, embed_dim)

        self.cond_pos_embed = nn.Parameter(torch.zeros(1, T_cond_total, embed_dim))

        # encoder
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.Mish(),
                nn.Linear(4 * embed_dim, embed_dim)
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(T_cond, T_cond)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            t, s = torch.meshgrid(
                torch.arange(T_cond),
                torch.arange(T_cond_total),
                indexing='ij'
            )
            mask = t >= (s - 1)  # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('memory_mask', mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, input_dim)

        # constants
        self.T_cond = T_cond
        self.T_cond_total = T_cond_total

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ConditionalTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor
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
        # process timestep input
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        # time embedding
        time_emb = self.time_emb(timesteps).unsqueeze(1) # (B,1,emb_dim)

        # input embedding
        input_emb = self.input_emb(sample)

        # encoder
        ## condition embedding
        cond_emb = self.cond_embed(cond) # (B, T_cond, emb_dim)
        total_cond_emb = torch.cat([time_emb, cond_emb], dim=1) # (B, T_cond_total, emb_dim)
        assert total_cond_emb.shape[1] == self.T_cond_total

        ## position embedding
        position_emb = self.cond_pos_embed[
            :, :self.T_cond_total, :
        ] # each position maps to a (learnable) vector

        x = self.drop(total_cond_emb + position_emb)
        x_enc = self.encoder(x) # (B, T_cond_total, emb_dim)

        # decoder
        ti = input_emb.shape[1]
        position_emb = self.pos_emb[
           :, :ti, :
        ] # each position maps to a (learnable) vector
        x = self.drop(input_emb + position_emb) # (B, T_cond, emb_dim)
        x_dec = self.decoder(
            tgt=x,
            memory=x_enc,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        ) # (B, T_cond, emb_dim)

        # head
        x = self.ln_f(x_dec)
        x = self.head(x) # (B, T_cond, input_dim)

        return x



