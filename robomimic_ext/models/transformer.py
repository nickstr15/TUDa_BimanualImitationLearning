from typing import Union

import torch
from torch import nn

from robomimic_ext.models.common import ModuleForDiffusion, SinusoidalPosEmb

class ConditionalTransformerForDiffusion(ModuleForDiffusion):
    """
    Conditional Transformer for Diffusion Models.

    This implementation is adapted from the original source:
    https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py

    It incorporates two fixed parameters:
        - time_as_cond = True
        - obs_as_cond = True

    Args:
        input_dim (int): Dimensionality of the input data.
        input_horizon (int): Horizon length for input data.
        cond_dim (int): Dimensionality of the conditional data.
        cond_horizon (int): Horizon length for conditional data.
        num_layers (int, optional): Number of decoder layers in the Transformer. Default is 8.
        num_heads (int, optional): Number of attention heads. Default is 4.
        embed_dim (int, optional): Dimension of embeddings. Default is 256.
        p_drop_embed (float, optional): Dropout probability for embeddings. Default is 0.0.
        p_drop_attn (float, optional): Dropout probability for attention layers. Default is 0.3.
        causal_attn (bool, optional): Whether to use causal attention masks. Default is True.
        n_cond_layers (int, optional): Number of encoder layers for conditional embeddings. Default is 0.

    Attributes:
        input_embed (nn.Linear): Embedding layer for input data.
        input_pos_embed (nn.Parameter): Positional embedding for inputs.
        drop_embed (nn.Dropout): Dropout layer for input embeddings.
        time_embed (nn.Linear): Embedding layer for time conditioning.
        cond_embed (nn.Linear): Embedding layer for conditional data.
        time_cond_pos_embed (nn.Parameter): Positional embedding for time-cond data.
        encoder (nn.Module): Encoder for processing conditional data.
        decoder (nn.TransformerDecoder): Decoder for generating outputs based on encoded conditions.
        mask (torch.Tensor): Causal attention mask for the decoder.
        memory_mask (torch.Tensor): Mask for the decoder's attention over the encoder's outputs.
        ln_f (nn.LayerNorm): Layer normalization applied to decoder outputs.
        head (nn.Linear): Final output projection layer.
        Ti (int): Horizon length for input data.
        Tc (int): Horizon length for conditional data.
        Ttc (int): Total number of tokens in the conditional input (time + condition = 1 + Tc).
    """
    def __init__(
        self,
        input_dim: int,
        input_horizon: int,
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

        Ti = input_horizon  # Horizon length for input data
        Tc = cond_horizon  # Horizon length for conditional data
        Ttc = 1 + Tc  # Total tokens in the conditional input (time + condition)

        # Input embedding layers
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, Ti, embed_dim))
        self.drop_embed = nn.Dropout(p_drop_embed)

        # Conditional embedding layers
        self.time_embed = SinusoidalPosEmb(embed_dim)
        self.cond_embed = nn.Linear(cond_dim, embed_dim)
        self.time_cond_pos_embed = nn.Parameter(torch.zeros(1, Ttc, embed_dim))

        # Encoder setup
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

        # Decoder setup
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        # Attention masks
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(Ti, Ti)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            # Memory mask
            t, s = torch.meshgrid(
                torch.arange(Ti),
                torch.arange(Ttc),
                indexing='ij'
            )
            mask = t >= (s - 1)  # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('memory_mask', mask)
        else:
            self.mask = None
            self.memory_mask = None

        # Decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, input_dim)

        # Constants
        self.Ti = Ti
        self.Tc = Tc
        self.Ttc = Ttc

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """
        Initialize weights for the model.

        Args:
            module (nn.Module): Module whose weights need to be initialized.
        """
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential
        )
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
            torch.nn.init.normal_(module.input_pos_embed, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.time_cond_pos_embed, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Conditional Transformer.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, Ti, input_dim), where B is the batch size,
                                   Ti is the sequence length, and input_dim is the feature dimension.
            timestep (Union[torch.Tensor, float, int]): Diffusion step. Can be a scalar or a tensor of shape (B,).
            cond (torch.Tensor): Conditioning vector of shape (B, Tc, cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, Ti, input_dim).
        """
        # Process timestep input
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension if necessary
        timesteps = timesteps.expand(sample.shape[0]) #(B,)

        # Embeddings
        ## Time embedding
        time_emb = self.time_embed(timesteps).unsqueeze(1) # (B,) -> (B,1,emb_dim)

        ## Condition embedding
        cond_emb = self.cond_embed(cond)  # (B,Tc,cond_dim) -> (B,Tc,emb_dim)
        total_cond_emb = torch.cat([time_emb, cond_emb], dim=1) # (B,1,emd_dim)+(B,Tc,emb_dim) -> (B,Ttc,emb_dim)
        assert total_cond_emb.shape[1] == self.Ttc

        ## Input embedding
        input_emb = self.input_embed(sample) # (B,T,input_dim) -> (B,T,emb_dim)

        # Encoder
        total_cond_emb = total_cond_emb + self.time_cond_pos_embed # (B,Ttc,emb_dim)
        x = self.drop_embed(total_cond_emb) # (B,Ttc,emb_dim)
        x_enc = self.encoder(x)

        # Decoder
        total_input_emb = input_emb + self.input_pos_embed # (B,Ti,emb_dim)
        x = self.drop_embed(total_input_emb) # (B,Ti,emb_dim)
        x_dec = self.decoder(
            tgt=x,
            memory=x_enc,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        ) # (B,Ti,emb_dim)

        # Final output
        x = self.ln_f(x_dec) # (B,Ti,emb_dim)
        out = self.head(x) # (B,Ti,input_dim)

        return out

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        It is separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).

        Args:
            weight_decay (float): Weight decay strength for regularization.

        Returns:
            list[dict]: List of parameter groups for the optimizer.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameters in the root GPT module as not decayed
        no_decay.add("input_pos_embed")
        no_decay.add("time_cond_pos_embed")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optim_groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups




