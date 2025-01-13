from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
from .positional_embedding import SinusoidalPosEmb
from .model import ModelBase

class TransformerForDiffusion(ModelBase):
    """
    A Transformer model for diffusion, supporting both encoder-only (BERT-like)
    and encoder-decoder (GPT-like) configurations.

    The model processes input sequences and optionally conditions on auxiliary information such as time steps or
    observed sequences.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            horizon (int): Length of the target sequence.
            n_obs_steps (int, optional): Number of observed steps for conditioning. Defaults to None.
            cond_dim (int, optional): Dimension of conditioning input. Defaults to 0.
            n_layer (int, optional): Number of Transformer layers. Defaults to 12.
            n_head (int, optional): Number of attention heads. Defaults to 12.
            n_emb (int, optional): Dimension of token embeddings. Defaults to 768.
            p_drop_emb (float, optional): Dropout rate for embeddings. Defaults to 0.1.
            p_drop_attn (float, optional): Dropout rate for attention layers. Defaults to 0.1.
            causal_attn (bool, optional): If True, applies causal masking for autoregressive tasks. Defaults to False.
            time_as_cond (bool, optional): If True, uses time steps as part of the conditioning. Defaults to True.
            obs_as_cond (bool, optional): If True, uses observed sequences as part of the conditioning. Defaults to False.
            n_cond_layers (int, optional): Number of layers in the condition encoder. Defaults to 0.
        """
        super().__init__(input_dim, output_dim)

        # Compute number of tokens for the main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond = 0

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # Input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop_emb = nn.Dropout(p_drop_emb)

        # Condition embedding
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            # Encoder
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                   d_model=n_emb,
                   nhead=n_head,
                   dim_feedforward=4*n_emb,
                   dropout=p_drop_attn,
                   activation='gelu',
                   batch_first=True,
                   norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                   encoder_layer,
                   num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4*n_emb),
                    nn.Mish(),
                    nn.Linear(4*n_emb, n_emb)
                )

            # Decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=n_layer
            )

        else:  # Encoder only (BERT-like)
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layer
            )

        # Attention mask
        if causal_attn:
            # Causal mask to ensure attention is only applied to the left in the input sequence
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('mask', mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing='ij')
                mask = t >= (s-1)  # Add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # Decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """
        Initialize weights for various components of the Transformer.

        Args:
            module (torch.nn.Module): Module whose parameters are being initialized.
        """
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
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass  # No parameters to initialize
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3) -> list:
        """
        Separate parameters into groups with and without weight decay for optimization.

        Args:
            weight_decay (float): Weight decay factor for regularization.

        Returns:
            list: Optimizer parameter groups.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Exclude position embeddings from weight decay
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # Validate parameter separation
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not categorized!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer with learning rate and weight decay.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for regularization.
            betas (Tuple[float, float]): Beta coefficients for AdamW.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the Transformer.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim).
            timestep (Union[torch.Tensor, float, int]): Diffusion step as a tensor or scalar.
            cond (Optional[torch.Tensor]): Conditioning tensor of shape (B, T', cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, output_dim).
        """
        # Time step
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        # Process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # Encoder-only (BERT-like)
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop_emb(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]  # Remove time token
        else:
            # Encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop_emb(cond_embeddings + position_embeddings)
            x = self.encoder(x)

            # Decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop_emb(token_embeddings + position_embeddings)
            x = self.decoder(tgt=x, memory=x, tgt_mask=self.mask, memory_mask=self.memory_mask)

        # Output head
        x = self.ln_f(x)
        x = self.head(x)
        return x