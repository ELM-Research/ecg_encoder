from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MAEViTConfig:
    patch_dim: int
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 4
    dropout: float = 0.1
    max_patches: int = 12
    decoder_d_model: int = 512


@dataclass
class MAEOutput:
    loss: Optional[torch.Tensor]
    recon: torch.Tensor
    pred: torch.Tensor
    mask: torch.Tensor
    enc: Optional[torch.Tensor] = None


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x, kp_mask: Optional[torch.Tensor] = None):
        y, _ = self.attn(x, x, x, key_padding_mask=kp_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.lin2(self.dropout_ff(F.gelu(self.lin1(x))))
        x = self.norm2(x + self.dropout(y))
        return x


class MAEViT(nn.Module):
    def __init__(self, cfg: MAEViTConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_enc = nn.Embedding(cfg.max_patches, cfg.d_model)

        self.encoder = nn.ModuleList([ViTBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_encoder_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_d_model))
        self.dec_pos = nn.Embedding(cfg.max_patches, cfg.decoder_d_model)
        self.enc_to_dec = nn.Linear(cfg.d_model, cfg.decoder_d_model) if cfg.decoder_d_model != cfg.d_model else nn.Identity()

        self.decoder = nn.ModuleList([ViTBlock(cfg.decoder_d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_decoder_layers)])
        self.dec_norm = nn.LayerNorm(cfg.decoder_d_model)
        self.pred_head = nn.Linear(cfg.decoder_d_model, cfg.patch_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_enc.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dec_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def _pos_ids(self, L: int, device: torch.device):
        return torch.arange(L, device=device).unsqueeze(0)

    def get_features(self, patches: torch.Tensor, visible_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_vis = self.encode(patches, visible_mask)
        return x_vis.mean(dim=1)

    def encode(self, patches: torch.Tensor, visible_mask: torch.Tensor):
        B, N, _ = patches.shape
        pos = self._pos_ids(N, patches.device).expand(B, N)

        x = self.patch_proj(patches) + self.pos_enc(pos)
        vis = visible_mask.bool()
        x_vis = x[vis].view(B, -1, x.size(-1))

        for blk in self.encoder:
            x_vis = blk(x_vis, kp_mask=None)
        x_vis = self.enc_norm(x_vis)

        return x_vis

    def decode(self, enc: torch.Tensor, visible_mask: torch.Tensor):
        B = enc.size(0)
        N = visible_mask.size(1)

        vis = visible_mask.bool()

        enc_dec = self.enc_to_dec(enc)
        D = enc_dec.size(-1)

        x_full = enc_dec.new_zeros(B, N, D)
        x_full[vis] = enc_dec.reshape(-1, D)

        mask_tok = self.mask_token.to(x_full.dtype)
        x_full[~vis] = mask_tok.expand(B, N, -1)[~vis]

        pos = self._pos_ids(N, enc.device).expand(B, N)
        x = x_full + self.dec_pos(pos)

        for blk in self.decoder:
            x = blk(x, kp_mask=None)
        x = self.dec_norm(x)
        pred = self.pred_head(x)
        return pred

    def forward(
        self,
        patches: torch.Tensor,
        visible_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_on_masked_only: bool = True,
    ) -> MAEOutput:
        enc = self.encode(patches, visible_mask)
        pred = self.decode(enc, visible_mask)
        recon = pred

        loss = None
        if targets is not None:
            if loss_on_masked_only:
                m = (~visible_mask.bool()).unsqueeze(-1).expand_as(pred)
                loss = F.mse_loss(pred[m], targets[m])
            else:
                loss = F.mse_loss(pred, targets)

        return MAEOutput(loss=loss, recon=recon, pred=pred, mask=~visible_mask.bool(), enc=enc)