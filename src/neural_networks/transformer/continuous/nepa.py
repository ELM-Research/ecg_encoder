from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NEPAConfig:
    input_dim: int = 12
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 2500


@dataclass
class NEPAOutput:
    loss: Optional[torch.Tensor]
    pred_embed: torch.Tensor
    input_embed: torch.Tensor


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        print("in", x.shape)
        y, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.lin2(self.dropout_ff(F.gelu(self.lin1(x))))
        x = self.norm2(x + self.dropout(y))
        print("out", x.shape)
        return x


class NEPATransformer(nn.Module):
    def __init__(self, cfg: NEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.signal_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        self.pred_head = nn.Linear(cfg.d_model, cfg.d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def _embed(self, x: torch.Tensor, modality_mask: Optional[torch.Tensor] = None):
        if modality_mask is not None:
            mask = (~modality_mask).unsqueeze(-1).float()
            x = x * mask

        x = x.transpose(1, 2)
        x = self.signal_proj(x)
        bsz, seq_len, _ = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.pos_emb(pos)
        return self.dropout(x)

    def forward(
        self,
        signal: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> NEPAOutput:
        input_embed = self._embed(signal, modality_mask)

        x = input_embed
        causal_mask = self._causal_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=padding_mask)

        pred_embed = self.pred_head(x)

        loss = self._compute_loss(input_embed, pred_embed, padding_mask)

        return NEPAOutput(loss=loss, pred_embed=pred_embed, input_embed=input_embed)

    def _compute_loss(
        self,
        input_embed: torch.Tensor,
        pred_embed: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target = input_embed.detach()
        pred = pred_embed[:, :-1, :]
        target = target[:, 1:, :]

        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

        loss = -(pred * target).sum(dim=-1)

        if padding_mask is not None:
            valid_mask = ~padding_mask[:, 1:]
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    @torch.inference_mode()
    def encode(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_embed = self._embed(signal)
        x = input_embed
        causal_mask = self._causal_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        return x

    def get_features(
        self,
    ):
        pass