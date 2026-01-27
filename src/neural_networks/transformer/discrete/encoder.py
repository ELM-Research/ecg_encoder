from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderTransformerConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class EncoderTransformerOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        y, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.lin2(self.dropout_ff(F.gelu(self.lin1(x))))
        x = self.norm2(x + self.dropout(y))
        return x


class EncoderTransformer(nn.Module):
    def __init__(self, cfg: EncoderTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _embed(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        return self.dropout(x)

    def _make_key_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.eq(self.cfg.pad_id)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> EncoderTransformerOutput:
        if src_key_padding_mask is None:
            src_key_padding_mask = self._make_key_padding_mask(src_ids)
        x = self._embed(src_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return EncoderTransformerOutput(loss=loss, logits=logits, hidden_states=x)