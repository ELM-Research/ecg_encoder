from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Seq2SeqTransformerConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class Seq2SeqTransformerOutput:
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


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        y, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y, _ = self.cross_attn(x, encoder_out, encoder_out, key_padding_mask=encoder_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout(y))
        y = self.lin2(self.dropout_ff(F.gelu(self.lin1(x))))
        x = self.norm3(x + self.dropout(y))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: Seq2SeqTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder_layers = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_decoder_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def _embed(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        return self.dropout(x)

    def _make_key_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.eq(self.cfg.pad_id)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        if src_key_padding_mask is None:
            src_key_padding_mask = self._make_key_padding_mask(src_ids)
        x = self._embed(src_ids)
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return x

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._make_key_padding_mask(tgt_ids)
        x = self._embed(tgt_ids)
        causal_mask = self._causal_mask(tgt_ids.size(1), x.device)
        for layer in self.decoder_layers:
            x = layer(
                x,
                attn_mask=causal_mask,
                key_padding_mask=tgt_key_padding_mask,
                encoder_out=encoder_out,
                encoder_key_padding_mask=encoder_key_padding_mask,
            )
        return x

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Seq2SeqTransformerOutput:
        if src_key_padding_mask is None:
            src_key_padding_mask = self._make_key_padding_mask(src_ids)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._make_key_padding_mask(tgt_ids)
        encoder_out = self.encode(src_ids, src_key_padding_mask)
        decoder_out = self.decode(tgt_ids, encoder_out, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.lm_head(decoder_out)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return Seq2SeqTransformerOutput(loss=loss, logits=logits, hidden_states=decoder_out)

    @torch.inference_mode()
    def generate(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        eos_id: Optional[int] = None,
        return_new_only: bool = False,
    ) -> torch.Tensor:
        src_key_padding_mask = self._make_key_padding_mask(src_ids)
        encoder_out = self.encode(src_ids, src_key_padding_mask)

        out = tgt_ids
        start_len = out.size(1)
        finished = torch.zeros(out.size(0), dtype=torch.bool, device=out.device)

        for _ in range(max_new_tokens):
            if out.size(1) >= self.cfg.max_seq_len or finished.all():
                break

            decoder_out = self.decode(out, encoder_out, self._make_key_padding_mask(out), src_key_padding_mask)
            logits = self.lm_head(decoder_out[:, -1, :])

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            if top_k is not None and 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None:
                next_id = torch.where(finished.unsqueeze(1), torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.squeeze(1).eq(eos_id)

            out = torch.cat([out, next_id], dim=1)

        return out[:, start_len:] if return_new_only else out