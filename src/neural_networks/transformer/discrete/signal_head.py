from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SignalFlowHeadConfig:
    signal_dim: int = 12
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 4
    dropout: float = 0.1
    max_signal_len: int = 2500
    num_steps: int = 50


@dataclass
class SignalFlowHeadOutput:
    loss: Optional[torch.Tensor]
    prediction: torch.Tensor


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class FlowBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * y
        y = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.ff(y)
        return x


class SignalFlowHead(nn.Module):
    def __init__(self, cfg: SignalFlowHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.signal_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_signal_len, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.SiLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.layers = nn.ModuleList([FlowBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 2 * cfg.d_model))
        self.output_proj = nn.Linear(cfg.d_model, cfg.signal_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _forward_net(self, xt: torch.Tensor, t: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = xt.shape
        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model)) + context_emb
        pos = torch.arange(seq_len, device=xt.device).unsqueeze(0).expand(bsz, -1)
        h = self.input_proj(xt) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, cond)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.output_proj(h)

    def forward(self, context_emb: torch.Tensor, signal: torch.Tensor) -> SignalFlowHeadOutput:
        x1 = signal.transpose(1, 2)
        bsz = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(bsz, device=x1.device)
        xt = (1 - t.view(bsz, 1, 1)) * x0 + t.view(bsz, 1, 1) * x1
        pred = self._forward_net(xt, t, context_emb)
        loss = F.mse_loss(pred, x1 - x0)
        return SignalFlowHeadOutput(loss=loss, prediction=pred.transpose(1, 2))

    @torch.inference_mode()
    def sample(self, context_emb: torch.Tensor, shape: tuple, device: torch.device, num_steps: Optional[int] = None) -> torch.Tensor:
        num_steps = num_steps or self.cfg.num_steps
        bsz, c, seq_len = shape
        x = torch.randn(bsz, seq_len, c, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((bsz,), (i + 0.5) / num_steps, device=device)
            v = self._forward_net(x, t, context_emb)
            x = x + v * dt
        return x.transpose(1, 2)


class DecoderWithSignalHead(nn.Module):
    def __init__(self, decoder: nn.Module, signal_head: SignalFlowHead, freeze_decoder: bool = False):
        super().__init__()
        self.decoder = decoder
        self.signal_head = signal_head
        self._freeze_decoder = freeze_decoder
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    @property
    def cfg(self):
        return self.decoder.cfg

    def _pool_hidden_states(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        mask = (~padding_mask).unsqueeze(-1).float()
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, tgt_ids: torch.Tensor, signal: torch.Tensor, labels: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> SignalFlowHeadOutput:
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.decoder._make_key_padding_mask(tgt_ids)
        decoder_out = self.decoder(tgt_ids, tgt_key_padding_mask, labels if not self._freeze_decoder else None)
        context_emb = self._pool_hidden_states(decoder_out.hidden_states, tgt_key_padding_mask)
        signal_out = self.signal_head(context_emb, signal)
        if decoder_out.loss is not None:
            signal_out = SignalFlowHeadOutput(
                loss=signal_out.loss + decoder_out.loss,
                prediction=signal_out.prediction,
            )
        return signal_out

    @torch.inference_mode()
    def generate_signal(self, tgt_ids: torch.Tensor, max_new_tokens: int, signal_shape: tuple,
                        num_steps: Optional[int] = None, **gen_kwargs) -> torch.Tensor:
        new_tokens = self.decoder.generate(tgt_ids, max_new_tokens, return_new_only=True, return_logits=False, **gen_kwargs)
        full_seq = torch.cat([tgt_ids, new_tokens], dim=1)
        if full_seq.size(1) > self.decoder.cfg.max_seq_len:
            full_seq = full_seq[:, -self.decoder.cfg.max_seq_len:]
        padding_mask = self.decoder._make_key_padding_mask(full_seq)
        decoder_out = self.decoder(full_seq, padding_mask)
        context_emb = self._pool_hidden_states(decoder_out.hidden_states, padding_mask)
        return self.signal_head.sample(context_emb, signal_shape, tgt_ids.device, num_steps)

    def generate(self, *args, **kwargs):
        return self.decoder.generate(*args, **kwargs)

    def resize_embeddings(self, new_vocab_size: int):
        self.decoder.resize_embeddings(new_vocab_size)
