from dataclasses import dataclass
from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    input_dim: int = 12
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    max_seq_len: int = 2500
    num_classes: Optional[int] = None
    loss_type: Literal["ddpm", "rectified_flow"] = "rectified_flow"
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class DiTOutput:
    loss: Optional[torch.Tensor]
    prediction: torch.Tensor


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.adaLN1 = AdaLN(d_model)
        self.adaLN2 = AdaLN(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_attn = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gate_ff = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.adaLN1(x, cond)
        y, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.gate_attn * self.dropout(y)
        y = self.adaLN2(x, cond)
        x = x + self.gate_ff * self.ff(y)
        return x


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.layers = nn.ModuleList([DiTBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, cfg.input_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        if cfg.loss_type == "ddpm":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _forward_net(self, xt: torch.Tensor, t: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = xt.shape
        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        pos = torch.arange(seq_len, device=xt.device).unsqueeze(0).expand(bsz, -1)
        h = self.input_proj(xt) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, cond, key_padding_mask=padding_mask)
        h = self.final_norm(h)
        return self.output_proj(h)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            return (F.mse_loss(pred, target, reduction="none") * mask).sum() / mask.sum()
        return F.mse_loss(pred, target)

    def forward(
        self,
        signal: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> DiTOutput:
        x1 = signal.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        bsz = x1.shape[0]

        if self.cfg.loss_type == "rectified_flow":
            x0 = torch.randn_like(x1)
            t = torch.rand(bsz, device=x1.device)
            t_expanded = t.view(bsz, 1, 1)
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            pred = self._forward_net(xt, t, padding_mask)
            target = x1 - x0
        elif self.cfg.loss_type == "ddpm":
            t_int = torch.randint(0, self.cfg.num_steps, (bsz,), device=x1.device)
            t = t_int.float() / self.cfg.num_steps
            noise = torch.randn_like(x1)
            sqrt_alpha = self.alphas_cumprod[t_int].sqrt().view(bsz, 1, 1)
            sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t_int]).sqrt().view(bsz, 1, 1)
            xt = sqrt_alpha * x1 + sqrt_one_minus_alpha * noise
            pred = self._forward_net(xt, t, padding_mask)
            target = noise

        loss = self._compute_loss(pred, target, padding_mask)
        return DiTOutput(loss=loss, prediction=pred.transpose(1, 2))

    @torch.inference_mode()
    def sample(
        self, shape: tuple, device: torch.device, num_steps: Optional[int] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_steps = num_steps or self.cfg.num_steps
        bsz, c, seq_len = shape
        x = torch.randn(bsz, seq_len, c, device=device)

        if self.cfg.loss_type == "rectified_flow":
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((bsz,), (i + 0.5) / num_steps, device=device)
                v = self._forward_net(x, t, padding_mask)
                x = x + v * dt
        elif self.cfg.loss_type == "ddpm":
            if num_steps != self.cfg.num_steps:
                raise ValueError("DDPM requires num_steps equal to training steps")
            for i in reversed(range(num_steps)):
                t = torch.full((bsz,), i / self.cfg.num_steps, device=device)
                pred_noise = self._forward_net(x, t, padding_mask)

                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)

                x0_pred = (x - (1 - alpha_cumprod).sqrt() * pred_noise) / alpha_cumprod.sqrt()
                x0_pred = x0_pred.clamp(0, 1)

                mean = (alpha_cumprod_prev.sqrt() * self.betas[i] * x0_pred + alpha.sqrt() * (1 - alpha_cumprod_prev) * x) / (1 - alpha_cumprod)

                if i > 0:
                    variance = self.betas[i] * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                    x = mean + variance.sqrt() * torch.randn_like(x)
                else:
                    x = mean

        return x.transpose(1, 2)

    def get_features(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = signal.transpose(1, 2)
        bsz, seq_len, _ = x.shape
        t = torch.zeros(bsz, device=x.device)
        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.input_proj(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, cond, key_padding_mask=padding_mask)
        h = self.final_norm(h)
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1)
        return h.mean(dim=1)