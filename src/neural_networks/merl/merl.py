import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import normalize
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel

from neural_networks.merl.blocks import AttentionPool2d, get_resnet
from utils.gpu_setup import get_rank, get_world_size


@dataclass
class MerlConfig:
    proj_hidden: int = 256
    proj_out: int = 256
    in_channels: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    seq_len: int = 2500
    lm: str = "ncbi/MedCPT-Query-Encoder"
    resnet_type: str = "resnet101"
    distributed: str = None
    spacial_dim: int = None
    d_model: int = 2048

    def __post_init__(self):
        if self.seq_len == 2500:
            self.spacial_dim = 157
        elif self.seq_len == 1250:
            self.spacial_dim = 79
        else:
            self.spacial_dim = 32

@dataclass
class MerlOutput:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]

class Merl(nn.Module):
    def __init__(self, cfg: MerlConfig):
        super().__init__()
        self.cfg = cfg
        self.resnet = get_resnet(cfg.resnet_type)
        self.lm = AutoModel.from_pretrained(cfg.lm)

        self.downconv = nn.Conv1d(in_channels=cfg.in_channels, out_channels=cfg.proj_out, kernel_size=1)
        self.att_pool_head = AttentionPool2d(spacial_dim=cfg.spacial_dim, embed_dim=cfg.proj_out, 
                                             num_heads=4, output_dim=cfg.proj_out)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=cfg.dropout)
        self.dropout2 = nn.Dropout(p=cfg.dropout)
        self.linear1 = nn.Linear(cfg.proj_out, cfg.proj_out, bias=False)
        self.linear2 = nn.Linear(cfg.proj_out, cfg.proj_out, bias=False)

        self.proj_t = nn.Sequential(
            nn.Linear(768, cfg.proj_hidden),
            nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_out),
        )

    def forward(self, signal: torch.Tensor, condition: torch.Tensor):
        out = self.resnet(signal)
        ecg_emb = self.downconv(out)
        proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
        proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
        ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
        ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)

        text_emb = self.get_text_emb(condition)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        combined_loss = self.calc_loss(ecg_emb1, ecg_emb2, proj_ecg_emb, proj_text_emb)

        return MerlOutput(
            loss=combined_loss,
            out=out,
        )

    @torch.no_grad()
    def get_text_emb(self, condition):
        text_emb = self.lm(**condition).pooler_output
        return text_emb

    def calc_loss(self, ecg_emb1, ecg_emb2, proj_ecg_emb, proj_text_emb):
        if self.cfg.distributed:
            world_size = get_world_size()
            rank = get_rank()

            with torch.no_grad():
                gathered_proj_ecg = [torch.zeros_like(proj_ecg_emb) for _ in range(world_size)]
                gathered_proj_text = [torch.zeros_like(proj_text_emb) for _ in range(world_size)]
                gathered_ecg1 = [torch.zeros_like(ecg_emb1) for _ in range(world_size)]
                gathered_ecg2 = [torch.zeros_like(ecg_emb2) for _ in range(world_size)]

                torch.distributed.all_gather(gathered_proj_ecg, proj_ecg_emb)
                torch.distributed.all_gather(gathered_proj_text, proj_text_emb)
                torch.distributed.all_gather(gathered_ecg1, ecg_emb1)
                torch.distributed.all_gather(gathered_ecg2, ecg_emb2)

            gathered_proj_ecg[rank] = proj_ecg_emb
            gathered_proj_text[rank] = proj_text_emb
            gathered_ecg1[rank] = ecg_emb1
            gathered_ecg2[rank] = ecg_emb2

            all_proj_ecg = torch.cat(gathered_proj_ecg, dim=0)
            all_proj_text = torch.cat(gathered_proj_text, dim=0)
            all_ecg1 = torch.cat(gathered_ecg1, dim=0)
            all_ecg2 = torch.cat(gathered_ecg2, dim=0)

            cma_loss = self.merl_loss(all_proj_ecg, all_proj_text)
            uma_loss = self.merl_loss(all_ecg1, all_ecg2)
        else:
            cma_loss = self.merl_loss(proj_ecg_emb, proj_text_emb)
            uma_loss = self.merl_loss(ecg_emb1, ecg_emb2)
        return cma_loss + uma_loss

    def merl_loss(self, x, y, temperature=0.07):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        sim = torch.einsum("i d, j d -> i j", x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(x.device)

        loss_t = F.cross_entropy(sim, labels)
        loss_i = F.cross_entropy(sim.T, labels)

        return loss_t + loss_i