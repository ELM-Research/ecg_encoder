import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ClassificationOutput:
    loss: Optional[torch.Tensor] = None
    logits: Dict[str, torch.Tensor] = None


class MultiLabelClassificationHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, labels: list[str]):
        super().__init__()
        self.backbone = backbone
        self.labels = labels
        self.heads = nn.Linear(hidden_dim, len(labels))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        targets = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self.labels}
        with torch.no_grad():
            features = self.backbone.get_features(**kwargs)
        logits = self.heads(features)
        loss = None
        if targets:
            target_tensor = torch.stack([targets[k] for k in self.labels], dim=-1).float()
            loss = self.loss_fn(logits, target_tensor)
        return ClassificationOutput(loss=loss, logits=logits)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self