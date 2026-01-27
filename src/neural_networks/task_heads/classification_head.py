import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

from configs.constants import BATCH_LABEL_CATS
from utils.gpu_setup import is_main

@dataclass
class ClassificationOutput:
    loss: Optional[torch.Tensor] = None
    logits: Dict[str, torch.Tensor] = None


class ClassificationHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, labels: list[str]):
        super().__init__()
        self.backbone = backbone
        self.labels = labels
        self.heads = nn.ModuleDict({label: nn.Linear(hidden_dim, len(BATCH_LABEL_CATS[label])) for label in labels})
        if is_main():
            print("Classification heads", self.heads)
        self.loss_fn = nn.CrossEntropyLoss()

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        targets = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self.labels}
        if is_main():
            print("kwargs", kwargs.keys())
            print("targets", targets)
        with torch.no_grad():
            features = self.backbone.get_features(**kwargs)

        logits = {label: head(features) for label, head in self.heads.items()}

        loss = None
        if targets:
            loss = sum(self.loss_fn(logits[k], targets[k]) for k in targets)

        return ClassificationOutput(loss=loss, logits=logits)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self