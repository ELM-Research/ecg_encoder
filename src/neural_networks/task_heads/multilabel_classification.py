class MultilabelClassificationHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, labels: list[str]):
        super().__init__()
        self.backbone = backbone
        self.labels = labels
        self.head = nn.Linear(hidden_dim, len(labels))  # one output per condition
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        targets = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self.labels}
        with torch.no_grad():
            features = self.backbone.get_features(**kwargs)
        logits = self.head(features)  # (B, num_labels)
        loss = None
        if targets:
            target_tensor = torch.stack([targets[k] for k in self.labels], dim=-1).float()  # (B, num_labels)
            loss = self.loss_fn(logits, target_tensor)
        return ClassificationOutput(loss=loss, logits=logits)
