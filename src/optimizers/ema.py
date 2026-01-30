import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMA(model, decay=0.999)
        # in training loop, after optimizer.step():
        ema.update()
        # for evaluation / checkpointing:
        ema.apply_shadow()   # swap shadow weights into model
        evaluate(model)
        ema.restore()        # swap original weights back
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        # shadow parameters (detached clones, not in any computation graph)
        self.shadow = {
            name: p.data.clone() for name, p in self._params()
        }
        # backup slot used during apply/restore
        self.backup = {}

    def _params(self):
        """Yield (name, param) for trainable parameters, unwrapping DDP."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        return ((n, p) for n, p in m.named_parameters() if p.requires_grad)

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        d = self.decay
        for name, param in self._params():
            self.shadow[name].lerp_(param.data, 1.0 - d)

    def apply_shadow(self):
        """Copy shadow parameters into the model (save originals for restore)."""
        self.backup = {}
        for name, param in self._params():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters after apply_shadow."""
        for name, param in self._params():
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: t.clone() for name, t in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name].copy_(state_dict[name])
