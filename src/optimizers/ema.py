import torch


class EMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {
            name: p.data.clone() for name, p in self._params()
        }
        self.backup = {}

    def _params(self):
        m = self.model.module if hasattr(self.model, "module") else self.model
        return ((n, p) for n, p in m.named_parameters() if p.requires_grad)

    @torch.no_grad()
    def update(self):
        d = self.decay
        for name, param in self._params():
            self.shadow[name].lerp_(param.data, 1.0 - d)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self._params():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self._params():
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: t.clone() for name, t in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name].copy_(state_dict[name])
