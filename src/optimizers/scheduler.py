import torch
import numpy as np
from abc import ABC
from torch.optim import Adam, AdamW
from utils.gpu_setup import get_world_size, is_main


def get_optimizer(args, model):
    registry = {
        "trans_discrete": DiscreteTransformerOptimizer,
        "trans_continuous": ContinuousTransformerOptimizer,
    }
    for key, cls in registry.items():
        if key in args.neural_network:
            return cls(model, args)
    return BaseOptimizer(model, args)


class BaseOptimizer(ABC):
    OPTIMIZERS = {"adam": Adam, "adamw": AdamW}

    def __init__(self, model, args):
        self.args = args
        self.n_current_steps = 0
        self.n_warmup_steps = args.warmup
        self.eff_bs = self._effective_global_bs()
        self.scale = self._compute_scale()
        self.optimizer = self._build_optimizer(model)
        self._log_config()

    def _world_size(self):
        ws = get_world_size()
        if ws == 1 and self.args.distributed:
            return max(1, torch.cuda.device_count())
        return ws

    def _effective_global_bs(self):
        return self.args.batch_size * self._world_size() * self.args.grad_accum_steps

    def _compute_scale(self):
        ref = self.args.ref_global_bs or (self.args.batch_size * self.args.grad_accum_steps)
        self.args.ref_global_bs = max(ref, 1)
        return max(self.eff_bs / self.args.ref_global_bs, 1e-8)

    def _optimizer_cls(self):
        return self.OPTIMIZERS.get(self.args.optimizer.lower(), AdamW)

    def _build_optimizer(self, model):
        cls = self._optimizer_cls()
        return cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self._base_lr(),
            betas=self._betas(),
            eps=self._eps(),
            weight_decay=self._weight_decay(),
        )

    def _base_lr(self):
        return getattr(self.args, "lr", 1e-3) * self.scale

    def _betas(self):
        return (getattr(self.args, "beta1", 0.9), getattr(self.args, "beta2", 0.999))

    def _eps(self):
        return getattr(self.args, "eps", 1e-8)

    def _weight_decay(self):
        wd = getattr(self.args, "weight_decay", 0.0)
        if self.args.scale_wd == "inv_sqrt":
            return wd / (self.scale**0.5)
        elif self.args.scale_wd == "linear":
            return wd / self.scale
        return wd

    def _log_config(self):
        if is_main():
            print(
                f"[{self.__class__.__name__}] eff_bs={self.eff_bs}, scale={self.scale:.4g}, "
                f"lr={self._base_lr():.3e}, betas={self._betas()}, wd={self._weight_decay():.3e}"
            )

    def get_lr(self):
        step = self.n_current_steps
        warmup = self.n_warmup_steps
        peak_lr = self._base_lr()
        if warmup > 0 and step < warmup:
            return peak_lr * (step + 1) / warmup
        return peak_lr

    def backward(self, loss):
        loss.backward()

    def step_and_update_lr(self):
        lr = self.get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.step()
        self.n_current_steps += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]


class DiscreteTransformerOptimizer(BaseOptimizer):
    def _optimizer_cls(self):
        return AdamW

    def _base_lr(self):
        return getattr(self.args, "lr", 3e-4) * self.scale

    def _betas(self):
        return (getattr(self.args, "beta1", 0.9), getattr(self.args, "beta2", 0.95))

    def _weight_decay(self):
        return getattr(self.args, "weight_decay", 0.1)

    def get_lr(self):
        step = self.n_current_steps
        warmup = self.n_warmup_steps
        max_steps = getattr(self.args, "max_steps", None)
        peak_lr = self._base_lr()
        min_lr = peak_lr * getattr(self.args, "min_lr_ratio", 0.1)

        if step < warmup:
            return peak_lr * (step + 1) / warmup

        if max_steps is None:
            return peak_lr / np.sqrt(max(1, step - warmup + 1))

        progress = (step - warmup) / max(1, max_steps - warmup)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * min(progress, 1.0)))


class ContinuousTransformerOptimizer(BaseOptimizer):
    def __init__(self, model, args):
        super().__init__(model, args)

    def _optimizer_cls(self):
        return Adam

    def _base_lr(self):
        return getattr(self.args, "lr", 1e-4) * self.scale

    def _betas(self):
        return (getattr(self.args, "beta1", 0.9), getattr(self.args, "beta2", 0.999))

    def _weight_decay(self):
        return getattr(self.args, "weight_decay", 0.0)

    def get_lr(self):
        step = self.n_current_steps
        warmup = self.n_warmup_steps
        peak_lr = self._base_lr()
        if warmup > 0 and step < warmup:
            return peak_lr * (step + 1) / warmup
        return peak_lr

    def step_and_update_lr(self):
        super().step_and_update_lr()