import torch
import numpy as np
from torch.optim import Adam, AdamW
from utils.gpu_setup import get_world_size, is_main


OPTIMIZERS = {"adam": Adam, "adamw": AdamW}


def get_optimizer(args, model):
    return Optimizer(model, args)


class Optimizer:

    def __init__(self, model, args):
        self.args = args
        self.n_current_steps = 0
        self.n_warmup_steps = args.warmup
        self.eff_bs = self._effective_global_bs()
        self.scale = self._compute_scale()
        self.peak_lr = args.lr * self.scale
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

    def _weight_decay(self):
        wd = self.args.weight_decay
        if self.args.scale_wd == "inv_sqrt":
            return wd / (self.scale**0.5)
        elif self.args.scale_wd == "linear":
            return wd / self.scale
        return wd

    def _build_optimizer(self, model):
        cls = OPTIMIZERS.get(self.args.optimizer.lower(), AdamW)
        return cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.peak_lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self._weight_decay(),
        )

    def _log_config(self):
        if is_main():
            print(
                f"[Optimizer] {self.args.optimizer} | lr_schedule={self.args.lr_schedule} | "
                f"eff_bs={self.eff_bs} | scale={self.scale:.4g} | "
                f"lr={self.peak_lr:.3e} | betas=({self.args.beta1}, {self.args.beta2}) | "
                f"wd={self._weight_decay():.3e}"
            )

    def get_lr(self):
        step = self.n_current_steps
        warmup = self.n_warmup_steps
        schedule = self.args.lr_schedule

        if warmup > 0 and step < warmup:
            return self.peak_lr * (step + 1) / warmup

        if schedule == "constant":
            return self.peak_lr

        if schedule == "inv_sqrt":
            return self.peak_lr / np.sqrt(max(1, step - warmup + 1))

        if schedule == "cosine":
            max_steps = getattr(self.args, "max_steps", None)
            min_lr = self.peak_lr * self.args.min_lr_ratio
            if max_steps is None:
                return self.peak_lr
            progress = (step - warmup) / max(1, max_steps - warmup)
            return min_lr + 0.5 * (self.peak_lr - min_lr) * (1 + np.cos(np.pi * min(progress, 1.0)))

        return self.peak_lr

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
