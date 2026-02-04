import torch
import numpy as np
from torch.optim import Adam, AdamW
from utils.gpu_setup import get_world_size, is_main


OPTIMIZERS = {"adam": Adam, "adamw": AdamW}


def get_optimizer(args, model):
    return Optimizer(model, args)


def _is_muon_param(name: str, param: torch.Tensor) -> bool:
    """
    Determine if a parameter should be optimized with Muon.

    Muon is designed for 2D weight matrices in hidden layers.
    Excludes: embeddings, biases, normalization, and head/output layers.

    Reference: https://kellerjordan.github.io/posts/muon/
    """
    if param.ndim != 2:
        return False
    name_lower = name.lower()
    # Exclude embeddings (input representations)
    if "emb" in name_lower:
        return False
    # Exclude output projections and classifier heads
    if "output_proj" in name_lower or "head" in name_lower:
        return False
    # Exclude normalization layers
    if "norm" in name_lower or "ln" in name_lower:
        return False
    return True


class MuonAdamW:
    """
    Combined Muon + AdamW optimizer.

    Muon optimizes 2D hidden layer weights; AdamW handles the rest.
    This follows the standard practice from the Muon paper.

    References:
    - PyTorch Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
    - Keller Jordan: https://github.com/KellerJordan/Muon
    - Diffusion benchmark: https://arxiv.org/abs/2510.19376
    """

    def __init__(self, muon_opt, adamw_opt, adamw_lr_ratio: float):
        self.muon = muon_opt
        self.adamw = adamw_opt
        self.adamw_lr_ratio = adamw_lr_ratio

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


class Optimizer:

    def __init__(self, model, args):
        self.args = args
        self.n_current_steps = 0
        self.n_warmup_steps = args.warmup
        self.eff_bs = self._effective_global_bs()
        self.scale = self._compute_scale()
        self.peak_lr = args.lr * self.scale
        self._is_muon = args.optimizer.lower() == "muon"
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

    def _weight_decay(self, wd=None):
        wd = wd if wd is not None else self.args.weight_decay
        if self.args.scale_wd == "inv_sqrt":
            return wd / (self.scale**0.5)
        elif self.args.scale_wd == "linear":
            return wd / self.scale
        return wd

    def _build_optimizer(self, model):
        opt_name = self.args.optimizer.lower()
        if opt_name == "muon":
            return self._build_muon_optimizer(model)
        cls = OPTIMIZERS.get(opt_name, AdamW)
        return cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.peak_lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self._weight_decay(),
        )

    def _build_muon_optimizer(self, model):
        """
        Build Muon for 2D hidden weights + AdamW for everything else.

        Hyperparameter rationale:
        - Muon: momentum=0.95, nesterov=True (empirically validated defaults)
        - Muon uses "match_rms_adamw" adjustment so same LR scale works
        - AdamW LR: 10% of Muon LR (tunable via muon_adamw_lr_ratio)
        """
        from torch.optim import Muon

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_muon_param(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        # Learning rates
        # With "match_rms_adamw", Muon LR can be similar to AdamW LR
        # Default ratio: AdamW gets 10% of Muon's LR (can be tuned)
        adamw_lr_ratio = getattr(self.args, "muon_adamw_lr_ratio", 0.1)
        adamw_lr = self.peak_lr * adamw_lr_ratio

        # Muon hyperparameters
        muon_momentum = getattr(self.args, "muon_momentum", 0.95)
        muon_nesterov = getattr(self.args, "muon_nesterov", True)
        muon_ns_steps = getattr(self.args, "muon_ns_steps", 5)

        wd = self._weight_decay()

        # Store for logging
        self._muon_param_count = len(muon_params)
        self._adamw_param_count = len(adamw_params)
        self._adamw_lr = adamw_lr
        self._adamw_lr_ratio = adamw_lr_ratio

        # Build separate optimizers
        muon_opt = Muon(
            muon_params,
            lr=self.peak_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            weight_decay=wd,
            adjust_lr_fn="match_rms_adamw",
        )

        adamw_opt = AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=wd,
        )

        return MuonAdamW(muon_opt, adamw_opt, adamw_lr_ratio)

    def _log_config(self):
        if is_main():
            if self._is_muon:
                muon_momentum = getattr(self.args, "muon_momentum", 0.95)
                print(
                    f"[Optimizer] Muon+AdamW | lr_schedule={self.args.lr_schedule} | "
                    f"eff_bs={self.eff_bs} | scale={self.scale:.4g}"
                )
                print(
                    f"  Muon: {self._muon_param_count} params | lr={self.peak_lr:.3e} | "
                    f"momentum={muon_momentum} | wd={self._weight_decay():.3e}"
                )
                print(
                    f"  AdamW: {self._adamw_param_count} params | lr={self._adamw_lr:.3e} | "
                    f"betas=({self.args.beta1}, {self.args.beta2}) | wd={self._weight_decay():.3e}"
                )
            else:
                print(
                    f"[Optimizer] {self.args.optimizer} | lr_schedule={self.args.lr_schedule} | "
                    f"eff_bs={self.eff_bs} | scale={self.scale:.4g} | "
                    f"lr={self.peak_lr:.3e} | betas=({self.args.beta1}, {self.args.beta2}) | "
                    f"wd={self._weight_decay():.3e}"
                )

    def _lr_multiplier(self):
        """Compute LR multiplier based on warmup and schedule."""
        step = self.n_current_steps
        warmup = self.n_warmup_steps
        schedule = self.args.lr_schedule

        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        if schedule == "constant":
            return 1.0

        if schedule == "inv_sqrt":
            return 1.0 / np.sqrt(max(1, step - warmup + 1))

        if schedule == "cosine":
            max_steps = getattr(self.args, "max_steps", None)
            if max_steps is None:
                return 1.0
            progress = (step - warmup) / max(1, max_steps - warmup)
            min_ratio = self.args.min_lr_ratio
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1 + np.cos(np.pi * min(progress, 1.0)))

        return 1.0

    def get_lr(self):
        return self.peak_lr * self._lr_multiplier()

    def backward(self, loss):
        loss.backward()

    def step_and_update_lr(self):
        mult = self._lr_multiplier()
        if self._is_muon:
            # Muon params: scale from peak_lr
            for g in self.optimizer.muon.param_groups:
                g["lr"] = self.peak_lr * mult
            # AdamW params: scale from adamw_lr (which is peak_lr * ratio)
            for g in self.optimizer.adamw.param_groups:
                g["lr"] = self._adamw_lr * mult
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = self.peak_lr * mult
        self.optimizer.step()
        self.n_current_steps += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
