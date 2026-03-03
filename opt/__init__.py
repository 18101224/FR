from __future__ import annotations

from argparse import Namespace
from typing import Any, Iterable, Optional

import torch
from torch import optim

from .lr_scheduler import WarmupCosineScheduler, WarmupPolyScheduler, WarmupStepScheduler


def get_optimizer(args: Namespace, model) -> tuple[torch.optim.Optimizer, Optional[object]]:
    params = _resolve_params(model)
    optimizer = _build_optimizer(args=args, params=params)
    scheduler = _build_scheduler(args=args, optimizer=optimizer)
    return optimizer, scheduler


def build_scheduler(args: Namespace, optimizer: torch.optim.Optimizer) -> Optional[object]:
    return _build_scheduler(args=args, optimizer=optimizer)


def scheduler_step(scheduler: Optional[object], global_step: Optional[int] = None) -> None:
    if scheduler is None:
        return
    if global_step is None:
        scheduler.step()
    else:
        scheduler.step(global_step)


def get_last_lr(optimizer: torch.optim.Optimizer) -> float:
    lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return sum(lrs) / len(lrs)


def _build_optimizer(args: Namespace, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    optimizer_name = str(_get_attr(args, "optimizer", "adamw")).lower()
    learning_rate = float(_get_attr(args, "learning_rate", _get_attr(args, "lr", 1e-4)))
    weight_decay = float(_get_attr(args, "weight_decay", 0.0))

    if optimizer_name == "adamw":
        return optim.AdamW(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return optim.SGD(
            params=params,
            lr=learning_rate,
            momentum=float(_get_attr(args, "momentum", 0.9)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Invalid optimizer: {optimizer_name}")


def _build_scheduler(args: Namespace, optimizer: torch.optim.Optimizer) -> Optional[object]:
    scheduler_name = str(_get_attr(args, "scheduler", "cosine")).lower()
    if scheduler_name == "none":
        return None

    total_steps = _resolve_total_steps(args)
    warmup_steps = _resolve_warmup_steps(args)
    base_lr = float(_get_attr(args, "learning_rate", _get_attr(args, "lr", 1e-4)))
    warmup_lr_init = float(_get_attr(args, "warmup_lr_init", 0.0))

    if scheduler_name == "cosine":
        min_lr = float(_get_attr(args, "min_lr", _get_attr(args, "eta_min", base_lr * 0.01)))
        return WarmupCosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            min_lr=min_lr,
        )

    if scheduler_name == "poly_2":
        return WarmupPolyScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            power=2.0,
        )

    if scheduler_name == "poly_0":
        return WarmupPolyScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            power=0.0,
        )

    if scheduler_name == "step":
        return WarmupStepScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            lr_milestones=_resolve_step_milestones(args),
            lr_lambda=float(_get_attr(args, "lr_lambda", 0.1)),
        )

    raise ValueError(f"Invalid scheduler: {scheduler_name}")


def _resolve_step_milestones(args: Namespace) -> list[int]:
    raw_milestones = _get_attr(args, "lr_milestones", ())
    milestones = [int(milestone) for milestone in raw_milestones]

    if not milestones:
        return []

    steps_per_epoch = _get_attr(args, "steps_per_epoch", None)
    if steps_per_epoch is None:
        return milestones

    return [milestone * int(steps_per_epoch) for milestone in milestones]


def _resolve_total_steps(args: Namespace) -> int:
    total_steps = _get_attr(args, "total_steps", None)
    if total_steps is not None:
        return int(total_steps)

    steps_per_epoch = _get_attr(args, "steps_per_epoch", None)
    n_epochs = _get_attr(args, "n_epochs", _get_attr(args, "num_epochs", None))
    if steps_per_epoch is not None and n_epochs is not None:
        return int(steps_per_epoch) * int(n_epochs)
    if n_epochs is not None:
        return int(n_epochs)

    raise ValueError("`total_steps` or (`steps_per_epoch` and `n_epochs`) must be provided.")


def _resolve_warmup_steps(args: Namespace) -> int:
    warmup_steps = _get_attr(args, "warmup_steps", None)
    if warmup_steps is not None:
        return max(int(warmup_steps), 0)

    steps_per_epoch = _get_attr(args, "steps_per_epoch", None)
    warmup_epochs = _get_attr(args, "warmup_epochs", _get_attr(args, "warmup_epoch", 0))
    if steps_per_epoch is None:
        return max(int(warmup_epochs), 0)

    return max(int(steps_per_epoch) * int(warmup_epochs), 0)


def _resolve_params(model) -> Iterable[torch.nn.Parameter]:
    if isinstance(model, (list, tuple)):
        params = []
        for module in model:
            if module is None:
                continue

            if hasattr(module, "parameters"):
                params.extend(param for param in module.parameters() if param.requires_grad)
            else:
                params.extend(param for param in module if param.requires_grad)

        return params

    if hasattr(model, "parameters"):
        return [param for param in model.parameters() if param.requires_grad]

    return model


def _get_attr(obj: Any, key: str, default):
    if obj is None:
        return default

    if isinstance(obj, dict):
        value = obj.get(key, default)
        return default if value is None else value

    if hasattr(obj, key):
        value = getattr(obj, key)
        return default if value is None else value

    return default


__all__ = [
    "WarmupCosineScheduler",
    "WarmupPolyScheduler",
    "WarmupStepScheduler",
    "build_scheduler",
    "get_last_lr",
    "get_optimizer",
    "scheduler_step",
]
