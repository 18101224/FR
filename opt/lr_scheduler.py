from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import torch


class BaseWarmupScheduler(ABC):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("`total_steps` must be a positive integer.")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_lr_init = float(warmup_lr_init)
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.last_step = -1

        if self.warmup_steps > 0:
            self._set_lrs([self.warmup_lr_init for _ in self.base_lrs])

    def step(self, global_step: int | None = None) -> list[float]:
        if global_step is None:
            global_step = self.last_step + 1

        self.last_step = int(global_step)
        lrs = self.get_lr(self.last_step)
        self._set_lrs(lrs)
        return lrs

    def state_dict(self) -> dict[str, int]:
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.last_step = int(state_dict.get("last_step", -1))
        if self.last_step < 0:
            if self.warmup_steps > 0:
                self._set_lrs([self.warmup_lr_init for _ in self.base_lrs])
            else:
                self._set_lrs(self.base_lrs)
            return
        self._set_lrs(self.get_lr(self.last_step))

    def get_lr(self, global_step: int) -> list[float]:
        if self.warmup_steps > 0 and global_step < self.warmup_steps:
            return self._get_warmup_lrs(global_step)
        return self._get_main_lrs(global_step)

    def _get_warmup_lrs(self, global_step: int) -> list[float]:
        progress = float(global_step + 1) / float(self.warmup_steps)
        return [
            self.warmup_lr_init + (base_lr - self.warmup_lr_init) * progress
            for base_lr in self.base_lrs
        ]

    def _set_lrs(self, lrs: Sequence[float]) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = float(lr)

    @property
    def decay_steps(self) -> int:
        return max(self.total_steps - self.warmup_steps, 1)

    @abstractmethod
    def _get_main_lrs(self, global_step: int) -> list[float]:
        raise NotImplementedError


class WarmupCosineScheduler(BaseWarmupScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
        )
        self.min_lr = float(min_lr)

    def _get_main_lrs(self, global_step: int) -> list[float]:
        decay_step = max(global_step - self.warmup_steps, 0)
        decay_progress = min(float(decay_step) / float(self.decay_steps), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]


class WarmupPolyScheduler(BaseWarmupScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
        power: float = 2.0,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
        )
        self.power = float(power)

    def _get_main_lrs(self, global_step: int) -> list[float]:
        decay_step = max(global_step - self.warmup_steps, 0)
        decay_progress = min(float(decay_step) / float(self.decay_steps), 1.0)
        alpha = pow(1.0 - decay_progress, self.power)
        return [base_lr * alpha for base_lr in self.base_lrs]


class WarmupStepScheduler(BaseWarmupScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
        lr_milestones: Iterable[int] | None = None,
        lr_lambda: float = 0.1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
        )
        self.lr_milestones = sorted(int(milestone) for milestone in (lr_milestones or ()))
        self.lr_lambda = float(lr_lambda)

    def _get_main_lrs(self, global_step: int) -> list[float]:
        alpha = 1.0
        for milestone in self.lr_milestones:
            if global_step > milestone:
                alpha *= self.lr_lambda
        return [base_lr * alpha for base_lr in self.base_lrs]
