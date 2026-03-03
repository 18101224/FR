from __future__ import annotations

import torch

from .base import BaseMarginLoss


class AdaFaceMarginLoss(BaseMarginLoss):
    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        interclass_filtering_threshold: float = 0.0,
        initial_running_mean: float = 20.0,
        initial_running_std: float = 100.0,
        eps: float = 1e-3,
        norm_clamp_min: float = 1e-3,
        norm_clamp_max: float = 100.0,
    ) -> None:
        super().__init__(
            scale=s,
            interclass_filtering_threshold=interclass_filtering_threshold,
        )
        self.margin = float(m)
        self.h = float(h)
        self.t_alpha = float(t_alpha)
        self.eps = float(eps)
        self.norm_clamp_min = float(norm_clamp_min)
        self.norm_clamp_max = float(norm_clamp_max)

        self.register_buffer(
            "running_mean",
            torch.tensor([float(initial_running_mean)], dtype=torch.float32),
        )
        self.register_buffer(
            "running_std",
            torch.tensor([float(initial_running_std)], dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        normalized_labels = self.normalize_labels(labels)
        positive_indices = self.positive_indices(normalized_labels)

        working_logits = self.apply_interclass_filtering(logits, normalized_labels, positive_indices)
        if positive_indices.numel() == 0:
            return working_logits * self.scale

        safe_norms = norms.reshape(-1, 1).detach().clamp(self.norm_clamp_min, self.norm_clamp_max)
        running_mean, running_std = self.update_running_stats(safe_norms)

        margin_scaler = (safe_norms - running_mean) / (running_std + self.eps)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1.0, 1.0).reshape(-1)
        margin_scaler = margin_scaler[positive_indices]

        target_labels = normalized_labels[positive_indices]
        target_logits = working_logits[positive_indices, target_labels].clamp(-1.0 + self.eps, 1.0 - self.eps)

        adaptive_angular_margin = -self.margin * margin_scaler
        angular_logits = torch.cos(torch.acos(target_logits) + adaptive_angular_margin)
        additive_margin = self.margin + (self.margin * margin_scaler)
        updated_target_logits = (angular_logits - additive_margin).to(working_logits.dtype)

        output_logits = working_logits.clone()
        output_logits[positive_indices, target_labels] = updated_target_logits
        return output_logits * self.scale

    def update_running_stats(self, safe_norms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            batch_mean = safe_norms.mean().to(self.running_mean.dtype)
            batch_std = safe_norms.std(unbiased=False).to(self.running_std.dtype)
            self.running_mean.mul_(1.0 - self.t_alpha).add_(batch_mean * self.t_alpha)
            self.running_std.mul_(1.0 - self.t_alpha).add_(batch_std * self.t_alpha)

        return self.running_mean.to(dtype=safe_norms.dtype), self.running_std.to(dtype=safe_norms.dtype)

AdaFaceLoss = AdaFaceMarginLoss
