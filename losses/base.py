from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseMarginLoss(torch.nn.Module, ABC):
    def __init__(
        self,
        scale: float = 64.0,
        interclass_filtering_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.interclass_filtering_threshold = float(interclass_filtering_threshold)

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
        return labels.reshape(-1).long()

    @staticmethod
    def positive_indices(labels: torch.Tensor) -> torch.Tensor:
        return torch.where(labels != -1)[0]

    def apply_interclass_filtering(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.interclass_filtering_threshold <= 0:
            return logits

        filtered_logits = logits.clone()
        filtered_mask = filtered_logits > self.interclass_filtering_threshold

        if positive_indices.numel() > 0:
            protected_mask = torch.ones(
                (positive_indices.numel(), filtered_logits.size(1)),
                device=filtered_logits.device,
                dtype=torch.bool,
            )
            protected_mask.scatter_(1, labels[positive_indices].view(-1, 1), False)
            filtered_mask[positive_indices] &= protected_mask

        filtered_logits.masked_fill_(filtered_mask, 0.0)
        return filtered_logits
