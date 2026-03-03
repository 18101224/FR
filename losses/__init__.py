from __future__ import annotations

from typing import Optional

from .adaface import AdaFaceLoss, AdaFaceMarginLoss
from .base import BaseMarginLoss

def get_margin_loss(name: Optional[str] = "adaface", **kwargs) -> Optional[BaseMarginLoss]:
    normalized_name = "none" if name is None else str(name).lower()
    if normalized_name == "none":
        return None

    if normalized_name != "adaface":
        raise ValueError("Unsupported margin loss. Supported losses: adaface, none")

    return AdaFaceMarginLoss(**kwargs)


def build_margin_loss(name: Optional[str] = "adaface", **kwargs) -> Optional[BaseMarginLoss]:
    return get_margin_loss(name=name, **kwargs)


__all__ = [
    "AdaFaceLoss",
    "AdaFaceMarginLoss",
    "BaseMarginLoss",
    "build_margin_loss",
    "get_margin_loss",
]
