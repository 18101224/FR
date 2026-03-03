from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FaceSampleRecord:
    record_index: int
    raw_label: Any
    class_index: int
    relative_path: str


class BaseFaceDataset(Dataset, ABC):
    def __init__(
        self,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        color_space: str = "RGB",
        repeated_augment_prob: float = 0.0,
        repeat_same_image: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.color_space = color_space
        self.repeated_augment_prob = float(repeated_augment_prob)
        self.repeat_same_image = bool(repeat_same_image)
        self._previous_index: Optional[int] = None
        self._previous_label: Optional[int] = None

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def read_sample(self, index: int) -> Tuple[Image.Image, int]:
        raise NotImplementedError

    @abstractmethod
    def get_sample_record(self, index: int) -> FaceSampleRecord:
        raise NotImplementedError

    def set_transform(self, transform: Optional[Callable[[Image.Image], Any]]) -> None:
        self.transform = transform

    def set_repeated_augmentation(self, repeated_augment_prob: float, repeat_same_image: bool = False) -> None:
        self.repeated_augment_prob = float(repeated_augment_prob)
        self.repeat_same_image = bool(repeat_same_image)
        self._previous_index = None
        self._previous_label = None

    def sample_index_for_class(self, class_index: int, fallback_index: int) -> int:
        return fallback_index

    def _resolve_index(self, index: int) -> int:
        if self.repeated_augment_prob <= 0.0:
            return index
        if self._previous_label is None or random.random() >= self.repeated_augment_prob:
            return index
        if self.repeat_same_image and self._previous_index is not None:
            return self._previous_index
        return self.sample_index_for_class(self._previous_label, self._previous_index if self._previous_index is not None else index)

    def __getitem__(self, index: int):
        resolved_index = self._resolve_index(index)
        sample, label = self.read_sample(resolved_index)
        self._previous_index = int(resolved_index)
        self._previous_label = int(label)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.tensor(label, dtype=torch.long)
