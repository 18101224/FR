from __future__ import annotations

import csv
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import BaseFaceDataset, FaceSampleRecord


class VGGFace2Dataset(BaseFaceDataset):
    dataset_name = "vgg2"

    def __init__(
        self,
        root_dir: str,
        transform=None,
        color_space: str = "RGB",
        repeated_augment_prob: float = 0.0,
        repeat_same_image: bool = False,
        cache_file: str = "train.tsv",
        split: str = "train",
    ) -> None:
        super().__init__(
            transform=transform,
            color_space=color_space,
            repeated_augment_prob=repeated_augment_prob,
            repeat_same_image=repeat_same_image,
        )
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = str(split)
        self.image_root = self._resolve_image_root()
        self.cache_path = self.root_dir / cache_file

        self._validate_root()
        self._relative_paths, self._raw_labels = self._load_or_create_cache()
        self._class_indices, self._label_mapping = self._build_label_mapping(self._raw_labels)
        self._inverse_label_mapping = {value: key for key, value in self._label_mapping.items()}
        self._indices_by_class = self._build_indices_by_class(self._class_indices)

    @property
    def num_classes(self) -> int:
        return len(self._label_mapping)

    @property
    def label_mapping(self) -> Dict[str, int]:
        return dict(self._label_mapping)

    @property
    def class_to_raw_label(self) -> Dict[int, str]:
        return dict(self._inverse_label_mapping)

    def __len__(self) -> int:
        return len(self._relative_paths)

    def get_sample_record(self, index: int) -> FaceSampleRecord:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")
        return FaceSampleRecord(
            record_index=int(index),
            raw_label=self._raw_labels[index],
            class_index=int(self._class_indices[index]),
            relative_path=self._relative_paths[index],
        )

    def read_sample(self, index: int) -> Tuple[Image.Image, int]:
        sample_record = self.get_sample_record(index)
        image_path = self.image_root / sample_record.relative_path
        image = Image.open(image_path).convert(self.color_space)
        return image, sample_record.class_index

    def sample_index_for_class(self, class_index: int, fallback_index: int) -> int:
        class_members = self._indices_by_class.get(int(class_index))
        if class_members is None or len(class_members) == 0:
            return int(fallback_index)
        if len(class_members) == 1:
            return int(class_members[0])
        fallback_index = int(fallback_index)
        candidates = [int(candidate) for candidate in class_members if int(candidate) != fallback_index]
        if not candidates:
            return fallback_index
        return random.choice(candidates)

    def _resolve_image_root(self) -> Path:
        if (self.root_dir / self.split).is_dir():
            return self.root_dir / self.split
        return self.root_dir

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")

    def _load_or_create_cache(self) -> Tuple[List[str], List[str]]:
        if self.cache_path.exists():
            return self._load_cache()
        return self._scan_images_and_cache()

    def _load_cache(self) -> Tuple[List[str], List[str]]:
        relative_paths: List[str] = []
        raw_labels: List[str] = []
        with self.cache_path.open("r", newline="") as cache_handle:
            reader = csv.reader(cache_handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError(f"Invalid cache row in {self.cache_path}: {row}")
                relative_paths.append(row[0])
                raw_labels.append(row[1])
        return relative_paths, raw_labels

    def _scan_images_and_cache(self) -> Tuple[List[str], List[str]]:
        relative_paths: List[str] = []
        raw_labels: List[str] = []
        temp_path: Optional[Path] = None

        image_paths = sorted(
            path for path in self.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            delete=False,
            dir=self.root_dir,
            prefix=f".{self.cache_path.stem}_",
            suffix=self.cache_path.suffix,
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            writer = csv.writer(temp_handle, delimiter="\t")

            for image_path in image_paths:
                relative_path = image_path.relative_to(self.image_root).as_posix()
                raw_label = image_path.parent.name
                writer.writerow((relative_path, raw_label))
                relative_paths.append(relative_path)
                raw_labels.append(raw_label)

        if temp_path is not None:
            os.replace(temp_path, self.cache_path)

        return relative_paths, raw_labels

    @staticmethod
    def _build_label_mapping(raw_labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
        mapping: Dict[str, int] = {}
        class_indices: List[int] = []
        for raw_label in raw_labels:
            class_index = mapping.setdefault(raw_label, len(mapping))
            class_indices.append(class_index)
        return class_indices, mapping

    @staticmethod
    def _build_indices_by_class(class_indices: List[int]) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        for dataset_index, class_index in enumerate(class_indices):
            grouped.setdefault(int(class_index), []).append(int(dataset_index))
        return grouped

    @classmethod
    def from_config(cls, dataset_cfg: Any, transform=None) -> "VGGFace2Dataset":
        root_dir = cls._resolve_root_dir(dataset_cfg)
        color_space = cls._get_config_value(dataset_cfg, "color_space", "RGB")
        cache_file = cls._get_config_value(dataset_cfg, "cache_file", "train.tsv")
        split = cls._get_config_value(dataset_cfg, "split", "train")
        repeated_augment_prob = cls._resolve_repeated_augment_prob(dataset_cfg)
        repeat_same_image = bool(
            cls._get_config_value(
                dataset_cfg,
                "repeat_same_image",
                cls._get_config_value(dataset_cfg, "use_same_image", False),
            )
        )
        return cls(
            root_dir=root_dir,
            transform=transform,
            color_space=color_space,
            repeated_augment_prob=repeated_augment_prob,
            repeat_same_image=repeat_same_image,
            cache_file=cache_file,
            split=split,
        )

    @classmethod
    def _resolve_root_dir(cls, dataset_cfg: Any) -> str:
        root_dir = cls._get_config_value(dataset_cfg, "root_dir", None)
        if root_dir:
            return str(root_dir)

        dataset_root = cls._get_config_value(dataset_cfg, "dataset_root", None)
        if dataset_root:
            return str(dataset_root)

        data_root = cls._get_config_value(dataset_cfg, "data_root", None)
        rec = cls._get_config_value(dataset_cfg, "rec", None)
        if data_root and rec:
            return os.path.join(str(data_root), str(rec))

        raise ValueError("VGGFace2Dataset requires `root_dir` or (`data_root` and `rec`).")

    @staticmethod
    def _get_config_value(dataset_cfg: Any, key: str, default):
        if dataset_cfg is None:
            return default
        if isinstance(dataset_cfg, dict):
            value = dataset_cfg.get(key, default)
            return default if value is None else value
        if hasattr(dataset_cfg, key):
            value = getattr(dataset_cfg, key)
            return default if value is None else value
        return default

    @classmethod
    def _resolve_repeated_augment_prob(cls, dataset_cfg: Any) -> float:
        explicit_prob = cls._get_config_value(dataset_cfg, "repeated_augment_prob", None)
        if explicit_prob is not None:
            return float(explicit_prob)

        architecture = str(cls._get_config_value(dataset_cfg, "architecture", ""))
        if architecture.startswith("kprpe"):
            return 0.1
        return 0.0
