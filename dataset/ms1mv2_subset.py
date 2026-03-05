from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any
import urllib.request

import numpy as np

from .ms1mv3 import MS1MV3Dataset


ADAFACE_MS1MV2_SUBSET_URL = (
    "https://raw.githubusercontent.com/mk-minchul/AdaFace/master/assets/ms1mv2_train_subset_index.txt"
)


class MS1MV2SubsetDataset(MS1MV3Dataset):
    """
    AdaFace ablation subset for MS1MV2.

    Behavior matches AdaFace:
    1) keep only indices listed in ms1mv2_train_subset_index.txt
    2) remove identities with fewer than 5 samples
    3) remap labels to contiguous class ids
    """

    dataset_name = "ms1mv2_subset"

    def __init__(
        self,
        root_dir: str,
        transform=None,
        color_space: str = "RGB",
        repeated_augment_prob: float = 0.0,
        repeat_same_image: bool = False,
        cache_file: str = "train.tsv",
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            color_space=color_space,
            repeated_augment_prob=repeated_augment_prob,
            repeat_same_image=repeat_same_image,
            cache_file=cache_file,
        )
        subset_index_path = self._resolve_subset_index_path(self.root_dir)
        subset_indices = self._load_subset_indices(subset_index_path)
        self._apply_subset_and_reindex(subset_indices=subset_indices, min_images_per_identity=5)

    @classmethod
    def from_config(cls, dataset_cfg: Any, transform=None) -> "MS1MV2SubsetDataset":
        root_dir = cls._resolve_root_dir(dataset_cfg)
        color_space = cls._get_config_value(dataset_cfg, "color_space", "RGB")
        cache_file = cls._get_config_value(dataset_cfg, "cache_file", "train.tsv")
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
        )

    @staticmethod
    def _resolve_subset_index_path(root_dir: Path) -> Path:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            root_dir / "ms1mv2_train_subset_index.txt",
            project_root / "assets" / "ms1mv2_train_subset_index.txt",
            project_root / "dataset" / "assets" / "ms1mv2_train_subset_index.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        download_target = root_dir / "ms1mv2_train_subset_index.txt"
        download_target.parent.mkdir(parents=True, exist_ok=True)
        MS1MV2SubsetDataset._download_subset_indices(download_target)
        return download_target

    @staticmethod
    def _download_subset_indices(path: Path) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=path.parent,
                prefix=f".{path.stem}_",
                suffix=path.suffix,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                with urllib.request.urlopen(ADAFACE_MS1MV2_SUBSET_URL) as response:
                    temp_file.write(response.read())
            if temp_path is not None:
                os.replace(temp_path, path)
        except Exception as exc:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to fetch AdaFace ms1mv2 subset index file. "
                f"Please place `ms1mv2_train_subset_index.txt` under `{path.parent}`."
            ) from exc

    @staticmethod
    def _load_subset_indices(path: Path) -> np.ndarray:
        raw_text = path.read_text().strip()
        if not raw_text:
            raise ValueError(f"Subset index file is empty: {path}")
        values = [int(token) for token in raw_text.split(",") if token.strip()]
        if not values:
            raise ValueError(f"No valid indices found in subset index file: {path}")
        return np.asarray(values, dtype=np.int64)

    def _apply_subset_and_reindex(self, subset_indices: np.ndarray, min_images_per_identity: int = 5) -> None:
        if subset_indices.ndim != 1:
            raise ValueError(f"subset_indices must be a 1D array. Got shape: {subset_indices.shape}")
        if len(subset_indices) == 0:
            raise ValueError("subset_indices is empty.")

        # AdaFace builds subset indices from ImageFolder.samples order.
        # convert.py saves images as `imgs/<label>/<record_index>.jpg`,
        # so lexicographic sort over this relative path matches that order.
        sort_order = np.argsort(np.asarray(self._relative_paths, dtype=object))
        sorted_record_indices = self._record_indices[sort_order]
        sorted_raw_labels = self._raw_labels[sort_order]
        sorted_relative_paths = [self._relative_paths[int(index)] for index in sort_order]

        dataset_size = len(sorted_record_indices)
        if subset_indices.min() < 0 or subset_indices.max() >= dataset_size:
            raise ValueError(
                f"Subset indices out of range. dataset_size={dataset_size}, "
                f"min={int(subset_indices.min())}, max={int(subset_indices.max())}"
            )

        self._record_indices = sorted_record_indices[subset_indices]
        self._raw_labels = sorted_raw_labels[subset_indices]
        self._relative_paths = [sorted_relative_paths[int(index)] for index in subset_indices]

        keep_mask = self._build_keep_mask_by_min_count(
            labels=self._raw_labels,
            min_count=max(int(min_images_per_identity), 1),
        )
        self._record_indices = self._record_indices[keep_mask]
        self._raw_labels = self._raw_labels[keep_mask]
        self._relative_paths = [path for path, keep in zip(self._relative_paths, keep_mask) if keep]

        self._class_indices, self._label_mapping = self._build_label_mapping(self._raw_labels)
        self._indices_by_class = self._build_indices_by_class(self._class_indices)
        self._inverse_label_mapping = {value: key for key, value in self._label_mapping.items()}

    @staticmethod
    def _build_keep_mask_by_min_count(labels: np.ndarray, min_count: int) -> np.ndarray:
        unique_labels, counts = np.unique(labels, return_counts=True)
        keep_labels = set(unique_labels[counts >= min_count].tolist())
        return np.asarray([int(label) in keep_labels for label in labels], dtype=np.bool_)
