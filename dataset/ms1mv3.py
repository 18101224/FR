from __future__ import annotations

import os
from typing import Any

from .recordio import MXRecordFaceDataset


class MS1MV3Dataset(MXRecordFaceDataset):
    dataset_name = "ms1mv3"
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
            record_file="train.rec",
            index_file="train.idx",
            cache_file=cache_file,
        )

    @classmethod
    def from_config(cls, dataset_cfg: Any, transform=None) -> "MS1MV3Dataset":
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

        raise ValueError("MS1MV3Dataset requires `root_dir` or (`data_root` and `rec`).")

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
        if architecture == "kprpe_small":
            # KP-RPE supplementary Table 6: ablation setting uses repeated augmentation 0.5.
            return 0.5
        if architecture.startswith("kprpe"):
            # Supplementary material reports repeated augmentation 0.1 for large-scale training.
            return 0.1
        return 0.0
