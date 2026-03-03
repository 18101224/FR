from __future__ import annotations

import atexit
import csv
import io
import os
import random
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image

from .base import BaseFaceDataset, FaceSampleRecord
from .vggface2 import VGGFace2Dataset


class CASIAWebFaceParquetDataset(BaseFaceDataset):
    dataset_name = "casia"

    IMAGE_COLUMN_CANDIDATES = ("image", "img")
    LABEL_COLUMN_CANDIDATES = ("label", "labels", "identity", "class", "class_id", "person_id")
    PATH_COLUMN_CANDIDATES = ("relative_path", "image_path", "path", "file_name", "filename")

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
            transform=transform,
            color_space=color_space,
            repeated_augment_prob=repeated_augment_prob,
            repeat_same_image=repeat_same_image,
        )
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.data_root = self.root_dir / "data" if (self.root_dir / "data").is_dir() else self.root_dir
        self.cache_path = self.root_dir / cache_file
        self._parquet_handles: Dict[int, pq.ParquetFile] = {}
        self._row_group_cache: Dict[int, Tuple[int, Any]] = {}

        self._validate_root()
        self._parquet_paths = sorted(self.data_root.glob("*.parquet"))
        if not self._parquet_paths:
            raise FileNotFoundError(f"No parquet shards found under: {self.data_root}")

        first_shard = pq.ParquetFile(self._parquet_paths[0])
        self._column_names = list(first_shard.schema.names)
        self._image_column = self._resolve_required_column(self.IMAGE_COLUMN_CANDIDATES)
        self._label_column = self._resolve_required_column(self.LABEL_COLUMN_CANDIDATES)
        self._path_column = self._resolve_optional_column(self.PATH_COLUMN_CANDIDATES)

        self._row_group_offsets = self._build_row_group_offsets()
        (
            self._shard_indices,
            self._row_indices,
            self._relative_paths,
            self._raw_labels,
        ) = self._load_or_create_cache()
        self._class_indices, self._label_mapping = self._build_label_mapping(self._raw_labels)
        self._inverse_label_mapping = {value: key for key, value in self._label_mapping.items()}
        self._indices_by_class = self._build_indices_by_class(self._class_indices)
        atexit.register(self.close)

    @property
    def num_classes(self) -> int:
        return len(self._label_mapping)

    @property
    def label_mapping(self) -> Dict[Any, int]:
        return dict(self._label_mapping)

    @property
    def class_to_raw_label(self) -> Dict[int, Any]:
        return dict(self._inverse_label_mapping)

    def __len__(self) -> int:
        return len(self._relative_paths)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parquet_handles"] = {}
        state["_row_group_cache"] = {}
        return state

    def close(self) -> None:
        self._parquet_handles = {}
        self._row_group_cache = {}

    def __del__(self) -> None:
        self.close()

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
        shard_index = self._shard_indices[index]
        row_index = self._row_indices[index]
        image = self._read_image_from_shard(shard_index, row_index)
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

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset data root does not exist: {self.data_root}")

    def _resolve_required_column(self, candidates: Tuple[str, ...]) -> str:
        for candidate in candidates:
            if candidate in self._column_names:
                return candidate
        raise ValueError(f"Missing required parquet column. Candidates: {candidates}, available: {self._column_names}")

    def _resolve_optional_column(self, candidates: Tuple[str, ...]) -> Optional[str]:
        for candidate in candidates:
            if candidate in self._column_names:
                return candidate
        return None

    def _build_row_group_offsets(self) -> Dict[int, List[int]]:
        offsets: Dict[int, List[int]] = {}
        for shard_index, parquet_path in enumerate(self._parquet_paths):
            parquet_file = pq.ParquetFile(parquet_path)
            shard_offsets = [0]
            running = 0
            for row_group_index in range(parquet_file.num_row_groups):
                running += parquet_file.metadata.row_group(row_group_index).num_rows
                shard_offsets.append(running)
            offsets[shard_index] = shard_offsets
        return offsets

    def _load_or_create_cache(self) -> Tuple[List[int], List[int], List[str], List[Any]]:
        if self.cache_path.exists():
            return self._load_cache()
        return self._scan_parquet_and_cache()

    def _load_cache(self) -> Tuple[List[int], List[int], List[str], List[Any]]:
        shard_indices: List[int] = []
        row_indices: List[int] = []
        relative_paths: List[str] = []
        raw_labels: List[Any] = []

        with self.cache_path.open("r", newline="") as cache_handle:
            reader = csv.reader(cache_handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if len(row) != 4:
                    raise ValueError(f"Invalid cache row in {self.cache_path}: {row}")
                shard_indices.append(int(row[0]))
                row_indices.append(int(row[1]))
                relative_paths.append(row[2])
                raw_labels.append(row[3])

        return shard_indices, row_indices, relative_paths, raw_labels

    def _scan_parquet_and_cache(self) -> Tuple[List[int], List[int], List[str], List[Any]]:
        shard_indices: List[int] = []
        row_indices: List[int] = []
        relative_paths: List[str] = []
        raw_labels: List[Any] = []
        temp_path: Optional[Path] = None

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

            for shard_index, parquet_path in enumerate(self._parquet_paths):
                parquet_file = pq.ParquetFile(parquet_path)
                columns = [self._label_column]
                if self._path_column is not None:
                    columns.append(self._path_column)
                table = parquet_file.read(columns=columns)
                shard_labels = table[self._label_column].to_pylist()
                if self._path_column is not None:
                    shard_paths = table[self._path_column].to_pylist()
                else:
                    shard_paths = [f"{parquet_path.name}:{row_index}" for row_index in range(len(shard_labels))]

                for row_index, raw_label in enumerate(shard_labels):
                    raw_label = self._normalize_scalar(raw_label)
                    relative_path = str(shard_paths[row_index])
                    writer.writerow((shard_index, row_index, relative_path, raw_label))
                    shard_indices.append(shard_index)
                    row_indices.append(row_index)
                    relative_paths.append(relative_path)
                    raw_labels.append(raw_label)

        if temp_path is not None:
            os.replace(temp_path, self.cache_path)

        return shard_indices, row_indices, relative_paths, raw_labels

    @staticmethod
    def _build_label_mapping(raw_labels: List[Any]) -> Tuple[List[int], Dict[Any, int]]:
        mapping: Dict[Any, int] = {}
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

    def _get_parquet_handle(self, shard_index: int) -> pq.ParquetFile:
        handle = self._parquet_handles.get(int(shard_index))
        if handle is None:
            handle = pq.ParquetFile(self._parquet_paths[int(shard_index)])
            self._parquet_handles[int(shard_index)] = handle
        return handle

    def _resolve_row_group(self, shard_index: int, row_index: int) -> Tuple[int, int]:
        offsets = self._row_group_offsets[int(shard_index)]
        row_group_index = bisect_right(offsets, int(row_index)) - 1
        row_group_start = offsets[row_group_index]
        return row_group_index, row_group_start

    def _get_row_group_table(self, shard_index: int, row_group_index: int):
        cached = self._row_group_cache.get(int(shard_index))
        if cached is not None and cached[0] == int(row_group_index):
            return cached[1]
        parquet_file = self._get_parquet_handle(shard_index)
        table = parquet_file.read_row_group(int(row_group_index), columns=[self._image_column])
        self._row_group_cache[int(shard_index)] = (int(row_group_index), table)
        return table

    def _read_image_from_shard(self, shard_index: int, row_index: int) -> Image.Image:
        row_group_index, row_group_start = self._resolve_row_group(shard_index, row_index)
        table = self._get_row_group_table(shard_index, row_group_index)
        local_row_index = int(row_index) - int(row_group_start)
        image_value = table[self._image_column][local_row_index].as_py()
        return self._decode_image_value(image_value)

    def _decode_image_value(self, value) -> Image.Image:
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
            if image_bytes is not None:
                return Image.open(io.BytesIO(image_bytes)).convert(self.color_space)
            if image_path:
                return self._open_image_path(str(image_path))

        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(value))).convert(self.color_space)

        if isinstance(value, str):
            return self._open_image_path(value)

        raise TypeError(f"Unsupported image value type from parquet: {type(value)!r}")

    def _open_image_path(self, image_path: str) -> Image.Image:
        path_obj = Path(image_path)
        if not path_obj.is_absolute():
            candidate = self.root_dir / image_path
            if candidate.exists():
                path_obj = candidate
            else:
                candidate = self.data_root / image_path
                if candidate.exists():
                    path_obj = candidate
        return Image.open(path_obj).convert(self.color_space)

    @staticmethod
    def _normalize_scalar(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return CASIAWebFaceParquetDataset._normalize_scalar(value[0])
        return value

    @classmethod
    def from_config(cls, dataset_cfg: Any, transform=None) -> "CASIAWebFaceParquetDataset":
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

        raise ValueError("CASIAWebFaceParquetDataset requires `root_dir` or (`data_root` and `rec`).")

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


class CASIAWebFaceFolderDataset(VGGFace2Dataset):
    dataset_name = "casia"
