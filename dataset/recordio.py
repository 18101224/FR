from __future__ import annotations

import atexit
import csv
import numbers
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if "bool" not in np.__dict__:
    np.bool = np.bool_
if "object" not in np.__dict__:
    np.object = np.object_
if "float" not in np.__dict__:
    np.float = float

import mxnet as mx
from PIL import Image

from .base import BaseFaceDataset, FaceSampleRecord


class MXRecordFaceDataset(BaseFaceDataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        color_space: str = "RGB",
        repeated_augment_prob: float = 0.0,
        repeat_same_image: bool = False,
        record_file: str = "train.rec",
        index_file: str = "train.idx",
        cache_file: str = "train.tsv",
    ) -> None:
        super().__init__(
            transform=transform,
            color_space=color_space,
            repeated_augment_prob=repeated_augment_prob,
            repeat_same_image=repeat_same_image,
        )
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.record_path = self.root_dir / record_file
        self.index_path = self.root_dir / index_file
        self.cache_path = self.root_dir / cache_file
        self._record_reader: Optional[mx.recordio.MXIndexedRecordIO] = None

        self._validate_files()
        self._available_record_indices = self._load_available_record_indices()
        self._record_indices, self._relative_paths, self._raw_labels = self._load_or_create_cache()
        self._class_indices, self._label_mapping = self._build_label_mapping(self._raw_labels)
        self._indices_by_class = self._build_indices_by_class(self._class_indices)
        self._inverse_label_mapping = {value: key for key, value in self._label_mapping.items()}
        atexit.register(self.close)

    @property
    def num_classes(self) -> int:
        return len(self._label_mapping)

    @property
    def label_mapping(self) -> Dict[int, int]:
        return dict(self._label_mapping)

    @property
    def class_to_raw_label(self) -> Dict[int, int]:
        return dict(self._inverse_label_mapping)

    def __len__(self) -> int:
        return len(self._record_indices)

    def get_sample_record(self, index: int) -> FaceSampleRecord:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")

        return FaceSampleRecord(
            record_index=int(self._record_indices[index]),
            raw_label=int(self._raw_labels[index]),
            class_index=int(self._class_indices[index]),
            relative_path=self._relative_paths[index],
        )

    def read_sample(self, index: int) -> Tuple[Image.Image, int]:
        sample_record = self.get_sample_record(index)
        record_reader = self._ensure_record_reader()
        packed = record_reader.read_idx(sample_record.record_index)
        _, image_buffer = mx.recordio.unpack(packed)
        image = mx.image.imdecode(image_buffer).asnumpy()
        return Image.fromarray(image), sample_record.class_index

    def close(self) -> None:
        if self._record_reader is None:
            return

        close_fn = getattr(self._record_reader, "close", None)
        if callable(close_fn):
            close_fn()
        self._record_reader = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_record_reader"] = None
        return state

    def _validate_files(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        if not self.record_path.exists():
            raise FileNotFoundError(f"Record file does not exist: {self.record_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file does not exist: {self.index_path}")

    def _ensure_record_reader(self) -> mx.recordio.MXIndexedRecordIO:
        if self._record_reader is None:
            self._record_reader = mx.recordio.MXIndexedRecordIO(
                str(self.index_path),
                str(self.record_path),
                "r",
            )
        return self._record_reader

    def _open_record_reader(self) -> mx.recordio.MXIndexedRecordIO:
        return mx.recordio.MXIndexedRecordIO(str(self.index_path), str(self.record_path), "r")

    def _load_available_record_indices(self) -> np.ndarray:
        record_reader = self._open_record_reader()
        try:
            packed = record_reader.read_idx(0)
            header, _ = mx.recordio.unpack(packed)
            if header.flag > 0:
                return np.arange(1, int(header.label[0]), dtype=np.int64)
            return np.asarray(sorted(record_reader.keys), dtype=np.int64)
        finally:
            close_fn = getattr(record_reader, "close", None)
            if callable(close_fn):
                close_fn()

    def _load_or_create_cache(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        if self.cache_path.exists():
            return self._load_cache()
        return self._scan_recordio_and_cache()

    def _load_cache(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        record_indices: List[int] = []
        relative_paths: List[str] = []
        raw_labels: List[int] = []

        with self.cache_path.open("r", newline="") as cache_handle:
            reader = csv.reader(cache_handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if len(row) != 3:
                    raise ValueError(f"Invalid cache row in {self.cache_path}: {row}")
                record_indices.append(int(row[0]))
                relative_paths.append(row[1])
                raw_labels.append(int(float(row[2])))

        return (
            np.asarray(record_indices, dtype=np.int64),
            relative_paths,
            np.asarray(raw_labels, dtype=np.int64),
        )

    def _scan_recordio_and_cache(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        record_indices: List[int] = []
        relative_paths: List[str] = []
        raw_labels: List[int] = []

        record_reader = self._open_record_reader()
        temp_path: Optional[Path] = None

        try:
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

                for record_index in self._available_record_indices:
                    packed = record_reader.read_idx(int(record_index))
                    header, _ = mx.recordio.unpack(packed)
                    raw_label = self._parse_label(header.label)
                    relative_path = f"{raw_label}/{int(record_index)}.jpg"

                    writer.writerow((int(record_index), relative_path, raw_label))
                    record_indices.append(int(record_index))
                    relative_paths.append(relative_path)
                    raw_labels.append(raw_label)

            if temp_path is not None:
                os.replace(temp_path, self.cache_path)
        finally:
            close_fn = getattr(record_reader, "close", None)
            if callable(close_fn):
                close_fn()

        return (
            np.asarray(record_indices, dtype=np.int64),
            relative_paths,
            np.asarray(raw_labels, dtype=np.int64),
        )

    def _build_label_mapping(self, raw_labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        mapping: Dict[int, int] = {}
        class_indices = np.empty(len(raw_labels), dtype=np.int64)

        for index, raw_label in enumerate(raw_labels):
            raw_label = int(raw_label)
            class_index = mapping.setdefault(raw_label, len(mapping))
            class_indices[index] = class_index

        return class_indices, mapping

    @staticmethod
    def _build_indices_by_class(class_indices: np.ndarray) -> Dict[int, np.ndarray]:
        grouped_indices: Dict[int, List[int]] = {}
        for dataset_index, class_index in enumerate(class_indices):
            grouped_indices.setdefault(int(class_index), []).append(int(dataset_index))
        return {
            class_index: np.asarray(indices, dtype=np.int64)
            for class_index, indices in grouped_indices.items()
        }

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

    @staticmethod
    def _parse_label(label_value) -> int:
        if isinstance(label_value, numbers.Number):
            return int(label_value)
        return int(label_value[0])
