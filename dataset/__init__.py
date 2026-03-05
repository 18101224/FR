from __future__ import annotations

from argparse import Namespace
from typing import Dict, Optional, Type

from .base import BaseFaceDataset, FaceSampleRecord
from .casia import CASIAWebFaceFolderDataset, CASIAWebFaceParquetDataset
from .ms1mv2_subset import MS1MV2SubsetDataset
from .ms1mv3 import MS1MV3Dataset
from .vggface2 import VGGFace2Dataset
from .webface import WebFace4MDataset, WebFace12MDataset
from torch.utils.data import DataLoader, DistributedSampler

DATASET_REGISTRY: Dict[str, Type[BaseFaceDataset]] = {
    "ms1mv2_subset": MS1MV2SubsetDataset,
    "ms1mv2-subset": MS1MV2SubsetDataset,
    "ms1m-v2-subset": MS1MV2SubsetDataset,
    "ms1m_v2_subset": MS1MV2SubsetDataset,
    "msv2_subset": MS1MV2SubsetDataset,
    "msv2-subset": MS1MV2SubsetDataset,
    "ms1mv3": MS1MV3Dataset,
    "ms1m_v3": MS1MV3Dataset,
    "ms1m-v3": MS1MV3Dataset,
    "msv3": MS1MV3Dataset,
    "webface4m": WebFace4MDataset,
    "webface_4m": WebFace4MDataset,
    "webface-4m": WebFace4MDataset,
    "webface12m": WebFace12MDataset,
    "webface_12m": WebFace12MDataset,
    "webface-12m": WebFace12MDataset,
    "vgg2": VGGFace2Dataset,
    "vgg2_raw": VGGFace2Dataset,
    "vgg2_aligned": VGGFace2Dataset,
    "vggface2": VGGFace2Dataset,
    "vggface2_raw": VGGFace2Dataset,
    "vggface2_aligned": VGGFace2Dataset,
    "vgg-face2": VGGFace2Dataset,
    "casia": CASIAWebFaceFolderDataset,
    "casia_aligned": CASIAWebFaceFolderDataset,
    "casia-webface": CASIAWebFaceFolderDataset,
    "casia-webface-aligned": CASIAWebFaceFolderDataset,
    "casia_webface": CASIAWebFaceFolderDataset,
    "casia_webface_aligned": CASIAWebFaceFolderDataset,
    "casiawebface": CASIAWebFaceFolderDataset,
    "casia_raw": CASIAWebFaceParquetDataset,
    "casia_parquet": CASIAWebFaceParquetDataset,
    "casia-parquet": CASIAWebFaceParquetDataset,
    "casia-webface-raw": CASIAWebFaceParquetDataset,
    "casia_webface_raw": CASIAWebFaceParquetDataset,
}


def get_dataset_class(name: str) -> Type[BaseFaceDataset]:
    normalized_name = name.lower()
    if normalized_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset `{name}`. Available datasets: {available}")
    return DATASET_REGISTRY[normalized_name]


def build_train_dataset(dataset_args: Namespace, train_transform=None) -> BaseFaceDataset:
    dataset_name = getattr(dataset_args, "name", getattr(dataset_args, "dataset_name", "ms1mv3"))
    dataset_class = get_dataset_class(str(dataset_name))
    return dataset_class.from_config(dataset_args, transform=train_transform)


def get_train_dataset(dataset_args: Namespace, train_transform=None, aug_args: Optional[Namespace] = None, local_rank: int = 0):
    del local_rank
    augmentation_version = getattr(aug_args, "augmentation_version", "none") if aug_args else "none"
    if augmentation_version != "none":
        raise NotImplementedError("Dataset-side augmentation is not implemented yet. Use `augmentation_version: none`.")
    dataset = build_train_dataset(dataset_args=dataset_args, train_transform=train_transform)
    num_classes = _resolve_num_classes(dataset=dataset, dataset_args=dataset_args)
    setattr(dataset_args, "num_classes", num_classes)
    return dataset, num_classes


def _resolve_num_classes(dataset: BaseFaceDataset, dataset_args: Namespace) -> int:
    dataset_num_classes = getattr(dataset, "num_classes", None)
    if dataset_num_classes is not None:
        return int(dataset_num_classes)

    args_num_classes = getattr(dataset_args, "num_classes", None)
    if args_num_classes is not None:
        return int(args_num_classes)

    raise ValueError("Could not determine `num_classes`. Provide it explicitly or use a dataset class that exposes `num_classes`.")

def get_loader(args, train_transform=None, use_distributed_sampler: bool = False):
    dataset, num_classes = get_train_dataset(args, train_transform=train_transform)
    sampler = None
    if use_distributed_sampler:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, num_classes, len(loader)


__all__ = [
    "BaseFaceDataset",
    "FaceSampleRecord",
    "CASIAWebFaceFolderDataset",
    "CASIAWebFaceParquetDataset",
    "MS1MV2SubsetDataset",
    "MS1MV3Dataset",
    "VGGFace2Dataset",
    "WebFace4MDataset",
    "WebFace12MDataset",
    "build_train_dataset",
    "get_dataset_class",
    "get_loader",
    "get_train_dataset",
]
