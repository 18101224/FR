from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aligners import build_mtcnn_aligner
from dataset import BaseFaceDataset, FaceSampleRecord, get_dataset_class


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class PreprocessingSourceDataset(Dataset):
    def __init__(self, source_dataset: BaseFaceDataset):
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> Tuple[Image.Image, FaceSampleRecord]:
        image, _ = self.source_dataset.read_sample(index)
        record = self.source_dataset.get_sample_record(index)
        return image, record


def identity_collate(batch):
    return batch


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_size", type=int, default=112)
    parser.add_argument("--skip_existing", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=500)
    return parser.parse_args()


def resolve_source_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).lower()
    if normalized.startswith("casia") and "raw" not in normalized and "parquet" not in normalized:
        return "casia_raw"
    if normalized in {"vggface2_aligned", "vgg2_aligned"}:
        return "vgg2"
    return normalized


def build_source_dataset(args: argparse.Namespace) -> BaseFaceDataset:
    source_dataset_name = resolve_source_dataset_name(args.dataset_name)
    dataset_class = get_dataset_class(source_dataset_name)
    dataset_cfg = argparse.Namespace(
        dataset_name=source_dataset_name,
        dataset_root=args.input_root,
        root_dir=args.input_root,
        split="train",
        color_space="RGB",
        repeated_augment_prob=0.0,
        repeat_same_image=False,
        architecture="",
    )
    return dataset_class.from_config(dataset_cfg, transform=None)


def build_output_path(dataset_name: str, output_root: Path, record: FaceSampleRecord) -> Path:
    dataset_name = resolve_source_dataset_name(dataset_name)
    train_root = output_root / "train"
    raw_label = str(record.raw_label)

    if dataset_name.startswith("casia") and ("raw" in dataset_name or "parquet" in dataset_name):
        source_name = Path(record.relative_path).name or f"sample_{record.record_index:08d}"
        source_name = source_name.replace(":", "_").replace("/", "_").replace("\\", "_")
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            source_name = f"{Path(source_name).stem or 'sample'}_{record.record_index:08d}.jpg"
        file_name = f"{record.record_index:08d}_{source_name}"
        return train_root / raw_label / file_name

    relative_path = Path(record.relative_path)
    return train_root / relative_path


def main():
    args = get_arguments()
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Build source dataset: name={args.dataset_name} root={args.input_root}")
    source_dataset = build_source_dataset(args)
    total_images = len(source_dataset)
    print(f"[INFO] Source dataset loaded. total_images={total_images}")
    loader = DataLoader(
        PreprocessingSourceDataset(source_dataset),
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=False,
        collate_fn=identity_collate,
    )

    aligner = build_mtcnn_aligner(device=device, output_size=args.output_size)

    saved = 0
    skipped = 0
    skipped_existing = 0
    processed = 0
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    progress = tqdm(
        total=total_images,
        desc=f"Preprocess {args.dataset_name}",
        unit="img",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
    )

    for batch in loader:
        images = [sample[0] for sample in batch]
        records = [sample[1] for sample in batch]

        aligned_images, _ = aligner.align_pil_batch(images)

        for aligned_image, record in zip(aligned_images, records):
            output_path = build_output_path(args.dataset_name, output_root, record)

            if args.skip_existing and output_path.exists():
                skipped_existing += 1
                continue

            if aligned_image is None:
                skipped += 1
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_image.save(output_path, quality=95)
            saved += 1

        processed += len(batch)
        progress.update(len(batch))
        progress.set_postfix(saved=saved, skipped=skipped, existed=skipped_existing)
        if (not is_tty) and args.log_interval > 0 and (processed % args.log_interval == 0 or processed >= total_images):
            print(
                f"[PROGRESS] processed={processed}/{total_images} "
                f"saved={saved} skipped={skipped} existed={skipped_existing}"
            )

    progress.close()

    print(
        f"Preprocessing finished. dataset={args.dataset_name} "
        f"saved={saved} skipped={skipped} skipped_existing={skipped_existing} "
        f"output_root={output_root}"
    )


if __name__ == "__main__":
    main()
