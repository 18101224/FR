from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from datasets import Dataset


DEFAULT_NAMES = ["lfw", "agedb_30", "cfp_fp", "cplfw", "calfw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_bin_path(bin_root: Path, name: str) -> Path:
    candidates = [
        bin_root / f"{name}.bin",
        bin_root / name / f"{name}.bin",
        bin_root / name / f"{name.lower()}.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {name}.bin under {bin_root}")


def load_bin(path: Path) -> Tuple[Sequence[bytes], Sequence[bool]]:
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")

    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        bins, is_same = payload
    else:
        raise ValueError(f"Unexpected .bin payload structure in {path}")

    if len(bins) % 2 != 0:
        raise ValueError(f"Invalid verification bin (odd image count): {path}")
    if len(is_same) * 2 != len(bins):
        raise ValueError(
            f"Pair count mismatch in {path}: len(bins)={len(bins)} len(is_same)={len(is_same)}"
        )
    return bins, is_same


def decode_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def build_dataset(bins: Sequence[bytes], is_same_pairs: Sequence[bool]) -> Dataset:
    images: List[Image.Image] = []
    indexes: List[int] = []
    is_same_flags: List[bool] = []

    for pair_idx, is_same in enumerate(is_same_pairs):
        image_idx_1 = pair_idx * 2
        image_idx_2 = image_idx_1 + 1
        images.append(decode_image(bins[image_idx_1]))
        images.append(decode_image(bins[image_idx_2]))
        indexes.extend([image_idx_1, image_idx_2])
        is_same_flags.extend([bool(is_same), bool(is_same)])

    return Dataset.from_dict(
        {
            "image": images,
            "index": indexes,
            "is_same": is_same_flags,
        }
    )


def prepare_one(name: str, bin_root: Path, out_root: Path, overwrite: bool) -> Dict[str, str]:
    bin_path = find_bin_path(bin_root, name)
    target_path = out_root / name

    if target_path.exists() and not overwrite:
        return {
            "name": name,
            "status": "skip",
            "reason": f"exists: {target_path} (use --overwrite)",
        }

    bins, is_same = load_bin(bin_path)
    dataset = build_dataset(bins, is_same)
    if target_path.exists() and overwrite:
        import shutil

        shutil.rmtree(target_path)
    dataset.save_to_disk(str(target_path))
    return {
        "name": name,
        "status": "ok",
        "images": str(len(dataset)),
        "pairs": str(len(is_same)),
        "path": str(target_path),
    }


def main():
    args = parse_args()
    bin_root = Path(args.bin_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        try:
            result = prepare_one(
                name=name,
                bin_root=bin_root,
                out_root=out_root,
                overwrite=bool(args.overwrite),
            )
            print(result)
        except Exception as e:
            print({"name": name, "status": "error", "error": str(e)})


if __name__ == "__main__":
    main()
