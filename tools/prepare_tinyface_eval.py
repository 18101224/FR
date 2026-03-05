from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
from datasets import Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aligners import build_mtcnn_aligner


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinyface_root", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--probe_list", type=str, default=None)
    parser.add_argument("--gallery_list", type=str, default=None)
    parser.add_argument("--distractor_list", type=str, default=None)
    return parser.parse_args()


def find_existing_dir(root: Path, candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return None


def find_named_dir_recursive(root: Path, names: Sequence[str]) -> Path | None:
    lower_names = {name.lower() for name in names}
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name.lower() in lower_names:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def list_images(root: Path) -> List[str]:
    files: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        files.append(path.relative_to(root).as_posix())
    return files


def load_list_file(path: Path) -> List[str]:
    items: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(line)
    return items


def resolve_tinyface_splits(args: argparse.Namespace, root: Path) -> Tuple[List[str], List[str], List[str]]:
    if args.probe_list and args.gallery_list and args.distractor_list:
        probe = load_list_file(Path(args.probe_list).expanduser().resolve())
        gallery = load_list_file(Path(args.gallery_list).expanduser().resolve())
        distractor = load_list_file(Path(args.distractor_list).expanduser().resolve())
        return probe, gallery, distractor

    probe_dir = find_existing_dir(
        root,
        [
            "Probe",
            "probe",
            "probe_set",
            "Testing_Set/Probe",
            "test/probe",
        ],
    )
    if probe_dir is None:
        probe_dir = find_named_dir_recursive(root, ["Probe", "probe", "probe_set"])

    gallery_dir = find_existing_dir(
        root,
        [
            "Gallery_Match",
            "gallery_match",
            "Gallery",
            "gallery",
        ],
    )
    if gallery_dir is None:
        gallery_dir = find_named_dir_recursive(root, ["Gallery_Match", "gallery_match", "gallery"])

    distractor_dir = find_existing_dir(
        root,
        [
            "Gallery_Distractor",
            "gallery_distractor",
            "Distractor",
            "distractor",
        ],
    )
    if distractor_dir is None:
        distractor_dir = find_named_dir_recursive(root, ["Gallery_Distractor", "gallery_distractor", "distractor"])

    if probe_dir is None or gallery_dir is None or distractor_dir is None:
        raise FileNotFoundError(
            "Could not infer TinyFace split directories. "
            "Pass --probe_list/--gallery_list/--distractor_list explicitly."
        )

    return (
        [p.as_posix() for p in sorted(path.relative_to(root) for path in probe_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)],
        [p.as_posix() for p in sorted(path.relative_to(root) for path in gallery_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)],
        [p.as_posix() for p in sorted(path.relative_to(root) for path in distractor_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)],
    )


def ensure_unique_basename_keys(paths: Iterable[str]) -> None:
    seen: Dict[str, str] = {}
    duplicates: List[Tuple[str, str, str]] = []
    for path in paths:
        key = Path(path).stem
        if key in seen and seen[key] != path:
            duplicates.append((key, seen[key], path))
        else:
            seen[key] = path
    if duplicates:
        examples = duplicates[:5]
        msg = "\n".join([f"key={k} a={a} b={b}" for k, a, b in examples])
        raise ValueError(
            "TinyFace path keys are not unique by basename stem. "
            "Current evaluator matches by basename only.\n"
            f"Examples:\n{msg}"
        )


def main():
    args = parse_args()
    tinyface_root = Path(args.tinyface_root).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    if not tinyface_root.exists():
        raise FileNotFoundError(f"TinyFace root not found: {tinyface_root}")

    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
        import shutil

        shutil.rmtree(out_path)

    probe_paths, gallery_paths, distractor_paths = resolve_tinyface_splits(args, tinyface_root)
    image_paths: List[str] = []
    seen: Set[str] = set()
    for path in [*probe_paths, *gallery_paths, *distractor_paths]:
        if path in seen:
            continue
        seen.add(path)
        image_paths.append(path)
    ensure_unique_basename_keys(image_paths)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    aligner = build_mtcnn_aligner(device=device, output_size=112) if args.align else None

    images: List[Image.Image] = []
    indexes: List[int] = []
    fallback_resize = 0
    detect_fail = 0

    for idx, rel_path in enumerate(image_paths):
        image = Image.open(tinyface_root / rel_path).convert("RGB")

        if aligner is not None:
            aligned_list, _ = aligner.align_pil_batch([image])
            aligned = aligned_list[0]
            if aligned is None:
                detect_fail += 1
                image = image.resize((112, 112), Image.BILINEAR)
                fallback_resize += 1
            else:
                image = aligned
        else:
            if image.size != (112, 112):
                image = image.resize((112, 112), Image.BILINEAR)
                fallback_resize += 1

        images.append(image)
        indexes.append(idx)

        if idx > 0 and idx % 1000 == 0:
            print(f"[TinyFace] processed={idx}/{len(image_paths)}")

    dataset = Dataset.from_dict(
        {
            "image": images,
            "index": indexes,
            "path": image_paths,
        }
    )
    dataset.save_to_disk(str(out_path))

    metadata = {
        "image_paths": image_paths,
        "probe_paths": probe_paths,
        "gallery_paths": gallery_paths,
        "distractor_paths": distractor_paths,
    }
    torch.save(metadata, out_path / "metadata.pt")

    print(
        f"[TinyFace] done images={len(image_paths)} "
        f"probe={len(probe_paths)} gallery={len(gallery_paths)} distractor={len(distractor_paths)} "
        f"detect_fail={detect_fail} resized={fallback_resize} out={out_path}"
    )


if __name__ == "__main__":
    main()
