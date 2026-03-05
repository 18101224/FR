from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aligners import build_mtcnn_aligner


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ijbs_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        yield path


def main():
    args = parse_args()
    ijbs_root = Path(args.ijbs_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not ijbs_root.exists():
        raise FileNotFoundError(f"IJB-S root not found: {ijbs_root}")
    if out_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_root} (use --overwrite)")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    aligner = build_mtcnn_aligner(device=device, output_size=112)

    paths = list(iter_images(ijbs_root))
    detect_fail = 0
    saved = 0

    for idx, path in enumerate(paths):
        rel = path.relative_to(ijbs_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(path).convert("RGB")
        aligned_list, _ = aligner.align_pil_batch([image])
        aligned = aligned_list[0]
        if aligned is None:
            detect_fail += 1
            aligned = image.resize((112, 112), Image.BILINEAR)
        aligned.save(out_path, quality=95)
        saved += 1

        if idx > 0 and idx % 2000 == 0:
            print(f"[IJB-S] processed={idx}/{len(paths)} saved={saved} detect_fail={detect_fail}")

    print(f"[IJB-S] done total={len(paths)} saved={saved} detect_fail={detect_fail} out={out_root}")


if __name__ == "__main__":
    main()
