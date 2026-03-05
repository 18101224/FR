from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from datasets import Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aligners import build_mtcnn_aligner
from aligners.retinaface_aligner import aligner_helper


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ijb_root", type=str, required=True)
    parser.add_argument("--subset", type=str, default="ijbc", choices=["ijbb", "ijbc", "both"])
    parser.add_argument("--out_root", type=str, default="/data/mj/facerec_val")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--face_tid_mid", type=str, default=None)
    parser.add_argument("--template_pair_label", type=str, default=None)
    parser.add_argument("--name_5pts_score", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_paths(root: Path, subset_upper: str) -> Dict[str, List[Path]]:
    lower = subset_upper.lower()
    dashed = subset_upper[:3] + "-" + subset_upper[3:]
    return {
        "image_dir": [
            root / subset_upper / "loose_crop",
            root / subset_upper / "images",
            root / dashed / "loose_crop",
            root / dashed / "images",
            root / lower / "loose_crop",
            root / lower / "images",
            root / f"{subset_upper}_images",
            root / "IJB_release" / subset_upper / "loose_crop",
            root / "IJB_release" / subset_upper / "images",
            root / "IJB_release" / dashed / "loose_crop",
            root / "IJB_release" / dashed / "images",
            root / "IJB_release" / lower / "loose_crop",
            root / "IJB_release" / lower / "images",
            root / "loose_crop",
            root / "images",
        ],
        "face_tid_mid": [
            root / subset_upper / "meta" / f"{lower}_face_tid_mid.txt",
            root / dashed / "meta" / f"{lower}_face_tid_mid.txt",
            root / lower / "meta" / f"{lower}_face_tid_mid.txt",
            root / "IJB_release" / subset_upper / "meta" / f"{lower}_face_tid_mid.txt",
            root / "IJB_release" / dashed / "meta" / f"{lower}_face_tid_mid.txt",
            root / "IJB_release" / lower / "meta" / f"{lower}_face_tid_mid.txt",
            root / "meta" / f"{lower}_face_tid_mid.txt",
        ],
        "template_pair_label": [
            root / subset_upper / "meta" / f"{lower}_template_pair_label.txt",
            root / dashed / "meta" / f"{lower}_template_pair_label.txt",
            root / lower / "meta" / f"{lower}_template_pair_label.txt",
            root / "IJB_release" / subset_upper / "meta" / f"{lower}_template_pair_label.txt",
            root / "IJB_release" / dashed / "meta" / f"{lower}_template_pair_label.txt",
            root / "IJB_release" / lower / "meta" / f"{lower}_template_pair_label.txt",
            root / "meta" / f"{lower}_template_pair_label.txt",
        ],
        "name_5pts_score": [
            root / subset_upper / "meta" / f"{lower}_name_5pts_score.txt",
            root / dashed / "meta" / f"{lower}_name_5pts_score.txt",
            root / lower / "meta" / f"{lower}_name_5pts_score.txt",
            root / "IJB_release" / subset_upper / "meta" / f"{lower}_name_5pts_score.txt",
            root / "IJB_release" / dashed / "meta" / f"{lower}_name_5pts_score.txt",
            root / "IJB_release" / lower / "meta" / f"{lower}_name_5pts_score.txt",
            root / "meta" / f"{lower}_name_5pts_score.txt",
        ],
    }


def find_first(paths: Sequence[Path], required: bool = True) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"Could not find required file/dir among candidates: {paths}")
    return None


def find_recursive_file(root: Path, file_name: str) -> Path | None:
    matches = sorted(root.rglob(file_name), key=lambda p: (len(p.parts), str(p)))
    if not matches:
        return None
    return matches[0]


def parse_tokens(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    line = line.replace(",", " ")
    return [token for token in line.split() if token]


def parse_face_tid_mid(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    image_paths: List[str] = []
    templates: List[int] = []
    medias: List[int] = []
    with path.open("r") as f:
        for line in f:
            tokens = parse_tokens(line)
            if len(tokens) < 3:
                continue
            image_paths.append(tokens[0])
            templates.append(int(tokens[1]))
            medias.append(int(tokens[2]))
    if len(image_paths) == 0:
        raise ValueError(f"No entries in {path}")
    return image_paths, np.asarray(templates, dtype=np.int64), np.asarray(medias, dtype=np.int64)


def parse_pair_labels(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1: List[int] = []
    p2: List[int] = []
    label: List[int] = []
    with path.open("r") as f:
        for line in f:
            tokens = parse_tokens(line)
            if len(tokens) < 3:
                continue
            p1.append(int(tokens[0]))
            p2.append(int(tokens[1]))
            label.append(int(tokens[2]))
    if len(p1) == 0:
        raise ValueError(f"No entries in {path}")
    return np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64), np.asarray(label, dtype=np.int64)


def parse_name_5pts_score(path: Path | None) -> Dict[str, Tuple[np.ndarray, float]]:
    result: Dict[str, Tuple[np.ndarray, float]] = {}
    if path is None or not path.exists():
        return result
    with path.open("r") as f:
        for line in f:
            tokens = parse_tokens(line)
            if len(tokens) < 12:
                continue
            name = tokens[0]
            points = np.asarray([float(x) for x in tokens[1:11]], dtype=np.float32).reshape(5, 2)
            score = float(tokens[11])
            result[name] = (points, score)
    return result


def _as_posix_path(path_str: str) -> str:
    return path_str.replace("\\", "/")


def resolve_image_dir(
    ijb_root: Path,
    candidates: Sequence[Path],
    sample_rel_path: str,
) -> Path:
    rel = _as_posix_path(sample_rel_path)
    for candidate in candidates:
        if candidate is None:
            continue
        if not candidate.exists():
            continue
        if (candidate / rel).exists():
            return candidate

    sample_name = Path(rel).name
    rel_parts = Path(rel).parts
    for match in sorted(ijb_root.rglob(sample_name)):
        match_posix = match.as_posix()
        if rel and match_posix.endswith(rel):
            base = Path(match_posix[: -len(rel)]).resolve()
            if base.exists():
                return base

    # Final fallback: choose the first candidate that exists.
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image_dir from sample path: {sample_rel_path}. "
        f"Pass --image_dir explicitly."
    )


def align_from_5pts(image: Image.Image, points: np.ndarray, output_size: int = 112) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    reference = aligner_helper.reference_landmark().astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), reference, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("estimateAffinePartial2D returned None")
    aligned = cv2.warpAffine(
        rgb,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Image.fromarray(aligned)


def open_image(image_dir: Path, rel_path: str) -> Image.Image:
    path = image_dir / rel_path
    if not path.exists():
        alt = image_dir / Path(rel_path).name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def build_one_subset(
    ijb_root: Path,
    subset_upper: str,
    out_root: Path,
    align: bool,
    device: str,
    overwrite: bool,
    image_dir_override: Path | None = None,
    face_tid_mid_override: Path | None = None,
    template_pair_label_override: Path | None = None,
    name_5pts_score_override: Path | None = None,
) -> None:
    cands = candidate_paths(ijb_root, subset_upper)
    face_tid_mid = face_tid_mid_override or find_first(cands["face_tid_mid"], required=False)
    if face_tid_mid is None:
        face_tid_mid = find_recursive_file(ijb_root, f"{subset_upper.lower()}_face_tid_mid.txt")
    if face_tid_mid is None:
        raise FileNotFoundError("Could not find face_tid_mid file. Pass --face_tid_mid explicitly.")

    pair_label = template_pair_label_override or find_first(cands["template_pair_label"], required=False)
    if pair_label is None:
        pair_label = find_recursive_file(ijb_root, f"{subset_upper.lower()}_template_pair_label.txt")
    if pair_label is None:
        raise FileNotFoundError("Could not find template_pair_label file. Pass --template_pair_label explicitly.")

    name_5pts_score = name_5pts_score_override or find_first(cands["name_5pts_score"], required=False)
    if name_5pts_score is None:
        name_5pts_score = find_recursive_file(ijb_root, f"{subset_upper.lower()}_name_5pts_score.txt")

    image_paths, templates, medias = parse_face_tid_mid(face_tid_mid)
    image_dir = (
        image_dir_override
        if image_dir_override is not None
        else resolve_image_dir(ijb_root=ijb_root, candidates=cands["image_dir"], sample_rel_path=image_paths[0])
    )

    p1, p2, label = parse_pair_labels(pair_label)
    ldmk_score = parse_name_5pts_score(name_5pts_score)

    eval_name = f"{subset_upper}_gt_aligned"
    out_path = out_root / eval_name
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists (use --overwrite)")
        import shutil

        shutil.rmtree(out_path)

    aligner = build_mtcnn_aligner(device=device, output_size=112) if align else None

    images: List[Image.Image] = []
    indexes: List[int] = []
    faceness_scores: List[float] = []
    detect_fail = 0
    resized_fallback = 0
    aligned_by_5pts = 0

    for idx, rel_path in enumerate(image_paths):
        image = open_image(image_dir=image_dir, rel_path=rel_path)
        score = 1.0

        if align:
            key = rel_path
            entry = ldmk_score.get(key)
            if entry is None:
                entry = ldmk_score.get(Path(rel_path).name)
            if entry is not None:
                try:
                    image = align_from_5pts(image=image, points=entry[0], output_size=112)
                    score = float(entry[1])
                    aligned_by_5pts += 1
                except Exception:
                    entry = None
            if entry is None and aligner is not None:
                aligned_list, score_tensor = aligner.align_pil_batch([image])
                aligned = aligned_list[0]
                if aligned is None:
                    detect_fail += 1
                    image = image.resize((112, 112), Image.BILINEAR)
                    resized_fallback += 1
                else:
                    image = aligned
                    score = float(score_tensor[0].item())
        else:
            if image.size != (112, 112):
                image = image.resize((112, 112), Image.BILINEAR)
                resized_fallback += 1

        images.append(image)
        indexes.append(idx)
        faceness_scores.append(score)

        if idx > 0 and idx % 2000 == 0:
            print(f"[{subset_upper}] processed={idx}/{len(image_paths)}")

    dataset = Dataset.from_dict(
        {
            "image": images,
            "index": indexes,
        }
    )
    dataset.save_to_disk(str(out_path))

    metadata = {
        "faceness_scores": np.asarray(faceness_scores, dtype=np.float32),
        "templates": templates.astype(np.int64),
        "medias": medias.astype(np.int64),
        "label": label.astype(np.int64),
        "p1": p1.astype(np.int64),
        "p2": p2.astype(np.int64),
    }
    torch.save(metadata, out_path / "metadata.pt")

    print(
        f"[{subset_upper}] done images={len(images)} pairs={len(label)} "
        f"align={align} by_5pts={aligned_by_5pts} detect_fail={detect_fail} "
        f"resize_fallback={resized_fallback} out={out_path}"
    )


def main():
    args = parse_args()
    ijb_root = Path(args.ijb_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    subsets: List[str]
    if args.subset == "both":
        subsets = ["IJBB", "IJBC"]
    else:
        subsets = [args.subset.upper()]

    for subset in subsets:
        build_one_subset(
            ijb_root=ijb_root,
            subset_upper=subset,
            out_root=out_root,
            align=bool(args.align),
            device=device,
            overwrite=bool(args.overwrite),
            image_dir_override=Path(args.image_dir).expanduser().resolve() if args.image_dir else None,
            face_tid_mid_override=Path(args.face_tid_mid).expanduser().resolve() if args.face_tid_mid else None,
            template_pair_label_override=Path(args.template_pair_label).expanduser().resolve() if args.template_pair_label else None,
            name_5pts_score_override=Path(args.name_5pts_score).expanduser().resolve() if args.name_5pts_score else None,
        )


if __name__ == "__main__":
    main()
