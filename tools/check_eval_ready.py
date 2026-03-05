from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


REQUIRED: Dict[str, Tuple[str, Set[str]]] = {
    "lfw": ("verification", {"image", "index", "is_same"}),
    "agedb_30": ("verification", {"image", "index", "is_same"}),
    "cfp_fp": ("verification", {"image", "index", "is_same"}),
    "cplfw": ("verification", {"image", "index", "is_same"}),
    "calfw": ("verification", {"image", "index", "is_same"}),
    "IJBB_gt_aligned": ("ijbbc", {"image", "index"}),
    "IJBC_gt_aligned": ("ijbbc", {"image", "index"}),
    "tinyface_aligned_pad_0.1": ("tinyface", {"image", "index", "path"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/mj")
    parser.add_argument("--facerec_val", type=str, default="facerec_val")
    return parser.parse_args()


def pick_dataset_path(root: Path, facerec_val: str, name: str) -> Path:
    candidates = [
        root / facerec_val / name,
        root / name,
        root / name.lower(),
        root / name.upper(),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_columns(path: Path):
    from datasets import Dataset, DatasetDict, load_from_disk

    obj = load_from_disk(str(path))
    if isinstance(obj, Dataset):
        return set(obj.column_names), len(obj)
    if isinstance(obj, DatasetDict):
        if len(obj) == 0:
            return set(), 0
        split = next(iter(obj.keys()))
        columns = set(obj[split].column_names)
        length = sum(len(obj[s]) for s in obj.keys())
        return columns, length
    raise TypeError(f"Unsupported object type from load_from_disk: {type(obj)!r}")


def find_bin_paths(root: Path) -> Iterable[Tuple[str, Path | None]]:
    bin_files = ["lfw.bin", "agedb_30.bin", "cfp_fp.bin", "cplfw.bin", "calfw.bin"]
    for file_name in bin_files:
        candidates = [
            root / file_name,
            root / "eval_bins" / file_name,
            root / "facerec_val" / file_name,
        ]
        found = next((path for path in candidates if path.exists()), None)
        yield file_name, found


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    print(f"[ROOT] {root}")
    print("\n== run_v1 eval format check ==")

    try:
        import datasets  # noqa: F401
        datasets_ready = True
        datasets_import_err = ""
    except Exception as e:
        datasets_ready = False
        datasets_import_err = str(e)

    ready_count = 0
    for name, (eval_type, required_cols) in REQUIRED.items():
        path = pick_dataset_path(root, args.facerec_val, name)
        exists = path.exists()
        load_ok = False
        cols_ok = False
        meta_ok = False
        n = None
        columns: Set[str] = set()
        error = "-"

        if exists and datasets_ready:
            try:
                columns, n = load_columns(path)
                load_ok = True
            except Exception as e:
                error = str(e).splitlines()[-1]
        elif exists and not datasets_ready:
            error = f"datasets import fail: {datasets_import_err}"

        cols_ok = required_cols.issubset(columns) if load_ok else False
        meta_ok = (path / "metadata.pt").is_file() if eval_type in {"ijbbc", "tinyface"} else True
        ready = exists and load_ok and cols_ok and meta_ok
        ready_count += int(ready)

        print(
            f"{name:24} ready={ready} exists={exists} load={load_ok} "
            f"cols={cols_ok} meta={meta_ok} n={n} path={path}"
        )
        if exists and not load_ok:
            print(f"  load_err: {error}")
        if load_ok and not cols_ok:
            print(f"  has_cols={sorted(columns)} need={sorted(required_cols)}")

    print(f"\nready {ready_count}/{len(REQUIRED)}")
    print("\n== .bin availability ==")
    for file_name, found in find_bin_paths(root):
        print(f"{file_name:12} found={bool(found)} path={str(found) if found else '-'}")


if __name__ == "__main__":
    main()
