#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Set


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create filtered VietOCR annotation files after manual quality review."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "ocr_data",
        help="VietOCR cell dataset folder containing annotation_train/val/test.txt.",
    )
    parser.add_argument(
        "--flagged-csv",
        type=Path,
        default=PROJECT_ROOT / "quality_audit" / "quality_flagged.csv",
        help="CSV produced by audit_cell_quality.py.",
    )
    parser.add_argument(
        "--exclude-list",
        type=Path,
        default=None,
        help="Text file with one image relative path per line to remove.",
    )
    parser.add_argument(
        "--exclude-all-flagged",
        action="store_true",
        help="Remove every image listed in flagged-csv. Use only after review.",
    )
    parser.add_argument(
        "--suffix",
        default="filtered",
        help="Output annotation suffix, e.g. filtered -> annotation_train_filtered.txt.",
    )
    return parser.parse_args()


def load_flagged(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["image"] for row in csv.DictReader(f)}


def load_exclude_list(path: Path) -> Set[str]:
    excluded: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            excluded.add(line)
    return excluded


def filter_annotation(input_path: Path, output_path: Path, excluded: Set[str]) -> Dict[str, int]:
    kept = 0
    removed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            image_relpath = line.rstrip("\n").split("\t", 1)[0]
            if image_relpath in excluded:
                removed += 1
                continue
            dst.write(line)
            kept += 1
    return {"kept": kept, "removed": removed}


def main() -> None:
    args = parse_args()
    excluded: Set[str] = set()

    if args.exclude_all_flagged:
        excluded.update(load_flagged(args.flagged_csv))
    if args.exclude_list:
        excluded.update(load_exclude_list(args.exclude_list))

    if not excluded:
        print("No images selected for exclusion. Filtered annotations will match originals.")

    totals = {}
    for split in ("train", "val", "test"):
        input_path = args.data_root / f"annotation_{split}.txt"
        output_path = args.data_root / f"annotation_{split}_{args.suffix}.txt"
        totals[split] = filter_annotation(input_path, output_path, excluded)
        print(split, totals[split], "->", output_path)

    print("excluded_unique_images", len(excluded))


if __name__ == "__main__":
    main()
