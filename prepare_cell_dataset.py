#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageOps


IMAGE_EXT = ".jpg"
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop each labeled cell bounding box into a VietOCR line dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets"),
        help="Source dataset folder containing images/, label/, and dataset_metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ocr_data",
        help="Output folder for cropped cell images and VietOCR annotation files.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Pixel padding added around each cell crop after EXIF orientation is applied.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep boxes whose label is empty. By default empty cells are skipped.",
    )
    parser.add_argument(
        "--no-strip-labels",
        action="store_true",
        help="Do not strip leading/trailing spaces from labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder before regenerating it.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_label(label: str, strip: bool = True) -> str:
    label = label.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return label.strip() if strip else label


def field_type_from_id(item_id: str) -> str:
    parts = item_id.split("_")
    return "_".join(parts[1:-1]) if len(parts) > 2 else item_id


def clip_box(
    coord: Iterable[float], width: int, height: int, padding: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in coord]
    left = max(0, math.floor(x1) - padding)
    top = max(0, math.floor(y1) - padding)
    right = min(width, math.ceil(x2) + padding)
    bottom = min(height, math.ceil(y2) + padding)
    return left, top, right, bottom


def annotation_line(image_relpath: str, label: str) -> str:
    return f"{image_relpath}\t{label}\n"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def crop_cells(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = args.source
    output_dir = args.output
    metadata_path = source_dir / "dataset_metadata.json"
    metadata = load_json(metadata_path)

    prepare_output(output_dir, args.overwrite)

    split_lines: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    cell_records: List[Dict[str, Any]] = []
    stats = Counter()
    by_split = Counter()
    by_category = Counter()
    by_type = Counter()
    exif_orientation = Counter()

    strip_labels = not args.no_strip_labels

    for item in metadata:
        split = item.get("split", "train")
        if split not in split_lines:
            split_lines[split] = []

        image_file = item["image_file"]
        image_path = source_dir / "images" / image_file
        label_path = source_dir / "label" / item["json_file"]
        label_data = load_json(label_path)

        with Image.open(image_path) as raw_image:
            exif_orientation[raw_image.getexif().get(274, 1)] += 1
            image = ImageOps.exif_transpose(raw_image).convert("RGB")

        width, height = image.size
        item_id = item["id"]
        field_type = field_type_from_id(item_id)
        category = item.get("category", "unknown")

        for box_idx, box in enumerate(label_data.get("boxes", [])):
            stats["boxes_total"] += 1
            label = clean_label(str(box.get("label", "")), strip=strip_labels)

            if not label and not args.keep_empty:
                stats["empty_skipped"] += 1
                continue

            coord = box.get("coordinate", [])
            if len(coord) != 4:
                stats["bad_coordinate"] += 1
                continue

            left, top, right, bottom = clip_box(coord, width, height, args.padding)
            if right <= left or bottom <= top:
                stats["invalid_crop"] += 1
                continue

            split_image_dir = output_dir / "images" / split / category
            split_image_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{item_id}_cell{box_idx:03d}{IMAGE_EXT}"
            crop_path = split_image_dir / crop_name
            crop_relpath = crop_path.relative_to(output_dir).as_posix()

            image.crop((left, top, right, bottom)).save(crop_path, quality=95)
            split_lines[split].append(annotation_line(crop_relpath, label))

            record = {
                "image": crop_relpath,
                "label": label,
                "split": split,
                "category": category,
                "grade": item.get("grade", "unknown"),
                "field_type": field_type,
                "source_image": image_file,
                "source_json": item["json_file"],
                "box_index": box_idx,
                "coordinate": [float(v) for v in coord],
                "crop_box": [left, top, right, bottom],
            }
            cell_records.append(record)

            stats["cells_written"] += 1
            by_split[split] += 1
            by_category[category] += 1
            by_type[field_type] += 1

    for split, lines in split_lines.items():
        annotation_path = output_dir / f"annotation_{split}.txt"
        with annotation_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

    with (output_dir / "metadata_cells.jsonl").open("w", encoding="utf-8") as f:
        for record in cell_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    chars = sorted({char for record in cell_records for char in record["label"]})
    with (output_dir / "vocab_chars.txt").open("w", encoding="utf-8") as f:
        f.write("".join(chars))
        f.write("\n")

    summary = {
        "source": str(source_dir),
        "output": str(output_dir),
        "padding": args.padding,
        "stats": dict(stats),
        "by_split": dict(by_split),
        "by_category": dict(by_category),
        "top_field_types": by_type.most_common(30),
        "raw_exif_orientation": dict(exif_orientation),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = crop_cells(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
