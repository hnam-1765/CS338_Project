#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cropped OCR cell image quality with blur/contrast/brightness checks."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "ocr_data",
        help="Prepared VietOCR cell dataset folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "quality_audit",
        help="Output folder for reports, contact sheets, and flagged samples.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of images to include in random manual-review contact sheets.",
    )
    parser.add_argument("--seed", type=int, default=338, help="Random seed for sampling.")
    parser.add_argument(
        "--copy-flagged",
        type=int,
        default=200,
        help="Copy up to N worst flagged images for quick inspection. Set 0 to disable.",
    )
    parser.add_argument(
        "--blur-percentile",
        type=float,
        default=2.0,
        help="Flag images with Laplacian blur score below this dataset percentile.",
    )
    parser.add_argument(
        "--contrast-percentile",
        type=float,
        default=1.0,
        help="Flag images with contrast below this dataset percentile.",
    )
    parser.add_argument(
        "--brightness-low",
        type=float,
        default=25.0,
        help="Flag images darker than this mean grayscale value.",
    )
    parser.add_argument(
        "--brightness-high",
        type=float,
        default=245.0,
        help="Flag images brighter than this mean grayscale value.",
    )
    parser.add_argument("--min-width", type=int, default=10)
    parser.add_argument("--min-height", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_annotation(path: Path) -> List[Tuple[str, str, str]]:
    split = path.stem.replace("annotation_", "")
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            image_relpath, label = line.split("\t", 1)
            rows.append((split, image_relpath, label))
    return rows


def load_rows(data_root: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for split in ("train", "val", "test"):
        annotation_path = data_root / f"annotation_{split}.txt"
        if annotation_path.exists():
            rows.extend(read_annotation(annotation_path))
    return rows


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), pct))


def tenengrad(gray: np.ndarray) -> float:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))


def image_metrics(path: Path) -> Dict[str, Any]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    lap_var_after_gaussian = float(cv2.Laplacian(gaussian, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())

    return {
        "width": w,
        "height": h,
        "aspect": float(w / h) if h else 0.0,
        "brightness": brightness,
        "contrast": contrast,
        "laplacian_var": lap_var,
        "laplacian_var_after_gaussian": lap_var_after_gaussian,
        "tenengrad": tenengrad(gray),
    }


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "flagged_samples").mkdir(parents=True, exist_ok=True)


def reasons_for(record: Dict[str, Any], thresholds: Dict[str, float], args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    if record["laplacian_var"] < thresholds["blur"]:
        reasons.append("low_laplacian_blur_score")
    if record["contrast"] < thresholds["contrast"]:
        reasons.append("low_contrast")
    if record["brightness"] < args.brightness_low:
        reasons.append("too_dark")
    if record["brightness"] > args.brightness_high:
        reasons.append("too_bright")
    if record["width"] < args.min_width:
        reasons.append("too_narrow")
    if record["height"] < args.min_height:
        reasons.append("too_short")
    return reasons


def shrink_to_box(image: Image.Image, box: Tuple[int, int]) -> Image.Image:
    max_w, max_h = box
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def create_contact_sheet(
    records: Sequence[Dict[str, Any]],
    data_root: Path,
    output_path: Path,
    title: str,
    cols: int = 5,
    cell_size: Tuple[int, int] = (220, 130),
) -> None:
    if not records:
        return
    rows = math.ceil(len(records) / cols)
    header_h = 34
    label_h = 48
    sheet = Image.new("RGB", (cols * cell_size[0], header_h + rows * cell_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)

    for idx, record in enumerate(records):
        col = idx % cols
        row = idx // cols
        x = col * cell_size[0]
        y = header_h + row * cell_size[1]
        draw.rectangle((x, y, x + cell_size[0] - 1, y + cell_size[1] - 1), outline=(210, 210, 210))

        image = Image.open(data_root / record["image"]).convert("RGB")
        thumb = shrink_to_box(image, (cell_size[0] - 12, cell_size[1] - label_h - 10))
        img_x = x + (cell_size[0] - thumb.width) // 2
        img_y = y + 6
        sheet.paste(thumb, (img_x, img_y))

        text = record["label"]
        if len(text) > 34:
            text = text[:31] + "..."
        meta = f"{record['split']} lap={record['laplacian_var']:.1f} c={record['contrast']:.1f}"
        draw.text((x + 6, y + cell_size[1] - 38), text, fill=(0, 0, 0), font=font)
        draw.text((x + 6, y + cell_size[1] - 20), meta, fill=(80, 80, 80), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "split",
        "image",
        "label",
        "width",
        "height",
        "brightness",
        "contrast",
        "laplacian_var",
        "laplacian_var_after_gaussian",
        "tenengrad",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)

    rows = load_rows(args.data_root)
    records: List[Dict[str, Any]] = []
    for split, image_relpath, label in rows:
        metrics = image_metrics(args.data_root / image_relpath)
        record = {"split": split, "image": image_relpath, "label": label, **metrics}
        records.append(record)

    thresholds = {
        "blur": percentile([r["laplacian_var"] for r in records], args.blur_percentile),
        "contrast": percentile([r["contrast"] for r in records], args.contrast_percentile),
    }

    flagged: List[Dict[str, Any]] = []
    reason_counts = Counter()
    for record in records:
        reasons = reasons_for(record, thresholds, args)
        record["reasons"] = "|".join(reasons)
        if reasons:
            flagged.append(record)
            reason_counts.update(reasons)

    write_csv(args.output / "quality_all.csv", records)
    write_csv(args.output / "quality_flagged.csv", flagged)

    random.seed(args.seed)
    sample_records = random.sample(records, min(args.sample_size, len(records)))
    create_contact_sheet(
        sample_records,
        args.data_root,
        args.output / "contact_sheet_random_300.jpg",
        f"Random review sample: {len(sample_records)} cells",
    )

    worst_blur = sorted(records, key=lambda r: r["laplacian_var"])[: min(args.sample_size, len(records))]
    create_contact_sheet(
        worst_blur,
        args.data_root,
        args.output / "contact_sheet_worst_blur_300.jpg",
        f"Worst blur-score sample: {len(worst_blur)} cells",
    )

    if args.copy_flagged:
        for idx, record in enumerate(sorted(flagged, key=lambda r: (r["laplacian_var"], r["contrast"]))[: args.copy_flagged]):
            src = args.data_root / record["image"]
            suffix = Path(record["image"]).suffix
            dst = args.output / "flagged_samples" / f"{idx:04d}_{record['split']}_{record['laplacian_var']:.1f}_{record['contrast']:.1f}{suffix}"
            shutil.copy2(src, dst)

    summary = {
        "data_root": str(args.data_root),
        "total_images": len(records),
        "flagged_images": len(flagged),
        "thresholds": thresholds,
        "reason_counts": dict(reason_counts),
        "by_split": dict(Counter(r["split"] for r in records)),
        "flagged_by_split": dict(Counter(r["split"] for r in flagged)),
        "outputs": {
            "all_csv": str(args.output / "quality_all.csv"),
            "flagged_csv": str(args.output / "quality_flagged.csv"),
            "random_contact_sheet": str(args.output / "contact_sheet_random_300.jpg"),
            "worst_blur_contact_sheet": str(args.output / "contact_sheet_worst_blur_300.jpg"),
        },
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
