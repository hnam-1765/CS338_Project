#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def add_local_vietocr_to_path() -> None:
    for candidate in (PROJECT_ROOT / "vietocr", WORKSPACE_ROOT / "vietocr"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


add_local_vietocr_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VietOCR prediction on one image.")
    parser.add_argument("--img", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vgg19_transformer_hocba.yml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "vgg19_transformer_hocba.pth",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["weights"] = str(args.weights)
    if args.device:
        config["device"] = args.device

    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    predictor = Predictor(Cfg(config))
    image = ImageOps.exif_transpose(Image.open(args.img)).convert("RGB")
    print(predictor.predict(image))


if __name__ == "__main__":
    main()
