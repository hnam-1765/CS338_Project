#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
LOCAL_VIETOCR = ROOT / "vietocr"
if str(LOCAL_VIETOCR) not in sys.path:
    sys.path.insert(0, str(LOCAL_VIETOCR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VietOCR prediction on one image.")
    parser.add_argument("--img", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "hocba_vietocr_fit" / "configs" / "vgg19_transformer_hocba.yml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "hocba_vietocr_fit" / "weights" / "vgg19_transformer_hocba.pth",
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
