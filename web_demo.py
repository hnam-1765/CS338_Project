#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vgg19_transformer_hocba.yml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "vgg19_transformer_hocba.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def add_local_vietocr_to_path() -> None:
    for candidate in (PROJECT_ROOT / "vietocr", PROJECT_ROOT.parent / "vietocr"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


class PredictorService:
    def __init__(self, config_path: Path, weights_path: Path, device: Optional[str]) -> None:
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self._predictor: Optional[Any] = None
        self._loaded_weights_mtime: Optional[float] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        config_exists = self.config_path.is_file()
        checkpoint_exists = self.weights_path.is_file()

        if not config_exists:
            state = "error"
            message = "The model configuration file was not found."
        elif not checkpoint_exists:
            state = "waiting"
            message = "The demo is running in preview mode, waiting for a model checkpoint to be added."
        elif self._load_error:
            state = "error"
            message = self._load_error
        elif self._predictor is None:
            state = "available"
            message = "The checkpoint is available. The model will load on the first OCR request."
        else:
            state = "ready"
            message = "The model is loaded and ready for OCR."

        return {
            "state": state,
            "message": message,
            "config_exists": config_exists,
            "checkpoint_exists": checkpoint_exists,
            "model_loaded": self._predictor is not None,
            "config_path": str(self.config_path),
            "weights_path": str(self.weights_path),
            "device": self.device or "from config",
        }

    def _load_predictor(self) -> Any:
        if not self.config_path.is_file():
            raise RuntimeError(f"Configuration file not found: {self.config_path}")
        if not self.weights_path.is_file():
            raise RuntimeError(f"Checkpoint not found: {self.weights_path}")

        weights_mtime = self.weights_path.stat().st_mtime
        if self._predictor is not None and self._loaded_weights_mtime == weights_mtime:
            return self._predictor

        add_local_vietocr_to_path()
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError(
                "The checkpoint was found, but VietOCR could not be imported. "
                "Install the VietOCR repository in the virtual environment."
            ) from exc

        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["weights"] = str(self.weights_path)
        if self.device:
            config["device"] = self.device

        self._predictor = Predictor(Cfg(config))
        self._loaded_weights_mtime = weights_mtime
        self._load_error = None
        return self._predictor

    def predict(self, image: Image.Image) -> Optional[str]:
        if not self.weights_path.is_file():
            return None

        with self._lock:
            try:
                predictor = self._load_predictor()
                return str(predictor.predict(image))
            except Exception as exc:
                self._load_error = f"Could not load or run the model: {exc}"
                raise RuntimeError(self._load_error) from exc


def load_uploaded_image() -> tuple[Image.Image, str]:
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("Select a cell image before running OCR.")

    raw = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")
    if not raw:
        raise ValueError("The image file is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    return image, upload.filename


def image_to_data_uri(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((1100, 540), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def create_app(service: PredictorService) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index() -> str:
        return render_template("web_demo.html", model=service.status())

    @app.post("/")
    def predict() -> tuple[str, int] | str:
        try:
            image, filename = load_uploaded_image()
            prediction = service.predict(image)
            return render_template(
                "web_demo.html",
                model=service.status(),
                filename=filename,
                image_size=f"{image.width} x {image.height} px",
                preview_uri=image_to_data_uri(image),
                prediction=prediction,
                preview_only=prediction is None,
            )
        except (ValueError, RuntimeError) as exc:
            return render_template("web_demo.html", model=service.status(), error=str(exc)), 400

    @app.get("/health")
    def health() -> Any:
        return jsonify(service.status())

    @app.errorhandler(413)
    def upload_too_large(_: Any) -> tuple[str, int]:
        return (
            render_template(
                "web_demo.html",
                model=service.status(),
                error="The image exceeds the 10 MB upload limit.",
            ),
            413,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HocBa VietOCR web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None, help="Override model device, e.g. cpu or cuda:0.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PredictorService(args.config.resolve(), args.weights.resolve(), args.device)
    app = create_app(service)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
