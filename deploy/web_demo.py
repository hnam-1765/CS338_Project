#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vgg19_transformer_hocba.yml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "vgg19_transformer_hocba.pth"
DEFAULT_DET_MODEL_DIR = PROJECT_ROOT / "models" / "paddleocr" / "ch_PP-OCRv4_det_infer"
SAMPLES_ROOT = PROJECT_ROOT / "static" / "samples"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_DETECTIONS = 80
BOX_PADDING = 3
SAMPLE_TITLES = {
    "grade-94": "Grade",
    "rating-good": "Rating",
    "school-year": "School year",
    "school-name": "School name",
}
SAMPLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

Point = Tuple[float, float]
Box = List[Point]


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
            message = "The VietOCR configuration file was not found."
        elif not checkpoint_exists:
            state = "waiting"
            message = "Waiting for the VietOCR checkpoint."
        elif self._load_error:
            state = "error"
            message = self._load_error
        elif self._predictor is None:
            state = "available"
            message = "The checkpoint is available. VietOCR will load on the first OCR request."
        else:
            state = "ready"
            message = "VietOCR is loaded and ready."

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
            raise RuntimeError(f"VietOCR configuration file not found: {self.config_path}")
        if not self.weights_path.is_file():
            raise RuntimeError(f"VietOCR checkpoint not found: {self.weights_path}")

        weights_mtime = self.weights_path.stat().st_mtime
        if self._predictor is not None and self._loaded_weights_mtime == weights_mtime:
            return self._predictor

        add_local_vietocr_to_path()
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError("VietOCR could not be imported in this environment.") from exc

        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["weights"] = str(self.weights_path)
        if self.device:
            config["device"] = self.device

        self._predictor = Predictor(Cfg(config))
        self._loaded_weights_mtime = weights_mtime
        self._load_error = None
        return self._predictor

    def predict(self, image: Image.Image) -> str:
        with self._lock:
            try:
                predictor = self._load_predictor()
                return str(predictor.predict(image))
            except Exception as exc:
                self._load_error = f"Could not load or run VietOCR: {exc}"
                raise RuntimeError(self._load_error) from exc


class DetectorService:
    def __init__(self, det_model_dir: Path) -> None:
        self.det_model_dir = det_model_dir
        self._detector: Optional[Any] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        det_model_exists = self.det_model_dir.is_dir()
        if self._load_error:
            return {
                "state": "error",
                "message": self._load_error,
                "detector_loaded": False,
                "det_model_exists": det_model_exists,
                "det_model_dir": str(self.det_model_dir),
            }
        if not det_model_exists:
            return {
                "state": "waiting",
                "message": "Waiting for the local PaddleOCR detection model.",
                "detector_loaded": False,
                "det_model_exists": False,
                "det_model_dir": str(self.det_model_dir),
            }
        if self._detector is None:
            return {
                "state": "available",
                "message": "The local PaddleOCR detector will load on the first page detection request.",
                "detector_loaded": False,
                "det_model_exists": True,
                "det_model_dir": str(self.det_model_dir),
            }
        return {
            "state": "ready",
            "message": "PaddleOCR detector is loaded and ready.",
            "detector_loaded": True,
            "det_model_exists": True,
            "det_model_dir": str(self.det_model_dir),
        }

    def _load_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        if not self.det_model_dir.is_dir():
            raise RuntimeError(f"PaddleOCR detection model directory not found: {self.det_model_dir}")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddlepaddle and paddleocr to enable page detection."
            ) from exc

        try:
            self._detector = PaddleOCR(
                det_model_dir=str(self.det_model_dir),
                use_angle_cls=False,
                use_gpu=False,
                lang="ch",
                det=True,
                rec=False,
                cls=False,
                show_log=False,
            )
        except TypeError:
            self._detector = PaddleOCR(
                det_model_dir=str(self.det_model_dir),
                use_angle_cls=False,
                use_gpu=False,
                lang="ch",
                det=True,
                rec=False,
                cls=False,
            )
        self._load_error = None
        return self._detector

    def detect(self, image: Image.Image) -> List[Box]:
        with self._lock:
            try:
                detector = self._load_detector()
                checked_image = np.asarray(image.convert("RGB"))
                dt_boxes, _ = detector.text_detector(checked_image)
                raw = [] if dt_boxes is None else [box.tolist() for box in dt_boxes]
                boxes = extract_boxes(raw)
                return sort_boxes(boxes)[:MAX_DETECTIONS]
            except Exception as exc:
                self._load_error = f"Could not run PaddleOCR detection: {exc}"
                raise RuntimeError(self._load_error) from exc


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def looks_like_point(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2 and is_number(value[0]) and is_number(value[1])


def looks_like_box(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 4 and all(looks_like_point(point) for point in value[:4])


def normalize_box(value: Any) -> Box:
    return [(float(point[0]), float(point[1])) for point in list(value)[:4]]


def extract_boxes(raw: Any) -> List[Box]:
    boxes: List[Box] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if looks_like_box(value):
            boxes.append(normalize_box(value))
            return
        if isinstance(value, dict):
            for key in ("dt_polys", "rec_polys", "det_polygons", "boxes", "points"):
                if key in value:
                    visit(value[key])
            return
        if isinstance(value, np.ndarray):
            visit(value.tolist())
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(raw)
    unique: List[Box] = []
    seen = set()
    for box in boxes:
        signature = tuple((round(x, 1), round(y, 1)) for x, y in box)
        if signature not in seen:
            seen.add(signature)
            unique.append(box)
    return unique


def sort_boxes(boxes: Sequence[Box]) -> List[Box]:
    def key(box: Box) -> Tuple[float, float]:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        return (min(ys), min(xs))

    return sorted(boxes, key=key)


def bounding_rect(box: Box, width: int, height: int, padding: int = BOX_PADDING) -> Tuple[int, int, int, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    left = max(0, int(np.floor(min(xs))) - padding)
    top = max(0, int(np.floor(min(ys))) - padding)
    right = min(width, int(np.ceil(max(xs))) + padding)
    bottom = min(height, int(np.ceil(max(ys))) + padding)
    if right <= left or bottom <= top:
        raise ValueError("Detected an invalid bounding box.")
    return left, top, right, bottom


def crop_box(image: Image.Image, box: Box) -> Image.Image:
    return image.crop(bounding_rect(box, image.width, image.height)).convert("RGB")


def box_to_payload(box: Box, width: int, height: int) -> Dict[str, Any]:
    left, top, right, bottom = bounding_rect(box, width, height, padding=0)
    polygon = [{"x": x, "y": y} for x, y in box]
    return {
        "polygon": polygon,
        "bbox": {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
            "x_pct": 100 * left / width,
            "y_pct": 100 * top / height,
            "width_pct": 100 * (right - left) / width,
            "height_pct": 100 * (bottom - top) / height,
        },
    }


def image_to_data_uri(image: Image.Image, max_size: Tuple[int, int] = (1400, 1000), quality: int = 90) -> str:
    preview = image.copy()
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_uploaded_page() -> tuple[Image.Image, str]:
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("Upload a report card page image before running the demo.")

    raw = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The image exceeds the 15 MB upload limit.")
    if not raw:
        raise ValueError("The image file is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    return image, upload.filename


def load_sample_image(sample_id: str) -> tuple[Image.Image, str]:
    sample = get_sample_images().get(sample_id)
    if sample is None:
        raise ValueError("The selected sample is not available.")

    sample_path = SAMPLES_ROOT / sample["filename"]
    if not sample_path.is_file():
        raise ValueError("The selected sample file is missing.")

    with Image.open(sample_path) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return image, f"Sample: {sample['title']}"


def get_sample_images() -> Dict[str, Dict[str, str]]:
    samples: Dict[str, Dict[str, str]] = {}
    if not SAMPLES_ROOT.is_dir():
        return samples

    for path in sorted(SAMPLES_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SAMPLE_EXTENSIONS:
            continue
        sample_id = path.stem
        samples[sample_id] = {
            "filename": path.name,
            "title": SAMPLE_TITLES.get(sample_id, sample_id.replace("-", " ").replace("_", " ").title()),
        }
    return samples


def load_requested_page() -> tuple[Image.Image, str]:
    upload = request.files.get("image")
    if upload is not None and upload.filename:
        return load_uploaded_page()

    sample_id = request.form.get("sample_id", "").strip()
    if sample_id:
        return load_sample_image(sample_id)

    raise ValueError("Upload an image or choose one of the samples.")


def run_page_pipeline(image: Image.Image, detector: DetectorService, predictor: PredictorService) -> List[Dict[str, Any]]:
    boxes = detector.detect(image)
    results: List[Dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        crop = crop_box(image, box)
        text = predictor.predict(crop)
        payload = box_to_payload(box, image.width, image.height)
        payload.update(
            {
                "index": index,
                "text": text,
                "crop_uri": image_to_data_uri(crop, max_size=(260, 120), quality=88),
            }
        )
        results.append(payload)
    return results


def create_app(predictor: PredictorService, detector: DetectorService) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    def status() -> Dict[str, Any]:
        return {"ocr": predictor.status(), "detector": detector.status()}

    def render_page(**context: Any) -> str:
        return render_template(
            "web_demo.html",
            status=status(),
            max_detections=MAX_DETECTIONS,
            samples=get_sample_images(),
            **context,
        )

    @app.get("/")
    def index() -> str:
        return render_page()

    @app.post("/")
    def predict_page() -> tuple[str, int] | str:
        try:
            image, filename = load_requested_page()
            results = run_page_pipeline(image, detector, predictor)
            return render_page(
                filename=filename,
                image_size=f"{image.width} x {image.height} px",
                preview_uri=image_to_data_uri(image),
                results=results,
                results_json=json.dumps(results, ensure_ascii=False),
                count=len(results),
            )
        except (ValueError, RuntimeError) as exc:
            return render_page(error=str(exc)), 400

    @app.get("/health")
    def health() -> Any:
        return jsonify(status())

    @app.errorhandler(413)
    def upload_too_large(_: Any) -> tuple[str, int]:
        return render_page(error="The image exceeds the 15 MB upload limit."), 413

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HocBa full-page OCR web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--det-model-dir", type=Path, default=DEFAULT_DET_MODEL_DIR)
    parser.add_argument("--device", default=None, help="Override VietOCR device, e.g. cpu or cuda:0.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = PredictorService(args.config.resolve(), args.weights.resolve(), args.device)
    detector = DetectorService(args.det_model_dir.resolve())
    app = create_app(predictor, detector)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
