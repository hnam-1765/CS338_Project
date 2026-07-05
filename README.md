# HocBa OCR

<p align="center"><em>Vietnamese report-card OCR with PaddleOCR text detection and fine-tuned VietOCR recognition.</em></p>

<p align="center">
  <a href="https://huggingface.co/spaces/SaitoHoujou/HocBa-OCR_Demo_Web"><img alt="Hugging Face Space" src="https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffcc4d?logo=huggingface&amp;logoColor=black"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Framework" src="https://img.shields.io/badge/OCR-PaddleOCR%20%2B%20VietOCR-green">
</p>

HocBa OCR is a research and deployment project for extracting text from Vietnamese school report cards. The pipeline detects text regions on a full-page report-card image with PaddleOCR, crops each detected region, then recognizes Vietnamese text with a VGG19-bn + Transformer VietOCR model fine-tuned on report-card cell crops.

Live demo: https://huggingface.co/spaces/SaitoHoujou/HocBa-OCR_Demo_Web

## What This Repository Contains

| Area | Purpose |
| --- | --- |
| `prepare_cell_dataset.py` | Crop labeled report-card cells into VietOCR training samples |
| `audit_cell_quality.py` | Measure blur, contrast, brightness, and create review sheets |
| `filter_annotations.py` | Build filtered annotations after manual quality review |
| `train_hocba.py` | Fine-tune VietOCR on prepared report-card cells |
| `predict_one.py` | Run recognition on one cropped cell image |
| `web_demo.py` | Lightweight local cell-level OCR demo |
| `deploy/` | Hugging Face Space deployment for full-page OCR |
| `configs/` | VietOCR training/inference configuration |
| `quality_audit/` | Audit reports and visual sheets from the prepared cell dataset |

The public-facing demo lives in `deploy/`. The root-level scripts are for dataset preparation, quality control, model training, and local experimentation.

## Pipeline Overview

```text
Annotated report-card pages
        |
        v
prepare_cell_dataset.py
        |
        v
Cropped OCR cells + VietOCR annotations
        |
        v
audit_cell_quality.py + optional filter_annotations.py
        |
        v
train_hocba.py
        |
        v
Fine-tuned VietOCR checkpoint
        |
        v
deploy/web_demo.py: PaddleOCR detection -> VietOCR recognition -> web results
```

## Demo

The Hugging Face Space accepts a full-page report-card image and returns detected text boxes with recognized text.

- Space: https://huggingface.co/spaces/SaitoHoujou/HocBa-OCR_Demo_Web
- Runtime: Docker Space
- App entrypoint: `deploy/web_demo.py`
- Detection model: local PaddleOCR `ch_PP-OCRv4_det_infer`
- Recognition model: fine-tuned VietOCR checkpoint expected at `deploy/weights/vgg19_transformer_hocba.pth`

## Dataset Preparation

The preparation script expects a source dataset with:

```text
datasets/
|-- images/
|-- label/
`-- dataset_metadata.json
```

Create cropped cell images and VietOCR annotation files:

```bash
python prepare_cell_dataset.py \
  --source datasets \
  --output ocr_data \
  --padding 2 \
  --overwrite
```

Output:

```text
ocr_data/
|-- images/{train,val,test}/...
|-- annotation_train.txt
|-- annotation_val.txt
|-- annotation_test.txt
|-- metadata_cells.jsonl
|-- summary.json
`-- vocab_chars.txt
```

Annotation format:

```text
relative/path/to/cell.jpg<TAB>label
```

The script applies EXIF orientation correction before cropping and skips empty labels by default.

## Quality Audit

OCR quality is sensitive to blur, low contrast, overexposure, tight crops, label noise, and train/validation leakage. This repo includes an audit step so bad crops can be reviewed before training.

Run the audit:

```bash
python audit_cell_quality.py \
  --data-root ocr_data \
  --output quality_audit \
  --sample-size 300 \
  --overwrite
```

Current audit summary:

| Metric | Value |
| --- | ---: |
| Total cropped cells | 10,026 |
| Train / Val / Test | 7,325 / 1,027 / 1,674 |
| Flagged for review | 244 |
| Main checks | Laplacian blur, contrast, brightness, minimum size |

 
 
After manual review, create an exclusion list and filtered annotations:

```bash
python filter_annotations.py \
  --data-root ocr_data \
  --exclude-list quality_audit/exclude_images.txt
```

Then update `configs/vgg19_transformer_hocba.yml`:

```yaml
dataset:
  train_annotation: annotation_train_filtered.txt
  valid_annotation: annotation_val_filtered.txt
```

## Training

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e ../vietocr
```

Train on GPU:

```bash
python train_hocba.py \
  --config configs/vgg19_transformer_hocba.yml \
  --device cuda:0 \
  --rebuild-lmdb
```

Small CPU smoke run:

```bash
python train_hocba.py \
  --device cpu \
  --batch-size 4 \
  --iters 100 \
  --valid-every 50 \
  --rebuild-lmdb
```

Default training settings:

| Setting | Value |
| --- | --- |
| Backbone | `vgg19_bn` |
| Sequence model | Transformer |
| Image height | 32 |
| Max image width | 768 |
| Batch size | 32 |
| Iterations | 6,000 |
| Max learning rate | `1e-4` |
| Validation interval | 250 |
| Label smoothing | 0.1 |
| Early stopping | enabled, patience 8 validations |

Best weights are exported to:

```text
weights/vgg19_transformer_hocba.pth
```

## Single-Image Prediction

Run recognition on one cropped cell:

```bash
python predict_one.py \
  --img ocr_data/images/test/<category>/<file>.jpg \
  --weights weights/vgg19_transformer_hocba.pth \
  --device cuda:0
```

## Local Web Demo

The root-level demo is a simple cell-level OCR interface:

```bash
python web_demo.py \
  --weights weights/vgg19_transformer_hocba.pth \
  --device cpu \
  --host 0.0.0.0 \
  --port 5000
```

Open `http://localhost:5000`.

## Hugging Face Space Deployment

The full-page demo is packaged under `deploy/`:

```text
deploy/
|-- Dockerfile
|-- README.md
|-- web_demo.py
|-- configs/
|-- models/paddleocr/ch_PP-OCRv4_det_infer/
|-- static/samples/
`-- templates/
```

Expected deployment-only files:

```text
deploy/weights/vgg19_transformer_hocba.pth
deploy/vietocr/
```

Upload `deploy/` to the Hugging Face Space repository. The Space metadata is stored in `deploy/README.md`.

## Factors That Affect Accuracy

- Detection quality: missed or merged PaddleOCR boxes propagate directly to recognition errors.
- Crop quality: blur, low contrast, skew, shadows, and tight boxes reduce VietOCR accuracy.
- Vocabulary coverage: Vietnamese diacritics, punctuation, and numeric grades must exist in the config vocabulary.
- Domain shift: printed text, handwriting styles, tables, stamps, and scanned/phone-captured pages differ in texture.
- Label quality: noisy cell labels can make validation accuracy look unstable and hurt fine-tuning.
- Split strategy: pages or students should not leak across train/validation/test when measuring generalization.

## Repository Notes

- The project name used in documentation is `HocBa OCR`; the original course folder may still be named `CS338_Project`.
- Root scripts focus on training and research workflow.
- `deploy/` is intentionally self-contained for the Hugging Face Space.
- Large generated artifacts such as `ocr_data/`, `weights/`, `checkpoints/`, and `logs/` should not be committed unless they are meant to be released.

## Citation / Acknowledgements

This project builds on:

- VietOCR VGG19-bn + Transformer recognition.
- PaddleOCR text detection.
- A custom Vietnamese report-card cell dataset prepared for CS338 coursework.

If you use this repository, please cite the underlying OCR frameworks and acknowledge this project where appropriate.
