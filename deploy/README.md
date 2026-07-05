---
title: HocBa OCR Demo
emoji: 📄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# HocBa OCR Demo

Upload a Vietnamese report-card page, detect text regions with PaddleOCR, then recognize each crop with a fine-tuned VietOCR model.

This Space is the deployment package for the `HocBa OCR` project. The app runs a full-page OCR pipeline:

```text
Input page image -> PaddleOCR detection -> cropped text regions -> VietOCR recognition -> detected text list
```

Repository: https://github.com/hnam-1765/CS338_Project
