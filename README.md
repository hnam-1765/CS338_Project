# HocBa VietOCR Fine-tuning

Folder này dùng pretrained `VGG19-bn + Transformer` của VietOCR để fine-tune trên dataset học bạ dạng cell.

## 1. Chuẩn bị cropped cell dataset

Chạy từ root repo `/home/namhoai/WorkSpace/Hoctap/CS338`:

```bash
python3 hocba_vietocr_fit/prepare_cell_dataset.py \
  --source datasets \
  --output hocba_vietocr_fit/ocr_data \
  --padding 2 \
  --overwrite
```

Output chính:

```text
hocba_vietocr_fit/ocr_data/images/{train,val,test}/...
hocba_vietocr_fit/ocr_data/annotation_train.txt
hocba_vietocr_fit/ocr_data/annotation_val.txt
hocba_vietocr_fit/ocr_data/annotation_test.txt
hocba_vietocr_fit/ocr_data/metadata_cells.jsonl
hocba_vietocr_fit/ocr_data/summary.json
```

Mỗi dòng annotation có format VietOCR:

```text
relative/path/to/cell.jpg<TAB>label
```

Script tự dùng `ImageOps.exif_transpose()` trước khi crop, vì một số ảnh học bạ có EXIF orientation.

## 1.5. Kiểm tra chất lượng ảnh cell

Sau khi crop, chạy audit blur/contrast/brightness và tạo contact sheet để xem thủ công:

```bash
python3 hocba_vietocr_fit/audit_cell_quality.py \
  --data-root hocba_vietocr_fit/ocr_data \
  --output hocba_vietocr_fit/quality_audit \
  --sample-size 300 \
  --overwrite
```

Output:

```text
hocba_vietocr_fit/quality_audit/quality_all.csv
hocba_vietocr_fit/quality_audit/quality_flagged.csv
hocba_vietocr_fit/quality_audit/contact_sheet_random_300.jpg
hocba_vietocr_fit/quality_audit/contact_sheet_worst_blur_300.jpg
hocba_vietocr_fit/quality_audit/flagged_samples/
```

Metric chính là `laplacian_var`: ảnh càng mờ thì score càng thấp. Script chọn ngưỡng theo percentile của chính dataset để tránh đặt ngưỡng cứng sai với ảnh học bạ.

Nếu sau khi xem ảnh xấu bạn muốn loại một số ảnh khỏi train, tạo file text, mỗi dòng là relative path trong `quality_flagged.csv`, ví dụ:

```text
images/train/hoctap/xxx_cell000.jpg
images/val/nhanxet/yyy_cell003.jpg
```

Rồi tạo annotation filtered:

```bash
python3 hocba_vietocr_fit/filter_annotations.py \
  --data-root hocba_vietocr_fit/ocr_data \
  --exclude-list hocba_vietocr_fit/quality_audit/exclude_images.txt
```

Sau đó đổi config:

```yaml
dataset:
  train_annotation: annotation_train_filtered.txt
  valid_annotation: annotation_val_filtered.txt
```

## 2. Fine-tune VGG19 Transformer

Nếu chưa cài dependency, cài tối thiểu:

```bash
pip install -r hocba_vietocr_fit/requirements.txt
pip install -e vietocr
```

Trên Kaggle P100, nếu log báo `Tesla P100 ... sm_60 is not compatible` hoặc
`CUDA error: no kernel image is available for execution on the device`, hãy cài
PyTorch CUDA 11.8 bản cũ hơn trước khi train:

```bash
python -m pip install --upgrade --force-reinstall \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Sau đó chạy lại từ đầu kernel/session để chắc chắn `torch` mới được import.

Nếu Kaggle báo `AttributeError: np.sctypes was removed in the NumPy 2.0 release`
khi import `imgaug`, pin NumPy về 1.x:

```bash
python -m pip install --force-reinstall numpy==1.26.4 imgaug==0.4.0
```

Train:

```bash
python3 hocba_vietocr_fit/train_hocba.py \
  --config hocba_vietocr_fit/configs/vgg19_transformer_hocba.yml \
  --device cuda:0 \
  --rebuild-lmdb
```

Chạy thử CPU hoặc giảm tải:

```bash
python3 hocba_vietocr_fit/train_hocba.py \
  --device cpu \
  --batch-size 4 \
  --iters 100 \
  --valid-every 50 \
  --rebuild-lmdb
```

Best weights sẽ được lưu tại:

```text
hocba_vietocr_fit/weights/vgg19_transformer_hocba.pth
```

Config mặc định hiện tại cho dataset cell này:

```text
batch_size: 32
iters: 6000
max_lr: 1e-4
valid_every: 250
label_smoothing: 0.1
early_stopping patience: 8 validations
wandb: tắt mặc định, bật bằng --wandb hoặc sửa config
```

Notebook Kaggle nằm tại:

```text
hocba_vietocr_fit/kaggle_train_vgg19_transformer_hocba.ipynb
```

Notebook luôn copy code sang `/kaggle/working` trước khi chạy, vì `/kaggle/input` là read-only.

## 3. Predict một ảnh

```bash
python3 hocba_vietocr_fit/predict_one.py \
  --img hocba_vietocr_fit/ocr_data/images/test/hoctap/<file>.jpg \
  --weights hocba_vietocr_fit/weights/vgg19_transformer_hocba.pth \
  --device cuda:0
```

## Ghi chú

- `prepare_cell_dataset.py` bỏ qua cell có label rỗng.
- Config giữ nguyên vocab mặc định của VietOCR để load pretrained head tốt nhất.
- `train_hocba.py` patch tạm lỗi off-by-one trong hàm tạo LMDB của repo VietOCR local, để không bị mất sample cuối.
