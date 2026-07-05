#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def add_local_vietocr_to_path() -> None:
    for candidate in (PROJECT_ROOT / "vietocr", WORKSPACE_ROOT / "vietocr"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


add_local_vietocr_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VietOCR VGG19-bn Transformer on HocBa cells.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vgg19_transformer_hocba.yml",
        help="Full VietOCR config file.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--iters", type=int, default=None, help="Override trainer.iters.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override trainer.batch_size.")
    parser.add_argument("--valid-every", type=int, default=None, help="Override trainer.valid_every.")
    parser.add_argument("--metrics", type=int, default=None, help="Override trainer.metrics sample count.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override trainer.label_smoothing.")
    parser.add_argument("--early-patience", type=int, default=None, help="Override trainer.early_stopping.patience.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=None, help="Override wandb.project.")
    parser.add_argument("--wandb-run-name", default=None, help="Override wandb.name.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of loading the VGG19 Transformer pretrained weights.",
    )
    parser.add_argument(
        "--rebuild-lmdb",
        action="store_true",
        help="Remove train/valid LMDB folders before training so annotations are rebuilt.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.device:
        config["device"] = args.device
    if args.iters is not None:
        config["trainer"]["iters"] = args.iters
    if args.batch_size is not None:
        config["trainer"]["batch_size"] = args.batch_size
    if args.valid_every is not None:
        config["trainer"]["valid_every"] = args.valid_every
    if args.metrics is not None:
        config["trainer"]["metrics"] = args.metrics
    if args.label_smoothing is not None:
        config["trainer"]["label_smoothing"] = args.label_smoothing
    if args.early_patience is not None:
        config["trainer"].setdefault("early_stopping", {})["patience"] = args.early_patience
    if args.wandb:
        config.setdefault("wandb", {})["enabled"] = True
    if args.wandb_project:
        config.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_run_name:
        config.setdefault("wandb", {})["name"] = args.wandb_run_name
    return config


def remove_lmdb_dirs(config: Dict[str, Any]) -> None:
    dataset_name = config["dataset"]["name"]
    for prefix in ("train", "valid"):
        lmdb_path = Path(f"{prefix}_{dataset_name}")
        if lmdb_path.exists():
            import shutil

            shutil.rmtree(lmdb_path)
            print(f"removed {lmdb_path}")


def patch_vietocr_lmdb_sample_count() -> None:
    """Fix an off-by-one in this vendored VietOCR createDataset implementation."""
    import cv2
    import lmdb
    import numpy as np
    from tqdm import tqdm

    import vietocr.loader.dataloader as dataloader
    import vietocr.tool.create_dataset as create_dataset

    def fixed_create_dataset(outputPath: str, root_dir: str, annotation_path: str) -> None:
        annotation_file = os.path.join(root_dir, annotation_path)
        with open(annotation_file, "r", encoding="utf-8") as ann_file:
            annotations = [line.rstrip("\n").split("\t", 1) for line in ann_file if line.strip()]

        env = lmdb.open(outputPath, map_size=1099511627776)
        cache = {}
        cnt = 0
        errors = 0

        for image_file, label in tqdm(annotations, ncols=100, desc=f"Create {outputPath}"):
            image_path = os.path.join(root_dir, image_file)
            if not os.path.exists(image_path):
                errors += 1
                continue

            with open(image_path, "rb") as f:
                image_bin = f.read()

            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            image = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                errors += 1
                continue

            img_h, img_w = image.shape[:2]
            cache[f"image-{cnt:09d}"] = image_bin
            cache[f"label-{cnt:09d}"] = label.encode()
            cache[f"path-{cnt:09d}"] = image_file.encode()
            cache[f"dim-{cnt:09d}"] = np.array([img_h, img_w], dtype=np.int32).tobytes()
            cnt += 1

            if cnt % 1000 == 0:
                create_dataset.writeCache(env, cache)
                cache = {}

        cache["num-samples"] = str(cnt).encode()
        create_dataset.writeCache(env, cache)
        if errors:
            print(f"Remove {errors} invalid images")
        print(f"Created dataset with {cnt} samples")

    create_dataset.createDataset = fixed_create_dataset
    dataloader.createDataset = fixed_create_dataset


def configure_label_smoothing(trainer: Any, config: Dict[str, Any]) -> None:
    from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss

    smoothing = float(config["trainer"].get("label_smoothing", 0.1))
    trainer.criterion = LabelSmoothingLoss(
        len(trainer.vocab), padding_idx=trainer.vocab.pad, smoothing=smoothing
    )
    print(f"label_smoothing={smoothing}")


def setup_wandb(config: Dict[str, Any]) -> Optional[Any]:
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is enabled but not installed; continuing without wandb logging.")
        return None

    run = wandb.init(
        project=wandb_config.get("project", "hocba-vietocr"),
        name=wandb_config.get("name", None),
        config=config,
    )
    return run


def log_line(trainer: Any, message: str) -> None:
    print(message)
    if hasattr(trainer, "logger"):
        trainer.logger.log(message)


def train_with_callbacks(trainer: Any, config: Dict[str, Any], wandb_run: Optional[Any]) -> None:
    total_loss = 0.0
    total_loader_time = 0.0
    total_gpu_time = 0.0
    best_acc = -1.0
    validations_without_improvement = 0

    early_config = config["trainer"].get("early_stopping", {})
    early_enabled = bool(early_config.get("enabled", True))
    patience = int(early_config.get("patience", 8))
    min_delta = float(early_config.get("min_delta", 0.0005))

    data_iter = iter(trainer.train_gen)
    for _ in range(trainer.num_iters):
        trainer.iter += 1
        start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.train_gen)
            batch = next(data_iter)

        total_loader_time += time.time() - start
        start = time.time()
        loss = trainer.step(batch)
        total_gpu_time += time.time() - start

        total_loss += loss
        trainer.train_losses.append((trainer.iter, loss))

        if trainer.iter % trainer.print_every == 0:
            avg_loss = total_loss / trainer.print_every
            lr = trainer.optimizer.param_groups[0]["lr"]
            message = (
                f"iter: {trainer.iter:06d} - train loss: {avg_loss:.3f} - "
                f"lr: {lr:.2e} - load time: {total_loader_time:.2f} - gpu time: {total_gpu_time:.2f}"
            )
            log_line(trainer, message)
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "time/load": total_loader_time,
                        "time/gpu": total_gpu_time,
                    },
                    step=trainer.iter,
                )

            total_loss = 0.0
            total_loader_time = 0.0
            total_gpu_time = 0.0

        if trainer.valid_annotation and trainer.iter % trainer.valid_every == 0:
            val_loss = trainer.validate()
            acc_full_seq, acc_per_char = trainer.precision(trainer.metrics)
            message = (
                f"iter: {trainer.iter:06d} - valid loss: {val_loss:.3f} - "
                f"acc full seq: {acc_full_seq:.4f} - acc per char: {acc_per_char:.4f}"
            )
            log_line(trainer, message)

            if wandb_run:
                wandb_run.log(
                    {
                        "valid/loss": val_loss,
                        "valid/acc_full_seq": acc_full_seq,
                        "valid/acc_per_char": acc_per_char,
                        "valid/best_acc_full_seq": max(best_acc, acc_full_seq),
                    },
                    step=trainer.iter,
                )

            if acc_full_seq > best_acc + min_delta:
                trainer.save_weights(trainer.export_weights)
                trainer.save_checkpoint(trainer.checkpoint)
                best_acc = acc_full_seq
                validations_without_improvement = 0
                log_line(trainer, f"new best full-sequence accuracy: {best_acc:.4f}")
            else:
                validations_without_improvement += 1
                log_line(
                    trainer,
                    "early stopping wait: "
                    f"{validations_without_improvement}/{patience} "
                    f"(best acc full seq: {best_acc:.4f})",
                )

            if early_enabled and validations_without_improvement >= patience:
                log_line(trainer, f"early stopping at iter {trainer.iter:06d}")
                break

    if wandb_run:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    patch_vietocr_lmdb_sample_count()

    if args.rebuild_lmdb:
        remove_lmdb_dirs(config)

    from vietocr.model.trainer import Trainer
    from vietocr.tool.config import Cfg

    trainer = Trainer(Cfg(config), pretrained=not args.no_pretrained)
    configure_label_smoothing(trainer, config)
    wandb_run = setup_wandb(config)
    train_with_callbacks(trainer, config, wandb_run)


if __name__ == "__main__":
    main()
