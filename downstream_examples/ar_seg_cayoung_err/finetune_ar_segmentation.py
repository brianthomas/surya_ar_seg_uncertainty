#!/usr/bin/env python3
"""
Fine-tuning / baseline script for active-region segmentation.

Forked from ``downstream_examples/template/3_finetune_template_1D.py``. As in the template,
only ``build_datasets()`` and ``build_model()`` carry task-specific content;
``build_trainer()`` and ``main()`` are unchanged.

Two models share one config, one dataset and one metrics class, so their numbers are
directly comparable. Run both from this directory:

    # 14-parameter per-pixel logistic regression (CPU is fine)
    python finetune_ar_segmentation.py --train_baseline --no-wandb

    # Surya backbone + LoRA + linear 2D head
    CUDA_VISIBLE_DEVICES=0 python finetune_ar_segmentation.py
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported (cuBLAS reads it once); see the template script.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functools import partial
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

import app_paths  # noqa: F401  -- puts this directory and workshop_infrastructure on sys.path

from configs import TrainingConfig, load_ar_config
from datasets.ar_dataset import ARSegmentationDataset
from lightning_modules.pl_segmentation import SegmentationLightningModule
from metrics.ar_metrics import ARMetrics
from models.pixel_logistic import (
    PixelLogisticModel,
    destandardize_channels,
)
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from workshop_infrastructure.utils import (
    UploadBestCheckpointToS3,
    apply_peft_lora,
    build_scalers,
    load_pretrained_weights,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"

_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the run config YAML (default: this app's config_script.yaml).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable WandB logging (useful for local runs)."
    )
    parser.add_argument(
        "--train_baseline",
        action="store_true",
        help="Train the per-pixel logistic regression instead of Surya.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Override training.max_epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training.batch_size."
    )
    parser.add_argument(
        "--s3-cache-dir",
        type=str,
        default=None,
        help="Override data.s3_cache_dir (local cache for S3 reads).",
    )
    parser.add_argument(
        "--deterministic",
        choices=tuple(_DETERMINISTIC_CLI),
        default=None,
        help="Override training.deterministic ('warn' when comparing runs).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task-specific: datasets and model
# ---------------------------------------------------------------------------


def build_datasets(cfg: TrainingConfig, scalers) -> Tuple[DataLoader, DataLoader]:
    """Train and validation loaders. Only the AR-mask arguments are this app's business."""
    return build_helio_dataloaders(
        cfg,
        ARSegmentationDataset,
        scalers=scalers,
        seed=cfg.seed,
        return_surya_stack=True,
        max_number_of_samples=cfg.data.max_samples,
        ds_mask_dir=cfg.data.ar_mask_dir,
        ds_mask_splits=cfg.data.ar_mask_splits,
        ds_mask_key=cfg.data.ar_mask_key,
        pooling=cfg.data.ar_pooling,
    )


def build_model(cfg: TrainingConfig, scalers, train_baseline: bool = False) -> L.LightningModule:
    """The baseline and Surya are scored by the same ARMetrics, so they are comparable."""
    metric_kwargs = dict(threshold=cfg.data.ar_threshold, pos_weight=cfg.data.ar_pos_weight)
    metrics = {
        "train_loss": ARMetrics("train_loss", **metric_kwargs),
        "val_loss": ARMetrics("val_loss", **metric_kwargs),  # what ModelCheckpoint monitors
        "train_metrics": ARMetrics("train_metrics", **metric_kwargs),
        "val_metrics": ARMetrics("val_metrics", **metric_kwargs),
    }

    if train_baseline:
        model = PixelLogisticModel(
            in_channels=len(cfg.data.channels),
            time_dim=cfg.model.time_embedding.time_dim,
        )
        # The baseline reads signum-log values, not z-scores: undo the per-channel
        # standardization before every forward pass.
        preprocess_fn = partial(
            destandardize_channels, channel_order=cfg.data.channels, scalers=scalers
        )
        _log_trainable_parameters(model)
        return SegmentationLightningModule(
            model,
            metrics,
            lr=cfg.learning_rate,
            batch_size=cfg.batch_size,
            preprocess_fn=preprocess_fn,
        )

    if cfg.data.ar_pooling != 1:
        raise ValueError(
            f"data.ar_pooling={cfg.data.ar_pooling} downsamples the input, but the Surya "
            f"backbone expects img_size={cfg.model.img_size}. Pooling is for --train_baseline only."
        )

    from workshop_infrastructure.models.finetune_models import HelioSpectformer2D

    model = HelioSpectformer2D.from_config(
        cfg.model,
        dtype=_DTYPES[cfg.dtype] if isinstance(cfg.dtype, str) else cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        ft_unembedding_type="linear",
        ft_out_chans=1,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)

    # Same three regimes as the template: LoRA (+ head), linear probe, or full fine-tune.
    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
    if cfg.model.use_lora:
        model = apply_peft_lora(model, cfg.model.lora_config)
    _log_trainable_parameters(model)

    return SegmentationLightningModule(
        model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size
    )


def _log_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Generic: trainer and entry point (unchanged from the template)
# ---------------------------------------------------------------------------


def build_trainer(
    cfg: TrainingConfig,
    no_wandb: bool = False,
    max_epochs_override: int | None = None,
) -> Tuple[L.Trainer, ModelCheckpoint]:
    max_epochs = max_epochs_override if max_epochs_override is not None else cfg.max_epochs

    loggers = []
    if not no_wandb:
        loggers.append(
            WandbLogger(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=cfg.job_id,
                log_model=False,
                save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
            )
        )
    loggers.append(CSVLogger("runs", name=cfg.job_id))

    Path(cfg.output.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=cfg.output.s3_bucket,
        prefix=cfg.output.s3_prefix,
        fixed_key_name=(cfg.output.s3_best_key or None),
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb],
        log_every_n_steps=2,
    )
    return trainer, checkpoint_cb


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_ar_config(args.config)
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])

    scalers = build_scalers(info=cfg.data.scalers_path)

    train_loader, val_loader = build_datasets(cfg, scalers)
    print(
        f"[DATA] train: {len(train_loader.dataset)} samples | val: {len(val_loader.dataset)} samples"
    )
    lit_model = build_model(cfg, scalers, train_baseline=args.train_baseline)
    trainer, checkpoint_cb = build_trainer(
        cfg, no_wandb=args.no_wandb, max_epochs_override=args.max_epochs
    )

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
