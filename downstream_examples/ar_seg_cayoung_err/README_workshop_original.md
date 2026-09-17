# Active-region segmentation

A downstream app forked from `downstream_examples/template/` (see `template/ADAPTING.md`).
Input: a 13-channel SDO stack. Target: a 4096x4096 binary mask of active regions that
contain a polarity inversion line, from the
[surya-bench-ar-segmentation](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation)
dataset (hourly, 2010-2024, 0/255 uint8 in HDF5).

The point of this app is the **baseline**: a 14-parameter per-pixel logistic regression
scored by exactly the same metrics, dataset and config as the Surya + LoRA model, so the
difference between the two numbers is the value the foundation model adds.

## Layout

| File | Role |
|------|------|
| `0_dataset_dataloader_ar.ipynb` | walk-through: pairing SDO stacks with masks, inspecting and plotting samples |
| `1_baseline_ar.ipynb` | walk-through: the 14-parameter baseline, the metrics, a Lightning training loop |
| `2_finetune_ar.ipynb` | walk-through: Surya + LoRA with the same data, metrics and loop (GPU) |
| `3_errors_ar.ipynb` | walk-through: bootstrap intervals, calibration, confidence contours, seed ensembles, the label floor (explained in `ERRORS_NOTEBOOK_EXPLAINED.md`) |
| `configs.py` | `ARDataConfig`: the task-specific `data:` keys (`ar_mask_dir`, `ar_mask_key`, `ar_pooling`, `ar_pos_weight`, `ar_threshold`) |
| `configs/config_script.yaml` | the single run config |
| `datasets/ar_dataset.py` | `ARSegmentationDataset`: exact-timestamp pairing of Surya index and mask index |
| `models/pixel_logistic.py` | `PixelLogisticModel`: 1x1 conv = logistic regression per pixel |
| `metrics/ar_metrics.py` | `ARMetrics`: BCE loss; IoU / Dice / precision / recall at validation |
| `lightning_modules/pl_segmentation.py` | reuses the template's task-agnostic Lightning module |
| `finetune_ar_segmentation.py` | training script; `--train_baseline` switches model |
| `download_masks.sh` | fetches and extracts the mask dataset into `data/` |
| `GROUND_TRUTH_EXPLAINED.md` | how the mask dataset is made, what it means, how faithfully it can be reproduced |
| `RESULTS_REPORT.md` | all measured results so far: the collapse and fixes, medium and real budgets, the limb effect, the label floor, machine findings |

Tests: `pytest tests/test_ar_segmentation.py` (CPU, seconds).

The notebooks follow the template's 0 / 1 / 2 sequence and read `configs/config_script.yaml`.
Set `AR_CONFIG=/path/to/other.yaml` before starting the kernel to run them against a different
config (for example one whose indices point at local NetCDF files) without editing cells.

## Setup

```bash
cd downstream_examples/ar_seg_cayoung_err
bash download_masks.sh          # ~1.3 GB download, ~6 GB extracted, no login needed
```

SDO inputs come from the public `s3://nasa-surya-bench` bucket via the shared indices in
`data/indices/`. Set a cache directory (about 1 GB per unique timestep):

```bash
export SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache
```

## Run

```bash
# from the repo root
python -m downstream_examples.ar_seg_cayoung_err.finetune_ar_segmentation --train_baseline --no-wandb
CUDA_VISIBLE_DEVICES=0 python -m downstream_examples.ar_seg_cayoung_err.finetune_ar_segmentation
```

`max_samples: 10` in the YAML keeps the first runs short; raise it for real numbers.

## The baseline ladder

All of these are the same script and metrics; only the config changes.

| Baseline | Config change | Parameters |
|----------|---------------|-----------:|
| learned \|B_los\| threshold | `channels: [hmi_m]`, `model.in_channels: 1` | 2 |
| per-pixel logistic on 13 channels | default | 14 |
| Surya + LoRA + linear 2D head | drop `--train_baseline` | ~3.2 M trainable |

The label recipe is itself a threshold on the magnetogram (+/-50 G, 100 px minimum,
10 px dilation, PIL required), so the first row measures how much of the task the
recipe already explains.

## Things to know

- **Pairing.** Masks are hourly; SDO is every 12 minutes. Only on-the-hour SDO
  timesteps pair, by exact match. The Surya index decides train vs validation, so all
  mask CSVs are listed in `ar_mask_splits` by default.
- **Metrics are pooled over the batch**, not averaged per sample. With `batch_size: 1`
  that is per-sample. Empty target + empty prediction scores IoU 1.
- **Class imbalance.** AR pixels are ~1.3 % of the disk. `ar_pos_weight` up-weights
  them in the BCE; the default (`null`) matches the released Surya example.
- **`ar_pooling`** block-averages the input stack and the mask (`>= 0.5` re-binarizes).
  It exists so the baseline can run on a laptop; the Surya path refuses it.
- **No vertical flips.** The base dataset would flip the inputs but not the mask.
- **`ar_mask_key: intersection`** switches the target to the PIL map itself, a much
  thinner target with harsher imbalance.
