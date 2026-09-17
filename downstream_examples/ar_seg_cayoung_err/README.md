# AR segmentation — uncertainty study

Four notebooks that segment solar active regions, plus the error-bar / calibration /
ensemble analysis in notebook 3.

| notebook | what it does |
|---|---|
| `0_dataset_dataloader_ar.ipynb` | pairs 13-channel SDO stacks with binary AR masks |
| `1_baseline_ar.ipynb` | 14-parameter per-pixel logistic regression, and the scorecard |
| `2_finetune_ar.ipynb` | Surya backbone + LoRA + linear 2D head, same metrics |
| `3_errors_ar.ipynb` | bootstrap CIs, calibration, contours, seed ensemble, label floor |

Longer prose: `NOTEBOOKS_EXPLAINED.md`, `ERRORS_NOTEBOOK_EXPLAINED.md`,
`GROUND_TRUTH_EXPLAINED.md`, `RESULTS_REPORT.md`.

## Provenance

This app was developed as `downstream_apps/ar_segmentation/` in the `surya_workshop`
repository. It has been moved here, and the infrastructure it depends on has been
vendored alongside it, so **nothing outside this repository is needed any more**:

```
surya_ar_seg_uncertainty/          <- repository root
├── workshop_infrastructure/       <- vendored: Surya backbone, HelioNetCDFDataset,
│                                     config dataclasses, dataloader builders, assets
├── data/indices/                  <- vendored: the Surya index CSVs (82 MB)
├── downstream_examples/
│   ├── template/                  <- vendored: the app template this one was forked from
│   └── ar_seg_cayoung_err/        <- this app
└── tests/                         <- vendored: 57 tests, 15 of them for this app
```

The only things still outside the repository are bulk data, which the workshop repository
never held either:

| what | where | set by |
|---|---|---|
| AR masks (122k `.h5`, ~6 GB) | `/home/jovyan/scratch_space/AR_Seg_Uncertainty/data/assets/surya-bench-ar-segmentation` | `data/surya-bench-ar-segmentation` symlink |
| SDO NetCDF inputs | public `nasa-surya-bench` S3 bucket, cached at `/home/jovyan/scratch_space/cayoung/surya_s3_cache` | `data.s3_cache_dir` |

`assets/scalers.yaml` and `assets/surya.366m.v1.pt` are local copies; `ensure_assets()`
re-downloads them from HuggingFace if they ever go missing.

## Running

Notebooks: open with the **Python (surya_ws)** kernel (`/home/jovyan/envs/surya_ws`,
Python 3.12) and run in this directory, in order 0 → 1 → 2 → 3.

Script, from this directory:

```bash
# 14-parameter per-pixel logistic baseline (CPU is fine)
python finetune_ar_segmentation.py --train_baseline --no-wandb

# Surya backbone + LoRA + linear 2D head
CUDA_VISIBLE_DEVICES=0 python finetune_ar_segmentation.py
```

`AR_CONFIG` (notebooks) and `--config` (script) both point at a different YAML without
editing anything.

Tests, from the repository root:

```bash
pytest tests/test_ar_segmentation.py tests/test_lora_setup.py tests/test_model_config.py
```

## Imports

`import app_paths` replaces the old `sys.path.append("../../")`. It derives both roots
from its own location — this directory, and the repository root two levels up — so the
app's modules are imported by plain name and the infrastructure by package name:

```python
from configs import load_ar_config
from datasets.ar_dataset import ARSegmentationDataset
from metrics.ar_metrics import ARMetrics
from models.pixel_logistic import PixelLogisticModel
from lightning_modules.pl_segmentation import SegmentationLightningModule
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
```

## Known gaps

- `runs/errors/surya_seed1_smoke_window_pw11_bce-dice_lr0.0001.ckpt` is a **truncated
  copy** (659 MB against 1775 MB at the source) and will fail to load.
- The `medium` and `real` checkpoints and the `runs/errors_*` CSV logs were never copied
  out of the workshop directory. Without them `3_errors_ar.ipynb` retrains at those
  budgets instead of reusing the cached runs.
