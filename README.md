# surya_ar_uncertainty

A trimmed copy of [Surya](https://github.com/NASA-IMPACT/Surya) containing only what is
needed to replicate the **AR (Active Region) Segmentation** downstream fine-tuning
example, as a base for uncertainty work.

Copied from `~/Code/Surya` (git `78b827e`).

## What's here

```
pyproject.toml, uv.lock         # environment definition (unmodified)
environment.yml	                # Conda environment (alt) 
surya/                          # the full surya package (datasets, models, utils)
tests/                          # test_surya.py — install sanity check
assets/                         # README figures
downstream_examples/
  download_sample_train_data.py # helper for pulling sample SDO training data
  ar_segmentation/
    finetune.py                 # <- the fine-tuning entry point
    infer.py, models.py, dataset.py, create_ar_csv.py
    config.yaml                 # fine-tuning config
    config_infer.yaml           # inference config
    download_data.sh            # HF downloads (weights, bench data, indices)
    ar_segmentation_tutorial.ipynb
    checkpoints/                # fine-tuning output dir (path_experiment)
    assets/
```

Deliberately **not** copied: the other downstream examples (`solar_flare_forcasting`,
`solar_wind_forcasting`, `euv_spectra_prediction`), `easy_inference/`, `.git`, and the
previous run's `wandb/`, `logs/`, and `inference_results/` output.

### Large assets are not included

The three multi-GB artifacts are **not** in this project — no copies, and no links into
`~/Code/Surya`. Fetch them with the example's own download script:

```bash
cd downstream_examples/ar_segmentation
bash download_data.sh
```

It writes each one to the path the configs already reference:

| path under `assets/` | size | needed for |
| --- | --- | --- |
| `surya.366m.v1.pt` | 1.7 GB | pretrained backbone — fine-tuning |
| `ar_segmentation_weights.pth` | 1.7 GB | released fine-tuned weights — inference |
| `infer_data/` | 9.4 GB | 16 `.nc` files — inference |

Everything else (`scalers.yaml`, `train/valid_index_surya_1_0.csv`,
`surya-bench-ar-segmentation/*.csv`, `ar_csv_files/`) is a real copy and is already here.

## Setup

```bash
cd ~/Code/surya_ar_uncertainty
uv sync
source .venv/bin/activate          # the tutorial notebook's kernel is named ".venv"
python -m pytest -s -o log_cli=true tests/test_surya.py   # optional sanity check
```

`surya` is installed as a package by `uv sync`, which is what lets `finetune.py` do
`from surya.utils import distributed` while running from the example directory.

## Before fine-tuning can run

Two things are still needed before `finetune.py` can run:

1. **The downloaded assets, plus the masks.** Run `bash download_data.sh` (above) to get the
   pretrained backbone and the AR benchmark archive, then unpack the segmentation masks,
   which arrive as a tarball:
   ```bash
   cd downstream_examples/ar_segmentation/assets/surya-bench-ar-segmentation
   mkdir -p data && tar -xvzf data.tar.gz -C data
   ```
   `dataset.py` reads each mask from `./assets/surya-bench-ar-segmentation/<file_path>`
   (e.g. `data/2010/05/20100513_0100.h5`); only the index CSVs were copied into this project.

2. **SDO NetCDF corpus.** `config.yaml`'s `sdo_data_root_path` still points at
   `/nobackupnfs1/sroy14/processed_data/Helio/nc`, a path from the original authors'
   cluster. Point it at a local NetCDF root (see
   `downstream_examples/download_sample_train_data.py`) and make sure
   `assets/train_index_surya_1_0.csv` paths line up with it.

`config.yaml` also lists `ar_index_test: ./assets/surya-bench-ar-segmentation/test.csv`,
which the HF dataset snapshot doesn't include — harmless, since `finetune.py` only reads
`ar_index_train` and `ar_index_valid`.

## Single change made to the copied files

`config.yaml`'s `pretrained_path` was `../../data/Surya-1.0/surya.366m.v1.pt`, a path that
does not exist in either repo. It now reads `./assets/surya.366m.v1.pt`, matching
`config_infer.yaml` and the path `download_data.sh` writes the checkpoint to.

## Running

```bash
cd downstream_examples/ar_segmentation

# fine-tune (single GPU)
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py

# inference with the released weights
python infer.py --checkpoint_path ./assets/ar_segmentation_weights.pth \
                --output_dir ./inference_results --num_samples 3 --device cuda
```
