# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A trimmed fork of [Surya](https://github.com/NASA-IMPACT/Surya) (a 366M-parameter heliophysics
foundation model trained on SDO/AIA+HMI data), kept down to only what's needed to run the
**Active Region (AR) Segmentation** downstream fine-tuning example, as a base for uncertainty
quantification work. See `README.md` for what was deliberately excluded from the original repo
(other downstream tasks, `easy_inference/`, prior run outputs) and the one deliberate config
change (`pretrained_path` in `config.yaml`). `README_Surya.md` is the original upstream README,
kept for architecture/model background — it describes downstream tasks not present in this repo.

## Setup

Two environment paths exist (`uv` is the original path, `conda` was added on top — both install
the same deps):

```bash
uv sync
source .venv/bin/activate
```
or
```bash
conda env create -f environment.yml
conda activate surya_ar_seg_uncertainty
pip install --no-deps -e .   # puts `surya` on the path without re-resolving deps from PyPI
```

`surya` must be installed as a package (editable or via `uv sync`) — `finetune.py` and `infer.py`
rely on `from surya.utils import ...` / `from surya.models import ...` while running from inside
`downstream_examples/ar_segmentation/`.

## Commands

```bash
# sanity check that the environment + pretrained weights work (downloads ~2GB from HF on first run)
python -m pytest -s -o log_cli=true tests/test_surya.py

# fetch the large assets this repo doesn't vendor (pretrained backbone, released
# fine-tuned weights, inference .nc files) — run from downstream_examples/ar_segmentation/
bash download_data.sh

# unpack segmentation masks (only the index CSVs are checked in, not the .h5 masks themselves)
cd downstream_examples/ar_segmentation/assets/surya-bench-ar-segmentation
mkdir -p data && tar -xvzf data.tar.gz -C data

# fine-tune (must run from downstream_examples/ar_segmentation/, torchrun required even for 1 GPU)
cd downstream_examples/ar_segmentation
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
torchrun --nnodes=1 --nproc_per_node=4 --standalone finetune.py   # multi-GPU

# inference with released fine-tuned weights
python infer.py --checkpoint_path ./assets/ar_segmentation_weights.pth \
                --output_dir ./inference_results --num_samples 3 --device cuda
```

There is no linter/formatter config actively enforced beyond the `black`/`isort`/`mypy` settings
declared in `pyproject.toml`'s `[tool.*]` sections (no pre-commit hook wired up in this trimmed
repo). `tests/test_surya.py` is the only test file; it's an end-to-end GPU-friendly (but CPU-
capable) integration test, not a unit test suite — there's no `test_ar_segmentation.py` yet.

## Before fine-tuning can run

Two things beyond `download_data.sh` are needed (see `README.md` for full detail):

1. The masks tarball must be unpacked (see above) — `dataset.py`'s `ArDSDataset` reads each mask
   from `./assets/surya-bench-ar-segmentation/<file_path>` relative to the CWD the script runs
   from.
2. `config.yaml`'s `data.sdo_data_root_path` still points at the original authors' cluster path
   (`/nobackupnfs1/sroy14/processed_data/Helio/nc`) — point it at a local NetCDF root (see
   `downstream_examples/download_sample_train_data.py`) and make sure
   `assets/train_index_surya_1_0.csv` paths line up with it.

## Architecture

**`surya/` — the reusable foundation-model package**, installed as a pip package:
- `surya/models/helio_spectformer.py` — `HelioSpectFormer`, the top-level model. Wraps an
  embedding stage (linear or perceiver-style, chosen by `time_embedding["type"]`), the
  `SpectFormer` backbone, and an optional unembed/decoder stage that's *skipped* when
  `finetune=True` (finetune configs read the backbone's raw token output and add their own head).
  Supports an optional `learned_flow` branch (`HelioFlowModel`, `flow.py`) that can be combined
  additively with the backbone forecast, and an `ensemble` mode that folds ensemble members into
  the batch dimension for the backbone pass and unfolds them at the end.
- `surya/models/spectformer.py` — `SpectFormer` backbone: a stack of `BlockSpectralGating`
  (frequency-domain gating) layers followed by `BlockAttention` (long-short/windowed attention)
  layers over patch tokens.
- `surya/models/embedding.py` — tokenization/detokenization: `PatchEmbed3D`, `LinearEmbedding`,
  `PerceiverChannelEmbedding` (encoders) and `LinearDecoder`, `PerceiverDecoder` (decoders). The
  embedding "type" (`linear` vs `perceiver`) is a config choice threaded through `time_embedding`.
- `surya/datasets/helio.py` — `HelioNetCDFDataset`: reads the SDO NetCDF (`.nc`) cube, builds
  `valid_indices` from an index CSV + timestamp deltas, applies channel scaling/normalization
  (`transformations.py`), and returns `ts`/`time_delta_input`/`forecast`/`*_latitude` tensors.
  Downstream datasets subclass this and override `__getitem__` to attach task-specific targets.
- `surya/utils/config.py` — `ExperimentConfig`/`DataConfig`/`ModelConfig`/`OptimizerConfig`:
  structured wrappers around the training YAML, with cross-field assertions (e.g. model
  `in_channels` must match `len(data.channels)`; linear embedding's `time_dim` must match
  `n_input_timestamps`). Only used by the upstream training path — the AR segmentation
  `finetune.py` reads the YAML into a plain dict instead of using this class.
- `surya/utils/distributed.py` — DDP/FSDP setup (`init_ddp`), `StatefulDistributedSampler`,
  rank-aware helpers (`print0`, `is_main_process`), checkpoint save/load.
- `surya/utils/data.py` — `build_scalers` (constructs per-channel normalization from
  `scalers.yaml`), `custom_collate_fn`.

**`downstream_examples/ar_segmentation/` — the fine-tuning example this fork exists for**:
- `dataset.py` — `ArDSDataset(HelioNetCDFDataset)`: intersects the base HelioFM valid timestamp
  index with the AR benchmark's own index CSVs (`train.csv`/`validation.csv`/`test.csv` under
  `assets/surya-bench-ar-segmentation/`), then loads a binary mask per sample from an `.h5` file
  (dataset key `union_with_intersect`) instead of a forecast target.
- `models.py` — task-specific heads built on top of `HelioSpectFormer`:
  - `HelioSpectformer2D` — adds a segmentation decoder (`LinearDecoder`/`PerceiverDecoder`, chosen
    by `ft_unembedding_type`) on top of the backbone's token output. This is the model
    `finetune.py` builds when `model_type: spectformer_lora`.
  - `HelioSpectformer1D` — pools tokens (avg/max/attention/transformer pooling, mutually
    exclusive, config-selected) down to a scalar-style output; used for non-spatial downstream
    tasks (not the AR segmentation path, but shares this file).
  - `UNet`/`UNetEncoder`/`UNetDecoder`/`DoubleConv` — a standalone baseline segmentation model
    (`model_type: unet`), independent of the Surya backbone.
  - `ChannelAdapter` — an optional `Conv3d` that remaps a smaller input channel set onto the
    13 channels the pretrained backbone expects (gated by `config["adapter"]["use_channel_adapter"]`).
- `finetune.py` — the training entry point. Loads the pretrained backbone weights (filtered to
  matching-shape keys only, so architecture drift doesn't hard-fail), optionally wraps the model
  in PEFT LoRA (`apply_peft_lora`, target modules configurable via `lora_config`), wraps DDP,
  and runs a standard train/eval loop with `BCEWithLogitsLoss` (Dice/IoU losses are defined but
  not currently wired into the training loop's `criterion`). Always uses `torchrun` (raises if
  run without `--gpu`); reads `config.yaml` as a plain dict, not through `surya.utils.config`.
- `infer.py` — loads a checkpoint (either LoRA fine-tuned or full) and runs segmentation
  inference, producing multi-panel PNG visualizations (input channels / prediction / ground truth).
- `create_ar_csv.py` — builds/regenerates the AR index CSVs consumed by `dataset.py`.
- `config.yaml` / `config_infer.yaml` — training/inference configs; see "Before fine-tuning can
  run" above for the two fields that need local overrides.

**Key config-driven branches to be aware of when changing code**: `model_type`
(`spectformer_lora` vs `unet`) picks the whole model class in both `finetune.py:get_model` and
`infer.py`; `time_embedding["type"]` (`linear` vs `perceiver`) picks embedding/decoder classes
inside `HelioSpectFormer`; `finetune: True` on `HelioSpectFormer` disables the base model's own
unembed layer so the downstream head (`HelioSpectformer2D`/`1D`) supplies its own.
