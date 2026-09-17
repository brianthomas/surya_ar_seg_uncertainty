# Active Region Segmentation

A fine-tuning example for solar Active Region (AR) segmentation using the Surya foundation model. This project demonstrates how to adapt Surya for downstream computer vision tasks on solar imagery.


## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space for dataset
- Hugging Face account for data access


## Setup and Data Download

```bash
cd downstream_examples/ar_segmentation

# Download AR segmentation dataset (requires Hugging Face login)
bash download_data.sh

# Create CSV indices for training data
python create_ar_csv.py

# unzip masks stored in data.tar.gz in data
cd assets/surya-bench-ar-segmentation
mkdir -p data
tar -xvzf data.tar.gz -C data
```

## Training

```bash
cd downstream_examples/ar_segmentation
# Single GPU training
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py

# Multi-GPU training (example for 4 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 --standalone finetune.py
```

## Inference

Run Active Region segmentation inference using either the interactive notebook or command-line scripts.
**Prerequisites**: Download all the data using the [download_data.sh](download_data.sh) script.

### Option A: Interactive Notebook (Recommended for beginners)

The [ar_segmentation_tutorial.ipynb](ar_segmentation_tutorial.ipynb) notebook provides step-by-step guidance with visualizations.

### Option B: Command-Line Inference

**Basic GPU Inference**
```bash
python infer.py --checkpoint_path ./assets/ar_segmentation_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cuda 
```

**CPU Inference** (slower but no GPU required)
```bash
python infer.py --checkpoint_path ./assets/ar_segmentation_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cpu
```

**Advanced Usage**
```bash
# Custom configuration and more samples
python infer.py --config_path ./config.yaml \
                --checkpoint_path ./assets/ar_segmentation_weights.pth \
                --output_dir ./custom_results \
                --num_samples 10 \
                --data_type valid \
                --device cuda
```

### Parameters Reference
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config_path` | `./config.yaml` | Path to model configuration file |
| `--checkpoint_path` | `./assets/ar_segmentation_weights.pth` | Path to trained model weights |
| `--output_dir` | `./inference_results` | Directory for saving results |
| `--num_samples` | `3` | Number of samples to process and visualize |
| `--data_type` | `test` | Dataset split to use (`test` or `valid`) |
| `--device` | `cuda` | Computing device (`cuda` or `cpu`) |

#### Output
- **Visualizations**: Multi-panel images showing input channels, predictions, and ground truth
- **Format**: High-resolution PNG files
- **Naming**: `test_0.png`, `test_1.png`, etc.
- **Location**: Specified `output_dir`


The output ![Sample output of Surya for 2014-01-07](../../assets/ar_seg_results.png)
## Dataset Information

### Input Data
- **Format**: SDO/AIA multi-channel solar images
- **Shape**: (13, 4096, 4096) - 13 channels including:
  - AIA channels: 94Å, 131Å, 171Å, 193Å, 211Å, 304Å, 335Å, 1600Å
  - HMI channels: Magnetogram, Bx, By, Bz, Velocity
- **Temporal coverage**: 2011-2014
- **Cadence**: 12-minute intervals

### Output Data
- **Format**: Binary segmentation masks
- **Shape**: (4096, 4096)
- **Classes**: Background (0) and Active Region (1)

### Data Source
The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/surya-bench-ar-segmentation](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation)
For more details on mask creation methodology, see [SuryaBench AR Segmentation](https://github.com/NASA-IMPACT/SuryaBench/tree/main/ar_segmentation).

### Finetuning on a single day

`config_feb15_2013.yaml` trains on one day and validates on a different held-out day:

| | Date | Official split |
|---|---|---|
| train | 2013-02-15 | `train.csv` -- earliest 2013 date in the train split |
| validation | 2013-01-15 | `validation.csv` |

```bash
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py \
    --config_path ./config_feb15_2013.yaml
```

The shipped indices have no January 2013 rows in the train split, and no single-day
index exists for either date, so the config uses generated indices under
`assets/single_day/`. Regenerate them (or make indices for another day) with:

```bash
python make_day_index.py --date 2013-02-15
python make_day_index.py --date 2013-01-15
```

The script lists the S3 bucket for the requested day, pulls the matching mask rows out
of whichever shipped AR split contains that date, and reports how many samples survive
the dataset's validity filter -- use that number for `iters_per_epoch_*`.

Both days give **21 samples**. SDO has a daily gap at 21:00, leaving 23 of 24 hourly
masks, and the `+60 min` target requirement then drops 20:00 (its target is the missing
21:00) and 23:00 (its target falls on the next day). Each generated SDO index matches
the corresponding shipped index exactly -- 119 files for 2013-02-15 against the train
index, 116 for 2013-01-15 against the valid index.

The two days come from different official splits, so validation is genuinely held out --
but 21 samples from a single day covers one active-region configuration, so read the
metric as a smoke signal, not as real performance. Checkpoints go to
`checkpoints_feb15_2013/` so they do not overwrite a full run.

### Relocating the AR mask files

The AR label masks (`.h5`) are read from `data.ar_mask_root_path`, which defaults to
`./assets/surya-bench-ar-segmentation` -- where `download_data.sh` puts them. To keep
them off this filesystem, download them elsewhere and point the config at the same
directory:

```bash
AR_MASK_DIR=/scratch/ar_masks ./download_data.sh
```

```yaml
data:
  ar_mask_root_path: /scratch/ar_masks
```

The index CSVs (`ar_index_train` / `ar_index_valid`) are configured separately, so they
can stay in `assets/` or move with the masks. The path is validated when the dataset is
constructed, so a wrong directory fails immediately rather than part-way into training.
If you moved the masks and also regenerate index CSVs, pass the same directory to
`create_ar_csv.main(..., mask_root=...)`.

### Reading SDO input from S3

`data.sdo_data_root_path` accepts an `s3://` URI as well as a local directory. The
default config reads the public bucket directly:

```yaml
data:
  sdo_data_root_path: s3://nasa-surya-bench
  s3_anon: true          # public bucket: without this, ambient AWS credentials sign
                         # the request and it is rejected with 403
  s3_scratch_dir: null   # null -> $SURYA_S3_SCRATCH, $SCRATCH, $TMPDIR, system temp
```

The bucket is laid out as `YYYY/MM/YYYYMMDD_HHMM.nc`, matching the `path` column of
`assets/*_index_surya_1_0.csv`, so no index files need editing. Only the SDO inputs
come from S3; the AR label masks are still read from `assets/` (see Setup above).

Each object is downloaded whole, read, and deleted. Peak scratch usage is therefore
`num_data_workers x world_size x ~0.6 GB` (~5 GB at the default 8 workers), not the
size of the dataset. Files are not cached between reads: at ~600 MB per timestep and
~70k training timesteps, no realistic disk holds enough of the dataset for a shuffled
sampler to hit it.

Reading one 13-channel timestep costs roughly 2 s of download plus 5 s of HDF5 decode,
so throughput is bound by decode rather than by S3 -- **`num_data_workers` is the knob
to raise, not `s3_boto3_max_concurrency`.**

If you repeatedly train on a small fixed subset, stage it once and use a local path
instead -- faster than any download-per-read and easier to reason about:

```bash
aws s3 cp --recursive --no-sign-request s3://nasa-surya-bench/2011/02/ /scratch/sdo/2011/02/
# then set sdo_data_root_path: /scratch/sdo
```

## File Structure

```
ar_segmentation/
├── README.md                    # This file
├── config.yaml                  # Training configuration
├── download_data.sh            # Data download script
├── create_ar_csv.py            # Dataset indexing script
├── finetune.py                 # Training script
├── infer.py                    # Inference script
├── run_inference_example.sh    # Inference example
├── dataset.py                  # Dataset class implementation
├── segmentation_models.py      # Model definitions
└── assets/                     # Data indices and downloaded data
```

## Pre-trained Models

Pre-trained weights are available on Hugging Face:
- **Repository**: [models/nasa-ibm-ai4science/ar_segmentation_surya](https://huggingface.co/nasa-ibm-ai4science/ar_segmentation_surya)
- **Model Type**: SpectFormer with LoRA adapters

### Custom Dataset
To use your own AR masks:

1. Organize masks in the expected directory structure
2. Update the CSV files with your data paths
3. Modify `config.yaml` to point to your indices
