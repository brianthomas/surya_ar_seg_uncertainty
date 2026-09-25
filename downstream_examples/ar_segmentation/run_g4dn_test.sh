#!/usr/bin/env bash
# Single-epoch test fine-tune on one T4 (config_feb15_2013_g4dn.yaml), logging to wandb.
# Needs `wandb login` (or WANDB_API_KEY) beforehand.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# The first backward pass on the T4 OOMs by ~1 GiB with >1 GiB reserved-but-fragmented;
# expandable segments lets the allocator reuse that fragmented memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py \
    --gpu --wandb --config_path ./config_feb15_2013_g4dn.yaml
