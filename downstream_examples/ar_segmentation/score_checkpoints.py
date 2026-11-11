#!/usr/bin/env python
"""Score saved AR-segmentation checkpoints on any AR index, and report calibration.

Fills two gaps finetune.py leaves open.

  * `ar_index_test` is carried in every bootstrap config but nothing reads it.
    `finetune.py:get_dataloaders` builds a train and a valid loader only, and
    `infer.py:get_dataloader` ignores its `data_type` argument and builds a mask-less
    `HelioNetCDFDataset`, so it cannot score against ground truth at all. The configured
    test split has therefore never been evaluated for any member of any regime.
  * Validation BCE is logged per epoch, but no calibration metric is. "The probability fit
    stopped improving" can then only be argued from BCE going flat, which is weak when BCE
    is also the training objective. ECE and Brier state it directly, and NOTES rule 2 --
    valid/iou peaks late while valid/loss floors early -- is a statement about calibration
    that deserves a calibration number.

Confusion counting is deliberately identical to `finetune.py:evaluate_model`: a 0.5 threshold
on sigmoid(logits), target > 0.5, micro-averaged TP/FP/FN/TN accumulated in float64 over every
pixel of every sample, and average loss as the unweighted mean of per-batch BCE means. The
validation sampler is reconstructed the same way too -- `StatefulDistributedSampler` defaults
to shuffle=True/seed=0 and `set_epoch` is never called, so its permutation is deterministic and
the run scored a fixed 200-frame prefix of it (NOTES rule 5).

That makes the script self-testing: scoring the validation index with a checkpoint whose IoU
the run already logged must reproduce that number. `--expect-iou` asserts it, and is how any
change here should be checked.

Single process on purpose -- `evaluate_model`'s `all_reduce` calls need a process group, and at
world size 1 there is nothing to reduce.

Note on dtype: the bootstrap configs say bfloat16 because they trained on an L40S. On a T4
(sm_75) bfloat16 is emulated -- slower and less accurate -- so `--dtype` defaults to the
config's value but should be set to float16 there. Compare numbers only across runs that used
the same dtype; see --expect-iou's tolerance.

Usage:
    # the never-scored test split. Build its SDO index first -- nothing shipped covers
    # leaky_validation's Jan 1-14 / Feb 1-14 inputs, so it yields 0 samples until then:
    #   python make_month_start_subsets.py --split leaky_validation --period 1day \
    #       --with-sdo-index --out-dir ./assets/subsamples_test_eval
    python score_checkpoints.py \
        --config_path ./configs_bootstrap/2016hr/config_boot01.yaml \
        --ar-index ./assets/subsamples_test_eval/ar_index_leaky_validation_1day.csv \
        --sdo-index ./assets/subsamples_test_eval/sdo_index_leaky_validation_1day.csv \
        --checkpoints /efs/bathomas/ar_seg_bootstrap/2016hr/boot01/best.pth \
        --dtype float16 --label test

    # per-epoch calibration on the validation index the run selected on
    python score_checkpoints.py \
        --config_path ./configs_bootstrap/2016hr/config_boot01.yaml \
        --checkpoints /efs/bathomas/ar_seg_bootstrap/2016hr/boot01/epoch_{0,1,2,3,4}.pth \
        --dtype float16 --max-samples 200 --label valid
"""

import argparse
import csv
import os
import time

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader

# Preloaded for the same reason finetune.py does it: the lazy dask import inside xarray's
# .load() path runs in forked DataLoader workers.
import dask.base  # noqa: F401

from dataset import ArDSDataset
from finetune import custom_collate_fn
from infer import load_model
from surya.datasets.disk_alignment import corner_scaling_kwargs
from surya.utils.data import build_scalers
from surya.utils.distributed import StatefulDistributedSampler

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_dataset(config, scalers, ar_index, sdo_index):
    """An ArDSDataset over `ar_index`, constructed exactly as get_dataloaders' valid branch.

    `sdo_index` plays the role `valid_data_path` plays in training: the SDO input index the AR
    labels are joined against. An AR index whose timestamps are absent from it yields zero
    samples rather than an error, which is why the caller checks len() below.
    """
    channels = (
        config["adapter"]["channels"]
        if config["adapter"]["use_channel_adapter"]
        else config["data"]["channels"]
    )
    s3_kwargs = dict(
        s3_anon=config["data"].get("s3_anon", False),
        s3_scratch_dir=config["data"].get("s3_scratch_dir"),
        s3_boto3_max_concurrency=config["data"].get("s3_boto3_max_concurrency", 4),
        s3_boto3_part_size_mb=config["data"].get("s3_boto3_part_size_mb", 64),
    )
    # Corner-scaling correction, exactly as finetune.py:get_dataloaders applies it: a checkpoint
    # must be scored with the setting it trained under (KNOWN_ISSUE_channel_offset.md). A no-op
    # for every config that leaves data.fix_corner_scaling off, i.e. every run before *_pxfix.
    s3_kwargs.update(corner_scaling_kwargs(config["data"]))
    return ArDSDataset(
        sdo_data_root_path=config["data"]["sdo_data_root_path"],
        index_path=sdo_index,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        scalers=scalers,
        num_mask_aia_channels=config["num_mask_aia_channels"],
        drop_hmi_probablity=config["drop_hmi_probablity"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        channels=channels,
        phase="valid",
        ds_ar_index_paths=[ar_index],
        ar_mask_root_path=config["data"].get("ar_mask_root_path"),
        **s3_kwargs,
    )


@torch.no_grad()
def score(model, loader, device, dtype, criterion, max_samples, n_bins, threshold=0.5):
    """Segmentation metrics plus ECE/Brier over `max_samples` batches of `loader`.

    Returns a dict. The TP/FP/FN/TN block mirrors finetune.py:evaluate_model exactly so the
    IoU/Dice/precision/recall it reports are the same quantities the training run logged.
    Calibration is accumulated as per-bin (count, sum of confidence, sum of label) in float64,
    which is the standard binned ECE estimator and needs one pass.
    """
    tp = torch.zeros((), device=device, dtype=torch.float64)
    fp = torch.zeros((), device=device, dtype=torch.float64)
    fn = torch.zeros((), device=device, dtype=torch.float64)
    tn = torch.zeros((), device=device, dtype=torch.float64)

    bin_n = torch.zeros(n_bins, device=device, dtype=torch.float64)
    bin_conf = torch.zeros(n_bins, device=device, dtype=torch.float64)
    bin_pos = torch.zeros(n_bins, device=device, dtype=torch.float64)
    brier_sum = torch.zeros((), device=device, dtype=torch.float64)

    # Interior bin edges: bucketize maps p into 0..n_bins-1 over [0,1].
    edges = torch.linspace(0, 1, n_bins + 1, device=device)[1:-1]

    running_loss, num_batches = 0.0, 0
    t0 = time.time()
    for i, (batch, _meta) in enumerate(loader):
        if i == max_samples:
            break
        curr = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type=device.type, dtype=dtype):
            outputs = model(curr)
            target = curr["forecast"].unsqueeze(1)
            loss = criterion(outputs, target)

        running_loss += loss.item()
        num_batches += 1

        prob = torch.sigmoid(outputs.float())
        true = target > 0.5

        pred = prob > threshold
        tp += (pred & true).sum()
        fp += (pred & ~true).sum()
        fn += (~pred & true).sum()
        tn += (~pred & ~true).sum()

        p = prob.reshape(-1).double()
        y = true.reshape(-1).double()
        idx = torch.bucketize(p, edges)
        bin_n.index_add_(0, idx, torch.ones_like(p))
        bin_conf.index_add_(0, idx, p)
        bin_pos.index_add_(0, idx, y)
        brier_sum += ((p - y) ** 2).sum()

        if num_batches % 25 == 0:
            print(f"    {num_batches} samples, {time.time() - t0:.0f}s", flush=True)

    tp, fp, fn, tn = tp.item(), fp.item(), fn.item(), tn.item()
    total = tp + fp + fn + tn
    if total == 0:
        raise SystemExit("Scored 0 pixels -- the AR index yielded no samples.")

    def ratio(num, den):
        return float("nan") if den == 0 else num / den

    n = bin_n.cpu()
    conf = bin_conf.cpu()
    pos = bin_pos.cpu()
    nz = n > 0
    # ECE: sum over occupied bins of n_b/N * |mean label - mean confidence|.
    ece = ((n[nz] / n.sum()) * (pos[nz] / n[nz] - conf[nz] / n[nz]).abs()).sum().item()

    return {
        "n_samples": num_batches,
        "loss": running_loss / max(num_batches, 1),
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, total),
        "pos_frac_true": ratio(tp + fn, total),
        "pos_frac_pred": ratio(tp + fp, total),
        "ece": ece,
        "brier": (brier_sum / total).item(),
        "pixels": int(total),
        "seconds": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", required=True,
                   help="The member's own training config, so the LoRA geometry matches the "
                        "checkpoint. config_infer.yaml describes a different LoRA (r=32) and "
                        "will not load these r=8 checkpoints.")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoints to score, in the order to report them.")
    p.add_argument("--ar-index", default=None,
                   help="AR label index. Defaults to the config's ar_index_valid[0].")
    p.add_argument("--sdo-index", default=None,
                   help="SDO input index the labels join against. Defaults to the config's "
                        "valid_data_path.")
    p.add_argument("--dtype", default=None, choices=sorted(DTYPES),
                   help="Autocast dtype. Defaults to the config's. Use float16 on a T4.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Batches to score. Defaults to the config's iters_per_epoch_valid "
                        "(200), which is the prefix the training run scored.")
    p.add_argument("--n-bins", type=int, default=15, help="ECE bins.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default="eval", help="Tag for the output rows.")
    p.add_argument("--out-csv", default=None, help="Append results here.")
    p.add_argument("--expect-iou", type=float, default=None,
                   help="Assert the FIRST checkpoint's IoU matches this, within --expect-tol. "
                        "Use the run's own logged value to verify this script reproduces it.")
    p.add_argument("--expect-tol", type=float, default=2e-3,
                   help="Tolerance for --expect-iou. Default 2e-3 absorbs autocast dtype "
                        "differences between this GPU and the one that trained.")
    args = p.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    dtype = DTYPES[args.dtype] if args.dtype else DTYPES[config["dtype"]]
    config["dtype"] = dtype  # get_model reads config["dtype"] and wants the torch dtype
    ar_index = args.ar_index or config["data"]["ar_index_valid"][0]
    sdo_index = args.sdo_index or config["data"]["valid_data_path"]
    max_samples = args.max_samples or config["iters_per_epoch_valid"]
    device = torch.device(args.device)

    print(f"AR index  : {ar_index}")
    print(f"SDO index : {sdo_index}")
    print(f"dtype     : {dtype}   max_samples: {max_samples}   threshold: {args.threshold}")

    scalers = build_scalers(info=yaml.safe_load(open(config["data"]["scalers_path"])))
    dataset = build_dataset(config, scalers, ar_index, sdo_index)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise SystemExit(
            f"{ar_index} yielded 0 samples against {sdo_index}. Its timestamps are absent from "
            "that SDO index -- build one with make_month_start_subsets.py --with-sdo-index."
        )
    if len(dataset) < max_samples:
        print(f"note: only {len(dataset)} samples available, scoring all of them")

    # Same sampler as training's valid loader: shuffle=True/seed=0 by default and set_epoch is
    # never called, so this is the identical fixed permutation the run scored a prefix of.
    loader = DataLoader(
        dataset=dataset,
        sampler=StatefulDistributedSampler(dataset, num_replicas=1, rank=0, drop_last=True),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_data_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    rows = []
    for ckpt in args.checkpoints:
        print(f"\n=== {ckpt}")
        model = load_model(config, ckpt, device)
        model.eval()
        m = score(model, loader, device, dtype, criterion, max_samples,
                  args.n_bins, args.threshold)
        m["checkpoint"] = os.path.basename(ckpt)
        m["label"] = args.label
        m["ar_index"] = ar_index
        rows.append(m)
        print(
            f"  n={m['n_samples']}  IoU {m['iou']:.6f}  Dice {m['dice']:.6f}  "
            f"P {m['precision']:.4f}  R {m['recall']:.4f}\n"
            f"  loss {m['loss']:.6f}  ECE {m['ece']:.6e}  Brier {m['brier']:.6e}  "
            f"pred/true area {m['pos_frac_pred'] / m['pos_frac_true']:.4f}  "
            f"({m['seconds']:.0f}s)"
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n{'checkpoint':14s} {'n':>4s} {'IoU':>9s} {'loss':>10s} {'ECE':>11s} "
          f"{'Brier':>11s} {'pred/true':>9s}")
    for m in rows:
        print(f"{m['checkpoint']:14s} {m['n_samples']:4d} {m['iou']:9.6f} {m['loss']:10.6f} "
              f"{m['ece']:11.4e} {m['brier']:11.4e} "
              f"{m['pos_frac_pred'] / m['pos_frac_true']:9.4f}")

    if args.out_csv:
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out_csv}")

    if args.expect_iou is not None:
        got = rows[0]["iou"]
        delta = abs(got - args.expect_iou)
        status = "OK" if delta <= args.expect_tol else "MISMATCH"
        print(f"\n--expect-iou {status}: got {got:.6f}, expected {args.expect_iou:.6f}, "
              f"|delta| {delta:.2e} (tol {args.expect_tol:.0e})")
        if delta > args.expect_tol:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
