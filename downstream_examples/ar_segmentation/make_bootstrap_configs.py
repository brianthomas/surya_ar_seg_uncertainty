#!/usr/bin/env python
"""Generate one finetune.py config per bootstrap replicate, plus a driver script.

Each replicate written by bootstrap_subsample.py becomes one fine-tuning run, and the spread
across the resulting models is the ensemble's uncertainty estimate. finetune.py already accepts
--config_path (finetune.py:670), so driving 10 runs needs no training-code change -- only 10
configs that differ in the right places and agree everywhere else.

Exactly three things may differ between members, or the ensemble stops measuring what it should:

    job_id              the wandb run name (finetune.py:559)
    ar_index_train      the replicate
    path_experiment     where checkpoints land

Everything else -- and in particular the evaluation indices -- is held identical. Members have to
be scored on the same data; if validation were resampled per member, the observed spread would be
model variance plus evaluation-set variance with no way to separate them.

A FOURTH, OPT-IN DIFFERENCE: --seed-start. Without it no `seed` key is written at all, so every
member trains at finetune.py's default seed 0 and member spread is purely finite-sample -- a
clean nonparametric bootstrap, and what all four existing ensembles ran under. With
--seed-start K, member i gets seed K+i-1, so the replicate AND the optimization path both vary
and member spread becomes the TOTAL of the two components. Pass --seed-start 1 to pair each arm
1:1 with its seed-0 counterpart in the parent ensemble: the per-replicate difference is then a
paired estimate of the optimization-path component, on 9 df rather than the 2 df the one-replicate
seed ablation in make_seed_ablation_configs.py can reach. See NOTES, "Seed ablation".

Three fixes are applied to every generated config because the shipped config.yaml cannot run the
ensemble as-is:

  1. ar_mask_root_path points at /home/jovyan/scratch_space/... which does not exist, so
     dataset.py:63 raises FileNotFoundError before training starts. Repointed at the assets copy.
  2. path_experiment is one global "checkpoints" dir, and finetune.py:657-659 writes
     epoch_{epoch}.pth there unconditionally every epoch -- all 10 runs would overwrite each
     other, and 15 epochs x 10 members at ~1.8 GB is ~250 GB against 26 GB free on /home/jovyan.
     Moved to a per-member directory under --experiment-root (default on /mnt/efs).
  3. iters_per_epoch_train is 2000 against a 168-sample epoch. Harmless (finetune.py:610 uses it
     only as a break cap) but it makes the logs misleading, so it is set to the replicate size.

Usage:
    python make_bootstrap_configs.py
    python make_bootstrap_configs.py --max-epochs 5
    python make_bootstrap_configs.py --replicate-dir assets/bootstrap_train_1week

The 2016hr ensemble, with the optimizer settings that keep a 2016-step epoch from diverging
(see --learning-rate and the three flags under it):

    python make_bootstrap_configs.py \\
        --replicate-dir ./assets/bootstrap_train_2016hr \\
        --out-dir ./configs_bootstrap/2016hr \\
        --runner ./run_bootstrap_ensemble_2016hr.sh \\
        --experiment-root /mnt/efs/bathomas/ar_seg_bootstrap/2016hr \\
        --job-id-prefix AR-seg_2016hr \\
        --valid-index ./assets/subsamples/ar_index_validation_1day.csv \\
        --test-index ./assets/subsamples/ar_index_leaky_validation_1day.csv \\
        --max-epochs 5 \\
        --learning-rate 0.0002 --lr-schedule cosine --warm-up-steps 200 --grad-clip-norm 1.0

The 672hr_stable ensemble -- the 672hr replicates re-run with clipping and warmup, keeping
lr 1e-3 because 672hr reaches the highest IoU of any regime despite its instability. Written
to its own out-dir/experiment-root/job-id-prefix so the original 672hr set stays intact:

    python make_bootstrap_configs.py \\
        --replicate-dir ./assets/bootstrap_train_1month \\
        --out-dir ./configs_bootstrap/672hr_stable \\
        --runner ./run_bootstrap_ensemble_672hr_stable.sh \\
        --experiment-root /mnt/efs/bathomas/ar_seg_bootstrap/672hr_stable \\
        --job-id-prefix AR-seg_672hr_stable \\
        --valid-index ./assets/subsamples/ar_index_validation_1day.csv \\
        --test-index ./assets/subsamples/ar_index_leaky_validation_1day.csv \\
        --max-epochs 6 \\
        --learning-rate 0.001 --lr-schedule cosine --warm-up-steps 70 --grad-clip-norm 1.0

6 epochs, not 10: boot01 ran at 10 and its IoU plateaued over the last four epochs (0.5926 to
0.6016, flat within noise) while valid/loss rose monotonically from epoch 4 -- so those epochs
cost ~1.6 TB each and bought overfitting. Note this is not the same as stopping a 10-epoch run
at 6: the cosine total is max_epochs x iters_per_epoch_train, so a 6-epoch config anneals to
min_lr by epoch 6 rather than sitting near 3.5e-4 there. boot01's own 10-epoch configs are
preserved under configs_bootstrap/672hr_stable.10ep.20261027/ since its results came from them.
"""

import argparse
import copy
import glob
import os
import stat

import pandas as pd
import yaml

HEADER = """\
# GENERATED by make_bootstrap_configs.py -- do not edit by hand.
#
# Bootstrap ensemble member {rep} of {total}.
#   training index : {train}  ({n_rows} samples)
#   checkpoints    : {experiment}
{seed_line}#
# Evaluation indices are identical across all {total} members on purpose: members must be
# scored on the same data, or model variance and evaluation-set variance are confounded.
# Regenerate with: python make_bootstrap_configs.py
"""

SEED_LINE = """\
#   model seed     : {seed}  (torch/numpy/random {offset}, train sampler {seed})
#
# This member varies BOTH its replicate and its seed, so across-member spread here is the total
# of the finite-sample and optimization-path components rather than the first alone. Its seed-0
# counterpart is member {rep:02d} of the parent ensemble; the difference between the two is a
# paired sample of the optimization-path component.
"""

RUNNER = """\
#!/usr/bin/env bash
# GENERATED by make_bootstrap_configs.py -- runs the bootstrap ensemble sequentially.
#
# Each member is an independent fine-tune on its own 1-week bootstrap replicate. Checkpoints go
# to per-member directories, so members never overwrite one another and the script deletes
# nothing.
{seed_note}#
# Cost per member: {n_rows} samples x ~590 MB of SDO NetCDF per epoch = ~{gb_epoch:.0f} GB/epoch,
# re-fetched each epoch because files are deleted after use. At {epochs} epochs that is
# ~{tb_member:.1f} TB per member and ~{tb_total:.0f} TB for the ensemble; max_epochs is the main
# lever on wall time.
set -euo pipefail

cd "$(dirname "${{BASH_SOURCE[0]}}")"

# The allocator fragments on long runs at 4096x4096; expandable segments lets it reuse that
# memory, as run_g4dn_test.sh does.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{runs}
echo "All {total} members complete."
"""

RUN_LINE = """\
echo "=== member {rep}/{total}: {config} ==="
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py \\
    --gpu{wandb} --config_path ./{config}
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-config", default="./config.yaml",
                   help="Config to derive every member from.")
    p.add_argument("--replicate-dir", default="./assets/bootstrap_train_1week",
                   help="Directory of replicate indices from bootstrap_subsample.py.")
    p.add_argument("--replicate-glob", default="*_boot[0-9][0-9][0-9].csv",
                   help="Which files in --replicate-dir are replicates (excludes the manifest).")
    p.add_argument("--out-dir", default="./configs_bootstrap",
                   help="Where the generated configs are written.")
    p.add_argument("--runner", default="./run_bootstrap_ensemble.sh",
                   help="Where the driver script is written. It must sit in this example's "
                        "directory, not in --out-dir: finetune.py and every asset path in the "
                        "configs resolve against the working directory, and the runner cds to "
                        "its own location.")
    p.add_argument("--experiment-root", default="/mnt/efs/bathomas/ar_seg_bootstrap",
                   help="Parent of the per-member checkpoint directories. Keep this off "
                        "/home/jovyan, which has ~26 GB free against ~250 GB of checkpoints.")
    p.add_argument("--valid-index", default="./assets/surya-bench-ar-segmentation/validation.csv",
                   help="Fixed validation index, identical for every member. The full split is "
                        "the default deliberately: iters_per_epoch_valid caps evaluation at 200 "
                        "batches either way, so a smaller index costs the same while covering "
                        "far less -- ar_index_validation_1day.csv spans only Jan 15 of each year "
                        "(9 days) against 153 days here. Per-member metric noise feeds straight "
                        "into the ensemble spread, so broader coverage is worth having free.")
    p.add_argument("--test-index", default="./assets/surya-bench-ar-segmentation/leaky_validation.csv",
                   help="Fixed leaky-validation index, identical for every member. The full split "
                        "is the default for the same reason as --valid-index: the 1day subsample "
                        "covers only Jan 1 and Feb 1 of each year (18 days) against 252 days "
                        "here. NOTE: finetune.py never reads ar_index_test -- it is carried for "
                        "post-hoc evaluation -- and nothing shipped indexes its Jan 1-14 / "
                        "Feb 1-14 SDO inputs, so it yields 0 samples until that index is built "
                        "from S3 (make_month_start_subsets.py --with-sdo-index).")
    p.add_argument("--ar-mask-root", default="./assets/surya-bench-ar-segmentation",
                   help="Directory holding the mask .h5 tree.")
    p.add_argument("--s3-scratch-dir", default="/tmp/surya_s3_scratch",
                   help="Staging dir for SDO objects. Needs ~590 MB per data worker "
                        "(~7 GB at the default 12).")
    p.add_argument("--job-id-prefix", default="AR-seg",
                   help="Prefix for job_id, which finetune.py:559 uses as the wandb run name: "
                        "'<prefix>_bootNN'. Set it per sample size (e.g. AR-seg_672hr) so "
                        "ensembles at different training-set sizes stay distinguishable in wandb.")
    p.add_argument("--num-data-workers", type=int, default=12,
                   help="data.num_data_workers. Set here rather than inherited so regeneration "
                        "cannot silently revert it: config.yaml ships 8, which is what let "
                        "boot01 drift from the rest of the ensemble. Scratch space and host RAM "
                        "both scale with this.")
    p.add_argument("--max-epochs", type=int, default=10,
                   help="optimizer.max_epochs. Pass 0 to inherit the base config's value.")
    p.add_argument("--seed-start", type=int, default=None,
                   help="Give each member its own model seed: member i gets --seed-start + i - 1. "
                        "Omit it (the default) to write no `seed` key at all, which is what the "
                        "four existing ensembles ran under -- every member at finetune.py's "
                        "default seed 0, so member spread is purely finite-sample. Pass 1 to "
                        "pair each arm with its seed-0 counterpart in the parent ensemble and "
                        "get a paired read on the optimization-path component. NOTE the "
                        "sentinel is None, not 0: `--seed-start 0` is a valid request that "
                        "writes seed 0..9, which is NOT the same as omitting the flag.")

    # Optimizer stability. The base config's lr 1e-3 with no schedule and no clipping is only
    # safe for short epochs: it held up at 168 steps/epoch, spiked on isolated epochs at 672
    # (run_672.log has one at validation loss 121.1, IoU 0.0000), and diverged outright at 2016
    # -- training BCE peaked at 930 in epoch 1 and validation IoU fell from 0.5672 to 0.1071
    # over the remaining four epochs. Anything at or above ~672 steps/epoch wants a lower LR
    # plus warmup and clipping; see the 2016hr invocation in the module docstring.
    p.add_argument("--learning-rate", type=float, default=0.0,
                   help="optimizer.learning_rate. Pass 0 to inherit the base config's value.")
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                   help="optimizer.lr_schedule, read by finetune.py:build_lr_scheduler. "
                        "'cosine' means linear warmup over --warm-up-steps then cosine decay to "
                        "optimizer.min_lr. 'constant' reproduces the pre-schedule behaviour that "
                        "the 168hrs and 672hr ensembles ran under.")
    p.add_argument("--warm-up-steps", type=int, default=0,
                   help="optimizer.warm_up_steps. Only has effect with --lr-schedule cosine. "
                        "~10%% of one epoch is a reasonable starting point.")
    p.add_argument("--grad-clip-norm", type=float, default=0.0,
                   help="optimizer.grad_clip_norm, the max_norm passed to clip_grad_norm_. "
                        "0 disables clipping, matching every run before this flag existed.")
    p.add_argument("--best-metric", default="valid/iou",
                   choices=["valid/iou", "valid/dice", "valid/loss"],
                   help="optimizer.best_metric -- which validation metric best.pth tracks. "
                        "IoU is the task metric and the default, but the two can disagree "
                        "sharply: on 672hr_stable boot01, IoU peaked at epoch 8 while valid/loss "
                        "bottomed out at epoch 0 and rose monotonically after epoch 4, so IoU "
                        "selection picked one of that run's worst-calibrated checkpoints. Prefer "
                        "valid/loss when the ensemble is being used for uncertainty estimates.")

    # Model capacity. The base config.yaml is rank-32/alpha-64 with no activation
    # checkpointing, which is tuned for a large GPU and a large training set. These defaults
    # match config_feb16_2013.yaml instead: far less adapter capacity for a 168-sample member,
    # and recompute on half the blocks so the run fits comfortably in memory.
    p.add_argument("--lora-r", type=int, default=8, help="model.lora_config.r.")
    p.add_argument("--lora-alpha", type=int, default=16, help="model.lora_config.lora_alpha.")
    p.add_argument("--checkpoint-layers", default="0,2,4,6,8",
                   help="Comma-separated block indices to activation-checkpoint, or 'none' for "
                        "no checkpointing. Default: 5 of the 10 blocks.")

    p.add_argument("--wandb-from", default="./config.yaml",
                   help="Copy every wandb_* key from this config. finetune.py:558,561 fall back "
                        "to entity 'nasa-impact' and mode 'offline' when the keys are absent, "
                        "so without this the runs would log to the wrong place. Pass 'none' to "
                        "keep the base config's wandb settings.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Omit --wandb from the generated runner. wandb.init only happens when "
                        "finetune.py is given that flag, so the wandb_* keys are inert without it.")
    args = p.parse_args()

    base = yaml.safe_load(open(args.base_config))
    reps = sorted(glob.glob(os.path.join(args.replicate_dir, args.replicate_glob)))
    if not reps:
        raise SystemExit(
            f"No replicates matching {args.replicate_glob} in {args.replicate_dir}. "
            f"Run bootstrap_subsample.py first."
        )

    for path in (args.valid_index, args.test_index):
        if not os.path.isfile(path):
            raise SystemExit(f"Evaluation index not found: {path}")
    if not os.path.isdir(args.ar_mask_root):
        raise SystemExit(f"Mask directory not found: {args.ar_mask_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    epochs = args.max_epochs or base["optimizer"]["max_epochs"]
    total = len(reps)
    run_lines, rows = [], []

    ckpt_layers = (
        [] if args.checkpoint_layers.strip().lower() == "none"
        else [int(x) for x in args.checkpoint_layers.split(",")]
    )

    wandb_keys = {}
    if args.wandb_from.strip().lower() != "none":
        if not os.path.isfile(args.wandb_from):
            raise SystemExit(f"--wandb-from config not found: {args.wandb_from}")
        src = yaml.safe_load(open(args.wandb_from))
        wandb_keys = {k: v for k, v in src.items() if k.startswith("wandb")}
        if not wandb_keys:
            raise SystemExit(f"No wandb_* keys found in {args.wandb_from}")

    for i, rep_path in enumerate(reps, start=1):
        n_rows = len(pd.read_csv(rep_path))
        cfg = copy.deepcopy(base)

        # --- the three per-member differences ---
        cfg["job_id"] = f"{args.job_id_prefix}_boot{i:02d}"
        cfg["data"]["ar_index_train"] = [f"./{os.path.relpath(rep_path, '.')}"]
        experiment = os.path.join(args.experiment_root, f"boot{i:02d}")
        cfg["path_experiment"] = experiment

        # --- the optional fourth: a per-member model seed ---
        # Inserted immediately after job_id rather than appended, so `seed` sits at the top of
        # the file beside the other thing that identifies this arm. Skipped entirely when
        # --seed-start is absent, which keeps regeneration of the seed-0 ensembles byte-exact.
        seed = None if args.seed_start is None else args.seed_start + i - 1
        if seed is not None:
            rebuilt = {}
            for key, value in cfg.items():
                rebuilt[key] = value
                if key == "job_id":
                    rebuilt["seed"] = seed
            cfg = rebuilt

        # --- held identical across members ---
        cfg["data"]["ar_index_valid"] = [args.valid_index]
        cfg["data"]["ar_index_test"] = [args.test_index]

        # --- corrections to the shipped config ---
        cfg["data"]["ar_mask_root_path"] = args.ar_mask_root
        cfg["data"]["s3_scratch_dir"] = args.s3_scratch_dir
        cfg["data"]["num_data_workers"] = args.num_data_workers
        cfg["iters_per_epoch_train"] = n_rows
        cfg["optimizer"]["max_epochs"] = epochs

        # --- optimizer stability ---
        if args.learning_rate:
            cfg["optimizer"]["learning_rate"] = args.learning_rate
        cfg["optimizer"]["lr_schedule"] = args.lr_schedule
        cfg["optimizer"]["warm_up_steps"] = args.warm_up_steps
        cfg["optimizer"]["grad_clip_norm"] = args.grad_clip_norm
        cfg["optimizer"]["best_metric"] = args.best_metric

        # --- capacity / memory, and wandb destination ---
        cfg["model"]["lora_config"]["r"] = args.lora_r
        cfg["model"]["lora_config"]["lora_alpha"] = args.lora_alpha
        cfg["model"]["checkpoint_layers"] = list(ckpt_layers)
        cfg.update(wandb_keys)

        name = f"config_boot{i:02d}.yaml"
        out = os.path.join(args.out_dir, name)
        with open(out, "w") as fh:
            seed_line = "" if seed is None else SEED_LINE.format(
                seed=seed, offset=42 + seed, rep=i)
            fh.write(HEADER.format(rep=i, total=total, train=rep_path, n_rows=n_rows,
                                   experiment=experiment, seed_line=seed_line))
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)

        run_lines.append(RUN_LINE.format(rep=i, total=total,
                                         wandb="" if args.no_wandb else " --wandb",
                                         config=os.path.relpath(out, ".")))
        row = {"member": i, "config": out, "ar_index_train": rep_path,
               "n_rows": n_rows, "path_experiment": experiment}
        if seed is not None:
            row["seed"] = seed
        rows.append(row)

    n_rows = rows[0]["n_rows"]
    gb_epoch = n_rows * 590 / 1000
    runner_path = args.runner
    seed_note = ""
    if args.seed_start is not None:
        seed_note = (
            f"#\n# SEED SWEEP: member i trains at model seed {args.seed_start}+i-1, i.e. seeds "
            f"{args.seed_start}..{args.seed_start + total - 1}, so\n"
            f"# replicate and optimization path vary together and across-member spread is the "
            f"TOTAL of\n"
            f"# the two variance components. Difference the arms against their seed-0 "
            f"counterparts in the\n"
            f"# parent ensemble, member by member, for a paired read on the optimization-path "
            f"part.\n"
        )
    with open(runner_path, "w") as fh:
        fh.write(RUNNER.format(total=total, n_rows=n_rows, epochs=epochs, gb_epoch=gb_epoch,
                               tb_member=gb_epoch * epochs / 1000,
                               tb_total=gb_epoch * epochs * total / 1000,
                               seed_note=seed_note,
                               runs="\n".join(run_lines)))
    os.chmod(runner_path, os.stat(runner_path).st_mode | stat.S_IXUSR | stat.S_IXGRP)

    index = pd.DataFrame(rows)
    index_path = os.path.join(args.out_dir, "ensemble_index.csv")
    index.to_csv(index_path, index=False)

    print(f"wrote {total} member config(s) to {args.out_dir}")
    print(f"  runner:   {runner_path}")
    print(f"  index:    {index_path}")
    print(f"  epochs:   {epochs}   samples/member: {n_rows}   "
          f"workers: {args.num_data_workers}")
    print(f"  lora:     r={args.lora_r} alpha={args.lora_alpha}   "
          f"checkpoint_layers={ckpt_layers}")
    lr = args.learning_rate or base["optimizer"]["learning_rate"]
    print(f"  optim:    lr={lr:g}  schedule={args.lr_schedule}  "
          f"warm_up_steps={args.warm_up_steps}  grad_clip_norm={args.grad_clip_norm:g}  "
          f"best_metric={args.best_metric}")
    if wandb_keys:
        print(f"  wandb:    from {args.wandb_from} -> "
              + ", ".join(f"{k}={v!r}" for k, v in sorted(wandb_keys.items())))
        if args.no_wandb:
            print("            NOTE: --no-wandb, so the runner omits --wandb and these are inert")
    if args.seed_start is None:
        print("  seed:     no `seed` key written -- every member at finetune.py's default 0 "
              "(pure finite-sample spread)")
    else:
        print(f"  seed:     {args.seed_start}..{args.seed_start + total - 1}, one per member "
              f"(spread = finite-sample + optimization-path)")
    print(f"  valid:    {args.valid_index}   (identical for all {total})")
    print(f"  test:     {args.test_index}   (carried only; finetune.py does not read it)")
    print(f"  masks:    {args.ar_mask_root}")
    print(f"  ckpt dir: {args.experiment_root}/boot01 .. boot{total:02d}")
    print(index[["member", "ar_index_train", "n_rows"]].to_string(index=False))
    print(f"\nrun with:  bash {runner_path}")


if __name__ == "__main__":
    main()
