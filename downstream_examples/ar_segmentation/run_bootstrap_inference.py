"""Run every bootstrap ensemble member's best checkpoint over every inference .nc file.

Produces the raw material for the uncertainty analysis: for each of the 15 HelioFM
cubes in `assets/infer_data`, the AR probability map predicted by each of the 10
bootstrap members. Member-to-member spread on a fixed frame is the quantity of
interest -- see `/efs/bathomas/ar_seg_bootstrap/168hrs/README.md`.

Differs from `infer_file.py` in three ways, all driven by running 150 inferences
rather than one:

  * The model config is read from each member's own training config, so the LoRA
    geometry (r=8, targets fc1/fc2/attn.qkv/attn.proj) always matches the
    checkpoint. `config_infer.yaml` describes a *different* LoRA (r=32, other
    targets) and will not load these checkpoints.
  * Output is one array per (member, file) rather than one figure per channel.
    The segmentation head is single-channel, so the mask does not vary by channel
    and writing it 8 times per member is pure duplication. Channel images are
    written once per file instead, shared by every member.
  * It is resumable. Existing outputs are skipped, so an interrupted run resumes
    where it stopped.

Three bootstrap regimes exist, selected by `--regime`: `168hrs` (168-sample, one-week
bootstrap replicates), `672hr` (672-sample, one-month replicates) and `672hr_stable`
(the same 672-sample replicates retrained with warmup + cosine decay + gradient
clipping, so no member collapses after its peak). They differ only in how much data each
member saw and how it was optimized -- same LoRA geometry, same backbone, same
validation set -- so the regime picks the weights/config/output roots and the best-epoch
table and changes nothing else.

A fourth key, `672hr_stable_seed`, is NOT a bootstrap: its three "members" are seed arms
on one shared replicate (672hr_stable's boot01), so member spread there measures the
optimization-path component of model uncertainty rather than the finite-sample one. It is
named after its parent regime because that is exactly what it is -- 672hr_stable's recipe
and replicate, reseeded -- and the name keeps it adjacent to `672hr_stable` on disk and
in this table. It follows the standard naming convention, so no --*_root flags are needed.

Example:
    PYTHONPATH=<repo root> python run_bootstrap_inference.py \
        --output_root /efs/bathomas/ar_seg_bootstrap/168hrs_infer

    PYTHONPATH=<repo root> python run_bootstrap_inference.py --regime 672hr \
        --nc_dir /efs/bathomas/assets/holdout_nc_11930 \
        --output_root /efs/bathomas/ar_seg_bootstrap/672hr_infer_holdout_top3

    PYTHONPATH=<repo root> python run_bootstrap_inference.py --regime 672hr_stable_seed \
        --nc_dir ./assets/holdout_nc_nonflaring30 \
        --output_root \
          /efs/bathomas/ar_seg_bootstrap/672hr_stable_seed_infer_holdout_nonflaring30
"""

import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from infer import load_model
from infer_file import normalize, read_hc_file, timestamp_from_file
from surya.datasets.disk_alignment import corner_scaling_kwargs
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed

# Best epoch per member per regime, by validation IoU on the shared 200-frame
# validation set. Keyed (epoch, val_iou, wandb_run_id).
#
# Re-derive with:
#   grep -o 'IoU: [0-9.]*' <wandb run>/files/output.log   # one line per epoch, 0-indexed
#
# `epoch_N.pth` is saved and *then* evaluated (finetune.py's epoch loop), so the Nth
# validation line is the score of epoch_N.pth. Map a wandb run to its member by the
# --config_path in its files/wandb-metadata.json `args`.
BEST_EPOCH_BY_REGIME = {
    # Confirmed 2026-10-22 against the output.log of every run listed in
    # /efs/bathomas/ar_seg_bootstrap/168hrs/README.md, which tabulates the same values.
    # boot01 trained 20 epochs, the rest 10; its peak at epoch 8 is never beaten in the
    # twelve later epochs.
    "168hrs": {
        "boot01": (8, 0.5170, "b919pvy1"),
        "boot02": (9, 0.5050, "zz6vdbsd"),
        "boot03": (9, 0.5256, "xu7fqbem"),
        "boot04": (5, 0.5344, "jkdf1iu9"),
        "boot05": (6, 0.5539, "zeddewff"),
        "boot06": (4, 0.4950, "m2mmkryf"),
        "boot07": (3, 0.4812, "noa3fm8f"),
        "boot08": (8, 0.4927, "ftqm9isf"),
        "boot09": (8, 0.4918, "v7ea6fcs"),
        "boot10": (7, 0.5194, "wepnogck"),
    },
    # Derived 2026-10-26 from /efs/bathomas/wandb. boot01 trained 10 epochs, the rest 5.
    # Peaks land early -- epoch 0 to 3 -- and most members then collapse hard (boot02 to
    # 0.0000, boot06 to 0.0572, boot03 to 0.1935), so "best epoch" selects much more
    # aggressively here than in the 168hrs regime. boot08 peaks at epoch 0, making it a
    # one-epoch model; it is kept because the ensemble is defined as every run's best.
    # There is a second config_boot01 run (xf5z2wlg) that logged zero epochs -- a crashed
    # start six minutes before nb9lm7a4; nb9lm7a4 is the real one.
    "672hr": {
        "boot01": (2, 0.6036, "nb9lm7a4"),
        "boot02": (1, 0.5608, "dtcinrgm"),
        "boot03": (2, 0.6169, "g03bele4"),
        "boot04": (1, 0.6314, "1gu2f1ak"),
        "boot05": (2, 0.6341, "ga1mu7nf"),
        "boot06": (1, 0.6089, "w9g5wx72"),
        "boot07": (3, 0.6104, "s32mafh0"),
        "boot08": (0, 0.5177, "9rb2iybc"),
        "boot09": (1, 0.5621, "6xowxfq6"),
        "boot10": (2, 0.5699, "0c76ze9m"),
    },
    # Derived 2026-10-29 from the 60 "Validation -- IoU:" lines of
    # /efs/bathomas/run_bootstrap_ensemble_672hr_stable_rerun.log, one clean sequential
    # pass of all ten members at 6 epochs (warmup 70 + cosine + grad_clip_norm 1.0).
    #
    # Unlike 672hr, no member collapses: every run's last epoch is within 0.012 IoU of its
    # own peak, so "best epoch" barely selects here (mean best-minus-final 0.0020) and the
    # table is far less load-bearing than it is above. Peaks land at epoch 3-5 rather than
    # 0-3, because the stabilized LR schedule keeps improving instead of blowing up.
    #
    # Two other 672hr_stable runs are in /efs/bathomas/wandb and NEITHER is a member:
    # k5gwnlfb is the 10-epoch boot01 pilot, whose epoch_6-9.pth are quarantined under
    # boot01/stale_10ep_k5gwnlfb/ (a 6720-step cosine, not comparable to these 4032-step
    # ones); 35u5h5xg is a boot02 start that was SIGINT-ed 365 batches into epoch 0 and
    # wrote no checkpoint at all. See NOTES, "672hr_stable full ensemble".
    "672hr_stable": {
        "boot01": (5, 0.6006, "j6huxcrn"),
        "boot02": (4, 0.6018, "s96wimmi"),
        "boot03": (5, 0.6574, "9ei4z7gv"),
        "boot04": (5, 0.6452, "vivfkjiv"),
        "boot05": (4, 0.6573, "wacgmsao"),
        "boot06": (5, 0.6178, "tcu7qblb"),
        "boot07": (5, 0.6597, "f76zjsdj"),
        "boot08": (4, 0.5968, "rkbzguhx"),
        "boot09": (4, 0.6170, "s9vokocg"),
        "boot10": (3, 0.6208, "bxx09je1"),
    },
    # Derived 2026-10-30 from the .wandb datastores of the three arms, cross-checked
    # against each run's six "Validation -- IoU:" lines. NOT a bootstrap ensemble: all
    # three arms train the SAME replicate (train_boot001.csv) and differ only in `seed`,
    # so "member" here means seed arm, and the spread measures the optimization-path
    # variance component the other three regimes hold fixed. See NOTES, "Seed ablation".
    #
    # seed00 IS 672hr_stable/boot01 -- set_global_seed(0) is manual_seed(42) and the
    # sampler's default seed is already 0 -- so it was never retrained. Its (epoch, IoU)
    # therefore matches boot01's above exactly, as it must. Its weights live under
    # 672hr_stable/boot01, so 672hr_stable_seed/seed00 is a SYMLINK to ../672hr_stable/
    # boot01, which keeps the naming convention intact rather than special-casing it here.
    #
    # seed02's selection sits on a 0.0076 IoU margin over its own epoch 5 (0.6053 vs
    # 0.5977). On a 200-frame validation set that is selection noise, and it is the
    # cleanest illustration in this repo of why IoU-based epoch picking is fragile: the
    # same recipe, perturbed only in its seed, hands best.pth a mid-run epoch whose
    # advantage would not survive a different validation draw. Both other arms pick ep5.
    "672hr_stable_seed": {
        "seed00": (5, 0.6006, "j6huxcrn"),
        "seed01": (5, 0.6004, "bm06c9dt"),
        "seed02": (2, 0.6053, "x7m7u8fz"),
    },
    # The diagonal seed sweep at the 672-sample budget, re-selected by `epoch_select.py` on
    # 2026-11-04. Like the two entries below, these epochs are NOT the valid/iou argmax; the
    # third value is validation IoU at the CHOSEN epoch, so it is at or below the run's peak.
    #     python epoch_select.py --sweep-arm 672hr_stable_seed_diag \
    #         --baseline-arm 672hr_stable
    #
    # THIS ARM MIXES THREE EPOCH BUDGETS, so epoch indices are NOT comparable across members and
    # per-member selection is mandatory rather than tidy:
    #     boot06, boot07, boot09, boot10   6 epochs, retrained 2026-11-03/04 (4032-step cosine)
    #     boot01                           5 epochs (3360-step cosine)
    #     boot02-boot05, boot08            4 epochs (2688-step cosine)
    # The 4/5-epoch budgets are the original drift -- configs regenerated below the parent's
    # uniform 6 after the runner was written, the same defect recorded for
    # 168hrs_stable_seed_diag below. Only four members were brought up to the parent's budget
    # (/efs/bathomas/run_seed_ablation_672hr_stable_short.log).
    #
    # THE RETRAIN DID NOT IMPROVE THOSE FOUR, which is why the other six were left alone.
    # Best-generalized IoU against their own archived 4-epoch counterparts: boot06 +0.0154,
    # boot10 +0.0029, but boot07 -0.0143 and boot09 -0.0130; all four still sit below their
    # parent picks (-0.0020, -0.0127, -0.0252, -0.0099). All four also moved from
    # `no-degradation` to `overfitting`. The mechanism is the schedule, not the seed: a 4-epoch
    # cosine is a COMPLETE schedule annealed to min_lr, so its last checkpoint is fully settled,
    # whereas 6 epochs spend two more at meaningful LR and buy IoU creep by degrading the
    # probability fit. best.pth is epoch 5 for all four, at +12.8%, +12.8%, +16.9% and +2.0%
    # over their own val-loss floors.
    #
    # Two entries not to read at face value:
    #   boot07 -- floor at epoch 1, +11.4% by epoch 3, IoU still climbing to epoch 5, so its
    #     window collapses to 3-3 and the pick is 8.43 sigma below its own peak via the
    #     "none within noise of run max" fallback. No checkpoint here is both calibrated and
    #     IoU-competitive; same pathology as 168hrs boot05 below.
    #   boot08 -- its val loss at the pick reads +17.0% over the floor, but that is NOT a
    #     penalty: the floor is epoch 3, the last epoch, so the epoch-2 pick sits on the
    #     pre-floor descent rather than past it.
    #
    # Across the merged arm the pick differs from best.pth in 9/10 members; mean IoU is 0.6116
    # at these picks against 0.6201 at best.pth, so 0.0085 mean IoU buys the calibration back.
    #
    # THE SEED CONTRAST IS STILL CONFOUNDED. Only 4 of 10 members reached the parent's budget,
    # so a paired difference against 672hr_stable still mixes seed with epoch budget -- see
    # CONFOUND_NOTES in epoch_select.py. These tuples are inference-ready; the seed contrast is
    # not, and needs the remaining six retrained at 6 epochs.
    #
    # PATHS: regime_defaults() now resolves both roots with no flags. boot01-boot05 and boot08
    # are symlinks under ar_seg_bootstrap/672hr_stable_seed_diag/ into
    # 672hr_stable_seed_diag.4ep.20261031/ (the 672hr_stable_seed/seed00 convention), because
    # their weights were moved aside on 2026-11-03 so the retrain could not overwrite them.
    # The unsuffixed config dir carries max_epochs 6 for all ten even though six trained at 4-5;
    # that is harmless here because inference reads no optimizer key, but it does mean the config
    # is no longer a record of how those six were trained -- the archived 4ep config dir is.
    # Note epoch_select.py can no longer rediscover the 4ep runs for boot06/07/09/10: discovery
    # keys on the config-dir string in each run's wandb-metadata.json, identical for both, and
    # resolves to the newest run per member. The durable records are
    # /efs/bathomas/ar_seg_bootstrap/672hr_stable_seed_diag_epoch_selection.csv (this selection)
    # and ..._epoch_selection.4ep.20261031.csv (the superseded all-4ep one).
    "672hr_stable_seed_diag": {
        "boot01": (3, 0.5942, "gcbilj9y"),
        "boot02": (2, 0.5833, "m34bg8j8"),
        "boot03": (1, 0.6396, "d9sqd0sl"),
        "boot04": (1, 0.6263, "3ufxzmx1"),
        "boot05": (2, 0.6469, "5iwqnbbb"),
        "boot06": (5, 0.6084, "4yzuigs0"),
        "boot07": (3, 0.6470, "ras44jxi"),
        "boot08": (2, 0.5830, "ub25vmph"),
        "boot09": (2, 0.5767, "nj2edxhx"),
        "boot10": (4, 0.6110, "en38qsxh"),
    },
    # The two entries below were selected by `epoch_select.py` (then named
    # `epoch_select_168hrs.py`) on 2026-10-31, which mechanizes the eight rules under
    # "HOW TO TELL IF A RUN IS OVER- OR UNDERTRAINED" in NOTES.
    # They are therefore, with the entry above and `672hr_stable_generalized` at the bottom, the
    # only entries in this table whose epoch is NOT
    # the valid/iou argmax: the third value is still each member's validation IoU, but at the
    # *chosen* epoch, so it is below its own run's peak by construction (every pick is within 2
    # jitter sigma of that peak, except 168hrs boot05 -- see below). Re-derive either entry with:
    #     python epoch_select.py --sweep-arm 168hrs_stable_seed_diag --baseline-arm 168hrs \
    #         --out /efs/bathomas/ar_seg_bootstrap/168hrs_epoch_selection.csv
    # which prints both blocks paste-ready and writes that CSV with the window, the jitter sigma,
    # the trend tests and a one-line reason per member. Pass --out for this pair only: its CSV is
    # named after the BASELINE arm, where the script now defaults to the sweep arm's name.
    #
    # The diagonal seed sweep: arm i trains replicate boot_i at seed i, stabilized recipe.
    # boot01 is on a 10-epoch/1680-step cosine while boot02-boot10 are on 4-epoch/672-step ones
    # (the budget drift NOTES flagged and nobody fixed before the run), so epoch indices are NOT
    # equivalent across members here and per-member selection is mandatory rather than tidy.
    "168hrs_stable_seed_diag": {
        "boot01": (2, 0.4816, "69tyui4d"),
        "boot02": (3, 0.5090, "onw713rt"),
        "boot03": (1, 0.4792, "mi7ff4jy"),
        "boot04": (2, 0.4644, "ys7gyg5q"),
        "boot05": (2, 0.5140, "ewdcm1lj"),
        "boot06": (0, 0.3795, "83vxunn2"),
        "boot07": (3, 0.4953, "sonnz3ww"),
        "boot08": (2, 0.4418, "fxhzd259"),
        "boot09": (1, 0.4814, "lx5xaa7l"),
        "boot10": (3, 0.4789, "lv2ka4t6"),
    },
    # The SHIPPED 168hrs ensemble re-selected at best-generalized epochs -- same ten checkpoints
    # directories as "168hrs" above, different epoch out of each. Deliberately does NOT follow the
    # naming convention, because there is no ar_seg_bootstrap/168hrs_generalized on disk: run it
    # with --weights_root/--config_dir pointed at 168hrs and its own --output_root, the same shape
    # as the 672hr_stable_minloss re-selection. "168hrs" is left untouched; it is the best-IoU
    # record that README.md and NOTES cite.
    #
    # Two members to know about. boot08 and boot09 pick epoch 0 because their jitter sigma is
    # 0.059 and 0.047 -- no later epoch ever takes new high ground by 2 sigma, so every epoch is
    # statistically tied and epoch 0 is the cheapest. That is the constant-lr operating-point
    # thrash the 168hrs README documents, not a selection bug. boot05 is the one pick further than
    # 2 sigma from its own peak (4.09): its val loss is already +28 % above its floor by epoch 2,
    # so its window collapses to a single epoch and no checkpoint is both calibrated and
    # IoU-competitive.
    "168hrs_generalized": {
        "boot01": (6, 0.4959, "b919pvy1"),
        "boot02": (8, 0.4974, "zz6vdbsd"),
        "boot03": (4, 0.4700, "xu7fqbem"),
        "boot04": (2, 0.5233, "jkdf1iu9"),
        "boot05": (2, 0.4984, "zeddewff"),
        "boot06": (3, 0.4904, "m2mmkryf"),
        "boot07": (1, 0.4608, "noa3fm8f"),
        "boot08": (0, 0.4089, "ftqm9isf"),
        "boot09": (0, 0.4195, "v7ea6fcs"),
        "boot10": (4, 0.4610, "wepnogck"),
    },
    # The SHIPPED 672hr_stable ensemble re-selected at best-generalized epochs -- same ten
    # checkpoint directories as "672hr_stable" above, a different epoch out of each. Like
    # 168hrs_generalized it does NOT follow the naming convention, because there is no
    # ar_seg_bootstrap/672hr_stable_generalized on disk: run it with --weights_root/--config_dir
    # pointed at 672hr_stable and its own --output_root. "672hr_stable" is left untouched; it is
    # the best-IoU record README.md and NOTES cite, and the root
    # 672hr_stable_infer_holdout_nonflaring30 was produced from it.
    #
    # WHY THIS ENTRY EXISTS: it is the epoch-matched seed-0 baseline for
    # 672hr_stable_seed_diag above. That arm's tuples are epoch_select picks, so pairing it
    # against the best-IoU 672hr_stable root would difference seed, epoch budget AND the
    # selection criterion at once. NOTES ("Under-training in 672hr_stable_seed_diag") measures
    # how much that matters on validation: re-selecting ONE arm leaves a paired difference of
    # -0.0101 (t(9) = -3.20), re-selecting BOTH shrinks it to -0.0041 (t(9) = -1.20, p 0.26).
    # README also warns that section 7 calibration at generalized picks is not comparable to any
    # best.pth figure in this project. This entry is what makes the holdout contrast epoch-policy
    # matched; it changes nothing about the sweep.
    #
    # Derived 2026-11-04 from the `672hr_stable` rows of
    # /efs/bathomas/ar_seg_bootstrap/672hr_stable_seed_diag_epoch_selection.csv, which
    #     python epoch_select.py --sweep-arm 672hr_stable_seed_diag --baseline-arm 672hr_stable
    # writes (it re-selects the baseline arm as well as the sweep, and prints both blocks
    # paste-ready). The pick differs from best.pth in 7 of 10 members -- only boot02, boot07 and
    # boot10 already sat at a generalized epoch -- and mean validation IoU falls 0.6274 -> 0.6166,
    # which is the 0.011 the calibration costs. 8 of 10 members diagnose as `overfitting` at
    # best.pth, which is rule 8's point: the valid/iou argmax is systematically the
    # worst-calibrated checkpoint of its run.
    "672hr_stable_generalized": {
        "boot01": (3, 0.5854, "j6huxcrn"),
        "boot02": (4, 0.6018, "s96wimmi"),
        "boot03": (2, 0.6279, "9ei4z7gv"),
        "boot04": (2, 0.6223, "vivfkjiv"),
        "boot05": (3, 0.6481, "wacgmsao"),
        "boot06": (2, 0.6104, "tcu7qblb"),
        "boot07": (5, 0.6597, "f76zjsdj"),
        "boot08": (2, 0.5880, "rkbzguhx"),
        "boot09": (2, 0.6018, "s9vokocg"),
        "boot10": (3, 0.6208, "bxx09je1"),
    },
}

DEFAULT_REGIME = "168hrs"
DEFAULT_NC_DIR = "./assets/infer_data"


def regime_defaults(regime: str) -> dict:
    """Weights / config / output roots for a regime, all named after it on disk.

    Every regime resolves here, `672hr_stable_seed` included -- that regime was renamed
    from `seed_ablation` (and its directories moved) precisely so it would, instead of
    needing three explicit --*_root flags. Its one irregularity is handled on disk rather
    than in code: seed00 was never retrained, so 672hr_stable_seed/seed00 is a symlink to
    ../672hr_stable/boot01.

    The two exceptions are `168hrs_generalized` and `672hr_stable_generalized`, neither of
    which is a training run at all: they are the shipped `168hrs` / `672hr_stable`
    checkpoints re-selected at a different epoch. There is nothing named after either on
    disk, so each needs --weights_root and --config_dir pointed at its parent plus its own
    --output_root; the paths this returns for them do not exist. A re-selection is not a
    regime, and giving one a directory tree of symlinks to pretend otherwise would
    duplicate 18 GB of checkpoints to save two flags.
    """
    return {
        "weights_root": f"/efs/bathomas/ar_seg_bootstrap/{regime}",
        "config_dir": f"/efs/bathomas/configs_bootstrap/{regime}",
        "output_root": f"/efs/bathomas/ar_seg_bootstrap/{regime}_infer",
    }


def load_member_config(config_dir: str, member: str, dtype: torch.dtype) -> dict:
    """Read a member's training config and adapt it for inference.

    Only three fields change. Everything describing the model -- depth, embed_dim,
    and above all `lora_config` -- is left exactly as trained, because the
    checkpoint is loaded with strict=True and any mismatch is a hard failure.
    """
    path = os.path.join(config_dir, f"config_{member}.yaml")
    config = yaml.safe_load(open(path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    # Activation checkpointing trades compute for memory during backward. Under
    # no_grad there is no backward, so it only costs time.
    config["model"]["checkpoint_layers"] = []
    config["dtype"] = dtype
    return config


def write_channels(path: str, ts: np.ndarray, channels: list[str], source: str, timestamp: str):
    """Write the normalized channel cube once per input file, shared by all members.

    Stored in the standardized domain the model consumes. `infer_file.to_display_units`
    converts to the signum-log domain the scalers' min/max (and so the colour bars)
    are expressed in; the notebook does that at display time.
    """
    ds = xr.Dataset(
        data_vars={ch: (("y", "x"), ts[i].astype(np.float16)) for i, ch in enumerate(channels)},
        attrs={
            "title": "Normalized HelioFM channels (model input)",
            "source_file": source,
            "data_time": timestamp,
            "units": "standardized: (signum-log(v) - mean) / (std + epsilon), per scalers.yaml",
        },
    )
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()


def write_probability(path: str, probability: np.ndarray, **attrs):
    """Write one member's probability map for one frame.

    float16 keeps the ensemble to ~5 GB. Its worst-case step near p=0.5 is ~5e-4,
    an order of magnitude below the ~5e-3 fp16-vs-fp32 spread of the forward pass
    itself, so it adds nothing material to the error budget.
    """
    ds = xr.Dataset(
        data_vars={"ar_probability": (("y", "x"), probability.astype(np.float16))},
        attrs={"title": "Surya AR segmentation probability", **attrs},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()


def predict(model, ts: np.ndarray, device, dtype, device_type: str) -> np.ndarray:
    """Run the model on one normalized cube and return an (H, W) probability map.

    Mirrors `infer_file.predict`, but builds the input tensor on the host in the
    autocast dtype so the 872 MB fp32 cube is never resident on the GPU.
    """
    batch = {
        # B, C, T, H, W -- a single input timestamp, so T = 1.
        "ts": torch.from_numpy(ts).unsqueeze(0).unsqueeze(2).to(device, dtype=dtype),
        # Offset of each input frame from the reference frame, in hours.
        "time_delta_input": torch.zeros(1, 1, dtype=torch.float32).to(device),
    }

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            logits = model(batch)

    if logits.ndim == 5:
        logits = logits[:, 0]

    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        "Run every bootstrap member's best checkpoint over every inference .nc file"
    )
    parser.add_argument(
        "--regime",
        default=DEFAULT_REGIME,
        choices=sorted(BEST_EPOCH_BY_REGIME),
        help=(
            "Which ensemble to run. Bootstrap regimes: 168hrs (one-week replicates), "
            "672hr (one-month), 672hr_stable (one-month, stabilized LR). "
            "672hr_stable_seed is the seed ablation -- one replicate, three seeds, so "
            "member spread is optimization-path rather than finite-sample variance. "
            "The two *_stable_seed_diag arms are diagonal seed sweeps (replicate i at "
            "seed i), so their spread is both components at once; both read at "
            "epoch_select picks and both on mixed epoch budgets. "
            "168hrs_generalized and 672hr_stable_generalized are not training runs but "
            "the shipped 168hrs / 672hr_stable checkpoints re-selected at generalized "
            "epochs, and are the epoch-matched baselines for the two seed sweeps; they "
            "have no directories of their own, so they need --weights_root/--config_dir "
            "pointed at the parent plus their own --output_root. "
            "Sets the best-epoch table and, for every other regime, the default "
            "weights/config/output roots; the three --*_root flags below override those "
            "defaults individually."
        ),
    )
    # Default None rather than a regime-derived path, so an explicit flag is
    # distinguishable from a default and the regime can fill the rest in below.
    parser.add_argument("--weights_root", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--nc_dir", default=DEFAULT_NC_DIR)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--members",
        nargs="+",
        default=None,
        help="Subset of members to run. Defaults to all ten of the regime.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Run only the first N .nc files. For smoke tests.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help=(
            "Autocast dtype. float16 on the T4 (sm_75): it has no native bfloat16, so "
            "bf16 is emulated at ~2.5x the wall time while tracking fp32 less closely "
            "(max |dp| 0.057 vs 0.005 on a reference frame)."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="AR call threshold.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    best_epoch = BEST_EPOCH_BY_REGIME[args.regime]
    defaults = regime_defaults(args.regime)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.members is None:
        args.members = sorted(best_epoch)

    set_global_seed(42)

    print(f"Regime {args.regime}:  weights {args.weights_root}  "
          f"configs {args.config_dir}  ->  {args.output_root}")

    dtype = getattr(torch, args.dtype)
    if args.device == "cuda" and torch.cuda.is_available():
        device, device_type = torch.device("cuda"), "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device, device_type = torch.device("cpu"), "cpu"
        print("Using CPU")

    nc_files = sorted(glob.glob(os.path.join(args.nc_dir, "*.nc")))
    assert nc_files, f"No .nc files under {args.nc_dir}"
    if args.max_files:
        nc_files = nc_files[: args.max_files]
    print(f"{len(args.members)} members x {len(nc_files)} files = "
          f"{len(args.members) * len(nc_files)} inferences")

    os.makedirs(args.output_root, exist_ok=True)
    rows = []

    # Member-outer: each checkpoint is 1.8 GB, so loading once per member costs 10
    # loads instead of 150. The cost is re-reading each cube per member, which is
    # the cheaper half (~5 s read vs ~7 s load) and parallel-free either way.
    for member in args.members:
        if member not in best_epoch:
            raise SystemExit(
                f"Unknown member {member!r} for regime {args.regime}; "
                f"known: {sorted(best_epoch)}"
            )
        epoch, val_iou, run_id = best_epoch[member]
        ckpt = os.path.join(args.weights_root, member, f"epoch_{epoch}.pth")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Missing checkpoint {ckpt}")

        print(f"\n=== {member}  epoch {epoch}  (val IoU {val_iou:.4f}, wandb {run_id}) ===")
        config = load_member_config(args.config_dir, member, dtype)
        channels = config["data"]["channels"]
        scalers = build_scalers(info=config["data"]["scalers"])

        model = load_model(copy.deepcopy(config), ckpt, device)

        for nc_path in nc_files:
            stem = os.path.splitext(os.path.basename(nc_path))[0]
            out_dir = os.path.join(args.output_root, stem)
            os.makedirs(out_dir, exist_ok=True)
            prob_path = os.path.join(out_dir, f"prob_{member}.nc")
            chan_path = os.path.join(out_dir, "channels.nc")

            if os.path.exists(prob_path) and not args.overwrite:
                print(f"  {stem}: exists, skipping")
                continue

            t0 = time.time()
            # Each member's own config decides, so inputs match how it was trained.
            raw, attrs = read_hc_file(
                nc_path, channels, corner_scaling=corner_scaling_kwargs(config["data"])
            )
            timestamp = timestamp_from_file(nc_path, attrs)
            ts = normalize(raw, channels, scalers)
            del raw
            t_read = time.time() - t0

            t0 = time.time()
            probability = predict(model, ts, device, dtype, device_type)
            t_fwd = time.time() - t0

            if not os.path.exists(chan_path) or args.overwrite:
                write_channels(chan_path, ts, channels, nc_path, timestamp)
            del ts

            frac = float((probability > args.threshold).mean())
            write_probability(
                prob_path,
                probability,
                member=member,
                regime=args.regime,
                epoch=epoch,
                val_iou=val_iou,
                wandb_run=run_id,
                source_file=nc_path,
                data_time=timestamp,
                threshold=args.threshold,
                checkpoint=ckpt,
                autocast_dtype=args.dtype,
            )
            rows.append(
                dict(member=member, regime=args.regime, epoch=epoch, val_iou=val_iou,
                     file=stem, data_time=timestamp, pred_frac=frac,
                     read_s=round(t_read, 1), fwd_s=round(t_fwd, 1))
            )
            print(f"  {stem}: AR {100 * frac:.3f}% of frame  "
                  f"(read {t_read:.1f}s, fwd {t_fwd:.1f}s)")

        del model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        manifest = os.path.join(args.output_root, "manifest.csv")
        df = pd.DataFrame(rows)
        # Append so a resumed run extends the manifest instead of truncating it.
        df.to_csv(manifest, mode="a", header=not os.path.exists(manifest), index=False)
        print(f"\nWrote {len(rows)} rows to {manifest}")
    print("Done.")


if __name__ == "__main__":
    main()
