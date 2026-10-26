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

Two training regimes exist, selected by `--regime`: `168hrs` (168-sample, one-week
bootstrap replicates) and `672hr` (672-sample, one-month replicates). They differ only
in how much data each member saw -- same LoRA geometry, same backbone, same validation
set -- so the regime picks the weights/config/output roots and the best-epoch table and
changes nothing else.

Example:
    PYTHONPATH=<repo root> python run_bootstrap_inference.py \
        --output_root /efs/bathomas/ar_seg_bootstrap/168hrs_infer

    PYTHONPATH=<repo root> python run_bootstrap_inference.py --regime 672hr \
        --nc_dir /efs/bathomas/assets/holdout_nc_11930 \
        --output_root /efs/bathomas/ar_seg_bootstrap/672hr_infer_holdout_top3
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
}

DEFAULT_REGIME = "168hrs"
DEFAULT_NC_DIR = "./assets/infer_data"


def regime_defaults(regime: str) -> dict:
    """Weights / config / output roots for a regime, all named after it on disk."""
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
            "Training regime: 168hrs (one-week replicates) or 672hr (one-month). Sets "
            "the best-epoch table and the default weights/config/output roots; the "
            "three --*_root flags below override those defaults individually."
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
            raw, attrs = read_hc_file(nc_path, channels)
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
