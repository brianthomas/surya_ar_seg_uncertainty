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

Example:
    PYTHONPATH=<repo root> python run_bootstrap_inference.py \
        --output_root /efs/bathomas/ar_seg_bootstrap/168hrs_infer
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

# Best epoch per member, by validation IoU on the shared 200-frame validation set.
#
# Re-derive with:
#   grep -o 'IoU: [0-9.]*' <wandb run>/files/output.log   # one line per epoch, 0-indexed
#
# Confirmed 2026-10-22 against the output.log of every run listed in
# /efs/bathomas/ar_seg_bootstrap/168hrs/README.md, which tabulates the same values.
# boot01 trained 20 epochs, the rest 10; its peak at epoch 8 is never beaten in the
# twelve later epochs.
BEST_EPOCH = {
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
}

DEFAULT_WEIGHTS_ROOT = "/efs/bathomas/ar_seg_bootstrap/168hrs"
DEFAULT_CONFIG_DIR = "/efs/bathomas/configs_bootstrap/168hrs"
DEFAULT_NC_DIR = "./assets/infer_data"
DEFAULT_OUTPUT_ROOT = "/efs/bathomas/ar_seg_bootstrap/168hrs_infer"


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
    parser.add_argument("--weights_root", default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--config_dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--nc_dir", default=DEFAULT_NC_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--members",
        nargs="+",
        default=sorted(BEST_EPOCH),
        help="Subset of members to run. Defaults to all ten.",
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

    set_global_seed(42)

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
        if member not in BEST_EPOCH:
            raise SystemExit(f"Unknown member {member!r}; known: {sorted(BEST_EPOCH)}")
        epoch, val_iou, run_id = BEST_EPOCH[member]
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
                dict(member=member, epoch=epoch, val_iou=val_iou, file=stem,
                     data_time=timestamp, pred_frac=frac,
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
