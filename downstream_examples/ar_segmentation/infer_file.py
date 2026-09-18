"""Run AR-segmentation inference on a single HelioFM NetCDF (HC) file.

Unlike `infer.py`, which drives `HelioNetCDFDataset` off an index CSV and emits one
wide multi-panel figure, this script takes a single `.nc` cube (local path or
`s3://` URI), runs the model once, and writes one output file *per AIA channel*.

The segmentation head produces a single-channel mask, so the mask itself is the
same for every channel -- what varies per file is the AIA channel it is paired
with. Each per-channel output holds that channel's image, the predicted AR
probability, and the thresholded binary mask.

Example:
    python infer_file.py \
        --input_file ./assets/infer_data/20110120_0100.nc \
        --checkpoint_path ./assets/ar_segmentation_weights.pth \
        --output_dir ./inference_results/20110120_0100
"""

import argparse
import os
import re
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.visualization.colormaps as sunpy_cm
import torch
import xarray as xr
import yaml
from matplotlib.colors import ListedColormap

from surya.datasets.helio import transform
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed
from surya.utils.s3 import default_scratch_dir, is_s3_path, make_s3_client, parse_s3_uri

# Reused so this script and infer.py cannot drift on how a checkpoint is loaded.
from infer import load_model

# Same mapping infer.py's plot_sun_sdo_cmap2 uses internally.
CHANNEL_CMAPS = {
    "aia94": sunpy_cm.cmlist["sdoaia94"],
    "aia131": sunpy_cm.cmlist["sdoaia131"],
    "aia171": sunpy_cm.cmlist["sdoaia171"],
    "aia193": sunpy_cm.cmlist["sdoaia193"],
    "aia211": sunpy_cm.cmlist["sdoaia211"],
    "aia304": sunpy_cm.cmlist["sdoaia304"],
    "aia335": sunpy_cm.cmlist["sdoaia335"],
    "aia1600": sunpy_cm.cmlist["sdoaia1600"],
    "hmi_m": sunpy_cm.cmlist["hmimag"],
    "hmi_bx": sunpy_cm.cmlist["hmimag"],
    "hmi_by": sunpy_cm.cmlist["hmimag"],
    "hmi_bz": sunpy_cm.cmlist["hmimag"],
    "hmi_v": plt.get_cmap("bwr"),
}


def read_hc_file(filepath: str, channels: list[str], s3_anon: bool = False) -> np.ndarray:
    """Read one HelioFM NetCDF cube into a (C, H, W) array, in `channels` order.

    Accepts a local path or an ``s3://bucket/key`` URI. S3 objects are staged to
    scratch, read, and deleted -- the same download-then-read path
    HelioNetCDFDataset uses, and for the same reason (these cubes stream badly).
    """
    if not is_s3_path(filepath):
        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None, cache=False) as ds:
            return ds[channels].to_array().load().to_numpy(), dict(ds.attrs)

    scratch = default_scratch_dir()
    os.makedirs(scratch, exist_ok=True)
    local_path = os.path.join(scratch, f"{uuid4().hex}.nc")
    bucket, key = parse_s3_uri(filepath)
    client = make_s3_client(anon=s3_anon)
    try:
        print(f"Downloading {filepath} to {local_path}")
        client.download_file(bucket, key, local_path)
        with xr.open_dataset(local_path, engine="h5netcdf", chunks=None, cache=False) as ds:
            return ds[channels].to_array().load().to_numpy(), dict(ds.attrs)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def normalize(data: np.ndarray, channels: list[str], scalers) -> np.ndarray:
    """Apply the signum-log + standardization the model was trained under.

    Mirrors HelioNetCDFDataset.transform_data, minus the optional pooling, which
    inference at native 4096 resolution does not use.
    """
    means = np.array([scalers[ch].mean for ch in channels])
    stds = np.array([scalers[ch].std for ch in channels])
    epsilons = np.array([scalers[ch].epsilon for ch in channels])
    sl_scale_factors = np.array([scalers[ch].sl_scale_factor for ch in channels])

    return transform(data, means, stds, sl_scale_factors, epsilons)


def to_display_units(normalized: np.ndarray, scaler) -> np.ndarray:
    """Undo standardization, leaving the signum-log values the scaler describes.

    This is deliberately only the first half of the inverse transform. The
    scaler's `min`/`max` -- which set the colour limits -- were fit in the
    signum-log domain, so undoing the log as well would put the image on a
    different scale than its own colour bar.

    Note this differs from infer.py's tensor_to_numpy, which inverts all the way
    to physical units and then re-applies log1p without the sl_scale_factor,
    landing above the scaler's max and saturating the bright half of the disk.
    """
    return normalized * (scaler.std + scaler.epsilon) + scaler.mean


def timestamp_from_file(filepath: str, attrs: dict) -> str:
    """Best-effort observation timestamp, as ``YYYY-MM-DDTHH:MM``.

    Prefers the file's own ``data_time`` attribute and falls back to the
    ``YYYYMMDD_HHMM`` stem these files are named with. Returns "" if neither
    parses, in which case titles simply omit the time.
    """
    candidates = [attrs.get("data_time", ""), os.path.basename(filepath)]
    for candidate in candidates:
        match = re.search(r"(\d{8})_(\d{4})", str(candidate))
        if match:
            return pd.to_datetime(
                f"{match.group(1)}{match.group(2)}", format="%Y%m%d%H%M"
            ).strftime("%Y-%m-%dT%H:%M")
    return ""


def predict(model, ts: np.ndarray, device, dtype, device_type: str) -> np.ndarray:
    """Run the model on one normalized cube and return an (H, W) probability map.

    Args:
        ts: Normalized data of shape (C, H, W).
    """
    batch = {
        # B, C, T, H, W -- a single input timestamp, so T = 1.
        "ts": torch.from_numpy(ts).unsqueeze(0).unsqueeze(2).to(device),
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


def save_channel_png(
    channel: str,
    image: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    scaler,
    save_path: str,
    threshold: float,
    title_txt: str,
    dpi: int,
):
    """Write a three-panel figure: channel image, AR probability, and overlay."""
    vmin, vmax = scaler.min, scaler.max
    if "hmi" in channel:
        vmin = -vmax  # HMI colormaps are diverging and must be centered on 0.

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=dpi)
    # y > 1 keeps the suptitle clear of the per-panel titles after tight_layout.
    fig.suptitle(title_txt, fontsize=14, fontweight="bold", y=1.02)

    im = axes[0].imshow(image, cmap=CHANNEL_CMAPS[channel], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Band {channel}")
    fig.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.046, pad=0.02)

    im = axes[1].imshow(probability, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("AR probability")
    fig.colorbar(im, ax=axes[1], orientation="horizontal", fraction=0.046, pad=0.02)

    axes[2].imshow(image, cmap=CHANNEL_CMAPS[channel], vmin=vmin, vmax=vmax)
    # Masked so only predicted-AR pixels are tinted; the rest stays transparent.
    axes[2].imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap=ListedColormap(["cyan"]),
        alpha=0.35,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(f"Overlay (p > {threshold})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {save_path}")


def save_channel_netcdf(
    channel: str,
    image: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    source_file: str,
    timestamp: str,
    threshold: float,
):
    """Write the channel image plus prediction arrays as a small NetCDF file."""
    ds = xr.Dataset(
        data_vars={
            channel: (("y", "x"), image.astype(np.float32)),
            "ar_probability": (("y", "x"), probability.astype(np.float32)),
            "ar_mask": (("y", "x"), mask.astype(np.uint8)),
        },
        attrs={
            "title": "Surya AR segmentation inference",
            "channel": channel,
            "source_file": source_file,
            "data_time": timestamp,
            "threshold": threshold,
            "channel_units": (
                "signum-log: sign(v)*log1p(|v*sl_scale_factor|), per scalers.yaml"
            ),
        },
    )
    ds.to_netcdf(save_path, engine="h5netcdf")
    ds.close()
    print(f"Saved {save_path}")


def run(
    config,
    input_file: str,
    checkpoint_path: str,
    output_dir: str,
    channels: list[str],
    device,
    device_type: str,
    threshold: float,
    save_netcdf: bool,
    dpi: int,
):
    all_channels = config["data"]["channels"]
    unknown = [ch for ch in channels if ch not in all_channels]
    assert not unknown, f"Channels {unknown} are not in the model's input channels."

    scalers = build_scalers(info=config["data"]["scalers"])
    model = load_model(config, checkpoint_path, device)

    print(f"Reading {input_file}")
    raw, attrs = read_hc_file(
        input_file, all_channels, s3_anon=config["data"].get("s3_anon", False)
    )
    timestamp = timestamp_from_file(input_file, attrs)
    ts = normalize(raw, all_channels, scalers)
    del raw

    print("Running inference.")
    probability = predict(model, ts, device, config["dtype"], device_type)
    mask = (probability > threshold).astype(np.uint8)
    print(
        f"Predicted AR coverage: {100.0 * mask.mean():.2f}% of the frame "
        f"at threshold {threshold}."
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_file))[0]

    for channel in channels:
        channel_idx = all_channels.index(channel)
        image = to_display_units(ts[channel_idx], scalers[channel])

        title = f"{stem} {channel}"
        if timestamp:
            title = f"{timestamp} | {channel}"

        save_channel_png(
            channel=channel,
            image=image,
            probability=probability,
            mask=mask,
            scaler=scalers[channel],
            save_path=os.path.join(output_dir, f"{stem}_{channel}.png"),
            threshold=threshold,
            title_txt=title,
            dpi=dpi,
        )

        if save_netcdf:
            save_channel_netcdf(
                channel=channel,
                image=image,
                probability=probability,
                mask=mask,
                save_path=os.path.join(output_dir, f"{stem}_{channel}.nc"),
                source_file=input_file,
                timestamp=timestamp,
                threshold=threshold,
            )

    print(f"Done. {len(channels)} channel(s) written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        "AR Segmentation inference on a single HelioFM NetCDF file"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        help="HelioFM NetCDF cube to run on. Local path or s3://bucket/key.",
    )
    parser.add_argument(
        "--config_path",
        default="./config_infer.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./assets/ar_segmentation_weights.pth",
        type=str,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./inference_results",
        type=str,
        help="Directory to write the per-channel output files to.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to emit files for. Defaults to every AIA channel in the config.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Probability above which a pixel is called active region.",
    )
    parser.add_argument(
        "--save_netcdf",
        action="store_true",
        help="Also write a per-channel NetCDF holding the image, probability and mask.",
    )
    parser.add_argument(
        "--dpi",
        default=100,
        type=int,
        help="Resolution of the saved figures.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    set_global_seed(42)

    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("Using CPU")

    channels = args.channels or [
        ch for ch in config["data"]["channels"] if ch.startswith("aia")
    ]

    run(
        config=config,
        input_file=args.input_file,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        channels=channels,
        device=device,
        device_type=device_type,
        threshold=args.threshold,
        save_netcdf=args.save_netcdf,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
