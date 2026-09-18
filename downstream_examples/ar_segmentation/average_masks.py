"""Average AR-segmentation masks pixel by pixel across inference runs, per channel.

Consumes the per-channel NetCDF files written by `infer_file.py --save_netcdf`
and, for each channel, averages the `ar_mask` variable across every run
directory given. The result is an agreement map in [0, 1]: 0 means no run called
that pixel active region, 1 means all of them did, and intermediate values mark
where the runs disagree.

Deliberately avoids importing torch/peft -- it only reads NetCDF and plots, so
it runs in seconds on a CPU-only box.

Example:
    python average_masks.py \
        --run_dirs ~/infer_run1 ~/infer_run2 ~/infer_run3 \
        --output_dir ~/infer_mask_mean
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as sunpy_cm
import xarray as xr
import yaml

# Same mapping infer_file.py uses, duplicated so this script stays importable
# without pulling in the model stack.
CHANNEL_CMAPS = {
    "aia94": "sdoaia94",
    "aia131": "sdoaia131",
    "aia171": "sdoaia171",
    "aia193": "sdoaia193",
    "aia211": "sdoaia211",
    "aia304": "sdoaia304",
    "aia335": "sdoaia335",
    "aia1600": "sdoaia1600",
    "hmi_m": "hmimag",
    "hmi_bx": "hmimag",
    "hmi_by": "hmimag",
    "hmi_bz": "hmimag",
}


def channel_cmap(channel: str):
    if channel == "hmi_v":
        return plt.get_cmap("bwr")
    return sunpy_cm.cmlist[CHANNEL_CMAPS[channel]]


def index_run(run_dir: str) -> dict[str, str]:
    """Map channel name -> NetCDF path for one run directory.

    Files are named `<stem>_<channel>.nc`, and the stem itself contains
    underscores, so the channel is taken from the file's own `channel`
    attribute rather than parsed out of the name.
    """
    found = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.nc"))):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            channel = ds.attrs.get("channel")
        if channel is None:
            print(f"  skipping {os.path.basename(path)}: no 'channel' attribute")
            continue
        found[channel] = path
    assert found, f"No per-channel NetCDF files found in {run_dir}"
    return found


def check_same_observation(paths: list[str]):
    """Refuse to average masks that came from different observations.

    Averaging across timestamps would silently produce a meaningless map, so
    this is an error rather than a warning.
    """
    times = {}
    for path in paths:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            times[path] = (ds.attrs.get("data_time"), ds.attrs.get("source_file"))
    distinct = set(times.values())
    if len(distinct) > 1:
        detail = "\n".join(f"    {p}: {t}" for p, t in times.items())
        raise ValueError(
            "Runs do not share an observation; refusing to average:\n" + detail
        )


def load_scaler_limits(scalers_path: str) -> dict:
    """Read per-channel display limits straight from scalers.yaml.

    Only `min`/`max` are needed, so the file is parsed as plain YAML instead of
    going through surya's build_scalers (which would drag in torch).
    """
    info = yaml.safe_load(open(scalers_path, "r"))
    return {ch: (float(v["min"]), float(v["max"])) for ch, v in info.items()}


def average_channel(paths: list[str], variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over runs for one channel, accumulated in float64.

    Accumulates rather than stacking so memory stays at a couple of 4096^2
    buffers regardless of how many runs are averaged.
    """
    total = None
    total_sq = None
    for path in paths:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert variable in ds, f"{path} has no variable '{variable}'"
            values = ds[variable].values.astype(np.float64)
        if total is None:
            total = np.zeros_like(values)
            total_sq = np.zeros_like(values)
        assert values.shape == total.shape, (
            f"{path}: shape {values.shape} does not match {total.shape}"
        )
        total += values
        total_sq += values * values

    n = len(paths)
    mean = total / n
    # Population variance; clipped because round-off can push it just below 0.
    variance = np.clip(total_sq / n - mean * mean, 0.0, None)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def save_png(
    channel: str,
    image: np.ndarray,
    mean: np.ndarray,
    n_runs: int,
    vlimits: tuple[float, float] | None,
    save_path: str,
    title_txt: str,
    dpi: int,
):
    """Two panels: the mean mask, and it overlaid on the channel image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    fig.suptitle(title_txt, fontsize=14, fontweight="bold", y=1.02)

    im = axes[0].imshow(mean, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"Mean mask over {n_runs} runs")
    cbar = fig.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.046, pad=0.02)
    # Ticks at the only values a mean of n binary masks can take.
    cbar.set_ticks([i / n_runs for i in range(n_runs + 1)])

    if vlimits is None:
        vmin, vmax = float(np.min(image)), float(np.max(image))
    else:
        vmin, vmax = vlimits
        if "hmi" in channel:
            vmin = -vmax  # HMI colormaps are diverging and centre on 0.
    axes[1].imshow(image, cmap=channel_cmap(channel), vmin=vmin, vmax=vmax)
    # Masked so full agreement is opaque, partial agreement fades, and pixels no
    # run selected stay fully transparent.
    axes[1].imshow(
        np.ma.masked_where(mean == 0, mean),
        cmap="cool",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"Agreement over {channel}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  saved {save_path}")


def save_netcdf(
    channel: str,
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    save_path: str,
    run_dirs: list[str],
    variable: str,
    attrs: dict,
):
    ds = xr.Dataset(
        data_vars={
            channel: (("y", "x"), image.astype(np.float32)),
            "mask_mean": (("y", "x"), mean),
            "mask_std": (("y", "x"), std),
        },
        attrs={
            "title": f"Pixelwise mean of '{variable}' across {len(run_dirs)} runs",
            "channel": channel,
            "n_runs": len(run_dirs),
            "runs": "; ".join(run_dirs),
            "averaged_variable": variable,
            "source_file": attrs.get("source_file", ""),
            "data_time": attrs.get("data_time", ""),
        },
    )
    ds.to_netcdf(save_path, engine="h5netcdf")
    ds.close()
    print(f"  saved {save_path}")


def main():
    parser = argparse.ArgumentParser(
        "Average AR segmentation masks across inference runs, per channel"
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Two or more directories of per-channel NetCDF files to average.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to write the per-channel averaged files to.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to average. Defaults to every channel present in all runs.",
    )
    parser.add_argument(
        "--variable",
        default="ar_mask",
        help="Variable to average. Use 'ar_probability' for a soft ensemble mean.",
    )
    parser.add_argument(
        "--scalers_path",
        default="./assets/scalers.yaml",
        type=str,
        help="Used only for figure colour limits. Ignored if missing.",
    )
    parser.add_argument("--no_png", action="store_true", help="Skip the figures.")
    parser.add_argument("--no_netcdf", action="store_true", help="Skip the NetCDF files.")
    parser.add_argument("--dpi", default=100, type=int, help="Figure resolution.")
    args = parser.parse_args()

    run_dirs = [os.path.expanduser(d) for d in args.run_dirs]
    assert len(run_dirs) >= 2, "Averaging needs at least two run directories."

    indexes = []
    for run_dir in run_dirs:
        print(f"Indexing {run_dir}")
        indexes.append(index_run(run_dir))

    common = set(indexes[0])
    for idx in indexes[1:]:
        common &= set(idx)
    assert common, "The run directories share no channels."

    for run_dir, idx in zip(run_dirs, indexes):
        missing = sorted(set(idx) - common)
        if missing:
            print(f"Note: {run_dir} has extra channels not in every run: {missing}")

    channels = args.channels or sorted(common)
    unknown = [ch for ch in channels if ch not in common]
    assert not unknown, f"Channels {unknown} are not present in every run."

    vlimits = {}
    if os.path.exists(args.scalers_path):
        vlimits = load_scaler_limits(args.scalers_path)
    else:
        print(f"Note: {args.scalers_path} not found; scaling figures to their own range.")

    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
    output_dir = os.path.expanduser(args.output_dir)

    for channel in channels:
        paths = [idx[channel] for idx in indexes]
        check_same_observation(paths)
        print(f"{channel}: averaging {len(paths)} runs")

        mean, std = average_channel(paths, args.variable)

        with xr.open_dataset(paths[0], engine="h5netcdf") as ds:
            image = ds[channel].values
            attrs = dict(ds.attrs)

        stem = os.path.splitext(os.path.basename(paths[0]))[0]
        stem = re.sub(rf"_{re.escape(channel)}$", "", stem)

        # The unanimous/contested split only means something for binary input;
        # for ar_probability every pixel is "partial" and the numbers mislead.
        is_binary = np.isin(np.unique(mean), np.arange(len(paths) + 1) / len(paths)).all()
        if is_binary:
            unanimous = float(np.mean(mean == 1.0)) * 100
            contested = float(np.mean((mean > 0) & (mean < 1.0))) * 100
            print(
                f"  all runs agree AR: {unanimous:.4f}% of frame | "
                f"partial agreement: {contested:.4f}% | mean {mean.mean():.6f}"
            )
        else:
            print(
                f"  mean {mean.mean():.6f} | max {mean.max():.4f} | "
                f"mean spread across runs {std.mean():.6f}"
            )

        title = f"{attrs.get('data_time', stem)} | {channel} | mean {args.variable}"

        if not args.no_png:
            save_png(
                channel=channel,
                image=image,
                mean=mean,
                n_runs=len(paths),
                vlimits=vlimits.get(channel),
                save_path=os.path.join(output_dir, f"{stem}_{channel}_mask_mean.png"),
                title_txt=title,
                dpi=args.dpi,
            )

        if not args.no_netcdf:
            save_netcdf(
                channel=channel,
                image=image,
                mean=mean,
                std=std,
                save_path=os.path.join(output_dir, f"{stem}_{channel}_mask_mean.nc"),
                run_dirs=run_dirs,
                variable=args.variable,
                attrs=attrs,
            )

    print(f"Done. {len(channels)} channel(s) written to {output_dir}")


if __name__ == "__main__":
    main()
