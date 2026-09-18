"""Render the averaged AR masks as false-colour JPEG images, one per channel.

Consumes the `*_mask_mean.nc` files written by `average_masks.py` and colours
each pixel by its `mask_mean` value -- the fraction of inference runs that called
that pixel active region. Output is a true 1:1 raster: one input pixel, one
output pixel, no axes, no margins, no resampling.

Colour choice follows the sequential rule, because agreement level is an ordered
magnitude ("how many runs agreed"), not an identity. The default is therefore a
single-hue ramp, light meaning high agreement against the black background. The
ramp steps and the optional categorical palette were both checked with the
data-viz palette validator against a dark surface:

    sequential  #1c5cab,#3987e5,#b7d3f6  --mode dark --ordinal    -> all pass
    categorical #3987e5,#d95926,#199e70  --mode dark --pairs all  -> all pass

Example:
    python false_color.py --input ~/infer_mask_mean --output_dir ~/infer_false_color
"""

import argparse
import glob
import os

import numpy as np
import xarray as xr
import yaml
from PIL import Image

# Blue sequential ramp, steps 100-600. Step 600 is the dark-surface limit: any
# darker and the low end stops clearing 2:1 against the background.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
]
# Indices into BLUE_RAMP for the validated 3-level ordinal palette, so the
# common 3-run case reproduces exactly what the validator was run on.
VALIDATED_3 = [9, 6, 1]  # #1c5cab, #3987e5, #b7d3f6

# Categorical slots 1-3, dark steps. Only for when levels must be told apart at a
# glance rather than read as an ordered magnitude; capped at 3 because the
# fourth slot fails the all-pairs floors.
CATEGORICAL = ["#3987e5", "#d95926", "#199e70"]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def ordinal_colors(n_levels: int) -> list[str]:
    """Pick `n_levels` steps off the blue ramp, darkest first.

    Any evenly spaced selection from a single-hue ramp keeps the properties the
    ordinal check tests -- one hue, monotone lightness, and a low end no darker
    than step 600 -- so this stays valid as the run count changes.
    """
    assert 1 <= n_levels <= len(BLUE_RAMP), f"Cannot build {n_levels} ordinal steps."
    if n_levels == 3:
        return [BLUE_RAMP[i] for i in VALIDATED_3]
    idx = np.linspace(len(BLUE_RAMP) - 1, 0, n_levels).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def build_lut(levels: np.ndarray, mode: str) -> dict:
    """Map each non-zero agreement level to an RGB triple."""
    non_zero = [lv for lv in levels if lv > 0]
    if mode == "categorical":
        assert len(non_zero) <= len(CATEGORICAL), (
            f"categorical mode handles at most {len(CATEGORICAL)} levels, "
            f"got {len(non_zero)}. Use --mode sequential."
        )
        colors = CATEGORICAL[: len(non_zero)]
    else:
        colors = ordinal_colors(len(non_zero))
    return {lv: hex_to_rgb(c) for lv, c in zip(non_zero, colors)}


def render_discrete(values: np.ndarray, lut: dict, base: np.ndarray) -> np.ndarray:
    """Paint each agreement level onto `base` (an H,W,3 uint8 backdrop)."""
    out = base.copy()
    for level, rgb in lut.items():
        out[np.isclose(values, level)] = rgb
    return out


def render_continuous(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Interpolate along the blue ramp for non-discrete data (e.g. mean probability).

    Interpolation happens in RGB along a ramp that is already monotone in
    lightness, so intermediate colours stay on the same single-hue scale.
    """
    ramp = np.array([hex_to_rgb(c) for c in reversed(BLUE_RAMP)], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, len(ramp))
    scaled = np.clip(values, 0.0, 1.0)

    out = base.astype(np.float64)
    painted = scaled > 0
    for channel in range(3):
        interpolated = np.interp(scaled[painted], positions, ramp[:, channel])
        out[..., channel][painted] = interpolated
    return out.round().astype(np.uint8)


def grayscale_backdrop(image: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    """Render the channel image as a dim grey backdrop, so ARs sit in context.

    Held to 55% brightness so the false colour stays clearly separable from the
    solar disc underneath it.
    """
    if limits is None:
        vmin, vmax = float(image.min()), float(image.max())
    else:
        vmin, vmax = limits
    span = vmax - vmin if vmax > vmin else 1.0
    norm = np.clip((image - vmin) / span, 0.0, 1.0)
    grey = (norm * 255 * 0.55).round().astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def load_scaler_limits(scalers_path: str) -> dict:
    info = yaml.safe_load(open(scalers_path, "r"))
    return {ch: (float(v["min"]), float(v["max"])) for ch, v in info.items()}


def collect_inputs(paths: list[str]) -> list[str]:
    files = []
    for entry in paths:
        entry = os.path.expanduser(entry)
        if os.path.isdir(entry):
            files.extend(sorted(glob.glob(os.path.join(entry, "*_mask_mean.nc"))))
        else:
            files.append(entry)
    assert files, f"No *_mask_mean.nc files found in {paths}"
    return files


def main():
    parser = argparse.ArgumentParser(
        "False-colour JPEG of averaged AR segmentation masks"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Directories of *_mask_mean.nc files, or individual .nc files.",
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Where to write the images."
    )
    parser.add_argument(
        "--variable",
        default="mask_mean",
        help="Variable to colour. 'mask_std' also works.",
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "categorical"],
        help="sequential = one hue by agreement level (default, correct for "
        "ordered data). categorical = distinct hues, max 3 levels.",
    )
    parser.add_argument(
        "--background",
        default="black",
        choices=["black", "channel"],
        help="black = mask only. channel = dim greyscale solar image behind it.",
    )
    parser.add_argument(
        "--scalers_path",
        default="./assets/scalers.yaml",
        type=str,
        help="Only used to scale the --background channel image.",
    )
    parser.add_argument(
        "--format",
        default="jpeg",
        choices=["jpeg", "png"],
        help="JPEG is lossy and rings at the hard colour edges; png is exact.",
    )
    parser.add_argument("--quality", default=95, type=int, help="JPEG quality (1-100).")
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Burn a colour key into the top-left corner. Off by default so the "
        "output stays a faithful 1:1 raster.",
    )
    args = parser.parse_args()

    files = collect_inputs(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    limits = {}
    if args.background == "channel":
        scalers_path = os.path.expanduser(args.scalers_path)
        if os.path.exists(scalers_path):
            limits = load_scaler_limits(scalers_path)
        else:
            print(f"Note: {scalers_path} not found; scaling backdrop to its own range.")

    ext = "jpg" if args.format == "jpeg" else "png"

    for path in files:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert args.variable in ds, f"{path} has no variable '{args.variable}'"
            values = ds[args.variable].values.astype(np.float64)
            channel = ds.attrs.get("channel", "")
            n_runs = ds.attrs.get("n_runs", None)
            image = ds[channel].values if channel in ds else None

        levels = np.unique(values)
        discrete = len(levels) <= len(BLUE_RAMP) + 1
        assert discrete or args.mode != "categorical", (
            f"{os.path.basename(path)}: '{args.variable}' takes {len(levels)} "
            "distinct values, which is continuous data -- categorical colours "
            "cannot encode it. Use --mode sequential."
        )

        if args.background == "channel" and image is not None:
            base = grayscale_backdrop(image, limits.get(channel))
        else:
            base = np.zeros(values.shape + (3,), dtype=np.uint8)

        print(f"{os.path.basename(path)}  [{channel}]")
        lut = None
        if discrete:
            lut = build_lut(levels, args.mode)
            rgb = render_discrete(values, lut, base)
            for level, colour in sorted(lut.items()):
                label = f"{level:.3f}"
                if n_runs:
                    label += f"  ({round(level * n_runs)} of {n_runs} runs)"
                count = int(np.isclose(values, level).sum())
                print(
                    f"    {label:<24} #{'%02x%02x%02x' % colour}"
                    f"   {count:>9d} px  {100 * count / values.size:.4f}%"
                )
        else:
            rgb = render_continuous(values, base)
            print(
                f"    continuous ramp {BLUE_RAMP[-1]} -> {BLUE_RAMP[0]} over "
                f"[0, 1]; data max {values.max():.4f}"
            )

        if args.legend:
            rgb = draw_legend(rgb, lut, n_runs)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{stem}_false_color.{ext}")
        img = Image.fromarray(rgb, mode="RGB")
        if args.format == "jpeg":
            # subsampling=0 keeps 4:4:4 chroma; the default 4:2:0 would smear
            # these saturated edges badly.
            img.save(out_path, "JPEG", quality=args.quality, subsampling=0)
        else:
            img.save(out_path, "PNG")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    wrote {out_path}  ({rgb.shape[1]}x{rgb.shape[0]}, {size_mb:.1f} MB)")

    print(f"Done. {len(files)} image(s) written to {output_dir}")


def draw_legend(rgb: np.ndarray, lut: dict | None, n_runs) -> np.ndarray:
    """Burn a small swatch column into the top-left corner.

    Only used with --legend; it overwrites image pixels, which is why the
    default is to leave the raster untouched.
    """
    if not lut:
        return rgb
    out = rgb.copy()
    pad, swatch, gap = 40, 80, 24
    for i, (_, colour) in enumerate(sorted(lut.items())):
        top = pad + i * (swatch + gap)
        out[top : top + swatch, pad : pad + swatch] = colour
    return out


if __name__ == "__main__":
    main()
