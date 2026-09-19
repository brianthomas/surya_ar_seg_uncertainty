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

Smoothing and contouring (--contour)
------------------------------------
The mask average cannot be contoured as it stands. It is the mean of binary
masks, so it only takes the values k/n -- with three runs, {0, 1/3, 2/3, 1} --
and its edges follow Surya's 16 px patch grid rather than anything on the Sun.
A contour drawn straight onto it traces that staircase plus every isolated
speckle, which is noise, not a region boundary.

So --contour smooths first, to the model's own resolution limit, and contours
the smoothed field. Two consequences worth knowing:

  * The contour is drawn on the *smoothed* field but painted over the *unsmoothed*
    fills, so it will not follow the colour boundaries underneath it exactly.
    That is intended -- the fills show the raw per-pixel vote, the contour shows
    the resolved region.
  * Smoothing shrinks the enclosed area slightly, because it trims the isolated
    single-run speckles that a hard threshold would otherwise keep.

Measured on the 3-run aia94 average, level 0.5:

    --smooth_px 0.5  (effectively raw)   27339 px of contour, 0.1234% enclosed
    --smooth_px 16   (default, 1 patch)   3016 px of contour, 0.1058% enclosed

Nine times the contour pixels for no extra information -- that ratio is the
argument for smoothing.

Example:
    python false_color.py --input ~/infer_mask_mean --output_dir ~/infer_false_color
    python false_color.py --input ~/infer_mask_mean/..._aia94_mask_mean.nc \
        --output_dir ~/out --contour --contour_levels 0.5
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


# Surya's resolution limit, and the default smoothing scale.
#
# The backbone tokenises the 4096 px frame into 16 px patches (config_infer.yaml:
# img_size 4096, patch_size 16) and the segmentation head's decoder emits one
# value per patch, which is then expanded back to 16x16 pixels. So the model has
# 256x256 independent output cells, not 4096x4096: structure finer than 16 px is
# an artefact of that expansion, not something the model resolved.
#
# This is visible in the output rather than merely implied by the config -- the
# disagreement regions in the mask average are rectangular blocks aligned to the
# 16 px grid, with unanimous pixels speckled inside them.
#
# Change this only if the model config changes. It is a property of Surya, not a
# display preference; --smooth_px overrides it per run for exploration.
SURYA_PATCH_PX = 16

# Contour accent. Validated against every fill it is drawn over, dark surface:
#   #1c5cab / #3987e5 / #b7d3f6 vs #d95926 -> CVD dE 22.9-28.0, normal 31.8-33.1
CONTOUR_COLOR = "#d95926"


def smooth_to_model_resolution(values: np.ndarray, fwhm_px: float) -> np.ndarray:
    """Blur the mask average down to the model's own resolution limit.

    Args:
        values: Mask average, (H, W), each pixel in [0, 1].
        fwhm_px: Full width at half maximum of the Gaussian, in pixels. Pass
            SURYA_PATCH_PX to make one resolution element equal one patch.

    Returns:
        Smoothed field, (H, W), float. Continuous -- the k/n quantisation of the
        input is gone, which is what makes a threshold crossing well defined.

    Why FWHM and not sigma directly: FWHM is the width at which the kernel has
    fallen to half its peak, so it is the honest "one resolution element"
    figure and is directly comparable to the patch size. Quoting sigma instead
    would understate the blur by a factor of ~2.35. The conversion is the
    standard one, from exp(-x^2 / 2*sigma^2) = 1/2 at x = FWHM/2:

        FWHM = 2 * sqrt(2 * ln 2) * sigma ~= 2.3548 * sigma

    so the default 16 px FWHM is sigma ~= 6.79 px.

    Why a Gaussian rather than a 16x16 box filter matched to the patch: a box
    filter is separable and exact but leaves its own square footprint in the
    result, reintroducing axis-aligned artefacts of the same kind being removed.
    The Gaussian is isotropic, so the smoothed field has no preferred direction
    and the contour is free to follow the region.

    mode="nearest" extends the edge pixel outward rather than treating outside
    the frame as zero. Zero-padding would pull the field down near the border
    and bend a contour inward there. Moot for a disc centred in the frame, but
    wrong for a region touching the edge.
    """
    from scipy.ndimage import gaussian_filter

    # 2*sqrt(2*ln2) ~= 2.3548; see the conversion in the docstring above.
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return gaussian_filter(values, sigma=sigma, mode="nearest")


def contour_mask(field: np.ndarray, level: float, width: int) -> np.ndarray:
    """Boundary pixels of `field >= level`, thickened to roughly `width` px.

    Args:
        field: Smoothed mask average, (H, W).
        level: Threshold in (0, 1]. 0.5 means "at least half the runs agreed".
        width: Target line thickness in pixels.

    Returns:
        Boolean mask, (H, W), true on the contour line.

    The contour is extracted morphologically rather than with matplotlib's
    contour(): this writes into a raster at 1:1, so what is needed is the set of
    *pixels* on the boundary, not a set of sub-pixel polygon vertices that would
    then have to be rasterised back. Thresholding and taking the morphological
    boundary gives that directly, is exact at pixel resolution, and keeps the
    output a pure array operation with no figure machinery involved.

    The boundary is the region minus its own erosion, i.e. the ring of pixels
    inside the region that touch its outside. So the line sits *inside* the
    thresholded area -- the enclosed pixel count reported by the caller is the
    threshold count, not the count inside the drawn line, and the two differ by
    the width of the line.

    border_value=0 makes erosion treat outside the frame as background, so a
    region running off the edge is closed along that edge instead of being left
    open. Without it the contour would silently break wherever a region is
    clipped by the frame.

    Thickening is `width // 2` dilations, and each dilation grows the line by one
    pixel on *each* side, so the drawn thickness is 1 + 2 * (width // 2):

        width 1 -> 0 dilations -> 1 px
        width 3 -> 1 dilation  -> 3 px
        width 4 -> 2 dilations -> 5 px
        width 5 -> 2 dilations -> 5 px

    A symmetric line around a single-pixel boundary can only be an odd number of
    pixels wide, so even widths round *up* to the next odd value -- width 4 and
    width 5 both give 5 px. Pass odd widths to get exactly what you asked for.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    binary = field >= level
    if not binary.any():
        # No crossing at this level; hand back an all-false mask of the right
        # shape so the caller can paint it without a special case.
        return binary
    edge = binary & ~binary_erosion(binary, border_value=0)
    if width > 1:
        edge = binary_dilation(edge, iterations=int(width) // 2)
    return edge


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
        "--contour",
        action="store_true",
        help="Overlay a contour of the mask average after smoothing it to the "
        "model's resolution limit.",
    )
    parser.add_argument(
        "--contour_levels",
        nargs="+",
        type=float,
        default=[0.5],
        help="Levels to contour, in [0, 1]. Default 0.5 (half the runs agree).",
    )
    parser.add_argument(
        "--smooth_px",
        type=float,
        default=float(SURYA_PATCH_PX),
        help=f"FWHM of the smoothing kernel in pixels. Default {SURYA_PATCH_PX} "
        "= Surya's patch size, the finest structure the decoder can represent.",
    )
    parser.add_argument(
        "--contour_width", type=int, default=3, help="Contour thickness in pixels."
    )
    parser.add_argument(
        "--contour_color", default=CONTOUR_COLOR, help="Contour colour as hex."
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Burn a colour key into the top-left corner. Off by default so the "
        "output stays a faithful 1:1 raster.",
    )
    args = parser.parse_args()

    bad = [lv for lv in args.contour_levels if not 0.0 < lv <= 1.0]
    assert not bad, f"--contour_levels must lie in (0, 1]; got {bad}"
    assert args.smooth_px > 0, "--smooth_px must be positive."

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

        if args.contour:
            # Smooth once and reuse for every level -- the smoothing does not
            # depend on the threshold, only the crossing does.
            smoothed = smooth_to_model_resolution(values, args.smooth_px)
            colour = hex_to_rgb(args.contour_color)
            print(
                f"    contour: smoothed to FWHM {args.smooth_px:g} px "
                f"(sigma {args.smooth_px / 2.3548:.2f}), colour {args.contour_color}"
            )
            # Ascending, so that with several levels the tighter, higher-level
            # contours are painted last and stay visible where they abut a
            # lower one. Note every level uses the same colour, so more than one
            # level is only legible when the contours are well separated.
            for level in sorted(args.contour_levels):
                edge = contour_mask(smoothed, level, args.contour_width)
                # Area of the thresholded region, not of the drawn line: the
                # line sits inside this area (see contour_mask).
                enclosed = int((smoothed >= level).sum())
                rgb[edge] = colour
                print(
                    f"      level {level:.3f}: {enclosed:>9d} px enclosed "
                    f"({100 * enclosed / values.size:.4f}%), "
                    f"{int(edge.sum()):>8d} px of contour"
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
