"""Measure the channel-to-channel disk offset in core-sdo cubes, and check the fix.

For each cube, every channel is registered against a reference channel (hmi_bz by
default, which lines up with the AR masks) by comparing disk centers from a robust fit to
each channel's limb. The script prints, per channel:

  measured   offset of the channel's content from the reference, (dy, dx) px; positive
             means toward larger array indices
  predicted  (976 / RSUN_OBS - 1) * 2048 px from the channel's own metadata, for the
             corner-scaled channels (0 for the others)
  residual   offset left after undo_corner_scaling with the chosen --sign

The disk centers come from a limb fit. HMI channels fit to ~0.1 px, and aia94, 131,
304, 335 and 1600 to a few px. The coronal channels (171, 193, 211) can be off by
10-30 px when bright off-limb loops pull the limb; the "limb rms" and "kept" columns
show how much to trust each row, and hmi_m against hmi_bz is the cleanest single test.

Go/no-go for data.fix_corner_scaling: on a summer cube the 8 corner-scaled channels
should measure close to the prediction (~70 px in early July) on both axes, the other 5
close to 0, and every residual close to 0. If the residuals come out at about twice the
prediction, the shift runs the other way: rerun with --sign -1 and, if that fixes it,
set data.corner_scaling_sign: -1.

    python check_disk_alignment.py ../../data/Surya-1.0_validation_data/20140107_1500.nc
    python check_disk_alignment.py s3://nasa-surya-bench/2011/07/20110701_0000.nc --s3_anon

See KNOWN_ISSUE_channel_offset.md.
"""

import argparse

import hdf5plugin  # noqa: F401  (registers the LZ4 filter the cubes are compressed with)
import numpy as np
import scipy.ndimage
import skimage.measure
import xarray as xr

from surya.datasets.disk_alignment import (
    CORNER_SCALED_CHANNELS,
    corner_shift_px,
    rsun_obs_from_attrs,
    undo_corner_scaling,
)

ALL_CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335",
    "aia1600", "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]


def disk_center(img: np.ndarray, is_hmi: bool, bin_factor: int = 4,
                expected_radius_px: float = 976.0 / 0.6, n_angles: int = 360):
    """Center (y, x) of the solar disk in full-resolution pixels, from a limb fit.

    Along ``n_angles`` rays from the image center, the limb is taken as the steepest
    outward drop of the smoothed image within 10% of the expected radius
    (the cubes fix it at 976 arcsec; the largest diagonal offset moves the limb ~6%): the edge of
    the on-disk support for HMI, of the signum-log intensity for AIA. A circle is fit
    to those points by least squares, twice, dropping points more than 3 MADs off the
    first fit (prominences, off-limb loops, flares).
    """
    x = np.nan_to_num(img.astype(np.float64))
    if is_hmi:
        x = (x != 0).astype(np.float64)
    else:
        x = np.sign(x) * np.log1p(np.abs(x))
    x = skimage.measure.block_reduce(x, (bin_factor, bin_factor), np.mean)
    x = scipy.ndimage.gaussian_filter(x, 1.5)

    n = x.shape[0]
    cy, cx = n / 2.0 - 0.5, n / 2.0 - 0.5  # start the rays at the image center
    r0 = expected_radius_px / bin_factor
    radii = np.linspace(0.9 * r0, 1.1 * r0, 200)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    # Re-centre the rays on each fit so a shifted disk is still sampled radially.
    for _ in range(3):
        yy = cy + np.outer(np.sin(theta), radii)
        xx = cx + np.outer(np.cos(theta), radii)
        profiles = scipy.ndimage.map_coordinates(x, [yy, xx], order=1, mode="nearest")
        limb_r = radii[np.argmin(np.gradient(profiles, axis=1), axis=1)]
        py, px = cy + limb_r * np.sin(theta), cx + limb_r * np.cos(theta)

        keep = np.ones_like(py, dtype=bool)
        for _ in range(2):
            cy, cx, r = _fit_circle(py[keep], px[keep])
            resid = np.hypot(py - cy, px - cx) - r
            # Floor the tolerance at 1 binned px: an HMI footprint is a near-perfect
            # circle, and a zero MAD would otherwise reject every point.
            mad = np.median(np.abs(resid - np.median(resid)))
            keep = np.abs(resid) < max(3 * 1.4826 * mad, 1.0)
    rms = float(np.sqrt(np.mean(resid[keep] ** 2))) * bin_factor
    # back to full-resolution pixel coordinates (pixel centers at i + 0.5)
    return (cy + 0.5) * bin_factor, (cx + 0.5) * bin_factor, rms, float(keep.mean())


def _fit_circle(y: np.ndarray, x: np.ndarray):
    """Algebraic least-squares circle fit; returns (cy, cx, r)."""
    a = np.column_stack([2 * y, 2 * x, np.ones_like(y)])
    (cy, cx, k), *_ = np.linalg.lstsq(a, y**2 + x**2, rcond=None)
    return cy, cx, np.sqrt(k + cy**2 + cx**2)


def offsets(cube: np.ndarray, channels, ref: str, bin_factor: int):
    """Disk-center offset of each channel relative to ``ref`` (full-resolution px),
    and each channel's limb-fit quality (RMS residual in px, fraction of rays kept)."""
    fits = {
        ch: disk_center(cube[i], ch.startswith("hmi"), bin_factor)
        for i, ch in enumerate(channels)
    }
    ref_c = np.array(fits[ref][:2])
    return (
        {ch: np.array(f[:2]) - ref_c for ch, f in fits.items()},
        {ch: f[2:] for ch, f in fits.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cubes", nargs="+", help="Local .nc paths or s3:// URIs.")
    parser.add_argument("--reference", default="hmi_bz", help="Channel whose disk center is the zero point.")
    parser.add_argument("--sign", type=int, default=1, choices=[1, -1],
                        help="Direction passed to undo_corner_scaling for the residual.")
    parser.add_argument("--bin", type=int, default=4, dest="bin_factor",
                        help="Block-average factor before the limb fit (default 4).")
    parser.add_argument("--s3_anon", action="store_true", help="Anonymous S3 access.")
    args = parser.parse_args()

    for path in args.cubes:
        raw, attrs, rsun = load_cube(path, args.s3_anon)
        fixed = undo_corner_scaling(raw, ALL_CHANNELS, rsun, sign=args.sign)
        before, quality = offsets(raw, ALL_CHANNELS, args.reference, args.bin_factor)
        after, _ = offsets(fixed, ALL_CHANNELS, args.reference, args.bin_factor)

        print(f"\n{path}  ({attrs.get('data_time', '')}, reference {args.reference}, "
              f"sign {args.sign:+d})")
        print(f"{'channel':9s} {'RSUN_OBS':>9s} {'measured dy,dx':>16s} "
              f"{'predicted':>9s} {'residual dy,dx':>16s} {'limb rms':>8s} {'kept':>5s}")
        for ch in ALL_CHANNELS:
            pred = corner_shift_px(rsun[ch]) if ch in CORNER_SCALED_CHANNELS else 0
            b, a = before[ch], after[ch]
            rms, kept = quality[ch]
            flag = "  <- unreliable limb fit" if rms > 25 or kept < 0.6 else ""
            print(f"{ch:9s} {rsun[ch]:9.2f} {b[0]:7.1f},{b[1]:7.1f}  {pred:9d} "
                  f"{a[0]:7.1f},{a[1]:7.1f} {rms:8.1f} {kept:5.2f}{flag}")


def load_cube(path: str, s3_anon: bool):
    """Raw (C, H, W) cube, global attrs and per-channel RSUN_OBS, staging S3 once."""
    import os
    from uuid import uuid4

    from surya.utils.s3 import default_scratch_dir, is_s3_path, make_s3_client, parse_s3_uri

    if not is_s3_path(path):
        return _read_local(path)
    scratch = default_scratch_dir()
    os.makedirs(scratch, exist_ok=True)
    local = os.path.join(scratch, f"{uuid4().hex}.nc")
    bucket, key = parse_s3_uri(path)
    print(f"Downloading {path}")
    make_s3_client(anon=s3_anon).download_file(bucket, key, local)
    try:
        return _read_local(local)
    finally:
        os.remove(local)


def _read_local(path: str):
    with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
        raw = ds[ALL_CHANNELS].to_array().load().to_numpy()
        return raw, dict(ds.attrs), rsun_obs_from_attrs(ds, ALL_CHANNELS)


if __name__ == "__main__":
    main()
