"""Undo the corner-anchored solar-disk rescaling in the released Surya core-sdo cubes.

The core-sdo pipeline rescales every image so the solar disk has a fixed radius of
976 arcsec (scale factor ``s = 976 / RSUN_OBS``). In the released cubes, 8 of the 13
channels appear to have been rescaled about array index (0, 0) instead of the image
center. That leaves their disk correctly sized but translated along the diagonal by
``(s - 1) * 2048`` px per axis: about 0 px in early January (``s`` ~ 1 at perihelion)
and about 70 px in early July. The other 5 channels and the AR masks are centered.

If the shifted image is ``I_bad(y) = I0(y / s)``, the centered image is
``I_good(x) = I_bad(x + d)`` with ``d = (s - 1) * c`` on both axes, so a pure
translation fixes it. The original warp used nearest-neighbour interpolation, so an
integer shift is as precise as the data allow.

See downstream_examples/ar_segmentation/KNOWN_ISSUE_channel_offset.md.
"""

import json
from collections.abc import Mapping, Sequence

import numpy as np

# Channels observed to be offset from aia1600, hmi_bx, hmi_by, hmi_bz, hmi_v and the AR masks.
CORNER_SCALED_CHANNELS = (
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "hmi_m",
)

TARGET_RSUN_ARCSEC = 976.0
IMAGE_CENTER_PX = 2048.0


def rsun_obs_from_attrs(ds, channels: Sequence[str]) -> dict[str, float]:
    """Read each channel's RSUN_OBS (arcsec) from its ``meta_1`` JSON attribute.

    Args:
        ds: An open xarray Dataset (or any mapping of name -> object with ``.attrs``).
        channels: Channel names to read.
    Returns:
        ``{channel: RSUN_OBS}``.
    """
    out = {}
    for ch in channels:
        attrs = ds[ch].attrs
        if "meta_1" not in attrs:
            raise KeyError(
                f"Channel {ch!r} has no 'meta_1' attribute, so its RSUN_OBS is unknown and "
                "the corner-scaling correction cannot be computed. Disable "
                "data.fix_corner_scaling or use cubes that carry FITS metadata."
            )
        meta = attrs["meta_1"]
        if isinstance(meta, (bytes, np.bytes_)):
            meta = meta.decode()
        meta = json.loads(meta) if isinstance(meta, str) else dict(meta)
        value = meta.get("RSUN_OBS", meta.get("rsun_obs"))
        if value is None:
            raise KeyError(f"Channel {ch!r} meta_1 has no RSUN_OBS keyword.")
        out[ch] = float(value)
    return out


def corner_shift_px(
    rsun_obs: float,
    target: float = TARGET_RSUN_ARCSEC,
    center: float = IMAGE_CENTER_PX,
) -> int:
    """Per-axis shift ``(s - 1) * center``, rounded to whole pixels."""
    return int(round((target / rsun_obs - 1.0) * center))


def _translate(img: np.ndarray, d: int) -> np.ndarray:
    """Return ``out[i, j] = img[i + d, j + d]``, zero-filled where that falls outside."""
    out = np.zeros_like(img)
    h, w = img.shape
    if abs(d) >= h or abs(d) >= w:
        return out
    if d >= 0:
        out[: h - d, : w - d] = img[d:, d:]
    else:
        out[-d:, -d:] = img[: h + d, : w + d]
    return out


def undo_corner_scaling(
    data: np.ndarray,
    channels: Sequence[str],
    rsun_obs: Mapping[str, float],
    affected: Sequence[str] | None = None,
    sign: int = 1,
    center: float | None = None,
) -> np.ndarray:
    """Translate the corner-scaled channels of a (C, H, W) cube back onto the image center.

    Args:
        data: Raw cube, channels in ``channels`` order, before scaling or pooling.
        channels: Channel name for each index of ``data``'s first axis.
        rsun_obs: ``{channel: RSUN_OBS}`` for at least the affected channels present.
        affected: Channels to correct; defaults to ``CORNER_SCALED_CHANNELS``.
        sign: +1 applies ``out(x) = in(x + d)``; -1 reverses the direction. Only for use if
            a measurement (check_disk_alignment.py) shows the shift runs the other way.
        center: Pixel about which the correct scaling should have been done. Defaults to
            half the image height (2048 for 4096 cubes).
    Returns:
        A new array; unaffected channels are copied unchanged.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if data.ndim != 3 or data.shape[0] != len(channels):
        raise ValueError(f"data shape {data.shape} does not match {len(channels)} channels")
    affected = set(CORNER_SCALED_CHANNELS if affected is None else affected)
    if center is None:
        center = data.shape[1] / 2.0
    out = data.copy()
    for i, ch in enumerate(channels):
        if ch not in affected:
            continue
        d = sign * corner_shift_px(rsun_obs[ch], center=center)
        if d:
            out[i] = _translate(data[i], d)
    return out


def corner_scaling_kwargs(data_config: Mapping) -> dict:
    """The correction's dataset kwargs from a config's ``data:`` block.

    Read with ``.get`` so configs written before this option keep their old behaviour.
    """
    return dict(
        fix_corner_scaling=bool(data_config.get("fix_corner_scaling", False)),
        corner_scaling_channels=data_config.get("corner_scaling_channels"),
        corner_scaling_sign=int(data_config.get("corner_scaling_sign", 1)),
    )
