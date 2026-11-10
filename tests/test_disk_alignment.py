"""Tests for the corner-scaling correction (surya/datasets/disk_alignment.py).

Standalone: needs only numpy for the core tests; the loader test also needs the
package's xarray/h5netcdf/torch stack and is skipped without it.
"""

import json
import logging

import numpy as np
import pytest

from surya.datasets.disk_alignment import (
    CORNER_SCALED_CHANNELS,
    corner_scaling_kwargs,
    corner_shift_px,
    rsun_obs_from_attrs,
    undo_corner_scaling,
)

N = 512  # small stand-in for 4096; the correction scales with the image center
C = N / 2.0
RSUN_JULY = 943.86  # arcsec, early July
RSUN_JAN = 975.95  # arcsec, early January


def _disk(center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.float32)


def _corner_scaled_disk(rsun_obs: float) -> np.ndarray:
    """A centered disk of the right final size, then scaled about (0, 0) as in the bug."""
    s = 976.0 / rsun_obs
    radius = 0.4 * N
    # Scaling about the origin maps the true center C to s*C.
    return _disk(s * C, s * C, radius)


def _centroid(img: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    w = img.sum()
    return (yy * img).sum() / w, (xx * img).sum() / w


class _Var:
    def __init__(self, attrs):
        self.attrs = attrs


def test_shift_magnitude_matches_prediction_at_4096():
    assert corner_shift_px(RSUN_JAN) == 0
    assert corner_shift_px(RSUN_JULY) == 70


def test_undo_recenters_affected_and_leaves_others_alone():
    channels = ["aia171", "hmi_bz"]
    bad = _corner_scaled_disk(RSUN_JULY)
    good = _disk(C, C, 0.4 * N)
    cube = np.stack([bad, good])
    rsun = {"aia171": RSUN_JULY, "hmi_bz": RSUN_JULY}

    out = undo_corner_scaling(cube, channels, rsun)

    cy, cx = _centroid(out[0])
    assert abs(cy - C) <= 1.0 and abs(cx - C) <= 1.0
    np.testing.assert_array_equal(out[1], cube[1])
    np.testing.assert_array_equal(cube[0], bad)  # input not modified in place


def test_wrong_sign_doubles_the_offset():
    cube = _corner_scaled_disk(RSUN_JULY)[None]
    d = corner_shift_px(RSUN_JULY, center=C)
    out = undo_corner_scaling(cube, ["aia171"], {"aia171": RSUN_JULY}, sign=-1)
    cy, _ = _centroid(out[0])
    assert abs((cy - C) - 2 * d) <= 1.0


def test_january_is_a_no_op():
    cube = np.random.default_rng(0).random((2, N, N)).astype(np.float32)
    out = undo_corner_scaling(cube, ["aia94", "hmi_m"], {"aia94": 976.0, "hmi_m": 976.0})
    np.testing.assert_array_equal(out, cube)


def test_vacated_edge_is_zero_filled_not_wrapped():
    cube = np.ones((1, N, N), dtype=np.float32)
    d = corner_shift_px(RSUN_JULY, center=C)
    out = undo_corner_scaling(cube, ["hmi_m"], {"hmi_m": RSUN_JULY})
    assert d > 0
    assert np.all(out[0, N - d :, :] == 0) and np.all(out[0, :, N - d :] == 0)
    assert np.all(out[0, : N - d, : N - d] == 1)


def test_affected_override_and_default_list():
    assert set(CORNER_SCALED_CHANNELS) == {
        "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "hmi_m",
    }
    cube = _corner_scaled_disk(RSUN_JULY)[None]
    out = undo_corner_scaling(cube, ["aia171"], {"aia171": RSUN_JULY}, affected=["hmi_m"])
    np.testing.assert_array_equal(out, cube)


def test_rsun_obs_from_attrs_parses_meta_1():
    ds = {
        "aia171": _Var({"meta_1": json.dumps({"RSUN_OBS": 950.5})}),
        "hmi_m": _Var({"meta_1": json.dumps({"rsun_obs": 951.0})}),
    }
    assert rsun_obs_from_attrs(ds, ["aia171", "hmi_m"]) == {"aia171": 950.5, "hmi_m": 951.0}


def test_rsun_obs_from_attrs_errors_clearly_without_metadata():
    with pytest.raises(KeyError, match="meta_1"):
        rsun_obs_from_attrs({"aia171": _Var({})}, ["aia171"])


def test_config_defaults_keep_old_behaviour():
    assert corner_scaling_kwargs({}) == dict(
        fix_corner_scaling=False, corner_scaling_channels=None, corner_scaling_sign=1
    )
    assert corner_scaling_kwargs({"fix_corner_scaling": True})["fix_corner_scaling"] is True


def test_helio_dataset_load_nc_data_applies_flag(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    helio = pytest.importorskip("surya.datasets.helio")

    channels = ["aia171", "hmi_bz"]
    bad = _corner_scaled_disk(RSUN_JULY)
    good = _disk(C, C, 0.4 * N)
    meta = json.dumps({"RSUN_OBS": RSUN_JULY})
    ds = xr.Dataset(
        {
            "aia171": xr.DataArray(bad, dims=["y", "x"], attrs={"meta_1": meta}),
            "hmi_bz": xr.DataArray(good, dims=["y", "x"], attrs={"meta_1": meta}),
        }
    )
    path = tmp_path / "20110701_0000.nc"
    ds.to_netcdf(path, engine="h5netcdf")

    loader = object.__new__(helio.HelioNetCDFDataset)
    loader.logger = logging.getLogger("test")
    loader.sdo_data_root_path = str(tmp_path)
    loader.corner_scaling_channels = None
    loader.corner_scaling_sign = 1

    loader.fix_corner_scaling = False
    off = loader.load_nc_data(path.name, None, channels)
    np.testing.assert_array_equal(off[0], bad)

    loader.fix_corner_scaling = True
    on = loader.load_nc_data(path.name, None, channels)
    cy, cx = _centroid(on[0])
    assert abs(cy - C) <= 1.0 and abs(cx - C) <= 1.0
    np.testing.assert_array_equal(on[1], good)
