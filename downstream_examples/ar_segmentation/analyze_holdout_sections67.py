#!/usr/bin/env python
"""Boundary-localization and calibration statistics for every frame of an ensemble run.

Replicates sections 6 and 7 of `view_harp11930_ensemble_mean_672hr.ipynb` -- written
there for one frame -- over an arbitrary number of frames, so the single-frame headline
numbers (uncertainty 5.25 px, error 10.88 px, ratio 2.07x, Brier 0.1148, bias -0.0603)
can be read as distributions instead of anecdotes.

Section 6, "how well located is the boundary":

  * Method B (headline) -- average symmetric surface distance between each member's own
    p=0.5 boundary and the ensemble-mean boundary, and the same measurement against the
    ground-truth mask. The first is UNCERTAINTY (what the bootstrap can see), the second
    is ERROR (how wrong the ensemble actually is). They are different quantities and are
    never merged.
  * Method A (cross-check) -- sigma_r = sigma_p / |grad p-bar| on the *raw* mean field,
    plus the two assumptions it rests on, measured rather than asserted: whether p-bar is
    smooth inside a patch, and the patch-seam gradient bias.

Section 7, "can p-bar be read as a probability": a 10-bin reliability table over the
pixels where at least one member fired, plus Brier, bias and ECE. Bin *accumulators*
(n, n_true, sum_p) are stored rather than the ratios, so pooling across frames gives an
exact pooled curve rather than an average of per-frame averages.

Deliberately CPU-only -- it reads NetCDF and HDF5 and never imports torch. The epoch and
validation IoU of each member come from the `prob_*.nc` attributes, which record what
actually ran, rather than from `run_bootstrap_inference.BEST_EPOCH_BY_REGIME` (importing
that pulls in the whole model stack). `--check-best-epoch` cross-checks the two.

Resumable: a frame whose `ensemble_mean.nc` already exists is reloaded from that cache
rather than recomputed, unless `--overwrite` is passed.

Example:
    # the 30-AR non-flaring cohort
    python analyze_holdout_sections67.py \
        --infer_root /efs/bathomas/ar_seg_bootstrap/672hr_infer_holdout_nonflaring30 \
        --frames_csv /efs/bathomas/assets/holdout_nc_nonflaring30/frames.csv

    # regression check: must reproduce the notebook's published numbers for HARP 11930
    python analyze_holdout_sections67.py \
        --infer_root /efs/bathomas/ar_seg_bootstrap/672hr_infer_holdout_top3 \
        --frames 20241001_1200 --out_dir /tmp/regression --no-cache
"""

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "." not in sys.path:
    sys.path.insert(0, ".")
import false_color as fcolor  # SURYA_PATCH_PX

THRESHOLD = 0.5   # matches evaluate_model's hardcoded threshold in finetune.py
VOTE_MIN = 6      # hard-vote majority: k >= 6 of 10
MIN_N = 1000      # reliability floor: a thinner bin's frequency is noise
WIN = 512         # side of the most-contested window, as in section 5
PATCH_PX = fcolor.SURYA_PATCH_PX
CALIB_EDGES = np.linspace(0.0, 1.0, 11)
TRAIN_TRUE_FRAC = 0.00643  # training-era validation positive fraction, for the density ratio

DEFAULT_MASK_ROOT = "./assets/surya-bench-ar-segmentation/data"

# Boundary-distance histogram edges, for the pooled ECDF. Fine and linear where the
# distances actually live (the single-frame median is 1.4 px), log-spaced out to the
# 4096*sqrt(2) diagonal for the detached-island tail. One patch (16 px) is an exact edge,
# so "fraction within one patch" is read off a bin boundary rather than interpolated.
DIST_EDGES = np.unique(np.concatenate([
    np.arange(0.0, 64.0 + 0.25, 0.25),
    np.geomspace(64.5, 5800.0, 80),
]))


# ---------------------------------------------------------------------------
# Loaders -- same raster for prediction, mask and cube: north-up, crpix 2048.5,
# 0.6 arcsec/px, 4096^2. No reprojection anywhere.
# ---------------------------------------------------------------------------


def load_probability(frame_dir: str, member: str) -> np.ndarray:
    """One member's AR probability map, as float32 (H, W)."""
    with xr.open_dataset(os.path.join(frame_dir, f"prob_{member}.nc"),
                         engine="h5netcdf") as ds:
        return ds.ar_probability.values.astype(np.float32)


def probability_attrs(frame_dir: str, member: str) -> dict:
    """The provenance attrs `run_bootstrap_inference.write_probability` stored."""
    with xr.open_dataset(os.path.join(frame_dir, f"prob_{member}.nc"),
                         engine="h5netcdf") as ds:
        return dict(ds.attrs)


def load_truth(truth_path: str, key: str = "union_with_intersect") -> np.ndarray:
    """Ground-truth AR mask as bool (H, W). Stored 0/255 uint8.

    Whole-disk: it covers every AR in the frame, not only the catalog's target HARP.
    """
    with h5py.File(truth_path, "r") as f:
        return f[key][:] > 0


def members_present(frame_dir: str) -> list[str]:
    """Member names with a probability map in `frame_dir`, sorted."""
    return sorted(
        os.path.basename(p)[len("prob_"):-len(".nc")]
        for p in glob.glob(os.path.join(frame_dir, "prob_*.nc"))
    )


def plate_scale(source_nc: str) -> tuple[float, float, float]:
    """(arcsec/px, rsun_obs arcsec, km/px at disk centre) from the cube's own WCS.

    Read from the cube rather than hardcoded: rsun_obs swings ~1.7 % over a year with the
    Earth-Sun distance, so a fixed constant would mis-scale every frame but one.
    `rsun_ref` is in metres, hence the /1000.
    """
    with xr.open_dataset(source_nc) as cube:  # lazy: attrs, not the 580 MB of data
        wcs = json.loads(cube[list(cube.data_vars)[0]].attrs["meta_1"])
    arcsec_per_px = float(wcs["cdelt1"])
    rsun_obs = float(wcs["rsun_obs"])
    return arcsec_per_px, rsun_obs, arcsec_per_px / rsun_obs * float(wcs["rsun_ref"]) / 1000.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """IoU / precision / recall for one boolean prediction against one truth mask.

    Denominators are guarded: a member predicting nothing would otherwise divide by zero
    in precision rather than scoring the 0.0 it earned.
    """
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "pred_frac": float(pred.mean()),
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary: mask pixels with at least one 8-neighbour outside the mask.

    Erosion with a full 3x3 removes everything not completely surrounded, so what is left
    of `mask & ~eroded` is exactly the one-pixel rind. 8-connectivity rather than 4 so
    diagonal steps of a staircased edge count once, not twice.
    """
    return mask & ~binary_erosion(mask, np.ones((3, 3), bool))


def surface_distances(b_ens: np.ndarray, dist_to_ens: np.ndarray, other: np.ndarray):
    """Both directions of boundary-to-boundary distance, in pixels.

    Returns (d_other_to_ens, d_ens_to_other), never pre-averaged: the asymmetry says
    which boundary is longer and more convoluted, and collapsing to ASSD first throws
    that away.

    `dist_to_ens` is the EDT of the ensemble boundary, computed once by the caller and
    reused -- it is the expensive half (~1.3 s on 4096^2) and is the same for every
    member and for truth.

    Returns (None, None) when either boundary is empty: a member that predicted nothing
    has no boundary to be a distance from, and scipy's EDT of an all-background frame is
    not a distance at all. The caller records NaN and says so.
    """
    b_other = boundary(other)
    if not b_other.any() or not b_ens.any():
        return None, None
    return dist_to_ens[b_other], distance_transform_edt(~b_other)[b_ens]


def distance_summary(d_other_to_ens, d_ens_to_other) -> dict:
    """ASSD / median / HD95 over one boundary pair, plus the pooled distances."""
    if d_other_to_ens is None:
        return {"d_to_ens": np.nan, "d_from_ens": np.nan, "assd_px": np.nan,
                "median_px": np.nan, "hd95_px": np.nan, "pooled": None}
    pooled = np.concatenate([d_other_to_ens, d_ens_to_other])
    return {
        "d_to_ens": float(d_other_to_ens.mean()),
        "d_from_ens": float(d_ens_to_other.mean()),
        "assd_px": 0.5 * (float(d_other_to_ens.mean()) + float(d_ens_to_other.mean())),
        "median_px": float(np.median(pooled)),
        "hd95_px": float(np.percentile(pooled, 95)),
        "pooled": pooled,
    }


def calibration_bins(p_mean: np.ndarray, truth: np.ndarray, fired: np.ndarray) -> pd.DataFrame:
    """Reliability accumulators over the pixels where at least one member fired.

    Stores `n`, `n_true` and `sum_p` -- not the ratios -- so summing across frames gives
    the exact pooled curve. The k=0 population is ~95 % of the frame and would sit
    entirely in the first bin, setting the curve's left end on its own; it is excluded
    here and reported separately by the caller.
    """
    pm, tm = p_mean[fired], truth[fired]
    idx = np.clip(np.digitize(pm, CALIB_EDGES) - 1, 0, len(CALIB_EDGES) - 2)
    return pd.DataFrame({
        "bin_lo": CALIB_EDGES[:-1],
        "bin_hi": CALIB_EDGES[1:],
        "n": np.bincount(idx, minlength=10).astype(np.int64),
        "n_true": np.bincount(idx, weights=tm.astype(np.float64),
                              minlength=10).astype(np.int64),
        "sum_p": np.bincount(idx, weights=pm.astype(np.float64), minlength=10),
    })


def reliability_scores(bins: pd.DataFrame) -> dict:
    """Bias and ECE from bin accumulators alone, so frames and pools share one path.

    ECE is the n-weighted mean |observed - mean p| over bins clearing the reliability
    floor, which is the notebook's definition. Bias is over every binned pixel.
    """
    n_total = int(bins.n.sum())
    if n_total == 0:
        return {"mean_p": np.nan, "observed": np.nan, "bias": np.nan, "ece": np.nan,
                "n_binned": 0}
    mean_p = float(bins.sum_p.sum() / n_total)
    observed = float(bins.n_true.sum() / n_total)
    ok = bins.n >= MIN_N
    if ok.any():
        w = bins.n[ok] / bins.n[ok].sum()
        per_bin = (bins.n_true[ok] / bins.n[ok] - bins.sum_p[ok] / bins.n[ok]).abs()
        ece = float((w * per_bin).sum())
    else:
        ece = np.nan
    return {"mean_p": mean_p, "observed": observed, "bias": mean_p - observed,
            "ece": ece, "n_binned": n_total}


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------


def analyze_frame(frame: str, frame_dir: str, truth_path: str, meta: dict) -> dict:
    """Sections 3, 6 and 7 for one frame. Returns the row/table bundle for the caller."""
    members = members_present(frame_dir)
    if not members:
        raise SystemExit(f"{frame}: no prob_*.nc under {frame_dir}")
    n = len(members)

    truth = load_truth(truth_path)
    truth_frac = float(truth.mean())
    h, w = truth.shape

    # --- section 3: accumulate one member at a time. Sum and sum-of-squares in float64;
    # holding ten float32 4096^2 maps at once is 1.3 GB for no reason.
    total = np.zeros((h, w), dtype=np.float64)
    total_sq = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.uint8)
    member_rows, member_frame_means = [], {}

    for member in members:
        prob = load_probability(frame_dir, member)
        total += prob
        total_sq += np.square(prob, dtype=np.float64)
        pred = prob > THRESHOLD
        count += pred
        attrs = probability_attrs(frame_dir, member)
        member_frame_means[member] = float(prob.mean())
        member_rows.append({
            "frame": frame, "member": member, "kind": "member",
            "epoch": attrs.get("epoch"), "val_iou": attrs.get("val_iou"),
            **binary_metrics(pred, truth),
        })
        del prob, pred

    p_mean = (total / n).astype(np.float32)
    # var = E[p^2] - E[p]^2, population then corrected to the n-1 sample variance.
    # Clipped at 0: the identity is exact in real arithmetic, but a pixel where all
    # members agree to the last fp16 bit can land a few 1e-17 below zero.
    p_var = np.clip(total_sq / n - np.square(total / n), 0.0, None) * (n / (n - 1))
    p_sd = np.sqrt(p_var).astype(np.float32)
    del total, total_sq, p_var

    soft_pred = p_mean > THRESHOLD
    hard_pred = count >= VOTE_MIN
    soft = binary_metrics(soft_pred, truth)
    hard = binary_metrics(hard_pred, truth)

    scores = pd.DataFrame(member_rows).sort_values("iou")
    best_member = scores.iloc[-1]
    # Lower median of the n, so the row is a real member rather than an interpolation.
    median_member = scores.iloc[(n - 1) // 2]

    # --- plate scale, from the cube the maps were predicted from
    source_nc = probability_attrs(frame_dir, members[0]).get("source_file", "")
    if not source_nc or not os.path.exists(source_nc):
        raise SystemExit(
            f"{frame}: source cube {source_nc!r} (from prob_{members[0]}.nc attrs) is not "
            "on disk, and section 6 needs its WCS for the plate scale. Re-stage it with "
            "download_holdout_nc.py."
        )
    arcsec_per_px, rsun_obs, km_per_px = plate_scale(source_nc)

    # --- section 6, Method B
    b_ens = boundary(soft_pred)
    if not b_ens.any():
        raise SystemExit(f"{frame}: the ensemble predicted no AR at all; section 6 is "
                         "undefined on this frame")
    dist_to_ens = distance_transform_edt(~b_ens)  # computed once, reused n+1 times

    dist_hist = {}
    empty_boundary = []
    for row in member_rows:
        mask = load_probability(frame_dir, row["member"]) > THRESHOLD
        summary = distance_summary(*surface_distances(b_ens, dist_to_ens, mask))
        row["n_boundary_px"] = int(np.count_nonzero(boundary(mask)))
        if summary["pooled"] is None:
            empty_boundary.append(row["member"])
        else:
            dist_hist[row["member"]] = np.histogram(summary["pooled"], bins=DIST_EDGES)[0]
        row.update({k: v for k, v in summary.items() if k != "pooled"})
        del mask

    truth_summary = distance_summary(*surface_distances(b_ens, dist_to_ens, truth))
    if truth_summary["pooled"] is not None:
        dist_hist["ground truth"] = np.histogram(truth_summary["pooled"], bins=DIST_EDGES)[0]
    del dist_to_ens

    assd = np.array([r["assd_px"] for r in member_rows], dtype=float)
    unc = float(np.nanmean(assd))
    unc_sd = float(np.nanstd(assd, ddof=1))
    err = float(truth_summary["assd_px"])

    # --- section 6, Method A: sigma_r = sigma_p / |grad p-bar|, on the RAW field.
    # p-bar is an average of ten continuous maps and varies smoothly inside a patch;
    # smoothing it to one patch FWHM first flattens the gradient and inflates sigma_r to
    # a property of the kernel rather than of the ensemble.
    sigma_prob = float(p_sd[b_ens].mean())
    gy, gx = np.gradient(p_mean)
    grad = np.hypot(gy, gx)
    del gy, gx
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_r = (p_sd / grad).astype(np.float32)
    sr_b = sigma_r[b_ens]
    sr_b = sr_b[np.isfinite(sr_b)]

    # The two assumptions Method A rests on, measured rather than asserted.
    blocks = p_mean.reshape(h // PATCH_PX, PATCH_PX, w // PATCH_PX, PATCH_PX)
    touching = b_ens.reshape(h // PATCH_PX, PATCH_PX, w // PATCH_PX, PATCH_PX).any(axis=(1, 3))
    within_patch = float(np.median(
        (blocks.max(axis=(1, 3)) - blocks.min(axis=(1, 3)))[touching]))
    yy, xx = np.indices(p_mean.shape)
    on_seam = ((yy % PATCH_PX) == 0) | ((xx % PATCH_PX) == 0) | \
              ((yy % PATCH_PX) == PATCH_PX - 1) | ((xx % PATCH_PX) == PATCH_PX - 1)
    seam_grad = float(grad[b_ens & on_seam].mean())
    interior_grad = float(grad[b_ens & ~on_seam].mean())
    del grad, on_seam

    # --- de-projection. Treating the displacement as radial (the foreshortened
    # direction) is an upper bound; the tangential case is the undeprojected figure.
    r_arcsec = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) * arcsec_per_px
    mu = np.sqrt(np.clip(1.0 - (r_arcsec / rsun_obs) ** 2, 0.0, None))
    del yy, xx, r_arcsec
    mu_b = mu[b_ens]
    keep = mu_b >= 0.3
    del mu

    # --- section 5's most-contested window, for the exemplar figures
    contested = (count > 0) & (count < n)
    density = uniform_filter(contested.astype(np.float32), size=WIN, mode="constant")
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    y0, x0 = max(0, cy - WIN // 2), max(0, cx - WIN // 2)
    y1, x1 = min(h, y0 + WIN), min(w, x0 + WIN)
    del density, contested

    # --- section 7
    fired = count > 0
    bins = calibration_bins(p_mean, truth, fired)
    scores7 = reliability_scores(bins)
    pm_fired, tm_fired = p_mean[fired], truth[fired]
    brier = float(np.mean((pm_fired - tm_fired) ** 2)) if pm_fired.size else np.nan
    n_quiet = int(np.count_nonzero(~fired))
    missed = int(np.count_nonzero(truth & ~fired))
    del pm_fired, tm_fired

    frame_row = {
        "frame": frame, **meta,
        "n_members": n, "truth_frac": truth_frac,
        "truth_density_vs_train": truth_frac / TRAIN_TRUE_FRAC,
        "soft_iou": soft["iou"], "soft_precision": soft["precision"],
        "soft_recall": soft["recall"], "soft_pred_frac": soft["pred_frac"],
        "hard_iou": hard["iou"],
        "best_member": best_member["member"], "best_member_iou": float(best_member.iou),
        "median_member": median_member["member"],
        "median_member_iou": float(median_member.iou),
        "member_iou_sd": float(scores.iou.std(ddof=1)),
        # Section 6. Uncertainty is member scatter about the ensemble boundary; error is
        # the ensemble boundary against truth. Different quantities, kept apart by name.
        "boundary_assd_px": unc, "boundary_assd_sd_px": unc_sd,
        "boundary_median_px": float(np.nanmedian([r["median_px"] for r in member_rows])),
        "boundary_hd95_px": float(np.nanmean([r["hd95_px"] for r in member_rows])),
        "boundary_error_assd_px": err,
        "boundary_error_median_px": float(truth_summary["median_px"]),
        "boundary_error_hd95_px": float(truth_summary["hd95_px"]),
        "boundary_error_over_uncertainty": err / unc if unc else np.nan,
        "boundary_sigma_prob": sigma_prob,
        "n_boundary_px_ens": int(np.count_nonzero(b_ens)),
        "sigma_r_mean_px": float(sr_b.mean()) if sr_b.size else np.nan,
        "sigma_r_median_px": float(np.median(sr_b)) if sr_b.size else np.nan,
        "sigma_r_p90_px": float(np.percentile(sr_b, 90)) if sr_b.size else np.nan,
        "within_patch_range": within_patch,
        "seam_grad": seam_grad, "interior_grad": interior_grad,
        "seam_bias_ratio": seam_grad / interior_grad if interior_grad else np.nan,
        "mu_median_boundary": float(np.median(mu_b)),
        "mu_below_0p3_frac": float((~keep).mean()),
        "deprojected_km": unc * km_per_px * float(np.mean(1 / mu_b[keep])) if keep.any() else np.nan,
        "undeprojected_km": unc * km_per_px,
        "arcsec_per_px": arcsec_per_px, "rsun_obs": rsun_obs, "km_per_px": km_per_px,
        # Section 7
        "brier": brier, **scores7,
        "n_quiet": n_quiet, "quiet_frac": n_quiet / count.size,
        "quiet_true_frac": float(truth[~fired].mean()) if n_quiet else np.nan,
        "missed_ar_px": missed,
        "missed_ar_frac": missed / int(np.count_nonzero(truth)) if truth.any() else np.nan,
        # Provenance / reproducibility
        "source_nc": source_nc, "truth_path": truth_path,
        "win_y0": y0, "win_x0": x0, "win_y1": y1, "win_x1": x1,
        "empty_boundary_members": " ".join(empty_boundary),
        # Self-check: the frame-mean of p-bar must equal the mean of the members' own
        # frame-means. Exact in real arithmetic; fp16 storage of each map is the only
        # source of disagreement, so the tolerance is set by fp16, not by the averaging.
        "mean_check_abs_diff": abs(float(p_mean.mean())
                                   - float(np.mean(list(member_frame_means.values())))),
    }

    return {
        "frame_row": frame_row, "member_rows": member_rows, "bins": bins,
        "dist_hist": dist_hist,
        "fields": {"p_mean": p_mean, "p_sd": p_sd, "count": count, "truth": truth},
    }


def write_cache(path: str, result: dict, regime: str) -> None:
    """Cache one frame's ensemble fields and section-6/7 scalars.

    Same variables and attribute names section 8 of the notebook established, so the
    existing HARP 11930 cache and these new ones are readable by one code path. The
    boundary-distance histogram goes in as a variable; sigma_r does not -- it recomputes
    from p_mean and p_sd in seconds and would add 67 MB to a 168 MB file.
    """
    f = result["fields"]
    row = result["frame_row"]
    hist = result["dist_hist"]
    series = sorted(hist)

    attrs = {"title": f"Bootstrap ensemble mean AR segmentation, {row['n_members']} members",
             "regime": regime, "threshold": THRESHOLD, "vote_min": VOTE_MIN,
             "truth_key": "union_with_intersect",
             "truth_positive_fraction": row["truth_frac"],
             "geometry": ("north-up, crpix 2048.5, 0.6 arcsec/px, 4096x4096 "
                          "-- same as the .nc cube"),
             "truth_scope": ("whole-disk: the mask covers every AR in the frame, not only "
                             "the catalog's target HARP"),
             "members": " ".join(r["member"] for r in result["member_rows"])}
    for k, v in row.items():
        attrs[k] = "" if v is None else v

    ds = xr.Dataset(
        data_vars={
            "ar_probability_mean": (("y", "x"), f["p_mean"]),
            "ar_probability_sd": (("y", "x"), f["p_sd"]),
            "agreement_count": (("y", "x"), f["count"]),
            "ground_truth": (("y", "x"), f["truth"].astype(np.uint8)),
            "distance_hist": (("series", "dist_bin"),
                              np.stack([hist[s] for s in series]).astype(np.int64)),
            "distance_bin_lo": (("dist_bin",), DIST_EDGES[:-1]),
            "distance_bin_hi": (("dist_bin",), DIST_EDGES[1:]),
        },
        coords={"series": series},
        attrs=attrs,
    )
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()


def read_cache(path: str) -> dict:
    """Rebuild the row/table bundle from a cached frame, without the 4096^2 fields."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        attrs = dict(ds.attrs)
        series = [str(s) for s in ds.series.values]
        hist = {s: ds.distance_hist.values[i] for i, s in enumerate(series)}
    return {"frame_row": attrs, "dist_hist": hist, "cached": True}


# ---------------------------------------------------------------------------


def frame_metadata(frames_csv: str | None, mask_root: str) -> dict:
    """frame -> {harpnum, rank, peak_timestamp, truth_path}, from a download manifest.

    Without a manifest the truth path is derived from the frame stem's own YYYY/MM, and
    harpnum/rank are simply absent -- which is the right behaviour for an inference root
    that was never built from an AR catalog.
    """
    if not frames_csv:
        return {}
    df = pd.read_csv(frames_csv)
    out = {}
    for row in df.itertuples(index=False):
        out[str(row.frame)] = {
            "harpnum": int(row.harpnum), "rank": int(row.rank),
            "peak_timestamp": str(row.peak_timestamp), "truth_path": str(row.truth_path),
        }
    return out


def derive_truth_path(frame: str, mask_root: str) -> str:
    """`<mask_root>/YYYY/MM/<frame>.h5` from a YYYYMMDD_HHMM stem."""
    return os.path.join(mask_root, frame[:4], frame[4:6], f"{frame}.h5")


def check_best_epoch(member_rows: list[dict], regime: str) -> None:
    """Cross-check the attrs against run_bootstrap_inference's table. Imports torch."""
    from run_bootstrap_inference import BEST_EPOCH_BY_REGIME
    table = BEST_EPOCH_BY_REGIME[regime]
    for r in member_rows:
        want = table.get(r["member"])
        if want and int(r["epoch"]) != want[0]:
            print(f"    MISMATCH {r['frame']} {r['member']}: ran epoch {r['epoch']}, "
                  f"table says {want[0]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer_root", required=True,
                   help="Inference output root: one subdirectory of prob_*.nc per frame.")
    p.add_argument("--regime", default="672hr", help="Recorded in the cache attrs.")
    p.add_argument("--frames_csv", default=None,
                   help="frames.csv from download_holdout_nc.py, for harpnum/rank/truth.")
    p.add_argument("--mask-root", default=DEFAULT_MASK_ROOT,
                   help="Mask root, used when --frames_csv is not given.")
    p.add_argument("--frames", nargs="+", default=None, help="Subset of frame stems.")
    p.add_argument("--out_dir", default=None, help="Where the CSVs go. Default: infer_root.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute frames that already have an ensemble_mean.nc.")
    p.add_argument("--no-cache", action="store_true",
                   help="Compute without reading or writing ensemble_mean.nc.")
    p.add_argument("--check-best-epoch", action="store_true",
                   help="Cross-check each member's epoch against BEST_EPOCH_BY_REGIME.")
    args = p.parse_args()

    out_dir = args.out_dir or args.infer_root
    meta_by_frame = frame_metadata(args.frames_csv, args.mask_root)

    frames = args.frames or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(args.infer_root, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "prob_*.nc"))
    )
    if not frames:
        raise SystemExit(f"No frame directories with prob_*.nc under {args.infer_root}")
    print(f"{len(frames)} frames under {args.infer_root}  (regime {args.regime})\n")

    frame_rows, member_rows, bin_rows, hist_rows = [], [], [], []
    for i, frame in enumerate(frames, 1):
        frame_dir = os.path.join(args.infer_root, frame)
        cache = os.path.join(frame_dir, "ensemble_mean.nc")
        meta = dict(meta_by_frame.get(frame, {}))
        truth_path = meta.pop("truth_path", None) or derive_truth_path(frame, args.mask_root)

        if os.path.exists(cache) and not args.overwrite and not args.no_cache:
            result = read_cache(cache)
            print(f"[{i}/{len(frames)}] {frame}: cached")
        else:
            if not os.path.exists(truth_path):
                raise SystemExit(f"{frame}: no ground-truth mask at {truth_path}")
            t0 = time.time()
            result = analyze_frame(frame, frame_dir, truth_path, meta)
            row = result["frame_row"]
            if not args.no_cache:
                write_cache(cache, result, args.regime)
            del result["fields"]
            print(f"[{i}/{len(frames)}] {frame}: "
                  f"UNC {row['boundary_assd_px']:.2f} +/- {row['boundary_assd_sd_px']:.2f} px   "
                  f"ERR {row['boundary_error_assd_px']:.2f} px   "
                  f"ratio {row['boundary_error_over_uncertainty']:.2f}x   "
                  f"Brier {row['brier']:.4f}  bias {row['bias']:+.4f}   "
                  f"({time.time() - t0:.0f}s)")
            if row["empty_boundary_members"]:
                print(f"    WARNING: no boundary for {row['empty_boundary_members']} "
                      "-- excluded from this frame's uncertainty")
            if row["mean_check_abs_diff"] > 1e-3:
                print(f"    WARNING: mean-is-a-mean check off by "
                      f"{row['mean_check_abs_diff']:.2e} (fp16 eps ~ 1e-3)")
            if args.check_best_epoch:
                check_best_epoch(result["member_rows"], args.regime)

            member_rows += result["member_rows"]
            bin_rows.append(result["bins"].assign(frame=frame))

        frame_rows.append(result["frame_row"])
        for series, counts in result["dist_hist"].items():
            keep = counts > 0
            hist_rows.append(pd.DataFrame({
                "frame": frame, "series": series,
                "bin_lo": DIST_EDGES[:-1][keep], "bin_hi": DIST_EDGES[1:][keep],
                "count": counts[keep],
            }))

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, df in [
        ("metrics_frame.csv", pd.DataFrame(frame_rows)),
        ("metrics_member_frame.csv", pd.DataFrame(member_rows)),
        ("calibration_bins.csv", pd.concat(bin_rows, ignore_index=True) if bin_rows else None),
        ("distance_hist.csv", pd.concat(hist_rows, ignore_index=True) if hist_rows else None),
    ]:
        if df is None or df.empty:
            print(f"  (skipped {name}: every frame came from cache)")
            continue
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        written.append((path, len(df)))

    fr = pd.DataFrame(frame_rows)
    print(f"\n{'=' * 78}\n{len(fr)} frames, regime {args.regime}\n")
    for label, col in [("uncertainty ASSD (px)", "boundary_assd_px"),
                       ("error ASSD (px)", "boundary_error_assd_px"),
                       ("error / uncertainty", "boundary_error_over_uncertainty"),
                       ("Brier", "brier"), ("bias (p-bar - observed)", "bias"),
                       ("ECE", "ece"), ("soft-mean IoU", "soft_iou"),
                       ("truth positive fraction", "truth_frac")]:
        v = pd.to_numeric(fr[col], errors="coerce").dropna()
        if v.empty:
            continue
        print(f"  {label:<26} mean {v.mean():8.4f}  sd {v.std(ddof=1):7.4f}  "
              f"median {v.median():8.4f}  [{v.min():.4f}, {v.max():.4f}]")

    bad = fr[pd.to_numeric(fr.mean_check_abs_diff, errors="coerce") > 1e-3]
    print(f"\n  frames failing the fp16 mean-is-a-mean check: "
          f"{len(bad) if len(bad) else 'none'}")
    empties = fr[fr.empty_boundary_members.astype(str).str.strip() != ""]
    print(f"  frames with a member that predicted nothing:   "
          f"{len(empties) if len(empties) else 'none'}")
    for path, n in written:
        print(f"\nwrote {path}  ({n} rows)")


if __name__ == "__main__":
    main()
