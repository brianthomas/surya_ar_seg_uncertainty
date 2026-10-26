#!/usr/bin/env python
"""Build a holdout catalog of the largest ARs in one split of the AR benchmark.

The shipped holdout catalogs (assets/surya-bench-ar-segmentation/holdout_ar_large_from_*.csv)
were originally produced by hand, so there was no way to regenerate them or to retarget
them at a different split. This rebuilds them from the two catalogs they derive from:

  harp_catalog_subset.csv  hourly HARP bounding-box geometry pulled from JSOC
                           hmi.Mharp_720s, covering 2010-05-13 to 2024-12-31
  unified_catalog.csv      every benchmark split concatenated, tagged by `source`

Each AR is summarised at its *peak* -- the hour its deprojected bounding box is largest --
and ranked by that area. `<split>_overlap_files` counts how many of the split's mask files
the AR is tracked in, i.e. how much data a run would lose by holding the AR out.

The 2020-2024 (`test`) era needs no holdout selection: it is held out by construction, since
training stops at 2019-12-31. Reproducing the 2010-2019 catalog does, hence --holdout-select.

Usage:
    # 2020-2024, the 88 largest cleanly-deprojected ARs
    python make_holdout_catalog.py --split test --out holdout_ar_large_from_test.csv

    # every test-era HARP, unfiltered (reproduces harp_catalog_test_full.csv)
    python make_holdout_catalog.py --split test --top 0 --no-deproj-ok-only --out full.csv

    # 2010-2019, reproducing holdout_ar_large_from_training.csv
    python make_holdout_catalog.py --split train --holdout-select 30 --out from_training.csv
"""

import argparse

import numpy as np
import pandas as pd

R_SUN_KM = 696000.0
DEG2RAD = np.pi / 180.0

# Columns read from harp_catalog_subset.csv that must be numeric for the geometry to work.
GEOMETRY_COLUMNS = ("lat_fwt", "lon_fwt", "lat_min", "lon_min", "lat_max", "lon_max")

# Splits consulted when selecting a holdout set: the one being held out, plus the splits an
# AR must not touch at all to be eligible.
HOLDOUT_EXCLUDE_SPLITS = ("validation", "leaky_validation")


def load_harp_geometry(path: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return the HARP tracking rows between `start` and `end`, inclusive.

    Missing keywords are written as the literal string MISSING by the JSOC export, so they
    have to be declared as NA values rather than left to pandas' own inference.
    """
    cat = pd.read_csv(path, na_values=["MISSING"])
    cat["timestamp"] = pd.to_datetime(cat["timestamp"])
    era = cat[(cat.timestamp >= start) & (cat.timestamp <= end)].copy()
    if era.empty:
        raise SystemExit(f"{path} has no rows between {start} and {end}")
    for col in GEOMETRY_COLUMNS:
        era[col] = pd.to_numeric(era[col], errors="coerce")
    return era


def add_deprojected_extents(df: pd.DataFrame) -> pd.DataFrame:
    """Add bounding-box extents in km and the area they enclose, dropping incomplete rows.

    Longitude degrees shrink towards the limb, so the longitudinal extent is scaled by
    cos(mean latitude) to approximate the AR's true size rather than its projected one.
    """
    geo = df.dropna(subset=list(GEOMETRY_COLUMNS)).copy()
    geo["dlat_deg"] = geo["lat_max"] - geo["lat_min"]
    geo["dlon_deg"] = geo["lon_max"] - geo["lon_min"]
    mean_lat = (geo["lat_max"] + geo["lat_min"]) / 2.0
    geo["extent_lat_km"] = geo["dlat_deg"] * DEG2RAD * R_SUN_KM
    geo["extent_lon_km"] = (
        geo["dlon_deg"] * DEG2RAD * R_SUN_KM * np.cos(mean_lat * DEG2RAD)
    )
    geo["area_km2"] = geo["extent_lat_km"] * geo["extent_lon_km"]
    return geo


def peak_rows_per_harp(geo: pd.DataFrame) -> pd.DataFrame:
    """Return one row per HARP: the hour at which its deprojected area is largest."""
    return geo.loc[geo.groupby("harpnum")["area_km2"].idxmax()].copy()


def duration_days_per_harp(era: pd.DataFrame) -> pd.Series:
    """Return each HARP's tracked span in days, indexed by harpnum.

    Computed over every tracking row in the era, including those dropped for missing
    geometry -- the AR was still being tracked in those hours.
    """
    span = era.groupby("harpnum")["timestamp"].agg(["min", "max"])
    return ((span["max"] - span["min"]).dt.total_seconds() / 3600.0 / 24.0).round(1)


def classify_position(lat_fwt: float, lon_fwt: float) -> tuple[str, str]:
    """Return (position, deproj) for a flux-weighted centroid.

    An AR near a limb is foreshortened badly enough that its deprojected area is not
    trustworthy, so it is flagged Poor and normally filtered out.
    """
    tags = []
    if abs(lon_fwt) < 70:
        tags.append("Face on")
    if lon_fwt <= -50:
        tags.append("East Limb")
    if lon_fwt >= 50:
        tags.append("West Limb")
    if abs(lat_fwt) >= 50:
        tags.append("Polar")
    position = ", ".join(tags)
    return position, "Poor" if "Limb" in position else "OK"


def split_window(unified: pd.DataFrame, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last timestamp `split` covers, so no era is hardcoded."""
    rows = unified[unified["source"] == split]
    if rows.empty:
        sources = sorted(unified["source"].dropna().unique())
        raise SystemExit(f"split {split!r} not in unified catalog; have {sources}")
    return rows["timestamp"].min(), rows["timestamp"].max()


def split_present_timestamps(unified: pd.DataFrame, split: str) -> set:
    """Return the timestamps `split` actually has a mask file for."""
    rows = unified[(unified["source"] == split) & (unified["present"] == 1.0)]
    return set(rows["timestamp"])


def count_overlap(tracked: pd.Series, present: set) -> pd.Series:
    """Count, per HARP, how many of its tracked hours have a mask file in the split."""
    return tracked.map(lambda stamps: len(stamps & present))


def build_catalog(
    harp_catalog: str,
    unified_catalog: str,
    split: str,
    top: int,
    deproj_ok_only: bool,
    holdout_select: int,
) -> pd.DataFrame:
    """Rank the ARs tracked during `split` and return the holdout catalog for it."""
    unified = pd.read_csv(unified_catalog)
    unified["timestamp"] = pd.to_datetime(unified["timestamp"])

    start, end = split_window(unified, split)
    era = load_harp_geometry(harp_catalog, start, end)
    peak = peak_rows_per_harp(add_deprojected_extents(era))

    peak["duration_days"] = peak["harpnum"].map(duration_days_per_harp(era))
    positions = [
        classify_position(lat, lon) for lat, lon in zip(peak["lat_fwt"], peak["lon_fwt"])
    ]
    peak["position"] = [p for p, _ in positions]
    peak["deproj"] = [d for _, d in positions]

    # Rank over every AR in the era, then filter, so a row's rank states its true size
    # standing rather than its position within the filtered subset.
    peak = peak.sort_values("area_km2", ascending=False).reset_index(drop=True)
    peak["rank"] = peak.index + 1

    peak["area_km2_million"] = (peak["area_km2"] / 1e6).round(0)
    for col in ("extent_lat_km", "extent_lon_km"):
        peak[col] = peak[col].round(0)
    for col in ("lat_fwt", "lon_fwt", "dlat_deg", "dlon_deg"):
        peak[col] = peak[col].round(2)

    tracked = era.groupby("harpnum")["timestamp"].apply(set)
    overlap_column = f"{split}_overlap_files"
    peak[overlap_column] = count_overlap(
        peak["harpnum"].map(tracked), split_present_timestamps(unified, split)
    ).values

    if top:
        peak = peak[peak["rank"] <= top]
    if deproj_ok_only:
        peak = peak[peak["deproj"] == "OK"]

    if holdout_select:
        # An AR is only eligible if holding it out costs no validation data at all; among
        # those, prefer the ones that cost the least training data.
        for other in HOLDOUT_EXCLUDE_SPLITS:
            if other == split:
                continue
            touched = count_overlap(
                peak["harpnum"].map(tracked), split_present_timestamps(unified, other)
            )
            peak = peak[touched.values == 0]
        peak = peak.sort_values(overlap_column, ascending=True)
        peak = peak.head(holdout_select).sort_values("rank")

    return peak.rename(columns={"timestamp": "peak_timestamp"})[
        [
            "rank", "harpnum", "peak_timestamp", "lat_fwt", "lon_fwt",
            "dlat_deg", "dlon_deg", "extent_lat_km", "extent_lon_km",
            "area_km2_million", "duration_days", "position", "deproj", overlap_column,
        ]
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--harp-catalog", default="./assets/harp_catalog_subset.csv",
                   help="Hourly HARP bounding-box geometry from JSOC hmi.Mharp_720s.")
    p.add_argument("--unified-catalog", default="./assets/unified_catalog.csv",
                   help="Benchmark splits concatenated, tagged by a `source` column.")
    p.add_argument("--split", default="test",
                   help="Which `source` of the unified catalog to catalog ARs for.")
    p.add_argument("--top", type=int, default=100,
                   help="Keep only ranks up to this; 0 keeps every AR in the era.")
    p.add_argument("--deproj-ok-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop limb ARs, whose deprojected area is not trustworthy.")
    p.add_argument("--holdout-select", type=int, default=0,
                   help="Narrow to this many ARs that cost the least data to hold out. "
                        "Only meaningful for splits a model trains on.")
    p.add_argument("--out", required=True, help="Where the catalog CSV is written.")
    args = p.parse_args()

    catalog = build_catalog(
        args.harp_catalog, args.unified_catalog, args.split,
        args.top, args.deproj_ok_only, args.holdout_select,
    )
    catalog.to_csv(args.out, index=False)

    print(f"{args.split}: {len(catalog)} ARs -> {args.out}")
    if len(catalog):
        print(f"  ranks {catalog['rank'].min()}-{catalog['rank'].max()}, "
              f"peaks {catalog['peak_timestamp'].min()} to {catalog['peak_timestamp'].max()}")
        print(f"  {args.split}_overlap_files total: "
              f"{int(catalog[f'{args.split}_overlap_files'].sum())}")


if __name__ == "__main__":
    main()
