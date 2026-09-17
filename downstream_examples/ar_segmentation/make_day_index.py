#!/usr/bin/env python
"""Build single-day HelioFM and AR index CSVs for finetuning on one date.

The shipped indices (assets/*_index_surya_1_0.csv) do not cover every calendar day in
every split: 2013-01-15 appears in valid_index_surya_1_0.csv but not the train index,
and 2013-01-01 is in neither. Restricting a run to one day therefore means generating
the pair of indices for it.

The SDO index is built by listing the S3 bucket, so every path it names is known to
exist. Where a shipped index also covers the day, the two agree -- for 2013-01-15 both
list the same 116 files. The AR index is filtered out of whichever shipped split
contains the date.

Usage:
    python make_day_index.py --date 2013-01-15
    python make_day_index.py --date 2013-01-15 --bucket s3://nasa-surya-bench
"""

import argparse
import os

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

# Shipped AR splits, searched in order for the requested date.
AR_SPLITS = ("train", "validation", "leaky_validation")


def list_sdo_day(bucket: str, date: pd.Timestamp) -> pd.DataFrame:
    """Return an index of every SDO NetCDF in `bucket` for `date`, sorted by time."""
    prefix = f"{date:%Y/%m/%Y%m%d}_"
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".nc")]
    if not keys:
        raise SystemExit(f"No objects under s3://{bucket}/{prefix}")

    timesteps = pd.to_datetime(
        [os.path.basename(k)[: -len(".nc")] for k in keys], format="%Y%m%d_%H%M"
    )
    # Paths stay relative so they join onto data.sdo_data_root_path, matching the
    # convention in the shipped indices.
    return pd.DataFrame({"path": keys, "timestep": timesteps, "present": 1}).sort_values(
        "timestep", ignore_index=True
    )


def find_ar_day(ar_root: str, date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Return the AR mask rows for `date` and the name of the split they came from."""
    for split in AR_SPLITS:
        path = os.path.join(ar_root, f"{split}.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        day = df[(df.timestamp >= date) & (df.timestamp < date + pd.Timedelta(days=1))]
        if len(day) > 0:
            return day.reset_index(drop=True), split
    raise SystemExit(f"{date:%Y-%m-%d} is not present in any of {AR_SPLITS} under {ar_root}")


def count_trainable(sdo: pd.DataFrame, ar: pd.DataFrame, target_minutes: int) -> int:
    """Count samples surviving HelioNetCDFDataset.filter_valid_indices then the AR join.

    A timestep is usable only if it and its +target_minutes counterpart are both in the
    SDO index, and it also has a mask. Mirrors the dataset so the config's
    iters_per_epoch can be set to something that is actually reachable.
    """
    stamps = set(sdo["timestep"])
    deltas = np.unique([np.timedelta64(0, "m"), np.timedelta64(target_minutes, "m")])
    valid = {t for t in stamps if all((t + d) in stamps for d in deltas)}
    return len(set(ar.loc[ar["present"] == 1, "timestamp"]) & valid)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="Day to extract, as YYYY-MM-DD.")
    p.add_argument("--bucket", default="s3://nasa-surya-bench",
                   help="Bucket holding the SDO NetCDF files.")
    p.add_argument("--ar-root", default="./assets/surya-bench-ar-segmentation",
                   help="Directory holding the shipped AR index CSVs.")
    p.add_argument("--out-dir", default="./assets/single_day",
                   help="Where the generated index CSVs are written.")
    p.add_argument("--target-minutes", type=int, default=60,
                   help="Must match data.time_delta_target_minutes in the config.")
    args = p.parse_args()

    date = pd.Timestamp(args.date)
    bucket = args.bucket.removeprefix("s3://").rstrip("/")
    os.makedirs(args.out_dir, exist_ok=True)

    sdo = list_sdo_day(bucket, date)
    ar, split = find_ar_day(args.ar_root, date)

    stem = f"{date:%Y%m%d}"
    sdo_out = os.path.join(args.out_dir, f"sdo_index_{stem}.csv")
    ar_out = os.path.join(args.out_dir, f"ar_index_{stem}.csv")
    sdo.to_csv(sdo_out)                      # leading unnamed column, as shipped
    ar.to_csv(ar_out, index=False)

    n = count_trainable(sdo, ar, args.target_minutes)
    print(f"{date:%Y-%m-%d}: {len(sdo)} SDO files, "
          f"{int((ar['present'] == 1).sum())} masks (from {split}.csv)")
    print(f"  wrote {sdo_out}")
    print(f"  wrote {ar_out}")
    print(f"  trainable samples: {n}  <- set iters_per_epoch_train to at most this")


if __name__ == "__main__":
    main()
