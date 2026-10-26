#!/usr/bin/env python
"""Stage the HelioFM input cubes named by a holdout AR catalog.

Each row of a holdout catalog (`holdout_ar_*.csv`) summarises one AR at its *peak* hour.
This downloads the one SDO NetCDF cube per AR that `run_bootstrap_inference.py` then runs
every ensemble member over, and writes the harpnum -> frame mapping alongside it so no
later step has to re-derive it.

The bucket is laid out `YYYY/MM/YYYYMMDD_HHMM.nc` at 12-minute cadence; catalog peak
timestamps are on the hour, so every peak has an exact object.

Resumable and verified: a cube whose local size already matches the object's
ContentLength is skipped, and transfers land on `<name>.nc.part` before being renamed, so
an interrupted run can never leave a truncated cube for h5netcdf to choke on later.

Example:
    python download_holdout_nc.py --dry-run
    python download_holdout_nc.py \
        --catalog ./assets/surya-bench-ar-segmentation/holdout_ar_nonflaring_30_from_test.csv \
        --out-dir /efs/bathomas/assets/holdout_nc_nonflaring30
"""

import argparse
import os
import sys

import pandas as pd

# `surya` is not pip-installed in every environment here, so locate the repo root from
# this file rather than relying on PYTHONPATH the way the training scripts do.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from surya.utils.s3 import make_s3_client

DEFAULT_CATALOG = (
    "./assets/surya-bench-ar-segmentation/holdout_ar_nonflaring_30_from_test.csv"
)
DEFAULT_OUT_DIR = "/efs/bathomas/assets/holdout_nc_nonflaring30"
DEFAULT_BUCKET = "s3://nasa-surya-bench"
# Where the AR benchmark's .h5 masks were unpacked. Recorded per frame rather than
# resolved later, so the analysis step never has to guess the YYYY/MM layout.
DEFAULT_MASK_ROOT = "./assets/surya-bench-ar-segmentation/data"


def frame_table(catalog: str, mask_root: str) -> pd.DataFrame:
    """One row per catalog AR: its peak frame, S3 key, and ground-truth mask path.

    `rank` is carried through because it is the catalog's own ordering (by deprojected
    peak area) and is non-contiguous -- it cannot be recovered from row position.
    """
    df = pd.read_csv(catalog)
    missing = {"harpnum", "peak_timestamp"} - set(df.columns)
    if missing:
        raise SystemExit(f"{catalog} is missing required column(s): {sorted(missing)}")

    peak = pd.to_datetime(df["peak_timestamp"])
    frame = peak.dt.strftime("%Y%m%d_%H%M")
    return pd.DataFrame({
        "harpnum": df["harpnum"],
        "rank": df["rank"] if "rank" in df.columns else pd.NA,
        "peak_timestamp": peak.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": frame,
        "key": peak.dt.strftime("%Y/%m/") + frame + ".nc",
        "truth_path": os.path.join(mask_root, "") + peak.dt.strftime("%Y/%m/") + frame + ".h5",
    })


def remote_size(client, bucket: str, key: str) -> int:
    """ContentLength of one object, or -1 if it is not there."""
    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception:
        return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=DEFAULT_CATALOG,
                   help="Holdout AR catalog CSV. Needs harpnum + peak_timestamp columns.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="Where the cubes and frames.csv are written.")
    p.add_argument("--bucket", default=DEFAULT_BUCKET,
                   help="Bucket holding the SDO NetCDF cubes.")
    p.add_argument("--mask-root", default=DEFAULT_MASK_ROOT,
                   help="Root of the unpacked AR masks, for the recorded truth_path.")
    p.add_argument("--signed", action="store_true",
                   help="Sign requests. The default is anonymous: nasa-surya-bench is "
                        "public and ambient credentials get a 403.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the keys and total bytes without transferring anything.")
    args = p.parse_args()

    bucket = args.bucket.removeprefix("s3://").rstrip("/")
    frames = frame_table(args.catalog, args.mask_root)
    client = make_s3_client(anon=not args.signed)

    print(f"{len(frames)} frames from {args.catalog}")
    print(f"  s3://{bucket}/<key>  ->  {args.out_dir}\n")

    rows, n_skipped, n_fetched, bytes_fetched = [], 0, 0, 0
    for row in frames.itertuples(index=False):
        dest = os.path.join(args.out_dir, f"{row.frame}.nc")
        size = remote_size(client, bucket, row.key)
        if size < 0:
            raise SystemExit(f"HARP {row.harpnum}: s3://{bucket}/{row.key} does not exist")

        if args.dry_run:
            print(f"  HARP {row.harpnum:>6}  {row.key}  {size / 1e6:,.0f} MB")
        elif os.path.exists(dest) and os.path.getsize(dest) == size:
            # Size match is the only cheap integrity check available; a partial
            # transfer is caught because .part files are never renamed.
            print(f"  HARP {row.harpnum:>6}  {row.frame}  present, skipping")
            n_skipped += 1
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            partial = f"{dest}.part"
            print(f"  HARP {row.harpnum:>6}  {row.frame}  downloading "
                  f"{size / 1e6:,.0f} MB ...", end="", flush=True)
            client.download_file(bucket, row.key, partial)
            got = os.path.getsize(partial)
            if got != size:
                os.remove(partial)
                raise SystemExit(f"\n  short read: got {got} bytes, expected {size}")
            os.replace(partial, dest)
            print(" ok")
            n_fetched += 1
            bytes_fetched += size

        if not os.path.exists(row.truth_path) and not args.dry_run:
            print(f"      WARNING: no ground-truth mask at {row.truth_path}")
        rows.append({**row._asdict(), "nc_path": dest, "bytes": size})

    out = pd.DataFrame(rows)[
        ["harpnum", "rank", "peak_timestamp", "frame", "key", "nc_path", "truth_path", "bytes"]
    ]
    if args.dry_run:
        print(f"\nwould transfer {len(out)} cubes, {out.bytes.sum() / 1e9:.1f} GB total")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = os.path.join(args.out_dir, "frames.csv")
    out.to_csv(manifest, index=False)
    print(f"\n{n_fetched} downloaded ({bytes_fetched / 1e9:.1f} GB), {n_skipped} already present")
    print(f"wrote {manifest}  ({len(out)} frames, {out.bytes.sum() / 1e9:.1f} GB on disk)")
    n_missing_truth = int((~out.truth_path.map(os.path.exists)).sum())
    print(f"ground-truth masks present: {len(out) - n_missing_truth}/{len(out)}")


if __name__ == "__main__":
    main()
