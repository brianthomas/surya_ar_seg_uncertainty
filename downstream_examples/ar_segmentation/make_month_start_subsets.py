#!/usr/bin/env python
"""Sub-sample month-start windows out of the shipped AR segmentation index CSVs.

The shipped indices (assets/surya-bench-ar-segmentation/{train,validation,leaky_validation}.csv)
run 2010-2019 at hourly cadence -- 74,760 rows for train alone. Short experiments want a small
reproducible slice instead. make_day_index.py already does this for a single calendar day; this
script generalizes it to a window of 1hr / 6hr / 1day / 3day / 1week / 2week taken from the start of
each month.

The splits do not cover whole months, which is the thing to watch:

    train.csv             107 blocks, Feb 15 - Dec 31    every February starts on the 15th,
                                                         and 2010-05 starts on the 13th
    validation.csv          9 blocks, Jan 15 - 31        no day-1 row exists anywhere
    leaky_validation.csv   18 blocks, days 1 - 14        a 2-week window is the entire block

So "beginning of the month" is taken to mean the first timestamp available in each (year, month)
block, not calendar day 1 -- anchoring strictly on day 1 yields nothing at all for
validation.csv and nothing for any February in train.csv. Pass --strict-month-start for the
literal reading; blocks without a day-1 row are then skipped with a warning.

Windows are labels only. HelioNetCDFDataset joins them against the SDO input index, and the
shipped indices mirror the AR splits (train_index_surya_1_0.csv covers 2011-02-15 onward,
valid_index_surya_1_0.csv covers Jan 15-31). Nothing shipped covers leaky_validation's
Jan 1-14 / Feb 1-14 or any of 2010, so those windows train on zero samples until the matching
SDO index is built from S3 -- pass --with-sdo-index to do that here.

By default every month's window is rolled up into a single index per split and period --
ar_index_train_1day.csv holds the first day of all 107 train months, which is what
ar_index_train in config.yaml wants to point at. Pass --per-month to get them broken out one
file per month instead.

Usage:
    python make_month_start_subsets.py
    python make_month_start_subsets.py --per-month
    python make_month_start_subsets.py --period 1day --year 2013
    python make_month_start_subsets.py --split leaky_validation --period 1week --with-sdo-index
    python make_month_start_subsets.py --strict-month-start
"""

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

# Shipped split name -> filename. The training split is train.csv, not training.csv.
SPLITS = {
    "train": "train.csv",
    "validation": "validation.csv",
    "leaky_validation": "leaky_validation.csv",
}

# Window lengths, shortest first. Keys double as output filename suffixes.
PERIODS = {
    "1hr": pd.Timedelta(hours=1),
    "6hr": pd.Timedelta(hours=6),
    "1day": pd.Timedelta(days=1),
    "3day": pd.Timedelta(days=3),
    "1week": pd.Timedelta(weeks=1),
    "2week": pd.Timedelta(weeks=2),
}

# Shipped SDO input index per split, used to report how many samples a window can actually
# train on. validation and leaky_validation are both validation-phase splits; leaky_validation
# has no shipped coverage at all, so the lookup simply finds nothing for it.
DEFAULT_SDO_INDEX = {
    "train": "./assets/train_index_surya_1_0.csv",
    "validation": "./assets/valid_index_surya_1_0.csv",
    "leaky_validation": "./assets/valid_index_surya_1_0.csv",
}


def load_split(ar_root: str, split: str) -> pd.DataFrame:
    """Read one shipped AR index, with timestamps parsed and sorted."""
    path = os.path.join(ar_root, SPLITS[split])
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} not found. Download the masks with ./download_data.sh, or point "
            f"--ar-root at the directory holding the shipped index CSVs."
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ignore_index=True)


def month_blocks(df: pd.DataFrame):
    """Yield (year, month, block) for each calendar month present, in time order.

    Rows within a month are contiguous and hourly in every shipped split, so the block's first
    timestamp is the month's anchor and its last is where a window has to stop.
    """
    for (year, month), block in df.groupby(
        [df["timestamp"].dt.year, df["timestamp"].dt.month], sort=True
    ):
        yield int(year), int(month), block.reset_index(drop=True)


def month_start_window(
    block: pd.DataFrame, year: int, month: int, period: pd.Timedelta, strict: bool
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Slice `period` from the start of one month-block.

    Returns (rows, anchor). The anchor is the block's first timestamp, or calendar day 1 00:00
    under `strict`. A strict anchor the block does not contain yields an empty frame and a None
    anchor, which the caller reports and skips.
    """
    if strict:
        anchor = pd.Timestamp(year=year, month=month, day=1)
        if anchor not in set(block["timestamp"]):
            return block.iloc[:0], None
    else:
        anchor = block["timestamp"].iloc[0]

    # Half-open [anchor, anchor + period). Bounded by the block, so a window longer than the
    # remaining days truncates here rather than bleeding into the next block -- which would be
    # a different year, not the next day.
    in_window = (block["timestamp"] >= anchor) & (block["timestamp"] < anchor + period)
    return block.loc[in_window].reset_index(drop=True), anchor


def load_sdo_index(path: str) -> pd.DataFrame | None:
    """Read a shipped SDO input index, or None if it is not there."""
    if not path or not os.path.isfile(path):
        return None
    sdo = pd.read_csv(path)
    sdo["timestep"] = pd.to_datetime(sdo["timestep"])
    return sdo


def build_sdo_window(bucket: str, rows: pd.DataFrame, deltas: set) -> pd.DataFrame:
    """List S3 for every calendar day the window needs and concatenate the day indices.

    Imported lazily so the default no-network path does not need boto3 or credentials.

    The range runs to the last timestamp plus the largest sequence delta, not just to the last
    timestamp: the final hour of a window ending at 23:00 needs its +target counterpart on the
    following day, and leaving that day unlisted would quietly make that hour untrainable.
    """
    from make_day_index import list_sdo_day

    # list_sdo_day passes the name straight to boto3, which rejects a URI -- make_day_index
    # strips the scheme in its own main() before calling it, so do the same here.
    bucket = bucket.removeprefix("s3://").rstrip("/")
    last = rows["timestamp"].iloc[-1] + max(deltas)
    days = pd.date_range(rows["timestamp"].iloc[0].normalize(), last.normalize(), freq="D")
    frames = [list_sdo_day(bucket, pd.Timestamp(day)) for day in days]
    return pd.concat(frames, ignore_index=True).sort_values("timestep", ignore_index=True)


def write_ar_index(rows: pd.DataFrame, path: str) -> None:
    """Write an AR index in the shipped format.

    index=False and the timestamp,file_path,present columns keep these byte-compatible with the
    shipped splits and assets/single_day outputs. date_format is explicit because pandas drops
    the time component when every value in the column is midnight, which is exactly the 1hr
    single-row case -- it would write "2013-03-01" where every shipped file has
    "2013-03-01 00:00:00".
    """
    rows.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def sequence_deltas(input_minutes: list[int], target_minutes: int, rollout_steps: int) -> set:
    """The offsets that must all be present for a timestep to be a usable sample.

    Mirrors helio.py:290-296: the input deltas plus one target delta per rollout step.
    """
    deltas = set(pd.to_timedelta(input_minutes, unit="m"))
    deltas |= {
        pd.Timedelta(minutes=iroll * target_minutes) for iroll in range(1, rollout_steps + 2)
    }
    return deltas


def observed_stamps(sdo: pd.DataFrame | None) -> set | None:
    """The SDO timesteps that actually hold an observation, or None if there is no index.

    helio.py:299 drops `present != 1` before building valid_indices, so unobserved timesteps
    cannot anchor a sample. Built once per split -- it is ~310k entries for the shipped train
    index, far too expensive to rebuild per window.
    """
    if sdo is None or len(sdo) == 0:
        return None
    return set(sdo.loc[sdo["present"] == 1, "timestep"])


def count_window_trainable(stamps: set | None, rows: pd.DataFrame, deltas: set) -> int:
    """Samples in `rows` that survive the SDO join, or -1 when no index is available.

    Computed here rather than via make_day_index.count_trainable, which overcounts: that helper
    ignores the SDO index's `present` column and assumes a single target delta. Both inflate the
    result -- it reported 23 where the dataset itself yields 22 for the 2013-02 1day window.
    """
    if stamps is None:
        return -1
    wanted = set(rows.loc[rows["present"] == 1, "timestamp"])
    return sum(1 for t in wanted if all((t + d) in stamps for d in deltas))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ar-root", default="./assets/surya-bench-ar-segmentation",
                   help="Directory holding the shipped AR index CSVs.")
    p.add_argument("--out-dir", default="./assets/subsamples",
                   help="Where the sub-sampled indices and manifest.csv are written.")
    p.add_argument("--split", action="append", choices=sorted(SPLITS),
                   help="Split to sub-sample; repeatable. Default: all three.")
    p.add_argument("--period", action="append", choices=list(PERIODS),
                   help="Window length; repeatable. Default: all five.")
    p.add_argument("--year", action="append", type=int,
                   help="Restrict to these years; repeatable. Default: all.")
    p.add_argument("--month", action="append", type=int,
                   help="Restrict to these months (1-12); repeatable. Default: all.")
    p.add_argument("--per-month", action="store_true",
                   help="Write one file per month instead of rolling every month's window into a "
                        "single file per split and period. Lays them out as "
                        "<out-dir>/<split>/<period>/ar_index_<split>_<YYYYMM>_<period>.csv.")
    p.add_argument("--strict-month-start", action="store_true",
                   help="Anchor on calendar day 1 00:00 and skip months without it. Yields "
                        "nothing for validation.csv and nothing for any February in train.csv.")
    p.add_argument("--with-sdo-index", action="store_true",
                   help="Also build the matching SDO input index by listing S3. Needed for "
                        "leaky_validation and 2010 windows, which no shipped index covers.")
    p.add_argument("--bucket", default="s3://nasa-surya-bench",
                   help="Bucket holding the SDO NetCDF files, for --with-sdo-index.")
    p.add_argument("--sdo-index", default=None,
                   help="Shipped SDO index used for the trainable count. Default: per-split, "
                        "train_index_surya_1_0.csv or valid_index_surya_1_0.csv.")
    # These three shape the trainable count and must match the config the windows will be used
    # with, since HelioNetCDFDataset only keeps a timestep whose whole input/target sequence is
    # present in the SDO index.
    p.add_argument("--target-minutes", type=int, default=60,
                   help="Must match data.time_delta_target_minutes in the config.")
    p.add_argument("--input-minutes", type=int, nargs="+", default=[0],
                   help="Must match data.time_delta_input_minutes in the config.")
    p.add_argument("--rollout-steps", type=int, default=0,
                   help="Must match rollout_steps in the config.")
    args = p.parse_args()

    splits = args.split or list(SPLITS)
    periods = args.period or list(PERIODS)
    years = set(args.year) if args.year else None
    months = set(args.month) if args.month else None

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    warnings = []
    # (split, period) -> the per-month windows to concatenate, unless --per-month.
    rollup_ar = defaultdict(list)
    rollup_sdo = defaultdict(list)

    deltas = sequence_deltas(args.input_minutes, args.target_minutes, args.rollout_steps)

    for split in splits:
        df = load_split(args.ar_root, split)
        sdo_shipped = load_sdo_index(args.sdo_index or DEFAULT_SDO_INDEX[split])
        shipped_stamps = observed_stamps(sdo_shipped)

        for year, month, block in month_blocks(df):
            if (years and year not in years) or (months and month not in months):
                continue
            block_end = block["timestamp"].iloc[-1]

            for period in periods:
                rows, anchor = month_start_window(
                    block, year, month, PERIODS[period], args.strict_month_start
                )
                if anchor is None:
                    warnings.append(
                        f"{split} {year}-{month:02d}: no {year}-{month:02d}-01 00:00 row "
                        f"(block starts {block['timestamp'].iloc[0]:%Y-%m-%d}); skipped under "
                        f"--strict-month-start"
                    )
                    continue
                if len(rows) == 0:
                    warnings.append(f"{split} {year}-{month:02d} {period}: empty window; skipped")
                    continue

                expected = int(PERIODS[period] / pd.Timedelta(hours=1))
                truncated = len(rows) < expected

                if args.per_month:
                    out_dir = os.path.join(args.out_dir, split, period)
                    os.makedirs(out_dir, exist_ok=True)
                    ar_out = os.path.join(out_dir, f"ar_index_{split}_{year}{month:02d}_{period}.csv")
                    write_ar_index(rows, ar_out)
                else:
                    # Rolled up: every month's window for this (split, period) lands in one file,
                    # written after the sweep. The manifest still carries a row per month so the
                    # per-month coverage stays visible.
                    out_dir = args.out_dir
                    ar_out = os.path.join(args.out_dir, f"ar_index_{split}_{period}.csv")
                    rollup_ar[(split, period)].append(rows)

                # Covered means the shipped index has at least one timestep inside the window;
                # a window straddling its edge is partially usable, which `trainable` quantifies.
                sdo_covered = sdo_shipped is not None and bool(
                    (
                        (sdo_shipped["timestep"] >= rows["timestamp"].iloc[0])
                        & (sdo_shipped["timestep"] <= rows["timestamp"].iloc[-1])
                    ).any()
                )

                sdo_out = ""
                if args.with_sdo_index:
                    sdo = build_sdo_window(args.bucket, rows, deltas)
                    stamps = observed_stamps(sdo)
                    if args.per_month:
                        sdo_out = os.path.join(
                            out_dir, f"sdo_index_{split}_{year}{month:02d}_{period}.csv"
                        )
                        sdo.to_csv(sdo_out)  # leading unnamed column, as shipped
                    else:
                        sdo_out = os.path.join(args.out_dir, f"sdo_index_{split}_{period}.csv")
                        rollup_sdo[(split, period)].append(sdo)
                else:
                    stamps = shipped_stamps
                    if not sdo_covered:
                        warnings.append(
                            f"{split} {year}-{month:02d} {period}: no shipped SDO index covers "
                            f"{rows['timestamp'].iloc[0]:%Y-%m-%d}..{rows['timestamp'].iloc[-1]:%Y-%m-%d}; "
                            f"this window trains on 0 samples. Re-run with --with-sdo-index."
                        )

                records.append({
                    "split": split,
                    "period": period,
                    "year": year,
                    "month": month,
                    "anchor": rows["timestamp"].iloc[0],
                    "end": rows["timestamp"].iloc[-1],
                    "n_rows": len(rows),
                    "n_present": int((rows["present"] == 1).sum()),
                    "expected_hours": expected,
                    "truncated": truncated,
                    "block_end": block_end,
                    "sdo_covered": bool(args.with_sdo_index or sdo_covered),
                    "trainable": count_window_trainable(stamps, rows, deltas),
                    "ar_index_path": ar_out,
                    "sdo_index_path": sdo_out,
                })

    # Warnings come first: when every window was skipped they are the only explanation of why.
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    if not records:
        raise SystemExit("No windows matched the requested splits/periods/years/months.")

    # Concatenated in month order, so each rolled-up file stays sorted the way the shipped
    # splits are. Windows never overlap -- one per month, bounded by that month's block.
    for (split, period), frames in sorted(rollup_ar.items()):
        out = os.path.join(args.out_dir, f"ar_index_{split}_{period}.csv")
        write_ar_index(pd.concat(frames, ignore_index=True), out)
    for (split, period), frames in sorted(rollup_sdo.items()):
        out = os.path.join(args.out_dir, f"sdo_index_{split}_{period}.csv")
        # Boundary days can repeat across windows, so drop_duplicates before writing; the index
        # is reset so the leading unnamed column is 0..N-1 as in the shipped indices.
        sdo = pd.concat(frames, ignore_index=True)
        sdo = sdo.drop_duplicates("timestep").sort_values("timestep", ignore_index=True)
        sdo.to_csv(out)

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    n_files = len(rollup_ar) if not args.per_month else len(records)
    layout = "one file per month" if args.per_month else "rolled up per split and period"
    print(f"wrote {n_files} index file(s) ({layout}) from {len(records)} month window(s) "
          f"under {args.out_dir}")
    print(f"  manifest: {manifest_path}")

    # In rollup mode the per-month rows are the manifest's job; what matters on screen is what
    # each written file contains, so aggregate to one line per file.
    if args.per_month:
        cols = ["split", "period", "year", "month", "n_rows", "n_present", "truncated",
                "sdo_covered", "trainable"]
        view = manifest[cols]
    else:
        view = (
            manifest.groupby(["split", "period"])
            .agg(months=("month", "size"), n_rows=("n_rows", "sum"),
                 n_present=("n_present", "sum"),
                 # -1 means no SDO index at all; it must not be summed into a real count.
                 trainable=("trainable", lambda s: -1 if (s == -1).any() else int(s.sum())),
                 months_without_sdo=("sdo_covered", lambda s: int((~s).sum())))
            .reset_index()
        )
    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print(view.to_string(index=False))
    print("\ntrainable = samples surviving the SDO join (-1: no SDO index available). "
          "Set iters_per_epoch to at most this.")


if __name__ == "__main__":
    main()
