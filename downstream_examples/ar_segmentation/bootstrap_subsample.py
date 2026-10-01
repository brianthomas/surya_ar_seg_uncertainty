#!/usr/bin/env python
"""Bootstrap-resample one AR segmentation index, producing replicates in the same format.

Treats the rows of a subsample index (the files make_month_start_subsets.py writes under
assets/subsamples/) as the population and draws from it **with replacement**, so each output is a
bootstrap replicate usable directly as data.ar_index_train / ar_index_valid in config.yaml. The
output schema is the shipped timestamp,file_path,present -- nothing downstream needs changing.

Duplicate timestamps are the point of the exercise and are written literally. ArDSDataset handles
them: dataset.py:86 inner-joins the AR index against the unique SDO valid_indices, so a timestamp
drawn k times yields k rows; dataset.py:95 rebuilds valid_indices from that merged frame, keeping
it positionally aligned with ar_valid_indices, which is only ever read via .iloc (dataset.py:101
and :139). So each duplicate is a separate training sample, and adjusted_length counts it.

Sampling is **block** bootstrap by default. This data is strongly autocorrelated in time -- it is
why leaky_validation exists as its own split -- so drawing individual hours would treat ~24
near-identical images as 24 independent observations and report tighter uncertainty than the data
supports. The population is tiled into contiguous non-overlapping blocks of --block-hours (default
24) and blocks are drawn with replacement. Pass --iid for plain row-wise draws.

Blocks never straddle a gap in time: the rolled-up indices concatenate non-adjacent months, so
tiling respects contiguous hourly runs. A run shorter than the block becomes one short block, so
a 1hr index (runs of 1) degenerates to i.i.d. on its own and a 6hr index resamples whole windows.

Note the effective sample size is the number of *blocks*, not rows: ar_index_train_1week.csv has
17,976 rows but only 749 blocks at 24h, and 749 is what bounds the bootstrap's resolution.

Usage:
    python bootstrap_subsample.py assets/subsamples/ar_index_train_1day.csv
    python bootstrap_subsample.py <input> --n-replicates 100 --seed 42 --write-oob
    python bootstrap_subsample.py <input> --block-hours 72
    python bootstrap_subsample.py <input> --iid
    python bootstrap_subsample.py <input> --population trainable --n-replicates 20
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from make_month_start_subsets import (
    DEFAULT_SDO_INDEX,
    load_sdo_index,
    observed_stamps,
    sequence_deltas,
    write_ar_index,
)


def split_from_filename(path: str) -> str | None:
    """Recover the split name from an ar_index_<split>_<period>.csv filename.

    Used only to pick the default shipped SDO index for --population trainable. Longest match
    wins so leaky_validation is not read as validation.
    """
    name = os.path.basename(path)
    for split in sorted(DEFAULT_SDO_INDEX, key=len, reverse=True):
        if re.match(rf"^ar_index_{re.escape(split)}_", name):
            return split
    return None


def load_population(path: str, population: str, args) -> pd.DataFrame:
    """Read the source index and reduce it to the rows eligible to be drawn."""
    pop = pd.read_csv(path)
    missing = {"timestamp", "file_path", "present"} - set(pop.columns)
    if missing:
        raise SystemExit(f"{path} is missing expected column(s): {sorted(missing)}")
    pop["timestamp"] = pd.to_datetime(pop["timestamp"])
    pop = pop.sort_values("timestamp", ignore_index=True)
    n_all = len(pop)

    if population != "all":
        # dataset.py:74 drops these anyway; keeping them would leave unusable rows in the
        # replicate and make its effective size vary between runs.
        pop = pop.loc[pop["present"] == 1].reset_index(drop=True)

    if population == "trainable":
        split = args.split or split_from_filename(path)
        if split is None:
            raise SystemExit(
                f"--population trainable needs to know the split to pick an SDO index; "
                f"could not infer it from {os.path.basename(path)}. Pass --split."
            )
        sdo_path = args.sdo_index or DEFAULT_SDO_INDEX[split]
        stamps = observed_stamps(load_sdo_index(sdo_path))
        if stamps is None:
            raise SystemExit(
                f"--population trainable needs an SDO index, but {sdo_path} was not found. "
                f"Build one with make_month_start_subsets.py --with-sdo-index."
            )
        deltas = sequence_deltas(args.input_minutes, args.target_minutes, args.rollout_steps)
        keep = [all((t + d) in stamps for d in deltas) for t in pop["timestamp"]]
        pop = pop.loc[keep].reset_index(drop=True)

    if len(pop) == 0:
        raise SystemExit(
            f"Population is empty after --population {population} ({n_all} rows in {path})."
        )
    return pop


def build_blocks(pop: pd.DataFrame, block_hours: int) -> list[np.ndarray]:
    """Tile the population into the units that get drawn with replacement.

    Rows are split into contiguous hourly runs first, so no block spans a gap in time -- the
    rolled-up indices concatenate non-adjacent months, and a block straddling that boundary would
    be a fabricated stretch of time. Each run is then cut into consecutive non-overlapping blocks
    of at most block_hours rows.

    Non-overlapping (rather than moving) blocks keep the statistics simple: drawing B blocks from
    B with replacement leaves ~1/e of them untouched, and because the blocks partition the rows
    that carries over to the row level unchanged.
    """
    gaps = pop["timestamp"].diff() != pd.Timedelta(hours=1)
    run_id = gaps.cumsum()
    blocks = []
    for _, run in pop.groupby(run_id, sort=True):
        pos = run.index.to_numpy()
        # A run shorter than block_hours yields a single short block rather than an error.
        blocks.extend(np.array_split(pos, max(1, int(np.ceil(len(pos) / block_hours)))))
    return blocks


def draw(blocks: list[np.ndarray], rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw blocks with replacement until `size` rows are accumulated, then trim to exactly size.

    Returns row positions (with duplicates), not rows, so out-of-bag membership is a set
    difference on positions.

    Fixing the row count rather than the block count leaves a small upward bias in coverage when
    block lengths vary: a run of short draws needs more of them to reach `size`, so more distinct
    blocks get touched. Measured over 100 replicates of ar_index_train_1day.csv, the unique
    fraction is 0.639 at 24h blocks against 0.632 for equal-size units (--iid gives 0.6318). The
    alternative -- a fixed block count -- would hit 1/e exactly but make the output size vary,
    which is worse for a file meant to be a drop-in index.
    """
    lengths = np.array([len(b) for b in blocks])
    # Enough draws that even all-shortest-blocks reaches size; trimmed back afterwards.
    n_draws = int(np.ceil(size / lengths.min()))
    picked = rng.integers(0, len(blocks), size=n_draws)

    taken, total = [], 0
    for i in picked:
        taken.append(blocks[i])
        total += len(blocks[i])
        if total >= size:
            break
    return np.concatenate(taken)[:size]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="Index CSV to treat as the population, e.g. "
                                "assets/subsamples/ar_index_train_1day.csv")
    p.add_argument("--out-dir", default=None,
                   help="Where replicates are written. Default: alongside the input.")
    p.add_argument("--n-replicates", type=int, default=1,
                   help="How many bootstrap replicates to draw. Default: 1.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the replicate sequence. Default: 0.")
    p.add_argument("--write-oob", action="store_true",
                   help="Also write the out-of-bag rows (those never drawn) per replicate, "
                        "usable as a validation index.")
    p.add_argument("--population", choices=("present", "trainable", "all"), default="present",
                   help="Which rows are eligible. present (default): rows with a mask. "
                        "trainable: also require the SDO input sequence to exist. "
                        "all: every row verbatim, including present=0.")

    blocks = p.add_mutually_exclusive_group()
    blocks.add_argument("--block-hours", type=int, default=24,
                        help="Block length in hours for the block bootstrap. Default: 24. Blocks "
                             "never cross a gap in time; a shorter run becomes one short block.")
    blocks.add_argument("--iid", action="store_true",
                        help="Draw individual rows instead of blocks (same as --block-hours 1). "
                             "Overstates the effective sample size on this autocorrelated data.")

    size = p.add_mutually_exclusive_group()
    size.add_argument("--size", type=int, default=None,
                      help="Rows per replicate. Default: the population size.")
    size.add_argument("--fraction", type=float, default=None,
                      help="Rows per replicate as a fraction of the population size.")

    # Only consulted for --population trainable.
    p.add_argument("--split", choices=sorted(DEFAULT_SDO_INDEX), default=None,
                   help="Split the input came from. Default: inferred from the filename.")
    p.add_argument("--sdo-index", default=None,
                   help="SDO index for --population trainable. Default: per-split shipped index.")
    p.add_argument("--target-minutes", type=int, default=60,
                   help="Must match data.time_delta_target_minutes in the config.")
    p.add_argument("--input-minutes", type=int, nargs="+", default=[0],
                   help="Must match data.time_delta_input_minutes in the config.")
    p.add_argument("--rollout-steps", type=int, default=0,
                   help="Must match rollout_steps in the config.")
    args = p.parse_args()

    block_hours = 1 if args.iid else args.block_hours
    if block_hours < 1:
        raise SystemExit("--block-hours must be at least 1.")
    if args.n_replicates < 1:
        raise SystemExit("--n-replicates must be at least 1.")

    pop = load_population(args.input, args.population, args)
    block_list = build_blocks(pop, block_hours)

    if args.fraction is not None:
        if args.fraction <= 0:
            raise SystemExit("--fraction must be positive.")
        size = max(1, int(round(args.fraction * len(pop))))
    else:
        size = args.size if args.size is not None else len(pop)
    if size < 1:
        raise SystemExit("--size must be at least 1.")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    rng = np.random.default_rng(args.seed)
    records = []
    for rep in range(1, args.n_replicates + 1):
        pos = draw(block_list, rng, size)

        # Sorted so duplicates sit adjacent and the file reads like the shipped indices.
        # dataset.py:79 sorts anyway, so this is free.
        rows = pop.iloc[pos].sort_values("timestamp", ignore_index=True)
        ar_out = os.path.join(out_dir, f"{stem}_boot{rep:03d}.csv")
        write_ar_index(rows, ar_out)

        drawn = set(pos)
        oob_out = ""
        if args.write_oob:
            oob = pop.drop(index=sorted(drawn)).reset_index(drop=True)
            oob_out = os.path.join(out_dir, f"{stem}_oob{rep:03d}.csv")
            write_ar_index(oob, oob_out)

        records.append({
            "source": args.input,
            "replicate": rep,
            "seed": args.seed,
            "block_hours": block_hours,
            "population_rows": len(pop),
            "population_blocks": len(block_list),
            "n_draws": len(rows),
            "n_unique_drawn": len(drawn),
            "unique_fraction": round(len(drawn) / len(pop), 4),
            "n_oob": len(pop) - len(drawn),
            "ar_index_path": ar_out,
            "oob_index_path": oob_out,
        })

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(out_dir, f"{stem}_bootstrap_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    mode = "i.i.d. rows" if block_hours == 1 else f"{block_hours}h blocks"
    print(f"population: {len(pop)} rows in {len(block_list)} blocks ({mode}) "
          f"from {args.input} [--population {args.population}]")
    if len(block_list) < len(pop):
        print(f"  effective sample size is the {len(block_list)} blocks, not the "
              f"{len(pop)} rows")
    print(f"wrote {len(records)} replicate(s) of {size} rows to {out_dir}")
    print(f"  manifest: {manifest_path}")
    cols = ["replicate", "n_draws", "n_unique_drawn", "unique_fraction", "n_oob"]
    with pd.option_context("display.max_rows", 20, "display.width", 200):
        print(manifest[cols].to_string(index=False))
    if len(records) > 1:
        uf = manifest["unique_fraction"].mean()
        if size >= len(pop):
            print(f"\nmean unique fraction: {uf:.4f} (expected ~0.632 at size == population)")
        else:
            # m-out-of-n: 0.632 does not apply, and n_oob is nearly the whole population, so
            # neither column means much. What matters is how much the members share.
            shared = manifest["n_draws"].sum() - len(
                set().union(*[set(pd.read_csv(r)["timestamp"]) for r in manifest["ar_index_path"]])
            )
            print(f"\nm-out-of-n draw: {size} rows from a {len(pop)}-row population, so the 0.632 "
                  f"unique fraction and the n_oob column do not apply here.")
            print(f"  duplicated rows within replicates: "
                  f"{int((manifest['n_draws'] - manifest['n_unique_drawn']).sum())} of "
                  f"{int(manifest['n_draws'].sum())} draws")
            print(f"  rows shared between replicates: {shared} "
                  f"(low means the members are near-independent)")


if __name__ == "__main__":
    main()
