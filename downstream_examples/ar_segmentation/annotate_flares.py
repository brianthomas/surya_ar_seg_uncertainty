#!/usr/bin/env python
"""Annotate an AR catalog with flare activity, and optionally select the quiet ones.

`make_holdout_catalog.py` ranks ARs by size but says nothing about whether they flared.
Ensemble spread on a large flaring AR cannot be attributed to either property alone, so
building a control set means being able to ask "which of these never flared". This adds
that column, and can cut the catalog down to the N quietest.

Two independent measures of activity are recorded, because neither alone is sufficient:

  * `n_flares` / `max_goes_class` -- flares with a GOES class, attributed to one of the
    AR's NOAA regions. This drives the non-flaring verdict: it is the conventional
    definition and the one a reader will assume.
  * `n_euv_events` -- AIA Flare Detective detections falling inside the AR's bounding
    box. GOES misses rise and gradual phases (see flare_context.py's docstring), so an AR
    with no GOES flare but many EUV events is not actually quiescent. Recorded rather
    than used, unless --require-euv-quiet is passed.

The two are matched differently out of necessity: Flare Detective rows carry **no**
`ar_noaanum` and no GOES class at all -- only `hgs_x`/`hgs_y` (verified against
2024-10-01: 132 of 132 EUV rows have positions, 0 have a NOAA number). They are therefore
matched positionally against the HARP's *hourly* bounding box track, which also handles
the AR rotating ~13 deg/day across the disk during a 7-16 day lifetime. A single fixed box
would be wrong by most of a solar radius by the end.

Example:
    PYTHONPATH=<repo root> python annotate_flares.py \\
        --catalog band.csv --select-nonflaring 30 --out holdout_nonflaring.csv
"""

import argparse
import os

import pandas as pd
import requests

from flare_context import (
    GOES_FRAMES,
    EUV_FRAME,
    JSOC_SHOW_INFO,
    MAX_EUV_DURATION_MIN,
    SHARP_SERIES,
    UpstreamError,
    goes_magnitude,
    hek_flares,
    parse_noaa_ars,
)

# HEK returns events overlapping the window, so chunks double-count across boundaries;
# these fields identify an event for dedup.
EVENT_KEY = ("frm_name", "event_starttime", "event_endtime", "hgs_x", "hgs_y",
             "fl_goescls", "ar_noaanum")
CACHE_COLUMNS = ("event_starttime", "event_peaktime", "event_endtime", "fl_goescls",
                 "ar_noaanum", "frm_name", "hgs_x", "hgs_y")

# Flare Detective positions are snapped to a coarse tile grid, so the bounding-box test
# needs slack. 2 deg is roughly the +/-30 px used for the pixel probe in flare_context.
EUV_MARGIN_DEG = 2.0

ANNOTATION_COLUMNS = ("lifetime_start", "lifetime_end", "noaa_regions", "n_flares",
                      "max_goes_class", "n_euv_events", "flaring_at_peak")


def build_flare_cache(path: str, start: pd.Timestamp, end: pd.Timestamp,
                      chunk_days: int, limit: int, refresh: bool) -> pd.DataFrame:
    """Return every HEK flare event in [start, end], fetching only uncached chunks.

    Queried once for the whole span rather than per AR: candidate lifetimes overlap
    heavily, so per-AR querying repeats the same windows. Chunks are cached by their start
    so an interrupted run resumes instead of refetching hundreds of windows.
    """
    chunk_starts = pd.date_range(start, end, freq=f"{chunk_days}D")
    cached = pd.DataFrame()
    if os.path.isfile(path) and not refresh:
        cached = pd.read_csv(path, parse_dates=["event_starttime", "event_peaktime",
                                                "event_endtime", "chunk_start"])
        have = set(cached["chunk_start"])
        todo = [c for c in chunk_starts if c not in have]
        print(f"flare cache: {len(cached)} events, {len(todo)} of "
              f"{len(chunk_starts)} chunks still to fetch")
    else:
        todo = list(chunk_starts)
        print(f"flare cache: building from scratch, {len(todo)} chunks")

    fetched = []

    def flush() -> pd.DataFrame:
        """Persist what has been fetched so far, so the run is resumable."""
        nonlocal cached, fetched
        if fetched:
            cached = pd.concat([cached] + fetched, ignore_index=True)
            cached.to_csv(path, index=False)
            fetched = []
        return cached

    for i, chunk in enumerate(todo, 1):
        upper = min(chunk + pd.Timedelta(days=chunk_days), end)
        try:
            frame = hek_flares(chunk, upper, limit=limit)
        except UpstreamError as exc:
            print(f"  [{i}/{len(todo)}] {chunk:%Y-%m-%d} FAILED: {exc}")
            flush()
            raise SystemExit(
                f"HEK fetch failed at {chunk:%Y-%m-%d}; {len(cached)} events cached. "
                "Rerun the same command to resume from here."
            )
        if len(frame):
            frame = frame.reindex(columns=list(CACHE_COLUMNS))
            frame["chunk_start"] = chunk
            fetched.append(frame)
        # Checkpointed rather than written once at the end: this loop is hundreds of
        # requests, and losing all of them to an interrupt would be painful.
        if i % 10 == 0 or i == len(todo):
            total = len(flush())
            print(f"  [{i}/{len(todo)}] through {chunk:%Y-%m-%d}, {total} events cached")

    flush()
    return cached


def dedup_events(flares: pd.DataFrame) -> pd.DataFrame:
    """Drop events duplicated by overlapping chunk windows."""
    if flares.empty:
        return flares
    return flares.drop_duplicates(subset=[c for c in EVENT_KEY if c in flares.columns])


def harp_tracks(harp_catalog: str, harpnums: set) -> dict:
    """Return each HARP's hourly bounding-box track, keyed by harpnum.

    The track is what makes positional EUV matching possible: the box follows the AR
    across the disk instead of pinning it to wherever it was at peak.
    """
    cat = pd.read_csv(
        harp_catalog, na_values=["MISSING"],
        usecols=["timestamp", "harpnum", "lat_min", "lat_max", "lon_min", "lon_max"],
    )
    cat = cat[cat["harpnum"].isin(harpnums)].copy()
    cat["timestamp"] = pd.to_datetime(cat["timestamp"])
    return {h: g.sort_values("timestamp") for h, g in cat.groupby("harpnum")}


def noaa_regions_for_harp(harpnum: int, start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Return every NOAA region the HARP carried at any point in its lifetime.

    Sampled daily across the whole track, not read at one instant: a HARP can acquire or
    lose its NOAA number mid-track, and a region number present on only some days still
    means flares can be attributed to it.
    """
    query = (f"{SHARP_SERIES}[{harpnum}]"
             f"[{start:%Y.%m.%d_%H:%M:%S}_TAI-{end:%Y.%m.%d_%H:%M:%S}_TAI@1d]")
    try:
        response = requests.get(
            JSOC_SHOW_INFO, params={"ds": query, "key": "HARPNUM,T_REC,NOAA_ARS"},
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UpstreamError(f"JSOC unreachable for {query}: {exc}") from exc

    regions = set()
    for line in response.text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].strip().isdigit():
            regions |= set(parse_noaa_ars(parts[2]))
    # HEK truncates the leading 1 on some region numbers and not others -- 2024-10-01
    # returns both 3839 and 13839 -- so comparison has to happen modulo 10000.
    return {r % 10000 for r in regions}


def goes_flares_for_regions(flares: pd.DataFrame, regions: set,
                            start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return GOES-classified flares attributed to `regions` within the lifetime."""
    if flares.empty or not regions:
        return flares.iloc[:0]
    hit = flares[
        flares["frm_name"].isin(GOES_FRAMES)
        & flares["ar_noaanum"].notna()
        & flares["fl_goescls"].fillna("").ne("")
        & flares["event_starttime"].between(start, end)
    ].copy()
    if hit.empty:
        return hit
    return hit[hit["ar_noaanum"].astype(int).mod(10000).isin(regions)]


def euv_events_in_box(flares: pd.DataFrame, track: pd.DataFrame,
                      margin: float) -> pd.DataFrame:
    """Return Flare Detective events falling inside the HARP's box at their own time.

    Long-duration detections are excluded: a 20-hour event in 193/211 is an arcade or
    dimming signature rather than a flare, the same exclusion flare_context applies.
    """
    if flares.empty or track is None or track.empty:
        return flares.iloc[:0]
    euv = flares[
        (flares["frm_name"] == EUV_FRAME)
        & flares["hgs_x"].notna()
        & flares["hgs_y"].notna()
        & flares["event_starttime"].between(track["timestamp"].min(),
                                            track["timestamp"].max())
    ].copy()
    if euv.empty:
        return euv

    duration = (euv["event_endtime"] - euv["event_starttime"]).dt.total_seconds() / 60.0
    euv = euv[duration.fillna(0) <= MAX_EUV_DURATION_MIN]
    if euv.empty:
        return euv

    boxes = track.dropna(subset=["lat_min", "lat_max", "lon_min", "lon_max"])
    if boxes.empty:
        return euv.iloc[:0]
    # Snap each event to the nearest tracked hour, then test that hour's box.
    nearest = pd.merge_asof(
        euv.sort_values("event_starttime"), boxes, left_on="event_starttime",
        right_on="timestamp", direction="nearest",
    )
    inside = (
        nearest["hgs_y"].between(nearest["lat_min"] - margin, nearest["lat_max"] + margin)
        & nearest["hgs_x"].between(nearest["lon_min"] - margin,
                                   nearest["lon_max"] + margin)
    )
    return nearest[inside]


def annotate(catalog: pd.DataFrame, flares: pd.DataFrame, tracks: dict,
             margin: float) -> pd.DataFrame:
    """Add the flare columns to every row of `catalog`."""
    rows = []
    for n, (_, row) in enumerate(catalog.iterrows(), 1):
        harpnum = int(row["harpnum"])
        track = tracks.get(harpnum)
        if track is None or track.empty:
            raise SystemExit(f"HARP {harpnum} has no rows in the HARP catalog")
        start, end = track["timestamp"].min(), track["timestamp"].max()

        regions = noaa_regions_for_harp(harpnum, start, end)
        goes = goes_flares_for_regions(flares, regions, start, end)
        euv = euv_events_in_box(flares, track, margin)

        peak = pd.Timestamp(row["peak_timestamp"])
        in_progress = False
        if len(goes):
            spans = goes.dropna(subset=["event_starttime", "event_endtime"])
            in_progress = bool(
                ((spans["event_starttime"] <= peak) & (spans["event_endtime"] >= peak)).any()
            )

        classes = [c for c in goes["fl_goescls"].dropna() if c] if len(goes) else []
        rows.append({
            "harpnum": harpnum,
            "lifetime_start": f"{start:%Y-%m-%d %H:%M:%S}",
            "lifetime_end": f"{end:%Y-%m-%d %H:%M:%S}",
            # Semicolon-separated: a comma would need quoting in the CSV.
            "noaa_regions": ";".join(str(r) for r in sorted(regions)),
            "n_flares": len(goes),
            "max_goes_class": max(classes, key=goes_magnitude) if classes else "",
            "n_euv_events": len(euv),
            "flaring_at_peak": in_progress,
        })
        if n % 25 == 0 or n == len(catalog):
            print(f"  annotated {n}/{len(catalog)}")

    return catalog.merge(pd.DataFrame(rows), on="harpnum", how="left")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--catalog", required=True,
                   help="Catalog CSV from make_holdout_catalog.py.")
    p.add_argument("--harp-catalog", default="./assets/harp_catalog_subset.csv",
                   help="Hourly HARP bounding-box geometry, for lifetimes and EUV boxes.")
    p.add_argument("--flare-cache", default="./assets/hek_flares.csv",
                   help="Where HEK events are cached between runs.")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Refetch every chunk instead of reusing the cache.")
    p.add_argument("--chunk-days", type=int, default=5,
                   help="HEK query window. 5 days is ~800 rows at solar maximum.")
    p.add_argument("--hek-limit", type=int, default=3000,
                   help="Per-query row cap; a warning is printed if it is reached.")
    p.add_argument("--euv-margin-deg", type=float, default=EUV_MARGIN_DEG,
                   help="Slack on the bounding-box test, for grid-snapped EUV positions.")
    p.add_argument("--select-nonflaring", type=int, default=0,
                   help="Keep the N quietest ARs, largest first. 0 annotates only.")
    p.add_argument("--require-noaa", action=argparse.BooleanOptionalAction, default=True,
                   help="Require a NOAA region, so quiescence is verified not merely "
                        "unattributable.")
    p.add_argument("--require-euv-quiet", action="store_true",
                   help="Also require zero EUV detections, not just zero GOES flares.")
    p.add_argument("--allow-below-class", default=None,
                   help="Count an AR as non-flaring if its largest flare is below this "
                        "GOES class, e.g. C1.0 to tolerate B-class microflares. By "
                        "default any classified flare disqualifies it. Pair with "
                        "--max-flares: class alone admits ARs with dozens of microflares.")
    p.add_argument("--max-flares", type=int, default=None,
                   help="Also cap how many flares an AR may have produced. Only "
                        "meaningful alongside --allow-below-class.")
    p.add_argument("--require-quiet-at-peak", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Exclude ARs with a flare in progress at the peak timestamp -- "
                        "the frame inference actually runs on.")
    p.add_argument("--from-annotated", action="store_true",
                   help="Treat --catalog as an already-annotated CSV and only re-run "
                        "selection. Skips all network access.")
    p.add_argument("--annotated-out", default=None,
                   help="Also write the fully annotated catalog before selection.")
    p.add_argument("--out", required=True, help="Where the result CSV is written.")
    args = p.parse_args()

    catalog = pd.read_csv(args.catalog)
    print(f"{args.catalog}: {len(catalog)} ARs")

    if args.from_annotated:
        missing = [c for c in ANNOTATION_COLUMNS if c not in catalog.columns]
        if missing:
            raise SystemExit(f"--from-annotated but {args.catalog} lacks {missing}")
        annotated = catalog
    else:
        annotated = annotate_from_sources(catalog, args)
    select_and_write(annotated, args)


def annotate_from_sources(catalog: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Fetch flare data and annotate `catalog`, writing --annotated-out if asked."""
    tracks = harp_tracks(args.harp_catalog, set(catalog["harpnum"].astype(int)))
    spans = [(t["timestamp"].min(), t["timestamp"].max()) for t in tracks.values()]
    if not spans:
        raise SystemExit("no catalog HARPs found in the HARP catalog")
    # Pad so a flare starting just before the earliest AR appears is still caught.
    start = min(s for s, _ in spans).floor("D") - pd.Timedelta(days=1)
    end = max(e for _, e in spans).ceil("D") + pd.Timedelta(days=1)
    print(f"lifetimes span {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    flares = dedup_events(build_flare_cache(
        args.flare_cache, start, end, args.chunk_days, args.hek_limit, args.refresh_cache
    ))
    print(f"{len(flares)} distinct HEK events after dedup")

    annotated = annotate(catalog, flares, tracks, args.euv_margin_deg)
    if args.annotated_out:
        annotated.to_csv(args.annotated_out, index=False)
        print(f"wrote {args.annotated_out}: {len(annotated)} ARs")
    return annotated


def select_and_write(annotated: pd.DataFrame, args: argparse.Namespace) -> None:
    """Apply the non-flaring criteria and write --out."""
    result = annotated
    if args.select_nonflaring:
        if args.allow_below_class:
            ceiling = goes_magnitude(args.allow_below_class)
            largest = result["max_goes_class"].fillna("").map(
                lambda c: goes_magnitude(c) if c else 0.0
            )
            quiet = largest < ceiling
        else:
            quiet = result["n_flares"] == 0
        if args.max_flares is not None:
            quiet &= result["n_flares"] <= args.max_flares
        if args.require_quiet_at_peak:
            quiet &= ~result["flaring_at_peak"].fillna(False).astype(bool)
        if args.require_noaa:
            quiet &= result["noaa_regions"].fillna("").astype(str) != ""
        if args.require_euv_quiet:
            quiet &= result["n_euv_events"] == 0
        eligible = result[quiet].sort_values("rank")
        if len(eligible) < args.select_nonflaring:
            raise SystemExit(
                f"only {len(eligible)} ARs meet the non-flaring criteria, "
                f"{args.select_nonflaring} requested"
            )
        print(f"{len(eligible)} eligible; taking the {args.select_nonflaring} largest")
        result = eligible.head(args.select_nonflaring)

    result.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(result)} ARs")
    if len(result):
        print(f"  area_km2_million {result.area_km2_million.min():.0f}"
              f"-{result.area_km2_million.max():.0f}, "
              f"ranks {result['rank'].min()}-{result['rank'].max()}")
        print(f"  with zero EUV detections too: "
              f"{int((result.n_euv_events == 0).sum())}/{len(result)}")


if __name__ == "__main__":
    main()
