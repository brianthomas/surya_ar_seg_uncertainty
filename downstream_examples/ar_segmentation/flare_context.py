"""Record the flare state of each inference frame, so member spread can be read against it.

The bootstrap ensemble's across-member spread on a fixed frame is the uncertainty
quantity of interest (see `/efs/bathomas/ar_seg_bootstrap/168hrs/README.md`). A flaring
region looks nothing like a quiescent one in AIA 94/131/335, and every member was
fine-tuned on 2011-2019 frames only -- so when spread turns out large on one frame and
small on another, flare state is the first covariate to check. This script writes it down
instead of leaving it to recollection.

Output keys on the `.nc` stem (e.g. `20240904_0700`), which is exactly the `file` column
`run_bootstrap_inference.py` writes into its `manifest.csv`, so the two join directly.

Three things make this less trivial than "query a flare list":

  * A holdout catalog row is a *HARP*, not a NOAA region, and these large ones enclose
    9-11 NOAA regions each. "Is this AR flaring" can only mean "did any constituent
    region flare", so the NOAA numbers come from the SHARP `NOAA_ARS` keyword.
  * GOES classification alone answers "no" on every frame we care about while AIA plainly
    shows emission -- it misses rise phases (it starts its clock later) and gradual
    phases (it closes its clock earlier). The AIA-based Flare Detective list is queried
    alongside it, and it is the one that changes the answer.
  * The SHARP bounding box is useless for locating anything in a HARP spanning 163 deg of
    longitude. Containment is tested against the actual AR mask pixels.

Example:
    PYTHONPATH=<repo root> python flare_context.py \
        --out /efs/bathomas/ar_seg_bootstrap/168hrs_infer_holdout_top3/flare_context.csv
"""

import argparse
import json
import math
import os
import re

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, get_earth

JSOC_SHOW_INFO = "http://jsoc.stanford.edu/cgi-bin/ajax/show_info"
HEK_SEARCH = "https://www.lmsal.com/hek/her"
SHARP_SERIES = "hmi.sharp_720s"

SHARP_KEYS = (
    "HARPNUM,T_REC,NOAA_AR,NOAA_ARS,NOAA_NUM,LAT_FWT,LON_FWT,"
    "LAT_MIN,LAT_MAX,LON_MIN,LON_MAX,AREA_ACR,USFLUX,QUALITY"
)
HEK_COLS = (
    "event_starttime,event_peaktime,event_endtime,fl_goescls,fl_peakflux,"
    "ar_noaanum,frm_name,hgs_x,hgs_y,obs_channelid,obs_instrument"
)

# Frames named by GOES class with a NOAA region attached. SWPC is the authoritative NOAA
# event list; SSW Latest Events duplicates most of it and occasionally disagrees on which
# region a flare belongs to, which is worth seeing rather than silently resolving.
GOES_FRAMES = ("SWPC", "SSW Latest Events")
# AIA-based detector, per-wavelength. Catches what GOES brackets out.
EUV_FRAME = "Flare Detective - Trigger Module"

# Geometry of both the level-1.5 `.nc` cubes and the AR mask rasters: north-up, disk
# centre at the array centre, 0.6 arcsec/px. Verified empirically -- see
# `probe_mask_orientation`. NOT the raw `hmi.m_720s` WCS, which carries CROTA2 ~ 180 and
# CDELT ~ 0.504; these products are de-rotated and replate-scaled.
MASK_CRPIX = 2048.5  # 1-based FITS reference pixel
MASK_CDELT = 0.6  # arcsec per pixel
MASK_SHAPE = (4096, 4096)
# Flare Detective positions are snapped to a coarse tile grid, so a single-pixel test is
# meaningless. +/- 30 px is +/- 18 arcsec.
PROBE_HALFWIDTH = 30

# A Flare Detective "event" can run 20+ hours in the 193/211 channels, which would count
# as spanning any instant inside it. Those are arcade/dimming signatures, not a flare in
# progress, so they are reported but excluded from the in-progress verdict.
MAX_EUV_DURATION_MIN = 240

GOES_LETTER_SCALE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
GOES_CLASS_RE = re.compile(r"^([ABCMX])([0-9.]*)$")

# Declared explicitly so a frame with no events still yields a DataFrame with these
# columns, and `summarize` can reach for them without special-casing emptiness.
EVENT_COLUMNS = (
    "frame", "frm_name", "channel", "goes_class", "goes_magnitude",
    "start", "peak", "end", "duration_min", "hours_from_frame",
    "ar_noaanum", "in_harp_by_noaa", "hgs_x", "hgs_y",
    "pixel_x", "pixel_y", "on_mask", "mask_frac", "in_harp_bbox",
    "in_harp", "spans_frame",
)

DEFAULT_CATALOG = "./assets/surya-bench-ar-segmentation/holdout_ar_large_from_test.csv"
DEFAULT_MASK_ROOT = "./assets/surya-bench-ar-segmentation"
DEFAULT_OUT = (
    "/efs/bathomas/ar_seg_bootstrap/168hrs_infer_holdout_top3/flare_context.csv"
)


class UpstreamError(RuntimeError):
    """A remote catalog (JSOC or HEK) could not be reached or returned nonsense.

    Raised rather than retried: nothing in the inference run depends on this script, so
    the right response to an outage is a clear message and a later rerun.
    """


def goes_magnitude(cls: str) -> float:
    """Return a GOES class as peak flux in W/m^2, or NaN if unparseable.

    Lets classes be compared and maxed across letters: C9.9 < M1.0 falls out of the
    arithmetic instead of needing a letter-then-number sort.
    """
    if not isinstance(cls, str):
        return math.nan
    m = GOES_CLASS_RE.match(cls.strip().upper())
    if not m:
        return math.nan
    letter, number = m.groups()
    return GOES_LETTER_SCALE[letter] * (float(number) if number else 1.0)


def goes_class_of(magnitude: float) -> str:
    """Inverse of `goes_magnitude`, for reporting a maximum back as a class string."""
    if not np.isfinite(magnitude):
        return ""
    for letter in ("X", "M", "C", "B", "A"):
        scale = GOES_LETTER_SCALE[letter]
        if magnitude >= scale or letter == "A":
            return f"{letter}{magnitude / scale:.1f}"
    return ""


def sharp_keywords(harpnum: int, t_rec: pd.Timestamp) -> dict:
    """Return the SHARP header keywords for one HARP at one T_REC.

    `t_rec` is interpreted as TAI, matching the holdout catalog's `peak_timestamp` --
    those values came from this series, and `LAT_FWT`/`LON_FWT` round-trip exactly.
    """
    query = f"{SHARP_SERIES}[{harpnum}][{t_rec:%Y.%m.%d_%H:%M:%S}_TAI]"
    try:
        response = requests.get(
            JSOC_SHOW_INFO, params={"ds": query, "key": SHARP_KEYS}, timeout=60
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UpstreamError(f"JSOC unreachable for {query}: {exc}") from exc

    # show_info emits unrelated warnings ("series ... not found") above the table, so the
    # header is located by its first column rather than assumed to be line 0.
    lines = [ln for ln in response.text.splitlines() if ln.strip()]
    header = next((i for i, ln in enumerate(lines) if ln.startswith("HARPNUM")), None)
    if header is None or len(lines) <= header + 1:
        raise UpstreamError(f"JSOC returned no record for {query}:\n{response.text}")

    keys = lines[header].split("\t")
    values = lines[header + 1].split("\t")
    return dict(zip(keys, values))


def parse_noaa_ars(raw: str) -> list[int]:
    """Parse the SHARP `NOAA_ARS` keyword into a list of NOAA region numbers."""
    if not raw or raw.strip() in ("", "0", "MISSING", "NONE"):
        return []
    return [int(tok) for tok in raw.split(",") if tok.strip().isdigit()]


def hek_flares(start: pd.Timestamp, end: pd.Timestamp, limit: int = 500) -> pd.DataFrame:
    """Return every HEK flare event overlapping [start, end], from all detection frames.

    HEK returns events that overlap the window rather than only those starting inside it,
    so a window wide enough to contain the frame is enough to catch a flare already under
    way when the frame was taken.
    """
    params = {
        "cosec": 2,
        "cmd": "search",
        "type": "column",
        "event_type": "fl",
        "event_starttime": f"{start:%Y-%m-%dT%H:%M:%S}",
        "event_endtime": f"{end:%Y-%m-%dT%H:%M:%S}",
        "event_coordsys": "helioprojective",
        "x1": -1200,
        "x2": 1200,
        "y1": -1200,
        "y2": 1200,
        "result_limit": limit,
        "return": HEK_COLS,
    }
    try:
        response = requests.get(HEK_SEARCH, params=params, timeout=120)
        response.raise_for_status()
        payload = json.loads(response.text)
    except (requests.RequestException, json.JSONDecodeError) as exc:
        raise UpstreamError(f"HEK unreachable or unparseable for {start}..{end}: {exc}") from exc

    if payload.get("overmax"):
        print(f"  WARNING: HEK hit its {limit}-row limit for {start}..{end}; "
              "events may be missing. Narrow --window_hours or raise --hek_limit.")

    frame = pd.DataFrame(payload.get("result", []))
    if frame.empty:
        return frame
    for column in ("event_starttime", "event_peaktime", "event_endtime"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("hgs_x", "hgs_y", "ar_noaanum", "fl_peakflux"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def world_to_pixel(lon: float, lat: float, obstime: Time) -> tuple[float, float]:
    """Project Stonyhurst (lon, lat) in degrees to 0-based (x, y) pixels in the raster.

    HeliographicStonyhurst takes no `observer` in sunpy 7.x -- the observer belongs to the
    Helioprojective frame being transformed into.
    """
    coord = SkyCoord(lon * u.deg, lat * u.deg, frame=HeliographicStonyhurst(obstime=obstime))
    hpc = coord.transform_to(Helioprojective(obstime=obstime, observer=get_earth(obstime)))
    x = (MASK_CRPIX - 1) + hpc.Tx.to_value(u.arcsec) / MASK_CDELT
    y = (MASK_CRPIX - 1) + hpc.Ty.to_value(u.arcsec) / MASK_CDELT
    return float(x), float(y)


def sample_mask(mask: np.ndarray, x: float, y: float) -> tuple[bool, float]:
    """Return (mask set at the pixel, positive fraction of the surrounding box).

    The fraction is the number to trust: Flare Detective positions are quantized, so a
    centre pixel can fall in a hole inside an otherwise solid region.
    """
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
        return False, math.nan
    half = PROBE_HALFWIDTH
    box = mask[
        max(0, yi - half) : yi + half + 1,
        max(0, xi - half) : xi + half + 1,
    ]
    return bool(mask[yi, xi] > 0), float((box > 0).mean())


def load_mask(mask_root: str, frame: pd.Timestamp) -> tuple[np.ndarray, str]:
    """Load the `union_with_intersect` AR mask for one frame.

    Stored 0/255 as uint8; `dataset.py` divides by 255 at load time. Comparisons here use
    `> 0` so the encoding does not matter.
    """
    path = os.path.join(mask_root, "data", f"{frame:%Y/%m/%Y%m%d_%H%M}.h5")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing AR mask {path}")
    with h5py.File(path, "r") as handle:
        return handle["union_with_intersect"][...], path


def probe_mask_orientation(mask: np.ndarray, lon: float, lat: float, obstime: Time) -> None:
    """Check the mask is north-up by probing a point known to sit inside an AR.

    This is the one assumption in the script that would fail quietly: a flipped raster
    still returns plausible-looking fractions, just for the wrong regions. The HARP's own
    flux-weighted centre is the natural probe -- except for HARPs so large that their
    centroid lands in the quiet gap between sub-regions, where *both* orientations read
    zero and the test simply cannot discriminate. Saying "confirmed" there would be a
    false reassurance, so that case reports itself as inconclusive.

    A high-latitude control point is probed too: it should read zero in either
    orientation, and a nonzero reading means the projection itself is off.
    """
    x, y = world_to_pixel(lon, lat, obstime)
    _, upright = sample_mask(mask, x, y)
    _, rotated = sample_mask(
        mask, 2 * (MASK_CRPIX - 1) - x, 2 * (MASK_CRPIX - 1) - y
    )
    _, control = sample_mask(mask, *world_to_pixel(0.0, 60.0, obstime))

    if max(upright, rotated) < 0.05:
        verdict = "INCONCLUSIVE: centroid is off-mask in both orientations"
    elif upright < rotated:
        verdict = "WARNING: looks rotated 180 deg -- check MASK_CRPIX/MASK_CDELT"
    else:
        verdict = "north-up confirmed"
    print(f"  orientation check: centre {upright:.3f} north-up / {rotated:.3f} rotated, "
          f"N60 control {control:.3f} -- {verdict}")
    if control > 0.05:
        print("  WARNING: high-latitude control point is on-mask; projection is suspect.")


def classify_events(
    events: pd.DataFrame,
    frame: pd.Timestamp,
    noaa_ars: list[int],
    mask: np.ndarray,
    bbox: dict,
) -> pd.DataFrame:
    """Annotate HEK events with HARP association, timing relative to the frame, and
    containment in the AR mask.

    Three association tests, because no single one is sufficient:

      * `in_harp_by_noaa` -- the event's NOAA region is one of the HARP's. Decisive when
        present. Matched modulo 10000, because HEK truncates five-digit region numbers to
        four (13807 is reported as 3807); some events carry 0 or null instead.
      * `on_mask` -- the event sits on a positive pixel of the AR mask. This says "inside
        *some* active region of the segmentation target", NOT "inside this HARP": the
        mask is full-disk and contains every AR with a PIL. On its own it will happily
        accept a flare from an unrelated region.
      * `in_harp_bbox` -- the event falls inside the SHARP lat/lon extent. Far too
        permissive alone (HARP 11786 spans 163 deg of longitude), but it excludes the
        unrelated-region case that `on_mask` lets through.

    `in_harp` combines them: trust the NOAA number when there is one, otherwise require
    both positional tests to agree.
    """
    obstime = Time(frame.to_pydatetime())
    harp_mod = {number % 10000 for number in noaa_ars}
    rows = []

    for _, event in events.iterrows():
        start, peak, end = (
            event["event_starttime"],
            event["event_peaktime"],
            event["event_endtime"],
        )
        timed = pd.notna(start) and pd.notna(end)
        duration = (end - start).total_seconds() / 60 if timed else math.nan
        noaa = event["ar_noaanum"]
        by_noaa = bool(np.isfinite(noaa) and noaa > 0 and int(noaa) % 10000 in harp_mod)

        on_mask, mask_frac = False, math.nan
        pixel_x, pixel_y = math.nan, math.nan
        in_bbox = False
        if np.isfinite(event["hgs_x"]) and np.isfinite(event["hgs_y"]):
            pixel_x, pixel_y = world_to_pixel(event["hgs_x"], event["hgs_y"], obstime)
            on_mask, mask_frac = sample_mask(mask, pixel_x, pixel_y)
            in_bbox = bool(
                bbox["lon_min"] <= event["hgs_x"] <= bbox["lon_max"]
                and bbox["lat_min"] <= event["hgs_y"] <= bbox["lat_max"]
            )

        has_noaa = bool(np.isfinite(noaa) and noaa > 0)
        in_harp = by_noaa if has_noaa else (on_mask and in_bbox)
        spans = bool(pd.notna(start) and pd.notna(end) and start <= frame <= end)
        rows.append(
            {
                "frame": f"{frame:%Y%m%d_%H%M}",
                "frm_name": event["frm_name"],
                "channel": event.get("obs_channelid", ""),
                "goes_class": event.get("fl_goescls", "") or "",
                "goes_magnitude": goes_magnitude(event.get("fl_goescls", "")),
                "start": start,
                "peak": peak,
                "end": end,
                "duration_min": duration,
                "hours_from_frame": (peak - frame).total_seconds() / 3600
                if pd.notna(peak)
                else math.nan,
                "ar_noaanum": noaa,
                "in_harp_by_noaa": by_noaa,
                "hgs_x": event["hgs_x"],
                "hgs_y": event["hgs_y"],
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "on_mask": on_mask,
                "mask_frac": mask_frac,
                "in_harp_bbox": in_bbox,
                "in_harp": in_harp,
                "spans_frame": spans,
            }
        )

    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def distinct_flares(goes: pd.DataFrame) -> pd.DataFrame:
    """Collapse SWPC/SSW duplicates of the same flare into one row.

    The two detection frames list most events twice, with peak times agreeing to the
    minute and classes agreeing to within a tenth. Counting both would report four
    M-flares as eight. Grouping on the rounded peak keeps the stronger of the two
    classifications; the event table retains both rows so attribution disagreements stay
    visible.
    """
    if goes.empty:
        return goes
    keyed = goes.assign(_peak_key=goes.peak.dt.round("2min"))
    return (
        keyed.sort_values("goes_magnitude", ascending=False)
        .drop_duplicates("_peak_key")
        .drop(columns="_peak_key")
    )


def summarize(events: pd.DataFrame, frame: pd.Timestamp, row: dict, mask: np.ndarray) -> dict:
    """Collapse one frame's annotated events into a single verdict row."""
    goes = events[events.frm_name.isin(GOES_FRAMES)]
    euv = events[events.frm_name == EUV_FRAME]

    in_harp = distinct_flares(goes[goes.in_harp])
    goes_live = in_harp[in_harp.spans_frame]
    # EUV events carry no NOAA number, so `in_harp` reduces to on-mask AND in-bbox there.
    euv_live = euv[
        euv.spans_frame & euv.in_harp & (euv.duration_min <= MAX_EUV_DURATION_MIN)
    ]

    # Nearest in-HARP GOES flare either side of the frame, for "how close did it come".
    nearest = (
        in_harp.loc[in_harp.hours_from_frame.abs().idxmin()]
        if not in_harp.empty
        else None
    )
    within_6h = in_harp[in_harp.hours_from_frame.abs() <= 6]

    return {
        "frame": f"{frame:%Y%m%d_%H%M}",
        "harpnum": row["harpnum"],
        "rank": row.get("rank"),
        "area_km2_million": row.get("area_km2_million"),
        "noaa_num": len(row["noaa_ars"]),
        "noaa_ars": " ".join(str(n) for n in row["noaa_ars"]),
        "goes_in_progress": bool(not goes_live.empty),
        "goes_in_progress_class": goes_class_of(goes_live.goes_magnitude.max())
        if not goes_live.empty
        else "",
        "euv_on_mask_in_progress": bool(not euv_live.empty),
        "euv_channels_in_progress": " ".join(
            sorted({str(c) for c in euv_live.channel if str(c)})
        ),
        "nearest_goes_class": nearest["goes_class"] if nearest is not None else "",
        "nearest_goes_hours": round(nearest["hours_from_frame"], 2)
        if nearest is not None
        else math.nan,
        "max_goes_class_in_window": goes_class_of(in_harp.goes_magnitude.max())
        if not in_harp.empty
        else "",
        "n_m_class_pm_6h": int(
            (within_6h.goes_magnitude >= GOES_LETTER_SCALE["M"]).sum()
        ),
        "n_goes_events_in_window": int(len(in_harp)),
        "mask_pos_frac": round(float((mask > 0).mean()), 5),
    }


def frames_from_catalog(path: str, top: int) -> list[dict]:
    """Return the `top` largest ARs from a holdout catalog, by deprojected area."""
    catalog = pd.read_csv(path)
    catalog = catalog.sort_values("area_km2_million", ascending=False).head(top)
    return [
        {
            "frame": pd.Timestamp(row.peak_timestamp),
            "harpnum": int(row.harpnum),
            "rank": int(row.rank) if hasattr(row, "rank") else None,
            "area_km2_million": float(row.area_km2_million),
        }
        for row in catalog.itertuples()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG,
                        help="Holdout catalog to take frames from.")
    parser.add_argument("--top", type=int, default=3,
                        help="How many of the largest ARs to process.")
    parser.add_argument("--frames", nargs="+", default=None,
                        help="Explicit frame timestamps (YYYY-MM-DDTHH:MM), bypassing "
                             "--catalog. Requires --harpnums.")
    parser.add_argument("--harpnums", nargs="+", type=int, default=None,
                        help="HARPNUM per --frames entry.")
    parser.add_argument("--window_hours", type=float, default=12.0,
                        help="Report events within this many hours either side.")
    parser.add_argument("--mask_root", default=DEFAULT_MASK_ROOT,
                        help="Root holding the AR benchmark's data/ tree.")
    parser.add_argument("--hek_limit", type=int, default=500,
                        help="HEK result cap per query.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Summary CSV path; the event table gets an _events suffix.")
    args = parser.parse_args()

    if args.frames:
        if not args.harpnums or len(args.harpnums) != len(args.frames):
            raise SystemExit("--frames requires one --harpnums value per frame")
        rows = [
            {"frame": pd.Timestamp(f), "harpnum": h, "rank": None,
             "area_km2_million": math.nan}
            for f, h in zip(args.frames, args.harpnums)
        ]
    else:
        rows = frames_from_catalog(args.catalog, args.top)

    window = pd.Timedelta(hours=args.window_hours)
    summaries, event_tables = [], []

    for row in rows:
        frame = row["frame"]
        print(f"\n=== {frame:%Y-%m-%d %H:%M}  HARP {row['harpnum']} ===")

        keywords = sharp_keywords(row["harpnum"], frame)
        row["noaa_ars"] = parse_noaa_ars(keywords.get("NOAA_ARS", ""))
        lat_fwt = float(keywords["LAT_FWT"])
        lon_fwt = float(keywords["LON_FWT"])
        bbox = {
            "lat_min": float(keywords["LAT_MIN"]),
            "lat_max": float(keywords["LAT_MAX"]),
            "lon_min": float(keywords["LON_MIN"]),
            "lon_max": float(keywords["LON_MAX"]),
        }
        print(f"  NOAA_ARS ({keywords.get('NOAA_NUM')}): "
              f"{' '.join(str(n) for n in row['noaa_ars'])}")
        print(f"  flux-weighted centre: lat {lat_fwt:+.2f} lon {lon_fwt:+.2f}  "
              f"QUALITY {keywords.get('QUALITY')}")
        print(f"  extent: lat {bbox['lat_min']:+.1f}..{bbox['lat_max']:+.1f}  "
              f"lon {bbox['lon_min']:+.1f}..{bbox['lon_max']:+.1f}")

        mask, mask_path = load_mask(args.mask_root, frame)
        print(f"  mask {mask_path}  positive fraction {(mask > 0).mean():.4f}")
        probe_mask_orientation(mask, lon_fwt, lat_fwt, Time(frame.to_pydatetime()))

        raw = hek_flares(frame - window, frame + window, args.hek_limit)
        if raw.empty:
            print("  no HEK flare events in window")
            annotated = pd.DataFrame(columns=list(EVENT_COLUMNS))
        else:
            annotated = classify_events(raw, frame, row["noaa_ars"], mask, bbox)
            event_tables.append(annotated)

        summary = summarize(annotated, frame, row, mask)
        summaries.append(summary)
        live_class = summary["goes_in_progress_class"]
        print(f"  GOES in progress: {summary['goes_in_progress']}"
              f"{f' ({live_class})' if live_class else ''}"
              f"   nearest {summary['nearest_goes_class'] or 'none'} "
              f"at {summary['nearest_goes_hours']:+.2f} h")
        print(f"  EUV on-mask in progress: {summary['euv_on_mask_in_progress']}"
              f"   channels: {summary['euv_channels_in_progress'] or 'none'}")
        print(f"  window max {summary['max_goes_class_in_window'] or 'none'}, "
              f"{summary['n_m_class_pm_6h']} M+ within 6 h")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out, index=False)
    print(f"\nWrote {len(summaries)} summary rows to {args.out}")

    if event_tables:
        events_out = re.sub(r"\.csv$", "", args.out) + "_events.csv"
        combined = pd.concat(event_tables, ignore_index=True).sort_values(
            ["frame", "peak"]
        )
        combined.to_csv(events_out, index=False)
        print(f"Wrote {len(combined)} event rows to {events_out}")


if __name__ == "__main__":
    # A JSOC or HEK outage is an expected operational condition, not a bug worth a
    # traceback -- nothing in the inference run depends on this script, so the useful
    # response is a one-line message and a rerun later.
    try:
        main()
    except UpstreamError as error:
        raise SystemExit(f"Remote catalog unavailable: {error}")
