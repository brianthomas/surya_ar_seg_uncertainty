# Known issue: 8 input channels are offset from the other 5 and from the AR masks

**Status:** suspected data bug in the released core-sdo cubes; correction implemented and
off by default (`data.fix_corner_scaling: false`) until a summer cube confirms its sign.
Runs made before this note (including every bootstrap and seed-ablation run) used the
uncorrected inputs.

Longer write-up with sources: [Surya core-sdo channel offset: corner-anchored disk rescaling](https://docs.google.com/document/d/1qlDvsl9SQn1I631GTlWqBrc4fj1K9V-BhQ1uM12LLZg/edit).

## Symptom

In the 13-channel cubes, these 8 channels are shifted along the image diagonal relative to
the other 5 channels and to the AR segmentation masks:

| Offset group | Aligned group (and AR masks) |
|---|---|
| aia94, aia131, aia171, aia193, aia211, aia304, aia335, hmi_m | aia1600, hmi_bx, hmi_by, hmi_bz, hmi_v |

The offset changes through the year: about 0 px in January and about 70 px per axis in July.

## Cause (inferred)

The core-sdo pipeline rescales each image so the solar disk has a fixed radius of 976″,
with scale factor `s = 976 / RSUN_OBS`. Scaling about a point `p` moves the disk center by
`(s − 1)(c − p)`, where `c = 2048` is the image center. Scaling about the center gives no
shift; scaling about array index (0, 0) gives `(s − 1) · 2048` px on both axes.

976″ is almost exactly the Sun's angular radius at perihelion, so `s ≈ 1` in early
January and `s ≈ 1.034` at aphelion in early July. The predicted shift under corner
anchoring matches the observation:

| Date | RSUN_OBS (″) | Shift per axis (px) |
|---|---|---|
| 3 Jan | 975.9 | 0 |
| 1 Mar | 968.5 | 16 |
| 1 Apr | 960.3 | 34 |
| 1 May | 952.4 | 51 |
| 1 Jun | 946.3 | 64 |
| 4 Jul | 943.9 | 70 |
| 1 Aug | 945.5 | 66 |
| 1 Sep | 950.9 | 54 |
| 1 Oct | 958.6 | 37 |
| 1 Nov | 966.9 | 19 |
| 1 Dec | 973.2 | 6 |

The public SuryaBench `core_sdo/helio.py` (`scale_solardisk`) scales about the center, but it
was committed on 2025-11-06, after the data were produced. The cubes in this repo carry
`production_date` 2025-05-22 and different author attributes, so they came from an earlier
version of the pipeline. The split does not follow instrument lines (hmi_m is in the offset
group, aia1600 is not), which suggests the channels were processed in separate batches.

## What has and hasn't been verified

- Each channel carries `RSUN_OBS` in its `meta_1` JSON attribute, so the correction can be
  computed per sample from the cube itself. `crpix1` is 2048.5 for every channel, so the
  headers do not reveal which channels are shifted.
- In the local January cubes (`data/Surya-1.0_validation_data/20140107_*.nc`), all channels
  are centered to within a few px, as predicted for January.
- On a January cube corner-scaled synthetically to July (s = 1.034), `check_disk_alignment.py`
  measures about 70 px on both axes for the scaled channels and under 2 px after the correction.
- **Not yet verified:** the size and sign of the shift on a real summer cube.

## Effect on this project

- **Pretraining.** Surya reads these cubes directly, so the backbone learned cross-channel
  relationships with a seasonally varying offset of up to ~70 px between the two groups.
- **AR segmentation labels.** The masks align with hmi_bz and the other aligned channels,
  but not with hmi_m or the EUV channels. Without the correction, supervision is
  inconsistent in a way that depends on the time of year, which would show up in the
  uncertainty estimates as apparent data uncertainty.
- **February 2013 runs** (`config_feb1*_2013*.yaml`): the predicted mid-February offset is
  about 8-10 px, so these change little. Summer samples change most.

## The correction

If the shifted image is `I_bad(y) = I0(y / s)`, the centered image is
`I_good(x) = I_bad(x + d)` with `d = (s − 1) · 2048` on both axes. That is a pure
translation, applied to the raw 4096² arrays before scaling or pooling, with zero fill at
the vacated edge. The original warp used nearest-neighbour interpolation, so an integer
shift is as precise as the data allow.

Code: `surya/datasets/disk_alignment.py` (`undo_corner_scaling`, `rsun_obs_from_attrs`,
`corner_scaling_kwargs`). It is applied in:

- `HelioNetCDFDataset` (`surya/datasets/helio.py`), which covers `finetune.py` (through
  `ArDSDataset`) and `infer.py`;
- `infer_file.read_hc_file`, which covers `infer_file.py` (also `--fix_corner_scaling`)
  and `run_bootstrap_inference.py` (which follows each member's own config).

Config, in the `data:` block:

```yaml
fix_corner_scaling: false      # true recenters the 8 channels on load
# corner_scaling_channels: [...]   # defaults to the 8 channels above
# corner_scaling_sign: 1           # -1 only if the check shows the shift runs the other way
```

Configs without these keys behave as before.

## Before turning it on

1. Run the check on one summer cube from a machine with bucket access:

   ```bash
   cd downstream_examples/ar_segmentation
   python check_disk_alignment.py s3://nasa-surya-bench/2011/07/20110701_0000.nc --s3_anon
   ```

   Expect about 70 px on both axes for the 8 channels, about 0 for the other 5, and
   residuals near 0. hmi_m against hmi_bz is the cleanest single row; the limb fit for
   aia171, aia193 and aia211 can be off by 10-40 px and is flagged in the output.
2. If the residuals come out at about twice the prediction instead, rerun with
   `--sign -1`; if that fixes it, set `corner_scaling_sign: -1`.
3. Set `fix_corner_scaling: true` for new runs, and use the same setting for training,
   validation and inference. A checkpoint trained without the correction should be
   evaluated without it.

A corrected vs. uncorrected comparison on a summer holdout would be the clean ablation.

## Tests

`tests/test_disk_alignment.py` (synthetic disks; the loader test needs the project
environment with torch).
