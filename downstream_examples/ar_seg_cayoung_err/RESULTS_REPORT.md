# Active-Region Segmentation: Results Report

**Status as of 2026-09-17 09:26.** The final (`real`-budget) run is in progress: seed 0 of 5 is
at epoch 10, still improving. Everything below marked *final* is from completed runs; the
`real` section is a live snapshot and will be superseded by the run's own output.

This report gathers what was measured across the four notebooks in
`downstream_examples/ar_seg_cayoung_err/` — the baseline ladder, the Surya fine-tune, and the
error analysis — for readers who want the numbers and what they mean without re-running
anything. The mechanics are documented in `NOTEBOOKS_EXPLAINED.md`,
`ERRORS_NOTEBOOK_EXPLAINED.md` and `GROUND_TRUTH_EXPLAINED.md`.

---

## 1. Executive summary

| finding | value |
|---|---|
| Surya + adapters, 400 training images, epoch 10 (seed 0, in progress) | **IoU 0.637** — above the 0.632 label floor measured on the medium run |
| Surya + adapters, 60 training images (final medium run) | IoU 0.437, 95 % CI [0.424, 0.449]; 0.465 at the best calibrated threshold |
| Seed-to-seed spread vs sampling uncertainty (medium) | 0.025 vs 0.021 — comparable; a single run is not quotable to three decimals |
| Label floor: agreement between ±40 G and ±60 G versions of the answer key | **0.632** (medium validation set) |
| Calibration | raw scores over-confident (scale 0.46, shift −1.98); stated-vs-observed gap 0.0098 → 0.0015 after correction |
| Strongest predictor of per-image score | distance of the labelled regions from disk centre, r = −0.94 (solar rotation + line-of-sight foreshortening) |
| First run of the error notebook | collapsed to predicting nothing; three causes found and fixed |

The short version: with enough data the foundation model reaches the point where its
disagreement with the answer key is the same size as the answer key's disagreement with
itself. Below that point, model error dominated; at it, further gains on pixel overlap
cannot be distinguished from a different choice of threshold in the labelling rule.

---

## 2. What was measured, and how

Every model is scored with the same code (`ARMetrics`) on the same pairs of SDO image and
hourly active-region outline. Scores are **overlap** (IoU, Dice), never accuracy — active
regions cover ~1 % of the frame, so "background everywhere" is 99 % accurate and useless.
The error notebook adds, for each score:

- a **bootstrap interval** over validation images (sampling uncertainty);
- a **calibration** of the model's probabilities, fitted on a random half of the validation
  set and reported on the other half;
- **confidence contours** at 0.25 / 0.5 / 0.75 on the calibrated map;
- an **ensemble** of independently seeded adapter sets (training uncertainty);
- a **label floor**: the answer key re-derived at ±40 G and ±60 G, and the agreement between
  those two equally defensible keys.

---

## 3. The collapse, and its causes

The first run of `3_errors_ar.ipynb` (8 images, 2 epochs) produced a textbook failure:

| epoch | loss | IoU | precision | recall |
|---|---|---|---|---|
| 0 | 0.021 | 0.0054 | 0.71 | 0.0054 |
| 1 | 0.013 ↓ | **0.000035** | 1.00 | 0.000035 |

The loss improved while the model learned to predict nothing. Three causes, all now handled
in the notebook:

| cause | evidence | fix |
|---|---|---|
| The images were the quietest in the archive | a naive date cap selected mid-January 2011: 0.101 % AR coverage, 987 background pixels per AR pixel | date-window selection (solar maximum, 2013–2015): 0.9–1.6 % coverage, up to 16× the signal per image |
| The rare pixels were not up-weighted | `ar_pos_weight: null`; the penalty for "predict nothing" was 0.24 | weight measured from the selected outlines (√ of the imbalance); with a Dice term the same prediction costs 8.4 |
| Learning rate 10× too high for Surya | the config's own comment said 1e-4; the notebook passed 1e-3 | rate chosen per model |

The diagnostic that exposed it — and the one to watch on any run — is **precision and
recall together**, not the loss.

---

## 4. Medium budget: 60 training images, 60 validation, 2 seeds

Two runs. The first used the last epoch's weights, a disc-shaped dilation in the label
floor, and a chronological calibration split; the second corrected all three (§6, §7).

| quantity | provisional | **final** |
|---|---|---|
| seed 0, IoU at 0.5, pooled over 60 images | 0.427 [0.411, 0.442] | **0.437 [0.424, 0.449]** |
| calibration scale *a*, shift *b* | 0.443, −0.915 | 0.459, −1.975 |
| stated-vs-observed probability gap, before → after | 0.0077 → 0.0031 | 0.0098 → **0.0015** |
| evaluation half, uncalibrated IoU | 0.369 (confounded) | **0.444** (matches the full set) |
| best calibrated threshold → IoU | 0.12 → 0.381 | 0.28 → **0.465** |
| seeds 0 / 1 at that threshold | 0.402 / 0.383 | **0.459 / 0.417** |
| seed spread vs one seed's bootstrap width | 0.010 vs 0.032 | **0.025 vs 0.021** |
| answer-key reconstruction agreement (60 images) | 60/60 ≥ 0.6, median 0.826 | 60/60, median 0.813 |
| **label floor** | 0.639 | **0.632** |
| verdict | headroom | best 0.459 < 0.632: model error dominates |

### Training curves at 60 images

Both seeds reached their lowest validation loss at **epoch 1 of 10** and then overfit: seed 0
val loss 0.533 → 0.652 by epoch 9 while IoU stayed flat at ~0.43. Recall drifted 0.82 → 0.60
and precision 0.44 → 0.58 — toward under-prediction, never collapse. The lesson taken into
the `real` run: more *images*, not more epochs, plus early stopping and keeping the best
epoch's weights.

### What the ensemble showed

With the best-epoch weights the two seeds differ by 0.025 in IoU, about the width of one
seed's bootstrap interval. Training randomness and sampling uncertainty are the same size at
this budget; the honest report is the pair (0.459, 0.417) or a mean with a spread.

---

## 5. The limb effect: what actually drives per-image score

Per-image IoU on the medium validation set fell from 0.53 on the first images to 0.36 on the
last. The validation images are chronological (three days, 2013-01-16 → 01-18), and:

| per-image IoU correlates with | r |
|---|---|
| mean distance of the labelled regions from disk centre | **−0.94** |
| labelled area (coverage) | +0.90 |
| number of regions | +0.19 |
| agreement between the labelling rule and the released key | −0.19 |

| day | IoU | coverage | radial position (0 = centre, 1 = limb) |
|---|---|---|---|
| 16 Jan | 0.491 | 1.5 % | 0.45 |
| 17 Jan | 0.411 | 1.2 % | 0.53 |
| 18 Jan | 0.360 | 0.9 % | 0.60 |

The same regions rotated toward the limb; their line-of-sight-thresholded labelled area
shrank by a third; the model's overlap fell with it. Agreement between rule and key stayed
flat, so this is solar rotation and foreshortening, not the model or the labels misbehaving.
It is the limitation `GROUND_TRUTH_EXPLAINED.md` predicts, now measured.

**Consequence for method.** The notebook had split the validation set for calibration by
index — day 1 to fit, days 2–3 to evaluate — which put the easy images in the fit (0.470) and
the hard ones in the evaluation (0.368). A seeded random split gives 0.435 vs 0.403, i.e.
noise. The split is now random.

---

## 6. The answer key

The outlines are the ARPIL task of SuryaBench (Roy *et al.*, *Scientific Data* 13, 712,
2026): from the HMI line-of-sight magnetogram, threshold at ±50 G, drop regions under 100 px,
dilate with "a rectangular filter of size 10 pixels", take the intersection of the dilated
polarity maps as the polarity-inversion-line map, and keep active-region blobs that contain
one. Findings from reproducing it:

| finding | evidence |
|---|---|
| Size filter must precede dilation | reversed, agreement with the released key collapses to 0.01–0.09 |
| "Rectangular, size 10" = square brush grown 10 px, not a disc, not a 10 × 10 box | 12 timestamps: PIL agreement better on 12/12 (median 0.74 → 0.80); footprint 9/12 (≈ 0.85 → 0.86); a 10 × 10 box scores 0.70 |
| Good days reproduce closely; May 2010 does not | 2011-01-16: 0.66–0.92; 2010-05-13: ≈ 0.0–0.18. The 5,291 masks from 2010 were absent from the preprint (119,454 masks) and present in the published version (121,963 — matches the release exactly) |
| The two stored maps use different encodings | `union_with_intersect` 0/255, `intersection` 0/1 |
| AR coverage varies 78× over the solar cycle | 0.023 % (2019) to 1.80 % (2014); "≈ 1 %" is a solar-maximum figure |

The residual gap on good days (footprints ≈ 0.85 rather than 1.0) most likely comes from how
an "AR containing a PIL" is delimited, which the paper does not specify.

---

## 7. Real budget: 400 training / 400 validation images, 5 seeds — in progress

**Setup.** Images spread evenly across 2013–2015 (`window-stratified`, ≈ 131–137 per year in
each split; the first-400 alternative would have been 17 consecutive days of March 2013, the
same regions rotating). Up to 20 epochs with early stopping (patience 3), best-epoch weights
kept. Download: 774 files, 457 GB, completed in under 1 h 50 m. Launched 2026-09-17 00:09.

**Seed 0 so far** (epochs ≈ 43 min each; validation loss selects the checkpoint):

| epoch | val_loss | IoU | precision | recall |
|---|---|---|---|---|
| 0 | 0.475 | 0.492 | 0.62 | 0.70 |
| 1 | 0.452 | 0.518 | 0.68 | 0.69 |
| 2 | 0.412 | 0.531 | 0.62 | 0.80 |
| 3 | 0.400 | 0.562 | 0.70 | 0.74 |
| 4 | 0.360 | 0.589 | 0.71 | 0.78 |
| 5 | 0.356 | 0.579 | 0.65 | 0.84 |
| 6 | 0.347 | 0.598 | 0.70 | 0.80 |
| 7 | 0.336 | 0.599 | 0.68 | 0.84 |
| 8 | 0.350 | 0.611 | 0.77 | 0.75 |
| 9 | **0.320** | 0.627 | 0.74 | 0.81 |
| 10 | 0.323 | **0.637** | 0.78 | 0.78 |

Three things this curve already establishes:

1. **400 images are not memorised after eleven passes.** At 60 images the best epoch was the
   second; here validation loss was still setting new lows at epoch 9, and IoU is still rising.
2. **Epoch 0 alone beat the entire medium run** (0.492 vs a best of 0.465), on a broader and
   harder validation set spanning three Januaries.
3. **IoU has crossed the medium run's label floor** (0.637 vs 0.632). If this run's own floor —
   recomputed on its own validation images in Step 6 — comes out similar, the model has
   reached the point where its remaining disagreement with the answer key is the size of the
   key's disagreement with itself.

**Timing.** Seed 0 will stop three epochs after its last improvement, so around epoch 12–14,
finishing ~11:30–13:00. At ~9–10 h per seed the five-seed total is **~48–52 h** (done
2026-09-19 morning); three seeds would be ~30 h. Given that each seed now trains on enough
data to be individually meaningful, three members are likely sufficient to measure the
seed-to-seed spread — a decision to take when seed 0 completes.

---

## 8. Machine findings

| finding | value |
|---|---|
| The pod's real limits (cgroup), vs what `free`/`nproc` report | **60 GiB, 7 cores**, `oom.group=1` — vs 124 GB, 16 cores on the node |
| What an OOM does | kills and restarts the whole container; `/tmp` and `nohup`'d jobs are lost (happened once, with 10 loader workers) |
| Page cache from the `.nc` reads | charged to the pod, fills to the limit within an epoch, reclaimed on demand: **14.2 million reclaims, 0 OOMs** |
| Unreclaimable memory (anon + shmem), 4 loader workers | ~25 GB steady (medium); 34 GB steady / **40 GB peak** (real, at 05:37); the guard trips at 54 |
| GPU during training | 100 % on an L40S with 4 workers (0 % with 2 — loader-starved) |
| Validation and Step 6 | loader- and EFS-bound, not GPU-bound: the medium analysis took 90 min cold and 19 min with files in page cache |
| Download from S3 | 44–70 MB/s with 16 threads |

---

## 9. Open items

- **3 vs 5 seeds** for the `real` run (§7).
- **Per-region scoring** — detection rate, false alarms, per-region position and area error —
  is what a forecaster would use, and what the limb effect (§5) argues for over pixel overlap.
- **Test-time augmentation** as a cheap extra uncertainty source (safe at inference; the
  training pipeline disables flips because the outline would not be flipped with the image).
- **Vary the other two rule parameters** (blob size, brush size) in the label floor, not just
  the threshold.
- This report is a snapshot; the `real` run's own output (`runs/nbconvert/3_errors_real.ipynb`)
  is the record once it finishes, and §7 should be replaced by its final tables.

---

## Provenance

| artefact | location |
|---|---|
| executed medium runs | `runs/nbconvert/3_errors_medium_run.ipynb` (provisional), `3_errors_medium_final.ipynb` (final) |
| real run (in progress) | `runs/nbconvert/3_errors_real.ipynb` when written; per-epoch metrics under `runs/errors_surya_seed*_real_*/` |
| memory / GPU log | `runs/nbconvert/mem_watch2.log` |
| answer-key dataset | Roy *et al.* 2026, DOI 10.1038/s41597-026-06552-5; `nasa-ibm-ai4science/surya-bench-ar-segmentation` |
