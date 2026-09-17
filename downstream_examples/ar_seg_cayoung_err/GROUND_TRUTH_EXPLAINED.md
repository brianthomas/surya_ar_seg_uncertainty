# How the "Ground Truth" Outlines Are Made

The active-region outlines that every model in this app is trained on and scored against
were not drawn by anyone. They were **computed by a fixed rule** from one measurement — a map
of the Sun's magnetic field — and published as a dataset. This document explains where they
come from, exactly what the rule does, what the files contain, what the outlines mean
physically, how the workshop uses them, how faithfully the rule can be reproduced, and what
their limits are.

It is written for readers who are not solar physicists or machine-learning specialists.

---

## Contents

1. [Where the outlines come from](#where-the-outlines-come-from)
2. [The input: a map of the magnetic field](#the-input-a-map-of-the-magnetic-field)
3. [The rule, step by step](#the-rule-step-by-step)
4. [What is in a file](#what-is-in-a-file)
5. [What the outlines mean physically](#what-the-outlines-mean-physically)
6. [How the workshop uses them](#how-the-workshop-uses-them)
7. [How faithfully can the rule be reproduced?](#how-faithfully-can-the-rule-be-reproduced)
8. [Limitations to keep in mind](#limitations-to-keep-in-mind)
9. [Quick reference](#quick-reference)

---

## Where the outlines come from

The outlines are the **ARPIL** dataset — *Active Regions with Polarity Inversion Lines* — one
of the benchmark tasks in **SuryaBench**, published alongside the Surya foundation model:

> Roy, S. *et al.* "SuryaBench: Benchmark Dataset for Advancing Machine Learning in
> Heliophysics and Space Weather Prediction." *Scientific Data* **13**, 712 (2026).
> https://doi.org/10.1038/s41597-026-06552-5 (preprint: arXiv:2508.14107)

The data production for this task is credited to Talwinder Singh and Berkay Aydin; the
Hugging Face release lists Jinsu Hong, Kang Yang and Berkay Aydin (Georgia State University)
as authors. The method extends an earlier polarity-inversion-line detection technique from the
same group to full-disk images.

The workshop downloads the released files from
[`nasa-ibm-ai4science/surya-bench-ar-segmentation`](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation)
on Hugging Face (licence CC BY 4.0) via `download_masks.sh`.

---

## The input: a map of the magnetic field

Everything starts from one instrument: the **Helioseismic and Magnetic Imager (HMI)** on
NASA's Solar Dynamics Observatory. Among its products is a *line-of-sight magnetogram* — a
4096 × 4096 image of the whole solar disk in which each pixel records the strength of the
magnetic field along the line from the Sun to the spacecraft, in **gauss**.

Three things about that image matter for what follows:

- **Sign is polarity.** Positive means the field points toward the spacecraft, negative means
  away. A region of strong field with both signs close together is the raw material of an
  active region.
- **Each pixel is about 0.36 Mm across** near the centre of the disk (the paper's conversion:
  100 pixels ≈ 13.3 Mm² of photosphere). So "10 pixels" is roughly 3,600 km on the Sun.
- **Noise is a few to roughly ten gauss per pixel**, so a 50 gauss cut is comfortably above
  it. Most of the disk is quiet — a mottled field of small, weak, mixed-polarity elements.

HMI produces a magnetogram every 12 minutes. The rule is applied **once per hour**, to
"each valid" magnetogram.

The same magnetogram is one of the 13 channels the models see as input (`hmi_m`). The answer
key is therefore derived from part of the model's own input, which is discussed under
[Limitations](#limitations-to-keep-in-mind).

---

## The rule, step by step

Quotations are from the paper. Each step turns one image into another.

### 1. Threshold: find strong field of each sign

> "applying a magnetic field strength threshold of +50 and −50 Gauss"

Two black-and-white maps are made. In the first, a pixel is on if its field is **above +50 G**;
in the second, if it is **below −50 G**. Everything weaker — the quiet Sun — is off in both.
The result is scattered specks and patches of strong positive field, and separately of strong
negative field.

### 2. Size filter: drop the small stuff

> "we apply a size filter that excludes regions smaller than 100 pixels (approximately
> 13.3 Mm²)"

Any connected patch of fewer than 100 pixels is removed from each map. This clears out noise
and small magnetic elements that are not part of an active region. It is applied **before**
the next step, and the order matters a great deal — see
[reproduction](#how-faithfully-can-the-rule-be-reproduced).

### 3. Dilation: grow every patch outward

> "we dilate the binary images using a rectangular filter of size 10 pixels"

Each surviving patch, in each map separately, is expanded outward by 10 pixels using a
square-shaped brush. Two effects: pinholes and ragged edges are filled in, and — the point of
the step — patches of opposite polarity that sit within about 20 pixels (~7 Mm) of each other
now **overlap**.

### 4. Intersection: find where opposite polarities meet

> "we identify the intersection of the dilated positive and negative polarity regions, which
> corresponds to areas containing PILs"

Wherever the grown-positive map and the grown-negative map overlap, opposite polarities are
close together. That overlap is the map of **polarity inversion lines (PILs)** — the
boundaries where field pointing toward us meets field pointing away. This is the first of the
two products stored per hour.

### 5. Selection: keep only active regions that have a dividing line

> "Only ARs that include PILs are reported."

The union of the two grown maps is a set of strong-field blobs. Only the blobs that **contain a
PIL** are kept as active regions. A large, strong patch that is entirely one polarity is
discarded on purpose. This is the second stored product, and the default target for training.

The paper does not spell out how a "blob" is delimited for this test; the workshop's
reconstruction treats each connected component of the union as one candidate region.

### 6. Output

> "The eventual AR masks are 2D bitmaps (containing zeros and ones) ... and have a size of
> 4096 × 4096"

One file per hour, two maps inside it.

---

## What is in a file

Each hour is one HDF5 file, `data/YYYY/MM/YYYYMMDD_HHMM.h5`, about 53 KB (gzip-compressed;
the maps are almost entirely zero). Inside:

| key | what it holds | stored values |
|---|---|---|
| `intersection` | the PIL map — where grown positive and grown negative overlap | 0 / **1** |
| `union_with_intersect` | active-region footprints that contain a PIL — **the default target** | 0 / **255** |

The two keys use **different encodings** (0/1 versus 0/255). Anything reading the raw arrays
must binarise with `> 0` rather than assuming either convention; the workshop's loader does.
The PIL map is always a strict subset of the footprint map.

The files carry no record of the parameters used to make them — the rule above is known from
the paper, not from the data.

### The catalogues

Four CSV files list every hour and whether a mask exists (`timestamp, file_path, present`):

| catalogue | covers |
|---|---|
| `train.csv` | 15 Feb – 31 Dec of each year 2010–2019 |
| `validation.csv` | 15 – 31 Jan of each year 2010–2019 |
| `leaky_validation.csv` | 1 – 14 Jan and 1 – 14 Feb of each year 2010–2019 |
| `test.csv` | all of 2020–2024 |

Across all four: **128,352** hourly rows, of which **121,963** have a mask
(`present == 1`); the rest are hours with no valid magnetogram. The masks run from
13 May 2010 to 31 Dec 2024. The published paper's count of 121,963 matches the release
exactly; the earlier preprint reported 119,454 masks from January 2011, so the 5,291 masks
from 2010 were added between preprint and publication.

---

## What the outlines mean physically

A solar active region is a place where strong magnetic field has surfaced from below.
Physically, the interesting structure is not the patch of field itself but the **boundary
between opposite polarities** — the polarity inversion line. Magnetic energy is stored where
field lines are sheared and twisted across such a boundary, and that is where flares and
eruptions are launched.

That is why the rule insists on step 5. A strong unipolar patch — the decaying remnant of an
old region, or plage without a companion of opposite sign — is magnetically quiet and is left
out. When a model later flags such a patch, it has not necessarily made an error of
perception; it has disagreed with a definition. The notebooks call this out because it is the
most common source of apparent false positives.

### How much of the Sun is covered

Measured from the released files, the fraction of the frame marked as active region varies
enormously with the 11-year solar cycle:

| year | AR coverage | background : AR |
|---|---|---|
| 2010 | 0.19 % | 516 : 1 |
| 2011 | 1.00 % | 100 : 1 |
| 2013 | 1.37 % | 73 : 1 |
| **2014** | **1.80 %** | **55 : 1** |
| 2016 | 0.47 % | 215 : 1 |
| 2018 | 0.06 % | 1653 : 1 |
| 2019 | 0.023 % | 4285 : 1 |

A 78-fold swing. Two consequences run through the whole app: the class imbalance a model
faces depends entirely on *which years* it sees, and any statement like "active regions cover
about 1 % of the disk" is a solar-maximum figure, not a general one.

---

## How the workshop uses them

- **Pairing.** The outlines are hourly; the model's input images arrive every 12 minutes. The
  loader (`datasets/ar_dataset.py`) pairs an image with an outline only on an **exact
  timestamp match**, so only on-the-hour images are ever used — about one in five.
- **Which split.** The four catalogues are all loaded, but only as a lookup table. Train
  versus validation is decided by the *image* index, whose validation set is mid-to-late
  January of each year. The two systems happen not to overlap, but the image index is the
  arbiter.
- **Which map.** `ar_mask_key: union_with_intersect` (default) trains on the footprints;
  `intersection` trains on the PILs themselves — a far thinner target with a harsher imbalance.
- **Binarisation.** Both maps are loaded as `> 0`, giving a float 0/1 image regardless of the
  stored encoding.
- **No vertical flips.** The base image loader can flip inputs as augmentation; the AR loader
  refuses, because the outline would not be flipped with them.
- **Missing hours.** Rows with `present == 0` are dropped before pairing.

---

## How faithfully can the rule be reproduced?

`3_errors_ar.ipynb` re-runs the rule at ±40, ±50 and ±60 gauss to measure how much the
"truth" moves when its arbitrary threshold is nudged (the *label floor*). That only means
something if the re-implementation reproduces the released outlines at ±50 in the first
place, so the notebook scores itself against them before reporting anything.

The reconstruction is short:

```python
pos = remove_small_objects(B >  50, 100)        # step 1 + 2, positive
neg = remove_small_objects(B < -50, 100)        # step 1 + 2, negative
pos = binary_dilation(pos, ones((3, 3)), iterations=10)   # step 3, square, 10 px
neg = binary_dilation(neg, ones((3, 3)), iterations=10)
pil = pos & neg                                 # step 4
labels = label(pos | neg)                       # step 5: components of the union ...
ar = isin(labels, unique(labels[pil]))          # ... that contain a PIL
```

How well it matches, measured as overlap (IoU) with the released maps on cached timestamps:

| finding | evidence |
|---|---|
| Order matters: size filter **before** dilation | reversing it collapses agreement to 0.01–0.09 |
| "Rectangular filter of size 10" = square brush grown 10 px, not a disc and not a 10 × 10 box | on 12 timestamps: PIL agreement better on 12/12 (median 0.74 → 0.80); footprint agreement better on 9/12 (median ≈ 0.85 → 0.86); a 10 × 10 box scores 0.70 |
| Well-behaved days reproduce closely | 2011-01-16: footprints 0.66–0.92, PILs 0.72–0.85 |
| On the sixty January-2013 validation images of the full run | all 60 above 0.6; median footprint agreement 0.81 with the square brush, 0.83 with a disc on the same images — the two are within noise of each other on footprints, and the paper's wording decides |
| The first days of the mission do not | 2010-05-13: footprints ≈ 0.0–0.18 |

The 2010 failure has a plausible explanation now that the dataset's history is known: those
months were not in the preprint's version of the dataset, and the magnetograms the workshop
serves for 2010 may not be the ones the rule was run on. The notebook therefore treats a
reconstruction that disagrees with the released map as *untrustworthy on that image* and
excludes it from the floor, rather than assuming the image is at fault.

What explains the remaining gap on good days — a footprint agreement of ~0.85 rather than
1.0 — is not settled. The leading candidate is step 5: the paper does not define what counts
as one "AR" for the contains-a-PIL test, and the reconstruction's choice (a connected
component of the union) can merge or split blobs differently from the original. The evidence
for this is that the three timestamps where the square brush scored *worse* on footprints
moved by 0.1–0.2 at once — the signature of a whole blob flipping in or out, not of edges
shifting. Smaller candidates: the exact brush semantics, how the edge of the disk is treated,
and the version of the magnetogram data.

None of this affects training; it affects only how finely the label floor can be read, and
the notebook gates on the agreement score for exactly that reason.

---

## Limitations to keep in mind

**The threshold is on the line-of-sight field.** Near the edge of the disk the field is seen
increasingly side-on, so its line-of-sight component shrinks even where the real field is
strong. No correction for this is stated. Regions near the limb are therefore harder to
detect and their outlines smaller than the same regions would be at disk centre. This is measurable: across three days of
validation images in January 2013, a model's per-image overlap fell from 0.49 to 0.36 as the
labelled regions rotated from 0.45 to 0.60 of the disk radius (r = −0.94) and their labelled area
shrank by a third, while agreement between the rule and the released outlines stayed flat.

**Sizes are in pixels, not kilometres.** A 100-pixel patch and a 10-pixel brush cover far more
of the Sun's surface near the limb, where the surface is foreshortened, than at the centre.
The rule does not compensate.

**The parameters are choices.** 50 gauss, 100 pixels and 10 pixels are reasonable but
arbitrary. Nudging any of them moves every outline. That is not a flaw to fix; it is the
reason a model's score against these outlines cannot be quoted more precisely than the
outlines themselves are defined — which is what the label floor in `3_errors_ar.ipynb`
measures.

**The answer key is derived from one of the inputs.** The magnetogram the rule reads is
channel `hmi_m` of the 13 the model sees. A model can in principle learn to imitate the rule
from that channel alone, which is why Notebook 1's two-parameter baseline (a learned threshold
on `hmi_m`) is the right first rung of the ladder: it measures how much of the task the rule
already explains.

**Hourly only.** Four of every five images have no outline and cannot be used for this task.

**"Valid magnetogram" is not defined** in the paper. About 5 % of hours have no mask.

---

## Quick reference

| quantity | value | source |
|---|---|---|
| Instrument / product | SDO/HMI line-of-sight magnetogram | paper |
| Image size | 4096 × 4096, ≈ 0.36 Mm per pixel at disk centre | paper (100 px ≈ 13.3 Mm²) |
| Threshold | +50 G and −50 G, separately | paper |
| Minimum region | 100 pixels, applied **before** dilation | paper; order confirmed by measurement |
| Dilation | "rectangular filter of size 10 pixels" — square brush, 10 px | paper; shape confirmed by measurement |
| PIL map | intersection of the two dilated maps | paper |
| AR map | union blobs that contain a PIL | paper; blob definition not stated |
| Cadence | hourly, on each valid magnetogram | paper |
| Coverage | 13 May 2010 – 31 Dec 2024 | release catalogues |
| Masks | 121,963 present of 128,352 hours | release catalogues; matches the paper |
| Stored values | `union_with_intersect` 0/255, `intersection` 0/1 | measured from the files |
| Reproduction, good days | footprints ≈ 0.85–0.92, PILs ≈ 0.72–0.85 IoU | measured |
| Reproduction, May 2010 | ≈ 0.0–0.18 | measured |

### Sources

- Roy, S. *et al.*, SuryaBench, *Scientific Data* 13, 712 (2026) — https://doi.org/10.1038/s41597-026-06552-5
- Preprint — https://arxiv.org/abs/2508.14107
- Dataset release — https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation
- Surya model paper — https://arxiv.org/abs/2508.14112
