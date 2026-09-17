# What the Error-Analysis Notebook Does

A plain-language guide to `3_errors_ar.ipynb`, the fourth notebook in the active-region
segmentation example. It assumes you have read `NOTEBOOKS_EXPLAINED.md`, which covers the
three notebooks before it.

Those three notebooks each end with **one number**: how well the model's outlines overlap the
true ones. This notebook's job is to say **how much that number can be trusted** — and it
turns out the honest answer involves three separate questions that a single number cannot
answer.

---

## Contents

1. [The one idea](#the-one-idea)
2. [Before you run it: the knobs](#before-you-run-it-the-knobs)
3. [What the machine needs](#what-the-machine-needs)
4. [Walkthrough](#walkthrough)
5. [How to tell whether it is working](#how-to-tell-whether-it-is-working)
6. [A worked example: what actually happened](#a-worked-example-what-actually-happened)
7. [Practical notes](#practical-notes)
8. [Quick reference](#quick-reference)

---

## The one idea

Suppose the model scores 0.72 and a simpler model scores 0.68. Is the model better?

You cannot tell from those two numbers alone. Three different things could be hiding inside
the gap:

| Question | What it is about | Where the notebook answers it |
|---|---|---|
| **Would I get the same number with different images?** | Sampling — the score is computed from a handful of images, and a different handful gives a different score | Step 2 |
| **How sure is the model about each pixel, and is that confidence honest?** | The model's own hesitation, and whether "80% sure" really means 80% | Steps 3 and 4 |
| **Would I get the same number if the model were trained again?** | Randomness in training — a different starting point gives a different model | Step 5 |
| **Is the answer key itself precise enough to be measured against at this resolution?** | The outlines were made by a rule with arbitrary settings; nudging them moves the "truth" | Step 6 |

The notebook replaces the single number with an **interval** for each of these, and then
asks whether the gap between two models is bigger than all of them. Only then is "better"
a defensible word.

---

## Before you run it: the knobs

Everything adjustable sits in one cell near the top. Five settings matter.

### `BUDGET` — how big a run

| preset | images | passes over the data | models trained | what it is for |
|---|---|---|---|---|
| `smoke` | 8 | 2 | 2 | Minutes. Proves the code runs. **Every number it prints is noise.** |
| `medium` | 60 | 10 | 2 | Hours. Enough images for an interval to mean something, enough passes to see whether training is heading somewhere. |
| `real` | 400 | 20 | 5 | Numbers you could report. Large download, long run. |

Start with `smoke` to see every figure appear, then `medium` to learn whether the setup is
sound, and only then `real`. Skipping straight to `real` risks spending a day training a
model that was going to fail anyway.

### `MODEL_KIND` — which model

`"surya"` is the 366-million-parameter foundation model with small adapters; it needs a large
GPU. `"baseline"` is the 14-number pixel model from Notebook 1; it runs anywhere. Every
figure works with either, which is what makes the two comparable.

### `SAMPLE_SELECTION` — which images

This one is easy to overlook and matters enormously. The dataset sorts images by date, so a
naive cap takes the **earliest** ones — January 2011, near the bottom of the solar cycle,
when active regions cover about **one pixel in a thousand**. Three years later they cover
one in fifty-five: an eighteen-fold difference in signal per image.

- `"window"` (default) restricts both training and validation to a date range — solar
  maximum, 2013–2015. This is a *stated* narrowing of scope, and it is honest.
- `"stratified"` spreads the sample evenly across the whole archive, for a model that must
  work at any point in the cycle.
- `"head"` reproduces the naive behaviour, so you can see the difference.

There is deliberately **no** option to pick the validation images with the most active
regions. That would make the test easier and every score higher, and it would not be honest.

### `POS_WEIGHT_MODE` — correcting the imbalance

With active-region pixels at around one percent of the disk, a model can score well on
the training objective by simply answering "background" everywhere. `pos_weight` makes each
active-region pixel count for more so that this stops being a good strategy.

The notebook **measures** the imbalance from the outlines it actually selected rather than
trusting a number in a comment. `"sqrt"` (default) is the usual compromise; `"full"`
balances the two classes exactly and tends to over-predict; `"off"` reproduces the failure.

### `LOSS` — what training optimises

The scores everyone reports are *overlap* scores, but the standard training objective is a
per-pixel penalty that is only loosely related to overlap. `"bce+dice"` (default) adds a term
that optimises overlap directly. `"bce"` is the plain objective, for comparison.

### Workers — set automatically

The notebook reads the container's real limits (see below) and derives how many data-loading
processes to use. It prints the arithmetic so you can check it.

---

## What the machine needs

**A large GPU** for the Surya model, and patience: each sample is a 13-channel image of
4096 × 4096 pixels, roughly 0.9 GB in memory.

**Disk for the image cache.** Each timestep is about 590 MB. The `medium` budget needs 120
unique timesteps (60 training, 60 validation), about 70 GB; `real` needs roughly 470 GB.
The config points the cache at scratch storage for this reason. Files are only downloaded
once.

**An honest reading of memory.** If you are running inside a container — a JupyterHub pod,
for example — the usual tools (`free`, `nproc`) report the *whole machine*, not your slice
of it. Your real limits live in `/sys/fs/cgroup/`, and exceeding the memory one does not
slow you down: the whole container is killed and restarted, taking any unsaved output with
it. The notebook reads those limits directly. It also knows that the container is charged for
the operating system's file cache as the image files are read, that this cache is harmlessly
reclaimed on demand, and that only *unreclaimable* memory can trigger the kill — so it sizes
against the right quantity.

---

## Walkthrough

### Setup

Same as the other notebooks: choose a GPU, set one environment variable *before* the deep
learning library loads (it is read once, at start-up, and is silently ignored if set later),
load the configuration, fetch the weights and normalisation statistics.

### Step 1 — Raise the sample cap and the epoch count

Applies the chosen `BUDGET` by overriding two values in the loaded configuration, without
editing the file on disk. A learning rate is chosen per model here too: the config says the
foundation model should train ten times more gently than the baseline, and the notebook
honours that rather than using one rate for both.

### Choosing which images

Builds the datasets uncapped, then narrows them with `select_timestamps()` according to
`SAMPLE_SELECTION`. Prints the date span and count of what was chosen. It also warns if there
are fewer than 30 validation images, because below that an interval stops carrying much
information.

### Measuring the imbalance

Reads the selected outlines directly and reports what fraction of pixels are active region,
the resulting imbalance, and the `pos_weight` derived from it. Under the default window this
is about 0.9 % of the disk, roughly 113 background pixels per active-region pixel.

### Warming the cache in parallel

Downloads every image the run will need up front, sixteen at a time, so training never stalls
waiting for a file. Already-cached files are skipped, so re-running is free. It reuses the
dataset's own file-naming and download code — the cache filename is a fingerprint of the
file's cloud location, so a hand-rolled name would be silently ignored and everything
re-downloaded.

### Model factory and training

One function builds whichever model `MODEL_KIND` names. Another trains it under a given
random seed and **caches the result**, so re-running the notebook reloads instead of
retraining. The cache key includes every setting that changes what training does; without
that, a checkpoint from a failed configuration would reload silently and a fix would appear
to do nothing.

Two more behaviours of the training function are worth knowing. It keeps the epoch with the
lowest validation loss — Lightning selects it, and the function restores those weights before
caching them, so later steps never analyse an overfit final epoch. And if an earlier run with
the same settings already left such a checkpoint on disk, the function **adopts** it instead of
training again, so re-running the analysis after a code change costs minutes rather than hours.
Training also stops early once validation loss has not improved for `EARLY_STOP_PATIENCE`
epochs; on the first sixty-image run the best epoch was the second of ten.

The training objective is defined here. Rather than editing the app's shared scoring code, the
notebook extends it in place, adding the optional overlap term. The app's own tests are left
untouched.

**This is the only expensive cell.** It trains one model per seed. Everything after it is
inference or arithmetic.

### The prediction cache

One pass over the validation images stores, for each image, a **histogram of the model's raw
scores** split by whether each pixel was truly active region — a few thousand numbers instead
of a 16.8-million-pixel map. From those histograms the notebook can recompute the overlap
score at *any* threshold, draw the reliability diagram and fit the calibration, exactly and
instantly. Full maps are kept only for the two or three images the contour figures need.

A check confirms the histogram route gives the same score as the app's scoring code (it
matches to within the histogram's bin width).

### Step 2 — Bootstrap confidence intervals

The score is computed from a sample of images, so it has sampling error. The notebook
re-draws that sample with replacement a few thousand times, recomputes the score each time,
and reports the spread as a 95 % interval.

Two versions are given, because they answer different questions. **Pooled** throws every pixel
into one bucket, so large active regions dominate and a quiet-Sun image barely counts; this is
what the app's scoring code reports. **Per-image** gives each image one vote, which is usually
what someone means by "how well does it do on a typical image".

A second cell shows how the interval narrows as images are added. Width falls roughly with the
square root of the count, so halving it costs four times the data. Read off where the curve
drops below the difference you care about: that is how many images you need.

### Step 3 — Reliability diagram and calibration

The model gives each pixel a number between 0 and 1. Before it can be read as "how likely",
it has to be checked: of all pixels the model scored near 0.7, were about 70 % actually active
region? The reliability diagram plots that for every confidence level. An honest model sits on
the diagonal; a typical neural network sits below it, overstating its confidence.

The fix is two numbers, fitted after training, that rescale and shift the raw scores. Crucially
the notebook fits them on a **random** half of the validation images and reports on the other half —
random rather than first-and-second, because the validation images are in time order and the
Sun rotates: on the sixty-image run the first half scored 0.47 and the second 0.37 as the
regions drifted toward the limb (per-image overlap tracks distance from disk centre at
r = −0.94). A random split gives 0.44 versus 0.40, which is noise.
Fitting and reporting on the same images is how calibration comes to look better than it is.

Calibration barely changes the best achievable overlap — it only rescales, so the ranking of
pixels is unchanged. What it buys is *meaning*: afterwards, 0.75 is a statement about the
world, which is the only reason the contours in Step 4 can be read as an error bar. A
threshold sweep then shows where the overlap-maximising cut actually sits; with this much
imbalance it is rarely 0.5.

### Step 4 — Confidence contours

Instead of one outline at 0.5, the notebook draws the boundary at 0.25, 0.5 and 0.75 on the
calibrated map. Where the three curves sit on top of one another the model is decisive; where
they spread into a band it is hedging, and the band *is* the uncertainty on the outline.

One physical caveat is built in: the model's final layer works on 16-pixel tiles, so contour
position is not meaningful below about 16 pixels. The figures downsample by four for display,
which stays well inside that.

### Step 5 — An ensemble of seeds

Train the same model from a different random start and you get a slightly different answer.
That difference is real uncertainty, and a single run is blind to it. Because the default
regime freezes the big model and trains only small adapters, several members cost far less
than several full trainings.

The notebook averages the members' confidence maps (a better prediction than any one of them)
and shows where they disagree (a genuine uncertainty map, which lights up along region
boundaries). It also compares the spread of the *score* across seeds with the width of one
seed's bootstrap interval. If those are similar, quoting a single run's number to three
decimals is not defensible.

### Step 6 — The label floor

This step changes how everything above should be read.

The outlines are not ground truth. They come from a rule — field stronger than 50 gauss,
blobs larger than 100 pixels, edges grown by 10 pixels, a dividing line required — and
nobody in the Sun chose 50. So the notebook re-runs that rule at 40 and 60 gauss and measures
how far the "truth" moves. Two equally defensible answer keys that agree with each other only
to some level set a **floor**: no model scored against one of them can be resolved any finer.

The rule is a reconstruction from its published description, not the original authors' code,
so the notebook **scores the reconstruction against the real outlines first** and computes the
floor only from images where they agree. Where they do not, it says so rather than inventing a
number. A figure at the end shows the three versions of the answer key drawn over the magnetic
map they came from — usually the most persuasive picture in the notebook for explaining why a
segmentation score cannot be quoted to three decimals.

### Putting it together

A final table lists every score with its interval, next to the floor. The question it answers
is not "what did the model score" but **"is the gap larger than the uncertainty in measuring
it, and larger than the ambiguity in the definition it is measured against?"**

---

## How to tell whether it is working

The training metrics print precision and recall together, and the pair is the diagnostic.
Watch them, not the loss.

| Signature | Precision | Recall | What it means |
|---|---|---|---|
| **Collapse** | rises toward 1.0 | falls toward 0 | The model has learned to predict nothing. The loss will *keep improving* while this happens, which is why the loss alone is misleading. Raise `pos_weight`, check the sample selection. |
| **Over-correction** | falls toward 0 | rises toward 1.0 | The model flags everything. `pos_weight` is too high; move from `"full"` toward `"sqrt"` or a smaller number. |
| **Healthy** | both well away from 0 and 1 | | Real predictions, wrong in both directions, improving over epochs. |

Two more things to check early:

- The second training-loss term (`dice`) should **move**. If it stays near 1.0 while the
  first term falls, the model is not localising anything yet.
- The interval in Step 2 should be narrower than the difference you care about. If it is not,
  nothing downstream can resolve that difference, and more images are the only fix.

---

## A worked example: what actually happened

The first run of this notebook, at the `smoke` budget, produced a textbook collapse:

| epoch | loss | overlap | precision | recall |
|---|---|---|---|---|
| 0 | 0.021 | 0.0054 | 0.71 | 0.0054 |
| 1 | 0.013 ↓ | **0.000035** | 1.00 | 0.000035 |

Loss improved; the model stopped predicting anything. Three causes were found, all now fixed
in the notebook:

1. **The images were the wrong ones.** The naive cap had selected January 2011 — 0.1 % active
   region, 987 background pixels per active pixel. The `"window"` default gives 1.6 %, a
   sixteen-fold improvement in signal per image.
2. **The rare pixels were not up-weighted.** `pos_weight` was unset, so "predict nothing" was
   close to optimal for the objective.
3. **The learning rate was ten times too high** for the foundation model. The config had said
   so; the notebook had not honoured it.

The same budget after the fixes: overlap 0.041, recall 0.043 — no longer at the floor, but
still a two-pass, eight-image model. At the `medium` budget the picture is a real segmentation: overlap 0.435 after the first pass
over sixty images, and — with the best epoch kept rather than the last, and the calibration
split at random — **0.437 (95 % interval 0.424–0.449) at the default cut, 0.465 at the best
calibrated cut**, with the `dice` term down from 0.98 to 0.23. The two seeds differ by 0.025,
about the width of one seed's interval, so the honest report is the pair, not a single number.
The label floor came out at 0.632, and the best model at 0.459 sits well below it: the
remaining error is the model's, not the definition's.

The lesson the example teaches is the one the notebook exists for: a loss going down told us
nothing. The pair of precision and recall told us everything.

---

## Practical notes

**Re-running is cheap.** Downloads, trained models and the prediction pass are all cached.
Change a setting that affects training and the cache key changes with it, so you get a fresh
model; change only an analysis setting and nothing retrains.

**Outputs.** Training logs and checkpoints go under `runs/`. The committed notebook is kept
free of outputs; run it to a separate copy (for example with `nbconvert --output-dir`) if you
want the figures saved, and put that copy somewhere that survives a container restart — not
`/tmp`.

**What the numbers are not.** At `smoke` they are noise, by design. At `medium` they tell you
whether the configuration is sound and roughly where the error lives. Only `real` produces
figures worth reporting, and even then the label floor from Step 6 bounds how finely they can
be read.

**Not covered here, but natural next steps.** Scoring by *region* rather than by pixel
(did we find each active region, and how far off are its position and size); test-time
augmentation as a cheap extra uncertainty source; varying the other two settings of the
labelling rule (blob size and edge growth), not just the threshold.

---

## Quick reference

### Settings

| knob | default | alternatives |
|---|---|---|
| `BUDGET` | `"medium"` | `"smoke"`, `"real"` |
| `MODEL_KIND` | `"surya"` | `"baseline"` |
| `SAMPLE_SELECTION` | `"window-stratified"` (spread across 2013–2015) | `"window"`, `"stratified"`, `"head"` |
| `POS_WEIGHT_MODE` | `"sqrt"` | `"full"`, `"off"`, or a number |
| `LOSS` | `"bce+dice"` | `"bce"` |
| `EARLY_STOP_PATIENCE` | `3` | `None` to train every epoch |
| workers | derived from the container's limits, capped at 4 | edit the cap if the GPU is idle |

### Outputs, in order

| step | figure or table | the thing to look at |
|---|---|---|
| training | precision and recall per epoch | neither near 0 or 1; `dice` term moving |
| 2 | score with 95 % interval; width vs image count | interval narrower than the gap you care about |
| 3 | reliability diagram, before and after | "after" closer to the diagonal |
| 3 | overlap vs threshold | where the best cut actually sits |
| 4 | three-level contours and uncertainty band | band width along boundaries |
| 5 | ensemble mean, disagreement map, per-seed scores | seed spread vs bootstrap width |
| 6 | agreement of the reconstruction, then the floor | agreement first; floor only where it is high |
| end | every score with interval, next to the floor | is the gap bigger than both? |
