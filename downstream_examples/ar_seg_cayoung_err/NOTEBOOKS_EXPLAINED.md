# What These Four Notebooks Actually Do

A plain-language walkthrough of the active-region (AR) segmentation example in
`downstream_examples/ar_seg_cayoung_err/`, written for readers who are not machine-learning
specialists.

The goal of the exercise: teach a computer to look at pictures of the Sun and outline the
**active regions** — the blotchy, magnetically violent patches that produce solar flares.
Four notebooks split this into *get the data*, *try something deliberately dumb*, *try
something big*, and *find out how much to trust the result*.

The whole exercise is really a **measurement**, not a model-building project. Notebook 1
exists to produce a number that Notebook 2 has to beat, and Notebook 3 exists to say whether
the difference between them is real. Everything else is arranged to make that comparison fair.

---

## Contents

1. [Before you start](#before-you-start)
2. [How the active regions get located and marked](#how-the-active-regions-get-located-and-marked)
3. [Notebook 0 — assembling the data](#notebook-0--assembling-the-data)
4. [Notebook 1 — the deliberately dumb baseline](#notebook-1--the-deliberately-dumb-baseline)
5. [What "the baseline image" is](#what-the-baseline-image-is)
6. [Notebook 2 — fine-tuning Surya](#notebook-2--fine-tuning-surya)
7. [Notebook 3 — measuring the error](#notebook-3--measuring-the-error)
8. [Estimating error: the reasoning behind Notebook 3](#estimating-error-confidence-contours-on-the-outlines)
9. [The order of work, and what it found](#a-concrete-order-of-work)
10. [Quick reference](#quick-reference)

---

## Before you start

Two things must be in place or the notebooks fail on import or on first data access.

**The answer-key dataset.** The outlines are a separate download, about 1.3 GB compressed
and roughly 6 GB extracted:

```bash
cd downstream_examples/ar_seg_cayoung_err
bash download_masks.sh
```

This populates `data/surya-bench-ar-segmentation/` with four catalog files
(`train.csv`, `validation.csv`, `test.csv`, `leaky_validation.csv`) and about 122,000 hourly
outline files under `data/YYYY/MM/*.h5`. If that directory is missing you get a
`FileNotFoundError` naming the missing catalog.

**The right Python environment.** The notebooks need `lightning`, `torch`, `peft`, `h5py`
and `sunpy`. Select the kernel backed by the environment built from `environment.yml`.
A kernel missing `lightning` fails immediately on
`from workshop_infrastructure.utils import build_scalers`.

The Sun images themselves are **not** downloaded ahead of time. They stream from a public
cloud bucket on first use and land in a local cache directory — budget roughly 1 GB per
unique timestamp.

---

## How the active regions get located and marked

This is the part most people assume involves a human expert. It does not, and that matters
enormously for everything downstream.

Nobody hand-drew these outlines. They were produced by a **fixed mechanical rule** applied to
one specific measurement (the rule is published: Roy *et al.*, *Scientific Data* 13, 712, 2026 —
see `GROUND_TRUTH_EXPLAINED.md` for the full account): a map of the Sun's magnetic field, made every hour by an instrument
aboard the Solar Dynamics Observatory. Think of that map as a grayscale image where bright
means "magnetic field pointing toward us," dark means "pointing away," and mid-gray means
"not much field here."

The rule runs in four steps.

**1. Threshold.** Keep only pixels where the field is stronger than ±50 gauss (a unit of
magnetic strength). Everything weaker is discarded as quiet Sun. This leaves scattered specks
of strong field, both bright and dark.

**2. Drop the small stuff.** Any surviving blob smaller than about 100 pixels is thrown away
as noise or a minor feature.

**3. Grow the edges.** Each remaining blob is expanded outward by 10 pixels with a square brush. This fills
pinholes and merges blobs that are obviously part of the same structure.

**4. Require a dividing line.** This is the crucial step. Keep a region **only if** it
contains a *polarity inversion line* — a boundary where toward-us field sits directly against
away-from-us field. Physically, this is where magnetic energy gets stored and where flares
erupt.

Step 4 is why a huge, strong patch of magnetic field that is **all one polarity** is
deliberately left unmarked. Notebook 0 calls this out explicitly, and it matters later: when
the model appears to "false-positive" on such a patch, it may be recognizing something real
that the labeling rule chose to exclude.

Each hourly file stores two outlines:

| Key | What it contains |
|---|---|
| `union_with_intersect` | The full region footprints. **The default target.** |
| `intersection` | The dividing lines themselves — much thinner, much harsher imbalance. |

Switch between them with `data.ar_mask_key` in the config.

**The number to hold onto: marked regions cover between 0.02 % and 1.8 % of the frame,
depending on the year.** Near solar maximum (2011–2015) it is around 1 % — roughly 55 to 100
background pixels for every active-region pixel; near minimum it is a tenth of that or less,
up to 4,000 to 1. Every scoring decision in this example follows from that imbalance, and
Notebook 3 shows what happens when the years are chosen carelessly.

---

## Notebook 0 — assembling the data

`0_dataset_dataloader_ar.ipynb`

### Step 1 — Environment setup

The first cell sets an obscure environment variable *before* anything else loads. This is not
tidiness. That setting is read exactly once, when the underlying math library starts up. Set
it a moment too late and it silently does nothing. If you have already imported the deep
learning library in that session, you must restart the kernel.

### Step 2 — Read the configuration

A single settings file, `configs/config_script.yaml`, drives everything: which channels to
use, where the data lives, how big a batch is, what probability counts as "yes." Both other
notebooks and the command-line training script read this same file, so they cannot drift apart.

Two useful properties: an unrecognized setting raises an error naming the valid alternatives
rather than being silently ignored, and paths are interpreted relative to the config file, so
a checked-in config works from any working directory.

### Step 3 — Fetch the normalization statistics

Raw solar images span an absurd range of brightness — several orders of magnitude between a
quiet patch and a flare. The pipeline squashes each channel onto a comparable scale, then
subtracts the typical value and divides by the typical spread, so every channel arrives at the
model in roughly the same numeric range. These per-channel statistics download once and are
cached.

### Step 4 — Build the dataset, labels only

The first version deliberately skips loading the images (`return_surya_stack=False`), so the
label side can be checked in seconds instead of minutes. It:

1. reads the outline catalogs, keeping only rows marked "a real outline exists here";
2. **matches timestamps exactly** against the Sun-image catalog;
3. loads each matched outline as a 0-or-1 map.

The exact-match requirement has a real consequence. The outlines are **hourly**; the satellite
images arrive **every 12 minutes**. Only on-the-hour images can ever pair, so roughly four out
of every five images are discarded before training begins.

A second consequence: because the answer is an outline rather than a future picture of the
Sun, the dataset switches off the parent class's requirement that a frame one hour in the
future must also exist.

### Step 5 — Sanity-check one item

Print the shape, confirm the values really are only 0 and 1, print the fraction of the disk
marked. That fraction is the notebook's main teaching point:

> A model that answers "no active regions anywhere" is about 99% pixel-accurate and completely
> worthless.

This is why the scorecard later measures **overlap**, never accuracy.

### Step 6 — Plot three outlines

Purely to confirm they look like blobs on a disk, and to see how much the coverage varies from
one timestamp to another.

### Step 7 — Rebuild with the images attached

Now the full 13-channel stack loads: 8 ultraviolet views of the solar atmosphere plus 5
magnetic measurements, each 4096 × 4096. Each sample is roughly 0.9 GB. The first access to
any timestamp downloads its file into the local cache; later accesses are fast.

### Step 8 — Plot the images beside the outline

The pictures are converted back to physical units for display. The instruction here is the
payoff of the whole notebook: **look at the magnetic map against the outline.** Every marked
region sits on a patch containing both polarities. Single-polarity patches, however strong,
are unmarked.

### Step 9 — Wrap it in a loader

A loader adds batching and a sampling strategy. Training data is shuffled; validation data is
not. The shared builder used by the other two notebooks also seeds the shuffle explicitly, so
the ordering does not depend on ambient randomness.

### A note on train/validation splits

Two different split systems are in play, and only one of them is in charge.

The answer-key dataset ships its own splits (train: Feb 15 – Dec 31 of 2010–2019; validation:
Jan 15–31 of those years; test: all of 2020–2024). The config nonetheless loads **all four**
catalogs. That is safe because the **Sun-image index decides train versus validation** — its
validation set is mid-to-late January of each year, its training set is everything else, and
the two do not overlap in time. The outline catalogs are pooled purely as a lookup table.

If you ever swap in a different image index, check this yourself. The arbiter is the image
index, not the outline catalogs.

---

## Notebook 1 — the deliberately dumb baseline

`1_baseline_ar.ipynb`

**Why this notebook exists.** The outlines were *made* by thresholding the magnetic map. So a
model that does nothing but threshold the magnetic map should already do respectably. If the
366-million-parameter model cannot clearly beat that, it is not earning its keep. A simple
model that cannot possibly memorize its training data tells you what the big model actually
adds.

### Step 1 — Same setup, same config, same loaders

Identical to Notebook 0, except the datasets and loaders are built by a shared helper that
fills in every generic argument from the config. Only the task-specific arguments are passed
by hand. This is what makes Notebooks 1 and 2 nearly line-for-line identical.

### Step 2 — Define the baseline

For each pixel independently: take its 13 channel values, multiply each by a weight, add them
up, add one offset. That is the entire model — **14 numbers**. It has nowhere to store
memories, so it cannot overfit.

Two details worth knowing:

- It takes the **absolute value** of the inputs first. An active region is defined by field
  *strength*, not direction, so the sign is noise for this purpose.
- It partially undoes the normalization first, working in a compressed but more physical space.
  The compression stays, because raw values span too many orders of magnitude for a single
  linear layer to cope with.

A second configuration is worth running: set the channels to the line-of-sight magnetic map
only. That reduces the model to **2 numbers** — a learned magnetic threshold, which is
essentially the recipe the answer key was generated with. It measures how much of the task the
labeling rule already explains.

### Step 3 — Shape check

Run one untrained batch through and print the input and output dimensions. The output is
meaningless. The point is confirming the shapes line up; the notebook is blunt that mismatched
shapes are the dominant source of bugs in this kind of work.

### Step 4 — Define the scorecard

Four modes, of which two matter:

| Mode | What it does | Affects training? |
|---|---|---|
| `train_loss` | Per-pixel penalty for being wrong and confident | **Yes** |
| `val_loss` | Same penalty on held-out data; **selects the saved checkpoint** | No |
| `train_metrics` | Empty on purpose — overlap over 16.8 M pixels every step is not worth the time | No |
| `val_metrics` | Overlap, precision, recall, and the marked-pixel fraction | No |

The reported scores are **overlap** measures. The main one asks: of all pixels marked in
either the true outline or the guess, what fraction are marked in **both**? A perfect guess
scores 1.0; the "nothing anywhere" cheat scores 0.0.

Two implementation details with real consequences:

- Scores are **pooled over all pixels in the batch**, not averaged per image. Large active
  regions therefore dominate the number, and a quiet-Sun image contributes almost nothing.
- An image where the truth is empty *and* the guess is empty scores a perfect 1.0. Predicting
  nothing on a quiet Sun is correct.

There is also a knob (`ar_pos_weight`) to up-weight the rare active-region pixels in the
training penalty. The default leaves it off, matching the released Surya example.

### Step 5 — Score the untrained model

A sanity check. The penalty should sit near 0.69 — exactly what pure coin-flipping produces —
and the overlap should sit near the marked-pixel fraction. An untrained model is about as good
as guessing, and the numbers should say so.

### Step 6 — The training loop

A standard wrapper handles epoch bookkeeping, logging, and checkpoint selection. **It is the
identical wrapper Notebook 2 uses.** It is task-agnostic: it calls the model, computes the
scorecard, and logs the result.

### Step 7 — Fix the random seeds

Training is stochastic. Fixing the seeds makes the exercise repeatable. Note that full
bit-for-bit determinism is off by default in this repo because it costs roughly 20% in speed;
there is a middle setting that warns instead, which is the one to use when comparing runs.

### Step 8 — Train

Two passes over the data. The model is tiny, so nearly all the time goes into reading images,
not computing.

### Step 9 — Read the results

The logger writes a CSV with every validation score per epoch. The overlap figure is the
number to carry into Notebook 2.

---

## What "the baseline image" is

The final cell produces the picture that anchors the entire exercise: **three panels side by
side** for one held-out timestamp.

| Panel | What it shows |
|---|---|
| **Left** | The model's *confidence map* — a heat image where each pixel's brightness is its estimated probability of being active region, from 0 to 1. Smooth and continuous. |
| **Middle** | The same map cut hard at 0.5: everything above becomes white, everything below black. **This is what gets scored.** |
| **Right** | The true outline from the answer key. |

The left panel is the one worth staring at, and it is the bridge to every question about
error bars.

**The model does not actually output an outline.** It outputs a continuous confidence surface.
The middle panel is that surface after all gradation has been thrown away. Every hesitant,
in-between value — which is precisely where the model is telling you it is unsure — gets
flattened into a confident yes or no *before* anything is measured.

Recovering that discarded information is the basis of everything in the error-estimation
section below.

---

## Notebook 2 — fine-tuning Surya

`2_finetune_ar.ipynb`

This notebook is identical to Notebook 1 except for the model. That is the entire design, and
the notebooks say so explicitly.

### Steps 1–3 — Identical

Same configuration, same data loaders, same batches.

### Step 4 — Build Surya with an outline head

The **backbone** is a 366-million-parameter model pre-trained on a large archive of solar
imagery. It chops each image into 16 × 16 tiles and builds a rich internal description of
each one, using two kinds of processing block: two that filter the image globally in terms of
its spatial frequencies, and eight that let tiles exchange information both locally and
across the whole disk.

Because outlining is a picture-in, picture-out task, this uses the two-dimensional variant: a
thin **head** converts each tile's internal description back into pixel-level scores.

Two consequences the notebook flags honestly:

- The head's name must begin with `head_`, because the adapter machinery finds trainable
  pieces by that prefix. Get it wrong and you train adapters feeding a frozen, random readout.
- Because the head is linear within each tile, **outlines are blocky at 16-pixel granularity.**
  That is a floor imposed by the architecture, not a data problem, and it caps how finely any
  boundary claim can be made.

### Step 5 — Load the pre-trained weights

The saved checkpoint stores its weights under flat names, while the fine-tuning model nests
the backbone one level deeper. The loader tries both spellings, skips anything whose shape
does not match, and reports how many weights matched. **A low count means the architecture in
the config does not match the checkpoint** — check it rather than training on a half-loaded
model.

### Step 6 — Choose a training regime

Three options, all selected from the config:

| Setting | Regime | What trains |
|---|---|---|
| `use_lora: true` | Small adapters plus the head (**default**) | ~3 M parameters |
| `use_lora: false`, `freeze_backbone: true` | Linear probe | The head only |
| `use_lora: false`, `freeze_backbone: false` | Full fine-tuning | All 366 M |

The default freezes all 366 M pre-trained parameters and inserts small adapter matrices into
the attention and feed-forward layers, training under 1% of the model. The notebook prints
exactly which modules received adapters and which head pieces stay trainable. **Read that
printout** — a misspelled target name is silently ignored by the adapter library, and you
would be training nothing.

### Steps 7–9 — Shape check, same scorecard, same wrapper, same seeds

The model's output has the same shape as the baseline's, so the scorecard and training loop
need no changes at all.

### Step 10 — Train

Uses reduced-precision arithmetic to roughly halve memory, and recomputes intermediate values
on demand rather than storing them. One 13-channel 4096 × 4096 sample plus the backbone needs
a large GPU; keep the batch size at 1.

### Step 11 — Compare

Put the two runs side by side. **The gap in overlap score is the answer** — the measured value
the foundation model adds over a 14-parameter pixel classifier, under identical data,
identical splits, and identical scoring.

### Step 12 — The same three-panel picture

Same figure as the baseline, so the two can be compared by eye as well as by number.

> ⚠️ **As shipped, neither notebook produces a scientifically meaningful number.** The config
> caps the dataset at 10 samples and the notebooks run 2 passes. That is a plumbing test.
> Real numbers need the sample cap raised substantially and more epochs.

---

## Notebook 3 — measuring the error

`3_errors_ar.ipynb`

Notebooks 1 and 2 each end with one number. This notebook asks how much that number can be
trusted, and the honest answer has four parts: would a different handful of images give a
different score; is the model's confidence honest; would training again give a different
model; and is the answer key precise enough to be measured against at this resolution. It
replaces the single number with an interval for each, then asks whether the gap between two
models is bigger than all of them.

It is the most detailed of the four and is documented on its own in
`ERRORS_NOTEBOOK_EXPLAINED.md`; every one of its cells also carries its explanation inline.
What follows is the shape of it.

### Step 1 — Choose a budget, and the five knobs

One cell holds everything adjustable. `BUDGET` picks `smoke` (8 images, minutes, numbers that
are noise by design), `medium` (60 images, hours, enough to tell whether the setup is sound)
or `real` (400 images, days). `MODEL_KIND` switches between the foundation model and the
14-number baseline. Four more knobs exist because of what the first runs of this notebook
revealed — see "What it found" below. The learning rate is chosen per model here, honouring the
config's own note that the foundation model needs ten times the gentleness of the baseline.

### Step 2 — Choose *which* images, not just how many

The dataset sorts by date, so a naive cap takes the earliest images — January 2011, near solar
minimum, one active-region pixel in a thousand. Coverage three years later is one in fifty-five.
`SAMPLE_SELECTION` restricts both training and validation to a date window by default; a
`stratified` option spreads across the whole cycle instead. There is deliberately no option to
pick the validation images with the most active regions, because that would make every score
look better than it is.

### Step 3 — Measure the imbalance, then weight for it

The notebook reads the selected outlines and reports the true coverage, the imbalance, and a
weight derived from it that makes each active-region pixel count for more. Without this, a
model can do well on the training objective by answering "background" everywhere.

### Step 4 — Warm the cache in parallel

Every image the run needs is downloaded up front, sixteen at a time, so training never stalls
on a file. It reuses the loader's own naming and download code; already-cached files are
skipped. The container's real memory and CPU limits are read from `/sys/fs/cgroup/` — not from
`free` and `nproc`, which describe the whole node — and the number of loading processes is sized
from them, with the arithmetic printed.

### Step 5 — Train the members

One function builds the model, another trains it under a seed and caches the result under a
key that includes every setting affecting training. It keeps the epoch with the lowest
validation loss rather than the last one, stops early once that stops improving, and adopts a
matching checkpoint from an earlier run instead of retraining if one exists. This is the only
expensive cell. Member 0 is the model the next three steps use; the rest exist for Step 10.

### Step 6 — Cache the predictions as histograms

One pass over the validation images stores, per image, a histogram of the model's raw scores
split by true label — a few thousand numbers instead of a 16.8-million-pixel map. From that,
the overlap at *any* threshold, the reliability diagram and the calibration all follow exactly
and instantly. Full maps are kept only for the two or three images the contour figures need.

### Step 7 — Bootstrap intervals

The score is computed from a sample, so it has sampling error. Re-drawing the images with
replacement a few thousand times gives a 95 % interval. Two versions — pooled over pixels, and
per image — because they answer different questions. A second plot shows how the interval
narrows as images are added, which is how to decide how many you need.

### Step 8 — Reliability diagram and calibration

Of all pixels the model scored near 0.7, were 70 % really active region? The diagram answers
that; the fix is two numbers fitted on a random half of the validation set and reported on the
other — random, because the validation images are in time order and the Sun rotates, so the
first half is systematically easier than the second (0.47 versus 0.37 on the sixty-image run). Calibration barely changes the achievable overlap — it only rescales — but it is what
makes the next step's contours mean something.

### Step 9 — Confidence contours

The boundary is drawn at 0.25, 0.5 and 0.75 on the calibrated map. Where the curves coincide
the model is decisive; where they spread, the band is its uncertainty on the outline. Nothing
finer than the model's 16-pixel tile is meaningful, and the figures respect that.

### Step 10 — An ensemble of seeds

Training again from a different random start gives a slightly different model. The members'
average is a better prediction than any one of them; their disagreement, pixel by pixel, is a
genuine uncertainty map that lights up along region boundaries. The spread of the *score*
across seeds is compared with one seed's bootstrap width: if they are similar, a single run's
number cannot be quoted to the precision people usually quote it at.

### Step 11 — The label floor

The outlines come from a rule with arbitrary settings. The notebook re-runs that rule at ±40
and ±60 gauss and measures how far the "truth" moves. Two equally defensible answer keys that
agree only to some level set a floor below which no model score can be resolved. Because the
rule is re-implemented from its published description, the notebook first scores the
re-implementation against the real outlines and computes the floor only where they agree.

### Step 12 — Everything against the floor

A final table: every score with its interval, next to the floor. The question it answers is
whether the gap between models is larger than the uncertainty in measuring it *and* larger
than the ambiguity in the definition it is measured against.

### What it found

The first run of the notebook collapsed — and diagnosing that is where three of the five knobs
came from.

| run | epoch | overlap | precision | recall | what happened |
|---|---|---|---|---|---|
| smoke, original | 1 | 0.000035 | 1.00 | 0.000035 | loss kept improving; the model stopped predicting anything |
| smoke, fixed | 1 | 0.041 | 0.40 | 0.043 | off the floor; still an 8-image model |
| medium, seed 0 | 0 | 0.435 | 0.52 | 0.73 | a real segmentation from the first pass |
| medium, final analysis | best epoch | 0.437 [0.424, 0.449] | | | best calibrated cut 0.465; seeds 0.459 / 0.417; label floor 0.632 |

The three causes: the images were the quietest in the archive (a naive date cap), the rare
pixels were not up-weighted, and the foundation model was training at ten times its
recommended rate. The sixty-image run then showed a fourth thing: the best epoch was the
second of ten, and the notebook had been caching the last — fixed, with early stopping added. The lesson is the one the notebook exists to teach — the loss going down told
us nothing; precision and recall together told us everything.

Two things it found about the *answer key* are recorded in `GROUND_TRUTH_EXPLAINED.md`: the
published dilation is a square brush, not a disc (the notebook was corrected to match), and
the rule does not reproduce the May 2010 outlines from the magnetograms served here, so the
floor is never computed from them.

---

## Estimating error: confidence contours on the outlines

*This section was written before Notebook 3 existed and explains the reasoning behind it. The
notebook implements it: Source 1 is Steps 8–9, Source 2 is Step 10 (adapter ensembles; the
test-time augmentation and dropout ideas below remain future work — both dropout settings in
the config are 0.0), Source 3 is Step 11, and "Error bars on the score itself" is Step 7. The
per-region framing at the end is not yet implemented.*

You want to know not just *where* the model thinks an active region is, but *how sure it is* —
and how much of any disagreement with the answer key is the model's fault versus the answer
key's.

There are **three distinct sources of error**, and they must be separated before any of them
can be quantified.

### Source 1 — The model's own hesitancy

**This is already being computed and thrown away.** The left panel of the three-panel figure
is a full confidence surface; the cut at 0.5 discards it.

**Confidence contours, directly.** Instead of drawing one outline at 0.5, draw several — at
0.25, 0.5 and 0.75. You get three nested curves. Where they sit almost on top of one another,
the model is decisive. Where they spread into a wide band, it is hedging. That band **is** the
error bar on the outline. It costs nothing: no retraining, no extra forward passes. Shade
between the outer two and you have a publication-ready uncertainty figure.

**Calibrate first — this is not optional.** Neural networks are notoriously overconfident.
Before "0.75" can be read as "75% likely," check it: take all held-out pixels the model scored
near 0.75 and ask what fraction were *actually* marked. Plot that across all confidence levels
(a reliability diagram). If the curve bends away from the diagonal, fit a single correction
number on held-out data — temperature scaling — and apply it before drawing anything.

> Until calibration is done, confidence contours are decorative. They will look far tighter
> than reality.

**A hard floor to respect.** Because of the 16-pixel tile head, contour positions are only
meaningful down to about 16 pixels. Do not report boundary uncertainty finer than that; the
architecture cannot express it.

### Source 2 — Which model you happened to train

A single trained model gives one answer. Train again with a different random seed and you get
a slightly different answer. That spread is real uncertainty, and a single run cannot see it.

**Adapter ensembles — the best fit for this setup by a wide margin.** Because the default
regime freezes the 366 M backbone and trains only ~3 M adapter parameters, you can train five
adapter sets from different seeds for close to the cost of one full fine-tune, and store all
five cheaply. At prediction time, run all five over the same image:

- The **average** confidence map is a better prediction than any single one.
- The **spread across the five** at each pixel is a genuine uncertainty map. It will light up
  almost entirely along region *boundaries* — which is the honest statement: the models agree
  a region exists and disagree about exactly where it ends.
- Contours drawn on the average, with a band from the spread, are defensible in a way
  single-model contours are not.

The config already has an `ensemble` slot sitting unused.

**Test-time augmentation — the cheapest version.** Flip or slightly rotate the input, predict,
flip the prediction back, repeat a few times, average. The spread across those is a quick
uncertainty proxy with no retraining at all.

One caution: the pipeline deliberately disables vertical flips during *training*, because the
base loader would flip the image without flipping the outline and silently misalign the label.
At prediction time you control both, so it is safe — but keep rotations small, since the Sun's
appearance genuinely varies from disk center to limb.

**Dropout-based sampling — note it needs enabling.** The standard trick is to leave dropout
switched on at prediction time and sample repeatedly. In this config **both dropout settings
are 0.0**, so this yields exactly zero spread until you change them and retrain. Worth knowing
before reaching for it.

### Source 3 — The answer key itself

This is the source to prioritize, and the one the current setup is completely blind to.

Those outlines are not ground truth. They are the output of a rule with **arbitrary knobs**:
±50 gauss, 100-pixel minimum, 10-pixel growth. Nobody in the Sun decided on 50 gauss.

**So propagate that.** Regenerate the outlines at ±40 and ±60 gauss, and with 5- and 15-pixel
growth. Measure how much the "truth" moves. You now have a **band on the answer key**, and
from it an **irreducible error floor**: if the target itself shifts by some amount when you
nudge a threshold nobody can justify to three digits, then no model can be meaningfully scored
tighter than that. A model scoring 0.72 overlap against a target that wobbles by ±0.06 is not
distinguishable from one scoring 0.76.

This reframes the exercise productively:

- Where the model disagrees with the answer key **inside** the label-wobble band, it is not
  wrong — it is within the definition's own ambiguity.
- Where it disagrees **outside** that band, that is genuine model error worth investigating.

It also gives the single-polarity false-positive question a real test. Are the regions the
model flags ones that the ±40 gauss version of the rule also picks up? If so, the model found
something and the rule excluded it.

### Error bars on the score itself

Separate from outline uncertainty: the reported overlap number is a **statistic computed from
a sample**, so it carries sampling error.

- **Bootstrap it.** Compute the overlap per held-out timestamp, then resample those timestamps
  with replacement a thousand times to get a confidence interval. Report `0.71 (0.64–0.77)`,
  never a bare number.
- **This is how the Surya-versus-baseline claim becomes honest.** The gap between the two is
  only real if the intervals do not overlap. With the shipped 10-sample cap they will overlap
  hugely.
- **Report two versions.** The built-in score pools across all pixels in a batch, so large
  regions dominate and quiet images barely count. Also compute a per-image average. They
  answer different questions, and the per-image one is usually what a scientist means.

### A more physical framing

Per-pixel overlap may not be the quantity you actually care about. Scientists usually want:
*did we find the region, and did we get its size and position right?*

Group the predicted pixels into connected blobs, do the same for the true outline, and match
them up. That yields:

- **Detection rate** — what fraction of real active regions were found at all;
- **False alarm rate** — how many blobs were invented;
- **Per-region errors** — centre position off by how many pixels, area off by what percentage.

Uncertainty now becomes per-region and physically interpretable: *"this region's area is
4,200 ± 600 pixels; its centre is uncertain by ±20 pixels."* That is far more useful to a
space-weather forecaster than a single disk-averaged overlap score, and confidence contours
express it naturally.

---

## A concrete order of work

This was the plan; all six items are now implemented in `3_errors_ar.ipynb`.

| # | item | where | status and what it found |
|---|---|---|---|
| 1 | Raise the sample cap and epoch count | Step 1 (`BUDGET` presets) | Done. Raising the cap alone was not enough — *which* images mattered more, hence Step 2. |
| 2 | Bootstrap intervals on the scores | Step 7 | Done, pooled and per-image, plus the width-vs-count plot. At 8 images the interval spans most of the range the score can take. |
| 3 | Reliability diagram and calibration | Step 8 | Done, fitted on half the validation set and reported on the other half. |
| 4 | Multi-level contours on the calibrated map | Step 9 | Done, at 0.25 / 0.5 / 0.75, respecting the 16-pixel floor. |
| 5 | An ensemble of adapter seeds | Step 10 | Done; the preset chooses two or five members. |
| 6 | Regenerate the answer key at ±40 / ±60 G | Step 11 | Done, and gated on the re-implementation agreeing with the real outlines first. |

Items 2–4 needed no new training. Item 6 remains the one that changes how everything else is
read — and it also surfaced the two answer-key findings recorded in `GROUND_TRUTH_EXPLAINED.md`.

---

## Quick reference

### Files in this app

| File | Role |
|---|---|
| `0_dataset_dataloader_ar.ipynb` | Pairing images with outlines; inspecting and plotting samples |
| `1_baseline_ar.ipynb` | The 14-parameter baseline, the scorecard, a training loop |
| `2_finetune_ar.ipynb` | Surya plus adapters, with the same data, scorecard and loop |
| `3_errors_ar.ipynb` | Error bars, calibration, confidence contours, ensembles, label floor |
| `ERRORS_NOTEBOOK_EXPLAINED.md` | Plain-language guide to that notebook |
| `GROUND_TRUTH_EXPLAINED.md` | How the answer-key outlines are made, and how faithfully the rule reproduces |
| `RESULTS_REPORT.md` | Every result measured so far, in one place |
| `configs/config_script.yaml` | The single run configuration |
| `configs.py` | The task-specific configuration keys |
| `datasets/ar_dataset.py` | Exact-timestamp pairing of image index and outline index |
| `models/pixel_logistic.py` | The baseline: one weighted vote per pixel |
| `metrics/ar_metrics.py` | The shared scorecard |
| `lightning_modules/pl_segmentation.py` | The shared training loop |
| `finetune_ar_segmentation.py` | Command-line training script for longer runs |
| `download_masks.sh` | Fetches and extracts the answer-key dataset |
| `diagrams/ar_pipeline.html` | Interactive flow chart of the whole pipeline |

### Numbers worth remembering

| Quantity | Value |
|---|---|
| Image size | 4096 × 4096 |
| Channels | 13 (8 ultraviolet, 5 magnetic) |
| Image cadence | every 12 minutes |
| Outline cadence | hourly — so only 1 image in 5 can pair |
| Active-region coverage | 0.02 %–1.8 % of the frame by year: ~55:1 at solar maximum, >4,000:1 at minimum |
| Baseline size | 14 parameters |
| Backbone size | 366 M parameters |
| Trained under the default regime | ~3 M (under 1%) |
| Tile size, and the boundary resolution floor | 16 pixels |
| Coin-flip penalty value | 0.69 |
| Re-implementation of the labelling rule | overlap ≈ 0.85–0.92 with the real outlines on good days; fails on May 2010 |
| Container limits the sizing uses | read from `/sys/fs/cgroup/` — here 60 GB and 7 cores, not the node's 124 GB and 16 |

### Config keys specific to this task

| Key | Meaning |
|---|---|
| `ar_mask_dir` | Directory holding the outline catalogs and files |
| `ar_mask_splits` | Which catalogs to load (the image index decides train vs validation) |
| `ar_mask_key` | Region footprints, or the dividing lines themselves |
| `ar_pooling` | Shrink images and outlines so the baseline runs on a small machine |
| `ar_pos_weight` | Up-weight the rare active-region pixels in the training penalty |
| `ar_threshold` | The probability cut used when scoring |
