#!/usr/bin/env python
"""Best-generalized epoch per member for a paired pair of ensemble arms, from wandb datastores.

Mechanizes the eight rules under "HOW TO TELL IF A RUN IS OVER- OR UNDERTRAINED" in NOTES, which
four separate entries in that file had each re-derived by hand. Produces the (epoch, val_iou,
run_id) tuples `run_bootstrap_inference.BEST_EPOCH_BY_REGIME` needs, so the ensembles can be
inferred at a checkpoint that is neither over- nor under-trained rather than at `best.pth` --
which tracks valid/iou and is therefore systematically the worst-calibrated checkpoint of its run
(rule 8).

Two arms per invocation, paired replicate-for-replicate: a `--sweep-arm` and the `--baseline-arm`
it is differenced against. Arm i of each must train the SAME replicate, which is asserted from
each arm's own ensemble_index.csv rather than assumed. An arm is named by its directory under
/efs/bathomas/configs_bootstrap -- that name is what each run's wandb-metadata.json records, so
member -> run discovery is by provenance rather than by a hand-maintained id list.

WHAT THE PAIRED DIFFERENCE MEANS IS NOT A PROPERTY OF THIS SCRIPT. It is whatever the two arms
actually differ in, which for every pair tried so far is more than the one factor named in the
arm's directory name. `CONFOUND_NOTES` records the known confound per pair and the report prints
it; a pair with no entry there prints a reminder to diff the configs before reading the sign.

Reads the `.wandb` datastores rather than the logs: the logs round validation IoU to 4 decimals,
which is coarser than the jitter scale rule 3 has to estimate. `parse_wandb_history.parse` does
the byte-level scan, and every value is cross-checked against the run's own output.log so a
mis-mapped run is caught instead of averaged in.

CPU-only, seconds to run, no torch.

Examples:
    # the two 168-sample arms -- reproduces the blocks already in run_bootstrap_inference.py
    python epoch_select.py --sweep-arm 168hrs_stable_seed_diag --baseline-arm 168hrs \\
        --out /efs/bathomas/ar_seg_bootstrap/168hrs_epoch_selection.csv

    # the 672-sample seed sweep against its stabilized parent
    python epoch_select.py --sweep-arm 672hr_stable_seed_diag --baseline-arm 672hr_stable
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

# parse() lives beside the ensembles on EFS rather than in the repo; it is the only reader that
# recovers full-precision history without a live wandb service.
_WANDB_TOOLS = "/efs/bathomas"
if _WANDB_TOOLS not in sys.path:
    sys.path.insert(0, _WANDB_TOOLS)
from parse_wandb_history import parse  # noqa: E402

WANDB_ROOT = "/efs/bathomas/wandb"
CONFIG_ROOT = "/efs/bathomas/configs_bootstrap"

# 168hrs boot01 ran a day before its config set was regenerated, under a path with no regime
# segment. Documented in ar_seg_bootstrap/168hrs/README.md; everything else about it matches
# boot02-boot10. Inert unless 168hrs is one of the two arms.
LEGACY_CONFIG_PATHS = {"./configs_bootstrap/config_boot01.yaml": ("168hrs", "boot01")}

# What each arm pair actually differs in, keyed (sweep, baseline). Printed with the paired
# difference so the sign is never read as the factor the arm is named after. A pair absent here
# gets a warning instead, because "we did not check" and "nothing else differs" are not the same
# claim and this script cannot tell them apart.
CONFOUND_NOTES = {
    ("168hrs_stable_seed_diag", "168hrs"): [
        "this difference is seed AND LR schedule together -- the clean seed-0 parent",
        "168hrs_stable was never trained, so the two factors cannot be separated here.",
        "boot01 of the sweep is also on a 10-epoch cosine while boot02-10 are on 4-epoch ones.",
    ],
    ("672hr_stable_seed_diag", "672hr_stable"): [
        "this difference is seed AND epoch budget together, and the budget is now MIXED rather",
        "than uniformly short: boot06/07/09/10 were retrained at the parent's 6 epochs on",
        "2026-11-03/04, while boot01 stays at 5 and boot02-05/08 at 4. So 6 of 10 pairs are",
        "still biased NEGATIVE by roughly the +0.0104 the parent earns over the epochs they",
        "never ran, and 4 are budget-matched -- the bias is now unevenly distributed across",
        "members, which is worse for a paired test than a uniform offset. Retraining the",
        "remaining six at 6 epochs is what retires this note; the pair then differs only in",
        "`seed`. Note the retrain did NOT raise those four (boot07 -0.0143, boot09 -0.0130 at",
        "their chosen epochs), so a budget-matched arm is not automatically a better one.",
        "",
        "The HOLDOUT contrast (view_seed_diag_672hr_stable_sections67.ipynb) removes one",
        "component of this confound but not this one. It reads the baseline through the",
        "672hr_stable_generalized key of run_bootstrap_inference.py -- the same ten",
        "672hr_stable checkpoints at the picks THIS script writes for the baseline arm -- so",
        "the selection criterion is matched across arms and only seed and budget remain. That",
        "key is a re-selection, not a config directory, so it can never appear as an --*-arm",
        "here; the pair above is the only form this contrast takes in this script.",
    ],
    ("672hr_stable_pxfix", "672hr_stable"): [
        "the twin differs from its parent ONLY in data.fix_corner_scaling (plus the _pxfix job_id",
        "and checkpoint path) -- same replicates, seed, LR schedule, epochs and LoRA; see",
        "make_pxfix_configs.py. The pretrained backbone still learned from OFFSET inputs, so the",
        "difference includes LoRA absorbing the pretraining-to-aligned shift, not only the cleaner",
        "registration. Each arm must also be scored under its own setting (score_checkpoints.py",
        "and run_bootstrap_inference.py read it from the member config).",
    ],
    ("168hrs_stable_pxfix", "168hrs_stable"): [
        "the twin differs from its parent ONLY in data.fix_corner_scaling (plus the _pxfix job_id",
        "and checkpoint path) -- same replicates, seed, LR schedule, epochs and LoRA; see",
        "make_pxfix_configs.py. The pretrained backbone still learned from OFFSET inputs, so the",
        "difference includes LoRA absorbing the pretraining-to-aligned shift, not only the cleaner",
        "registration. Each arm must also be scored under its own setting (score_checkpoints.py",
        "and run_bootstrap_inference.py read it from the member config).",
    ],
    ("672hr_stable_seed_diag_pxfix", "672hr_stable_seed_diag"): [
        "the twin differs from its parent ONLY in data.fix_corner_scaling (plus the _pxfix job_id",
        "and checkpoint path); seeds, replicates and per-member epoch budgets are copied as-is,",
        "so any budget mix in the parent set is reproduced, not removed. Backbone caveat as for",
        "the 672hr_stable_pxfix pair.",
    ],
    ("672hr_stable_seed_diag_pxfix", "672hr_stable_pxfix"): [
        "the fix-on mirror of (672hr_stable_seed_diag, 672hr_stable): copied configs, so it",
        "inherits that pair's confounds (see its entry) rather than being a clean seed contrast.",
    ],
    ("168hrs_stable_seed_diag_pxfix", "168hrs_stable_seed_diag"): [
        "the twin differs from its parent ONLY in data.fix_corner_scaling (plus the _pxfix job_id",
        "and checkpoint path); seeds, replicates and per-member epoch budgets are copied as-is,",
        "so any budget mix in the parent set is reproduced, not removed. Backbone caveat as for",
        "the 168hrs_stable_pxfix pair.",
    ],
    ("168hrs_stable_seed_diag_pxfix", "168hrs_stable_pxfix"): [
        "the fix-on mirror of (168hrs_stable_seed_diag, 168hrs_stable): copied configs, so it",
        "inherits that pair's confounds (see its entry) rather than being a clean seed contrast.",
    ],
}

# Rule thresholds, named once so the report can print them and a reader can disagree with them
# explicitly rather than by reverse-engineering the code.
K_SIGMA = 2.0        # a gain or deficit must exceed this many jitter sigmas to be called real
LOSS_TOL = 0.10      # val loss "materially above its floor" (rule 6)
ALPHA = 0.05         # significance for the trend tests (rules 1, 4)


def dedupe(values: list) -> list:
    """Every history item lands twice in the datastore; keep every other.

    Verified against the `epoch` key, which must come back as 0..n-1 after deduping -- that is
    the assertion in `load_run`, not an assumption here.
    """
    return values[::2]


def discover_runs(arms: list) -> dict:
    """{arm: {member: run_id}} from every run's wandb-metadata.json --config_path."""
    found = {arm: {} for arm in arms}
    for meta_path in sorted(glob.glob(os.path.join(WANDB_ROOT, "run-*", "files",
                                                   "wandb-metadata.json"))):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(meta_path))).split("-")[-1]
        try:
            args = json.load(open(meta_path)).get("args") or []
        except (json.JSONDecodeError, OSError):
            continue
        if "--config_path" not in args:
            continue
        config_path = args[args.index("--config_path") + 1]

        if config_path in LEGACY_CONFIG_PATHS:
            arm, member = LEGACY_CONFIG_PATHS[config_path]
            if arm in found:
                found[arm][member] = run_id
            continue

        match = re.match(r"\./configs_bootstrap/([^/]+)/config_(boot\d+)\.yaml$", config_path)
        if not match:
            continue
        config_dir, member = match.groups()
        if config_dir in found:
            found[config_dir][member] = run_id
    return found


def load_run(run_id: str) -> dict:
    """Per-epoch history for one run, cross-checked against its own output.log."""
    (run_dir,) = glob.glob(os.path.join(WANDB_ROOT, f"run-*-{run_id}"))
    history = parse(os.path.join(run_dir, f"run-{run_id}.wandb"))
    out = {key: dedupe(values) for key, values in history.items()}

    epochs = out.get("epoch", [])
    assert epochs == list(range(len(epochs))), (
        f"{run_id}: deduped epoch index is {epochs[:12]}, expected 0..n-1 -- the two-copies-"
        f"per-item layout this reader assumes does not hold for this datastore"
    )

    log_path = os.path.join(run_dir, "files", "output.log")
    log_iou = [float(line.split("IoU:")[1].split()[0])
               for line in open(log_path, errors="ignore").read().splitlines()
               if "Validation —" in line]
    iou = out["valid/iou"]
    agree = len(log_iou) == len(iou) and all(
        abs(round(a, 4) - b) < 5e-5 for a, b in zip(iou, log_iou)
    )

    out["_run_id"] = run_id
    out["_dir"] = run_dir
    out["_n_epochs"] = len(iou)
    out["_agree"] = agree
    return out


def jitter_sigma(iou: np.ndarray) -> float:
    """Rule 3: sigma =~ sd(successive IoU deltas)/sqrt(2), excluding the ep0->ep1 learning jump.

    Two successive epochs each carry independent jitter, so the sd of their difference is
    sqrt(2) times the sd of a single epoch's -- hence the divisor.
    """
    deltas = np.diff(iou)[1:]
    if len(deltas) < 2:
        return float("nan")
    return float(np.std(deltas, ddof=1) / np.sqrt(2))


def trend(values: np.ndarray, lo: int) -> tuple:
    """Spearman rho and its p over epochs [lo, end]. Rule 4: test the trend, do not eyeball."""
    y = values[lo:]
    if len(y) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(np.arange(len(y)), y)
    return float(rho), float(p)


def plateau_start(iou: np.ndarray) -> int:
    """Earliest epoch from which the IoU trend to the end is no longer significant.

    Everything from there on is statistically tied, which is what licenses picking the cheapest
    epoch in the window rather than the argmax (rule 4).
    """
    for lo in range(1, max(len(iou) - 2, 2)):
        rho, p = trend(iou, lo)
        if not np.isfinite(p) or p >= ALPHA:
            return lo
    return max(len(iou) - 3, 1)


def diagnose(iou: np.ndarray, vloss: np.ndarray, tloss: np.ndarray) -> dict:
    """Apply rules 1, 3, 4, 6 to one member's curves and choose an epoch."""
    n = len(iou)
    sigma = jitter_sigma(iou)
    floor_ep = int(np.argmin(vloss))
    best_iou_ep = int(np.argmax(iou))

    # Rule 1: overfitting moves train and val loss APART; divergence moves them together.
    # Measured from the loss floor onward, not over the whole run: every run falls steeply from
    # its random head over epoch 0->1, and including that descent makes the full-range trend
    # report "no degradation" for a run that is plainly overfitting after its floor.
    v_rho, v_p = trend(vloss, floor_ep)
    t_rho, t_p = trend(tloss, floor_ep)
    ratio = vloss / np.where(tloss > 0, tloss, np.nan)
    r_rho, r_p = trend(ratio, floor_ep)
    if not np.isfinite(v_rho) or v_rho <= 0:
        regime = "no-degradation"
    elif t_rho > 0 and (not np.isfinite(r_p) or r_p >= ALPHA):
        regime = "divergence"
    elif t_rho < 0:
        regime = "overfitting"
    else:
        regime = "no-degradation"

    # Rule 6, lower edge: the last epoch that bought a real IoU gain. Measured against the
    # RUNNING MAXIMUM, not the previous epoch -- a gain only counts if it takes new high ground.
    # A rebound out of a dip is the cancellation signature rule 3 names as jitter (the 168-sample
    # sweep's boot01 drops 0.0329 at epoch 6 and recovers 0.0321 at epoch 7), and scoring
    # successive deltas would read that recovery as a real gain and push the window to epoch 7.
    # When no epoch ever took new high ground, the lower edge is 0, not 1: nothing after epoch 0
    # bought a real gain, so epoch 0 is a legitimate candidate -- and for that sweep's boot06 it
    # is the checkpoint that dominates on BOTH criteria (highest IoU of its window and the loss
    # floor itself), which a default of 1 would have silently excluded.
    real = [e for e in range(1, n)
            if np.isfinite(sigma) and iou[e] - iou[:e].max() > K_SIGMA * sigma]
    window_lo = max(real) if real else 0

    # Rule 6, upper edge: the first epoch AFTER the floor whose val loss is materially above it.
    # Restricted to post-floor epochs because epoch 0 is usually well above the floor too -- that
    # is the pre-floor descent, not overfitting, and counting it collapses the window onto 0.
    over = [e for e in range(floor_ep + 1, n) if vloss[e] > vloss[floor_ep] * (1 + LOSS_TOL)]
    window_hi = min(over) if over else n - 1
    window_lo = min(window_lo, window_hi)

    # Within the window, take the earliest epoch whose IoU is within noise of the run maximum:
    # cheapest probability fit among the checkpoints that are statistically tied on overlap.
    candidates = [e for e in range(window_lo, window_hi + 1)
                  if np.isfinite(sigma) and iou.max() - iou[e] <= K_SIGMA * sigma]
    if candidates:
        chosen = candidates[0]
        reason = (f"earliest epoch in window {window_lo}-{window_hi} within "
                  f"{K_SIGMA:g} sigma of max IoU")
    else:
        chosen = int(window_lo + np.argmax(iou[window_lo:window_hi + 1]))
        reason = f"max IoU inside window {window_lo}-{window_hi} (none within noise of run max)"

    p_rho, p_p = trend(iou, plateau_start(iou))
    return dict(
        n_epochs=n, chosen_epoch=chosen, window_lo=window_lo, window_hi=window_hi,
        best_iou_epoch=best_iou_ep, loss_floor_epoch=floor_ep, sigma=sigma, regime=regime,
        plateau_from=plateau_start(iou), plateau_rho=p_rho, plateau_p=p_p,
        vloss_rho=v_rho, vloss_p=v_p, tloss_rho=t_rho, tloss_p=t_p,
        ratio_rho=r_rho, ratio_p=r_p,
        iou_chosen=float(iou[chosen]), iou_best=float(iou[best_iou_ep]),
        iou_deficit_sigma=float((iou.max() - iou[chosen]) / sigma) if sigma else float("nan"),
        vloss_chosen=float(vloss[chosen]), vloss_floor=float(vloss[floor_ep]),
        vloss_excess_pct=float(100 * (vloss[chosen] / vloss[floor_ep] - 1)),
        vloss_excess_at_best_iou_pct=float(100 * (vloss[best_iou_ep] / vloss[floor_ep] - 1)),
        reason=reason,
    )


def replicate_of(arm: str, member: str) -> str:
    """The replicate CSV a member trained on, from its arm's ensemble_index.csv."""
    index = pd.read_csv(os.path.join(CONFIG_ROOT, arm, "ensemble_index.csv"))
    row = index[index.config.str.contains(f"config_{member}.yaml", regex=False)]
    return os.path.basename(row.ar_index_train.iloc[0]) if len(row) else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-arm", required=True,
                    help="Config directory name of the arm under test, e.g. "
                         "672hr_stable_seed_diag.")
    ap.add_argument("--baseline-arm", required=True,
                    help="Config directory name of the arm it is differenced against, e.g. "
                         "672hr_stable.")
    ap.add_argument("--out", default=None,
                    help="Where the per-member CSV goes. Defaults to "
                         "/efs/bathomas/ar_seg_bootstrap/<sweep-arm>_epoch_selection.csv. The "
                         "168hr pair's existing CSV is named after its BASELINE arm, so pass "
                         "--out explicitly to overwrite that one in place.")
    args = ap.parse_args()

    sweep_arm, base_arm = args.sweep_arm, args.baseline_arm
    arms = [sweep_arm, base_arm]
    out_path = args.out or os.path.join(
        "/efs/bathomas/ar_seg_bootstrap", f"{sweep_arm}_epoch_selection.csv")

    runs = discover_runs(arms)
    print("=== run discovery, from each run's wandb-metadata.json --config_path")
    for arm in arms:
        members = runs[arm]
        print(f"  {arm:26s} {len(members)} members: "
              f"{', '.join(f'{m}={r}' for m, r in sorted(members.items()))}")
    for arm in arms:
        assert runs[arm], f"{arm}: no runs found -- is that the config directory name?"
    assert set(runs[sweep_arm]) == set(runs[base_arm]), (
        f"members differ across arms, so there is nothing to pair: "
        f"{sweep_arm} has {sorted(runs[sweep_arm])}, {base_arm} has {sorted(runs[base_arm])}"
    )

    history = {arm: {m: load_run(r) for m, r in sorted(runs[arm].items())} for arm in arms}

    print("\n=== cross-check: datastore valid/iou vs each run's own output.log")
    for arm in arms:
        members = history[arm]
        bad = [m for m, h in members.items() if not h["_agree"]]
        epochs = {m: h["_n_epochs"] for m, h in members.items()}
        print(f"  {arm:26s} all agree: {not bad}"
              f"{'  MISMATCH: ' + ', '.join(bad) if bad else ''}")
        print(f"  {'':26s} epochs: {epochs}")
        assert not bad, f"{arm}: datastore disagrees with output.log for {bad}"

    print("\n=== replicate pairing: arm i of each sweep must train the same replicate")
    for member in sorted(history[base_arm]):
        reps = {arm: replicate_of(arm, member) for arm in arms}
        same = len(set(reps.values())) == 1
        print(f"  {member}  {reps[base_arm]:22s} {'OK' if same else 'MISMATCH ' + str(reps)}")
        assert same, f"{member}: replicates differ across arms: {reps}"

    print(f"\n=== per-member diagnosis  (K_SIGMA={K_SIGMA:g}, LOSS_TOL={LOSS_TOL:.0%}, "
          f"ALPHA={ALPHA})")
    rows = []
    for arm in arms:
        print(f"\n--- {arm}")
        print(f"  {'mem':8s}{'n':>3s}{'regime':>16s}{'sigma':>8s}{'win':>7s}"
              f"{'pick':>5s}{'IoU':>8s}{'dIoU/s':>8s}{'vloss':>9s}{'+%flr':>7s}"
              f"{'bIoU':>5s}{'flr':>4s}{'+%@bIoU':>9s}")
        for member, h in history[arm].items():
            iou = np.array(h["valid/iou"], dtype=float)
            vloss = np.array(h["valid/loss"], dtype=float)
            tloss = np.array(h["epoch_loss"], dtype=float)
            d = diagnose(iou, vloss, tloss)
            d.update(arm=arm, member=member, run_id=h["_run_id"],
                     replicate=replicate_of(arm, member))
            rows.append(d)
            window = f"{d['window_lo']}-{d['window_hi']}"
            print(f"  {member:8s}{d['n_epochs']:>3d}{d['regime']:>16s}{d['sigma']:>8.4f}"
                  f"{window:>7s}{d['chosen_epoch']:>5d}{d['iou_chosen']:>8.4f}"
                  f"{d['iou_deficit_sigma']:>8.2f}{d['vloss_chosen']:>9.5f}"
                  f"{d['vloss_excess_pct']:>7.1f}{d['best_iou_epoch']:>5d}"
                  f"{d['loss_floor_epoch']:>4d}{d['vloss_excess_at_best_iou_pct']:>9.1f}")

    table = pd.DataFrame(rows)

    print("\n=== rule 2 and 8: the two criteria disagree, and by how much")
    for arm in arms:
        a = table[table.arm == arm]
        n = len(a)
        print(f"  {arm:26s} best-IoU epochs {list(a.best_iou_epoch)}")
        print(f"  {'':26s} loss-floor      {list(a.loss_floor_epoch)}")
        print(f"  {'':26s} chosen          {list(a.chosen_epoch)}")
        print(f"  {'':26s} best.pth sits {(a.best_iou_epoch - a.loss_floor_epoch).mean():.1f} "
              f"epochs past the loss floor on average, costing "
              f"{a.vloss_excess_at_best_iou_pct.mean():+.1f}% val loss")
        # The calibration penalty the pick itself still carries. Stated rather than implicit:
        # choosing inside the rule-6 window bounds this by LOSS_TOL per member, but the mean is
        # the number to quote next to the IoU the pick gives up.
        print(f"  {'':26s} the chosen epochs sit "
              f"{(a.chosen_epoch - a.loss_floor_epoch).mean():+.1f} epochs from the floor, "
              f"costing {a.vloss_excess_pct.mean():+.1f}% val loss "
              f"(worst {a.vloss_excess_pct.max():+.1f}%), and give up "
              f"{(a.iou_best - a.iou_chosen).mean():.4f} IoU against each run's own peak")
        n_differ = int((a.chosen_epoch != a.best_iou_epoch).sum())
        print(f"  {'':26s} chosen != best-IoU for {n_differ}/{n} members")

    print("\n=== training-side spread at the chosen epochs (no GPU needed)")
    for arm in arms:
        a = table[table.arm == arm]
        print(f"  {arm:26s} IoU {a.iou_chosen.mean():.4f} +/- {a.iou_chosen.std(ddof=1):.4f}  "
              f"[{a.iou_chosen.min():.4f}, {a.iou_chosen.max():.4f}]")

    sweep = table[table.arm == sweep_arm].set_index("replicate")
    base = table[table.arm == base_arm].set_index("replicate")
    common = sorted(set(sweep.index) & set(base.index))
    diff = (sweep.loc[common, "iou_chosen"] - base.loc[common, "iou_chosen"]).astype(float)
    w_stat, w_p = stats.wilcoxon(diff)
    t_stat, t_p = stats.ttest_rel(sweep.loc[common, "iou_chosen"].astype(float),
                                 base.loc[common, "iou_chosen"].astype(float))
    print(f"\n  paired per-replicate difference ({sweep_arm} - {base_arm}), n={len(common)}:")
    print(f"    mean {diff.mean():+.4f}  median {diff.median():+.4f}  "
          f"sd {diff.std(ddof=1):.4f}  [{diff.min():+.4f}, {diff.max():+.4f}]")
    print(f"    Wilcoxon p {w_p:.3f}   paired t({len(common) - 1}) = {t_stat:.2f}, p {t_p:.3f}")
    print(f"    sd of the paired difference {diff.std(ddof=1):.4f} vs across-replicate sd "
          f"{base.iou_chosen.std(ddof=1):.4f} "
          f"(ratio {diff.std(ddof=1) / base.iou_chosen.std(ddof=1):.2f})")
    note = CONFOUND_NOTES.get((sweep_arm, base_arm))
    if note:
        for line in note:
            print(f"    NOTE {line}" if line is note[0] else f"         {line}")
    else:
        print("    NOTE no CONFOUND_NOTES entry for this pair. Diff the two config sets before")
        print("         reading the sign of that mean as the factor the arm is named after.")

    var_sweep = sweep.iou_chosen.var(ddof=1)
    var_base = base.iou_chosen.var(ddof=1)
    df = len(common) - 1
    print(f"\n  across-member variance: sweep {var_sweep:.6f}  baseline {var_base:.6f}  "
          f"F = {max(var_sweep, var_base) / min(var_sweep, var_base):.2f} on ({df},{df}) df, "
          f"p = {2 * min(stats.f.cdf(var_sweep / var_base, df, df), stats.f.sf(var_sweep / var_base, df, df)):.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["arm", "member", "run_id", "replicate", "n_epochs", "chosen_epoch", "window_lo",
            "window_hi", "best_iou_epoch", "loss_floor_epoch", "sigma", "regime", "plateau_from",
            "plateau_rho", "plateau_p", "vloss_rho", "vloss_p", "tloss_rho", "tloss_p",
            "ratio_rho", "ratio_p", "iou_chosen", "iou_best", "iou_deficit_sigma", "vloss_chosen",
            "vloss_floor", "vloss_excess_pct", "vloss_excess_at_best_iou_pct", "reason"]
    table[cols].to_csv(out_path, index=False)
    print(f"\nWrote {len(table)} rows to {out_path}")

    # The sweep arm keeps its own name as its regime key; the baseline's re-selection gets a
    # `_generalized` suffix, because the unsuffixed key is the best-IoU record README.md and
    # NOTES cite and must stay untouched.
    print("\n=== BEST_EPOCH_BY_REGIME entries, paste-ready")
    for arm, key in [(sweep_arm, sweep_arm), (base_arm, f"{base_arm}_generalized")]:
        print(f'    "{key}": {{')
        for _, r in table[table.arm == arm].iterrows():
            print(f'        "{r.member}": ({r.chosen_epoch}, {r.iou_chosen:.4f}, "{r.run_id}"),')
        print("    },")


if __name__ == "__main__":
    main()
