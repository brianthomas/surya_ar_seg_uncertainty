# Pixel-shift-fix (`_pxfix`) config twins

Context for the "with fix" half of the comparison described in
`finetune_with_and_without_pixel_shift_fix.txt`. The data bug itself is in
`KNOWN_ISSUE_channel_offset.md`: 8 channels are shifted by corner-anchored disk rescaling.

## What a twin is

Each `_pxfix` config is a copy of an existing config. It differs from its source in exactly
three keys, plus a 4-line header comment:

| key | change | why |
|---|---|---|
| `data.fix_corner_scaling` | `true` (inserted after the `s3_*` keys, or flipped from `false`) | turns on the recentering in `HelioNetCDFDataset` / `read_hc_file` |
| `job_id` | `+ _pxfix` | keeps wandb runs distinguishable |
| `path_experiment` | top-level configs: `+ _pxfix`; bootstrap sets: regime dir renamed, `.../<regime>/bootNN` → `.../<regime>_pxfix/bootNN` | the uncorrected checkpoints can't be overwritten, and `run_bootstrap_inference.py` builds `weights_root` from the regime name |

`corner_scaling_channels` and `corner_scaling_sign` are not written, so the defaults from
`surya/datasets/disk_alignment.py:corner_scaling_kwargs` apply: the 8 affected channels and sign `+1`.
Everything else is identical to the source: seeds, replicates, epoch budgets, LR schedule,
LoRA rank, and the validation/test indices. The originals are untouched, and their
checkpoints must still be evaluated **without** the fix.

## Generator: `make_pxfix_configs.py`

```bash
# single configs -> <name>_pxfix.yaml beside the source
python make_pxfix_configs.py --files config.yaml config_feb15_2013.yaml

# a bootstrap set -> configs_bootstrap/<set>_pxfix/ plus <runner>_pxfix.sh
python make_pxfix_configs.py --src-dir configs_bootstrap/672hr_stable \
    --runner-src run_bootstrap_ensemble_672hr_stable.sh
```

- It edits text line by line rather than round-tripping the YAML, so comments, GENERATED headers and key order
  are preserved, and `diff src dst` shows only the intended lines.
- It reloads every output and fails if the dict differs from the source beyond the three keys.
- It refuses to overwrite an existing file or directory. To regenerate, delete the twin first.
- It copies `ensemble_index.csv` unchanged. In the runner it swaps `configs_bootstrap/<set>/` for `<set>_pxfix/`.

## What exists (created 2026-11-10)

Top-level configs: `config_pxfix.yaml`, `config_feb15_2013_pxfix.yaml`, `config_feb16_2013_pxfix.yaml`,
`config_feb17_2013_pxfix.yaml`, `config_feb15_2013_g4dn_pxfix.yaml`, `config_infer_pxfix.yaml`.

| bootstrap set | runner | checkpoints (`/efs/bathomas/ar_seg_bootstrap/…`) |
|---|---|---|
| `672hr_stable_pxfix` | `run_bootstrap_ensemble_672hr_stable_pxfix.sh` | `672hr_stable_pxfix/bootNN` |
| `168hrs_stable_pxfix` | `run_bootstrap_ensemble_168hrs_stable_pxfix.sh` | `168hrs_stable_pxfix/bootNN` |
| `672hr_stable_seed_diag_pxfix` | `run_seed_ablation_672hr_stable_pxfix.sh` | `672hr_stable_seed_diag_pxfix/bootNN` |
| `168hrs_stable_seed_diag_pxfix` | `run_seed_ablation_168hrs_stable_pxfix.sh` | `168hrs_stable_seed_diag_pxfix/bootNN` |

`run_seed_ablation_168hr_and_672hr_stable_pxfix.sh` chains both seed-diag twins (20 members, 168hrs first).
The bootstrap configs use the training host's `/mnt/efs/bathomas/...` path, which is the same EFS as `/efs/bathomas/...`.

These sets were deliberately not twinned: the archived ones (`*.pre-lrfix.*`, `*.10ep.*`, `*.4ep.*`) and the
unstable `168hrs`, `672hr` and `2016hr`. Each can be added later with one `--src-dir` command.

## Training status (as of 2026-11-13)

- `168hrs_stable_seed_diag_pxfix`: **all 10 members trained**, 2026-11-11 to 13. boot01 has 10 epochs
  and boot02–10 have 4, matching the source set's budgets.
- `672hr_stable_seed_diag_pxfix`: **in progress**. boot01 is at epoch 1.
- `672hr_stable_pxfix`, `168hrs_stable_pxfix` and every top-level twin: not trained.

## Tooling changes made alongside

- `score_checkpoints.py:build_dataset` now applies `corner_scaling_kwargs(config["data"])`, as
  `finetune.py:get_dataloaders` does. Before this it ignored the setting, so it would have scored
  `_pxfix` checkpoints on uncorrected inputs. It is a no-op for every config with the fix off.
- `epoch_select.py:CONFOUND_NOTES` gained entries for `(<set>_pxfix, <set>)` on both stable sets and
  both seed-diag sets. It also gained entries for `(<regime>_stable_seed_diag_pxfix, <regime>_stable_pxfix)`.
  Arm discovery needed no change, because its regex matches `configs_bootstrap/<set>_pxfix/config_bootNN.yaml`.
- `run_bootstrap_inference.py` and `infer_file.py` already read the setting from each member's config.

## Caveats when reading results

1. **The sign is still unverified on a real summer cube** (`KNOWN_ISSUE_channel_offset.md`, "What has
   and hasn't been verified"). The correction was checked only on a January cube that was synthetically
   rescaled to July. If the sign is wrong, the fix-on runs *double* the misregistration on year-spanning
   sets (up to ~140 px instead of ~70). Feb-2013 configs are low-risk (~8–10 px). Check with
   `python check_disk_alignment.py s3://nasa-surya-bench/2011/07/20110701_0000.nc --s3_anon` in the surya env.
   If it shows the shift runs the other way, add `corner_scaling_sign: -1` to the twins' `data:` block
   and retrain.
2. **Twins inherit the source set's confounds.** In `168hrs_stable_seed_diag`, boot01 runs 10 epochs
   and boot02–10 run 4. The seed-diag vs. stable-parent contrast mixes seed with LR schedule and
   epoch budget, in both the fix-off and fix-on arms (see `CONFOUND_NOTES`). The fix-on vs. fix-off
   contrast on the *same* set is clean, apart from caveat 3.
3. **The backbone was pretrained on offset inputs.** A fix-on minus fix-off difference therefore includes LoRA
   absorbing the change from pretraining to aligned inputs, not only cleaner registration.
4. **Evaluate each checkpoint under its own setting.** Twin checkpoints with the fix on, originals with it off.
5. `config_infer_pxfix.yaml` keeps `config_infer.yaml`'s r=32 LoRA, so it can't load the r=8 bootstrap
   checkpoints. Use each member's own config instead, as `score_checkpoints.py` and
   `run_bootstrap_inference.py` already do.

## Verification that was run

- Every twin passed the generator's dict-equality check: it differs from its source only in the three keys.
- `corner_scaling_kwargs` gives `fix_corner_scaling=True`, sign `+1` for all 46 twins, and `False` for every source.
- All seed-diag twins keep their source `seed`.
- `py_compile` passed on the edited scripts, `epoch_select.py --help` runs, and `bash -n` passed on every generated runner.
- Not run: an `--expect-iou` reproduction confirming `score_checkpoints.py` is unchanged for fix-off
  checkpoints (needs GPU + S3).
