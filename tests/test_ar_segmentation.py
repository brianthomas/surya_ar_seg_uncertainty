"""Tests for the ar_segmentation downstream app.

CPU-only and fast: tiny synthetic masks and a tiny backbone from conftest. They pin
1. the per-pixel logistic baseline's shape and parameter count,
2. the segmentation metrics on known inputs (perfect, empty, and inverted predictions),
3. the dataset's timestamp intersection between Surya index and mask index, its mask
   decoding (0/255 -> 0/1), and block pooling,
4. that the Surya 2D path keeps its head trainable under LoRA and emits (B, 1, H, W).
"""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import DEPTH, EMBED_DIM, IMG_SIZE, IN_CHANS, N_SPECTRAL_BLOCKS, PATCH_SIZE, make_batch
from datasets.ar_dataset import ARSegmentationDataset
from metrics.ar_metrics import ARMetrics
from models.pixel_logistic import PixelLogisticModel
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.models.finetune_models import HelioSpectformer2D
from workshop_infrastructure.utils import apply_peft_lora, build_scalers

CHANNELS = ["aia94", "hmi_m"]


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------


def test_pixel_logistic_is_one_weight_per_channel_plus_bias():
    model = PixelLogisticModel(in_channels=13, time_dim=1)
    assert sum(p.numel() for p in model.parameters()) == 14

    out = model({"ts": torch.randn(2, 13, 1, 32, 32)})
    assert out.shape == (2, 1, 32, 32)


def test_pixel_logistic_abs_makes_polarity_irrelevant():
    model = PixelLogisticModel(in_channels=1, time_dim=1, abs_input=True)
    x = torch.randn(1, 1, 1, 8, 8)
    torch.testing.assert_close(model({"ts": x}), model({"ts": -x}))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mask(fraction_on: float, size: int = 16) -> torch.Tensor:
    m = torch.zeros(1, 1, size, size)
    n_on = int(fraction_on * size * size)
    m.view(-1)[:n_on] = 1.0
    return m


def test_val_metrics_perfect_prediction():
    target = _mask(0.25)
    logits = torch.where(target > 0, 10.0, -10.0)
    out, weights = ARMetrics("val_metrics")(logits, target)
    assert set(out) == {"iou", "dice", "precision", "recall", "pos_frac"}
    assert len(weights) == len(out)
    for key in ("iou", "dice", "precision", "recall"):
        assert out[key].item() == pytest.approx(1.0)
    assert out["pos_frac"].item() == pytest.approx(0.25)


def test_val_metrics_all_background_prediction_scores_zero_on_nonempty_target():
    target = _mask(0.25)
    out, _ = ARMetrics("val_metrics")(torch.full_like(target, -10.0), target)
    assert out["iou"].item() == pytest.approx(0.0)
    assert out["recall"].item() == pytest.approx(0.0)


def test_val_metrics_empty_target_and_empty_prediction_is_perfect():
    """Predicting nothing on a quiet Sun is correct, not a division-by-zero failure."""
    target = _mask(0.0)
    out, _ = ARMetrics("val_metrics")(torch.full_like(target, -10.0), target)
    assert out["iou"].item() == pytest.approx(1.0)
    assert out["dice"].item() == pytest.approx(1.0)


def test_train_loss_is_bce_and_pos_weight_upweights_positives():
    target = _mask(0.25)
    logits = torch.zeros_like(target)
    plain, _ = ARMetrics("train_loss")(logits, target)
    weighted, _ = ARMetrics("train_loss", pos_weight=10.0)(logits, target)
    assert plain["bce"].item() == pytest.approx(float(np.log(2)))
    assert weighted["bce"] > plain["bce"]


def test_train_metrics_are_intentionally_empty():
    out, weights = ARMetrics("train_metrics")(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert out == {} and weights == []


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _scalers():
    entry = {
        "class": "StandardScaler",
        "mean": 0.0,
        "std": 1.0,
        "epsilon": 1e-8,
        "is_fitted": "true",
        "min": 0.0,
        "max": 1.0,
        "sl_scale_factor": 1.0,
    }
    return build_scalers({ch: dict(entry) for ch in CHANNELS})


def _write_masks(mask_dir, timestamps, size=8):
    """Write one 0/255 uint8 mask per timestamp with the AR in the top-left quadrant."""
    rows = []
    for ts in timestamps:
        rel = f"data/{ts:%Y/%m}/{ts:%Y%m%d_%H%M}.h5"
        path = mask_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        ar = np.zeros((size, size), dtype=np.uint8)
        ar[: size // 2, : size // 2] = 255
        with h5py.File(path, "w") as f:
            f["union_with_intersect"] = ar
            f["intersection"] = np.zeros_like(ar)
        rows.append({"timestamp": ts, "file_path": rel, "present": 1.0})
    # One mask that is listed but absent, as in the real CSVs.
    rows.append(
        {"timestamp": timestamps[-1] + pd.Timedelta(hours=1), "file_path": "", "present": 0.0}
    )
    pd.DataFrame(rows).to_csv(mask_dir / "train.csv", index=False)


def _write_surya_index(path, timestamps):
    pd.DataFrame(
        {
            "path": [f"{ts:%Y/%m/%Y%m%d_%H%M}.nc" for ts in timestamps],
            "timestep": timestamps,
            "present": 1,
        }
    ).to_csv(path, index=False)


@pytest.fixture
def synthetic(tmp_path):
    hourly = pd.date_range("2014-01-07 00:00", periods=3, freq="1h")
    _write_masks(tmp_path, hourly)
    # Surya index at 12-min cadence: only the on-the-hour rows can pair with a mask, and
    # the first hour has no mask at all.
    surya_ts = pd.date_range("2014-01-06 23:00", "2014-01-07 02:00", freq="12min")
    index_csv = tmp_path / "surya_index.csv"
    _write_surya_index(index_csv, surya_ts)
    return tmp_path, index_csv, hourly


def _dataset(mask_dir, index_csv, **overrides):
    kwargs = dict(
        index_path=str(index_csv),
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
        n_input_timestamps=1,
        rollout_steps=0,
        scalers=_scalers(),
        channels=CHANNELS,
        ds_mask_dir=mask_dir,
        ds_mask_splits=["train"],
        return_surya_stack=False,
    )
    kwargs.update(overrides)
    return ARSegmentationDataset(**kwargs)


def test_dataset_keeps_only_timestamps_with_both_sdo_and_mask(synthetic):
    mask_dir, index_csv, hourly = synthetic
    ds = _dataset(mask_dir, index_csv)
    assert len(ds) == 3
    assert ds.valid_indices == [pd.Timestamp(t) for t in hourly]


def test_dataset_decodes_mask_to_binary_float(synthetic):
    mask_dir, index_csv, hourly = synthetic
    sample = _dataset(mask_dir, index_csv)[1]
    mask = sample["forecast"]
    assert mask.dtype == np.float32 and mask.shape == (8, 8)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[:4, :4].all() and not mask[4:, 4:].any()
    assert sample["ds_index"] == hourly[1].isoformat()


def test_dataset_pooling_block_reduces_the_mask(synthetic):
    mask_dir, index_csv, _ = synthetic
    mask = _dataset(mask_dir, index_csv, pooling=4)[0]["forecast"]
    assert mask.shape == (2, 2)
    np.testing.assert_array_equal(mask, [[1.0, 0.0], [0.0, 0.0]])


def test_dataset_max_samples_truncates(synthetic):
    mask_dir, index_csv, _ = synthetic
    assert len(_dataset(mask_dir, index_csv, max_number_of_samples=2)) == 2


def test_dataset_rejects_vertical_flip(synthetic):
    mask_dir, index_csv, _ = synthetic
    with pytest.raises(ValueError, match="random_vert_flip"):
        _dataset(mask_dir, index_csv, random_vert_flip=True)


def test_dataset_rejects_unknown_mask_key(synthetic):
    mask_dir, index_csv, _ = synthetic
    with pytest.raises(ValueError, match="ds_mask_key"):
        _dataset(mask_dir, index_csv, ds_mask_key="nope")


def test_dataset_errors_when_nothing_overlaps(synthetic, tmp_path):
    mask_dir, _, _ = synthetic
    index_csv = tmp_path / "far_away.csv"
    _write_surya_index(index_csv, pd.date_range("2020-01-01", periods=3, freq="1h"))
    with pytest.raises(ValueError, match="No timestamp overlap"):
        _dataset(mask_dir, index_csv)


# ---------------------------------------------------------------------------
# Surya 2D path
# ---------------------------------------------------------------------------


def _tiny_2d_model():
    return HelioSpectformer2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=DEPTH,
        n_spectral_blocks=N_SPECTRAL_BLOCKS,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        ft_unembedding_type="linear",
        ft_out_chans=1,
    )


def test_surya_2d_emits_one_logit_channel_and_keeps_head_trainable_under_lora():
    model = apply_peft_lora(_tiny_2d_model(), LoraAdapterConfig())
    out = model(make_batch(batch_size=2))
    assert out.shape == (2, 1, IMG_SIZE, IMG_SIZE)

    head_params = [
        p for n, p in model.named_parameters() if "head_unembed" in n and "original_module" not in n
    ]
    assert head_params and all(p.requires_grad for p in head_params)

    # The metrics accept the model's output directly.
    target = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE)
    losses, _ = ARMetrics("train_loss")(out, target)
    assert torch.isfinite(losses["bce"])
