"""
Loss and evaluation metrics for binary AR segmentation.

Four modes, following the template contract (see ``downstream_examples/template/metrics``):
- "train_loss"    — BCE with logits, backpropagated. Optional ``pos_weight`` for the
                    ~80:1 background:AR imbalance.
- "val_loss"      — same BCE; this is what ModelCheckpoint monitors.
- "train_metrics" — intentionally empty: IoU over 16.8M pixels per image at every step
                    is not worth the time, and it would not change the weights.
- "val_metrics"   — IoU, Dice, precision, recall at ``threshold``, plus the positive
                    pixel fraction of the target. Reported only.

Shape contract: predictions and targets are (B, 1, H, W); everything is flattened, so the
validation metrics are POOLED over all pixels in the batch, not averaged per sample. With
batch_size 1 that is per-sample. A batch whose target and prediction are both empty scores
IoU = Dice = 1 (``zero_division=1.0``): predicting nothing on a quiet Sun is correct.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_jaccard_index,
    binary_precision,
    binary_recall,
)


class ARMetrics:
    def __init__(self, mode: str, threshold: float = 0.5, pos_weight: float | None = None):
        self.mode = mode
        self.threshold = threshold
        self.pos_weight = None if pos_weight is None else torch.tensor(float(pos_weight))

    def _bce(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pw = None if self.pos_weight is None else self.pos_weight.to(preds.device)
        return F.binary_cross_entropy_with_logits(preds.float(), target.float(), pos_weight=pw)

    def train_loss(self, preds, target) -> tuple[dict[str, torch.Tensor], list[float]]:
        return {"bce": self._bce(preds, target)}, [1]

    def val_loss(self, preds, target) -> tuple[dict[str, torch.Tensor], list[float]]:
        return self.train_loss(preds, target)

    def train_metrics(self, preds, target) -> tuple[dict[str, torch.Tensor], list[float]]:
        return {}, []

    def val_metrics(self, preds, target) -> tuple[dict[str, torch.Tensor], list[float]]:
        probs = torch.sigmoid(preds.float()).reshape(-1)
        t = (target.reshape(-1) > 0.5).int()
        kw = dict(threshold=self.threshold)
        out = {
            "iou": binary_jaccard_index(probs, t, zero_division=1.0, **kw),
            "dice": binary_f1_score(probs, t, zero_division=1.0, **kw),
            "precision": binary_precision(probs, t, zero_division=1.0, **kw),
            "recall": binary_recall(probs, t, zero_division=1.0, **kw),
            "pos_frac": t.float().mean(),
        }
        return out, [1] * len(out)

    def __call__(self, preds, target) -> tuple[dict[str, torch.Tensor], list[float]]:
        match self.mode.lower():
            case "train_loss":
                return self.train_loss(preds, target)
            case "val_loss":
                return self.val_loss(preds, target)
            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)
            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)
            case _:
                raise NotImplementedError(f"{self.mode} is not a valid metric mode.")
