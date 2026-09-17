"""
Per-pixel logistic regression: the simplest possible AR segmentation baseline.

One weight per input channel plus a bias, applied independently at every pixel. In
PyTorch that is a 1x1 convolution, so with the 13 Surya channels the whole model has 14
parameters and cannot overfit. It is the segmentation analogue of the template's
``RegressionFlareModel``.

Two configurations are worth running:
  * all 13 channels          -> "how far does a linear pixel classifier get?"
  * channels: [hmi_m] only   -> a learned |B_los| threshold, i.e. essentially the recipe
                                 the labels were generated with (+/-50 G).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


def destandardize_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' moved from normalized space to signum-log space.

    Undoes the per-channel z-score ONLY; the signum-log compression stays, so values are
    ``sign(x*s) * log1p(|x*s|)`` rather than raw DN/Gauss. Same helper as the template's
    ``simple_baseline.destandardize_channels``; copied because apps are forks of the
    template and must not depend on one another.
    """
    x = batch["ts"].clone()
    with torch.no_grad():
        for i, channel in enumerate(channel_order):
            x[:, i, ...] = scalers[channel].inverse_transform(x[:, i, ...])
    return {**batch, "ts": x}


class PixelLogisticModel(nn.Module):
    """Logistic regression on each pixel's channel values, implemented as a 1x1 conv.

    Args:
        in_channels: Number of input channels C.
        time_dim: Number of input timesteps T; channel and time are flattened together.
        abs_input: Take ``|x|`` before the linear layer. AIA channels are non-negative
            already; for the signed HMI channels this turns the input into field
            strength, which is what defines an active region regardless of polarity.

    Input: batch dict with ``ts`` of shape (B, C, T, H, W) in signum-log space
    (use ``destandardize_channels`` as the Lightning ``preprocess_fn``).
    Output: logits of shape (B, 1, H, W).
    """

    def __init__(self, in_channels: int, time_dim: int = 1, abs_input: bool = True):
        super().__init__()
        self.abs_input = abs_input
        self.linear = nn.Conv2d(in_channels * time_dim, 1, kernel_size=1)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["ts"]
        if self.abs_input:
            x = x.abs()
        x = rearrange(x, "b c t h w -> b (c t) h w")
        return self.linear(x)
