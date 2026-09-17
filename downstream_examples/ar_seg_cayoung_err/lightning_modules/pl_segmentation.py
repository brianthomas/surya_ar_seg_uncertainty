"""
Lightning wrapper for AR segmentation.

The template's ``FlareLightningModule`` is already task-agnostic: it calls the model on
the batch, computes ``metrics[...]`` on ``(output, batch["forecast"].unsqueeze(1))`` and
logs. For a (B, H, W) mask that unsqueeze yields (B, 1, H, W), exactly the shape the 1x1
conv and ``HelioSpectformer2D`` produce, so it is reused unchanged under a task-neutral
name. If it ever moves into ``workshop_infrastructure/``, only this import changes.

``downstream_examples.template`` lives at the repository root, not in this folder; see
``app_paths.py`` for how it is found.
"""

import app_paths  # noqa: F401  -- puts the repository root on sys.path

from downstream_examples.template.lightning_modules.pl_simple_baseline import (
    FlareLightningModule as SegmentationLightningModule,
)

__all__ = ["SegmentationLightningModule"]
