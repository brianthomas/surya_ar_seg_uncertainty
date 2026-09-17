"""
Task-specific configuration for the active-region (AR) segmentation app.

Everything generic lives in ``workshop_infrastructure/configs.py`` and is imported, never
copied. This file adds only the handful of ``data:`` keys that describe the AR mask
dataset and how it is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, List, Optional

import app_paths  # noqa: F401  -- puts workshop_infrastructure on sys.path

from workshop_infrastructure.configs import (  # re-exported for convenience
    DataConfig,
    LoraAdapterConfig,
    ModelConfig,
    OutputConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    load_config,
)


@dataclass
class ARDataConfig(DataConfig):
    """DataConfig plus the AR-mask settings used by ``ARSegmentationDataset``.

    The masks come from the ``nasa-ibm-ai4science/surya-bench-ar-segmentation`` dataset:
    hourly 4096x4096 binary maps stored as ``.h5`` files, indexed by split CSVs. The
    train/validation split is decided by the *Surya* index (``train_data_path`` /
    ``valid_data_path``), so all mask CSVs can be listed here and the timestamp
    intersection selects what each split actually uses.
    """

    # Directory holding {train,validation,leaky_validation,test}.csv and data/YYYY/MM/*.h5.
    ar_mask_dir: str = ""
    # Which of the mask CSVs to load. The Surya index decides the split, so listing all of
    # them is the default; narrow this to exclude e.g. the leaky_validation period.
    ar_mask_splits: List[str] = field(
        default_factory=lambda: ["train", "validation", "leaky_validation", "test"]
    )
    # HDF5 key to use as the target: "union_with_intersect" (ARs that contain a PIL) or
    # "intersection" (the polarity inversion lines themselves).
    ar_mask_key: str = "union_with_intersect"
    # Spatial block-mean pooling applied to BOTH the SDO stack and the mask. 1 = native
    # 4096x4096. Values > 1 are for cheap baseline runs on small machines; the Surya
    # backbone needs 1 (its img_size is fixed at 4096).
    ar_pooling: int = 1
    # Loss / metric settings that depend on the label statistics.
    # AR pixels are ~1.3% of the disk, so BCE with pos_weight=1 is dominated by background.
    ar_pos_weight: Optional[float] = None
    ar_threshold: float = 0.5

    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + ("ar_mask_dir",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ar_pooling < 1:
            raise ValueError(f"data.ar_pooling must be >= 1, got {self.ar_pooling}.")
        if not 0.0 < self.ar_threshold < 1.0:
            raise ValueError(f"data.ar_threshold must be in (0, 1), got {self.ar_threshold}.")


load_ar_config = partial(load_config, data_cls=ARDataConfig)


__all__ = [
    "ARDataConfig",
    "load_ar_config",
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
