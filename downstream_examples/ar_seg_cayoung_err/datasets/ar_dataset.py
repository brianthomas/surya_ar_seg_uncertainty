"""
Active-region segmentation dataset: SDO image stacks paired with binary AR masks.

Masks come from ``nasa-ibm-ai4science/surya-bench-ar-segmentation``: one ``.h5`` file per
hour, each holding two 4096x4096 uint8 maps (0/255) under the keys ``intersection`` (the
polarity inversion lines) and ``union_with_intersect`` (the active regions that contain
one). The split CSVs list ``timestamp, file_path, present``.

Pairing is an exact timestamp match between the Surya index and the mask index. Masks
are hourly and SDO is at 12-minute cadence, so only on-the-hour SDO timesteps can pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

import app_paths  # noqa: F401  -- puts workshop_infrastructure on sys.path

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset

MASK_KEYS = ("union_with_intersect", "intersection")


class ARSegmentationDataset(HelioNetCDFDataset):
    """HelioNetCDFDataset whose ``forecast`` is a binary AR mask instead of a future frame.

    All ``HelioNetCDFDataset`` keyword arguments are accepted via ``**kwargs``.
    ``load_forecast_frames`` defaults to ``False`` (the label is the mask, so future SDO
    frames are never fetched), which also removes the "t+60 min must exist" filter.

    Additional Args:
        ds_mask_dir: Directory with the split CSVs and the ``data/YYYY/MM/*.h5`` masks.
        ds_mask_splits: Which CSVs to read. The Surya index decides train vs validation,
            so listing every split is safe; narrow it to exclude a period entirely.
        ds_mask_key: ``"union_with_intersect"`` (AR footprints) or ``"intersection"`` (PILs).
        return_surya_stack: If False, skip loading SDO data and return only the mask.
            Useful for label inspection and for tests.
        max_number_of_samples: Cap the dataset length for quick experiments.

    Raises:
        ValueError: on an unknown mask key, if ``random_vert_flip`` is requested (the
            base class would flip the inputs but not the mask), or if no timestamp
            overlaps between the Surya index and the mask index.
        FileNotFoundError: if a requested split CSV is missing.
    """

    def __init__(
        self,
        ds_mask_dir: str | Path,
        ds_mask_splits: Iterable[str] = ("train", "validation", "leaky_validation", "test"),
        ds_mask_key: str = "union_with_intersect",
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        **kwargs,
    ):
        if ds_mask_key not in MASK_KEYS:
            raise ValueError(f"ds_mask_key must be one of {MASK_KEYS}, got {ds_mask_key!r}.")
        if kwargs.get("random_vert_flip"):
            raise ValueError(
                "random_vert_flip is not supported: HelioNetCDFDataset flips the input "
                "stack but not a task-supplied mask, which would misalign the label."
            )
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.mask_dir = Path(ds_mask_dir)
        self.mask_key = ds_mask_key
        self.return_surya_stack = return_surya_stack

        mask_index = self._read_mask_index(self.mask_dir, ds_mask_splits)

        # Exact-timestamp intersection with the Surya index. Order is chronological.
        matched = pd.DatetimeIndex(self.valid_indices).intersection(mask_index.index).sort_values()
        if len(matched) == 0:
            raise ValueError(
                f"No timestamp overlap between the Surya index ({len(self.valid_indices)} "
                f"timesteps) and the mask index ({len(mask_index)} masks from {self.mask_dir})."
            )

        self.mask_paths = {
            pd.Timestamp(ts): self.mask_dir / mask_index.loc[ts, "file_path"] for ts in matched
        }
        self.valid_indices = [pd.Timestamp(ts) for ts in matched]
        if max_number_of_samples is not None and max_number_of_samples < len(self.valid_indices):
            self.valid_indices = self.valid_indices[:max_number_of_samples]
        self.adjusted_length = len(self.valid_indices)

    @staticmethod
    def _read_mask_index(mask_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
        """Concatenate the requested split CSVs, keeping present masks, indexed by timestamp."""
        frames = []
        for split in splits:
            csv_path = mask_dir / f"{split}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"Mask index {csv_path} not found. data.ar_mask_dir must point at the "
                    "surya-bench-ar-segmentation directory (see download_masks.sh)."
                )
            df = pd.read_csv(csv_path)
            frames.append(df.loc[df["present"] == 1, ["timestamp", "file_path"]])
        index = pd.concat(frames, ignore_index=True)
        index["timestamp"] = pd.to_datetime(index["timestamp"]).values.astype("datetime64[ns]")
        index = index.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        return index

    def _load_mask(self, path: Path) -> np.ndarray:
        """Read one mask as float32 in {0, 1}, block-pooled to match ``self.pooling``."""
        with h5py.File(path, "r") as f:
            mask = (f[self.mask_key][...] > 0).astype(np.float32)
        if self.pooling > 1:
            p = self.pooling
            h, w = mask.shape
            # Block mean, then re-binarize: a block is AR if at least half its pixels are.
            mask = mask[: h - h % p, : w - w % p].reshape(h // p, p, w // p, p).mean(axis=(1, 3))
            mask = (mask >= 0.5).astype(np.float32)
        return mask

    def __len__(self) -> int:
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary containing:
                forecast (np.ndarray): (H, W) float32 binary mask, the segmentation target.
                ds_index (str): ISO-format timestamp of the sample.
            When ``return_surya_stack=True``, also the ``HelioNetCDFDataset`` keys
            (``ts`` of shape (C, T, H, W), ``time_delta_input``).
        """
        timestep = self.valid_indices[idx]
        sample = super().__getitem__(idx) if self.return_surya_stack else {}
        sample["forecast"] = self._load_mask(self.mask_paths[timestep])
        sample["ds_index"] = timestep.isoformat()
        return sample
