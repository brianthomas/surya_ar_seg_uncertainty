"""Shared palette and raster helpers for the AR segmentation viewer notebooks.

Chrome only -- the palette, the rcParams, the block downsamplers and the
per-channel colormap/stretch. No data loading and no science: the notebooks keep
those inline, because the paths and the questions differ per notebook while the
palette must not.

Exists because `block_max` was already copy-pasted between `view_results.ipynb`
and `view_bootstrap_ensemble.ipynb` (the latter's docstring says so), and the
validated palette is ~60 lines on top of that. A third notebook would have made
a third copy. The two existing notebooks keep their inline copies; they can
migrate whenever they are next touched.

Usage:
    import ensemble_viz as ev
    ev.apply_rcparams()
    ax.imshow(ev.block_mean(img, 4), cmap=ev.channel_cmap("aia94"),
              **dict(zip(("vmin", "vmax"), ev.channel_clim("aia94", img))))
"""

import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as sunpy_cm  # registers the sdoaia* colormaps
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette
#
# One sequential blue ramp carries every magnitude (agreement, AR fraction,
# probability), so the same visual weight always means the same thing.
#
# Checked with the data-viz palette validator:
#
#   | use                                 | colours              | mode  | result    |
#   |-------------------------------------|----------------------|-------|-----------|
#   | member dots vs ensemble mean        | #2a78d6,#eb6834      | light | all pass  |
#   | agreement contours over imagery     | #d95926,#199e70      | dark  | all pass  |
#   | mask fill, used alone on a panel    | #d55181              | dark  | all pass  |
#
# Two deliberate departures, both forced by the data; see SEQ_AGREE and
# MASK_FILL below for the full reasoning.
# ---------------------------------------------------------------------------

# Sequential blue, steps 100 -> 700 of the validated ramp. Used where the light
# end genuinely means "near zero" and may recede toward the surface.
BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("surya_blue", BLUE_STEPS)

# Agreement ramp, steps 250 -> 700. On the agreement map the lightest value drawn
# means "1 member called this AR" -- a real detection, not nothing (k=0 is masked
# out) -- so the light end must clear the surface. Step 250 is the lightest that
# does, at 2.06:1.
#
# Read as a gradient against the colorbar, not as 10 nameable steps: ten discrete
# levels of a single hue sit ~0.047 apart in lightness, well under what the eye
# resolves, and the validator fails this ladder under --ordinal for exactly that
# reason. No single-hue ramp can carry 10 distinguishable steps.
SEQ_AGREE = LinearSegmentedColormap.from_list("surya_blue_agree", BLUE_STEPS[3:])

SERIES_1 = "#2a78d6"   # categorical slot 1 (light surface) - individual members
SERIES_2 = "#eb6834"   # categorical slot 2 (light surface) - ensemble mean
CONTOUR = {0.5: "#d95926", 0.8: "#199e70"}  # dark-surface steps, drawn over imagery

# Mask fill for the per-member panels. Magenta because the AIA colormaps run
# black -> green/gold -> white: cyan sits inside that range and vanishes over
# bright AR cores, which is exactly where the mask is. Magenta is the one
# strongly off-colormap hue. Used alone on a panel, so it needs no pairwise
# CVD check -- it would fail one against the aqua contour, but the two never
# share a figure.
MASK_FILL = "#d55181"

INK = "#0b0b0b"        # primary ink
INK_2 = "#52514e"      # secondary ink
MUTED = "#898781"      # axis / tick labels
GRID = "#e1e0d9"       # hairline gridline

RCPARAMS = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
}


def apply_rcparams():
    """Install the shared figure style. Call once per notebook, after imports."""
    plt.rcParams.update(RCPARAMS)


def recessive(ax, grid_axis="y"):
    """Push grid and axes behind the data, where chrome belongs."""
    ax.grid(True, axis=grid_axis, lw=0.8, color=GRID)
    ax.set_axisbelow(True)
    return ax


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------


def block_max(img, factor):
    """Downsample by taking the max over each factor x factor block.

    Averaging would erase sparse features; max keeps any flagged pixel visible.
    At the 0.6 % positive rate of the training-era frames a factor-4 mean loses
    isolated mask pixels entirely, which is why this helper exists.
    """
    h, w = img.shape[:2]
    h, w = h - h % factor, w - w % factor  # crop to a multiple of factor
    img = img[:h, :w]
    if img.ndim == 3:
        return img.reshape(h // factor, factor, w // factor, factor, img.shape[2]).max(axis=(1, 3))
    return img.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def block_mean(img, factor):
    """Downsample by averaging. For the smooth channel images, not for masks."""
    h, w = img.shape[:2]
    h, w = h - h % factor, w - w % factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def channel_clim(channel, img):
    """Per-frame display limits, from percentiles rather than the scaler.

    The scaler's min/max span the whole 2010-2019 archive and are set by flares.
    On a typical frame aia94 reaches a 99.8th percentile of ~0.08 against a
    scaler max of 5.10, so scaling to the scaler puts the entire disk in the
    bottom 2 % of the colour range and renders it black. A per-frame percentile
    stretch is the usual solar-imaging choice and is what makes active regions
    legible under the mask overlay.

    Subsampled 4x in each axis: these are display limits, and sorting 16.8M
    values for them is wasted work.
    """
    sub = img[::4, ::4]
    if "hmi" in channel:
        # Diverging colormaps must stay centred on zero or the sign reads wrong.
        v = float(np.percentile(np.abs(sub), 99.5))
        return -v, v
    lo, hi = np.percentile(sub, [0.5, 99.8])
    return float(lo), float(hi)


def channel_cmap(channel):
    """The community colormap for one HelioFM channel."""
    if channel == "hmi_v":
        return plt.get_cmap("bwr")
    key = "hmimag" if channel.startswith("hmi") else f"sdo{channel}"
    return sunpy_cm.cmlist[key]
