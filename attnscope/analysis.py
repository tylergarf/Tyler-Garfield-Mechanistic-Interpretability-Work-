"""Numpy-only analysis helpers (no MLX required)."""

# =============================================================================
# HOW THIS FILE MAPS TO THE ORIGINAL SUMMER CODE
#
# dominant_dims() is new: there was no equivalent function in the summer code.
# It puts a number on what the bottom panels of the final_lamma_vis.ipynb
# figures show (the "COS MLP SOFTMAX" heatmaps): which hidden dimension is the
# single bright pixel in each layer.
# =============================================================================

from __future__ import annotations

import numpy as np


def dominant_dims(last_hidden: np.ndarray, step: int = -1) -> np.ndarray:
    """Index of the largest residual dimension per layer at generation ``step``.

    ``last_hidden``: ``(steps, layers, d_model)``. Returns ``(layers,)``.
    (Softmax is monotonic, so this is also the bright pixel in the
    hidden-state panels.)
    """
    # last_hidden is the original mlp_out_mone / cos_last array: (100 steps, 28 layers, 3072).
    #   last_hidden[step]     (28, 3072)   every layer's last-token state at that step
    #                                      (step=-1 is the final step, the one the figures show)
    #   .argmax(axis=-1)      (28,)        index of the largest of the 3072 values, per layer
    # The figure's softmax doesn't change which value is largest, so this is the
    # same pixel that lights up. To find it in the 24 x 128 heatmap:
    # row = dim // 128, col = dim % 128 (e.g. dim 962 -> row 7, col 66).
    return last_hidden[step].argmax(axis=-1)
