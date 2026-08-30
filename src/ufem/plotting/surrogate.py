"""Figures for the surrogate and fold honest validation phase, drawn from artifacts only.

Build spec section 8, 10.5 and 20: the same one style module, and no figure is where a number
is first computed. The validate stage's own leave one out predictions are the input; this
module only draws them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ufem.plotting.style import (
    C_INK_2,
    C_MUTED,
    C_SERIES_1,
    C_SERIES_2,
    FIG_WIDTH_IN,
    annotation_style,
)

#: Consistent colors for the surrogate against the two baselines worth showing on one panel.
#: The training mean and the 3 nearest neighbour baseline are left off: the point of the
#: figure is the surrogate against its closest competitors, not every row of the table.
_MODEL_COLOR: dict[str, str] = {
    "gaussian_process": C_SERIES_1,
    "linear": C_SERIES_2,
    "quadratic_chaos": C_MUTED,
}
_MODEL_LABEL: dict[str, str] = {
    "gaussian_process": "Gaussian process",
    "linear": "linear",
    "quadratic_chaos": "quadratic chaos",
}


def predicted_vs_actual(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    r2_by_model: dict[str, float],
    unit_scale: float,
    unit_label: str,
) -> Any:
    """Out of sample predicted against actual for one scalar QoI, one panel per model.

    Three panels: the surrogate and the two baselines it is closest to out of sample
    (linear and quadratic chaos). Each shares axes so the reader compares scatter tightness
    directly rather than across independently scaled plots. The diagonal is the perfect
    prediction line, not a fit to these points.
    """
    order = ("gaussian_process", "linear", "quadratic_chaos")
    models = [name for name in order if name in predictions]
    fig, axes = plt.subplots(
        1, len(models), figsize=(FIG_WIDTH_IN, 2.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    truth_scaled = np.asarray(truth, dtype=float) * unit_scale
    scaled = {m: np.asarray(predictions[m], dtype=float) * unit_scale for m in models}
    lo = float(min(truth_scaled.min(), *(values.min() for values in scaled.values())))
    hi = float(max(truth_scaled.max(), *(values.max() for values in scaled.values())))
    pad = 0.05 * (hi - lo)
    for ax, model in zip(axes, models):
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=C_INK_2, lw=1.0,
                ls=(0, (4, 2.5)), zorder=2)
        ax.scatter(truth_scaled, scaled[model], s=14, color=_MODEL_COLOR[model], lw=0,
                   alpha=0.75, zorder=3)
        ax.annotate(
            f"{_MODEL_LABEL[model]}\n$R^2$ = {r2_by_model[model]:.3f}",
            xy=(0.04, 0.96), xycoords="axes fraction", **annotation_style(),
        )
        ax.set_xlabel(f"Measured {unit_label}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(f"Predicted {unit_label}")
    diagonal = Line2D(
        [], [], color=C_INK_2, lw=1.0, ls=(0, (4, 2.5)), label="perfect prediction"
    )
    axes[0].legend(handles=[diagonal], loc="lower right", fontsize=8.0)
    fig.tight_layout(pad=0.4)
    return fig
