"""Figures for the audit stage: the completion surface and the calibration plot.

Build spec 9.4 and 20. Both figures visualize artifacts the audit stage already wrote;
neither fits anything or recomputes a statistic. A figure that computed its own number would
be a second implementation of that number, and the two would eventually disagree.

Units: strength in MPa, covers in mm. Probabilities are dimensionless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ufem.config import FEATURE_ORDER
from ufem.plotting.style import (
    C_INK_2,
    C_SERIES_1,
    FIG_WIDTH_IN,
    annotation_style,
)

#: Marker fill for a completed run on the sequential surface. Near black rather than the
#: report's identity blue, because on a viridis ramp a mid blue disappears into the surface
#: it is meant to sit on top of.
MARKER_COMPLETED = "#111111"

#: Axis labels for the three inputs, in the feature contract order.
AXIS_LABELS: dict[str, str] = {
    "Fcm_MPa": "$f_{cm}$ [MPa]",
    "c_nom_bottom_mm": "$c_{nom,bot}$ [mm]",
    "c_nom_top_mm": "$c_{nom,top}$ [mm]",
}


def completion_surface(
    model: Any,
    design: np.ndarray,
    completed: np.ndarray,
    threshold: float,
    resolution: int = 120,
) -> Any:
    """Two dimensional slices of the fitted completion probability surface.

    One panel per pair of inputs, each drawn with the third input held at its median over
    the executed design, which is what makes a two dimensional slice of a three dimensional
    function honest: the slice is taken somewhere the design actually is, not at an
    arbitrary origin. The design points are overlaid so the reader can see the density
    behind every part of the surface, and the threshold contour is drawn because that
    contour, not the color ramp, is the boundary the validity domain uses.

    ``model`` is the fitted completion classifier, ``design`` the ``(n, 3)`` feature matrix,
    ``completed`` the boolean outcome per design row, ``threshold`` the domain threshold.
    """
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"completion_surface needs an (n, {len(FEATURE_ORDER)}) design matrix in the "
            f"feature order {list(FEATURE_ORDER)}; got shape {matrix.shape}."
        )
    flags = np.asarray(completed, dtype=bool)
    if flags.shape != (matrix.shape[0],):
        raise ValueError(
            f"completion_surface needs one outcome per design row, got {flags.shape} "
            f"against {matrix.shape[0]} rows."
        )
    median = np.median(matrix, axis=0)
    pairs = [(0, 2), (0, 1), (2, 1)]

    # constrained_layout, not tight_layout: each panel carries its own y label, and with a
    # shared colorbar the tight solver leaves them overlapping the neighbouring axes.
    fig, axes = plt.subplots(
        1, 3, figsize=(FIG_WIDTH_IN, 2.55), layout="constrained"
    )
    mesh = None
    for ax, (i, j) in zip(axes, pairs):
        held = [index for index in range(len(FEATURE_ORDER)) if index not in (i, j)][0]
        xs = np.linspace(matrix[:, i].min(), matrix[:, i].max(), resolution)
        ys = np.linspace(matrix[:, j].min(), matrix[:, j].max(), resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        query = np.empty((grid_x.size, len(FEATURE_ORDER)), dtype=float)
        query[:, i] = grid_x.ravel()
        query[:, j] = grid_y.ravel()
        query[:, held] = median[held]
        surface = model.predict_proba(query)[:, 1].reshape(grid_x.shape)

        # Sequential, not diverging: completion probability has no meaningful midpoint that
        # a diverging ramp would encode, and 0.5 is a decision threshold rather than a
        # neutral value. The threshold is drawn as a contour instead, where it belongs.
        mesh = ax.pcolormesh(
            grid_x, grid_y, surface, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto",
            rasterized=True, zorder=1,
        )
        ax.contour(
            grid_x, grid_y, surface, levels=[threshold], colors=["white"],
            linewidths=1.4, zorder=3,
        )
        # The design points sit on a dark to light ramp, so identity is carried by fill
        # (hollow against solid) rather than by hue, which no two colors would survive.
        ax.scatter(
            matrix[~flags, i], matrix[~flags, j], s=6, facecolor="none",
            edgecolor="white", linewidth=0.5, alpha=0.85, zorder=4,
        )
        ax.scatter(
            matrix[flags, i], matrix[flags, j], s=6, color=MARKER_COMPLETED,
            edgecolor="white", linewidth=0.25, alpha=0.95, zorder=5,
        )
        ax.set_xlabel(AXIS_LABELS[FEATURE_ORDER[i]])
        ax.set_ylabel(AXIS_LABELS[FEATURE_ORDER[j]])
        ax.grid(False)

    bar = fig.colorbar(mesh, ax=axes, fraction=0.030, pad=0.02)
    bar.set_label("P(complete) [-]")
    bar.outline.set_linewidth(0.6)
    handles = [
        Line2D(
            [], [], marker="o", ls="none", ms=4.5, color=MARKER_COMPLETED,
            markeredgecolor="white", markeredgewidth=0.5, label="completed",
        ),
        Line2D(
            [], [], marker="o", ls="none", ms=4.5, markerfacecolor="none",
            markeredgecolor=C_INK_2, markeredgewidth=0.9, label="failed",
        ),
        Line2D([], [], color=C_INK_2, lw=1.4, label=f"P = {threshold:.2f} contour"),
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncol=3, borderaxespad=0.2
    )
    return fig


def calibration_plot(
    table: list[dict[str, Any]], auc: float, brier: float, ece: float
) -> Any:
    """Reliability diagram: predicted against empirical completion rate, per bin.

    The diagonal is perfect calibration. Marker area is proportional to the number of design
    points in the bin, because a bin holding two samples and a bin holding seventy carry very
    different evidence and drawing them the same size would invite reading noise as
    miscalibration. Empty bins are simply absent, which is honest: the model made no
    predictions there.
    """
    populated = [row for row in table if row["n"] > 0]
    if not populated:
        raise ValueError(
            "the calibration table has no populated bin, so there is nothing to plot. That "
            "would mean the model produced no predictions at all."
        )
    predicted = np.array([row["mean_predicted"] for row in populated], dtype=float)
    empirical = np.array([row["empirical_rate"] for row in populated], dtype=float)
    counts = np.array([row["n"] for row in populated], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.62, 3.2))
    ax.plot([0, 1], [0, 1], color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ax.plot(predicted, empirical, color=C_SERIES_1, lw=1.2, zorder=3)
    ax.scatter(
        predicted, empirical, s=12.0 + 90.0 * counts / counts.max(),
        color=C_SERIES_1, lw=0, alpha=0.85, zorder=4,
    )
    ax.set_xlabel("Mean predicted P(complete) [-]")
    ax.set_ylabel("Empirical completion rate [-]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.annotate(
        f"AUC = {auc:.3f}\nBrier = {brier:.3f}\nECE = {ece:.3f}",
        xy=(0.04, 0.96), xycoords="axes fraction", **annotation_style(),
    )
    handles = [
        Line2D([], [], color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), label="perfect calibration"),
        Line2D(
            [], [], color=C_SERIES_1, lw=1.2, marker="o", ms=5.0,
            label="observed (area proportional to $n$)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout(pad=0.4)
    return fig
