"""Figures for the calibration phase, drawn from the calibrate stage's artifacts only.

Build spec 8, 11.4 and 20. Every number these functions draw was measured by
``ufem.calibrate`` and written to a Parquet or a JSON; nothing here recomputes a coverage, a
band or a scaling factor. The paired before and after of build spec 11.4 is the organizing
idea of all four figures: the recalibration is only credible if what it changed is visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ufem.plotting.style import (
    C_BAND,
    C_INK_2,
    C_MUTED,
    C_SERIES_1,
    C_SERIES_2,
    FIG_WIDTH_IN,
    annotation_style,
)


def coverage_sweep(
    sweep: Any, targets: list[str], labels: dict[str, str]
) -> Any:
    """Empirical against nominal coverage for each headline quantity, before and after.

    One panel per quantity. The diagonal is perfect calibration. The raw Gaussian predictive
    is the before, the same after the measured variance scaling is the after, and the conformal
    band is drawn alongside because it is the one with a guarantee rather than a hope. Wilson
    intervals are drawn as vertical bars: at n = 198 a coverage is worth about two points
    either way, and a figure without that on it invites the reader to over read a wiggle.
    """
    fig, axes = plt.subplots(
        1, len(targets), figsize=(FIG_WIDTH_IN, 2.5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        rows = sweep.loc[sweep["target"] == target].sort_values("nominal")
        nominal = rows["nominal"].to_numpy(dtype=float)
        ax.plot([0.45, 1.0], [0.45, 1.0], color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
        for column, color, marker in (
            ("before", C_SERIES_2, "o"),
            ("after", C_SERIES_1, "s"),
            ("conformal", C_MUTED, "^"),
        ):
            empirical = rows[f"{column}_empirical"].to_numpy(dtype=float)
            low = rows[f"{column}_wilson_low"].to_numpy(dtype=float)
            high = rows[f"{column}_wilson_high"].to_numpy(dtype=float)
            finite = np.isfinite(empirical)
            ax.vlines(
                nominal[finite], low[finite], high[finite], color=color, lw=0.8, alpha=0.5,
                zorder=3,
            )
            ax.plot(
                nominal[finite], empirical[finite], color=color, lw=1.2, marker=marker,
                ms=3.0, zorder=4,
            )
        ax.annotate(labels.get(target, target), xy=(0.04, 0.96), xycoords="axes fraction",
                    **annotation_style())
        ax.set_xlabel("Nominal")
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Empirical coverage")
    handles = [
        Line2D([], [], color=C_SERIES_2, lw=1.2, marker="o", ms=3.0, label="before scaling"),
        Line2D([], [], color=C_SERIES_1, lw=1.2, marker="s", ms=3.0, label="after scaling"),
        Line2D([], [], color=C_MUTED, lw=1.2, marker="^", ms=3.0, label="conformal"),
        Line2D([], [], color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), label="perfect"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.0,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.4)
    return fig


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell boundaries around a set of sample abscissae, for a flat shaded mesh."""
    values = np.asarray(centers, dtype=float)
    if values.size < 2:
        raise ValueError("a heatmap needs at least two abscissae.")
    middles = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        [[values[0] - (middles[0] - values[0])], middles, [values[-1] + (values[-1] - middles[-1])]]
    )


def pit_heatmap(pit: Any, signal: str, nominal_density: float) -> Any:
    """The probability integral transform along the displacement axis, before and after.

    Two panels sharing one color scale, centered on the density a calibrated model would give
    every bin, so the color says how far from flat each column is and in which direction. The
    softening branch is where build spec 11.4 expects calibration to break, and it is the right
    hand three quarters of each panel.
    """
    rows = pit.loc[pit["signal"] == signal]
    bins = [name for name in rows.columns if name.startswith("bin_")]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 3.4), sharex=True)
    grids = []
    for stage in ("before", "after"):
        block = rows.loc[rows["stage"] == stage].sort_values("u_mm")
        grids.append((stage, block["u_mm"].to_numpy(dtype=float),
                      block[bins].to_numpy(dtype=float).T))
    # The color scale is set by the 99th percentile of the deviations from flat, not by the
    # maximum: one abscissa near the origin, where the force is nearly deterministic, is four
    # times more extreme than anything else and would otherwise flatten the whole picture into
    # one color. Cells past the scale are drawn at its ends, so nothing is hidden, only capped.
    span = float(
        np.percentile(
            np.concatenate([np.abs(grid - nominal_density).ravel() for _s, _u, grid in grids]), 99
        )
    )
    for ax, (stage, u_mm, grid) in zip(axes, grids):
        mesh = ax.pcolormesh(
            _cell_edges(u_mm), np.linspace(0.0, 1.0, len(bins) + 1), grid, cmap="RdBu_r",
            shading="flat", vmin=nominal_density - span, vmax=nominal_density + span,
        )
        ax.set_ylabel("PIT")
        ax.annotate(
            f"{stage} the variance scaling", xy=(0.015, 0.94), xycoords="axes fraction",
            **annotation_style(),
        )
        ax.grid(False)
    axes[-1].set_xlabel("Displacement [mm]")
    bar = fig.colorbar(mesh, ax=axes, fraction=0.045, pad=0.015)
    bar.set_label(f"density per decile (flat = {nominal_density:.2f}, scale capped)")
    return fig


def conformal_band(examples: Any, signal: str, unit_scale: float, unit_label: str) -> Any:
    """Three runs with their simultaneous 90 percent band, worst case on the right.

    The band is the one the calibrate stage wrote: the leave one out mean, plus and minus the
    conformal multiplier times the recalibrated modulation. Behind it, dashed, is what the
    uncalibrated model offered at the same nominal level, a pointwise Gaussian interval, so
    the figure shows what calibration cost rather than asserting that it was cheap.
    """
    order = ("median", "p90", "worst")
    titles = {"median": "median run", "p90": "90th percentile", "worst": "worst run"}
    rows = examples.loc[examples["signal"] == signal]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.4), sharex=True, sharey=True)
    for ax, key in zip(np.atleast_1d(axes), order):
        block = rows.loc[rows["example"] == key].sort_values("u_mm")
        u_mm = block["u_mm"].to_numpy(dtype=float)
        ax.fill_between(
            u_mm, block["lower"].to_numpy(dtype=float) * unit_scale,
            block["upper"].to_numpy(dtype=float) * unit_scale,
            color=C_BAND, alpha=0.55, lw=0, zorder=2,
        )
        ax.plot(u_mm, block["upper_gaussian"].to_numpy(dtype=float) * unit_scale,
                color=C_MUTED, lw=0.7, ls=(0, (3, 2)), zorder=3)
        ax.plot(u_mm, block["lower_gaussian"].to_numpy(dtype=float) * unit_scale,
                color=C_MUTED, lw=0.7, ls=(0, (3, 2)), zorder=3)
        ax.plot(u_mm, block["loo_mean"].to_numpy(dtype=float) * unit_scale,
                color=C_SERIES_1, lw=1.4, zorder=4)
        ax.plot(u_mm, block["truth"].to_numpy(dtype=float) * unit_scale,
                color=C_SERIES_2, lw=1.4, zorder=5)
        ax.annotate(
            f"{titles[key]}\nscore {float(block['sup_score'].iloc[0]):.2f}",
            xy=(0.04, 0.96), xycoords="axes fraction", **annotation_style(),
        )
        ax.set_xlabel("Displacement [mm]")
    np.atleast_1d(axes)[0].set_ylabel(unit_label)
    handles = [
        Line2D([], [], color=C_SERIES_2, lw=1.4, label="simulated"),
        Line2D([], [], color=C_SERIES_1, lw=1.4, label="leave one out mean"),
        Patch(facecolor=C_BAND, alpha=0.55, edgecolor="none",
              label="simultaneous 90 percent band"),
        Line2D([], [], color=C_MUTED, lw=0.7, ls=(0, (3, 2)),
               label="pointwise Gaussian, uncalibrated"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.0,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(pad=0.4)
    return fig


def crps_skill(records: dict[str, Any], targets: list[str], labels: dict[str, str]) -> Any:
    """Continuous ranked probability skill against climatology, before and after scaling.

    Zero is the leave one out training mean and spread. The bars are what the whole surrogate
    buys probabilistically, which is a stricter question than the R2 of build spec 10.5: a
    model can beat the mean on point error and still lose on distribution.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.2))
    positions = np.arange(len(targets), dtype=float)
    width = 0.36
    for offset, key, color, label in (
        (-width / 2, "crps_skill_before", C_SERIES_2, "before scaling"),
        (width / 2, "crps_skill_after", C_SERIES_1, "after scaling"),
    ):
        values = [float(records[name][key]) for name in targets]
        ax.bar(positions + offset, values, width, color=color, lw=0, label=label, zorder=3)
    ax.axhline(0.0, color=C_INK_2, lw=0.8, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels.get(name, name) for name in targets], fontsize=8.5)
    ax.set_ylabel("CRPS skill against climatology")
    ax.legend(loc="upper right", fontsize=8.0)
    fig.tight_layout(pad=0.4)
    return fig
