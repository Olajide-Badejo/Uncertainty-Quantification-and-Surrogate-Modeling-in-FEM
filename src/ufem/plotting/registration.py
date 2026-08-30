"""Figures for the registration and reduction phase, drawn from artifacts only.

Build spec section 8 and 20: every figure is a function of an artifact, the style comes from
the one style module, and no figure is the place a number is first computed. These functions
take arrays and return figures; the caller reads the artifacts and records any annotated
quantity into ``figure_stats.json``.
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


def registration_before_after(
    stations: np.ndarray,
    unregistered: np.ndarray,
    registered: np.ndarray,
) -> Any:
    """Two spaghetti panels: the family before and after elastic registration.

    The point the figure has to carry is that the peaks line up on the right and do not on
    the left, so both panels share a vertical axis and the pointwise median is drawn over
    each: a smeared median on the left against a sharp one on the right is the visual form of
    the argument that registration exists to make.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.0), sharey=True)
    for ax, family, label in (
        (axes[0], unregistered, "before registration"),
        (axes[1], registered, "after registration"),
    ):
        for row in family:
            ax.plot(
                stations, row / 1000.0, color=C_MUTED, lw=0.35, alpha=0.16,
                solid_capstyle="round", zorder=1,
            )
        median = np.percentile(family, 50, axis=0) / 1000.0
        ax.plot(stations, median, color=C_SERIES_1, lw=1.8, zorder=3)
        peak = float(median.max())
        ax.annotate(
            f"{label}\nmedian peak {peak:.2f} kN",
            xy=(0.03, 0.97), xycoords="axes fraction", **annotation_style(),
        )
        ax.set_xlabel("Normalized arc length $s$ [-]")
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel("Reaction force $RF_2$ [kN]")
    axes[0].set_ylim(0, None)
    fig.tight_layout(pad=0.4)
    return fig


def warp_family(stations: np.ndarray, gamma: np.ndarray) -> Any:
    """The warping functions, with the identity for reference.

    Every curve here is monotone from (0, 0) to (1, 1); departure from the diagonal is the
    phase the registration removed from the amplitude functions.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.62, 3.0))
    for row in gamma:
        ax.plot(stations, row, color=C_MUTED, lw=0.4, alpha=0.22, zorder=1)
    ax.plot(
        stations, np.percentile(gamma, 50, axis=0), color=C_SERIES_1, lw=1.8, zorder=3
    )
    ax.plot(
        [0.0, 1.0], [0.0, 1.0], color=C_SERIES_2, lw=1.1, ls=(0, (4, 2.5)), zorder=2
    )
    ax.set_xlabel("Normalized arc length $s$ [-]")
    ax.set_ylabel("Warp $\\gamma(s)$ [-]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(
        handles=[
            Line2D([], [], color=C_MUTED, lw=0.9, alpha=0.6,
                   label=f"warps (n = {gamma.shape[0]})"),
            Line2D([], [], color=C_SERIES_1, lw=1.8, label="pointwise median"),
            Line2D([], [], color=C_SERIES_2, lw=1.1, ls=(0, (4, 2.5)), label="identity"),
        ],
        loc="upper left",
    )
    fig.tight_layout(pad=0.4)
    return fig


def scree(
    registered_ratio: np.ndarray,
    unregistered_ratio: np.ndarray,
    target: float,
    n_registered: int,
    n_unregistered: int,
    n_show: int = 16,
) -> Any:
    """Cumulative explained variance, registered against unregistered.

    Cumulative rather than per component, because the quantity the ablation reports is where
    each curve crosses the variance target, and a cumulative plot shows that crossing directly
    instead of asking the reader to sum bars.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN * 0.72, 3.0))
    for ratio, color, label, count in (
        (registered_ratio, C_SERIES_1, "registered", n_registered),
        (unregistered_ratio, C_SERIES_2, "unregistered", n_unregistered),
    ):
        cumulative = np.cumsum(np.asarray(ratio, dtype=float))[:n_show]
        x = np.arange(1, cumulative.size + 1)
        ax.plot(x, cumulative, color=color, lw=1.6, marker="o", ms=3.4, zorder=3,
                label=f"{label} ({count} to {target:.0%})")
        ax.axvline(count, color=color, lw=0.8, ls=(0, (3, 2.5)), zorder=2)
    ax.axhline(target, color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ax.annotate(
        f"{target:.0%} of variance",
        xy=(0.98, target), xycoords=("axes fraction", "data"),
        xytext=(0, -4), textcoords="offset points",
        ha="right", va="top", fontsize=9.0, color=C_INK_2,
    )
    ax.set_xlabel("Number of components [-]")
    ax.set_ylabel("Cumulative explained variance [-]")
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    return fig


def amplitude_loadings(
    stations: np.ndarray, components: np.ndarray, ratios: np.ndarray, n_show: int = 3
) -> Any:
    """The leading amplitude PC loadings, one panel each with its variance share."""
    n = min(n_show, components.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH_IN, 2.5), sharex=True)
    axes = np.atleast_1d(axes)
    for index in range(n):
        ax = axes[index]
        ax.axhline(0.0, color=C_MUTED, lw=0.7, zorder=1)
        ax.plot(stations, components[index], color=C_SERIES_1, lw=1.5, zorder=3)
        ax.set_xlabel("Arc length $s$ [-]")
        ax.annotate(
            f"PC{index + 1}\n{ratios[index] * 100:.1f}\\,\\% of variance",
            xy=(0.04, 0.96), xycoords="axes fraction", **annotation_style(),
        )
    axes[0].set_ylabel("Loading [-]")
    fig.tight_layout(pad=0.4)
    return fig
