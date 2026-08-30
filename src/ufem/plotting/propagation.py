"""Figures for the propagation phase, drawn from the propagate stage's artifacts.

Build spec 8, 13 and 20. Three figures, one per thing build spec 13 asks to be shown: what the
propagated quantities of interest look like against the thresholds that judge them, what the
predicted response family looks like as a fan, and whether the independent analytic model lands
where the surrogate does.

Two conventions are enforced here rather than left to the caller, because both are ways a
reliability figure misleads. The out of domain mass fraction is annotated on every panel, so a
reader cannot see a distribution without seeing how much of it the surrogate is entitled to
speak about. And the aleatory and predictive densities are drawn as two curves rather than one
merged shape, because they are two layers of build spec 13.1 and adding them on a page is the
same mistake as adding them in a number.
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


def qoi_densities(
    density: Any,
    targets: list[str],
    thresholds: dict[str, float],
    labels: dict[str, str],
    scales: dict[str, float],
    units: dict[str, str],
    out_of_domain_fraction: float,
) -> Any:
    """Propagated densities of the limit state quantities, with their thresholds marked.

    One panel per quantity. The filled curve is the aleatory layer, the input distributions
    through the mean surrogate; the outlined curve is the predictive layer, the same inputs
    through the calibrated predictive distribution. The shaded side of the threshold is the
    failure region, so the failure probability is the area a reader can see rather than a
    number they have to take on trust.
    """
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH_IN, 2.7))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, targets):
        rows = density.loc[density["target"] == name].sort_values("value")
        scale = scales[name]
        x = rows["value"].to_numpy(dtype=float) * scale
        aleatory = rows["aleatory_density"].to_numpy(dtype=float) / scale
        predictive = rows["predictive_density"].to_numpy(dtype=float) / scale
        threshold = thresholds[name] * scale
        ax.fill_between(x, aleatory, color=C_BAND, alpha=0.75, lw=0, zorder=2)
        ax.plot(x, aleatory, color=C_SERIES_1, lw=1.5, zorder=4)
        ax.plot(x, predictive, color=C_SERIES_2, lw=1.2, ls=(0, (4, 2.0)), zorder=5)
        ax.axvline(threshold, color=C_INK_2, lw=1.0, zorder=6)
        top = max(float(aleatory.max()), float(predictive.max())) * 1.28
        failing = x <= threshold if threshold > float(np.median(x)) else x >= threshold
        ax.fill_between(
            x[failing],
            np.zeros(int(failing.sum())),
            np.full(int(failing.sum()), top),
            color=C_MUTED,
            alpha=0.14,
            lw=0,
            zorder=1,
        )
        ax.set_xlabel(f"{labels[name]} [{units[name]}]")
        ax.set_ylim(0.0, top)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("probability density")
    handles = [
        Patch(facecolor=C_BAND, edgecolor=C_SERIES_1, lw=1.2, label="aleatory layer"),
        Line2D([], [], color=C_SERIES_2, lw=1.2, ls=(0, (4, 2.0)), label="predictive layer"),
        Line2D([], [], color=C_INK_2, lw=1.0, label="limit state"),
        Patch(
            facecolor=C_MUTED,
            alpha=0.3,
            edgecolor="none",
            label=f"failure region ({100.0 * out_of_domain_fraction:.0f}\\,\\% of the mass "
            "is outside the validity domain)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        fontsize=8.0,
    )
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.90))
    return fig


def envelope_fan(
    envelope: Any,
    prefix: str,
    scale: float,
    ylabel: str,
    out_of_domain_fraction: float,
    n_curves: int,
) -> Any:
    """The curve envelope fan of build spec 13.1: median, 50 and 90 percent envelopes.

    This is a population spread, not a band on one prediction. The two are drawn differently
    everywhere in this report for that reason: a band on a single predicted curve is the
    simultaneous conformal product of build spec 11.2 and appears in the calibration figures
    with its measured coverage attached, while this fan says where the beams themselves land.
    """
    frame = envelope.sort_values("u_mm")
    u = frame["u_mm"].to_numpy(dtype=float)
    quantiles = {
        level: frame[f"{prefix}_p{level:02d}"].to_numpy(dtype=float) * scale
        for level in (5, 25, 50, 75, 95)
    }
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4))
    ax.fill_between(u, quantiles[5], quantiles[95], color=C_BAND, alpha=0.45, lw=0, zorder=2)
    ax.fill_between(u, quantiles[25], quantiles[75], color=C_BAND, alpha=0.75, lw=0, zorder=3)
    ax.plot(u, quantiles[50], color=C_SERIES_1, lw=1.8, zorder=4)
    ax.set_xlabel("Imposed displacement $U_2$ [mm]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(u.min()), float(u.max()))
    ax.set_ylim(0.0, None)
    handles = [
        Line2D([], [], color=C_SERIES_1, lw=1.8, label="median predicted curve"),
        Patch(facecolor=C_BAND, alpha=0.75, edgecolor="none", label="50\\,\\% envelope"),
        Patch(facecolor=C_BAND, alpha=0.45, edgecolor="none", label="90\\,\\% envelope"),
    ]
    ax.legend(handles=handles, loc="lower left")
    ax.annotate(
        f"{n_curves} propagated draws\n"
        f"{100.0 * out_of_domain_fraction:.0f}\\,\\% outside the validity domain",
        xy=(0.985, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        **{key: value for key, value in annotation_style().items() if key not in ("ha", "va")},
    )
    fig.tight_layout(pad=0.4)
    return fig


def analytic_comparison(
    frame: Any,
    surrogate_quantiles: dict[str, float],
    analytic_quantiles: dict[str, float],
    scale: float,
    model_error: float,
) -> Any:
    """The cross check of build spec 13.4: two densities overlaid with quantile markers.

    The markers are the 5th, 50th and 95th percentiles of each distribution, drawn on the
    baseline so the comparison can be read as a comparison of locations and widths rather than
    of two curve shapes. The band around the analytic median is the stated model error, which
    is what the bracketing verdict is measured against.
    """
    x = frame["peak_load_N"].to_numpy(dtype=float) * scale
    surrogate = frame["surrogate_density"].to_numpy(dtype=float) / scale
    analytic = frame["analytic_density"].to_numpy(dtype=float) / scale
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4))
    top = max(float(surrogate.max()), float(analytic.max())) * 1.30
    median = analytic_quantiles["median"] * scale
    ax.axvspan(
        median * (1.0 - model_error),
        median * (1.0 + model_error),
        color=C_MUTED,
        alpha=0.13,
        lw=0,
        zorder=1,
    )
    ax.fill_between(x, surrogate, color=C_BAND, alpha=0.7, lw=0, zorder=2)
    ax.plot(x, surrogate, color=C_SERIES_1, lw=1.6, zorder=4)
    ax.plot(x, analytic, color=C_SERIES_2, lw=1.6, zorder=5)
    for values, color, offset in (
        (surrogate_quantiles, C_SERIES_1, 0.035),
        (analytic_quantiles, C_SERIES_2, 0.075),
    ):
        for key in ("p05", "median", "p95"):
            ax.plot(
                [values[key] * scale],
                [offset * top],
                marker="|",
                ms=10.0,
                markeredgewidth=1.6,
                color=color,
                zorder=6,
            )
    ax.set_xlabel("Peak load $RF_{2,\\max}$ [kN]")
    ax.set_ylabel("probability density")
    ax.set_ylim(0.0, top)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor=C_BAND, edgecolor=C_SERIES_1, lw=1.3, label="surrogate, aleatory layer"),
        Line2D([], [], color=C_SERIES_2, lw=1.6, label="analytic mechanics model"),
        Patch(
            facecolor=C_MUTED,
            alpha=0.3,
            edgecolor="none",
            label=f"stated model error, {100.0 * model_error:.0f}\\,\\%",
        ),
        Line2D(
            [], [], marker="|", ls="none", ms=10.0, markeredgewidth=1.6, color=C_INK_2,
            label="5th, 50th, 95th percentiles",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    return fig
