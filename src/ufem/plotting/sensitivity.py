"""Figures for the global sensitivity phase, drawn from the sensitivity stage's artifacts.

Build spec 8, 12 and 20. Two figures, and between them they carry the whole of what build spec
12 asks to be shown: that the two independent constructions agree, and what the response's
sensitivity does along the softening curve rather than at one summary point.

The palette follows the style module's rule of two identity colors, with the muted ink as the
third series. That is not a workaround. The bottom cover really is the negligible input here,
so drawing it in the recessive color is the honest encoding rather than a compromise forced by
a two color palette.
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
)

#: One color per input, in ``ufem.config.FEATURE_ORDER``.
INPUT_COLORS: tuple[str, str, str] = (C_SERIES_1, C_MUTED, C_SERIES_2)

#: The fill used for whatever the first order indices do not account for.
INTERACTION_COLOR = C_BAND


def q2_gate(
    q2: np.ndarray,
    ceiling: np.ndarray,
    labels: list[str],
    thresholds: tuple[float, float],
    readable: np.ndarray | None = None,
) -> Any:
    """Every expansion's corrected leave one out Q2 against the two publication thresholds.

    This is the phase's own result rather than a diagnostic of it, so it ships whatever the
    outcome. Each target is a bar at its measured Q2 with an open marker at the explainable
    variance ceiling the fitted Gaussian process nugget implies. A bar short of the 0.80 line
    but close to its own marker is a response the design cannot resolve; a bar far below its
    marker would be an expansion that is the problem. The distinction is the whole reason the
    marker is on the figure.

    ``readable`` names the targets whose ceiling means what it says. A process that pinned
    every lengthscale at its lower bound reports a nugget near zero and therefore a ceiling
    near one, which is the interpolate the scatter corner rather than a statement that the
    response is explainable, so those markers are omitted rather than drawn misleadingly.
    """
    values = np.asarray(q2, dtype=float)
    limit = np.asarray(ceiling, dtype=float)
    usable = (
        np.ones(values.size, dtype=bool)
        if readable is None
        else np.asarray(readable, dtype=bool)
    )
    positions = np.arange(values.size, dtype=float)
    height = max(2.6, 0.20 * values.size + 1.0)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, height))
    ax.barh(
        positions,
        np.clip(values, 0.0, None),
        height=0.62,
        color=C_SERIES_1,
        lw=0,
        zorder=3,
    )
    ax.plot(
        limit[usable], positions[usable], marker="|", ls="none", ms=9.0, color=C_SERIES_2,
        markeredgewidth=1.6, zorder=5,
    )
    for value, threshold, style in zip(
        thresholds, thresholds, ((0, (4, 2.5)), (0, (1, 1.6)))
    ):
        ax.axvline(threshold, color=C_INK_2, lw=0.9, ls=style, zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Corrected leave one out $Q^2$ [-]")
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor=C_SERIES_1, edgecolor="none", label="measured $Q^2$"),
        Line2D(
            [], [], marker="|", ls="none", ms=9.0, color=C_SERIES_2, markeredgewidth=1.6,
            label="explainable ceiling",
        ),
        Line2D(
            [], [], color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)),
            label=f"rankings at {thresholds[0]:.2f}",
        ),
        Line2D(
            [], [], color=C_INK_2, lw=0.9, ls=(0, (1, 1.6)),
            label=f"values at {thresholds[1]:.2f}",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.0)
    fig.tight_layout(pad=0.4)
    return fig


def sobol_agreement(
    pce: Any,
    posterior: Any,
    targets: list[str],
    inputs: list[str],
    labels: dict[str, str],
    input_labels: dict[str, str],
    levels: dict[str, str] | None = None,
) -> Any:
    """First order and total indices per quantity: chaos values against posterior spread.

    One panel per quantity of interest. Within a panel, each input carries a wide open bar for
    the total index and a narrow filled bar for the first order index, both at the Gaussian
    process posterior median, with a whisker spanning the posterior 90 percent interval. The
    diamonds are the analytic polynomial chaos values, which have no interval of their own
    because they are a decomposition of the fitted expansion rather than an estimate of
    anything. Agreement is therefore read as whether every diamond sits inside its whisker.

    The whiskers are the surrogate's uncertainty about the index. They are not the Monte Carlo
    error of a Saltelli estimate, which is roughly two orders of magnitude smaller here and is
    reported separately in the artifact.

    ``levels`` carries each quantity's publication level from the Q2 gate of build spec 12.1.
    A panel whose level is ``not_published`` is drawn hatched, and says so in its title rather
    than in a box over the data: the withheld marking has to be unmissable, which is not the
    same as putting it where it covers the bars it is marking.
    """
    fig, axes = plt.subplots(
        1, len(targets), figsize=(FIG_WIDTH_IN, 3.1), sharey=True
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(len(inputs), dtype=float)
    gate = levels or {}
    for ax, target in zip(axes, targets):
        withheld = gate.get(target) == "not_published"
        for offset, name in enumerate(inputs):
            color = INPUT_COLORS[offset % len(INPUT_COLORS)]
            for kind, width, filled in (("total_order", 0.66, False), ("first_order", 0.34, True)):
                row = posterior.loc[
                    (posterior["target"] == target)
                    & (posterior["input"] == name)
                    & (posterior["kind"] == kind)
                ]
                if row.empty:
                    continue
                median = float(row["median"].iloc[0])
                low = float(row["low"].iloc[0])
                high = float(row["high"].iloc[0])
                ax.bar(
                    positions[offset],
                    median,
                    width=width,
                    facecolor=color if filled else "none",
                    edgecolor=color,
                    linewidth=0.0 if filled else 1.1,
                    alpha=(0.32 if withheld else 0.85) if filled else 1.0,
                    hatch="///" if (filled and withheld) else None,
                    zorder=3 if filled else 2,
                )
                ax.vlines(
                    positions[offset], low, high, color=C_INK_2, lw=0.9, zorder=5
                )
                chaos = pce.loc[
                    (pce["target"] == target) & (pce["input"] == name)
                ]
                if not chaos.empty:
                    ax.plot(
                        positions[offset],
                        float(chaos[kind].iloc[0]),
                        marker="D",
                        ms=3.6,
                        color="white",
                        markeredgecolor=C_INK_2,
                        markeredgewidth=0.9,
                        zorder=6,
                    )
        ax.set_xticks(positions)
        ax.set_xticklabels([input_labels.get(name, name) for name in inputs])
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="x", visible=False)
        title = labels.get(target, target)
        ax.set_title(
            f"{title}\n(withheld)" if withheld else title,
            fontsize=9.0,
            color=C_INK_2,
            pad=4.0,
        )
    axes[0].set_ylabel("Sobol index [-]")
    handles = [
        Patch(facecolor=C_INK_2, edgecolor="none", alpha=0.85, label="first order $S_i$"),
        Patch(facecolor="none", edgecolor=C_INK_2, linewidth=1.1, label="total $T_i$"),
        Line2D([], [], color=C_INK_2, lw=0.9, label="GP posterior 90 percent"),
        Line2D(
            [], [], marker="D", ls="none", ms=3.6, color="white",
            markeredgecolor=C_INK_2, markeredgewidth=0.9, label="sparse PCE",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=8.0,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.tight_layout(pad=0.4)
    return fig


def functional_bands(
    functional: Any,
    block: str,
    inputs: list[str],
    input_labels: dict[str, str],
    response_label: str,
    level: str | None = None,
) -> Any:
    """Pointwise first order indices stacked along the response, over the variance profile.

    The upper panel stacks ``S_i(u)`` for the three inputs; whatever is left under the unit
    line is interaction, that is variance no single input explains on its own. The lower panel
    is the pointwise variance of the field itself, on a log scale, and it is there because a
    stacked index band is meaningless where there is nothing to decompose: a station whose
    variance is a thousandth of the peak's will happily show a confident looking split of
    almost nothing. Stations the observed family does not vary at are absent from both panels
    rather than filled in.

    A withheld block says so in the panel title and the legend sits outside the axes, because a
    stacked band fills its whole panel and anything drawn on top of it hides exactly the data
    the reader came for.
    """
    rows = functional.loc[
        (functional["block"] == block) & functional["usable"].astype(bool)
    ].sort_values("u_mm")
    if rows.empty:
        raise ValueError(
            f"no usable stations for block {block!r}: every station is either outside the "
            "observed family's varying domain or carries zero explained variance."
        )
    u = rows["u_mm"].to_numpy(dtype=float)
    shares = np.vstack([rows[f"S_{name}"].to_numpy(dtype=float) for name in inputs])
    interaction = np.clip(1.0 - shares.sum(axis=0), 0.0, None)
    fig, axes = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_IN, 4.3), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0]},
    )
    colors = [INPUT_COLORS[index % len(INPUT_COLORS)] for index in range(len(inputs))]
    axes[0].stackplot(
        u,
        *list(shares),
        interaction,
        colors=[*colors, INTERACTION_COLOR],
        edgecolor="none",
        zorder=2,
    )
    axes[0].set_ylabel("Share of variance [-]")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="x", visible=False)
    handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.6,
              label=input_labels.get(name, name))
        for color, name in zip(colors, inputs)
    ]
    handles.append(
        Patch(facecolor=INTERACTION_COLOR, edgecolor="white", linewidth=0.6,
              label="interaction")
    )
    if level == "not_published":
        axes[0].set_title(
            "withheld by the $Q^2$ gate of build spec 12.1: "
            "no share below is a published index",
            fontsize=9.0,
            color=C_INK_2,
            pad=5.0,
        )
    axes[0].legend(
        handles=handles, loc="lower center", ncol=4, fontsize=8.0,
        bbox_to_anchor=(0.5, 1.0) if level != "not_published" else (0.5, 1.10),
    )
    axes[1].plot(
        u, rows["variance"].to_numpy(dtype=float), color=C_INK_2, lw=1.3, zorder=3
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Var [$\\mathrm{N}^2$]" if block == "amplitude" else "Var [-]")
    axes[1].set_xlabel(response_label)
    axes[1].set_xlim(float(u.min()), float(u.max()))
    fig.tight_layout(pad=0.4)
    return fig


def crossover_annotation(u: np.ndarray, lead: np.ndarray, rival: np.ndarray) -> float:
    """The abscissa where one index overtakes another, by linear interpolation.

    Returns ``nan`` when there is no sign change, which is a result and not an error: it says
    one input leads over the whole usable domain, and the report says which.
    """
    difference = np.asarray(lead, dtype=float) - np.asarray(rival, dtype=float)
    stations = np.asarray(u, dtype=float)
    sign_change = np.flatnonzero(np.sign(difference[:-1]) * np.sign(difference[1:]) < 0.0)
    if sign_change.size == 0:
        return float("nan")
    index = int(sign_change[0])
    left, right = difference[index], difference[index + 1]
    weight = left / (left - right)
    return float(stations[index] + weight * (stations[index + 1] - stations[index]))
