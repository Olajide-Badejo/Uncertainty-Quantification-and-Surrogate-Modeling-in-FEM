"""Plotly figures for UFEM Lab: data in, figure out.

Every function here takes arrays or frames that a caller read from the artifact store and
returns a figure. None of them opens a file, and none of them carries a number that is not a
presentation constant from :mod:`ufem.ui.layout` (binding law 5).

The house style is applied through :func:`_chrome`, which is the Plotly counterpart of
:func:`ufem.plotting.style.apply_style`: the same palette, the same recessive gridlines, no
title inside the frame, units on every axis label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ufem.config import FEATURE_ORDER
from ufem.ui import layout
from ufem.ui.store import INPUT_LABELS

#: Axis titles for the two curve families and the input axes, built from the label table so a
#: unit never appears in two spellings.
FORCE_AXIS = "Reaction force [N]"
DAMAGE_AXIS = "Compressive damage [-]"
DISPLACEMENT_AXIS = "Imposed displacement [mm]"


def _axis_label(name: str) -> str:
    label, unit = INPUT_LABELS[name]
    return f"{label} [{unit}]"


def _chrome(figure: go.Figure, height: int, legend: bool = True) -> go.Figure:
    """Apply the house style to a finished figure."""
    figure.update_layout(
        height=height,
        margin={
            "l": layout.MARGIN_PX,
            "r": layout.MARGIN_PX,
            "t": layout.TOP_MARGIN_PX,
            "b": layout.MARGIN_PX,
        },
        paper_bgcolor=layout.PAPER_COLOR,
        plot_bgcolor=layout.PAPER_COLOR,
        font={"color": layout.INK_COLOR, "size": layout.TICK_FONT},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": layout.LEGEND_PAD,
            "xanchor": "right",
            "x": 1,
            "bgcolor": layout.PAPER_COLOR,
        },
        hovermode="closest",
    )
    figure.update_xaxes(
        gridcolor=layout.GRID_COLOR,
        linecolor=layout.AXIS_COLOR,
        zeroline=False,
        title={"font": {"size": layout.LABEL_FONT}},
    )
    figure.update_yaxes(
        gridcolor=layout.GRID_COLOR,
        linecolor=layout.AXIS_COLOR,
        zeroline=False,
        title={"font": {"size": layout.LABEL_FONT}},
    )
    return figure


def _band_fill(color: str, opacity: float) -> str:
    """An rgba fill from a hex color, so a band never reads as a line."""
    red, green, blue = bytes.fromhex(color.lstrip("#"))
    return f"rgba({red}, {green}, {blue}, {opacity})"


def _padded_range(values: np.ndarray) -> list[float]:
    """A data range with the house padding, so a curve never touches the frame."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = float(finite.min())
    high = float(finite.max())
    pad = layout.AXIS_PAD_RATIO * (high - low)
    return [low - pad, high + pad]


def curve_figure(
    u_grid: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_title: str,
    band_label: str,
    height: int,
    grayed: bool = False,
) -> go.Figure:
    """One predicted curve with its calibrated band.

    ``grayed`` is the validity domain rule of build spec 15: outside the domain the curve is
    still drawn, because its shape is still the model's answer, but it is drawn in gray and the
    panel puts a warning above it. Hiding the curve would suggest the model refused; graying it
    says the model answered and the answer is not evidence.
    """
    line_color = layout.GRAYED_COLOR if grayed else layout.PREDICTION_COLOR
    fill_opacity = layout.GRAYED_OPACITY if grayed else layout.BAND_OPACITY
    fill_color = layout.GRAYED_COLOR if grayed else layout.BAND_COLOR
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([u_grid, u_grid[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=_band_fill(fill_color, fill_opacity),
            line={"width": 0},
            hoverinfo="skip",
            name=band_label,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=u_grid,
            y=mean,
            mode="lines",
            line={"color": line_color, "width": layout.LINE_WIDTH},
            name="Predicted mean",
        )
    )
    figure.update_xaxes(title=DISPLACEMENT_AXIS)
    figure.update_yaxes(title=y_title, range=_padded_range(np.concatenate([lower, upper])))
    return _chrome(figure, height)


def overlay_figure(
    u_grid: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    job: str,
    y_title: str,
) -> go.Figure:
    """The finite element curve of one completed run against the surrogate at its inputs.

    Build spec 15 calls this the single most convincing view in the app, and the reason is that
    it is the only one where the reader can see the model being wrong. The run is a training
    run, so this is an in sample comparison and the panel says so; the out of sample version of
    the same picture is the calibration stage's band examples, which the model card shows.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([u_grid, u_grid[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=_band_fill(layout.BAND_COLOR, layout.BAND_OPACITY),
            line={"width": 0},
            hoverinfo="skip",
            name="Calibrated band",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=u_grid,
            y=mean,
            mode="lines",
            line={"color": layout.PREDICTION_COLOR, "width": layout.LINE_WIDTH},
            name="Surrogate at these inputs",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=u_grid,
            y=observed,
            mode="lines",
            line={
                "color": layout.COMPARISON_COLOR,
                "width": layout.LINE_WIDTH,
                "dash": layout.DASH_PATTERN_DASH,
            },
            name=f"Finite element run {job}",
        )
    )
    figure.update_xaxes(title=DISPLACEMENT_AXIS)
    figure.update_yaxes(title=y_title)
    return _chrome(figure, layout.OVERLAY_PANEL_HEIGHT_PX)


def design_matrix_figure(frame: pd.DataFrame, selected: str | None = None) -> go.Figure:
    """The 400 point design as a scatter matrix, completed against failed.

    Two traces rather than a color scale, because completion is a category and a continuous
    ramp over a binary would invite reading an ordering into it. The selected point, if there
    is one, is drawn as a third trace so a click has somewhere to land.
    """
    dimensions = [
        {"label": _axis_label(name), "values": frame[name].to_numpy(dtype=float)}
        for name in FEATURE_ORDER
    ]
    figure = go.Figure()
    for completed, color, name in (
        (True, layout.COMPLETED_COLOR, "Completed"),
        (False, layout.FAILED_COLOR, "Failed"),
    ):
        mask = frame["completed"].to_numpy(dtype=bool) == completed
        figure.add_trace(
            go.Splom(
                dimensions=[
                    {"label": item["label"], "values": item["values"][mask]}
                    for item in dimensions
                ],
                name=name,
                text=frame["job"].to_numpy()[mask],
                marker={
                    "color": color,
                    "size": layout.SMALL_MARKER_SIZE,
                    "opacity": layout.SCATTER_OPACITY,
                    "line": {"width": 0},
                },
                diagonal={"visible": False},
                showupperhalf=False,
            )
        )
    if selected is not None and (frame["job"] == selected).any():
        row = frame.loc[frame["job"] == selected]
        figure.add_trace(
            go.Splom(
                dimensions=[
                    {"label": _axis_label(name), "values": row[name].to_numpy(dtype=float)}
                    for name in FEATURE_ORDER
                ],
                name=f"Selected: {selected}",
                marker={
                    "color": layout.COMPARISON_COLOR,
                    "size": layout.SELECTED_MARKER_SIZE,
                    "symbol": "circle-open",
                    "line": {"width": layout.LINE_WIDTH, "color": layout.COMPARISON_COLOR},
                },
                diagonal={"visible": False},
                showupperhalf=False,
            )
        )
    return _chrome(figure, layout.MATRIX_PANEL_HEIGHT_PX)


def completion_surface_figure(
    slices: list[dict[str, Any]], threshold: float, frame: pd.DataFrame
) -> go.Figure:
    """Slices of the fitted completion probability surface, one per held fixed input.

    The contour at the stamped threshold is drawn heavier than the rest, because that contour
    is the boundary the whole validity domain contract of build spec 9.4 is about: inside it
    the surrogate is interpolating between runs that exist, outside it the campaign has
    nothing to say. The failed design points are drawn on top so the surface can be read
    against the evidence that produced it rather than on its own.
    """
    figure = make_subplots(
        rows=1,
        cols=len(slices),
        subplot_titles=[item["title"] for item in slices],
        horizontal_spacing=layout.SUBPLOT_SPACING_RATIO,
    )
    for position, item in enumerate(slices, start=1):
        figure.add_trace(
            go.Contour(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                colorscale="Blues",
                contours={
                    "start": 0,
                    "end": 1,
                    "size": layout.CONTOUR_SIZE_RATIO,
                    "showlines": False,
                },
                showscale=position == len(slices),
                showlegend=False,
                colorbar={"title": "P(complete)"},
                hovertemplate="P(complete) = %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        figure.add_trace(
            go.Contour(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                showscale=False,
                contours={
                    "start": threshold,
                    "end": threshold,
                    "coloring": "none",
                    "showlabels": True,
                },
                line={"color": layout.INK_COLOR, "width": layout.LINE_WIDTH},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=position,
        )
        failed = frame.loc[~frame["completed"].to_numpy(dtype=bool)]
        figure.add_trace(
            go.Scatter(
                x=failed[item["x_input"]].to_numpy(dtype=float),
                y=failed[item["y_input"]].to_numpy(dtype=float),
                mode="markers",
                marker={
                    "color": layout.FAILED_COLOR,
                    "size": layout.SMALL_MARKER_SIZE,
                    "opacity": layout.SCATTER_OPACITY,
                },
                name="Failed runs",
                showlegend=position == 1,
                hovertemplate="%{text}<extra></extra>",
                text=failed["job"].to_numpy(),
            ),
            row=1,
            col=position,
        )
        figure.update_xaxes(title=_axis_label(item["x_input"]), row=1, col=position)
        figure.update_yaxes(title=_axis_label(item["y_input"]), row=1, col=position)
    return _chrome(figure, layout.SURFACE_PANEL_HEIGHT_PX)


def q2_gate_figure(
    frame: pd.DataFrame, publish_values: float, publish_rankings: float
) -> go.Figure:
    """The Q2 of every chaos expansion against the two publication thresholds.

    This is what the sensitivity panel draws instead of Sobol bars. The bars are corrected
    leave one out Q2 values, which are measurements of the expansions, and the two vertical
    rules are the thresholds of build spec 12.1. A reader can see at a glance that every bar
    falls left of both, which is the finding.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["q2_corrected"].to_numpy(dtype=float),
            y=frame["target"].to_numpy(),
            orientation="h",
            marker={"color": layout.MUTED_COLOR},
            name="Corrected leave one out Q2",
            hovertemplate="%{y}: Q2 = %{x:.4f}<extra></extra>",
        )
    )
    for value, name, color in (
        (publish_rankings, "Rankings publishable", layout.COMPARISON_COLOR),
        (publish_values, "Values publishable", layout.PREDICTION_COLOR),
    ):
        figure.add_vline(
            x=value,
            line={"color": color, "width": layout.LINE_WIDTH, "dash": layout.DASH_PATTERN_DASH},
            annotation={"text": f"{name} at {value:g}", "font": {"size": layout.TICK_FONT}},
        )
    figure.update_xaxes(title="Corrected leave one out Q2 [-]")
    figure.update_yaxes(title="", autorange="reversed")
    return _chrome(figure, layout.SENSITIVITY_PANEL_HEIGHT_PX, legend=False)


def posterior_sobol_figure(frame: pd.DataFrame, target: str) -> go.Figure:
    """The Gaussian process posterior Sobol distributions for one target.

    Drawn as intervals rather than as bars with whiskers, and labeled indicative only, because
    the expansions these cross check failed their gate. What a posterior Sobol distribution
    says when its chaos counterpart is withheld is that two surrogate families disagree about
    the decomposition of a variance neither of them explains well, and a bar chart would state
    it far more confidently than that.
    """
    subset = frame.loc[frame["target"] == target]
    figure = go.Figure()
    for kind, color in (
        ("first_order", layout.PREDICTION_COLOR),
        ("total_order", layout.COMPARISON_COLOR),
    ):
        rows = subset.loc[subset["kind"] == kind]
        median = rows["median"].to_numpy(dtype=float)
        low = rows["low"].to_numpy(dtype=float)
        high = rows["high"].to_numpy(dtype=float)
        figure.add_trace(
            go.Scatter(
                x=median,
                y=[_axis_label(name) for name in rows["input"]],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": high - median,
                    "arrayminus": median - low,
                    "color": color,
                    "width": layout.MARKER_SIZE,
                },
                mode="markers",
                marker={"color": color, "size": layout.MARKER_SIZE},
                name=kind.replace("_", " "),
                hovertemplate="%{x:.3f}<extra></extra>",
            )
        )
    figure.update_xaxes(title="Posterior Sobol index, indicative only [-]")
    figure.update_yaxes(title="")
    return _chrome(figure, layout.DIAGNOSTIC_PANEL_HEIGHT_PX)


def agreement_figure(frame: pd.DataFrame) -> go.Figure:
    """Chaos index against posterior median, with the posterior interval as an error bar."""
    figure = go.Figure()
    for kind, color in (
        ("first_order", layout.PREDICTION_COLOR),
        ("total_order", layout.COMPARISON_COLOR),
    ):
        rows = frame.loc[frame["kind"] == kind]
        median = rows["gp_median"].to_numpy(dtype=float)
        figure.add_trace(
            go.Scatter(
                x=rows["pce"].to_numpy(dtype=float),
                y=median,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": rows["gp_high"].to_numpy(dtype=float) - median,
                    "arrayminus": median - rows["gp_low"].to_numpy(dtype=float),
                    "color": color,
                    "width": layout.SMALL_MARKER_SIZE,
                },
                mode="markers",
                marker={"color": color, "size": layout.MARKER_SIZE},
                name=kind.replace("_", " "),
                text=rows["target"].to_numpy(),
                hovertemplate="%{text}<br>chaos %{x:.3f}, posterior %{y:.3f}<extra></extra>",
            )
        )
    identity = np.array([0.0, 1.0])
    figure.add_trace(
        go.Scatter(
            x=identity,
            y=identity,
            mode="lines",
            line={
                "color": layout.MUTED_COLOR,
                "width": layout.HAIRLINE_WIDTH,
                "dash": layout.DASH_PATTERN_DOT,
            },
            name="Agreement",
            hoverinfo="skip",
        )
    )
    figure.update_xaxes(title="Sparse chaos index, withheld [-]")
    figure.update_yaxes(title="Posterior Sobol median, indicative [-]")
    return _chrome(figure, layout.DIAGNOSTIC_PANEL_HEIGHT_PX)


def density_figure(
    values: np.ndarray,
    aleatory: np.ndarray,
    predictive: np.ndarray,
    threshold: float,
    direction: str,
    x_title: str,
) -> go.Figure:
    """One propagated distribution with its limit state marker and the failing region.

    The failing side is shaded, and which side that is comes from the limit state's declared
    direction rather than from a guess, because shading the wrong side is a defect this project
    has already shipped once and recorded (docs/DEFECT_LOG.md).
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=values,
            y=aleatory,
            mode="lines",
            line={"color": layout.PREDICTION_COLOR, "width": layout.LINE_WIDTH},
            name="Aleatory layer",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=values,
            y=predictive,
            mode="lines",
            line={
                "color": layout.COMPARISON_COLOR,
                "width": layout.LINE_WIDTH,
                "dash": layout.DASH_PATTERN_DASH,
            },
            name="Predictive layer",
        )
    )
    failing = values <= threshold if direction == "below" else values >= threshold
    if failing.any():
        figure.add_trace(
            go.Scatter(
                x=values[failing],
                y=aleatory[failing],
                fill="tozeroy",
                fillcolor=_band_fill(layout.WARNING_COLOR, layout.ENVELOPE_OPACITY),
                line={"width": 0},
                name="Failing region",
                hoverinfo="skip",
            )
        )
    figure.add_vline(
        x=threshold,
        line={
            "color": layout.WARNING_COLOR,
            "width": layout.LINE_WIDTH,
            "dash": layout.DASH_PATTERN_DASH,
        },
        annotation={"text": "Limit state", "font": {"size": layout.TICK_FONT}},
    )
    figure.update_xaxes(title=x_title, range=_padded_range(values))
    figure.update_yaxes(title="Density [1/unit]")
    return _chrome(figure, layout.DENSITY_PANEL_HEIGHT_PX)


def coverage_figure(frame: pd.DataFrame, target: str) -> go.Figure:
    """Nominal against empirical coverage for one target, before and after calibration."""
    subset = frame.loc[frame["target"] == target].sort_values("nominal")
    nominal = subset["nominal"].to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=nominal,
            y=nominal,
            mode="lines",
            line={
                "color": layout.MUTED_COLOR,
                "width": layout.HAIRLINE_WIDTH,
                "dash": layout.DASH_PATTERN_DOT,
            },
            name="Nominal",
            hoverinfo="skip",
        )
    )
    for column, color, name in (
        ("before_empirical", layout.COMPARISON_COLOR, "Before variance scaling"),
        ("after_empirical", layout.PREDICTION_COLOR, "After variance scaling"),
        ("conformal_empirical", layout.MUTED_COLOR, "Deployed jackknife+"),
    ):
        if column not in subset.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=nominal,
                y=subset[column].to_numpy(dtype=float),
                mode="lines+markers",
                line={"color": color, "width": layout.LINE_WIDTH},
                marker={"size": layout.SMALL_MARKER_SIZE},
                name=name,
            )
        )
    figure.update_xaxes(title="Nominal coverage [-]")
    figure.update_yaxes(title="Empirical leave one out coverage [-]")
    return _chrome(figure, layout.DIAGNOSTIC_PANEL_HEIGHT_PX)


def band_example_figure(frame: pd.DataFrame, signal: str, example: str) -> go.Figure:
    """One held out curve inside its simultaneous band, from the calibration artifact."""
    subset = frame.loc[(frame["signal"] == signal) & (frame["example"] == example)]
    if subset.empty:
        raise KeyError(
            f"the calibration artifact carries no {example!r} example for signal {signal!r}; "
            f"it holds {sorted(frame['example'].unique())}."
        )
    u_grid = subset["u_mm"].to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([u_grid, u_grid[::-1]]),
            y=np.concatenate(
                [
                    subset["upper"].to_numpy(dtype=float),
                    subset["lower"].to_numpy(dtype=float)[::-1],
                ]
            ),
            fill="toself",
            fillcolor=_band_fill(layout.BAND_COLOR, layout.BAND_OPACITY),
            line={"width": 0},
            hoverinfo="skip",
            name="Simultaneous band",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=u_grid,
            y=subset["loo_mean"].to_numpy(dtype=float),
            mode="lines",
            line={"color": layout.PREDICTION_COLOR, "width": layout.LINE_WIDTH},
            name="Out of fold mean",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=u_grid,
            y=subset["truth"].to_numpy(dtype=float),
            mode="lines",
            line={
                "color": layout.COMPARISON_COLOR,
                "width": layout.LINE_WIDTH,
                "dash": layout.DASH_PATTERN_DASH,
            },
            name=f"Held out run {subset['job'].iloc[0]}",
        )
    )
    figure.update_xaxes(title=DISPLACEMENT_AXIS)
    figure.update_yaxes(title=FORCE_AXIS if signal == "force" else DAMAGE_AXIS)
    return _chrome(figure, layout.DIAGNOSTIC_PANEL_HEIGHT_PX)
