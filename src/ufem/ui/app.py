"""The five panels of UFEM Lab, wired to the artifact store (build spec section 15).

``ufem lab`` calls :func:`run_lab`, which loads a :class:`ufem.ui.store.LabStore` once and
serves it at ``http://127.0.0.1:8080``. The store is loaded before the server starts, so a
pipeline that has not run produces a named error on the command line rather than a dashboard
full of empty panels.

The panels, and what each one is careful about:

1. **Predict.** Three sliders bounded to the executed design, the calibrated curves morphing as
   they move, and a gray curve with a named censoring warning outside the validity domain.
2. **Dataset.** The 400 point design, completed against failed, the completion probability
   surface, and the click through from a completed point to its finite element curve with the
   surrogate drawn over it.
3. **Sensitivity.** The honest panel. Every chaos expansion in this campaign failed the Q2 gate
   of build spec 12.1, so this panel shows the gate outcome and the measurements behind it, and
   it draws no Sobol bar at all. The posterior cross check is shown, labeled indicative only in
   the same words the P6 report uses.
4. **Reliability.** The propagated distributions with their limit state markers, every
   probability with its binomial standard error, its conservative bound, the out of domain mass
   and the resolvable floor, and a threshold slider that recounts the persisted Monte Carlo rows
   through :func:`ufem.propagate.recompute_limit_state`.
5. **Model card.** Versions, hashes, the calibration gate, the baselines, and the three caveats
   this project carries: the design roughness, the withheld indices, and the censoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

from ufem import __version__
from ufem.calibrate import SIGNAL_DAMAGE, SIGNAL_FORCE
from ufem.config import FEATURE_ORDER, Config
from ufem.propagate import recompute_limit_state
from ufem.sensitivity import PUBLICATION_TEX
from ufem.ui import figures, layout
from ufem.ui.predict import Prediction, export_payload, predict
from ufem.ui.store import INPUT_LABELS, LabStore
from ufem.validate import QOI_LABELS

#: The window title. The host and the port are command line defaults and live in
#: :mod:`ufem.runner` with the rest of the CLI: an address is an environment fact rather than
#: a presentational choice, and this package holds no numeric literal that is neither.
PAGE_TITLE = "UFEM Lab"


def _fmt(value: float, decimals: int) -> str:
    """A number at a stated precision, or a marker for one that is not a number."""
    if value is None or not np.isfinite(value):
        return "not defined"
    return f"{value:,.{decimals}f}"


def _fmt_share(value: float) -> str:
    """A fraction as a percentage, at the coarse precision a share deserves."""
    if value is None or not np.isfinite(value):
        return "not defined"
    return f"{value:.{layout.COARSE_DECIMALS}%}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table, so a panel can state a table without inventing a widget."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Panel 1: predict
# ---------------------------------------------------------------------------


def build_predict_panel(store: LabStore) -> None:
    """Three sliders, two morphing curves, the scalar readouts, and the export button."""
    bounds = store.design_bounds
    values: dict[str, float] = dict(store.design_midpoints)
    state: dict[str, Any] = {"prediction": predict(store, values)}

    ui.markdown(
        "### Predict\n"
        "The calibrated surrogate at one design point. The band is the simultaneous "
        f"{1.0 - store.band_alpha:.0%} sup norm band of build spec 11.2, whose coverage the "
        "calibration stage measured on held out curves; the scalar intervals are the deployed "
        "jackknife+ intervals at the same level. Sliders are bounded to the box of the "
        "executed design, because outside it there are no runs."
    )

    warning = ui.markdown("").classes("text-sm")
    warning.style(f"color: {layout.WARNING_COLOR}")

    with ui.grid(columns=len(FEATURE_ORDER)).classes("w-full gap-4"):
        sliders: dict[str, Any] = {}
        readouts: dict[str, Any] = {}
        for name in FEATURE_ORDER:
            low, high = bounds[name]
            label, unit = INPUT_LABELS[name]
            with ui.column().classes("w-full"):
                readouts[name] = ui.label(
                    f"{label}: {_fmt(values[name], layout.INPUT_DECIMALS)} {unit}"
                ).classes("text-sm font-medium")
                sliders[name] = ui.slider(
                    min=low,
                    max=high,
                    step=(high - low) / layout.SLIDER_STEPS,
                    value=values[name],
                )
                ui.label(
                    f"design range {_fmt(low, layout.INPUT_DECIMALS)} to "
                    f"{_fmt(high, layout.INPUT_DECIMALS)} {unit}"
                ).classes("text-xs").style(f"color: {layout.MUTED_COLOR}")

    prediction = state["prediction"]
    with ui.row().classes("w-full no-wrap"):
        force_plot = ui.plotly(
            figures.curve_figure(
                prediction.u_grid,
                prediction.force_mean,
                prediction.force_lower,
                prediction.force_upper,
                figures.FORCE_AXIS,
                f"Simultaneous {1.0 - store.band_alpha:.0%} band",
                layout.CURVE_PANEL_HEIGHT_PX,
                grayed=not prediction.validity.inside,
            )
        ).classes("w-full")
    with ui.row().classes("w-full no-wrap"):
        damage_plot = ui.plotly(
            figures.curve_figure(
                prediction.u_grid,
                prediction.damage_mean,
                prediction.damage_lower,
                prediction.damage_upper,
                figures.DAMAGE_AXIS,
                f"Simultaneous {1.0 - store.band_alpha:.0%} band",
                layout.DAMAGE_PANEL_HEIGHT_PX,
                grayed=not prediction.validity.inside,
            )
        ).classes("w-full")

    scalar_table = ui.markdown("")

    def _scalar_markdown(current: Prediction) -> str:
        rows = []
        for readout in current.scalars:
            mean, low, high = readout.displayed()
            rows.append(
                [
                    readout.label,
                    readout.unit,
                    _fmt(mean, layout.VALUE_DECIMALS),
                    f"[{_fmt(low, layout.VALUE_DECIMALS)}, {_fmt(high, layout.VALUE_DECIMALS)}]",
                ]
            )
        return _markdown_table(
            [
                "Quantity",
                "Unit",
                "Predicted",
                f"Jackknife+ {1.0 - current.band_alpha:.0%} interval",
            ],
            rows,
        )

    def _refresh() -> None:
        current = predict(store, values)
        state["prediction"] = current
        grayed = not current.validity.inside
        force_plot.figure = figures.curve_figure(
            current.u_grid,
            current.force_mean,
            current.force_lower,
            current.force_upper,
            figures.FORCE_AXIS,
            f"Simultaneous {1.0 - current.band_alpha:.0%} band",
            layout.CURVE_PANEL_HEIGHT_PX,
            grayed=grayed,
        )
        damage_plot.figure = figures.curve_figure(
            current.u_grid,
            current.damage_mean,
            current.damage_lower,
            current.damage_upper,
            figures.DAMAGE_AXIS,
            f"Simultaneous {1.0 - current.band_alpha:.0%} band",
            layout.DAMAGE_PANEL_HEIGHT_PX,
            grayed=grayed,
        )
        force_plot.update()
        damage_plot.update()
        scalar_table.set_content(_scalar_markdown(current))
        warning.set_content(
            ""
            if current.validity.inside
            else f"**Outside the validity domain.** {current.validity.reason()}"
        )
        for name in FEATURE_ORDER:
            label, unit = INPUT_LABELS[name]
            readouts[name].set_text(
                f"{label}: {_fmt(values[name], layout.INPUT_DECIMALS)} {unit}"
            )

    def _make_handler(name: str) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            values[name] = float(event.value)
            _refresh()

        return handler

    for name in FEATURE_ORDER:
        sliders[name].on_value_change(_make_handler(name))

    def _export() -> None:
        payload = export_payload(store, state["prediction"])
        ui.download.content(
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            f"ufem_lab_prediction_{store.config_sha256[: layout.SHORT_HASH_CHARS]}.json",
        )

    with ui.row().classes("items-center gap-4"):
        ui.button("Export prediction as JSON", on_click=_export)
        ui.label(
            "The export carries the config hash, the commit, and every stage manifest's "
            "output digests, so the numbers in it can be traced back to the files they came "
            "from."
        ).classes("text-xs").style(f"color: {layout.MUTED_COLOR}")

    _refresh()


# ---------------------------------------------------------------------------
# Panel 2: dataset
# ---------------------------------------------------------------------------


def completion_slices(store: LabStore) -> list[dict[str, Any]]:
    """The completion probability surface on each pair of inputs, third held at its median.

    The grid and the probabilities come from the fitted completion model in the audit
    artifact, evaluated here rather than stored, which is the one place the dashboard runs a
    model instead of reading a table. It is the same model object the validity domain check
    uses, loaded from the same pickle whose digest the artifact records, so what is drawn is
    what the domain contract is made of.
    """
    bounds = store.design_bounds
    medians = store.design_midpoints
    slices = []
    for held in FEATURE_ORDER:
        pair = [name for name in FEATURE_ORDER if name != held]
        x_input, y_input = pair
        x_axis = np.linspace(*bounds[x_input], layout.SURFACE_GRID_STEPS)
        y_axis = np.linspace(*bounds[y_input], layout.SURFACE_GRID_STEPS)
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        query = np.empty((x_mesh.size, len(FEATURE_ORDER)))
        for position, name in enumerate(FEATURE_ORDER):
            if name == x_input:
                query[:, position] = x_mesh.ravel()
            elif name == y_input:
                query[:, position] = y_mesh.ravel()
            else:
                query[:, position] = medians[name]
        probability = store.domain.completion_probability(query).reshape(x_mesh.shape)
        label, unit = INPUT_LABELS[held]
        slices.append(
            {
                "title": f"{label} held at {medians[held]:.{layout.INPUT_DECIMALS}f} {unit}",
                "x": x_axis,
                "y": y_axis,
                "z": probability,
                "x_input": x_input,
                "y_input": y_input,
            }
        )
    return slices


def build_dataset_panel(store: LabStore) -> None:
    """The design, the completion surface, and the click through to a finite element run."""
    frame = store.design_with_status
    completed = frame.loc[frame["completed"].to_numpy(dtype=bool)]
    n_designed = len(frame)
    n_completed = len(completed)

    ui.markdown(
        "### Dataset\n"
        f"The {n_designed} point Latin hypercube design as it was executed: {n_completed} runs "
        f"completed and {n_designed - n_completed} produced nothing. The survivors are a biased "
        "subsample, which is why the completion probability surface below is a product of the "
        "pipeline rather than a footnote."
    )

    matrix_plot = ui.plotly(figures.design_matrix_figure(frame)).classes("w-full")

    ui.markdown(
        "#### Completion probability\n"
        "The fitted completion model of build spec 9.4, sliced with the third input at its "
        f"design median. The heavy contour is the stamped threshold, P(complete) = "
        f"{store.domain.threshold:g}; the markers are the runs that produced nothing."
    )
    ui.plotly(
        figures.completion_surface_figure(
            completion_slices(store), float(store.domain.threshold), frame
        )
    ).classes("w-full")

    ui.markdown(
        "#### One run against the surrogate\n"
        "Click a completed point in the scatter matrix, or choose a run. The finite element "
        "curve is drawn against the surrogate's prediction at the same three inputs. This is a "
        "training run, so the comparison is in sample; the out of sample version is on the "
        "model card, where the curves were held out."
    )

    jobs = sorted(completed["job"].tolist())
    selection: dict[str, str] = {"job": jobs[0]}
    caption = ui.markdown("")
    force_overlay = ui.plotly(go_placeholder()).classes("w-full")
    damage_overlay = ui.plotly(go_placeholder()).classes("w-full")

    def _show(job: str) -> None:
        selection["job"] = job
        observed_force, observed_damage = store.observed_curves(job)
        inputs = store.job_inputs(job)
        current = predict(store, inputs)
        force_overlay.figure = figures.overlay_figure(
            current.u_grid,
            observed_force,
            current.force_mean,
            current.force_lower,
            current.force_upper,
            job,
            figures.FORCE_AXIS,
        )
        damage_overlay.figure = figures.overlay_figure(
            current.u_grid,
            observed_damage,
            current.damage_mean,
            current.damage_lower,
            current.damage_upper,
            job,
            figures.DAMAGE_AXIS,
        )
        force_overlay.update()
        damage_overlay.update()
        row = frame.loc[frame["job"] == job].iloc[0]
        caption.set_content(
            f"**{job}** at "
            + ", ".join(
                f"{INPUT_LABELS[name][0].lower()} "
                f"{inputs[name]:.{layout.INPUT_DECIMALS}f} {INPUT_LABELS[name][1]}"
                for name in FEATURE_ORDER
            )
            + f". Completion probability {row['completion_probability']:.{layout.VALUE_DECIMALS}f}"
            + f", inside the validity domain: {bool(row['inside_domain'])}."
        )
        matrix_plot.figure = figures.design_matrix_figure(frame, selected=job)
        matrix_plot.update()

    picker = ui.select(jobs, value=jobs[0], label="Completed run").classes("w-64")
    picker.on_value_change(lambda event: _show(str(event.value)))

    def _on_click(event: Any) -> None:
        """Map a Plotly click on the scatter matrix back to a job identifier.

        Plotly reports the trace and the point within it. The traces are built in a known
        order, completed then failed then the selection marker, so the completed trace is the
        one a click through is defined for; a click on a failed run has no finite element curve
        to show and says so rather than selecting the nearest completed neighbour.
        """
        points = (event.args or {}).get("points") or []
        if not points:
            return
        point = points[0]
        curve = int(point.get("curveNumber", -1))
        index = int(point.get("pointNumber", point.get("pointIndex", -1)))
        if curve != 0:
            ui.notify(
                "That run produced nothing, so there is no finite element curve to overlay.",
                color="warning",
            )
            return
        if index < 0 or index >= n_completed:
            return
        job = str(completed["job"].to_numpy()[index])
        picker.set_value(job)

    matrix_plot.on("plotly_click", _on_click)
    _show(selection["job"])


def go_placeholder() -> go.Figure:
    """An empty figure to attach to a plot element before its first real content."""
    return go.Figure()


# ---------------------------------------------------------------------------
# Panel 3: sensitivity
# ---------------------------------------------------------------------------


def build_sensitivity_panel(store: LabStore) -> None:
    """The honest panel: what the gate decided, and why there are no Sobol bars here."""
    record = store.sensitivity
    context = record["context"]
    outcome = record["gate_outcome"]
    counts = record["publication_counts"]
    n_targets = len(context["targets"])

    ui.markdown(
        "### Sensitivity\n"
        f"**Publication gate outcome: {PUBLICATION_TEX[outcome]}.** Build spec 12.1 ties "
        "publication of a Sobol index to the corrected leave one out Q2 of the chaos expansion "
        f"it came from: at or above {context['q2_publish_values']:g} the values may be "
        f"published, at or above {context['q2_publish_rankings']:g} only the rankings, and "
        f"below that nothing. In this campaign {counts[outcome]} of {n_targets} expansions "
        f"fell below the lower threshold and {counts['values'] + counts['rankings']} cleared "
        "either one, so **no index value and no input ranking is published from this "
        "campaign**, and this panel draws no Sobol bar. What follows is the evidence for that "
        "decision and the diagnostics that say whether the failure belongs to the expansion or "
        "to the campaign."
    )

    rows = []
    for name in context["targets"]:
        chaos = record["targets"][name]["pce"]
        ceiling = chaos["explainable_variance_ceiling"]
        rows.append(
            [
                name,
                str(int(chaos["n_terms"])),
                _fmt(float(chaos["q2_corrected"]), layout.PROBABILITY_DECIMALS),
                _fmt(float(ceiling), layout.VALUE_DECIMALS)
                if chaos["ceiling_readable"]
                else "not readable",
                _fmt(float(chaos["design_roughness"]["roughness_ratio"]), layout.VALUE_DECIMALS),
                PUBLICATION_TEX[chaos["publication_level"]],
            ]
        )
    ui.markdown(
        _markdown_table(
            ["Target", "Terms", "Q2 corrected", "Explainable ceiling", "Roughness", "Publish"],
            rows,
        )
    )
    ui.markdown(f"_{record['diagnostic_note']}._").classes("text-sm")

    gate_frame = pd.DataFrame(
        {
            "target": [name for name in context["targets"]],
            "q2_corrected": [
                float(record["targets"][name]["pce"]["q2_corrected"])
                for name in context["targets"]
            ],
        }
    )
    ui.plotly(
        figures.q2_gate_figure(
            gate_frame,
            float(context["q2_publish_values"]),
            float(context["q2_publish_rankings"]),
        )
    ).classes("w-full")

    agreement = record["agreement"]
    ui.markdown(
        "#### The posterior cross check, indicative only\n"
        f"{context['gp_realizations']} conditional realizations were drawn from each fitted "
        "Gaussian process posterior and a Saltelli estimate computed on each, so every index "
        "below is a distribution rather than a number. It is shown because a cross check that "
        "is only reported when it agrees is not a cross check. It is **indicative only**: the "
        "expansions it was meant to validate did not pass their gate, so what these intervals "
        "describe is a decomposition of a variance that neither surrogate family explains "
        f"well. Agreement over all assessed rows: {agreement['n_agree']} of "
        f"{agreement['n_rows']}, by the criterion that {agreement['criterion']}. Split by "
        "kind: "
        + ", ".join(
            f"{kind.replace('_', ' ')} {counts['n_agree']} of {counts['n_rows']}"
            for kind, counts in agreement["diagnosis"]["by_kind"].items()
        )
        + "."
    )

    headline = list(context["headline"])
    target_select = ui.select(headline, value=headline[0], label="Target").classes("w-64")
    posterior_plot = ui.plotly(
        figures.posterior_sobol_figure(store.gp_indices, headline[0])
    ).classes("w-full")

    def _on_target(event: Any) -> None:
        posterior_plot.figure = figures.posterior_sobol_figure(store.gp_indices, str(event.value))
        posterior_plot.update()

    target_select.on_value_change(_on_target)

    ui.markdown(
        "#### Chaos against posterior\n"
        "The horizontal axis carries the chaos indices the gate withheld, so the figure is a "
        "comparison of two unpublished quantities and is labeled as one."
    )
    ui.plotly(figures.agreement_figure(store.agreement)).classes("w-full")

    functional_rows = []
    for block, block_record in record["functional"].items():
        functional_rows.append(
            [block, PUBLICATION_TEX[block_record["publication_level"]]]
        )
    ui.markdown(
        "#### Functional indices\n"
        "The pointwise decomposition of build spec 12.3 was computed for the curve blocks "
        "below and carries the publication level of the expansions behind it. It is withheld "
        "at the same gate, so the stacked bands build spec 15 asks for are not drawn: a stack "
        "implies the shares sum to one and are worth reading, and neither is established here. "
        "The values are in `functional_indices.parquet` for anyone who wants to look at what "
        "was withheld.\n\n"
        + _markdown_table(["Block", "Publish"], functional_rows)
    )


# ---------------------------------------------------------------------------
# Panel 4: reliability
# ---------------------------------------------------------------------------


def build_reliability_panel(store: LabStore) -> None:
    """The propagated distributions, the reliability readouts, and the live threshold."""
    context = store.propagation["context"]
    validity = store.propagation["validity"]
    states = store.limit_states
    labels = {state["short_label"]: state for state in states}
    subsample_context = store.propagation["subsample"]

    ui.markdown(
        "### Reliability\n"
        f"{int(context['n_samples']):,} Monte Carlo draws of the three inputs through the "
        "calibrated surrogate. The aleatory layer is the input spread through the mean "
        "surrogate; the predictive layer adds the surrogate's own calibrated uncertainty. "
        f"{_fmt_share(float(validity['out_of_domain_fraction']))} of that mass falls outside "
        "the validity domain, and no probability below "
        f"{float(context['resolvable_pf_floor']):g} is claimed at all: with "
        f"{int(store.propagation['context']['n_training_runs'])} training runs that is the "
        "floor the campaign imposes, whatever the sample size makes it possible to print."
    )

    ui.markdown(
        _markdown_table(
            [
                "Limit state",
                "Threshold, configured units",
                "Pf",
                "Binomial SE",
                "Conservative bound",
                "Pf inside domain",
            ],
            [
                [
                    state["short_label"],
                    _fmt(float(state["threshold"]), layout.VALUE_DECIMALS),
                    _fmt(float(state["pf_point"]), layout.PROBABILITY_DECIMALS),
                    _fmt(float(state["pf_standard_error"]), layout.PROBABILITY_DECIMALS),
                    _fmt(float(state["pf_conservative"]), layout.PROBABILITY_DECIMALS),
                    _fmt(float(state["pf_inside_domain"]), layout.PROBABILITY_DECIMALS),
                ]
                for state in states
            ],
        )
    )

    first = states[0]
    chosen: dict[str, Any] = {"state": first, "threshold": float(first["threshold"])}

    selector = ui.select(
        list(labels), value=first["short_label"], label="Limit state"
    ).classes("w-96")

    ui.markdown(
        "#### Threshold\n"
        f"The slider recounts the {int(subsample_context['n_rows']):,} Monte Carlo rows the "
        "propagate stage persisted, through the same function that produced the table above. "
        "It is a subsample of the headline sample, so its probability carries the larger "
        "standard error printed beside it; the headline numbers in the table are counted on "
        f"all {int(context['n_samples']):,} draws."
    )

    def _threshold_bounds(state: dict[str, Any]) -> tuple[float, float]:
        column = f"{state['target']}_mean"
        values = store.mc_subsample[column].to_numpy(dtype=float)
        return float(values.min()), float(values.max())

    slider_slot = ui.column().classes("w-full")
    readout = ui.markdown("")
    plot = ui.plotly(go_placeholder()).classes("w-full")

    def _density(target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subset = store.density.loc[store.density["target"] == target]
        return (
            subset["value"].to_numpy(dtype=float),
            subset["aleatory_density"].to_numpy(dtype=float),
            subset["predictive_density"].to_numpy(dtype=float),
        )

    def _refresh() -> None:
        state = chosen["state"]
        threshold = chosen["threshold"]
        target = state["target"]
        fresh = recompute_limit_state(
            store.mc_subsample, target, state["direction"], threshold
        )
        values, aleatory, predictive = _density(target)
        plot.figure = figures.density_figure(
            values,
            aleatory,
            predictive,
            threshold,
            state["direction"],
            f"{QOI_LABELS[target]}, in the propagated units of {target}",
        )
        plot.update()
        resolvable = (
            "above the resolvable floor"
            if fresh["resolvable"]
            else f"**below the resolvable floor of {float(context['resolvable_pf_floor']):g}, "
            "so it is reported as unresolved rather than as a number**"
        )
        readout.set_content(
            f"**{state['label']}**, failure when {QOI_LABELS[target]} is "
            f"{state['direction']} {_fmt(threshold, layout.VALUE_DECIMALS)} "
            f"(the configured threshold is "
            f"{_fmt(float(state['threshold']), layout.VALUE_DECIMALS)}).\n\n"
            + _markdown_table(
                ["Quantity", "Value"],
                [
                    [
                        "Pf, point estimate",
                        _fmt(fresh["pf_point"], layout.PROBABILITY_DECIMALS),
                    ],
                    [
                        "Binomial standard error",
                        _fmt(fresh["pf_standard_error"], layout.PROBABILITY_DECIMALS),
                    ],
                    [
                        "Wilson 95 percent interval",
                        f"[{_fmt(fresh['pf_wilson_low'], layout.PROBABILITY_DECIMALS)}, "
                        f"{_fmt(fresh['pf_wilson_high'], layout.PROBABILITY_DECIMALS)}]",
                    ],
                    [
                        "Conservative bound, band crossing",
                        _fmt(fresh["pf_conservative"], layout.PROBABILITY_DECIMALS),
                    ],
                    [
                        "Pf restricted to the validity domain",
                        _fmt(fresh["pf_inside_domain"], layout.PROBABILITY_DECIMALS),
                    ],
                    [
                        "Out of domain mass fraction",
                        _fmt_share(fresh["out_of_domain_fraction"]),
                    ],
                    ["Failures counted", f"{fresh['n_failures']:,} of {fresh['n_samples']:,}"],
                    ["Resolvable", resolvable],
                ],
            )
        )

    def _on_threshold(event: Any) -> None:
        chosen["threshold"] = float(event.value)
        _refresh()

    def _rebuild_slider() -> None:
        """A new slider per limit state, because its range is that quantity's own range.

        Rebuilt rather than reconfigured: the three limit states live on three quantities
        whose sampled ranges have nothing to do with each other, and a slider that kept the
        previous bounds would offer thresholds the Monte Carlo sample never reaches.
        """
        state = chosen["state"]
        bottom, top = _threshold_bounds(state)
        slider_slot.clear()
        with slider_slot:
            ui.slider(
                min=bottom,
                max=top,
                step=(top - bottom) / layout.SLIDER_STEPS,
                value=chosen["threshold"],
            ).on_value_change(_on_threshold)
            ui.label(
                f"sampled range of {QOI_LABELS[state['target']]}: "
                f"{_fmt(bottom, layout.VALUE_DECIMALS)} to {_fmt(top, layout.VALUE_DECIMALS)}"
            ).classes("text-xs").style(f"color: {layout.MUTED_COLOR}")

    def _on_state(event: Any) -> None:
        state = labels[str(event.value)]
        chosen["state"] = state
        chosen["threshold"] = float(state["threshold"])
        _rebuild_slider()
        _refresh()

    selector.on_value_change(_on_state)
    _rebuild_slider()

    ui.markdown(
        "#### What is not claimed\n"
        f"{store.propagation['roughness_caveat']}"
    ).classes("text-sm")

    _refresh()


# ---------------------------------------------------------------------------
# Panel 5: model card
# ---------------------------------------------------------------------------


def build_model_card_panel(store: LabStore) -> None:
    """Versions, hashes, coverage, baselines, and the caveats, all out of the manifests."""
    git = store.git_state
    calibration = store.calibration
    gate = calibration["gate"]
    record = store.surrogate_record

    ui.markdown(
        "### Model card\n"
        f"UFEM {__version__}, config `{store.config_sha256}`, surrogate fitted at commit "
        f"`{git['commit'][: layout.SHORT_HASH_CHARS]}` on branch `{git['branch']}`"
        + (", working tree dirty" if git.get("dirty") else "")
        + "."
    )

    ui.markdown(
        "#### Resolved stack\n"
        + _markdown_table(
            ["Component", "Version"],
            [[name, value] for name, value in sorted(store.package_versions.items())],
        )
    )

    ui.markdown(
        "#### Scope and validity domain\n"
        f"Trained on {int(record['n_training_runs'])} completed runs of a "
        f"{int(store.censoring['n_designed'])} point design. Inputs: "
        + ", ".join(
            f"{INPUT_LABELS[name][0].lower()} in "
            f"[{store.design_bounds[name][0]:.{layout.INPUT_DECIMALS}f}, "
            f"{store.design_bounds[name][1]:.{layout.INPUT_DECIMALS}f}] "
            f"{INPUT_LABELS[name][1]}"
            for name in FEATURE_ORDER
        )
        + f". A query is inside the validity domain when the completion model gives "
        f"P(complete) at or above {store.domain.threshold:g} and the point lies inside that "
        f"box; that holds for "
        f"{_fmt_share(float(store.domain_record['dense_inside_fraction']))} of the box by "
        "volume."
    )

    def _wilson(check: dict[str, Any]) -> str:
        """The interval a check carries, or a stated reason it carries none.

        Not every gate check is a counted proportion. The probability integral transform mass
        check is a share of a histogram rather than a number of successes out of a number of
        trials, so it has no binomial interval, and printing one would be inventing a
        confidence statement the artifact does not make.
        """
        interval = check.get("wilson")
        if interval is None:
            return "not a counted proportion"
        return (
            f"[{_fmt(float(interval[0]), layout.VALUE_DECIMALS)}, "
            f"{_fmt(float(interval[1]), layout.VALUE_DECIMALS)}]"
        )

    ui.markdown(
        "#### Calibration gate\n"
        f"Gate passed: **{bool(gate['passed'])}** at alpha {float(gate['alpha']):g}, with "
        f"{len(gate['checks'])} checks and {len(gate['failing'])} failing.\n\n"
        + _markdown_table(
            ["Criterion", "Measured", "95 percent Wilson interval", "Passed"],
            [
                [
                    check["criterion"],
                    _fmt(float(check["measured"]), layout.PROBABILITY_DECIMALS),
                    _wilson(check),
                    str(bool(check["passed"])),
                ]
                for check in gate["checks"]
            ],
        )
    )

    headline = list(calibration["context"]["headline_qoi"])
    coverage_select = ui.select(headline, value=headline[0], label="Target").classes("w-64")
    coverage_plot = ui.plotly(
        figures.coverage_figure(store.coverage_sweep, headline[0])
    ).classes("w-full")

    def _on_coverage(event: Any) -> None:
        coverage_plot.figure = figures.coverage_figure(store.coverage_sweep, str(event.value))
        coverage_plot.update()

    coverage_select.on_value_change(_on_coverage)

    ui.markdown(
        "#### The worst held out curve inside its band\n"
        "Chosen by its own sup norm score, not for looking comfortable. A simultaneous band is "
        "only worth claiming if it contains the run the model handles worst."
    )
    with ui.row().classes("w-full no-wrap"):
        ui.plotly(
            figures.band_example_figure(store.band_examples, SIGNAL_FORCE, "worst")
        ).classes("w-full")
        ui.plotly(
            figures.band_example_figure(store.band_examples, SIGNAL_DAMAGE, "worst")
        ).classes("w-full")

    baselines = store.baselines
    baseline_gate = baselines["gate"]
    models = sorted({row["model"] for row in baselines["scalar"]})
    baseline_rows = []
    for target in baseline_gate["headline_qoi"]:
        row = [QOI_LABELS[target]]
        for model in models:
            match = [
                item
                for item in baselines["scalar"]
                if item["target"] == target and item["model"] == model
            ]
            row.append(
                _fmt(float(match[0]["r2_test"]), layout.VALUE_DECIMALS) if match else "not run"
            )
        baseline_rows.append(row)
    ui.markdown(
        "#### Baselines, out of sample only\n"
        f"Leave one out test R2. Gate passed: **{bool(baseline_gate['passed'])}**.\n\n"
        + _markdown_table(["Quantity"] + models, baseline_rows)
    )

    corners = store.enriched_corners
    ui.markdown(
        "#### Known failure modes and caveats\n"
        f"- **Design roughness.** {store.propagation['roughness_caveat']}\n"
        f"- **Withheld sensitivity indices.** Every one of "
        f"{len(store.sensitivity['context']['targets'])} chaos expansions failed the Q2 gate "
        "of build spec 12.1, so no Sobol index value and no input ranking is published from "
        "this campaign.\n"
        f"- **Censoring.** {int(store.censoring['n_failed'])} of "
        f"{int(store.censoring['n_designed'])} designed runs produced nothing, and the "
        "failures are not spread evenly: "
        + "; ".join(
            f"with {corner['label'].lower()} in [{corner['low']:.1f}, {corner['high']:.1f}] "
            f"{corner['unit']}, {corner['n_failed']} of {corner['n']} failed"
            for corner in corners
        )
        + ". The surrogate carries a machine checked validity domain that excludes that "
        "corner, and the predict panel grays out inside it.\n"
        "- **Damage saturation.** The damage scalar is capped by the material table in every "
        "run of this campaign, so a threshold near the cap asks when saturation is reached "
        "rather than whether it is.\n"
        "- **Fixed model parameters.** Dilation angle, eccentricity, viscosity and fracture "
        "energy were frozen in the inherited campaign, so nothing here quantifies their "
        "contribution."
    )


# ---------------------------------------------------------------------------
# The page and the server
# ---------------------------------------------------------------------------


def build_page(store: LabStore) -> None:
    """Assemble the five panels into one page."""
    ui.page_title(PAGE_TITLE)
    with ui.header().classes("items-center justify-between"):
        ui.label(
            f"{PAGE_TITLE}: {store.config_sha256[: layout.SHORT_HASH_CHARS]}"
        ).classes("text-lg font-medium")
        ui.label(
            "Every number on this page was read from an artifact the pipeline wrote."
        ).classes("text-xs")
    with ui.tabs().classes("w-full") as tabs:
        predict_tab = ui.tab("Predict")
        dataset_tab = ui.tab("Dataset")
        sensitivity_tab = ui.tab("Sensitivity")
        reliability_tab = ui.tab("Reliability")
        card_tab = ui.tab("Model card")
    with ui.tab_panels(tabs, value=predict_tab).classes("w-full"):
        with ui.tab_panel(predict_tab):
            build_predict_panel(store)
        with ui.tab_panel(dataset_tab):
            build_dataset_panel(store)
        with ui.tab_panel(sensitivity_tab):
            build_sensitivity_panel(store)
        with ui.tab_panel(reliability_tab):
            build_reliability_panel(store)
        with ui.tab_panel(card_tab):
            build_model_card_panel(store)


def run_lab(
    repo_root: Path | str,
    host: str,
    port: int,
    config: Config | None = None,
    show: bool = True,
) -> None:
    """Load the artifact store and serve UFEM Lab. Blocks until the server stops.

    The host and the port are required rather than defaulted here, so there is exactly one
    place in the repository that decides what address `ufem lab` listens on, and it is the
    command line parser that a reader would look in.
    """
    store = LabStore.load(repo_root, config)

    @ui.page("/")
    def index() -> None:
        build_page(store)

    ui.run(
        host=host,
        port=port,
        title=PAGE_TITLE,
        show=show,
        reload=False,
        favicon="\N{BAR CHART}",
        uvicorn_logging_level="warning",
        show_welcome_message=False,
    )
