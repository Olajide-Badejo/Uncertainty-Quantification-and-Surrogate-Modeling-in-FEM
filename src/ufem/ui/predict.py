"""One prediction: the curves, the calibrated band, the scalar intervals, the verdict.

This is the compute side of the predict panel of build spec section 15, kept out of the
NiceGUI code so it can be measured and tested without a browser. The latency budget of the
panel is a budget on :func:`predict` plus serialization, and ``tests/test_ui.py`` asserts it
over a hundred seeded slider positions.

Every constant in the arithmetic comes from an artifact:

* the mean curves and the pointwise variance come from the fitted surrogate;
* the simultaneous band half width is ``band_scale * tau * sigma(u)``, and ``band_scale`` and
  ``tau`` are the two numbers the calibration stage measured against held out curves;
* the scalar interval is :func:`ufem.propagate.calibrated_band`, the same deployed jackknife+
  construction the reliability numbers were counted with, reading the same conformal scores;
* the validity verdict is :mod:`ufem.validity`, which is the single place that question is
  decided for the whole project.

Nothing here decides anything. It assembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ufem.calibrate import SIGNAL_DAMAGE, SIGNAL_FORCE
from ufem.config import FEATURE_ORDER
from ufem.manifest import MANIFEST_NAME
from ufem.propagate import QOI_DISPLAY, calibrated_band, predict_mean_and_variance
from ufem.ui.store import INPUT_LABELS, LabStore
from ufem.validate import QOI_LABELS


@dataclass(frozen=True)
class ScalarReadout:
    """One scalar quantity of interest with its calibrated jackknife+ interval."""

    name: str
    label: str
    unit: str
    display_scale: float
    mean: float
    sigma: float
    lower: float
    upper: float

    def displayed(self) -> tuple[float, float, float]:
        """``(mean, lower, upper)`` in the unit the report reads this quantity in."""
        return (
            self.mean * self.display_scale,
            self.lower * self.display_scale,
            self.upper * self.display_scale,
        )


@dataclass(frozen=True)
class ValidityVerdict:
    """Whether the surrogate is allowed to speak here, and if not, which corner this is.

    ``censored_corners`` carries the enriched failure corners of the campaign, from the audit
    stage's censoring statistics, filtered to the ones this query actually sits in. That is
    what turns a warning from "outside the validity domain" into a sentence naming the region
    the solver could not complete, which is what build spec 15 asks the panel for.
    """

    inside: bool
    completion_probability: float
    threshold: float
    inside_design_box: bool
    outside_inputs: tuple[dict[str, Any], ...]
    censored_corners: tuple[dict[str, Any], ...]

    def reason(self) -> str:
        """One sentence saying why the prediction is grayed out, or that it is not."""
        if self.inside:
            return ""
        parts: list[str] = []
        for record in self.outside_inputs:
            parts.append(
                f"{record['label']} is {record['value']:g} {record['unit']}, outside the "
                f"executed design range [{record['low']:g}, {record['high']:g}] {record['unit']}"
            )
        if self.completion_probability < self.threshold:
            parts.append(
                f"the completion model gives P(complete) = {self.completion_probability:.3f}, "
                f"below the stamped threshold of {self.threshold:g}"
            )
        for corner in self.censored_corners:
            parts.append(
                f"this is the censored corner: with {corner['label'].lower()} in "
                f"[{corner['low']:.1f}, {corner['high']:.1f}] {corner['unit']}, "
                f"{corner['n_failed']} of {corner['n']} designed runs produced nothing"
            )
        return "; ".join(parts) + "."


@dataclass(frozen=True)
class Prediction:
    """The whole payload one slider position produces."""

    inputs: dict[str, float]
    u_grid: np.ndarray
    force_mean: np.ndarray
    force_lower: np.ndarray
    force_upper: np.ndarray
    damage_mean: np.ndarray
    damage_lower: np.ndarray
    damage_upper: np.ndarray
    scalars: tuple[ScalarReadout, ...]
    validity: ValidityVerdict
    band_alpha: float


def _feature_row(values: dict[str, float]) -> np.ndarray:
    """The three inputs as a single design row, in the pinned feature order."""
    missing = [name for name in FEATURE_ORDER if name not in values]
    if missing:
        raise KeyError(
            f"a prediction needs every input of the feature contract; missing {missing}. The "
            f"contract is {list(FEATURE_ORDER)}."
        )
    return np.array([[float(values[name]) for name in FEATURE_ORDER]], dtype=float)


def validity_verdict(store: LabStore, values: dict[str, float]) -> ValidityVerdict:
    """Ask :mod:`ufem.validity` where this point stands, and name the corner if it is out."""
    row = _feature_row(values)
    bounds = store.design_bounds
    probability = float(store.domain.completion_probability(row)[0])
    inside_box = bool(store.domain.inside_design_box(row)[0])
    outside = []
    for name in FEATURE_ORDER:
        low, high = bounds[name]
        value = float(values[name])
        if value < low or value > high:
            label, unit = INPUT_LABELS[name]
            outside.append(
                {"input": name, "label": label, "unit": unit, "value": value,
                 "low": low, "high": high}
            )
    corners = tuple(
        corner
        for corner in store.enriched_corners
        if corner["low"] <= float(values[corner["input"]]) <= corner["high"]
    )
    return ValidityVerdict(
        inside=bool(probability >= store.domain.threshold and inside_box),
        completion_probability=probability,
        threshold=float(store.domain.threshold),
        inside_design_box=inside_box,
        outside_inputs=tuple(outside),
        censored_corners=corners,
    )


def predict(store: LabStore, values: dict[str, float]) -> Prediction:
    """Predict at one design point, with the calibrated band and the validity verdict.

    The curve band is the simultaneous sup norm band of build spec 11.2 at the level the
    calibration measured coverage for, evaluated with the same two factors the artifact
    records. It is a band on this one predicted curve, not the spread of the beam population;
    the propagation stage owns that fan and the reliability panel is where it is drawn.
    """
    row = _feature_row(values)
    surrogate = store.surrogate
    standardized = surrogate.feature_standardizer.transform(row)
    pieces = store.posterior_pieces

    moments = {
        name: predict_mean_and_variance(
            surrogate.models[name],
            surrogate.target_standardizers[name],
            standardized,
            pieces[name],
        )
        for name in surrogate.models
    }
    means = {
        block: np.array([[moments[name][0][0] for name in names]], dtype=float)
        for block, names in surrogate.score_targets.items()
    }
    variances = {
        block: np.array([[moments[name][1][0] for name in names]], dtype=float)
        for block, names in surrogate.score_targets.items()
    }
    curve = surrogate.curve_from_scores(means, variances)

    force_scale, force_tau = store.functional_band(SIGNAL_FORCE)
    damage_scale, damage_tau = store.functional_band(SIGNAL_DAMAGE)
    force_half = force_scale * force_tau * curve.force_std()[0]
    damage_half = damage_scale * damage_tau * curve.damage_std()[0]

    scalars = []
    scores = store.conformal_scores
    scaling = store.scalar_scaling
    for name in surrogate.scalar_targets:
        lower, upper = calibrated_band(
            surrogate.models[name],
            surrogate.target_standardizers[name],
            standardized,
            pieces[name],
            scores[name],
            scaling[name],
            alpha=store.band_alpha,
        )
        unit, display_scale = QOI_DISPLAY[name]
        scalars.append(
            ScalarReadout(
                name=name,
                label=QOI_LABELS[name],
                unit=unit,
                display_scale=float(display_scale),
                mean=float(moments[name][0][0]),
                sigma=float(scaling[name] * np.sqrt(moments[name][1][0])),
                lower=float(lower[0]),
                upper=float(upper[0]),
            )
        )

    return Prediction(
        inputs={name: float(values[name]) for name in FEATURE_ORDER},
        u_grid=curve.u_grid,
        force_mean=curve.force_mean[0],
        force_lower=curve.force_mean[0] - force_half,
        force_upper=curve.force_mean[0] + force_half,
        damage_mean=curve.damage_mean[0],
        damage_lower=curve.damage_mean[0] - damage_half,
        damage_upper=curve.damage_mean[0] + damage_half,
        scalars=tuple(scalars),
        validity=validity_verdict(store, values),
        band_alpha=store.band_alpha,
    )


def export_payload(store: LabStore, prediction: Prediction) -> dict[str, Any]:
    """The prediction as a JSON document that can be traced back to its artifacts.

    Build spec 15 asks the export button to write the prediction with its manifest hash. What
    it writes is the config hash, the commit the surrogate was fitted at, and for every stage
    the dashboard read, the stage directory and the digest of its manifest's own output
    records. A reader who has the repository can resolve every one of them; a reader who does
    not can at least tell whether two exported predictions came from the same pipeline.

    The exported curve is the full grid rather than a summary, because the point of the export
    is to be checkable against a rerun.
    """
    verdict = prediction.validity
    return {
        "kind": "ufem_lab_prediction",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": store.config_sha256,
        "git": store.git_state,
        "packages": store.package_versions,
        "inputs": prediction.inputs,
        "input_units": {name: INPUT_LABELS[name][1] for name in FEATURE_ORDER},
        "band_alpha": prediction.band_alpha,
        "band_construction": (
            "simultaneous sup norm band over the displacement grid (build spec 11.2), "
            "half width band_scale times the measured variance scaling factor times the "
            "propagated pointwise standard deviation; the scalar intervals are the deployed "
            "jackknife+ intervals of build spec 11.1"
        ),
        "validity": {
            "inside": verdict.inside,
            "completion_probability": verdict.completion_probability,
            "completion_threshold": verdict.threshold,
            "inside_design_box": verdict.inside_design_box,
            "outside_inputs": [record["input"] for record in verdict.outside_inputs],
            "censored_corners": [corner["input"] for corner in verdict.censored_corners],
            "reason": verdict.reason(),
        },
        "curves": {
            "u_mm": prediction.u_grid.tolist(),
            "force_N": prediction.force_mean.tolist(),
            "force_lower_N": prediction.force_lower.tolist(),
            "force_upper_N": prediction.force_upper.tolist(),
            "damage": prediction.damage_mean.tolist(),
            "damage_lower": prediction.damage_lower.tolist(),
            "damage_upper": prediction.damage_upper.tolist(),
        },
        "scalars": {
            readout.name: {
                "label": readout.label,
                "unit": readout.unit,
                "mean": readout.mean,
                "sigma": readout.sigma,
                "lower": readout.lower,
                "upper": readout.upper,
            }
            for readout in prediction.scalars
        },
        "manifests": {
            stage: {
                "directory": str(
                    (store.artifact_root / stage / store.config_sha256).relative_to(
                        store.repo_root
                    )
                ).replace("\\", "/"),
                "manifest": MANIFEST_NAME,
                "cache_key": manifest.get("extra", {}).get("cache_key"),
                "outputs": {
                    record["name"]: record["sha256"] for record in manifest.get("outputs", [])
                },
            }
            for stage, manifest in store.manifests.items()
        },
    }
