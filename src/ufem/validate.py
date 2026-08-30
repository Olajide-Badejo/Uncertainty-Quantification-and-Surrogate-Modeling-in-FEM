"""Stage ``validate``: the one cross validation harness, used by every model in the project.

Build spec 16.3 and binding law 3. There is exactly one harness, and the Gaussian process
reaches it through the same two protocols the baselines do, so the comparison cannot be
rigged by the surrogate getting a privilege the baselines did not. The predecessor project had
three different evaluation paths reporting three different numbers for the same model, and one
of them silently evaluated on its training set; that is the failure this file is built against.

Two harnesses, because the two kinds of target have honestly different costs.

**Scalars, leave one out over all 198 runs.** A scalar quantity of interest is measured
directly off a curve, so no reduction basis stands between the design and the target and
nothing about a leave one out fold needs refitting except the model. The baselines are refitted
from scratch on the other 197 rows for every one of the 198 folds, which is exact. The
Gaussian process uses the closed form leave one out of Dubrule 1983 and Rasmussen and Williams
section 5.4.2 at fixed hyperparameters, and the approximation is stated rather than buried:
the kernel hyperparameters and the constant mean come from a fit on all 198 points, so each
fold's model saw its own held out point through them. The grouped fold harness below refits
those hyperparameters per fold and reports both, which is the cross check build spec 11.1 asks
for. If the two disagree materially, that is a finding and the report says so.

**Curves, grouped ten fold, everything refitted inside the fold.** The registration reference,
the reduction bases, the feature standardization and the target standardization are all
recomputed on the training 90 percent of every fold. Leave one out at curve level would be the
honest ideal and it is out of budget by arithmetic rather than by preference: the SRVF
registration costs 12.8 seconds on this family, so 198 folds is 42 minutes for one model and
over three hours for the five. Ten folds is 10 registrations, about 2 minutes, and every fold
still refits every basis. The compromise is pre authorized, it is recorded in
docs/DESIGN_DECISIONS.md with this arithmetic, and the leak test of build spec 16.3 applies to
the fold harness exactly as it would to the leave one out one.

The gate of build spec 10.5: the surrogate must beat all four baselines out of sample on the
headline quantities of interest, or the failure is reported in the table and in the README
status. It is not tuned away and it is not omitted.

Units: every metric here is dimensionless (R2, relative L2) or carries the target's own unit
(RMSE). The curve error is a relative L2 norm on the physical load displacement curve in N
against displacement in mm, so it is comparable across models and across folds.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.baselines import (
    MeanCurveModel,
    NearestNeighborCurveModel,
    Regressor,
    build_baseline_regressors,
)
from ufem.config import Config, features
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET, displacement_grid
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.register import LANDMARKS_PARQUET, curve_matrix
from ufem.register import STAGE_NAME as REGISTER_STAGE
from ufem.surrogate import (
    CURVE_BLOCKS,
    LANDMARK_QOI,
    SCALAR_QOI,
    CurveBasis,
    GPSettings,
    SurrogateModel,
    configure_torch,
    fit_all,
    score_target_names,
)
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE

STAGE_NAME = "validate"

#: Output file names inside the stage directory.
BASELINES_JSON = "baselines.json"
BASELINES_TEX = "baselines_table.tex"
BASELINES_MD = "baselines_summary.md"
SCALAR_PREDICTIONS_PARQUET = "scalar_predictions.parquet"
CURVE_ERRORS_PARQUET = "curve_errors.parquet"

#: The surrogate's row label wherever a table lists models side by side.
GP_MODEL = "gaussian_process"

#: Display names, so the generated table reads as a table rather than as identifiers.
MODEL_LABELS: dict[str, str] = {
    GP_MODEL: "Gaussian process",
    "climatology": "Training mean",
    "linear": "Linear",
    "quadratic_chaos": "Quadratic chaos",
    "nearest_neighbour": "3 nearest neighbour",
}

QOI_LABELS: dict[str, str] = {
    "P_max_N": "Peak load",
    "u_peak_mm": "Displacement at peak",
    "k0_N_per_mm": "Initial stiffness",
    "E_abs_Nmm": "Absorbed energy",
    "P_residual_N": "Residual load",
    "softening_ratio": "Softening ratio",
    "u_damage_half_sat_mm": "Displacement at half damage",
    "damage_at_10mm": "Damage at 10 mm",
    "u_knee_mm": "Cracking knee displacement",
    "P_knee_N": "Cracking knee load",
    "arclength_total": "Normalized arc length",
}


class LeakDetected(ValueError):
    """A job appears in more than one place where it may appear exactly once."""


def make_folds(
    job_ids: list[str], n_folds: int, seed_sequence: np.random.SeedSequence
) -> list[np.ndarray]:
    """Grouped folds over unique job identifiers, as arrays of test row indices.

    The leak test of build spec 16.3 lives here rather than in a test file, because a harness
    that only refuses a duplicate when a test remembers to ask is a harness that will one day
    be called by something that does not ask. A repeated job identifier means the same
    simulation would sit in a training set and a test set at once, which is how the
    predecessor's augmented children ended up on both sides of its split, so it raises.

    The assignment is a seeded permutation dealt round robin, which keeps the folds within one
    row of the same size without any dependence on the order the jobs arrived in.
    """
    labels = [str(job) for job in job_ids]
    if len(set(labels)) != len(labels):
        seen: dict[str, int] = {}
        duplicates = []
        for label in labels:
            seen[label] = seen.get(label, 0) + 1
        duplicates = sorted(name for name, count in seen.items() if count > 1)
        raise LeakDetected(
            f"the fold harness was given {len(labels)} rows carrying only "
            f"{len(set(labels))} distinct job identifiers; {duplicates[:5]} appear more than "
            "once. The same simulation would land in a training set and a test set at the "
            "same time, which makes every out of sample number in this run a lie."
        )
    if not 2 <= n_folds <= len(labels):
        raise ValueError(
            f"n_folds must lie between 2 and the {len(labels)} available runs, got {n_folds}."
        )
    order = np.random.default_rng(seed_sequence).permutation(len(labels))
    folds = [np.sort(order[index::n_folds]) for index in range(n_folds)]
    covered = np.sort(np.concatenate(folds))
    if not np.array_equal(covered, np.arange(len(labels))):
        raise LeakDetected(
            "the folds do not partition the runs exactly once each. Every run must be a test "
            "row in exactly one fold, or the reported out of sample metric is an average over "
            "an unknown sample."
        )
    return folds


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination against the variance of the truth, dimensionless.

    The denominator is the total sum of squares of the held out truth. That makes a model
    which predicts the training mean score near zero and a model worse than that score
    negative, which is the point of reporting it.
    """
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError(
            "the held out truth has zero variance, so an R2 against it is undefined. A "
            "constant target is not a target."
        )
    return float(1.0 - np.sum((y - p) ** 2) / denominator)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean squared error in the target's own unit."""
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def relative_l2(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per curve relative L2 error, dimensionless, one value per row."""
    y = np.atleast_2d(np.asarray(truth, dtype=float))
    p = np.atleast_2d(np.asarray(prediction, dtype=float))
    if y.shape != p.shape:
        raise ValueError(f"relative_l2 needs matching shapes, got {y.shape} and {p.shape}.")
    norms = np.linalg.norm(y, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("a reference curve has zero norm, so a relative error is undefined.")
    return np.linalg.norm(y - p, axis=1) / norms


def error_summary(errors: np.ndarray) -> dict[str, float]:
    """The distribution of a per curve error, as the report quotes it."""
    array = np.asarray(errors, dtype=float)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "n": int(array.size),
    }


# ---------------------------------------------------------------------------
# The scalar leave one out harness
# ---------------------------------------------------------------------------


def baseline_leave_one_out(
    model_factory: Any, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Exact leave one out predictions for a baseline: n refits, no shortcuts.

    A hat matrix would give the linear and the chaos rows in closed form, and it would not
    give the neighbour row, so all four are done the same slow honest way. At n = 198 with ten
    basis terms this costs milliseconds, and having one code path for all four means the
    comparison cannot differ by which shortcut was available to whom.
    """
    design = np.atleast_2d(np.asarray(X, dtype=float))
    targets = np.asarray(Y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    n_rows = design.shape[0]
    out = np.empty_like(targets)
    for index in range(n_rows):
        keep = np.ones(n_rows, dtype=bool)
        keep[index] = False
        fitted = model_factory().fit(design[keep], targets[keep])
        out[index] = fitted.predict(design[index : index + 1])[0]
    return out


def gp_leave_one_out(
    surrogate: SurrogateModel, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Closed form leave one out mean and variance for one scalar target, in its own units."""
    mean, variance = surrogate.models[name].leave_one_out()
    standardizer = surrogate.target_standardizers[name]
    return (
        standardizer.inverse_mean(mean.reshape(-1, 1)).ravel(),
        standardizer.inverse_variance(variance.reshape(-1, 1)).ravel(),
    )


def scalar_harness(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    surrogate: SurrogateModel,
    n_neighbors: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Every scalar target, every model, train and test metrics side by side."""
    names = list(targets)
    Y = np.column_stack([targets[name] for name in names])
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    gp_test = np.empty_like(Y)
    gp_train = np.empty_like(Y)
    for column, name in enumerate(names):
        mean, _variance = gp_leave_one_out(surrogate, name)
        gp_test[:, column] = mean
        gp_train[:, column] = surrogate.predict_target(name, X)[0]
    predictions[GP_MODEL] = gp_test
    for column, name in enumerate(names):
        rows.append(
            {
                "harness": "leave_one_out",
                "target": name,
                "model": GP_MODEL,
                "r2_test": r2_score(Y[:, column], gp_test[:, column]),
                "rmse_test": rmse(Y[:, column], gp_test[:, column]),
                "r2_train": r2_score(Y[:, column], gp_train[:, column]),
                "rmse_train": rmse(Y[:, column], gp_train[:, column]),
                "n": int(Y.shape[0]),
            }
        )

    for template in build_baseline_regressors(n_neighbors):
        factory = _factory_for(template, n_neighbors)
        test = baseline_leave_one_out(factory, X, Y)
        train = factory().fit(X, Y).predict(X)
        predictions[template.name] = test
        for column, name in enumerate(names):
            rows.append(
                {
                    "harness": "leave_one_out",
                    "target": name,
                    "model": template.name,
                    "r2_test": r2_score(Y[:, column], test[:, column]),
                    "rmse_test": rmse(Y[:, column], test[:, column]),
                    "r2_train": r2_score(Y[:, column], train[:, column]),
                    "rmse_train": rmse(Y[:, column], train[:, column]),
                    "n": int(Y.shape[0]),
                }
            )

    frame_records = []
    for model, matrix in predictions.items():
        for column, name in enumerate(names):
            frame_records.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "target": name,
                        "truth": Y[:, column],
                        "prediction": matrix[:, column],
                    }
                )
            )
    return rows, pd.concat(frame_records, ignore_index=True)


def _factory_for(template: Regressor, n_neighbors: int) -> Any:
    """A zero argument constructor for one baseline, so the harness can refit it per fold."""
    kind = type(template)
    if template.name == "nearest_neighbour":
        return lambda: kind(n_neighbors=n_neighbors)
    return kind


# ---------------------------------------------------------------------------
# The grouped fold harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Everything one fold measured, before the summary flattens it."""

    fold: int
    n_train: int
    n_test: int
    component_counts: dict[str, int]
    force_test: dict[str, np.ndarray]
    force_train: dict[str, np.ndarray]
    damage_test: dict[str, np.ndarray]
    damage_train: dict[str, np.ndarray]
    scalar_rows: list[dict[str, Any]]


def _block_scores(basis: CurveBasis) -> tuple[np.ndarray, dict[str, slice]]:
    """The fold's training scores as one matrix, with each block's column span."""
    spans: dict[str, slice] = {}
    columns: list[np.ndarray] = []
    offset = 0
    for block in CURVE_BLOCKS:
        scores = basis.scores[block]
        spans[block] = slice(offset, offset + scores.shape[1])
        offset += scores.shape[1]
        columns.append(scores)
    return np.column_stack(columns) if columns else np.zeros((0, 0)), spans


def _reconstruct(
    basis: CurveBasis, matrix: np.ndarray, spans: dict[str, slice]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a predicted score matrix back into force and damage curves."""
    blocks = {block: matrix[:, spans[block]] for block in CURVE_BLOCKS}
    force = basis.reconstruct_force(
        blocks["amplitude"], blocks["phase"], blocks["displacement"]
    )
    damage = basis.reconstruct_damage(blocks["damage"])
    return force, damage


def _gp_score_predictions(
    basis: CurveBasis,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
    X_train: np.ndarray,
    X_query: list[np.ndarray],
) -> list[np.ndarray]:
    """Fit one GP per fold score and predict the requested designs, in score units."""
    names = score_target_names(basis)
    targets: dict[str, np.ndarray] = {}
    for block in CURVE_BLOCKS:
        for index, name in enumerate(names[block]):
            targets[name] = basis.scores[block][:, index]
    models, feature_standardizer, target_standardizers, _log = fit_all(
        X_train, targets, settings, seed_sequence
    )
    ordered = [name for block in CURVE_BLOCKS for name in names[block]]
    out = []
    for design in X_query:
        standardized = feature_standardizer.transform(design)
        matrix = np.empty((standardized.shape[0], len(ordered)))
        for column, name in enumerate(ordered):
            mean, _variance = models[name].predict(standardized)
            matrix[:, column] = (
                target_standardizers[name].inverse_mean(mean.reshape(-1, 1)).ravel()
            )
        out.append(matrix)
    return out


def _fold_scalars(
    X_train: np.ndarray,
    X_test: np.ndarray,
    targets: dict[str, np.ndarray],
    train_index: np.ndarray,
    test_index: np.ndarray,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
    n_neighbors: int,
) -> list[dict[str, Any]]:
    """Refit every scalar model inside one fold, hyperparameters and all.

    This is the honest cross check of build spec 11.1 against the closed form leave one out of
    the scalar harness, which reuses hyperparameters fitted on everything. Here nothing is
    reused: the Gaussian process gets its eight restarts on the training rows of this fold and
    has never seen the held out ones in any capacity.
    """
    names = list(targets)
    fold_targets = {name: targets[name][train_index] for name in names}
    models, feature_standardizer, target_standardizers, _log = fit_all(
        X_train, fold_targets, settings, seed_sequence
    )
    standardized = feature_standardizer.transform(X_test)
    rows = []
    for name in names:
        mean, _variance = models[name].predict(standardized)
        rows.append(
            {
                "model": GP_MODEL,
                "target": name,
                "index": test_index,
                "prediction": target_standardizers[name]
                .inverse_mean(mean.reshape(-1, 1))
                .ravel(),
            }
        )
    Y_train = np.column_stack([targets[name][train_index] for name in names])
    for template in build_baseline_regressors(n_neighbors):
        fitted = _factory_for(template, n_neighbors)().fit(X_train, Y_train)
        predicted = fitted.predict(X_test)
        for column, name in enumerate(names):
            rows.append(
                {
                    "model": template.name,
                    "target": name,
                    "index": test_index,
                    "prediction": predicted[:, column],
                }
            )
    return rows


def run_fold(
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    X: np.ndarray,
    force: np.ndarray,
    damage: np.ndarray,
    scalar_targets: dict[str, np.ndarray],
    config: Config,
    u_grid: np.ndarray,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> FoldResult:
    """One grouped fold: refit the registration, the bases, and every model on the training side."""
    n_neighbors = int(config.pipeline.validation.n_neighbors)
    basis = CurveBasis.fit(u_grid, force[train_index], damage[train_index], config)
    matrix, spans = _block_scores(basis)
    X_train, X_test = X[train_index], X[test_index]

    children = seed_sequence.spawn(2)
    gp_test, gp_train = _gp_score_predictions(
        basis, settings, children[0], X_train, [X_test, X_train]
    )
    force_test: dict[str, np.ndarray] = {}
    force_train: dict[str, np.ndarray] = {}
    damage_test: dict[str, np.ndarray] = {}
    damage_train: dict[str, np.ndarray] = {}

    predicted_force, predicted_damage = _reconstruct(basis, gp_test, spans)
    force_test[GP_MODEL] = relative_l2(force[test_index], predicted_force)
    damage_test[GP_MODEL] = relative_l2(damage[test_index], predicted_damage)
    predicted_force, predicted_damage = _reconstruct(basis, gp_train, spans)
    force_train[GP_MODEL] = relative_l2(force[train_index], predicted_force)
    damage_train[GP_MODEL] = relative_l2(damage[train_index], predicted_damage)

    for template in build_baseline_regressors(n_neighbors):
        if template.name in ("climatology", "nearest_neighbour"):
            continue
        fitted = _factory_for(template, n_neighbors)().fit(X_train, matrix)
        for label, design, truth_force, truth_damage, into_force, into_damage in (
            ("test", X_test, force[test_index], damage[test_index], force_test, damage_test),
            (
                "train",
                X_train,
                force[train_index],
                damage[train_index],
                force_train,
                damage_train,
            ),
        ):
            del label
            curves, damages = _reconstruct(basis, fitted.predict(design), spans)
            into_force[template.name] = relative_l2(truth_force, curves)
            into_damage[template.name] = relative_l2(truth_damage, damages)

    for curve_model in (
        MeanCurveModel.fit(X_train, force[train_index], damage[train_index]),
        NearestNeighborCurveModel.fit(
            X_train, force[train_index], damage[train_index], n_neighbors
        ),
    ):
        curves, damages = curve_model.predict_curves(X_test)
        force_test[curve_model.name] = relative_l2(force[test_index], curves)
        damage_test[curve_model.name] = relative_l2(damage[test_index], damages)
        curves, damages = curve_model.predict_curves(X_train)
        force_train[curve_model.name] = relative_l2(force[train_index], curves)
        damage_train[curve_model.name] = relative_l2(damage[train_index], damages)

    scalar_rows = _fold_scalars(
        X_train,
        X_test,
        scalar_targets,
        train_index,
        test_index,
        settings,
        children[1],
        n_neighbors,
    )
    return FoldResult(
        fold=fold,
        n_train=int(train_index.size),
        n_test=int(test_index.size),
        component_counts=dict(basis.block_counts),
        force_test=force_test,
        force_train=force_train,
        damage_test=damage_test,
        damage_train=damage_train,
        scalar_rows=scalar_rows,
    )


def summarize_curves(results: list[FoldResult]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Pool the per fold curve errors into one row per model and signal."""
    records = []
    rows = []
    for signal, test_key, train_key in (
        ("force", "force_test", "force_train"),
        ("damage", "damage_test", "damage_train"),
    ):
        models = sorted({name for result in results for name in getattr(result, test_key)})
        for model in models:
            test = np.concatenate(
                [getattr(result, test_key)[model] for result in results]
            )
            train = np.concatenate(
                [getattr(result, train_key)[model] for result in results]
            )
            rows.append(
                {
                    "harness": "grouped_fold",
                    "signal": signal,
                    "model": model,
                    "test": error_summary(test),
                    "train": error_summary(train),
                }
            )
            for result in results:
                for value in getattr(result, test_key)[model]:
                    records.append(
                        {
                            "fold": result.fold,
                            "signal": signal,
                            "model": model,
                            "relative_l2": float(value),
                        }
                    )
    return rows, pd.DataFrame.from_records(records)


def summarize_fold_scalars(
    results: list[FoldResult], targets: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Pool the per fold scalar predictions into out of sample metrics per target and model."""
    names = list(targets)
    collected: dict[tuple[str, str], np.ndarray] = {}
    for result in results:
        for row in result.scalar_rows:
            key = (str(row["model"]), str(row["target"]))
            if key not in collected:
                collected[key] = np.full(len(targets[names[0]]), np.nan)
            collected[key][row["index"]] = row["prediction"]
    rows = []
    for (model, target), prediction in sorted(collected.items()):
        if np.any(~np.isfinite(prediction)):
            raise ValueError(
                f"the grouped folds left {int(np.sum(~np.isfinite(prediction)))} runs without "
                f"a prediction for {model} on {target}. Every run is a test row in exactly "
                "one fold, so a gap means the folds did not partition the data."
            )
        truth = targets[target]
        rows.append(
            {
                "harness": "grouped_fold",
                "target": target,
                "model": model,
                "r2_test": r2_score(truth, prediction),
                "rmse_test": rmse(truth, prediction),
                "n": int(truth.size),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The gate and the generated documents
# ---------------------------------------------------------------------------


def evaluate_gate(
    scalar_rows: list[dict[str, Any]], headline: list[str]
) -> dict[str, Any]:
    """Build spec 10.5: does the surrogate beat all four baselines on the headline QoIs?

    Decided on the leave one out harness, because that is the one every model ran through
    identically. The verdict is per quantity as well as overall, so a mixed result is reported
    as a mixed result rather than collapsing to a single pass or fail that hides which half is
    which.
    """
    lookup = {
        (str(row["target"]), str(row["model"])): row
        for row in scalar_rows
        if row["harness"] == "leave_one_out"
    }
    per_target = {}
    for target in headline:
        gp = lookup.get((target, GP_MODEL))
        if gp is None:
            raise KeyError(
                f"the headline quantity {target!r} has no Gaussian process row in the leave "
                "one out harness, so the gate cannot be decided on it."
            )
        beaten = {}
        for (name, model), row in lookup.items():
            if name != target or model == GP_MODEL:
                continue
            beaten[model] = bool(gp["r2_test"] > row["r2_test"])
        losses = sorted(model for model, won in beaten.items() if not won)
        per_target[target] = {
            "gp_r2_test": float(gp["r2_test"]),
            "beats": beaten,
            "beats_all": not losses,
            "lost_to": losses,
            "best_baseline": max(
                (
                    (model, float(lookup[(target, model)]["r2_test"]))
                    for model in beaten
                ),
                key=lambda item: item[1],
            ),
        }
    failing = sorted(name for name, record in per_target.items() if not record["beats_all"])
    return {
        "criterion": (
            "out of sample R2 in the leave one out harness, surrogate against every baseline, "
            "on the headline quantities of build spec 10.5"
        ),
        "headline_qoi": list(headline),
        "per_target": per_target,
        "passed": not failing,
        "failing_targets": failing,
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_tex_table(
    scalar_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]], headline: list[str]
) -> str:
    """The report fragment: out of sample R2 per model on the headline QoIs, plus curve error.

    A tabular body only, so ``main.tex`` supplies the caption and the label and this stays a
    pure data fragment with no formatting opinions of its own.
    """
    lookup = {
        (str(row["target"]), str(row["model"])): row
        for row in scalar_rows
        if row["harness"] == "leave_one_out"
    }
    curves = {
        str(row["model"]): row for row in curve_rows if row["signal"] == "force"
    }
    models = [GP_MODEL, "climatology", "linear", "quadratic_chaos", "nearest_neighbour"]
    lines = [
        "% Generated by ufem.validate. Do not edit: regenerate with `ufem run validate`.",
        "\\begin{tabular}{l" + "r" * (len(headline) + 1) + "}",
        "\\toprule",
        "Model & "
        + " & ".join(QOI_LABELS.get(name, name) for name in headline)
        + " & Curve $L_2$ \\\\",
        "\\midrule",
    ]
    for model in models:
        cells = []
        for target in headline:
            row = lookup.get((target, model))
            cells.append("{--}" if row is None else _fmt(float(row["r2_test"])))
        curve = curves.get(model)
        cells.append(
            "{--}"
            if curve is None
            else _fmt(100.0 * float(curve["test"]["p50"]), 2) + "\\,\\%"
        )
        lines.append(f"{MODEL_LABELS.get(model, model)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines).replace("{--}", "\\textendash")


def build_markdown_summary(
    scalar_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """The human readable summary that ships beside the table."""
    lookup = {
        (str(row["target"]), str(row["model"])): row
        for row in scalar_rows
        if row["harness"] == "leave_one_out"
    }
    fold_lookup = {
        (str(row["target"]), str(row["model"])): row
        for row in scalar_rows
        if row["harness"] == "grouped_fold"
    }
    models = [GP_MODEL, "climatology", "linear", "quadratic_chaos", "nearest_neighbour"]
    targets = sorted({name for name, _model in lookup})
    out: list[str] = []
    add = out.append
    add("# Baselines and out of sample validation")
    add("")
    add(
        "Generated by the `validate` stage from the artifact store. Every number is measured "
        "in one harness that the surrogate and the four baselines of build spec 10.5 go "
        "through identically. Train metrics sit next to their test counterparts throughout, "
        "because a train number on its own says nothing (binding law 3)."
    )
    add("")
    add(
        f"Leave one out over {context['n_runs']} runs for the scalars; "
        f"{context['n_folds']} grouped folds with the registration, both principal component "
        "bases and every standardization refitted inside each fold for the curves."
    )
    add("")
    add("## Out of sample R2, leave one out")
    add("")
    add("| Quantity | " + " | ".join(MODEL_LABELS[m] for m in models) + " |")
    add("|---" * (len(models) + 1) + "|")
    for target in targets:
        cells = []
        for model in models:
            row = lookup.get((target, model))
            cells.append("not run" if row is None else _fmt(float(row["r2_test"])))
        add(f"| {QOI_LABELS.get(target, target)} | " + " | ".join(cells) + " |")
    add("")
    add("## The surrogate, train against test")
    add("")
    add("| Quantity | Train R2 | Test R2 (leave one out) | Test R2 (grouped fold refit) |")
    add("|---|---|---|---|")
    for target in targets:
        row = lookup[(target, GP_MODEL)]
        fold = fold_lookup.get((target, GP_MODEL))
        add(
            f"| {QOI_LABELS.get(target, target)} | {_fmt(float(row['r2_train']))} | "
            f"{_fmt(float(row['r2_test']))} | "
            f"{'not run' if fold is None else _fmt(float(fold['r2_test']))} |"
        )
    add("")
    add(
        "The third column refits the kernel hyperparameters inside every fold, where the "
        "second reuses the hyperparameters fitted on all runs and applies the closed form "
        "leave one out of Dubrule at those fixed values. The gap between them is the size of "
        "that approximation, measured rather than assumed."
    )
    add("")
    add("## Curve error, grouped folds")
    add("")
    add("| Signal | Model | Median relative L2 | 90th percentile | Train median |")
    add("|---|---|---|---|---|")
    for row in curve_rows:
        add(
            f"| {row['signal']} | {MODEL_LABELS.get(str(row['model']), row['model'])} | "
            f"{_fmt(100.0 * float(row['test']['p50']), 2)} % | "
            f"{_fmt(100.0 * float(row['test']['p90']), 2)} % | "
            f"{_fmt(100.0 * float(row['train']['p50']), 2)} % |"
        )
    add("")
    add("## The gate of build spec 10.5")
    add("")
    add(f"**{'PASSED' if gate['passed'] else 'FAILED'}.** {gate['criterion']}.")
    add("")
    for target, record in gate["per_target"].items():
        best_model, best_value = record["best_baseline"]
        verdict = (
            "beats every baseline"
            if record["beats_all"]
            else "loses to " + ", ".join(MODEL_LABELS.get(m, m) for m in record["lost_to"])
        )
        add(
            f"- **{QOI_LABELS.get(target, target)}**: surrogate R2 "
            f"{_fmt(record['gp_r2_test'])}, best baseline "
            f"{MODEL_LABELS.get(best_model, best_model)} at {_fmt(best_value)}. It {verdict}."
        )
    add("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, Path, dict[str, str]]:
    artifact_root = root / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, config_sha256)
    surrogate_dir = stage_dir(artifact_root, SURROGATE_STAGE, config_sha256)
    hashes: dict[str, str] = {}
    for directory, name, stage in (
        (grid_dir, RF2_GRID_PARQUET, GRID_STAGE),
        (grid_dir, DAMAGE_GRID_PARQUET, GRID_STAGE),
        (grid_dir, QOI_PARQUET, GRID_STAGE),
        (register_dir, LANDMARKS_PARQUET, REGISTER_STAGE),
        (surrogate_dir, "surrogate.json", SURROGATE_STAGE),
        (surrogate_dir, "gp_state.npy", SURROGATE_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the validate stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first."
            )
        hashes[name] = sha256_file(path)
    return grid_dir, register_dir, surrogate_dir, hashes


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the validate stage and return its artifact directory."""
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    grid_dir, register_dir, _surrogate_dir, input_hashes = _load_inputs(
        root, config, config_sha256
    )
    artifact_root = root / config.pipeline.paths.artifact_root

    force_frame = pd.read_parquet(grid_dir / RF2_GRID_PARQUET)
    damage_frame = pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET)
    qoi = pd.read_parquet(grid_dir / QOI_PARQUET)
    landmarks = pd.read_parquet(register_dir / LANDMARKS_PARQUET)
    jobs, force = curve_matrix(force_frame)
    damage_jobs, damage = curve_matrix(damage_frame)
    if jobs != damage_jobs or list(qoi["job"].astype(str)) != jobs:
        raise AssertionError(
            "the gridded curves, the damage curves and the QoI table carry different job "
            "orders, so no row wise comparison between them is valid."
        )

    X = features(qoi)
    u_grid = displacement_grid(config)
    scalar_targets: dict[str, np.ndarray] = {
        name: qoi[name].to_numpy(dtype=float) for name in SCALAR_QOI
    }
    for name in LANDMARK_QOI:
        scalar_targets[name] = landmarks[name].to_numpy(dtype=float)

    surrogate = SurrogateModel.load(artifact_root, config_sha256)
    settings = GPSettings.from_config(config)
    validation = config.pipeline.validation
    n_neighbors = int(validation.n_neighbors)

    scalar_started = _time.perf_counter()
    scalar_rows, scalar_predictions = scalar_harness(
        X, scalar_targets, surrogate, n_neighbors
    )
    scalar_seconds = _time.perf_counter() - scalar_started

    root_sequence = np.random.SeedSequence(config.pipeline.seed_entropy)
    fold_seed, model_seed = root_sequence.spawn(2)
    folds = make_folds(jobs, int(validation.n_folds), fold_seed)
    fold_children = model_seed.spawn(len(folds))

    fold_started = _time.perf_counter()
    results: list[FoldResult] = []
    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(len(jobs)), test_index, assume_unique=False)
        results.append(
            run_fold(
                index,
                train_index,
                test_index,
                X,
                force,
                damage,
                scalar_targets,
                config,
                u_grid,
                settings,
                fold_children[index],
            )
        )
        print(
            f"[validate] fold {index + 1}/{len(folds)}: {train_index.size} train, "
            f"{test_index.size} test, median curve error "
            f"{np.median(results[-1].force_test[GP_MODEL]) * 100:.2f} percent"
        )
    fold_seconds = _time.perf_counter() - fold_started

    curve_rows, curve_frame = summarize_curves(results)
    scalar_rows = scalar_rows + summarize_fold_scalars(results, scalar_targets)
    gate = evaluate_gate(scalar_rows, list(validation.headline_qoi))

    directory = stage_dir(artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    context = {
        "n_runs": len(jobs),
        "n_folds": len(folds),
        "n_neighbors": n_neighbors,
        "fold_component_counts": [result.component_counts for result in results],
    }
    payload = {
        "config_sha256": config_sha256,
        "context": context,
        "scalar": scalar_rows,
        "curve": curve_rows,
        "gate": gate,
        "loo_approximation": (
            "closed form leave one out (Dubrule 1983; Rasmussen and Williams 5.4.2) at "
            "hyperparameters fitted on all runs, cross checked by the grouped fold refit"
        ),
    }
    outputs = []
    for name, text in (
        (BASELINES_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (
            BASELINES_TEX,
            build_tex_table(scalar_rows, curve_rows, list(validation.headline_qoi)),
        ),
        (
            BASELINES_MD,
            build_markdown_summary(scalar_rows, curve_rows, gate, context),
        ),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8", newline="\n")
        outputs.append(path)
    for frame, name in (
        (scalar_predictions, SCALAR_PREDICTIONS_PARQUET),
        (curve_frame, CURVE_ERRORS_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    force_summary = {
        str(row["model"]): row["test"]["p50"] for row in curve_rows if row["signal"] == "force"
    }
    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "scalar_harness_wall_time_s": scalar_seconds,
        "fold_harness_wall_time_s": fold_seconds,
        "n_runs": len(jobs),
        "n_folds": len(folds),
        "gate": gate,
        "curve_median_relative_l2_force": force_summary,
        "fold_component_counts": context["fold_component_counts"],
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=sorted(outputs),
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    verdict = "PASSED" if gate["passed"] else f"FAILED on {', '.join(gate['failing_targets'])}"
    print(
        f"[validate] {len(jobs)} runs, leave one out for {len(scalar_targets)} scalars and "
        f"{len(folds)} grouped folds for the curves; surrogate median curve error "
        f"{force_summary[GP_MODEL] * 100:.2f} percent against "
        f"{force_summary['climatology'] * 100:.2f} percent for the training mean; "
        f"baseline gate {verdict}"
    )
    return directory

