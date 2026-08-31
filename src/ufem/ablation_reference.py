"""The production pipeline's out of fold curve predictions, so the ablations have a rival.

Build spec 10.6. Ablations 2 through 4 all ask the same question in different words: is the
registered principal component representation plus one Gaussian process per score worth its
place against some other way of turning three inputs into a curve? A question like that is only
answerable if both sides are measured in the same harness on the same folds, and three of the
metrics build spec 10.6.3 names, pointwise root mean square error, negative log predictive
density and coverage, need a *predictive distribution* from the production side rather than the
per curve error summary the ``validate`` stage stores.

So this module recomputes the production side once: the same ten grouped folds from the same
seed the validation stage uses, the registration and every basis refitted inside each fold, one
Gaussian process per retained score, and the reconstruction of build spec 10.4 with its
propagated pointwise variance. The result is cached in the artifact store under its own stage
name with a manifest, so the three ablation scripts pay for it once between them.

Two things this module deliberately does not do. It does not reimplement the prediction path:
it assembles a :class:`ufem.surrogate.SurrogateModel` per fold out of the same pieces the
surrogate stage assembles and calls ``predict_curve`` on it, because a second copy of the
variance propagation is exactly how two numbers for one band come about. And it does not
replace the ``validate`` stage: it reproduces that stage's per fold curve errors and asserts
the agreement, so a drift between this reference and the shipped numbers is a stop condition
rather than a discrepancy nobody notices.

Nothing here is on the production path. The ablations are measurements about the production
path, and this is the yardstick they are measured against.

Units: force in N on the common displacement grid in mm, damage dimensionless. Every variance
is in the square of its signal's unit.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.config import Config, features
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET, displacement_grid
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import sha256_file, stage_dir, verify_manifest, write_manifest
from ufem.register import curve_matrix
from ufem.surrogate import (
    CURVE_BLOCKS,
    CurveBasis,
    GPSettings,
    SurrogateModel,
    configure_torch,
    fit_all,
    score_target_names,
)
from ufem.validate import BASELINES_JSON, GP_MODEL, make_folds, relative_l2
from ufem.validate import STAGE_NAME as VALIDATE_STAGE

STAGE_NAME = "ablation_reference"

#: Output file names inside the stage directory.
FORCE_MEAN_NPY = "force_mean.npy"
FORCE_VARIANCE_NPY = "force_variance.npy"
DAMAGE_MEAN_NPY = "damage_mean.npy"
DAMAGE_VARIANCE_NPY = "damage_variance.npy"
FOLD_OF_RUN_NPY = "fold_of_run.npy"
REFERENCE_JSON = "reference.json"

#: How far the reproduced per fold curve errors may sit from the ones the ``validate`` stage
#: committed before the disagreement is a defect rather than float64 round off. Both sides run
#: the same code on the same folds from the same spawned seeds, so the honest expectation is
#: exact agreement; this is an allowance for summation order, not a tolerance to hide behind.
VALIDATE_AGREEMENT_TOLERANCE = 1.0e-9


class ReferenceUnavailable(RuntimeError):
    """The artifacts this reference is computed from are not on disk."""


@dataclass(frozen=True)
class CurveData:
    """The gridded campaign, in the one row order every ablation shares."""

    jobs: list[str]
    u_grid: np.ndarray
    X: np.ndarray
    force: np.ndarray
    damage: np.ndarray
    qoi: pd.DataFrame

    @property
    def n_runs(self) -> int:
        return len(self.jobs)


@dataclass(frozen=True)
class FoldReference:
    """The production pipeline's out of fold predictive distribution, one row per run."""

    jobs: list[str]
    u_grid: np.ndarray
    fold_of_run: np.ndarray
    force_mean: np.ndarray
    force_variance: np.ndarray
    damage_mean: np.ndarray
    damage_variance: np.ndarray
    metadata: dict[str, Any]


def load_curve_data(repo_root: Path | str, config: Config, config_sha256: str) -> CurveData:
    """Read the gridded curves, the damage family and the QoI table for one config hash.

    Raises rather than falling back when the grid stage has not run: an ablation with no data
    is a stop condition, not an empty result (ground rule 8).
    """
    artifact_root = Path(repo_root) / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    for name in (RF2_GRID_PARQUET, DAMAGE_GRID_PARQUET, QOI_PARQUET):
        if not (grid_dir / name).is_file():
            raise ReferenceUnavailable(
                f"the ablations need {grid_dir / name}, which does not exist. Run "
                f"`ufem run {GRID_STAGE}` first: an ablation measures artifacts, it does not "
                "recompute the pipeline that made them."
            )
    jobs, force = curve_matrix(pd.read_parquet(grid_dir / RF2_GRID_PARQUET))
    damage_jobs, damage = curve_matrix(pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET))
    qoi = pd.read_parquet(grid_dir / QOI_PARQUET)
    if jobs != damage_jobs or list(qoi["job"].astype(str)) != jobs:
        raise AssertionError(
            "the gridded curves, the damage curves and the QoI table carry different job "
            "orders, so no row wise comparison between them is valid."
        )
    return CurveData(
        jobs=jobs,
        u_grid=displacement_grid(config),
        X=features(qoi),
        force=force,
        damage=damage,
        qoi=qoi,
    )


def production_folds(jobs: list[str], config: Config) -> list[np.ndarray]:
    """The ten grouped folds the ``validate`` stage used, rebuilt from the same seed.

    The spawn order is copied from ``ufem.validate.run`` deliberately rather than shared
    through a helper, because the thing that has to be true is that both derive the same folds
    from the same entropy, and that is asserted by the agreement check in
    :func:`compute_reference` rather than assumed by a shared call.
    """
    root_sequence = np.random.SeedSequence(config.pipeline.seed_entropy)
    fold_seed, _model_seed = root_sequence.spawn(2)
    return make_folds(jobs, int(config.pipeline.validation.n_folds), fold_seed)


def production_fold_seeds(config: Config, n_folds: int) -> list[np.random.SeedSequence]:
    """The per fold seed each fold's score processes were fitted from, in fold order."""
    root_sequence = np.random.SeedSequence(config.pipeline.seed_entropy)
    _fold_seed, model_seed = root_sequence.spawn(2)
    fold_children = model_seed.spawn(n_folds)
    return [child.spawn(2)[0] for child in fold_children]


def fold_surrogate(
    basis: CurveBasis,
    X_train: np.ndarray,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> SurrogateModel:
    """One fold's production surrogate: a GP per retained score over the fold's own basis.

    Assembled as a real :class:`SurrogateModel` so the prediction and the variance propagation
    are the shipped ones rather than a copy. It carries no scalar quantity of interest, because
    the reference exists to compare curves; the scalar comparison reads the ``validate``
    stage's own numbers, which were measured on these same folds.
    """
    names = score_target_names(basis)
    targets: dict[str, np.ndarray] = {}
    for block in CURVE_BLOCKS:
        for index, name in enumerate(names[block]):
            targets[name] = basis.scores[block][:, index]
    models, feature_standardizer, target_standardizers, _log = fit_all(
        X_train, targets, settings, seed_sequence
    )
    return SurrogateModel(
        feature_standardizer=feature_standardizer,
        target_standardizers=target_standardizers,
        models=models,
        basis=basis,
        score_targets=names,
        scalar_targets=[],
        settings=settings,
        metadata={"source": STAGE_NAME},
    )


def _validate_medians(repo_root: Path, config: Config, config_sha256: str) -> dict[str, float]:
    """The per signal median curve error the ``validate`` stage committed for the surrogate."""
    artifact_root = Path(repo_root) / config.pipeline.paths.artifact_root
    path = stage_dir(artifact_root, VALIDATE_STAGE, config_sha256) / BASELINES_JSON
    if not path.is_file():
        raise ReferenceUnavailable(
            f"the reference cross check needs {path}, which does not exist. Run "
            f"`ufem run {VALIDATE_STAGE}` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["signal"]): float(row["test"]["p50"])
        for row in payload["curve"]
        if str(row["model"]) == GP_MODEL
    }


def compute_reference(
    repo_root: Path | str, config: Config, config_sha256: str
) -> FoldReference:
    """Refit the production pipeline in every fold and keep its out of fold predictions.

    Expensive by construction: ten elastic registrations and one Gaussian process per retained
    score per fold. The wall time is recorded in the manifest, because build spec 10.6 asks the
    ablations to report what they cost.
    """
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    data = load_curve_data(root, config, config_sha256)
    settings = GPSettings.from_config(config)
    folds = production_folds(data.jobs, config)
    seeds = production_fold_seeds(config, len(folds))

    n_runs, n_grid = data.force.shape
    force_mean = np.full((n_runs, n_grid), np.nan)
    force_variance = np.full((n_runs, n_grid), np.nan)
    damage_mean = np.full((n_runs, n_grid), np.nan)
    damage_variance = np.full((n_runs, n_grid), np.nan)
    fold_of_run = np.full(n_runs, -1, dtype=int)
    fold_errors: list[dict[str, Any]] = []

    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(n_runs), test_index, assume_unique=False)
        basis = CurveBasis.fit(
            data.u_grid, data.force[train_index], data.damage[train_index], config
        )
        model = fold_surrogate(basis, data.X[train_index], settings, seeds[index])
        prediction = model.predict_curve(data.X[test_index])
        force_mean[test_index] = prediction.force_mean
        force_variance[test_index] = prediction.force_variance
        damage_mean[test_index] = prediction.damage_mean
        damage_variance[test_index] = prediction.damage_variance
        fold_of_run[test_index] = index
        fold_errors.append(
            {
                "fold": index,
                "force": [
                    float(value)
                    for value in relative_l2(data.force[test_index], prediction.force_mean)
                ],
                "damage": [
                    float(value)
                    for value in relative_l2(data.damage[test_index], prediction.damage_mean)
                ],
            }
        )
        print(
            f"[{STAGE_NAME}] fold {index + 1}/{len(folds)}: median curve error "
            f"{np.median(relative_l2(data.force[test_index], prediction.force_mean)) * 100:.2f}"
            " percent"
        )

    if np.any(fold_of_run < 0) or np.any(~np.isfinite(force_mean)):
        raise AssertionError(
            "the folds left runs without a prediction, so the reference does not cover the "
            "campaign it claims to."
        )

    committed = _validate_medians(root, config, config_sha256)
    reproduced = {
        "force": float(np.median(np.concatenate([np.asarray(r["force"]) for r in fold_errors]))),
        "damage": float(
            np.median(np.concatenate([np.asarray(r["damage"]) for r in fold_errors]))
        ),
    }
    deviations = {
        signal: abs(reproduced[signal] - committed[signal]) for signal in reproduced
    }
    if max(deviations.values()) > VALIDATE_AGREEMENT_TOLERANCE:
        raise AssertionError(
            f"this reference reproduces median curve errors {reproduced} where the validate "
            f"stage committed {committed}, a deviation of {deviations}. The ablations compare "
            "against the shipped pipeline, so a reference that is not the shipped pipeline is "
            "a stop condition rather than a difference to note."
        )

    metadata = {
        "config_sha256": config_sha256,
        "n_runs": n_runs,
        "n_folds": len(folds),
        "fold_sizes": [int(fold.size) for fold in folds],
        "median_relative_l2": reproduced,
        "validate_median_relative_l2": committed,
        "validate_agreement_deviation": deviations,
        "wall_time_s": _time.perf_counter() - started,
    }
    return FoldReference(
        jobs=list(data.jobs),
        u_grid=data.u_grid,
        fold_of_run=fold_of_run,
        force_mean=force_mean,
        force_variance=force_variance,
        damage_mean=damage_mean,
        damage_variance=damage_variance,
        metadata=metadata,
    )


def _reference_directory(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    return stage_dir(
        Path(repo_root) / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256
    )


def _write_reference(
    directory: Path, reference: FoldReference, config: Config, config_sha256: str
) -> None:
    """Persist the reference and its manifest, arrays as plain ``.npy`` for byte stability."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        FORCE_MEAN_NPY: reference.force_mean,
        FORCE_VARIANCE_NPY: reference.force_variance,
        DAMAGE_MEAN_NPY: reference.damage_mean,
        DAMAGE_VARIANCE_NPY: reference.damage_variance,
        FOLD_OF_RUN_NPY: reference.fold_of_run,
    }
    outputs = []
    for name, array in arrays.items():
        path = directory / name
        np.save(path, np.ascontiguousarray(array))
        outputs.append(path)
    json_path = directory / REFERENCE_JSON
    json_path.write_text(
        json.dumps(
            {"jobs": reference.jobs, "u_grid_mm": [float(v) for v in reference.u_grid],
             **reference.metadata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs.append(json_path)
    grid_dir = stage_dir(
        directory.parents[1], GRID_STAGE, config_sha256
    )
    input_hashes = {
        name: sha256_file(grid_dir / name)
        for name in (RF2_GRID_PARQUET, DAMAGE_GRID_PARQUET, QOI_PARQUET)
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=sorted(outputs),
        seed_entropy=config.pipeline.seed_entropy,
        extra=dict(reference.metadata),
    )


def load_or_compute_reference(
    repo_root: Path | str, config: Config, config_sha256: str, force: bool = False
) -> FoldReference:
    """The cached reference, recomputed only when it is absent, incomplete, or forced."""
    directory = _reference_directory(repo_root, config, config_sha256)
    manifest_path = directory / "manifest.json"
    if not force and manifest_path.is_file() and verify_manifest(directory):
        payload = json.loads((directory / REFERENCE_JSON).read_text(encoding="utf-8"))
        metadata = {
            key: value for key, value in payload.items() if key not in ("jobs", "u_grid_mm")
        }
        return FoldReference(
            jobs=[str(job) for job in payload["jobs"]],
            u_grid=np.asarray(payload["u_grid_mm"], dtype=float),
            fold_of_run=np.load(directory / FOLD_OF_RUN_NPY),
            force_mean=np.load(directory / FORCE_MEAN_NPY),
            force_variance=np.load(directory / FORCE_VARIANCE_NPY),
            damage_mean=np.load(directory / DAMAGE_MEAN_NPY),
            damage_variance=np.load(directory / DAMAGE_VARIANCE_NPY),
            metadata=metadata,
        )
    reference = compute_reference(repo_root, config, config_sha256)
    _write_reference(directory, reference, config, config_sha256)
    return reference


# ---------------------------------------------------------------------------
# The shared metric definitions
# ---------------------------------------------------------------------------


def gaussian_nlpd(
    truth: np.ndarray, mean: np.ndarray, variance: np.ndarray
) -> float:
    """Mean Gaussian negative log predictive density per station, in nats.

    One definition, used by both sides of every ablation that quotes an NLPD, because two
    definitions of a log density differing by a constant would make a comparison meaningless
    while looking like a measurement.
    """
    y = np.asarray(truth, dtype=float)
    mu = np.asarray(mean, dtype=float)
    var = np.asarray(variance, dtype=float)
    if not (y.shape == mu.shape == var.shape):
        raise ValueError(
            f"an NLPD needs matching shapes, got {y.shape}, {mu.shape} and {var.shape}."
        )
    if np.any(var <= 0.0):
        raise ValueError(
            "a predictive variance is zero or negative somewhere, so the log density is not "
            "defined there. A model that claims certainty is a model to fix, not to floor."
        )
    return float(
        np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))
    )


def pointwise_coverage(
    truth: np.ndarray, mean: np.ndarray, variance: np.ndarray, level: float
) -> float:
    """Share of stations whose Gaussian interval at ``level`` contains the truth."""
    from scipy.stats import norm

    if not 0.0 < level < 1.0:
        raise ValueError(f"a coverage level must lie in (0, 1), got {level}.")
    z = float(norm.ppf(0.5 + level / 2.0))
    half = z * np.sqrt(np.asarray(variance, dtype=float))
    inside = np.abs(np.asarray(truth, dtype=float) - np.asarray(mean, dtype=float)) <= half
    return float(inside.mean())


def informative_stations(truth: np.ndarray) -> np.ndarray:
    """Stations where the observed family actually varies, as a boolean mask.

    Every run in this campaign is displacement controlled from zero, so the force and the
    damage are identically zero at the first station across all 198 curves. A model that
    reproduces that exactly has zero predictive variance there, and a log density against a
    zero variance is not a number. The calibration stage of build spec 11.2 met the same
    degeneracy and answered it the same way, by excluding the station from the score rather
    than by flooring a variance to keep it (ground rule 4). This is that rule, reused here so
    both sides of every ablation are scored on the same abscissae.
    """
    return np.asarray(truth, dtype=float).std(axis=0) > 0.0


def scored_stations(truth: np.ndarray, *variances: np.ndarray | None) -> np.ndarray:
    """Stations both sides of a comparison can be scored on, as a boolean mask.

    The informative stations of :func:`informative_stations`, intersected with the stations
    where every model in the comparison reports a strictly positive predictive variance. The
    intersection matters on the damage family: it is identically zero over an initial span in
    almost every run, so a fold whose training half is entirely zero there produces a basis
    with a zero mean, zero loadings and a zero truncation residual, and therefore a predictive
    variance of exactly zero at a station where the full family does vary. That is a real
    degeneracy of the family rather than a model claiming certainty, and the answer to it is
    the same as the calibration stage's: drop the station from the score and record how many
    were dropped, never floor a variance to keep it (ground rule 4).

    Both sides are then scored on the same stations by construction, which is the property
    that makes the comparison a comparison.
    """
    mask = informative_stations(truth)
    for variance in variances:
        if variance is not None:
            mask = mask & np.all(np.asarray(variance, dtype=float) > 0.0, axis=0)
    return mask


def curve_metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray | None,
    coverage_level: float = 0.9,
    stations: np.ndarray | None = None,
) -> dict[str, float]:
    """Every metric the curve ablations report, from one place so both sides match.

    ``variance`` may be None for a model that offers no predictive distribution, in which case
    the density and coverage entries are absent rather than filled with a placeholder. The
    error metrics run over every station; the density and the coverage run over ``stations``,
    which defaults to the informative ones and is passed explicitly by a caller comparing two
    models, so both are scored on the same abscissae. The count of what was excluded is
    reported beside them.
    """
    y = np.asarray(truth, dtype=float)
    mu = np.asarray(mean, dtype=float)
    errors = relative_l2(y, mu)
    out = {
        "relative_l2_mean": float(errors.mean()),
        "relative_l2_p50": float(np.percentile(errors, 50)),
        "relative_l2_p90": float(np.percentile(errors, 90)),
        # In the signal's own unit: N for a load displacement curve, dimensionless for damage.
        # Ground rule 14 keeps units out of a key that two signals share.
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mae": float(np.mean(np.abs(y - mu))),
        "n_curves": int(y.shape[0]),
    }
    if variance is not None:
        usable = informative_stations(y) if stations is None else np.asarray(stations, dtype=bool)
        out["nlpd"] = gaussian_nlpd(y[:, usable], mu[:, usable], variance[:, usable])
        out["coverage"] = pointwise_coverage(
            y[:, usable], mu[:, usable], variance[:, usable], coverage_level
        )
        out["coverage_level"] = float(coverage_level)
        out["n_stations_scored"] = int(usable.sum())
        out["n_stations_excluded"] = int(usable.size - usable.sum())
    return out


def peak_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Peak load agreement: signed bias in N, the R2 of the peak, and the curvature error.

    The curvature is the second difference of the curve at its own maximum, divided by the
    grid spacing squared, so it carries N per mm squared and a blunter peak reads as a smaller
    magnitude. It is measured at each curve's own peak station rather than at a common one,
    because a fixed station would measure where the peak moved rather than how sharp it is.
    """
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"peak metrics need matching shapes, got {y.shape} and {p.shape}.")
    true_peak = y.max(axis=1)
    predicted_peak = p.max(axis=1)
    signed = predicted_peak - true_peak
    denominator = float(np.sum((true_peak - true_peak.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("the observed peak loads have zero variance, so an R2 is undefined.")
    return {
        "peak_bias_N": float(signed.mean()),
        "peak_bias_relative": float((signed / true_peak).mean()),
        "peak_rmse_N": float(np.sqrt(np.mean(signed**2))),
        "peak_r2": float(1.0 - np.sum(signed**2) / denominator),
    }


def peak_curvature(curves: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Second difference at each curve's maximum, in N per mm squared, one value per curve."""
    matrix = np.asarray(curves, dtype=float)
    u = np.asarray(u_grid, dtype=float)
    spacing = float(u[1] - u[0])
    if not spacing > 0.0:
        raise ValueError("the displacement grid is not increasing, so a curvature is undefined.")
    index = np.clip(matrix.argmax(axis=1), 1, matrix.shape[1] - 2)
    rows = np.arange(matrix.shape[0])
    second = matrix[rows, index - 1] - 2.0 * matrix[rows, index] + matrix[rows, index + 1]
    return second / spacing**2
