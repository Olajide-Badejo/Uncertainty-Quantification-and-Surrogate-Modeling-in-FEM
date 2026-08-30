"""Ablation 4 of build spec 10.6.4: B-spline coefficient regression, the interpretable rival.

The prediction this script tests was committed to ``docs/ABLATIONS.md`` before the script
existed (ground rule 12), and the commit order in ``git log`` is the evidence.

Sixteen cubic B-spline basis functions on the displacement grid, knots denser near the peak,
one coefficient per basis function per curve by ordinary least squares, and one production
Gaussian process per coefficient. Prediction is a linear combination of fixed basis functions
with predicted coefficients, which is the whole appeal: a coefficient is the local height of the
curve near its knot, its Gaussian process is a statement about how that local height moves with
strength and cover, and there is no warp, no tangent space and no truncated eigenbasis between
the model and the reader.

Because the basis is fixed and the reconstruction is linear in the coefficients, the pointwise
predictive variance is the same propagation the production pipeline uses on its amplitude
scores, ``sum_j B_j(u)^2 sigma_j^2``, under the same independence assumption between the
processes. It is reported here for the same reason it is reported there, and it is uncalibrated
on both sides.

**Knot placement, stated so it is not mistaken for tuning.** The twelve interior knots sit at
the quantiles of a fifty fifty mixture of a uniform density over the full 0 to 20 mm stroke and
a normal density centered on the training folds' median displacement at peak with a standard
deviation of 2.5 mm. The mixture is inverted numerically on a fine grid. The peak location is a
statistic of the training half of each fold and is recomputed inside it, so no held out curve
influences where a knot goes.

Run it from the repository root after `ufem run all`:

    .venv/Scripts/python scripts/ablation_4_bspline.py
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ufem.ablation_reference import (  # noqa: E402
    curve_metrics,
    load_curve_data,
    load_or_compute_reference,
    peak_curvature,
    peak_metrics,
    production_folds,
    scored_stations,
)
from ufem.config import config_hash, load_config  # noqa: E402
from ufem.manifest import sha256_file, stage_dir, write_manifest  # noqa: E402
from ufem.surrogate import GPSettings, configure_torch, fit_all  # noqa: E402
from ufem.validate import relative_l2  # noqa: E402

STAGE_NAME = "ablation_4_bspline"
OUT_JSON = "ablation_4_bspline.json"

#: Build spec 10.6.4 asks for about sixteen cubic basis functions.
N_BASIS = 16
DEGREE = 3

#: The knot density mixture: half the interior knots' probability mass is spread uniformly over
#: the stroke and half is concentrated on the peak region, at this width in mm.
PEAK_WEIGHT = 0.5
PEAK_SIGMA_MM = 2.5

#: Resolution of the grid the mixture cumulative distribution is inverted on. Fine enough that
#: a knot lands within 0.002 mm of its quantile, which is a fiftieth of the displacement grid
#: spacing and therefore invisible to the basis.
CDF_GRID_POINTS = 10001

COVERAGE_LEVEL = 0.9

#: Spawn key for this ablation's seed tree (ground rule 13).
SPAWN_KEY: tuple[int, int] = (9, 4)


def peak_weighted_knots(
    u_min: float, u_max: float, n_basis: int, degree: int, peak_center: float
) -> np.ndarray:
    """The full knot vector: clamped ends, interior knots denser near ``peak_center``.

    A clamped cubic B-spline basis of ``n_basis`` functions needs ``n_basis + degree + 1``
    knots, of which ``degree + 1`` are repeated at each end, leaving ``n_basis - degree - 1``
    interior ones. Those sit at equally spaced quantiles of the mixture density described in
    the module docstring, which is a deterministic function of the training peak location and
    of nothing else.
    """
    from scipy.stats import norm

    n_interior = n_basis - degree - 1
    if n_interior < 1:
        raise ValueError(
            f"{n_basis} basis functions of degree {degree} leave no interior knot, so the "
            "basis is a single polynomial and the 'denser near the peak' placement is empty."
        )
    if not u_min < peak_center < u_max:
        raise ValueError(
            f"the peak center {peak_center} must lie inside ({u_min}, {u_max}) for the knot "
            "density to be a density on the stroke."
        )
    grid = np.linspace(u_min, u_max, CDF_GRID_POINTS)
    uniform_cdf = (grid - u_min) / (u_max - u_min)
    peak_cdf = norm.cdf(grid, loc=peak_center, scale=PEAK_SIGMA_MM)
    mixture = (1.0 - PEAK_WEIGHT) * uniform_cdf + PEAK_WEIGHT * peak_cdf
    # Renormalize onto [0, 1] over the stroke: the normal component puts a little mass outside
    # it, and a cumulative distribution that does not reach 1 at the right end would push every
    # quantile left by that much.
    mixture = (mixture - mixture[0]) / (mixture[-1] - mixture[0])
    quantiles = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    interior = np.interp(quantiles, mixture, grid)
    return np.concatenate(
        [np.full(degree + 1, u_min), interior, np.full(degree + 1, u_max)]
    )


def spline_basis(u_grid: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """The ``(n_points, n_basis)`` design matrix of the B-spline basis on the grid."""
    from scipy.interpolate import BSpline

    u = np.asarray(u_grid, dtype=float)
    matrix = BSpline.design_matrix(
        np.clip(u, knots[0], knots[-1]), np.asarray(knots, dtype=float), degree, extrapolate=False
    ).toarray()
    return np.asarray(matrix, dtype=float)


def fit_coefficients(basis: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Least squares coefficients of each curve on the basis, ``(n_curves, n_basis)``."""
    values = np.atleast_2d(np.asarray(curves, dtype=float))
    if values.shape[1] != basis.shape[0]:
        raise ValueError(
            f"the basis is defined at {basis.shape[0]} stations but the curves carry "
            f"{values.shape[1]}."
        )
    solution, _residuals, _rank, _singular = np.linalg.lstsq(basis, values.T, rcond=None)
    return solution.T


def coefficient_gp_predictions(
    coefficients: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> tuple[np.ndarray, np.ndarray]:
    """One production Gaussian process per coefficient, with its posterior variance."""
    targets = {
        f"coefficient_{index + 1}": coefficients[:, index]
        for index in range(coefficients.shape[1])
    }
    models, feature_standardizer, target_standardizers, _log = fit_all(
        X_train, targets, settings, seed_sequence
    )
    standardized = feature_standardizer.transform(X_test)
    mean = np.empty((standardized.shape[0], coefficients.shape[1]))
    variance = np.empty_like(mean)
    for index, name in enumerate(targets):
        column_mean, column_variance = models[name].predict(standardized)
        standardizer = target_standardizers[name]
        mean[:, index] = standardizer.inverse_mean(column_mean.reshape(-1, 1)).ravel()
        variance[:, index] = standardizer.inverse_variance(
            column_variance.reshape(-1, 1)
        ).ravel()
    return mean, variance


def main() -> int:
    started = _time.perf_counter()
    configure_torch()
    config = load_config(REPO_ROOT)
    digest = config_hash(config)
    data = load_curve_data(REPO_ROOT, config, digest)
    reference = load_or_compute_reference(REPO_ROOT, config, digest)
    if reference.jobs != data.jobs:
        raise AssertionError(
            "the production reference and the gridded curves carry different job orders."
        )
    settings = GPSettings.from_config(config)
    folds = production_folds(data.jobs, config)
    root = np.random.SeedSequence(config.pipeline.seed_entropy, spawn_key=SPAWN_KEY)
    fold_seeds = root.spawn(len(folds))

    u_grid = data.u_grid
    u_peak = data.qoi["u_peak_mm"].to_numpy(dtype=float)
    n_runs, n_grid = data.force.shape
    prediction = np.full((n_runs, n_grid), np.nan)
    prediction_variance = np.full((n_runs, n_grid), np.nan)
    projection = np.full((n_runs, n_grid), np.nan)
    knot_log: list[dict[str, Any]] = []

    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(n_runs), test_index, assume_unique=False)
        peak_center = float(np.median(u_peak[train_index]))
        knots = peak_weighted_knots(
            float(u_grid[0]), float(u_grid[-1]), N_BASIS, DEGREE, peak_center
        )
        basis = spline_basis(u_grid, knots, DEGREE)
        coefficients = fit_coefficients(basis, data.force[train_index])
        mean, variance = coefficient_gp_predictions(
            coefficients, data.X[train_index], data.X[test_index], settings, fold_seeds[index]
        )
        prediction[test_index] = mean @ basis.T
        prediction_variance[test_index] = variance @ (basis.T**2)
        # The projection of the held out truth onto this fold's basis, which separates what the
        # basis cannot represent from what the regression got wrong. It is not a prediction and
        # is labeled as such wherever it appears.
        projection[test_index] = fit_coefficients(basis, data.force[test_index]) @ basis.T
        knot_log.append(
            {
                "fold": index,
                "peak_center_mm": peak_center,
                "interior_knots_mm": [float(value) for value in knots[DEGREE + 1 : -DEGREE - 1]],
            }
        )
        fold_error = np.median(
            relative_l2(data.force[test_index], prediction[test_index])
        )
        print(
            f"[ablation 4] fold {index + 1}/{len(folds)}: knots centered on "
            f"{peak_center:.2f} mm, median relative L2 {fold_error * 100:.2f} percent"
        )

    if np.any(~np.isfinite(prediction)) or np.any(~np.isfinite(prediction_variance)):
        raise AssertionError(
            "the folds left runs without a spline prediction, so the pooled metrics would be "
            "an average over an unknown sample."
        )

    truth_curvature = peak_curvature(data.force, u_grid)
    ablation_curvature = peak_curvature(prediction, u_grid)
    production_curvature = peak_curvature(reference.force_mean, u_grid)
    stations = scored_stations(data.force, reference.force_variance, prediction_variance)
    ablation = {
        "force": curve_metrics(
            data.force, prediction, prediction_variance, COVERAGE_LEVEL, stations
        ),
        "peak": peak_metrics(data.force, prediction),
        "curvature": {
            "mean_signed_error": float(np.mean(ablation_curvature - truth_curvature)),
            "mean_absolute_error": float(np.mean(np.abs(ablation_curvature - truth_curvature))),
            "mean_magnitude": float(np.mean(np.abs(ablation_curvature))),
        },
        "basis_projection_force": curve_metrics(data.force, projection, None),
    }
    production = {
        "force": curve_metrics(
            data.force,
            reference.force_mean,
            reference.force_variance,
            COVERAGE_LEVEL,
            stations,
        ),
        "peak": peak_metrics(data.force, reference.force_mean),
        "curvature": {
            "mean_signed_error": float(np.mean(production_curvature - truth_curvature)),
            "mean_absolute_error": float(
                np.mean(np.abs(production_curvature - truth_curvature))
            ),
            "mean_magnitude": float(np.mean(np.abs(production_curvature))),
        },
    }
    results: dict[str, Any] = {
        "config_sha256": digest,
        "n_runs": n_runs,
        "n_folds": len(folds),
        "basis": {
            "n_basis": N_BASIS,
            "degree": DEGREE,
            "n_interior_knots": N_BASIS - DEGREE - 1,
            "peak_weight": PEAK_WEIGHT,
            "peak_sigma_mm": PEAK_SIGMA_MM,
            "placement": (
                "quantiles of a mixture of a uniform density over the stroke and a normal "
                "density on the training median displacement at peak, recomputed per fold"
            ),
        },
        "ablation": ablation,
        "production": production,
        "truth_curvature_mean_magnitude": float(np.mean(np.abs(truth_curvature))),
        "coverage_level": COVERAGE_LEVEL,
        "knot_log": knot_log,
        "wall_time_s": _time.perf_counter() - started,
    }

    artifact_root = REPO_ROOT / config.pipeline.paths.artifact_root
    directory = stage_dir(artifact_root, STAGE_NAME, digest)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUT_JSON
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    reference_dir = stage_dir(artifact_root, "ablation_reference", digest)
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=digest,
        input_hashes={
            "ablation_reference/reference.json": sha256_file(reference_dir / "reference.json")
        },
        outputs=[path],
        seed_entropy=config.pipeline.seed_entropy,
        extra={
            "wall_time_s": results["wall_time_s"],
            "spawn_key": list(SPAWN_KEY),
            "basis": results["basis"],
            "ablation_force": ablation["force"],
            "production_force": production["force"],
        },
    )

    print(f"\nAblation 4, B-spline coefficients, {n_runs} runs over {len(folds)} grouped folds")
    print(
        f"  median relative L2   ablation {ablation['force']['relative_l2_p50'] * 100:.2f} "
        f"percent, production {production['force']['relative_l2_p50'] * 100:.2f} percent"
    )
    print(
        f"  peak load bias       ablation {ablation['peak']['peak_bias_N']:+.1f} N, "
        f"production {production['peak']['peak_bias_N']:+.1f} N"
    )
    print(
        f"  peak curvature error ablation "
        f"{ablation['curvature']['mean_absolute_error']:.1f}, production "
        f"{production['curvature']['mean_absolute_error']:.1f} N per mm squared, truth "
        f"magnitude {results['truth_curvature_mean_magnitude']:.1f}"
    )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
