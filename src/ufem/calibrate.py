"""Stage ``calibrate``: measured predictive uncertainty with a finite sample guarantee.

Build spec section 11, and the direct answer to the fatal defect of section 5.1. The
predecessor project floored a standard deviation at 0.01, multiplied every variance by 1.1,
and injected half a percent of multiplicative noise, then published the resulting intervals.
Nothing of that kind happens here. There are exactly two operations on a predictive
distribution in this file, and both are measurements:

1. **One out of fold variance scaling factor per target** (build spec 11.3). It is the square
   root of the mean squared standardized leave one out residual, so applying it makes the
   predictive variance adequacy zero by construction. It is not chosen, it is measured, and
   whatever comes out is reported, including if it comes out far from 1.
2. **A conformal quantile on top of the recalibrated model** (build spec 11.1 and 11.2). This
   is where the coverage guarantee comes from, and it is distribution free: it holds under
   exchangeability whether or not step 1 left the Gaussian assumption in good shape. Step 1
   only shapes the band; step 2 is what makes it cover.

There is no third operation. No floor, no clip, no factor chosen to make a gate pass. If the
gate at the end of this file fails, the fix is model revision, and the stage exits nonzero
rather than widening anything (build spec 11.5, ground rule 4).

**What is out of fold here, and what is not.** The scalar leave one out is the closed form of
Dubrule 1983 and Rasmussen and Williams 5.4.2, at hyperparameters fitted on all 198 points;
the stated consequence is that each fold's model saw its own held out point through the
hyperparameter fit. The honest cross check is the 10 fold CV+ below, which refits the
hyperparameters inside every fold, and both numbers go in the table exactly as build spec 11.1
asks. The functional bands use the same closed form leave one out on the score processes,
propagated through the reduction; the reduction basis and the registration reference are the
ones fitted on all 198 curves, which is an approximation stated here, in the manifest, and in
docs/DESIGN_DECISIONS.md. The P4 grouped fold harness, which does refit both inside every
fold, is the cross check on what that approximation costs at curve level.

Units: the scalar intervals carry their target's unit (N, mm, N/mm, N mm, or dimensionless);
the curve bands are in N against displacement in mm; every score, coverage, PIT value and
scaling factor is dimensionless.
"""

from __future__ import annotations

import json
import math
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.config import Config, features
from ufem.conformal_functional import (
    band_scale,
    conformal_rank,
    coverage_bounds,
    leave_one_out_coverage,
    sup_norm_scores,
)
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.register import LANDMARKS_PARQUET, curve_matrix
from ufem.register import STAGE_NAME as REGISTER_STAGE
from ufem.surrogate import (
    CURVE_BLOCKS,
    LANDMARK_QOI,
    SCALAR_QOI,
    FittedGP,
    GPSettings,
    Standardizer,
    SurrogateModel,
    configure_torch,
    fit_all,
)
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.validate import QOI_LABELS, gp_leave_one_out, make_folds
from ufem.validate import STAGE_NAME as VALIDATE_STAGE

STAGE_NAME = "calibrate"

#: Output file names inside the stage directory.
CALIBRATION_JSON = "calibration.json"
SCALAR_CONFORMAL_PARQUET = "scalar_conformal.parquet"
CURVE_CONFORMAL_PARQUET = "curve_conformal.parquet"
PIT_PARQUET = "pit_by_abscissa.parquet"
COVERAGE_SWEEP_PARQUET = "coverage_sweep.parquet"
BAND_EXAMPLES_PARQUET = "band_examples.parquet"
CONFORMAL_TEX = "conformal_scalars.tex"
DIAGNOSTICS_TEX = "conformal_diagnostics.tex"
CALIBRATION_MD = "calibration_summary.md"

#: The two curve families the functional bands are built for.
SIGNAL_FORCE = "force"
SIGNAL_DAMAGE = "damage"
CURVE_SIGNALS: tuple[str, str] = (SIGNAL_FORCE, SIGNAL_DAMAGE)

#: Nominal coverage levels the before and after sweep of build spec 11.4 is measured on. A
#: plotting and reporting grid rather than a threshold: nothing is decided on these, the gate
#: reads the 90 percent level the spec names, and that level is in this list.
COVERAGE_LEVELS: tuple[float, ...] = (
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
)

#: Confidence level of every Wilson interval reported here. At n = 198 the standard error on a
#: 90 percent coverage is about 2 percent, which build spec 11.4 is explicit about: a naked
#: coverage number is not evidence, so no coverage is reported without this interval beside it.
WILSON_LEVEL = 0.95

#: Bins of the probability integral transform histogram at each abscissa.
PIT_BINS = 10

#: The gate of build spec 11.5, thresholds and their reasoning.
#:
#: ``GATE_ALPHA`` is the 90 percent level the spec names. ``GATE_PIT_OUTER_MASS_MAX`` makes
#: "no gross U shape on the softening branch" a number, because a visual criterion in a
#: machine checked gate is not a criterion. The statistic is the fraction of PIT values in the
#: outer two deciles (below 0.1 or above 0.9) over the post peak part of every curve. A
#: calibrated predictive distribution puts exactly 0.2 there. A U shaped PIT histogram is one
#: with too much mass in both tails, which is what an overconfident variance produces, so the
#: statistic rises above 0.2 exactly when the U appears; 0.35 is 75 percent more outer mass
#: than a calibrated model has, and a U shape severe enough to be called gross by eye runs well
#: past it (a predictive standard deviation half of what it should be puts 0.52 there). The
#: threshold gates the recalibrated model, which is the one the bands are built on, and the
#: before and after values are both reported so the reader sees what the scaling did.
GATE_ALPHA = 0.1
GATE_PIT_OUTER_MASS_MAX = 0.35
#: Expected outer decile mass under a calibrated predictive distribution, by definition.
PIT_OUTER_MASS_NOMINAL = 0.2

#: Curves drawn from the posterior for the modulation cross check, from the config's
#: ``conformal.K_posterior_draws``. See :func:`modulation_cross_check`.
MODULATION_CHECK_CURVES = 12


class CalibrationGateFailed(AssertionError):
    """The gate of build spec 11.5 did not pass, so no propagated number may be computed."""


# ---------------------------------------------------------------------------
# Small measurements, each with its formula in the docstring
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int, trials: int, level: float = WILSON_LEVEL
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``(p + z^2/2n +/- z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)``. Preferred over the normal
    approximation because the coverages measured here sit near 0.9 and 0.95, where the naive
    interval is both too narrow and capable of exceeding 1.
    """
    from scipy import stats

    if trials <= 0:
        raise ValueError("a Wilson interval needs at least one trial.")
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes out of {trials} trials is not a proportion.")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - spread, center + spread


def predictive_variance_adequacy(standardized: np.ndarray) -> float:
    """The one number summary of build spec 11.4, in its log form.

    ``PVA = log((1/n) sum_i z_i^2)`` with ``z_i = (y_i - mu_-i) / sigma_-i``, natural log,
    zero when the predictive variance is exactly adequate out of fold. Positive means the
    model is overconfident (residuals larger than the variance claims), negative means it is
    underconfident. The log form is used because a factor of two either way should read as the
    same distance from adequate, which the raw ratio does not do.
    """
    z = np.asarray(standardized, dtype=float).ravel()
    if z.size == 0:
        raise ValueError("predictive variance adequacy needs at least one residual.")
    mean_square = float(np.mean(z**2))
    if mean_square <= 0.0:
        raise ValueError(
            "every standardized residual is zero, so the predictive variance adequacy is not "
            "defined. A model that reproduces its held out points exactly is a leak."
        )
    return math.log(mean_square)


def variance_scaling_factor(standardized: np.ndarray) -> float:
    """The out of fold factor that brings the predictive variance adequacy to zero.

    ``tau = sqrt((1/n) sum_i z_i^2)``, so that ``z_i / tau`` has mean square exactly one. This
    is the recalibration build spec 11.3 puts first, and it is a measurement of the residuals
    the model has already been shown, not a knob: there is no target for it other than what
    the data says, and the value is reported per target whatever it turns out to be. Ground
    rule 4 bans a variance multiplier chosen to make a band cover; this is the opposite object,
    a multiplier read off the out of fold residuals and published.
    """
    z = np.asarray(standardized, dtype=float).ravel()
    return math.sqrt(float(np.mean(z**2)))


def leave_one_out_scaling_factors(standardized: np.ndarray) -> np.ndarray:
    """``tau_-i``: the scaling factor measured without point i, one per point.

    The global factor is fitted on the same residuals it is then evaluated on, which makes the
    post scaling adequacy zero by construction and therefore uninformative. Leaving each point
    out of its own factor removes that circularity at no cost, and the adequacy computed with
    these is the honest out of sample number the report quotes next to the trivial one.
    """
    z = np.asarray(standardized, dtype=float).ravel()
    n = z.size
    if n < 2:
        raise ValueError("a leave one out scaling factor needs at least two residuals.")
    total = float(np.sum(z**2))
    return np.sqrt((total - z**2) / (n - 1))


def crps_gaussian(truth: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed form continuous ranked probability score of a Gaussian forecast.

    ``CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]`` with ``z = (y - mu)/sigma``
    (Gneiting and Raftery 2007). Lower is better, and it carries the target's own unit, which
    is why the report quotes it as a skill score against climatology rather than raw.
    """
    from scipy import stats

    y = np.asarray(truth, dtype=float)
    mu = np.asarray(mean, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("the CRPS of a Gaussian needs a positive standard deviation.")
    z = (y - mu) / s
    return s * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def negative_log_predictive_density(
    truth: np.ndarray, mean: np.ndarray, sigma: np.ndarray
) -> float:
    """Mean NLPD of a Gaussian predictive: ``0.5 log(2 pi sigma^2) + z^2/2``, nats."""
    y = np.asarray(truth, dtype=float)
    mu = np.asarray(mean, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("the NLPD of a Gaussian needs a positive standard deviation.")
    z = (y - mu) / s
    return float(np.mean(0.5 * np.log(2.0 * math.pi * s**2) + 0.5 * z**2))


def climatology_leave_one_out(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The baseline every skill score is measured against: the other points' mean and spread.

    Leave one out rather than the whole sample, so the baseline is out of sample in exactly
    the sense the model is. Returns ``(mean_-i, sigma_-i)``.
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.size
    if n < 3:
        raise ValueError("a leave one out climatology needs at least three points.")
    total = y.sum()
    mean = (total - y) / (n - 1)
    total_square = float(np.sum(y**2))
    variance = (total_square - y**2) / (n - 1) - mean**2
    variance = variance * (n - 1) / (n - 2)
    if np.any(variance <= 0.0):
        raise ValueError("the leave one out climatology has a non positive variance.")
    return mean, np.sqrt(variance)


# ---------------------------------------------------------------------------
# Jackknife+ and CV+
# ---------------------------------------------------------------------------


def jackknife_plus_intervals(
    cross_mean: np.ndarray, cross_sigma: np.ndarray, scores: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """The jackknife+ interval at every training point, leaving that point's model out.

    Barber, Candes, Ramdas and Tibshirani (2021). The interval at a query point is not the
    prediction plus and minus a residual quantile; it is a quantile over the ensemble of leave
    one out models evaluated at that query point, each shifted by its own residual:

        upper = the kth smallest of { mu_-i(x) + R_i sigma_-i(x) }
        lower = the (m + 1 - k)th smallest of { mu_-i(x) - R_i sigma_-i(x) }

    with ``m`` models in the ensemble and ``k = ceil((1 - alpha)(m + 1))``. The scores ``R_i``
    are sigma normalized (build spec 11.1), so each is multiplied back by the standard
    deviation of the model that produced it, at the query point rather than at its own point.

    ``cross_mean``, ``cross_sigma`` and ``scores`` are ``(n_query, n_models)``; ``scores`` may
    also be one row shared by every query. An entry that is ``nan`` in any of the three is not
    part of that query's ensemble, which is how the caller excludes a model that must not vote:
    the deployed band excludes the query's own leave one out model, and the nested evaluation
    of :func:`nested_jackknife_plus` excludes everything that ever saw the query's response.
    """
    mu = np.atleast_2d(np.asarray(cross_mean, dtype=float))
    sigma = np.atleast_2d(np.asarray(cross_sigma, dtype=float))
    r = np.asarray(scores, dtype=float)
    if r.ndim == 1:
        r = np.tile(r, (mu.shape[0], 1))
    if mu.shape != sigma.shape or mu.shape != r.shape:
        raise ValueError(
            f"jackknife+ needs matching shapes: means {mu.shape}, sigmas {sigma.shape}, "
            f"scores {r.shape}."
        )
    half = r * sigma
    upper_all = mu + half
    lower_all = mu - half
    usable = np.isfinite(upper_all) & np.isfinite(lower_all)
    n_query = mu.shape[0]
    lower = np.empty(n_query)
    upper = np.empty(n_query)
    for j in range(n_query):
        keep = usable[j]
        m = int(keep.sum())
        rank = math.ceil((1.0 - alpha) * (m + 1))
        if rank > m:
            raise ValueError(
                f"a {100 * (1 - alpha):.1f} percent jackknife+ interval needs the {rank}th of "
                f"{m} leave one out models, which does not exist at this sample size."
            )
        upper[j] = np.sort(upper_all[j][keep])[rank - 1]
        lower[j] = np.sort(lower_all[j][keep])[m - rank]
    return lower, upper


def nested_jackknife_plus(
    model: FittedGP, standardizer: Standardizer, truth: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Jackknife+ intervals whose every ingredient is blind to the point they are tested on.

    This is the honest empirical coverage of build spec 11.5, and getting it honest changed
    the answer. Evaluating the deployed jackknife+ band at a training point measures something
    conservative, because the ``n - 1`` leave one out models that set the interval at ``x_j``
    were all fitted on ``y_j`` and interpolate it. On this campaign that inflated the measured
    coverage of a nominal 90 percent interval to 95.5 percent for the displacement at peak and
    96.5 percent for the initial stiffness, the two quantities whose in sample fit most exceeds
    their out of sample fit. The band was not wrong; the measurement was.

    So the query point is removed first, and the whole construction happens inside what is
    left: the ensemble is the leave ``{i, j}`` out models, the nonconformity scores are the
    leave one out residuals of the reduced problem, and the variance scaling factor is measured
    on those same reduced residuals. Nothing that touches ``y_j`` enters the interval at
    ``x_j``, so the resulting coverage is what a genuinely new run would see. The closed form
    makes it cheap: see :meth:`ufem.surrogate.FittedGP.nested_leave_one_out`.
    """
    nested = model.nested_leave_one_out()
    query_mean, query_variance = in_units(
        standardizer, nested["query_mean"], nested["query_variance"]
    )
    inner_mean, inner_variance = in_units(
        standardizer, nested["inner_mean"], nested["inner_variance"]
    )
    inner_sigma = np.sqrt(inner_variance)
    standardized = (truth[None, :] - inner_mean) / inner_sigma
    with np.errstate(invalid="ignore"):
        tau = np.sqrt(np.nanmean(standardized**2, axis=1))
    scores = np.abs(standardized) / tau[:, None]
    sigma = tau[:, None] * np.sqrt(query_variance)
    lower, upper = jackknife_plus_intervals(query_mean, sigma, scores, alpha)
    covered = (truth >= lower) & (truth <= upper)
    hits = int(covered.sum())
    low, high = wilson_interval(hits, truth.size)
    return {
        "nominal": 1.0 - alpha,
        "empirical": hits / truth.size,
        "wilson_low": low,
        "wilson_high": high,
        "median_width": float(np.median(upper - lower)),
        "mean_width": float(np.mean(upper - lower)),
        "median_scaling_factor": float(np.median(tau)),
        "n": int(truth.size),
    }


def cross_plus_intervals(
    cross_mean: np.ndarray,
    cross_sigma: np.ndarray,
    scores: np.ndarray,
    fold_of: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ (Barber et al. 2021, section 3), the same construction over fold models.

    ``cross_mean[j, i]`` is the prediction at point ``j`` of the model fitted without the fold
    containing point ``i``, so the ensemble has one entry per training point exactly as
    jackknife+ does, with ties across the points of a fold. The point's own fold is excluded
    from its ensemble, which is the CV+ analogue of removing the diagonal.
    """
    mu = np.atleast_2d(np.asarray(cross_mean, dtype=float))
    sigma = np.atleast_2d(np.asarray(cross_sigma, dtype=float))
    r = np.asarray(scores, dtype=float).ravel()
    folds = np.asarray(fold_of, dtype=int).ravel()
    if mu.shape != sigma.shape or mu.shape[1] != r.size or folds.size != r.size:
        raise ValueError("CV+ needs matching cross predictions, scores and fold labels.")
    n_query = mu.shape[0]
    half = r[None, :] * sigma
    upper_all = mu + half
    lower_all = mu - half
    lower = np.empty(n_query)
    upper = np.empty(n_query)
    for j in range(n_query):
        keep = folds != folds[j]
        m = int(keep.sum())
        rank = math.ceil((1.0 - alpha) * (m + 1))
        if rank > m:
            raise ValueError(
                f"a {100 * (1 - alpha):.1f} percent CV+ interval needs the {rank}th of {m} "
                "fold models, which does not exist at this fold count."
            )
        upper[j] = np.sort(upper_all[j][keep])[rank - 1]
        lower[j] = np.sort(lower_all[j][keep])[m - rank]
    return lower, upper


def in_units(
    standardizer: Standardizer, mean: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Undo a target standardization on a mean and a variance of any shape."""
    shape = np.asarray(mean).shape
    back_mean = standardizer.inverse_mean(np.asarray(mean, dtype=float).reshape(-1, 1))
    back_variance = standardizer.inverse_variance(np.asarray(variance, dtype=float).reshape(-1, 1))
    return back_mean.reshape(shape), back_variance.reshape(shape)


# ---------------------------------------------------------------------------
# The scalar branch
# ---------------------------------------------------------------------------


def scalar_calibration(
    surrogate: SurrogateModel,
    targets: dict[str, np.ndarray],
    alphas: list[float],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Jackknife+ with sigma normalized scores for every scalar target (build spec 11.1)."""
    records: dict[str, dict[str, Any]] = {}
    frames: list[pd.DataFrame] = []
    for name, truth in targets.items():
        model = surrogate.models[name]
        standardizer = surrogate.target_standardizers[name]
        mean, variance = gp_leave_one_out(surrogate, name)
        sigma = np.sqrt(variance)
        raw_z = (truth - mean) / sigma

        tau = variance_scaling_factor(raw_z)
        tau_loo = leave_one_out_scaling_factors(raw_z)
        scaled_z = raw_z / tau_loo
        scores = np.abs(scaled_z)

        cross_mean_std, cross_variance_std = model.leave_one_out_cross_predictions()
        cross_mean, cross_variance = in_units(standardizer, cross_mean_std, cross_variance_std)
        cross_sigma = tau * np.sqrt(cross_variance)
        # The deployed band at a training point must not let that point's own leave one out
        # model vote; masking the diagonal is how the ensemble is told so.
        deployed_mean = cross_mean.copy()
        np.fill_diagonal(deployed_mean, np.nan)

        per_alpha: dict[str, Any] = {}
        interval_columns: dict[str, np.ndarray] = {}
        for alpha in alphas:
            lower, upper = jackknife_plus_intervals(
                deployed_mean, cross_sigma, scores, alpha
            )
            honest = nested_jackknife_plus(model, standardizer, truth, alpha)
            per_alpha[f"{alpha:g}"] = {
                **honest,
                "deployed_median_width": float(np.median(upper - lower)),
                "rank_coverage": _rank_coverage(scores, alpha),
            }
            interval_columns[f"lower_{alpha:g}"] = lower
            interval_columns[f"upper_{alpha:g}"] = upper

        climate_mean, climate_sigma = climatology_leave_one_out(truth)
        records[name] = {
            "n": int(truth.size),
            "variance_scaling_factor": tau,
            "pva_before": predictive_variance_adequacy(raw_z),
            "pva_after": predictive_variance_adequacy(scaled_z),
            "crps_before": float(np.mean(crps_gaussian(truth, mean, sigma))),
            "crps_after": float(np.mean(crps_gaussian(truth, mean, tau * sigma))),
            "crps_climatology": float(np.mean(crps_gaussian(truth, climate_mean, climate_sigma))),
            "nlpd_before": negative_log_predictive_density(truth, mean, sigma),
            "nlpd_after": negative_log_predictive_density(truth, mean, tau * sigma),
            "nlpd_climatology": negative_log_predictive_density(truth, climate_mean, climate_sigma),
            "jackknife_plus": per_alpha,
            "coverage_sweep": _gaussian_coverage_sweep(raw_z, scaled_z, scores),
        }
        records[name]["crps_skill_before"] = 1.0 - (
            records[name]["crps_before"] / records[name]["crps_climatology"]
        )
        records[name]["crps_skill_after"] = 1.0 - (
            records[name]["crps_after"] / records[name]["crps_climatology"]
        )
        frames.append(
            pd.DataFrame(
                {
                    "target": name,
                    "truth": truth,
                    "loo_mean": mean,
                    "loo_sigma": sigma,
                    "loo_sigma_scaled": tau * sigma,
                    "standardized_residual": raw_z,
                    "score": scores,
                    **interval_columns,
                }
            )
        )
    return records, pd.concat(frames, ignore_index=True)


def _rank_coverage(scores: np.ndarray, alpha: float) -> dict[str, float]:
    """Coverage of the plain conformal band, measured by the exact leave one out rank rule."""
    covered = leave_one_out_coverage(scores, alpha)
    hits = int(covered.sum())
    low, high = wilson_interval(hits, covered.size)
    bracket_low, bracket_high = coverage_bounds(covered.size - 1, alpha)
    return {
        "empirical": hits / covered.size,
        "wilson_low": low,
        "wilson_high": high,
        "bracket_low": bracket_low,
        "bracket_high": bracket_high,
    }


def _gaussian_coverage_sweep(
    raw_z: np.ndarray, scaled_z: np.ndarray, scores: np.ndarray
) -> list[dict[str, float]]:
    """Coverage against nominal at every level of ``COVERAGE_LEVELS``, before and after.

    Three curves per target: the raw Gaussian predictive, the same after the measured variance
    scaling, and the conformal band. The first two are what build spec 11.4 means by the paired
    before and after; the third is the one with a guarantee attached.
    """
    from scipy import stats

    rows: list[dict[str, float]] = []
    n = raw_z.size
    for level in COVERAGE_LEVELS:
        z_critical = float(stats.norm.ppf(0.5 + level / 2.0))
        row: dict[str, float] = {"nominal": level}
        for label, values in (("before", raw_z), ("after", scaled_z)):
            hits = int(np.sum(np.abs(values) <= z_critical))
            low, high = wilson_interval(hits, n)
            row[f"{label}_empirical"] = hits / n
            row[f"{label}_wilson_low"] = low
            row[f"{label}_wilson_high"] = high
        alpha = 1.0 - level
        try:
            conformal = _rank_coverage(scores, alpha)
        except ValueError:
            # The sample cannot support this level. Recorded as absent rather than as a number
            # the data does not have (ground rule 1); the sweep still reports the Gaussian rows.
            row["conformal_empirical"] = float("nan")
            row["conformal_wilson_low"] = float("nan")
            row["conformal_wilson_high"] = float("nan")
        else:
            row["conformal_empirical"] = conformal["empirical"]
            row["conformal_wilson_low"] = conformal["wilson_low"]
            row["conformal_wilson_high"] = conformal["wilson_high"]
        rows.append(row)
    return rows


def cv_plus_calibration(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    jobs: list[str],
    config: Config,
    settings: GPSettings,
    alphas: list[float],
) -> dict[str, dict[str, Any]]:
    """The honest cross check of build spec 11.1: 10 fold CV+ with per fold refits.

    Nothing is reused from the production fit. Every fold refits the kernel hyperparameters,
    the constant mean and the noise on its own training rows, so the held out points were never
    seen in any capacity, not even through a hyperparameter. The folds are the ones
    ``ufem.validate`` uses, drawn from the same spawned ``SeedSequence`` positions, so this is
    the same fold assignment and the same fitted models the P4 baseline comparison reports,
    read for their predictive distributions rather than their point errors.
    """
    root_sequence = np.random.SeedSequence(config.pipeline.seed_entropy)
    fold_seed, model_seed = root_sequence.spawn(2)
    folds = make_folds(jobs, int(config.pipeline.validation.n_folds), fold_seed)
    fold_children = model_seed.spawn(len(folds))

    n = len(jobs)
    names = list(targets)
    fold_of = np.empty(n, dtype=int)
    cross_mean = {name: np.empty((n, n)) for name in names}
    cross_variance = {name: np.empty((n, n)) for name in names}
    for index, test_index in enumerate(folds):
        fold_of[test_index] = index
        train_index = np.setdiff1d(np.arange(n), test_index, assume_unique=False)
        fold_targets = {name: targets[name][train_index] for name in names}
        # Mirror ufem.validate.run_fold's spawn tree exactly: it spawns two children per fold
        # and hands the second to the scalar refits.
        children = fold_children[index].spawn(2)
        models, feature_standardizer, standardizers, _log = fit_all(
            X[train_index], fold_targets, settings, children[1]
        )
        design = feature_standardizer.transform(X)
        for name in names:
            mean, variance = models[name].predict(design)
            mean, variance = in_units(standardizers[name], mean, variance)
            cross_mean[name][:, test_index] = mean[:, None]
            cross_variance[name][:, test_index] = variance[:, None]

    records: dict[str, dict[str, Any]] = {}
    for name in names:
        truth = targets[name]
        own = np.take_along_axis(
            cross_mean[name], np.arange(n)[:, None], axis=1
        )[:, 0]
        own_variance = np.take_along_axis(
            cross_variance[name], np.arange(n)[:, None], axis=1
        )[:, 0]
        raw_z = (truth - own) / np.sqrt(own_variance)
        tau = variance_scaling_factor(raw_z)
        scores = np.abs(raw_z / leave_one_out_scaling_factors(raw_z))
        sigma = tau * np.sqrt(cross_variance[name])
        per_alpha: dict[str, Any] = {}
        for alpha in alphas:
            lower, upper = cross_plus_intervals(
                cross_mean[name], sigma, scores, fold_of, alpha
            )
            covered = (truth >= lower) & (truth <= upper)
            hits = int(covered.sum())
            low, high = wilson_interval(hits, n)
            per_alpha[f"{alpha:g}"] = {
                "nominal": 1.0 - alpha,
                "empirical": hits / n,
                "wilson_low": low,
                "wilson_high": high,
                "median_width": float(np.median(upper - lower)),
                "n": n,
                # The fold models that set the interval at a point were all fitted on data
                # containing it (only its own fold's model was not), so this coverage is
                # conservative for the same reason the deployed jackknife+ one is. The rank
                # coverage beside it has no such contamination: every point is scored by the
                # one model that excluded it, and compared against the other folds' scores.
                "rank_coverage": _rank_coverage(scores, alpha),
            }
        records[name] = {
            "variance_scaling_factor": tau,
            "pva_before": predictive_variance_adequacy(raw_z),
            "n_folds": len(folds),
            "cv_plus": per_alpha,
        }
    return records


# ---------------------------------------------------------------------------
# The functional branch
# ---------------------------------------------------------------------------


def informative_abscissae(truth: np.ndarray) -> np.ndarray:
    """The abscissae a band can be calibrated on: those where the family is not constant.

    Both curve families have stretches where every one of the 198 runs takes the same value,
    and a standardized residual there is 0/0. The load displacement family has one such point,
    the origin, where displacement control makes the force exactly zero by construction. The
    damage family has 79 of them: the first 0.4 mm, where no damage has initiated, and
    everything past 12.2 mm, where every run has reached the same saturated value. That is the
    damage saturation of build spec 5.6 showing up as a degenerate domain rather than as a
    weak correlation, and the honest response is to say where the band exists rather than to
    put a floor under a variance that is genuinely zero (ground rule 4).

    The rule is computed from the observed family alone, so it does not depend on the model,
    and the excluded span is reported in the artifact.
    """
    y = np.atleast_2d(np.asarray(truth, dtype=float))
    spread = y.max(axis=0) - y.min(axis=0)
    mask = spread > 0.0
    if not mask.any():
        raise ValueError(
            "every abscissa of this family is constant across all runs, so there is nothing "
            "to calibrate a band on."
        )
    return mask


def leave_one_out_curves(surrogate: SurrogateModel) -> dict[str, np.ndarray]:
    """Out of fold mean and variance for every training curve, both signals.

    The score processes get the same closed form leave one out the scalars do, and the result
    is pushed through the reduction with :meth:`SurrogateModel.curve_from_scores`, which is the
    same propagation ``predict_curve`` uses. What is *not* refitted per fold is the reduction
    basis and the registration reference, so this is a leave one out of the Gaussian processes
    inside a fixed representation. Build spec 16.3's fully refitted fold harness lives in
    ``ufem.validate`` and measures what that costs at curve level; the approximation is stated
    in the manifest and in docs/DESIGN_DECISIONS.md rather than left for a reader to infer.
    """
    names = surrogate.score_targets
    means: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for block in CURVE_BLOCKS:
        block_names = names[block]
        n_rows = surrogate.models[block_names[0]].train_y.size if block_names else 0
        block_mean = np.empty((n_rows, len(block_names)))
        block_variance = np.empty((n_rows, len(block_names)))
        for column, name in enumerate(block_names):
            mean, variance = gp_leave_one_out(surrogate, name)
            block_mean[:, column] = mean
            block_variance[:, column] = variance
        means[block] = block_mean
        variances[block] = block_variance
    prediction = surrogate.curve_from_scores(means, variances)
    return {
        "u_grid": prediction.u_grid,
        f"{SIGNAL_FORCE}_mean": prediction.force_mean,
        f"{SIGNAL_FORCE}_sigma": prediction.force_std(),
        f"{SIGNAL_DAMAGE}_mean": prediction.damage_mean,
        f"{SIGNAL_DAMAGE}_sigma": prediction.damage_std(),
    }


def functional_calibration(
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    alphas: list[float],
) -> dict[str, Any]:
    """Simultaneous sup norm bands with the measured variance scaling applied first.

    One scaling factor per signal, not one per abscissa. A factor that varied along the
    displacement axis would be reshaping the band rather than correcting how adequate it is,
    and the shape is the part the Gaussian process is supposed to supply; the sup norm score
    then judges that shape rather than repairing it.
    """
    raw_z = (truth - mean) / sigma
    tau = variance_scaling_factor(raw_z)
    scaled_sigma = tau * sigma
    scores = sup_norm_scores(truth, mean, scaled_sigma)
    per_alpha: dict[str, Any] = {}
    for alpha in alphas:
        covered = leave_one_out_coverage(scores, alpha)
        hits = int(covered.sum())
        low, high = wilson_interval(hits, covered.size)
        bracket_low, bracket_high = coverage_bounds(covered.size - 1, alpha)
        scale = band_scale(scores, alpha)
        per_alpha[f"{alpha:g}"] = {
            "nominal": 1.0 - alpha,
            "empirical": hits / covered.size,
            "wilson_low": low,
            "wilson_high": high,
            "bracket_low": bracket_low,
            "bracket_high": bracket_high,
            "band_scale": scale,
            "rank": conformal_rank(scores.size, alpha),
            "median_half_width": float(np.median(scale * scaled_sigma)),
            "n": int(covered.size),
        }
    return {
        "n": int(truth.shape[0]),
        "n_grid": int(truth.shape[1]),
        "variance_scaling_factor": tau,
        "pva_before": predictive_variance_adequacy(raw_z),
        "pva_after": predictive_variance_adequacy(raw_z / tau),
        "sup_score_median": float(np.median(scores)),
        "sup_score_max": float(scores.max()),
        "bands": per_alpha,
        "scores": scores,
        "scaled_sigma": scaled_sigma,
        "standardized": raw_z / tau,
    }


def band_examples(
    signal: str,
    jobs: list[str],
    u_grid: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    record: dict[str, Any],
    alpha: float,
) -> pd.DataFrame:
    """Three curves with their bands, written out so the report figure only reads.

    The three are chosen by their own sup norm score: the median run, the 90th percentile run,
    and the worst one. Picking the worst deliberately is the point. A band figure that showed
    three comfortable curves would be a decoration; the run the model handles worst is the one
    a reader needs to see, and the simultaneous band is only worth claiming if it contains it.
    """
    from scipy import stats

    scores = np.asarray(record["scores"], dtype=float)
    scale = record["bands"][f"{alpha:g}"]["band_scale"]
    gaussian = float(stats.norm.ppf(1.0 - alpha / 2.0))
    order = np.argsort(scores)
    chosen = {
        "median": int(order[order.size // 2]),
        "p90": int(order[int(0.9 * (order.size - 1))]),
        "worst": int(order[-1]),
    }
    frames = []
    for label, index in chosen.items():
        # The honest before: the pointwise Gaussian interval the uncalibrated model would
        # have given at the same nominal level. It is pointwise rather than simultaneous and
        # it trusts the Gaussian, which is exactly what the comparison is about.
        half_before = gaussian * sigma[index]
        half_after = scale * np.asarray(record["scaled_sigma"], dtype=float)[index]
        frames.append(
            pd.DataFrame(
                {
                    "signal": signal,
                    "example": label,
                    "job": jobs[index],
                    "sup_score": float(scores[index]),
                    "u_mm": u_grid,
                    "truth": truth[index],
                    "loo_mean": mean[index],
                    "sigma_raw": sigma[index],
                    "sigma_scaled": np.asarray(record["scaled_sigma"], dtype=float)[index],
                    "lower_gaussian": mean[index] - half_before,
                    "upper_gaussian": mean[index] + half_before,
                    "lower": mean[index] - half_after,
                    "upper": mean[index] + half_after,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def pit_by_abscissa(standardized: np.ndarray, n_bins: int = PIT_BINS) -> np.ndarray:
    """Probability integral transform histogram at every abscissa, as a ``(n_bins, n_grid)``.

    Column ``u`` is the density of ``Phi(z_i(u))`` over the training curves, normalized so a
    calibrated model gives a flat column at ``1 / n_bins``. Rendered as a heatmap along the
    displacement axis, which is where build spec 11.4 expects the softening branch to show any
    break in calibration.
    """
    from scipy import stats

    pit = stats.norm.cdf(np.asarray(standardized, dtype=float))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.empty((n_bins, pit.shape[1]))
    for column in range(pit.shape[1]):
        counts, _ = np.histogram(pit[:, column], bins=edges)
        out[:, column] = counts / pit.shape[0]
    return out


def pit_outer_mass(
    standardized: np.ndarray, u_grid: np.ndarray, u_peak: np.ndarray
) -> dict[str, float]:
    """The gate statistic of build spec 11.5, made into a number.

    The fraction of PIT values landing in the outer two deciles, over the softening branch of
    every curve, meaning every abscissa past that curve's own displacement at peak. A
    calibrated predictive distribution puts ``PIT_OUTER_MASS_NOMINAL`` there. A U shaped
    histogram is one with too much mass at both ends, so this statistic rising above nominal is
    the U appearing, and the split between the two tails says whether it is a U or a skew.
    """
    from scipy import stats

    z = np.asarray(standardized, dtype=float)
    grid = np.asarray(u_grid, dtype=float)
    peak = np.asarray(u_peak, dtype=float)
    if z.shape[0] != peak.size or z.shape[1] != grid.size:
        raise ValueError(
            f"the standardized residuals are {z.shape} against {peak.size} peak displacements "
            f"and {grid.size} abscissae."
        )
    softening = grid[None, :] > peak[:, None]
    if not softening.any():
        raise ValueError("no abscissa lies past a peak, so there is no softening branch.")
    pit = stats.norm.cdf(z)[softening]
    below = float(np.mean(pit < 0.1))
    above = float(np.mean(pit > 0.9))
    return {
        "n_values": int(pit.size),
        "outer_mass": below + above,
        "lower_tail_mass": below,
        "upper_tail_mass": above,
        "nominal": PIT_OUTER_MASS_NOMINAL,
        "mean_pit": float(np.mean(pit)),
    }


def modulation_cross_check(
    surrogate: SurrogateModel, X: np.ndarray, n_draws: int, seed_sequence: np.random.SeedSequence
) -> dict[str, float]:
    """How much of the curve's spread the linear propagation leaves out.

    The band modulation is the linearly propagated amplitude variance plus the truncation
    residual, which by construction carries no phase or displacement uncertainty: those enter
    the reconstruction nonlinearly. Drawing ``K_posterior_draws`` realizations through the full
    nonlinear reconstruction and comparing their pointwise spread against the propagated one
    says how large that omission is. It is a measurement, not a correction: nothing here is
    multiplied into the band (ground rule 4). The conformal quantile absorbs the difference,
    which is exactly why the guarantee does not depend on the modulation being right.
    """
    rows = np.linspace(0, X.shape[0] - 1, MODULATION_CHECK_CURVES, dtype=int)
    query = X[rows]
    draws = surrogate.draw_curves(query, n_draws, seed_sequence)
    sampled = draws.std(axis=1, ddof=1)
    propagated = surrogate.predict_curve(query).force_std()
    ratio = sampled / propagated
    return {
        "n_curves": int(rows.size),
        "n_draws": int(n_draws),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def evaluate_gate(
    scalar: dict[str, dict[str, Any]],
    functional: dict[str, dict[str, Any]],
    pit: dict[str, dict[str, Any]],
    headline: list[str],
) -> dict[str, Any]:
    """Build spec 11.5, machine checked, with every criterion recorded as it was measured."""
    key = f"{GATE_ALPHA:g}"
    nominal = 1.0 - GATE_ALPHA
    checks: list[dict[str, Any]] = []
    for signal in CURVE_SIGNALS:
        band = functional[signal]["bands"][key]
        checks.append(
            {
                "criterion": f"simultaneous {100 * nominal:.0f} percent band, {signal} curves",
                "measured": band["empirical"],
                "wilson": [band["wilson_low"], band["wilson_high"]],
                "passed": bool(band["wilson_low"] <= nominal <= band["wilson_high"]),
            }
        )
    for target in headline:
        band = scalar[target]["jackknife_plus"][key]
        checks.append(
            {
                "criterion": f"jackknife+ {100 * nominal:.0f} percent interval, {target}",
                "measured": band["empirical"],
                "wilson": [band["wilson_low"], band["wilson_high"]],
                "passed": bool(band["wilson_low"] <= nominal <= band["wilson_high"]),
            }
        )
    outer = pit[SIGNAL_FORCE]["after"]["outer_mass"]
    checks.append(
        {
            "criterion": (
                "PIT outer decile mass on the softening branch at or below "
                f"{GATE_PIT_OUTER_MASS_MAX:.2f}"
            ),
            "measured": outer,
            "wilson": None,
            "passed": bool(outer <= GATE_PIT_OUTER_MASS_MAX),
        }
    )
    failing = [check["criterion"] for check in checks if not check["passed"]]
    return {
        "passed": not failing,
        "alpha": GATE_ALPHA,
        "wilson_level": WILSON_LEVEL,
        "pit_outer_mass_max": GATE_PIT_OUTER_MASS_MAX,
        "checks": checks,
        "failing": failing,
    }


def enforce_gate(gate: dict[str, Any], directory: Path) -> None:
    """Stop the stage when the gate of build spec 11.5 failed, naming what failed.

    The diagnostics stay on disk so the failure can be read, but the caller writes no manifest
    after this raises, which is the point: a failed gate must never become a cache hit, and
    build spec 11.5 forbids any propagated number until it passes. Separate from
    :func:`evaluate_gate` so the failure path itself is testable rather than only its verdict.
    """
    if gate["passed"]:
        return
    raise CalibrationGateFailed(
        "the calibration gate of build spec 11.5 failed on: "
        + "; ".join(gate["failing"])
        + f". Diagnostics were written to {directory} and no manifest was recorded, so the "
        "stage will rerun rather than serve this result. Build spec 11.5: the fix is model "
        "revision, never band styling."
    )


# ---------------------------------------------------------------------------
# Generated fragments
# ---------------------------------------------------------------------------


#: Presentation units for the interval width column, so a table of five different physical
#: quantities reads in the units the report uses everywhere else (kN, mm, kN/mm, J).
WIDTH_UNITS: dict[str, tuple[str, float]] = {
    "P_max_N": ("kN", 1.0e-3),
    "u_peak_mm": ("mm", 1.0),
    "k0_N_per_mm": ("kN/mm", 1.0e-3),
    "E_abs_Nmm": ("J", 1.0e-3),
    "P_residual_N": ("kN", 1.0e-3),
    "P_knee_N": ("kN", 1.0e-3),
    "u_knee_mm": ("mm", 1.0),
    "u_damage_half_sat_mm": ("mm", 1.0),
    "softening_ratio": ("-", 1.0),
    "damage_at_10mm": ("-", 1.0),
    "arclength_total": ("-", 1.0),
}


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "{--}"
    # A rounded negative zero prints as "-0.000", which reads as a measurement rather than as
    # the zero it is.
    if abs(value) < 0.5 * 10.0 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def build_conformal_table(
    scalar: dict[str, dict[str, Any]],
    cv: dict[str, dict[str, Any]],
    headline: list[str],
    alphas: list[float],
) -> str:
    """The jackknife+ table of build spec 11.1, per QoI, with the CV+ cross check beside it."""
    lines = [
        "% Generated by the calibrate stage (ufem.calibrate). Do not edit.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Quantity & Nominal & Jackknife+ & 95\,\% Wilson & CV+ & Median width \\",
        r"\midrule",
    ]
    for target in headline:
        unit, scale = WIDTH_UNITS[target]
        label = f"{QOI_LABELS.get(target, target)} [{unit}]"
        for position, alpha in enumerate(sorted(alphas, reverse=True)):
            key = f"{alpha:g}"
            row = scalar[target]["jackknife_plus"][key]
            cross = cv[target]["cv_plus"][key]["rank_coverage"]
            name = label if position == 0 else ""
            lines.append(
                f"{name} & {100 * row['nominal']:.0f}\\,\\% & {_fmt(row['empirical'])} & "
                f"[{_fmt(row['wilson_low'])}, {_fmt(row['wilson_high'])}] & "
                f"{_fmt(cross['empirical'])} & {_fmt(row['median_width'] * scale, 2)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_diagnostics_table(
    scalar: dict[str, dict[str, Any]],
    functional: dict[str, dict[str, Any]],
    headline: list[str],
) -> str:
    """Variance scaling, adequacy, CRPS skill and NLPD, paired as build spec 11.4 asks."""
    lines = [
        "% Generated by the calibrate stage (ufem.calibrate). Do not edit.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Quantity & Scaling $\tau$ & PVA before & PVA after & CRPS skill & NLPD after \\",
        r"\midrule",
    ]
    for target in headline:
        record = scalar[target]
        lines.append(
            f"{QOI_LABELS.get(target, target)} & {_fmt(record['variance_scaling_factor'])} & "
            f"{_fmt(record['pva_before'])} & {_fmt(record['pva_after'])} & "
            f"{_fmt(record['crps_skill_after'])} & {_fmt(record['nlpd_after'], 2)} \\\\"
        )
    lines.append(r"\midrule")
    for signal in CURVE_SIGNALS:
        record = functional[signal]
        lines.append(
            f"{signal.capitalize()} curve & {_fmt(record['variance_scaling_factor'])} & "
            f"{_fmt(record['pva_before'])} & {_fmt(record['pva_after'])} & {{--}} & {{--}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_markdown_summary(payload: dict[str, Any]) -> str:
    """The stage's own readable summary, regenerated on every run."""
    out: list[str] = []
    add = out.append
    gate = payload["gate"]
    add("# Calibration summary")
    add("")
    add(
        f"Generated by the `calibrate` stage for config `{payload['config_sha256'][:12]}`, "
        f"over {payload['context']['n_runs']} runs."
    )
    add("")
    add(f"**Gate (build spec 11.5): {'PASSED' if gate['passed'] else 'FAILED'}.**")
    add("")
    add("| Criterion | Measured | 95 percent Wilson | Verdict |")
    add("|---|---|---|---|")
    for check in gate["checks"]:
        interval = (
            f"[{check['wilson'][0]:.3f}, {check['wilson'][1]:.3f}]"
            if check["wilson"] is not None
            else "not applicable"
        )
        add(
            f"| {check['criterion']} | {check['measured']:.4f} | {interval} | "
            f"{'pass' if check['passed'] else 'FAIL'} |"
        )
    add("")
    add("## Out of fold variance scaling factors")
    add("")
    add("| Target | tau | PVA before | PVA after (leave one out tau) |")
    add("|---|---|---|---|")
    for name, record in payload["scalar"].items():
        add(
            f"| {name} | {record['variance_scaling_factor']:.4f} | "
            f"{record['pva_before']:+.4f} | {record['pva_after']:+.4f} |"
        )
    for signal in CURVE_SIGNALS:
        record = payload["functional"][signal]
        add(
            f"| {signal} curve | {record['variance_scaling_factor']:.4f} | "
            f"{record['pva_before']:+.4f} | {record['pva_after']:+.4f} |"
        )
    add("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, dict[str, str]]:
    artifact_root = root / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, config_sha256)
    surrogate_dir = stage_dir(artifact_root, SURROGATE_STAGE, config_sha256)
    validate_dir = stage_dir(artifact_root, VALIDATE_STAGE, config_sha256)
    hashes: dict[str, str] = {}
    for directory, name, stage in (
        (grid_dir, RF2_GRID_PARQUET, GRID_STAGE),
        (grid_dir, DAMAGE_GRID_PARQUET, GRID_STAGE),
        (grid_dir, QOI_PARQUET, GRID_STAGE),
        (register_dir, LANDMARKS_PARQUET, REGISTER_STAGE),
        (surrogate_dir, "surrogate.json", SURROGATE_STAGE),
        (surrogate_dir, "gp_state.npy", SURROGATE_STAGE),
        (validate_dir, "baselines.json", VALIDATE_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the calibrate stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first."
            )
        hashes[name] = sha256_file(path)
    return grid_dir, register_dir, hashes


def declared_input_hashes(
    repo_root: Path | str, config: Config, config_sha256: str
) -> dict[str, str]:
    """Hash this stage's declared inputs as they are on disk right now (see ``ufem.runner``)."""
    return _load_inputs(Path(repo_root), config, config_sha256)[-1]


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the calibrate stage and return its artifact directory."""
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    grid_dir, register_dir, input_hashes = _load_inputs(root, config, config_sha256)
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
    surrogate = SurrogateModel.load(artifact_root, config_sha256)
    settings = GPSettings.from_config(config)
    alphas = list(config.pipeline.conformal.alphas)
    headline = list(config.pipeline.validation.headline_qoi)

    scalar_targets: dict[str, np.ndarray] = {
        name: qoi[name].to_numpy(dtype=float) for name in SCALAR_QOI
    }
    for name in LANDMARK_QOI:
        scalar_targets[name] = landmarks[name].to_numpy(dtype=float)

    scalar_started = _time.perf_counter()
    scalar, scalar_frame = scalar_calibration(surrogate, scalar_targets, alphas)
    scalar_seconds = _time.perf_counter() - scalar_started

    curve_started = _time.perf_counter()
    loo = leave_one_out_curves(surrogate)
    u_grid = loo["u_grid"]
    observed = {SIGNAL_FORCE: force, SIGNAL_DAMAGE: damage}
    functional: dict[str, dict[str, Any]] = {}
    pit: dict[str, dict[str, Any]] = {}
    curve_rows: list[pd.DataFrame] = []
    pit_rows: list[pd.DataFrame] = []
    band_rows: list[pd.DataFrame] = []
    u_peak = qoi["u_peak_mm"].to_numpy(dtype=float)
    for signal in CURVE_SIGNALS:
        mask = informative_abscissae(observed[signal])
        u_signal = u_grid[mask]
        truth_signal = observed[signal][:, mask]
        mean_signal = loo[f"{signal}_mean"][:, mask]
        sigma_signal = loo[f"{signal}_sigma"][:, mask]
        record = functional_calibration(truth_signal, mean_signal, sigma_signal, alphas)
        raw_z = (truth_signal - mean_signal) / sigma_signal
        pit[signal] = {
            "before": pit_outer_mass(raw_z, u_signal, u_peak),
            "after": pit_outer_mass(record["standardized"], u_signal, u_peak),
        }
        for label, values in (("before", raw_z), ("after", record["standardized"])):
            histogram = pit_by_abscissa(values)
            frame = pd.DataFrame(
                histogram.T,
                columns=[f"bin_{index + 1}" for index in range(histogram.shape[0])],
            )
            frame.insert(0, "u_mm", u_signal)
            frame.insert(0, "stage", label)
            frame.insert(0, "signal", signal)
            pit_rows.append(frame)
        curve_rows.append(
            pd.DataFrame(
                {
                    "signal": signal,
                    "job": jobs,
                    "sup_score": record["scores"],
                    "median_sigma": np.median(record["scaled_sigma"], axis=1),
                    **{
                        f"covered_{alpha:g}": leave_one_out_coverage(record["scores"], alpha)
                        for alpha in alphas
                    },
                }
            )
        )
        band_rows.append(
            band_examples(
                signal,
                jobs,
                u_signal,
                truth_signal,
                mean_signal,
                sigma_signal,
                record,
                GATE_ALPHA,
            )
        )
        functional[signal] = {
            key: value
            for key, value in record.items()
            if key not in ("scores", "scaled_sigma", "standardized")
        }
        functional[signal]["domain"] = {
            "n_abscissae": int(mask.sum()),
            "n_excluded": int((~mask).sum()),
            "u_min_mm": float(u_signal.min()),
            "u_max_mm": float(u_signal.max()),
            "excluded_u_mm": u_grid[~mask].tolist(),
            "rule": (
                "abscissae where all 198 observed curves take the same value carry no "
                "calibration information and are excluded from the supremum; no variance is "
                "floored to keep them (ground rule 4)"
            ),
        }
    curve_seconds = _time.perf_counter() - curve_started

    check_seed, = np.random.SeedSequence(config.pipeline.seed_entropy).spawn(1)
    modulation = modulation_cross_check(
        surrogate, X, int(config.pipeline.conformal.K_posterior_draws), check_seed
    )

    cv_started = _time.perf_counter()
    cv = cv_plus_calibration(X, scalar_targets, jobs, config, settings, alphas)
    cv_seconds = _time.perf_counter() - cv_started

    gate = evaluate_gate(scalar, functional, pit, headline)

    directory = stage_dir(artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_sha256": config_sha256,
        "context": {
            "n_runs": len(jobs),
            "n_grid": int(u_grid.size),
            "alphas": alphas,
            "headline_qoi": headline,
            "wilson_level": WILSON_LEVEL,
            "coverage_levels": list(COVERAGE_LEVELS),
            "pit_bins": PIT_BINS,
        },
        "scalar": scalar,
        "cv_plus": cv,
        "functional": functional,
        "pit_softening": pit,
        "modulation_cross_check": modulation,
        "gate": gate,
        "loo_approximation": (
            "closed form leave one out (Dubrule 1983; Rasmussen and Williams 5.4.2) on the "
            "score and scalar processes at hyperparameters fitted on all runs, inside a "
            "reduction basis and registration reference fitted on all runs; the per fold refit "
            "cross check is the 10 fold CV+ in this artifact and the fully refitted curve fold "
            "harness in the validate stage"
        ),
    }
    outputs: list[Path] = []
    for name, text in (
        (CALIBRATION_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (CONFORMAL_TEX, build_conformal_table(scalar, cv, headline, alphas)),
        (DIAGNOSTICS_TEX, build_diagnostics_table(scalar, functional, headline)),
        (CALIBRATION_MD, build_markdown_summary(payload)),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8", newline="\n")
        outputs.append(path)
    sweep = pd.concat(
        [
            pd.DataFrame(record["coverage_sweep"]).assign(target=name)
            for name, record in scalar.items()
        ],
        ignore_index=True,
    )
    for frame, name in (
        (scalar_frame, SCALAR_CONFORMAL_PARQUET),
        (pd.concat(curve_rows, ignore_index=True), CURVE_CONFORMAL_PARQUET),
        (pd.concat(pit_rows, ignore_index=True), PIT_PARQUET),
        (pd.concat(band_rows, ignore_index=True), BAND_EXAMPLES_PARQUET),
        (sweep, COVERAGE_SWEEP_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    print(
        f"[calibrate] {len(jobs)} runs; scalar jackknife+ in {scalar_seconds:.1f} s, functional "
        f"bands in {curve_seconds:.1f} s, 10 fold CV+ refits in {cv_seconds:.1f} s"
    )
    for check in gate["checks"]:
        print(f"[calibrate]   {'pass' if check['passed'] else 'FAIL'}  {check['criterion']}: "
              f"{check['measured']:.4f}")
    enforce_gate(gate, directory)

    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "scalar_wall_time_s": scalar_seconds,
        "functional_wall_time_s": curve_seconds,
        "cv_plus_wall_time_s": cv_seconds,
        "n_runs": len(jobs),
        "gate": gate,
        "variance_scaling_factors": {
            **{name: record["variance_scaling_factor"] for name, record in scalar.items()},
            **{
                f"{signal}_curve": functional[signal]["variance_scaling_factor"]
                for signal in CURVE_SIGNALS
            },
        },
        "functional_coverage": {
            signal: functional[signal]["bands"][f"{GATE_ALPHA:g}"]["empirical"]
            for signal in CURVE_SIGNALS
        },
        "modulation_cross_check": modulation,
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
    print("[calibrate] gate PASSED")
    return directory
