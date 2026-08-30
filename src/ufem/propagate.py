"""Monte Carlo propagation, limit states, and reliability with honest bounds.

Build spec section 13. The predecessor advertised 15,000 propagated samples and actually ran
1,000 inputs times 15 posterior draws through a surrogate whose predicted peak force
correlated with the concrete strength at r = -0.006, then published failure probabilities
against a damage threshold above the reachable maximum. This stage is the answer to all three
halves of that: the sample size is what it says it is, the two uncertainty layers are kept
apart and labeled, and every probability carries the three qualifications that decide whether
it means anything.

**The two layers (build spec 13.1).** They answer different questions and are never added.

* *Aleatory*: the input distributions of build spec 9.1 pushed through the mean surrogate,
  ``mc.n_samples`` draws. This is the spread the beam population would show if the surrogate
  were exact. Every quantile in the report is this layer.
* *Epistemic*: the surrogate's own uncertainty, by posterior sampling per input draw from the
  calibrated predictive distribution at that draw, on a seeded subsample of the aleatory
  sample. This layer is what the calibration stage earned the right to state: the predictive
  standard deviation is the fitted one multiplied by the variance scaling factor the
  jackknife+ calibration measured, so it is a width that was checked against held out data
  rather than a width the model asserted.

The epistemic layer draws independently at each input draw, which treats the surrogate error
as uncorrelated between two nearby designs. That understates its effect on a population
quantile, where a correlated error would move the whole quantile rather than blurring it. The
statement that does not depend on it is the conservative bound below, which is a worst case
over the calibrated band and holds whatever the correlation is; the report says so.

**The bound (build spec 13.2).** Every failure probability comes with a second number obtained
by counting a failure whenever the calibrated 90 percent jackknife+ interval at that input draw
crosses the threshold. That is the honest answer to surrogate error near a limit state: the
point estimate asks where the mean prediction falls, the bound asks where the prediction could
fall without contradicting the calibration.

**What is not claimed (build spec 13.3).** With 198 training runs, no failure probability below
about 1e-4 is resolvable, whatever the Monte Carlo sample size makes it possible to print. The
floor is stated in the artifact, in the tables, and in the report.

**The inherited caveat.** Phase P6 measured, model free, that this campaign's response varies
between nearest neighbour designs by a median share of the response standard deviation that is
recorded in the sensitivity artifact. That roughness is not in any of the intervals here: it is
error the surrogate family cannot express, so it is carried as a stated unmodeled error beside
every probability rather than folded into one. The sentence is generated from the P6
measurement, so it cannot drift from it.

**Batching.** The Monte Carlo through the Gaussian processes is matrix algebra and is done as
such. One evaluation of the prior cross covariance between a chunk of query points and the 198
training points yields, for that chunk, the posterior mean, the posterior variance, and all 198
leave one out cross predictions the jackknife+ interval needs. Chunks are
:data:`PREDICTION_CHUNK` rows for the mean and variance and :data:`BAND_CHUNK` rows for the
band, which holds six arrays of chunk by 198 doubles at once; at the committed sizes the peak
working set of the stage is under 100 MB and does not grow with ``mc.n_samples``.

Units: force in N, displacement in mm, strength in MPa, everywhere.
"""

from __future__ import annotations

import json
import math
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

from ufem.analytic import (
    DEFAULT_BEAM,
    MODEL_ERROR_FRACTION,
    cracked_second_moment_mm4,
    cracking_load_N,
    cross_check,
    empirical_log_elasticities,
    log_elasticities,
    peak_load_N,
    tip_stiffness_N_per_mm,
)
from ufem.calibrate import CALIBRATION_JSON, SCALAR_CONFORMAL_PARQUET, wilson_interval
from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import (
    FEATURE_ORDER,
    Config,
    derived_E,
    features,
    input_distributions,
)
from ufem.grid import QOI_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.sensitivity import SENSITIVITY_JSON
from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE
from ufem.surrogate import (
    LANDMARK_QOI,
    SCALAR_QOI,
    SURROGATE_JSON,
    FittedGP,
    Standardizer,
    SurrogateModel,
    configure_torch,
)
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.validate import QOI_LABELS
from ufem.validity import in_validity_domain, load_validity_domain

STAGE_NAME = "propagate"

#: Artifacts. One JSON carrying every measurement, four tabular products, three report
#: fragments, and the human readable summary.
PROPAGATION_JSON = "propagation.json"
QUANTILES_PARQUET = "qoi_quantiles.parquet"
DENSITY_PARQUET = "qoi_density.parquet"
RELIABILITY_PARQUET = "reliability.parquet"
CURVE_ENVELOPE_PARQUET = "curve_envelope.parquet"
ANALYTIC_PARQUET = "analytic_peak_load.parquet"
RELIABILITY_TEX = "reliability.tex"
QUANTILES_TEX = "propagated_quantiles.tex"
ANALYTIC_TEX = "analytic_cross_check.tex"
PROPAGATION_MD = "propagation_summary.md"

#: The quantiles build spec 13.1 names, as probabilities.
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

#: The band level of build spec 13.2. It is 90 percent because that is the level the
#: calibration gate of build spec 11.5 measured coverage at; a bound read off a band nobody
#: checked would be decoration.
BAND_ALPHA = 0.10

#: The characteristic value logic of build spec 13.2: a characteristic capacity is the 5th
#: percentile of the capacity distribution.
CHARACTERISTIC_LEVEL = 0.05

#: Relative gap at which the committed peak load threshold and the recomputed characteristic
#: value are called materially different, which is the trigger build spec 13.2 gives for
#: revising the configured number. One percent of a capacity is inside the rounding a declared
#: threshold deserves; more than that is a different limit state.
CHARACTERISTIC_TOLERANCE = 0.01

#: Build spec 13.3. Not a Monte Carlo limit: 1e5 samples resolve 1e-5 per draw. It is the
#: limit the training set imposes, and rarer events need the active learning of Track B.
RESOLVABLE_PF_FLOOR = 1.0e-4

#: Rows per chunk. The mean and variance path holds three chunk by 198 arrays, the band path
#: six, so the band gets the smaller chunk. Both are part of the artifact contract: changing
#: either changes the last bits of the stage's outputs through the order of the reductions.
PREDICTION_CHUNK = 20000
BAND_CHUNK = 8192

#: Points in each reported density estimate, and the plotting range in sample quantiles. The
#: range is quantile based rather than min to max so one extreme draw cannot flatten a figure.
DENSITY_GRID_POINTS = 256
DENSITY_RANGE = (0.001, 0.999)
DENSITY_PAD = 0.05

#: The envelopes of build spec 13.1: a median, a 50 percent envelope, a 90 percent envelope.
ENVELOPE_LEVELS: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)

#: Display scale and unit per reported quantity. The report reads forces in kN and energies in
#: J, as every other table in this project does.
QOI_DISPLAY: dict[str, tuple[str, float]] = {
    "P_max_N": ("kN", 1.0e-3),
    "u_peak_mm": ("mm", 1.0),
    "k0_N_per_mm": ("kN/mm", 1.0e-3),
    "E_abs_Nmm": ("J", 1.0e-3),
    "P_residual_N": ("kN", 1.0e-3),
    "softening_ratio": ("-", 1.0),
    "u_damage_half_sat_mm": ("mm", 1.0),
    "damage_at_10mm": ("-", 1.0),
    "u_knee_mm": ("mm", 1.0),
    "P_knee_N": ("kN", 1.0e-3),
    "arclength_total": ("-", 1.0),
}


class PropagationInputMissing(FileNotFoundError):
    """An upstream artifact this stage needs is absent, and nothing is invented."""


@dataclass(frozen=True)
class LimitState:
    """One declared limit state: which quantity, which side of which configured threshold.

    ``config_field`` is the name in the ``limit_states`` block of ``configs/pipeline.yaml``, so
    the threshold is read from the config rather than restated here. What lives here is the
    part that is a modeling statement rather than a number: which quantity of interest the
    threshold applies to, which side of it counts as a failure, and the engineering reason the
    limit state exists at all, which build spec 13.2 requires the report to give.
    """

    config_field: str
    target: str
    direction: str
    label: str
    short_label: str
    justification: str


#: The three limit states of build spec 13.2, in report order.
LIMIT_STATES: tuple[LimitState, ...] = (
    LimitState(
        config_field="peak_load_below_N",
        target="P_max_N",
        direction="below",
        label="Peak load below the characteristic value",
        short_label="Peak load below characteristic",
        justification=(
            "the characteristic capacity is the 5th percentile of the capacity distribution, "
            "so this limit state asks how often the realized member falls below the value a "
            "design would have been carried out with. The predecessor's two sigma threshold is "
            "recorded in build spec 13.2 as the counterexample: a multiple of a standard "
            "deviation is a statement about a distribution's shape, not about a design value"
        ),
    ),
    LimitState(
        config_field="residual_ratio_below",
        target="softening_ratio",
        direction="below",
        label="Residual capacity ratio at 20 mm below one half",
        short_label="Residual ratio below one half",
        justification=(
            "a member that retains less than half its peak load at 20 mm of imposed "
            "displacement has lost the post peak reserve that makes a warning of collapse "
            "possible. The ratio is used rather than the residual load itself so the limit "
            "state is about ductility rather than about strength twice"
        ),
    ),
    LimitState(
        config_field="damage_at_10mm_above",
        target="damage_at_10mm",
        direction="above",
        label="Compressive damage at 10 mm above 0.93",
        short_label="Damage at 10 mm above threshold",
        justification=(
            "the damage scalar saturates at the material table cap of 0.947 in every run of "
            "this campaign, so a threshold near that cap asks whether saturation is reached "
            "early, at half the imposed displacement range, rather than whether it is reached "
            "at all. Build spec 5.6 records the predecessor's threshold of 0.9591, which is "
            "above the reachable maximum and therefore has a failure probability of exactly "
            "zero for reasons that have nothing to do with the beam"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def draw_inputs(
    config: Config, n_samples: int, seed_sequence: np.random.SeedSequence
) -> np.ndarray:
    """``n_samples`` independent draws of the three inputs, in the feature contract order.

    Inverse transform sampling on uniforms, with one spawned child generator per input so a
    column's values depend on that column's seed and on nothing else: adding a fourth input in
    Track B, or reordering the report, cannot silently reshuffle the strength column. The
    distributions come from :func:`ufem.config.input_distributions` and are constructed nowhere
    else (binding law 2).

    Plain Monte Carlo rather than a low discrepancy sequence, deliberately. Build spec 13.2
    asks for a binomial standard error on every failure probability, and that standard error is
    a statement about independent draws; a randomized quasi Monte Carlo estimate would need its
    own replication based error and would make the two numbers in the same table mean different
    things.
    """
    if n_samples < 1:
        raise ValueError(f"the aleatory layer needs at least one draw, got {n_samples}.")
    distributions = input_distributions(config)
    children = seed_sequence.spawn(len(FEATURE_ORDER))
    columns = []
    for name, child in zip(FEATURE_ORDER, children):
        generator = np.random.default_rng(child)
        columns.append(
            np.asarray(distributions[name].ppf(generator.random(n_samples)), dtype=float)
        )
    return np.column_stack(columns)


def subsample_indices(
    n_samples: int, n_subsample: int, seed_sequence: np.random.SeedSequence
) -> np.ndarray:
    """A seeded subsample of the aleatory row indices, sorted, without replacement."""
    if n_subsample > n_samples:
        raise ValueError(
            f"cannot subsample {n_subsample} rows from an aleatory sample of {n_samples}."
        )
    generator = np.random.default_rng(seed_sequence)
    return np.sort(generator.choice(n_samples, size=n_subsample, replace=False))


# ---------------------------------------------------------------------------
# The Gaussian process algebra, batched
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorPieces:
    """Everything about one fitted process that does not depend on the query point.

    Assembled once per target and reused across every chunk, which is what turns a 1e5 sample
    propagation into a few matrix products. ``weights`` is the leave one out weight matrix of
    :meth:`ufem.surrogate.FittedGP.leave_one_out_cross_predictions`, whose column ``i`` is the
    weight vector of the model that left training point ``i`` out; the same identity gives the
    cross predictions at a training point and at a query point, and only the cross covariance
    block in front of it changes.
    """

    name: str
    constant_mean: float
    noise: float
    prior_variance: float
    alpha: np.ndarray
    inverse: np.ndarray
    weights: np.ndarray
    inverse_diagonal: np.ndarray


def posterior_pieces(gp: FittedGP) -> PosteriorPieces:
    """Precompute the query independent part of one process's posterior."""
    covariance = gp.kernel_matrix()
    inverse = np.linalg.inv(covariance)
    diagonal = np.diag(inverse)
    if np.any(diagonal <= 0.0):
        raise ValueError(
            f"target {gp.name!r} has a non positive diagonal in the inverse covariance, so "
            "neither the posterior nor its leave one out family is defined here."
        )
    constant = gp.constant_mean()
    alpha = inverse @ (gp.train_y - constant)
    weights = alpha[:, None] - inverse * (alpha / diagonal)[None, :]
    return PosteriorPieces(
        name=gp.name,
        constant_mean=constant,
        noise=gp.noise(),
        prior_variance=gp.prior_variance(),
        alpha=alpha,
        inverse=inverse,
        weights=weights,
        inverse_diagonal=diagonal,
    )


def predict_mean_and_variance(
    gp: FittedGP,
    standardizer: Standardizer,
    standardized_X: np.ndarray,
    pieces: PosteriorPieces,
    chunk: int = PREDICTION_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and predictive variance at every row, in the target's own units.

    The same numbers :meth:`ufem.surrogate.FittedGP.predict` returns, computed from one cross
    covariance block per chunk instead of one GPyTorch prediction per chunk. The equality is a
    test, not a claim: ``tests/test_propagate.py`` asserts agreement to 1e-9 on the training
    design and on random queries, so this path is checked against the library rather than
    trusted because the algebra looks right.

    The variance includes the fitted observation noise, because the quantity being propagated
    is what a new run of the finite element model would produce, not the latent mean of an
    ensemble of them.
    """
    matrix = np.atleast_2d(np.asarray(standardized_X, dtype=float))
    n_rows = matrix.shape[0]
    mean = np.empty(n_rows)
    variance = np.empty(n_rows)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = gp.cross_covariance(matrix[start:stop])
        projected = block @ pieces.inverse
        mean[start:stop] = pieces.constant_mean + block @ pieces.alpha
        variance[start:stop] = (
            pieces.prior_variance + pieces.noise - np.einsum("ij,ij->i", projected, block)
        )
    if np.any(variance <= 0.0):
        raise ValueError(
            f"target {pieces.name!r} produced a non positive predictive variance at "
            f"{int((variance <= 0.0).sum())} of {n_rows} query points. Nothing is clipped "
            "here (ground rule 4); a negative variance means the covariance is numerically "
            "indefinite at these hyperparameters."
        )
    return (
        standardizer.inverse_mean(mean.reshape(-1, 1)).ravel(),
        standardizer.inverse_variance(variance.reshape(-1, 1)).ravel(),
    )


def jackknife_plus_rows(
    cross_mean: np.ndarray, cross_sigma: np.ndarray, scores: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """The jackknife+ interval for many query points at once, without a Python loop.

    The construction is exactly :func:`ufem.calibrate.jackknife_plus_intervals`: with ``m``
    leave one out models and ``k = ceil((1 - alpha)(m + 1))``, the upper end is the ``k`` th
    smallest of ``mu_-i + R_i sigma_-i`` and the lower end the ``(m + 1 - k)`` th smallest of
    ``mu_-i - R_i sigma_-i``. The difference is arithmetic only: a partial sort along the model
    axis rather than a full sort per query, which is what makes 1e5 queries affordable. A test
    asserts bitwise agreement with the calibration stage's implementation, which stays the
    reference, on the same inputs.

    Every model votes at a query point. The calibration stage masks the diagonal because there
    it is evaluating the band at a training point, where that point's own leave one out model
    must not vote; a Monte Carlo draw is not a training point and has no diagonal to mask.
    """
    mu = np.atleast_2d(np.asarray(cross_mean, dtype=float))
    sigma = np.atleast_2d(np.asarray(cross_sigma, dtype=float))
    r = np.asarray(scores, dtype=float).ravel()
    if mu.shape != sigma.shape or mu.shape[1] != r.size:
        raise ValueError(
            f"jackknife+ needs matching shapes: means {mu.shape}, sigmas {sigma.shape}, "
            f"scores {r.shape}."
        )
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError(
            "the cross predictions carry a non finite entry, so the interval would be taken "
            "over an ensemble whose size is not what it appears to be."
        )
    m = r.size
    rank = math.ceil((1.0 - alpha) * (m + 1))
    if rank > m:
        raise ValueError(
            f"a {100 * (1 - alpha):.1f} percent jackknife+ interval needs the {rank}th of {m} "
            "leave one out models, which does not exist at this sample size."
        )
    half = r[None, :] * sigma
    upper = np.partition(mu + half, rank - 1, axis=1)[:, rank - 1]
    lower = np.partition(mu - half, m - rank, axis=1)[:, m - rank]
    return lower, upper


def calibrated_band(
    gp: FittedGP,
    standardizer: Standardizer,
    standardized_X: np.ndarray,
    pieces: PosteriorPieces,
    scores: np.ndarray,
    scaling_factor: float,
    alpha: float = BAND_ALPHA,
    chunk: int = BAND_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """The deployed jackknife+ interval at every query row, in the target's own units.

    ``scores`` and ``scaling_factor`` are read from the calibrate stage rather than recomputed:
    they are the calibration, and a second computation of them here would be a second answer to
    a question build spec 11.3 says has one.

    The leave one out cross predictions at a query point come from the same block inverse
    identity the calibration stage uses at a training point. With ``A`` the inverse training
    covariance, ``C`` the cross covariance of the chunk against the training design, and
    ``P = C A``: the means are ``m + C W`` for the weight matrix ``W`` in
    :class:`PosteriorPieces`, and the variance of the model that dropped point ``i`` is the
    full predictive variance plus ``P[:, i]**2 / A_ii``, a correction that is positive because
    dropping a point can only widen a posterior.
    """
    matrix = np.atleast_2d(np.asarray(standardized_X, dtype=float))
    n_rows = matrix.shape[0]
    lower = np.empty(n_rows)
    upper = np.empty(n_rows)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = gp.cross_covariance(matrix[start:stop])
        projected = block @ pieces.inverse
        cross_mean = pieces.constant_mean + block @ pieces.weights
        full_variance = (
            pieces.prior_variance + pieces.noise - np.einsum("ij,ij->i", projected, block)
        )
        cross_variance = full_variance[:, None] + projected**2 / pieces.inverse_diagonal[None, :]
        if np.any(cross_variance <= 0.0):
            raise ValueError(
                f"target {pieces.name!r} produced a non positive leave one out variance at a "
                "query point, which means the training covariance is numerically indefinite."
            )
        shape = cross_mean.shape
        units_mean = standardizer.inverse_mean(cross_mean.reshape(-1, 1)).reshape(shape)
        units_variance = standardizer.inverse_variance(
            cross_variance.reshape(-1, 1)
        ).reshape(shape)
        lower[start:stop], upper[start:stop] = jackknife_plus_rows(
            units_mean, scaling_factor * np.sqrt(units_variance), scores, alpha
        )
    return lower, upper


# ---------------------------------------------------------------------------
# The two layers
# ---------------------------------------------------------------------------


def epistemic_draws(
    mean: np.ndarray,
    sigma: np.ndarray,
    n_draws: int,
    seed_sequence: np.random.SeedSequence,
) -> np.ndarray:
    """``n_draws`` posterior draws at each of the given points, shape ``(n_points, n_draws)``.

    The epistemic layer of build spec 13.1. Each draw is the calibrated predictive distribution
    at that input draw, so ``sigma`` is the fitted predictive standard deviation already
    multiplied by the calibration's variance scaling factor.

    The property that makes this a layer rather than a decoration: a zero ``sigma`` returns the
    mean at every draw, exactly, so the epistemic spread of a surrogate that claims no
    uncertainty is zero rather than a small number nobody chose. That is the test in
    ``tests/test_propagate.py``, and it is reachable through this signature without touching
    anything private.
    """
    center = np.asarray(mean, dtype=float).ravel()
    width = np.asarray(sigma, dtype=float).ravel()
    if center.shape != width.shape:
        raise ValueError(
            f"the epistemic layer needs one standard deviation per point: means {center.shape} "
            f"against sigmas {width.shape}."
        )
    if np.any(width < 0.0):
        raise ValueError("a negative predictive standard deviation is not a distribution.")
    if n_draws < 1:
        raise ValueError(f"the epistemic layer needs at least one draw, got {n_draws}.")
    generator = np.random.default_rng(seed_sequence)
    normal = generator.standard_normal((center.size, int(n_draws)))
    return center[:, None] + width[:, None] * normal


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def quantile_standard_error(level: float, n_samples: int, density: float) -> float:
    """Asymptotic standard error of a sample quantile: ``sqrt(p (1 - p) / n) / f(q_p)``.

    The classical result: the sample quantile is asymptotically normal about the true quantile
    with that standard deviation, so the error of a reported quantile is governed by how thin
    the density is where it sits, not by the sample size alone. A quantile in a flat tail is
    expensive to resolve, and this is the number that says so.

    The density is an argument rather than something estimated inside, so the formula can be
    tested against a case where the density is known exactly.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"a quantile level must lie in (0, 1), got {level}.")
    if n_samples < 1:
        raise ValueError(f"a quantile standard error needs at least one sample, got {n_samples}.")
    if not density > 0.0:
        raise ValueError(
            f"the density at the quantile is {density}, so the asymptotic standard error is "
            "not defined there. A quantile in a region the sample never visits has no "
            "standard error to report."
        )
    return math.sqrt(level * (1.0 - level) / n_samples) / density


def kernel_density(sample: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Gaussian kernel density estimate of ``sample`` evaluated at ``at``.

    Scott's rule for the bandwidth, which is what the estimator's default is; the choice is
    named here rather than inherited silently because the quantile standard errors below divide
    by this number.
    """
    values = np.asarray(sample, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("a density estimate needs at least two samples.")
    return np.asarray(stats.gaussian_kde(values, bw_method="scott")(np.asarray(at, dtype=float)))


def quantile_summary(sample: np.ndarray, levels: tuple[float, ...] = QUANTILE_LEVELS) -> dict:
    """Quantiles with their Monte Carlo standard errors, plus the sample's moments."""
    values = np.asarray(sample, dtype=float).ravel()
    quantiles = np.quantile(values, levels)
    density = kernel_density(values, quantiles)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "cov": float(values.std(ddof=1) / values.mean()) if values.mean() != 0.0 else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
        "levels": [float(level) for level in levels],
        "quantiles": [float(value) for value in quantiles],
        "density_at_quantile": [float(value) for value in density],
        "standard_error": [
            quantile_standard_error(float(level), values.size, float(value))
            for level, value in zip(levels, density)
        ],
    }


def density_curve(sample: np.ndarray, n_points: int = DENSITY_GRID_POINTS) -> tuple:
    """A density estimate on a padded interquantile grid, for the report figures."""
    values = np.asarray(sample, dtype=float).ravel()
    low, high = np.quantile(values, DENSITY_RANGE)
    pad = DENSITY_PAD * (high - low)
    grid = np.linspace(low - pad, high + pad, int(n_points))
    return grid, kernel_density(values, grid)


def failure_mask(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """True where a value violates the limit state."""
    array = np.asarray(values, dtype=float)
    if direction == "below":
        return array < threshold
    if direction == "above":
        return array > threshold
    raise ValueError(f"a limit state direction is 'below' or 'above', got {direction!r}.")


def binomial_standard_error(probability: float, n_samples: int) -> float:
    """``sqrt(p (1 - p) / n)``, the Monte Carlo error of a counted proportion."""
    if n_samples < 1:
        raise ValueError(f"a binomial standard error needs at least one sample, got {n_samples}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"a probability must lie in [0, 1], got {probability}.")
    return math.sqrt(probability * (1.0 - probability) / n_samples)


def limit_state_result(
    state: LimitState,
    threshold: float,
    mean_prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    predictive: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_sigma: np.ndarray,
    inside_domain: np.ndarray,
) -> dict[str, Any]:
    """Every number this project is willing to state about one limit state.

    Four probabilities, and they are four different questions:

    * ``pf_point``: the aleatory layer through the mean surrogate. Build spec 13.2's point
      estimate, with its binomial standard error and a 95 percent Wilson interval, which is the
      interval that still says something when the count is zero.
    * ``pf_conservative``: the bound of build spec 13.2. A draw counts as a failure when its
      calibrated 90 percent interval crosses the threshold, or when the mean prediction fails.
      The union is deliberate: the jackknife+ interval is a quantile over an ensemble rather
      than an interval centered on the full model's mean, so it is not guaranteed to contain
      that mean, and a bound that could fall below its own point estimate would not be a bound.
      How often the union was needed is recorded in ``n_point_outside_band``.
    * ``pf_predictive``: the epistemic layer, the failure probability under the calibrated
      predictive distribution rather than under the mean. Reported with a closed form value
      beside it, ``pf_predictive_closed_form``, which integrates the same Gaussian analytically;
      the two agreeing is the check that the sampling is doing what the formula says.
    * ``pf_inside_domain``: the point estimate restricted to the draws inside the validity
      domain of build spec 9.4. Not a correction, and not a substitute for the headline number:
      it is what the probability would be if the population were truncated to the region the
      solver actually reached, and the difference between the two is the size of the question
      the censoring leaves open.
    """
    n_samples = int(np.asarray(mean_prediction).size)
    point = failure_mask(mean_prediction, threshold, state.direction)
    band = (
        failure_mask(lower, threshold, state.direction)
        if state.direction == "below"
        else failure_mask(upper, threshold, state.direction)
    )
    conservative = band | point
    pf_point = float(point.mean())
    hits = int(point.sum())
    wilson_low, wilson_high = wilson_interval(hits, n_samples)
    inside = np.asarray(inside_domain, dtype=bool)
    n_inside = int(inside.sum())
    sigma = np.asarray(predictive_sigma, dtype=float)
    centered = (threshold - np.asarray(predictive_mean, dtype=float)) / sigma
    tail = ndtr(centered) if state.direction == "below" else 1.0 - ndtr(centered)
    predictive_flat = np.asarray(predictive, dtype=float).ravel()
    pf_predictive = float(failure_mask(predictive_flat, threshold, state.direction).mean())
    return {
        "config_field": state.config_field,
        "target": state.target,
        "direction": state.direction,
        "label": state.label,
        "short_label": state.short_label,
        "justification": state.justification,
        "threshold": float(threshold),
        "n_samples": n_samples,
        "n_failures": hits,
        "pf_point": pf_point,
        "pf_standard_error": binomial_standard_error(pf_point, n_samples),
        "pf_wilson_low": float(wilson_low),
        "pf_wilson_high": float(wilson_high),
        "pf_conservative": float(conservative.mean()),
        "n_point_outside_band": int((point & ~band).sum()),
        "pf_predictive": pf_predictive,
        "pf_predictive_standard_error": binomial_standard_error(
            pf_predictive, int(predictive_flat.size)
        ),
        "pf_predictive_closed_form": float(tail.mean()),
        "n_predictive_draws": int(predictive_flat.size),
        "pf_inside_domain": float(point[inside].mean()) if n_inside > 0 else float("nan"),
        "n_inside_domain": n_inside,
        "median_band_width": float(np.median(upper - lower)),
        "resolvable": bool(pf_point >= RESOLVABLE_PF_FLOOR),
    }


def curve_envelope(
    surrogate: SurrogateModel, design: np.ndarray, levels: tuple[float, ...] = ENVELOPE_LEVELS
) -> pd.DataFrame:
    """Pointwise percentiles of the predicted force and damage curves over a subsample.

    The fan of build spec 13.1, and it is the aleatory layer: each curve is one input draw
    through the mean surrogate, so the envelope is the spread of the beam population and not a
    band on any one prediction. The calibrated band on a single curve is the functional product
    of build spec 11.2 and lives in the calibration stage, where it was measured against held
    out curves; drawing the two on the same axes without saying which is which is how a
    population spread gets read as a confidence band.
    """
    prediction = surrogate.predict_curve(design)
    frame = pd.DataFrame({"u_mm": np.asarray(prediction.u_grid, dtype=float)})
    for name, matrix in (
        ("force_N", prediction.force_mean),
        ("damage", prediction.damage_mean),
    ):
        percentiles = np.percentile(np.asarray(matrix, dtype=float), levels, axis=0)
        for level, row in zip(levels, percentiles):
            frame[f"{name}_p{int(level):02d}"] = row
    frame["n_curves"] = int(np.asarray(design).shape[0])
    return frame


# ---------------------------------------------------------------------------
# Report fragments
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 3) -> str:
    """Fixed point, or a stated marker for a value that is not a number."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_probability(value: float, n_samples: int) -> str:
    """A probability at four decimals, with the two cases that must not print as a number.

    A count of zero is printed as a bound at the resolution of the sample, never as
    ``0.0000``: no draw failing says the probability is below what this many draws can see,
    which is a statement about the sample, while a printed zero invites a reader to hear
    impossibility. The exponent is floored so the bound is true rather than rounded, whatever
    sample size it is asked about. A value that is not a number, which is what a probability
    conditioned on an empty subset would be, prints as a stated marker rather than as ``nan``.
    """
    if not np.isfinite(value):
        return "not defined"
    if value == 0.0:
        exponent = int(math.floor(math.log10(max(int(n_samples), 1))))
        return f"$< 10^{{-{exponent}}}$"
    if value < RESOLVABLE_PF_FLOOR:
        return f"{value:.2e}"
    return f"{value:.4f}"


def roughness_sentence(roughness: dict[str, Any]) -> str:
    """The P6 caveat of build spec 13.3, written from the P6 measurement.

    Not a fixed string. The share is read from the sensitivity stage's own artifact, so if the
    campaign or the measurement changes, the sentence changes with it rather than becoming a
    remembered number in a docstring.
    """
    share = 100.0 * float(roughness["roughness_ratio"])
    pairs = int(roughness["n_pairs"])
    return (
        f"Unmodeled error: phase P6 measured that the peak load differs by a median of "
        f"{share:.0f} percent of its campaign standard deviation between the {pairs} closest "
        "nearest neighbour pairs of the training design. The response is rough at the scale "
        "this design resolves, and no interval in this table contains that error."
    )


def build_reliability_table(payload: dict[str, Any]) -> str:
    """The reliability table of build spec 13.2, one row per limit state."""
    out_of_domain = 100.0 * float(payload["validity"]["out_of_domain_fraction"])
    lines = [
        "% Generated by the propagate stage (ufem.propagate). Do not edit.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Limit state & Threshold & $P_f$ & Binomial SE & Conservative & Outside domain \\",
        r"\midrule",
    ]
    for record in payload["limit_states"]:
        unit, scale = QOI_DISPLAY[record["target"]]
        threshold = record["threshold"] * scale
        digits = 2 if scale != 1.0 else 3
        lines.append(
            f"{record['short_label']} & {threshold:.{digits}f}{_tex_unit_suffix(unit)} & "
            f"{format_probability(record['pf_point'], record['n_samples'])} & "
            f"{_fmt(record['pf_standard_error'], 5)} & "
            f"{format_probability(record['pf_conservative'], record['n_samples'])} & "
            f"{out_of_domain:.1f}\\,\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _tex_unit(unit: str) -> str:
    """A unit as LaTeX text. The dimensionless marker is spelled out, never as a bare dash."""
    if unit == "-":
        return "dimensionless"
    return unit.replace("/", "$/$")


def _tex_unit_suffix(unit: str) -> str:
    """A unit appended to a number, or nothing at all when the quantity has none."""
    if unit == "-":
        return ""
    return f"\\,{_tex_unit(unit)}"


def build_quantile_table(payload: dict[str, Any]) -> str:
    """The propagated distribution of every scalar QoI of build spec 9.5."""
    lines = [
        "% Generated by the propagate stage (ufem.propagate). Do not edit.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Quantity & Mean & 5\,\% & 50\,\% & 95\,\% & SE of 5\,\% & Predictive 5\,\% \\",
        r"\midrule",
    ]
    for name in payload["context"]["reported_targets"]:
        record = payload["targets"][name]
        unit, scale = QOI_DISPLAY[name]
        digits = 2 if scale != 1.0 else 3
        aleatory = record["aleatory"]
        predictive = record["predictive"]
        levels = aleatory["levels"]
        index_05 = levels.index(0.05)
        index_50 = levels.index(0.5)
        index_95 = levels.index(0.95)
        lines.append(
            f"{QOI_LABELS.get(name, name)} [{_tex_unit(unit)}] & "
            f"{aleatory['mean'] * scale:.{digits}f} & "
            f"{aleatory['quantiles'][index_05] * scale:.{digits}f} & "
            f"{aleatory['quantiles'][index_50] * scale:.{digits}f} & "
            f"{aleatory['quantiles'][index_95] * scale:.{digits}f} & "
            f"{aleatory['standard_error'][index_05] * scale:.{digits + 2}f} & "
            f"{predictive['quantiles'][index_05] * scale:.{digits}f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_analytic_table(payload: dict[str, Any]) -> str:
    """The cross check of build spec 13.4, statistic by statistic."""
    analytic = payload["analytic"]
    comparison = analytic["comparison"]
    left = comparison["surrogate"]
    right = comparison["analytic"]
    scale = 1.0e-3
    rows = [
        ("Mean [kN]", left["mean"] * scale, right["mean"] * scale, 2),
        ("5th percentile [kN]", left["p05"] * scale, right["p05"] * scale, 2),
        ("Median [kN]", left["median"] * scale, right["median"] * scale, 2),
        ("95th percentile [kN]", left["p95"] * scale, right["p95"] * scale, 2),
        ("Coefficient of variation", left["cov"], right["cov"], 4),
    ]
    lines = [
        "% Generated by the propagate stage (ufem.propagate). Do not edit.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Peak load statistic & Surrogate & Analytic & Ratio \\",
        r"\midrule",
    ]
    for label, first, second, digits in rows:
        lines.append(
            f"{label} & {first:.{digits}f} & {second:.{digits}f} & {second / first:.3f} \\\\"
        )
    lines.append(r"\addlinespace")
    measured = analytic["empirical_elasticities"]["elasticities"]
    modeled = analytic["analytic_elasticities"]
    for name in FEATURE_ORDER:
        lines.append(
            f"Elasticity, {_tex_input(name)} & {measured[name]:+.3f} & "
            f"{modeled[name]:+.3f} & {modeled[name] / measured[name]:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _tex_input(name: str) -> str:
    return {
        "Fcm_MPa": r"$f_{cm}$",
        "c_nom_bottom_mm": r"$c_{\mathrm{bot}}$",
        "c_nom_top_mm": r"$c_{\mathrm{top}}$",
    }[name]


def build_markdown_summary(payload: dict[str, Any]) -> str:
    """The stage's own readable report, for the engineering log and for a human check."""
    context = payload["context"]
    validity = payload["validity"]
    analytic = payload["analytic"]
    out: list[str] = []

    def add(line: str = "") -> None:
        out.append(line)

    add("# Propagation and reliability")
    add()
    add(f"Config SHA-256: `{payload['config_sha256']}`.")
    add()
    add(
        f"Aleatory layer: {context['n_samples']} input draws through the mean surrogate. "
        f"Epistemic layer: {context['posterior_draws']} calibrated posterior draws at each of "
        f"{context['epistemic_subsample']} seeded subsampled draws. Curve fan: "
        f"{context['curve_subsample']} predicted curves."
    )
    add()
    add(
        f"**{100.0 * validity['out_of_domain_fraction']:.1f} percent of the Monte Carlo mass "
        f"falls outside the validity domain** of build spec 9.4, so every number below "
        "describes a population the surrogate is only partly entitled to speak about."
    )
    add()
    add(payload["roughness_caveat"])
    add()
    add(
        f"No failure probability below {RESOLVABLE_PF_FLOOR:.0e} is claimed (build spec 13.3): "
        f"at {context['n_training_runs']} training runs, rarer events need the active learning "
        "of Track B."
    )
    add()
    add("## Limit states")
    add()
    add("| Limit state | Threshold | Pf | Binomial SE | Conservative bound | Pf inside domain |")
    add("|---|---|---|---|---|---|")
    for record in payload["limit_states"]:
        unit, scale = QOI_DISPLAY[record["target"]]
        # A dimensionless quantity gets no unit at all here. Printing the marker after the
        # number gives "0.5 -", which reads as an arithmetic sign rather than as an absence.
        suffix = "" if unit == "-" else f" {unit}"
        add(
            f"| {record['label']} | {record['threshold'] * scale:.4g}{suffix} | "
            f"{format_probability(record['pf_point'], record['n_samples'])} | "
            f"{record['pf_standard_error']:.5f} | "
            f"{format_probability(record['pf_conservative'], record['n_samples'])} | "
            f"{format_probability(record['pf_inside_domain'], record['n_inside_domain'])} |"
        )
    add()
    characteristic = payload["characteristic_value"]
    add("## The peak load threshold")
    add()
    add(
        f"Characteristic value, the {100 * CHARACTERISTIC_LEVEL:.0f}th percentile of the "
        f"aleatory peak load: {characteristic['value_N']:.1f} N. Configured threshold: "
        f"{characteristic['configured_N']:.1f} N. Relative gap: "
        f"{100.0 * characteristic['relative_gap']:+.2f} percent, which is "
        f"{'inside' if characteristic['agrees'] else 'outside'} the "
        f"{100.0 * CHARACTERISTIC_TOLERANCE:.0f} percent tolerance."
    )
    add()
    add("## Analytic cross check")
    add()
    comparison = analytic["comparison"]
    add(
        f"Analytic median {comparison['analytic']['median'] / 1000.0:.2f} kN against surrogate "
        f"median {comparison['surrogate']['median'] / 1000.0:.2f} kN, a ratio of "
        f"{comparison['median_ratio']:.3f} against a stated model error of "
        f"{100.0 * comparison['model_error']:.0f} percent. Central tendency brackets: "
        f"{comparison['central_tendency_brackets']}. Dispersion brackets: "
        f"{comparison['dispersion_brackets']} (analytic coefficient of variation "
        f"{comparison['analytic']['cov']:.4f} against {comparison['surrogate']['cov']:.4f}, a "
        f"ratio of {comparison['dispersion_ratio']:.3f})."
    )
    add()
    add("| Input | Campaign elasticity | Analytic elasticity |")
    add("|---|---|---|")
    for name in FEATURE_ORDER:
        add(
            f"| {name} | "
            f"{analytic['empirical_elasticities']['elasticities'][name]:+.4f} | "
            f"{analytic['analytic_elasticities'][name]:+.4f} |"
        )
    add()
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, dict[str, str]]:
    artifact_root = root / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    surrogate_dir = stage_dir(artifact_root, SURROGATE_STAGE, config_sha256)
    calibrate_dir = stage_dir(artifact_root, CALIBRATE_STAGE, config_sha256)
    sensitivity_dir = stage_dir(artifact_root, SENSITIVITY_STAGE, config_sha256)
    hashes: dict[str, str] = {}
    for directory, name, stage in (
        (grid_dir, QOI_PARQUET, GRID_STAGE),
        (surrogate_dir, SURROGATE_JSON, SURROGATE_STAGE),
        (calibrate_dir, CALIBRATION_JSON, CALIBRATE_STAGE),
        (calibrate_dir, SCALAR_CONFORMAL_PARQUET, CALIBRATE_STAGE),
        (sensitivity_dir, SENSITIVITY_JSON, SENSITIVITY_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise PropagationInputMissing(
                f"the propagate stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first."
            )
        hashes[name] = sha256_file(path)
    return grid_dir, calibrate_dir, hashes


def declared_input_hashes(
    repo_root: Path | str, config: Config, config_sha256: str
) -> dict[str, str]:
    """Hash this stage's declared inputs as they are on disk right now (see ``ufem.runner``)."""
    return _load_inputs(Path(repo_root), config, config_sha256)[-1]


def _calibration_gate_passed(calibration: dict[str, Any]) -> dict[str, Any]:
    """Refuse to propagate behind a calibration gate that did not pass.

    Build spec 11.5 blocks the propagated numbers of section 13 on this gate, and this stage is
    exactly what it blocks: the conservative bound is a jackknife+ interval and the epistemic
    layer is a calibrated predictive distribution, so both are statements the gate is about.
    """
    gate = calibration["gate"]
    if not gate.get("passed", False):
        raise AssertionError(
            "the calibration gate of build spec 11.5 did not pass for this config hash, so "
            "neither the epistemic layer nor the conservative bound of build spec 13.2 means "
            f"what it says. Failing checks: {gate.get('failing')}. Fix the model, not this "
            "stage."
        )
    return {"passed": True, "failing": list(gate.get("failing", []))}


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the propagation stage and return its artifact directory."""
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    grid_dir, calibrate_dir, input_hashes = _load_inputs(root, config, config_sha256)
    artifact_root = root / config.pipeline.paths.artifact_root

    calibration = json.loads(
        (calibrate_dir / CALIBRATION_JSON).read_text(encoding="utf-8")
    )
    calibration_gate = _calibration_gate_passed(calibration)
    conformal = pd.read_parquet(calibrate_dir / SCALAR_CONFORMAL_PARQUET)
    sensitivity = json.loads(
        (
            stage_dir(artifact_root, SENSITIVITY_STAGE, config_sha256) / SENSITIVITY_JSON
        ).read_text(encoding="utf-8")
    )
    qoi = pd.read_parquet(grid_dir / QOI_PARQUET)
    surrogate = SurrogateModel.load(artifact_root, config_sha256)
    settings = config.pipeline.mc

    if BAND_ALPHA not in set(config.pipeline.conformal.alphas):
        raise ValueError(
            f"the conservative bound of build spec 13.2 reads a {100 * (1 - BAND_ALPHA):.0f} "
            f"percent band, but the calibration stage was configured for alphas "
            f"{config.pipeline.conformal.alphas}. The band this stage counts against has to be "
            "one the calibration actually measured coverage for."
        )

    seed_root = np.random.SeedSequence(config.pipeline.seed_entropy)
    input_seed, subsample_seed, epistemic_seed, curve_seed = seed_root.spawn(4)

    # ---- the aleatory layer ------------------------------------------------
    sampling_started = _time.perf_counter()
    design = draw_inputs(config, settings.n_samples, input_seed)
    standardized = surrogate.feature_standardizer.transform(design)
    inside = np.asarray(in_validity_domain(design, root, config), dtype=bool)
    sampling_seconds = _time.perf_counter() - sampling_started

    targets = list(SCALAR_QOI) + [
        name for name in LANDMARK_QOI if name in surrogate.scalar_targets
    ]
    missing = [name for name in targets if name not in surrogate.models]
    if missing:
        raise KeyError(
            f"the surrogate artifact carries no process for {missing}, so those quantities "
            "cannot be propagated. The target list must be a subset of what it fitted."
        )

    subsample = subsample_indices(
        settings.n_samples, settings.epistemic_subsample, subsample_seed
    )
    epistemic_children = dict(zip(targets, epistemic_seed.spawn(len(targets))))

    aleatory_started = _time.perf_counter()
    pieces = {name: posterior_pieces(surrogate.models[name]) for name in targets}
    means: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    quantile_rows: list[dict[str, Any]] = []
    density_frames: list[pd.DataFrame] = []
    for position, name in enumerate(targets, start=1):
        scaling = float(calibration["scalar"][name]["variance_scaling_factor"])
        mean, variance = predict_mean_and_variance(
            surrogate.models[name],
            surrogate.target_standardizers[name],
            standardized,
            pieces[name],
        )
        means[name] = mean
        sigmas[name] = scaling * np.sqrt(variance)
        draws = epistemic_draws(
            mean[subsample],
            sigmas[name][subsample],
            settings.posterior_draws,
            epistemic_children[name],
        )
        aleatory = quantile_summary(mean)
        predictive = quantile_summary(draws.ravel())
        grid, aleatory_density = density_curve(mean)
        predictive_density = kernel_density(draws.ravel(), grid)
        records[name] = {
            "label": QOI_LABELS.get(name, name),
            "unit": QOI_DISPLAY[name][0],
            "variance_scaling_factor": scaling,
            "median_predictive_sigma": float(np.median(sigmas[name])),
            "aleatory": aleatory,
            "predictive": predictive,
            # The law of total variance splits the predictive spread exactly: the variance of
            # the mean prediction over the input draws is the aleatory part, and the mean of
            # the calibrated predictive variance is the epistemic part. Reporting the second
            # directly rather than as a difference of two sampled standard deviations keeps a
            # small epistemic layer from being reported as the round off of a large aleatory
            # one.
            "epistemic_std": float(np.sqrt(np.mean(sigmas[name] ** 2))),
            "epistemic_to_aleatory_std": float(
                np.sqrt(np.mean(sigmas[name] ** 2)) / aleatory["std"]
            ),
        }
        for level, value, error, density in zip(
            aleatory["levels"],
            aleatory["quantiles"],
            aleatory["standard_error"],
            aleatory["density_at_quantile"],
        ):
            index = aleatory["levels"].index(level)
            quantile_rows.append(
                {
                    "target": name,
                    "unit": QOI_DISPLAY[name][0],
                    "level": level,
                    "aleatory": value,
                    "standard_error": error,
                    "density": density,
                    "predictive": predictive["quantiles"][index],
                    "predictive_standard_error": predictive["standard_error"][index],
                }
            )
        density_frames.append(
            pd.DataFrame(
                {
                    "target": name,
                    "value": grid,
                    "aleatory_density": aleatory_density,
                    "predictive_density": predictive_density,
                }
            )
        )
        del draws
        print(
            f"[propagate] {position}/{len(targets)} {name}: aleatory mean "
            f"{aleatory['mean']:.4g}, predictive 5th percentile "
            f"{predictive['quantiles'][1]:.4g}"
        )
    aleatory_seconds = _time.perf_counter() - aleatory_started

    # ---- limit states and the conservative bound ----------------------------
    band_started = _time.perf_counter()
    thresholds = config.pipeline.limit_states.model_dump()
    limit_records: list[dict[str, Any]] = []
    for state in LIMIT_STATES:
        name = state.target
        scores = conformal.loc[conformal["target"] == name, "score"].to_numpy(dtype=float)
        scaling = float(calibration["scalar"][name]["variance_scaling_factor"])
        lower, upper = calibrated_band(
            surrogate.models[name],
            surrogate.target_standardizers[name],
            standardized,
            pieces[name],
            scores,
            scaling,
        )
        draws = epistemic_draws(
            means[name][subsample],
            sigmas[name][subsample],
            settings.posterior_draws,
            epistemic_children[name],
        )
        limit_records.append(
            limit_state_result(
                state,
                float(thresholds[state.config_field]),
                means[name],
                lower,
                upper,
                draws,
                means[name][subsample],
                sigmas[name][subsample],
                inside,
            )
        )
        del draws
        print(
            f"[propagate] limit state {state.config_field}: Pf "
            f"{limit_records[-1]['pf_point']:.5f}, conservative "
            f"{limit_records[-1]['pf_conservative']:.5f}"
        )
    band_seconds = _time.perf_counter() - band_started

    # ---- the characteristic value ------------------------------------------
    characteristic = float(np.quantile(means["P_max_N"], CHARACTERISTIC_LEVEL))
    configured = float(config.pipeline.limit_states.peak_load_below_N)
    relative_gap = configured / characteristic - 1.0

    # ---- the curve fan ------------------------------------------------------
    curve_started = _time.perf_counter()
    curve_rows = subsample_indices(settings.n_samples, settings.curve_subsample, curve_seed)
    envelope = curve_envelope(surrogate, design[curve_rows])
    curve_seconds = _time.perf_counter() - curve_started

    # ---- the analytic cross check -------------------------------------------
    analytic_started = _time.perf_counter()
    analytic_sample = peak_load_N(design[:, 0], design[:, 1], design[:, 2])
    comparison = cross_check(means["P_max_N"], analytic_sample, MODEL_ERROR_FRACTION)
    training_design = features(qoi)
    empirical = empirical_log_elasticities(
        training_design, qoi["P_max_N"].to_numpy(dtype=float), FEATURE_ORDER
    )
    medians = [float(np.median(design[:, position])) for position in range(len(FEATURE_ORDER))]
    modulus = float(derived_E(medians[0]))
    analytic_grid = np.linspace(
        min(
            float(np.quantile(means["P_max_N"], DENSITY_RANGE[0])),
            float(np.quantile(analytic_sample, DENSITY_RANGE[0])),
        ),
        max(
            float(np.quantile(means["P_max_N"], DENSITY_RANGE[1])),
            float(np.quantile(analytic_sample, DENSITY_RANGE[1])),
        ),
        DENSITY_GRID_POINTS,
    )
    analytic_frame = pd.DataFrame(
        {
            "peak_load_N": analytic_grid,
            "surrogate_density": kernel_density(means["P_max_N"], analytic_grid),
            "analytic_density": kernel_density(analytic_sample, analytic_grid),
        }
    )
    training_stiffness = qoi["k0_N_per_mm"].to_numpy(dtype=float)
    gross_stiffness = float(
        tip_stiffness_N_per_mm(modulus, DEFAULT_BEAM.gross_second_moment_mm4)
    )
    cracked_stiffness = float(
        tip_stiffness_N_per_mm(
            modulus, float(cracked_second_moment_mm4(modulus, medians[1], medians[2]))
        )
    )
    analytic_record = {
        "beam": {
            "span_mm": DEFAULT_BEAM.span_mm,
            "overhang_mm": DEFAULT_BEAM.overhang_mm,
            "depth_mm": DEFAULT_BEAM.depth_mm,
            "thickness_mm": DEFAULT_BEAM.thickness_mm,
            "area_top_mm2": DEFAULT_BEAM.area_top_mm2,
            "area_bottom_mm2": DEFAULT_BEAM.area_bottom_mm2,
            "gross_second_moment_mm4": DEFAULT_BEAM.gross_second_moment_mm4,
        },
        "comparison": comparison,
        "analytic_elasticities": log_elasticities(*medians),
        "empirical_elasticities": empirical,
        "median_inputs": {name: value for name, value in zip(FEATURE_ORDER, medians)},
        "modulus_MPa": modulus,
        "cracking_load_N": float(cracking_load_N(medians[0])),
        "stiffness": {
            "gross_N_per_mm": gross_stiffness,
            "cracked_N_per_mm": cracked_stiffness,
            "campaign_mean_N_per_mm": float(training_stiffness.mean()),
            "campaign_brackets": bool(
                cracked_stiffness <= float(training_stiffness.mean()) <= gross_stiffness
            ),
        },
        "pointwise": {
            "pearson_on_training_design": float(
                np.corrcoef(
                    peak_load_N(
                        training_design[:, 0], training_design[:, 1], training_design[:, 2]
                    ),
                    qoi["P_max_N"].to_numpy(dtype=float),
                )[0, 1]
            ),
            "mean_relative_error": float(
                np.mean(
                    peak_load_N(
                        training_design[:, 0], training_design[:, 1], training_design[:, 2]
                    )
                    / qoi["P_max_N"].to_numpy(dtype=float)
                    - 1.0
                )
            ),
        },
    }
    analytic_seconds = _time.perf_counter() - analytic_started

    roughness = sensitivity["targets"]["P_max_N"]["pce"]["design_roughness"]
    payload: dict[str, Any] = {
        "config_sha256": config_sha256,
        "context": {
            "n_samples": int(settings.n_samples),
            "epistemic_subsample": int(settings.epistemic_subsample),
            "posterior_draws": int(settings.posterior_draws),
            "curve_subsample": int(settings.curve_subsample),
            "n_training_runs": int(len(qoi)),
            "feature_order": list(FEATURE_ORDER),
            "targets": targets,
            "reported_targets": list(SCALAR_QOI),
            "quantile_levels": list(QUANTILE_LEVELS),
            "band_alpha": BAND_ALPHA,
            "resolvable_pf_floor": RESOLVABLE_PF_FLOOR,
            "prediction_chunk": PREDICTION_CHUNK,
            "band_chunk": BAND_CHUNK,
            "calibration_gate": calibration_gate,
            "seed_entropy": str(config.pipeline.seed_entropy),
        },
        "validity": {
            "n_inside": int(inside.sum()),
            "n_outside": int((~inside).sum()),
            "out_of_domain_fraction": float((~inside).mean()),
            "completion_threshold": float(load_validity_domain(root, config).threshold),
        },
        "characteristic_value": {
            "level": CHARACTERISTIC_LEVEL,
            "value_N": characteristic,
            "configured_N": configured,
            "relative_gap": relative_gap,
            "tolerance": CHARACTERISTIC_TOLERANCE,
            "agrees": bool(abs(relative_gap) <= CHARACTERISTIC_TOLERANCE),
        },
        "targets": records,
        "limit_states": limit_records,
        "analytic": analytic_record,
        "roughness": roughness,
        "roughness_caveat": roughness_sentence(roughness),
    }

    directory = stage_dir(artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, text in (
        (PROPAGATION_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (RELIABILITY_TEX, build_reliability_table(payload)),
        (QUANTILES_TEX, build_quantile_table(payload)),
        (ANALYTIC_TEX, build_analytic_table(payload)),
        (PROPAGATION_MD, build_markdown_summary(payload)),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8", newline="\n")
        outputs.append(path)
    for frame, name in (
        (pd.DataFrame(quantile_rows), QUANTILES_PARQUET),
        (pd.concat(density_frames, ignore_index=True), DENSITY_PARQUET),
        (pd.DataFrame(limit_records), RELIABILITY_PARQUET),
        (envelope, CURVE_ENVELOPE_PARQUET),
        (analytic_frame, ANALYTIC_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "sampling_wall_time_s": sampling_seconds,
        "aleatory_wall_time_s": aleatory_seconds,
        "band_wall_time_s": band_seconds,
        "curve_wall_time_s": curve_seconds,
        "analytic_wall_time_s": analytic_seconds,
        "n_samples": int(settings.n_samples),
        "n_targets": len(targets),
        "out_of_domain_fraction": payload["validity"]["out_of_domain_fraction"],
        "resolvable_pf_floor": RESOLVABLE_PF_FLOOR,
        "roughness_ratio": float(roughness["roughness_ratio"]),
        "characteristic_value": payload["characteristic_value"],
        "limit_states": {
            record["config_field"]: {
                "pf_point": record["pf_point"],
                "pf_standard_error": record["pf_standard_error"],
                "pf_conservative": record["pf_conservative"],
                "pf_predictive": record["pf_predictive"],
            }
            for record in limit_records
        },
        "analytic_cross_check": {
            "median_ratio": comparison["median_ratio"],
            "central_tendency_brackets": comparison["central_tendency_brackets"],
            "dispersion_brackets": comparison["dispersion_brackets"],
            "model_error": comparison["model_error"],
        },
        "calibration_gate": calibration_gate,
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
    print(
        f"[propagate] {settings.n_samples} aleatory draws over {len(targets)} targets, "
        f"{100.0 * payload['validity']['out_of_domain_fraction']:.1f} percent outside the "
        f"validity domain; characteristic peak load {characteristic:.0f} N against the "
        f"configured {configured:.0f} N; analytic median ratio "
        f"{comparison['median_ratio']:.3f}"
    )
    return directory
