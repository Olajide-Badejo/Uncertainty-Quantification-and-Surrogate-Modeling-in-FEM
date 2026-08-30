"""Simultaneous sup norm conformal bands for the curve outputs (build spec 11.2).

The construction, in one paragraph. Take out of fold predictions for the n training curves:
a mean curve ``mu_-i(u)`` and a modulation ``s_i(u)``, which here is the Gaussian process
posterior predictive standard deviation of the curve, truncation residual included. Score
each curve by its worst standardized pointwise miss,

    R_i = sup_u |y_i(u) - mu_-i(u)| / s_i(u),

sort the n scores, and take the ``k = ceil((n + 1)(1 - alpha))``th of them. The band
``mu(u) +/- R_(k) s(u)`` then covers a whole exchangeable new curve, at every abscissa at
once, with probability at least ``1 - alpha`` and at most ``1 - alpha + 1/(n + 1)``. The
guarantee is finite sample and distribution free: it needs exchangeability and nothing else,
in particular nothing about the Gaussian process being right.

Two things this module deliberately does not do. It never modifies a variance: the modulation
arrives as an argument, is used as a shape, and is returned untouched, because a band widened
by a factor chosen to make it cover is the fabricated uncertainty of build spec 5.1 wearing a
conformal hat (ground rule 4). And it never falls back when the sample is too small for the
requested alpha: ``k > n`` means the data cannot support that level, and it raises saying so
rather than returning the maximum and calling it a quantile.

Units: whatever the curves carry. The scores are dimensionless because the modulation carries
the same unit as the residual, which is the point of standardizing before taking a supremum
over an axis where the curve's own scale changes by an order of magnitude.

The name to reach for in the literature: Diquigiovanni, Fontana and Vantini (2022) call this
the modulated sup norm nonconformity measure for functional data, and it is what the R package
``conformalInference.fd`` implements. R is not installed on this machine, so the cross check
build spec 11.2 asks for is done against a hand computable analytic construction instead; see
``tests/test_conformal_functional.py`` for the substitution and its reasoning.
"""

from __future__ import annotations

import math

import numpy as np


def _as_curve_matrix(array: np.ndarray, role: str) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(array, dtype=float))
    if matrix.ndim != 2:
        raise ValueError(f"{role} must be an (n_curves, n_grid) array, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{role} carries {int((~np.isfinite(matrix)).sum())} non finite values. A conformal "
            "score over a curve with a NaN in it is not a number, and dropping the row would "
            "silently change the calibration set size."
        )
    return matrix


def sup_norm_scores(
    truth: np.ndarray, mean: np.ndarray, modulation: np.ndarray
) -> np.ndarray:
    """The nonconformity score of build spec 11.2, one per curve.

    ``truth``, ``mean`` and ``modulation`` are ``(n_curves, n_grid)``; ``modulation`` may also
    be a single row, which is broadcast when every curve shares one shape. Returns ``(n,)``.
    """
    y = _as_curve_matrix(truth, "the observed curves")
    mu = _as_curve_matrix(mean, "the predicted mean curves")
    s = _as_curve_matrix(modulation, "the modulation")
    if y.shape != mu.shape:
        raise ValueError(
            f"the observed curves are {y.shape} and the predicted means are {mu.shape}; a "
            "conformal score needs them on the same grid and in the same row order."
        )
    if s.shape[0] == 1 and y.shape[0] != 1:
        s = np.repeat(s, y.shape[0], axis=0)
    if s.shape != y.shape:
        raise ValueError(
            f"the modulation is {s.shape} against curves of {y.shape}. Pass one modulation per "
            "curve, or a single row to be shared by all of them."
        )
    if np.any(s <= 0.0):
        raise ValueError(
            "the modulation has a non positive entry, so a standardized residual is undefined "
            "there. Nothing is floored here (ground rule 4); fix the variance that produced it."
        )
    return np.max(np.abs(y - mu) / s, axis=1)


def conformal_rank(n_scores: int, alpha: float) -> int:
    """``k = ceil((n + 1)(1 - alpha))``, the order statistic that scales the band."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    if n_scores < 1:
        raise ValueError("a conformal quantile needs at least one calibration score.")
    rank = math.ceil((n_scores + 1) * (1.0 - alpha))
    if rank > n_scores:
        raise ValueError(
            f"a {100 * (1 - alpha):.1f} percent band needs the {rank}th of {n_scores} "
            f"calibration scores, which does not exist. At n = {n_scores} the finest level "
            f"with a finite conformal quantile is {100 * (1 - 1 / (n_scores + 1)):.2f} percent. "
            "Returning the largest score instead would be an unbounded band reported as a "
            "quantile."
        )
    return rank


def band_scale(scores: np.ndarray, alpha: float) -> float:
    """The conformal band multiplier: the ``conformal_rank``th smallest score."""
    values = np.asarray(scores, dtype=float).ravel()
    rank = conformal_rank(values.size, alpha)
    return float(np.partition(values, rank - 1)[rank - 1])


def simultaneous_band(
    mean: np.ndarray, modulation: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """``mu(u) -/+ scale * s(u)``: the lower and upper band, same shape as ``mean``."""
    mu = _as_curve_matrix(mean, "the predicted mean curves")
    s = _as_curve_matrix(modulation, "the modulation")
    if s.shape[0] == 1 and mu.shape[0] != 1:
        s = np.repeat(s, mu.shape[0], axis=0)
    if s.shape != mu.shape:
        raise ValueError(f"the modulation is {s.shape} against means of {mu.shape}.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"the band scale must be a positive finite number, got {scale}.")
    half_width = scale * s
    return mu - half_width, mu + half_width


def leave_one_out_coverage(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Per curve indicator of whether the band calibrated on the *other* curves covers it.

    This is the honest empirical coverage of the construction, and it is not the same thing as
    checking a curve against a band its own score helped set. For curve i the calibration set
    is the other ``n - 1`` scores, so the threshold is their ``ceil(n(1 - alpha))``th order
    statistic, and by exchangeability the indicator has exactly the coverage a genuinely new
    curve would get. The mean of the returned array is therefore an unbiased estimate of the
    coverage the finite sample theory brackets in ``[1 - alpha, 1 - alpha + 1/n]``.
    """
    values = np.asarray(scores, dtype=float).ravel()
    n = values.size
    if n < 2:
        raise ValueError("leave one out coverage needs at least two calibration scores.")
    rank = conformal_rank(n - 1, alpha)
    ordered = np.sort(values)
    # For curve i the sorted others are the sorted whole with one element removed. An element
    # at or below the threshold shifts the ranks above it down by one, which is exactly the
    # two cases below; no per curve sort is needed.
    threshold_when_below = ordered[rank]
    threshold_when_above = ordered[rank - 1]
    covered = np.empty(n, dtype=bool)
    for index, value in enumerate(values):
        position = int(np.searchsorted(ordered, value, side="left"))
        threshold = threshold_when_below if position < rank else threshold_when_above
        covered[index] = bool(value <= threshold)
    return covered


def coverage_bounds(n_calibration: int, alpha: float) -> tuple[float, float]:
    """The exact finite sample bracket a conformal band's coverage must fall in."""
    if n_calibration < 1:
        raise ValueError("the bracket needs at least one calibration point.")
    lower = 1.0 - alpha
    return lower, lower + 1.0 / (n_calibration + 1)
