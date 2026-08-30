"""Unit tests for the core computation of each ablation script (build spec 10.6, 16.1).

An ablation is evidence about a design choice, so the thing that has to be right is not the
number it prints but the machinery that produced it. Each of the four scripts of phase P9 has
one property that, if it broke, would turn its result into a plausible number with nothing
behind it, and each of those properties is pinned here:

- the monotone damage decoder of ablation 2 cannot emit a decreasing curve, whatever its
  weights are, and it does not carry the salvaged file's terminal renormalization;
- the deep ensemble of ablation 3 gives a variance that grows with member disagreement, which
  is the entire reason an ensemble is a predictive distribution rather than five guesses;
- the B-spline basis of ablation 4 reproduces a curve exactly when the basis can express it,
  which is the identity every projection argument in that script rests on;
- the subsamplers of ablation 5 are functions of their seed alone, because a design study whose
  designs move between runs measures nothing.

The torch tests take an ``importorskip`` guard rather than a module level one, so the spline
and subsampling tests still run on the light dependency stack the fast CI job installs. That
guard is the fourth instance of a lesson recorded in docs/DEFECT_LOG.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from ablation_2_autoencoder import build_autoencoder
from ablation_3_deep_ensemble import ensemble_mixture
from ablation_4_bspline import (
    DEGREE,
    N_BASIS,
    fit_coefficients,
    peak_weighted_knots,
    spline_basis,
)
from ablation_5_design import random_subset, sobol_guided_subset

from ufem.ablation_reference import (
    curve_metrics,
    gaussian_nlpd,
    peak_curvature,
    peak_metrics,
    pointwise_coverage,
)

U_GRID = np.linspace(0.0, 20.0, 201)


def _seed(*key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(20260831, spawn_key=key)


class TestTheMonotoneDamageDecoder:
    """Ablation 2, the salvaged idea of build spec 6.4 item 7."""

    def test_it_cannot_emit_a_decreasing_curve(self):
        torch = pytest.importorskip("torch")

        torch.manual_seed(11)
        _encoder, decoder = build_autoencoder(n_points=201, latent=4, monotone=True)
        codes = torch.randn(32, 4, dtype=torch.get_default_dtype()) * 5.0
        with torch.no_grad():
            curves = decoder(codes).numpy()
        assert curves.shape == (32, 201)
        assert np.min(np.diff(curves, axis=1)) >= 0.0, (
            "a cumulative sum of softplus increments is non decreasing whatever the weights "
            "are; a negative increment means the head is no longer that construction."
        )

    def test_the_unconstrained_decoder_is_genuinely_unconstrained(self):
        """The control: without the head, a random decoder does produce decreasing curves.

        Without this the monotonicity test could pass on a decoder that happened to be
        monotone, which would make it a test of an accident rather than of a construction.
        """
        torch = pytest.importorskip("torch")

        torch.manual_seed(11)
        _encoder, decoder = build_autoencoder(n_points=201, latent=4, monotone=False)
        codes = torch.randn(32, 4, dtype=torch.get_default_dtype()) * 5.0
        with torch.no_grad():
            curves = decoder(codes).numpy()
        assert np.min(np.diff(curves, axis=1)) < 0.0

    def test_it_does_not_renormalize_its_terminal_value(self):
        """The defect deliberately left behind, asserted so it cannot come back.

        The salvaged ``ae_model.py`` divided every decoded damage curve by its own last value,
        which forces all of them to end at exactly 1 and destroys the amplitude the surrogate
        exists to predict. A decoder that reintroduced it would make this spread zero.
        """
        torch = pytest.importorskip("torch")

        torch.manual_seed(11)
        _encoder, decoder = build_autoencoder(n_points=201, latent=4, monotone=True)
        codes = torch.randn(16, 4, dtype=torch.get_default_dtype()) * 5.0
        with torch.no_grad():
            terminal = decoder(codes).numpy()[:, -1]
        assert float(terminal.std()) > 1.0e-6


class TestTheEnsembleMixture:
    """Ablation 3, the law of total variance rather than a spread quoted beside a mean."""

    def test_the_variance_grows_with_member_disagreement(self):
        base = np.zeros((5, 3, 7))
        variances = np.full((5, 3, 7), 0.25)
        agreeing_mean, agreeing_variance = ensemble_mixture(base, variances)
        disagreeing = base.copy()
        disagreeing[:, :, :] = np.arange(5).reshape(-1, 1, 1)
        _mean, disagreeing_variance = ensemble_mixture(disagreeing, variances)
        assert np.allclose(agreeing_mean, 0.0)
        assert np.allclose(agreeing_variance, 0.25)
        assert np.all(disagreeing_variance > agreeing_variance)
        assert np.allclose(disagreeing_variance, 0.25 + np.var(np.arange(5)))

    def test_the_variance_is_nonzero_away_from_the_member_mean(self):
        """Off the members' common mean the mixture must still carry the members' own variance.

        A mixture implemented as the variance of the member means alone would report exactly
        zero wherever the five members happen to agree, which is a confident prediction
        manufactured by an implementation detail.
        """
        rng = np.random.default_rng(_seed(3, 1))
        means = rng.normal(size=(5, 4, 11))
        variances = rng.gamma(shape=2.0, scale=0.5, size=(5, 4, 11))
        mixture_mean, mixture_variance = ensemble_mixture(means, variances)
        assert np.all(mixture_variance > 0.0)
        assert np.all(mixture_variance >= variances.mean(axis=0) - 1.0e-12)
        assert np.allclose(mixture_mean, means.mean(axis=0))

    def test_it_refuses_mismatched_inputs(self):
        with pytest.raises(ValueError, match="matching 3D arrays"):
            ensemble_mixture(np.zeros((5, 3, 7)), np.zeros((5, 3, 6)))


class TestTheSplineBasis:
    """Ablation 4, the identity the whole projection argument rests on."""

    def test_a_full_rank_basis_reproduces_any_curve_exactly(self):
        """At as many basis functions as stations the least squares fit is an interpolation.

        The knot vector here is the classical averaged one, interior knot ``j`` being the mean
        of the next ``degree`` stations, because that is the placement that satisfies the
        Schoenberg and Whitney condition and therefore makes the square design matrix
        invertible. The peak weighted placement the ablation ships does not: it deliberately
        bunches knots near the peak, which at full rank leaves a knot interval with no station
        in it and drops the matrix to rank 20 of 21. That is a property of a density weighted
        placement rather than a defect, and the ablation never runs at full rank; the identity
        under test is the one belonging to :func:`spline_basis` and :func:`fit_coefficients`.
        """
        grid = np.linspace(0.0, 20.0, 21)
        interior = np.array(
            [grid[index + 1 : index + 1 + DEGREE].mean() for index in range(grid.size - DEGREE - 1)]
        )
        knots = np.concatenate(
            [np.full(DEGREE + 1, grid[0]), interior, np.full(DEGREE + 1, grid[-1])]
        )
        basis = spline_basis(grid, knots, DEGREE)
        assert basis.shape == (21, 21)
        assert np.linalg.matrix_rank(basis) == 21
        rng = np.random.default_rng(_seed(4, 1))
        curves = rng.normal(size=(3, 21))
        rebuilt = fit_coefficients(basis, curves) @ basis.T
        np.testing.assert_allclose(rebuilt, curves, atol=1.0e-8)

    def test_a_curve_that_is_a_spline_is_recovered_in_its_own_coefficients(self):
        knots = peak_weighted_knots(0.0, 20.0, N_BASIS, DEGREE, peak_center=11.0)
        basis = spline_basis(U_GRID, knots, DEGREE)
        assert basis.shape == (U_GRID.size, N_BASIS)
        rng = np.random.default_rng(_seed(4, 2))
        coefficients = rng.normal(size=(4, N_BASIS))
        recovered = fit_coefficients(basis, coefficients @ basis.T)
        np.testing.assert_allclose(recovered, coefficients, atol=1.0e-8)

    def test_the_basis_is_a_partition_of_unity(self):
        knots = peak_weighted_knots(0.0, 20.0, N_BASIS, DEGREE, peak_center=11.0)
        basis = spline_basis(U_GRID, knots, DEGREE)
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1.0e-10)

    def test_the_knots_are_clamped_increasing_and_denser_near_the_peak(self):
        peak = 11.0
        knots = peak_weighted_knots(0.0, 20.0, N_BASIS, DEGREE, peak_center=peak)
        assert knots.size == N_BASIS + DEGREE + 1
        assert np.all(knots[: DEGREE + 1] == 0.0)
        assert np.all(knots[-DEGREE - 1 :] == 20.0)
        interior = knots[DEGREE + 1 : -DEGREE - 1]
        assert interior.size == N_BASIS - DEGREE - 1
        assert np.all(np.diff(interior) > 0.0)
        spacing = np.diff(np.concatenate([[0.0], interior, [20.0]]))
        centers = 0.5 * (
            np.concatenate([[0.0], interior]) + np.concatenate([interior, [20.0]])
        )
        near = spacing[np.abs(centers - peak) <= 2.5]
        far = spacing[np.abs(centers - peak) > 5.0]
        assert near.size and far.size
        assert near.mean() < far.mean(), (
            "the knot density is the one modeling choice this ablation makes, so a placement "
            "that is not denser near the peak is not the placement that was committed."
        )

    def test_it_refuses_a_peak_center_outside_the_stroke(self):
        with pytest.raises(ValueError, match="must lie inside"):
            peak_weighted_knots(0.0, 20.0, N_BASIS, DEGREE, peak_center=25.0)


@pytest.fixture(scope="module")
def design() -> np.ndarray:
    """A synthetic design with the campaign's own marginal spreads, in the feature order."""
    rng = np.random.default_rng(_seed(5, 0))
    return np.column_stack(
        [
            rng.normal(28.0, 2.8, size=60),
            rng.normal(27.0, 3.0, size=60),
            rng.normal(223.0, 5.0, size=60),
        ]
    )


class TestTheSubsamplers:
    """Ablation 5, where a design that moves between runs would measure nothing."""

    def test_the_random_subsampler_is_a_function_of_its_seed(self):
        first = random_subset(60, 24, _seed(5, 1))
        again = random_subset(60, 24, _seed(5, 1))
        different = random_subset(60, 24, _seed(5, 2))
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, different)
        assert first.size == 24 and np.unique(first).size == 24
        assert np.all(np.diff(first) > 0)

    def test_the_sobol_subsampler_is_a_function_of_its_seed(self, design):
        pytest.importorskip("scipy")
        first = sobol_guided_subset(design, 24, _seed(5, 3))
        again = sobol_guided_subset(design, 24, _seed(5, 3))
        different = sobol_guided_subset(design, 24, _seed(5, 4))
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, different)
        assert first.size == 24 and np.unique(first).size == 24

    def test_both_selections_return_the_whole_population_at_full_size(self, design):
        """The degeneracy at n = 198, asserted here rather than discovered in the results."""
        expected = np.arange(design.shape[0])
        np.testing.assert_array_equal(random_subset(60, 60, _seed(5, 5)), expected)
        np.testing.assert_array_equal(sobol_guided_subset(design, 60, _seed(5, 5)), expected)

    def test_the_sobol_selection_fills_the_space_better_on_average(self):
        """The mechanism the ablation predicts, checked on clean uniform synthetic designs.

        Measured by the covering radius, the largest distance from any design point to its
        nearest selected point, which is what a space filling selection exists to reduce. The
        property is statistical rather than per draw: on individual designs the random subset
        wins about two times in five, which is why this averages over eight seeded designs and
        why the ablation itself reports the spread across repetitions beside every mean.

        A uniform population on purpose. On a Gaussian one the Sobol points spread over a box
        whose corners hold almost no runs, so the greedy claim drags the selection outward and
        the covering radius gets worse. That is exactly the censoring failure mode written down
        in the prediction, and it is a finding about the campaign rather than a bug here.
        """
        radii = {"sobol": [], "random": []}
        for trial in range(8):
            rng = np.random.default_rng(_seed(5, 60 + trial))
            design = rng.uniform(size=(120, 3)) * np.array([10.0, 5.0, 20.0])
            seed = _seed(5, 80 + trial)
            for label, index in (
                ("sobol", sobol_guided_subset(design, 30, seed)),
                ("random", random_subset(design.shape[0], 30, seed)),
            ):
                distances = np.linalg.norm(design[:, None, :] - design[None, index, :], axis=2)
                radii[label].append(float(distances.min(axis=1).max()))
        assert np.mean(radii["sobol"]) < np.mean(radii["random"])

    def test_it_refuses_to_take_more_than_the_population(self, design):
        with pytest.raises(ValueError, match="cannot take"):
            random_subset(60, 61, _seed(5, 7))
        with pytest.raises(ValueError, match="cannot take"):
            sobol_guided_subset(design, 61, _seed(5, 7))


class TestTheSharedMetrics:
    """One definition per metric, because two would make every comparison meaningless."""

    def test_the_negative_log_density_matches_the_closed_form(self):
        truth = np.array([[1.0, 2.0]])
        mean = np.array([[1.5, 2.0]])
        variance = np.array([[0.25, 4.0]])
        expected = np.mean(
            [
                0.5 * (np.log(2 * np.pi * 0.25) + 0.25 / 0.25),
                0.5 * (np.log(2 * np.pi * 4.0) + 0.0),
            ]
        )
        assert gaussian_nlpd(truth, mean, variance) == pytest.approx(expected)

    def test_it_refuses_a_non_positive_variance_rather_than_flooring_it(self):
        with pytest.raises(ValueError, match="zero or negative"):
            gaussian_nlpd(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[1.0, 0.0]]))

    def test_coverage_of_a_calibrated_gaussian_is_the_nominal_level(self):
        rng = np.random.default_rng(_seed(6, 1))
        truth = rng.normal(size=(400, 50))
        mean = np.zeros_like(truth)
        variance = np.ones_like(truth)
        assert pointwise_coverage(truth, mean, variance, 0.9) == pytest.approx(0.9, abs=0.01)

    def test_the_curve_metrics_report_no_density_without_a_variance(self):
        truth = np.ones((3, 5))
        prediction = np.ones((3, 5)) * 1.1
        metrics = curve_metrics(truth, prediction, None)
        assert "nlpd" not in metrics and "coverage" not in metrics
        assert metrics["rmse"] == pytest.approx(0.1)
        assert metrics["n_curves"] == 3

    def test_the_peak_metrics_carry_the_sign_of_the_error(self):
        truth = np.array([[0.0, 10.0, 5.0], [0.0, 20.0, 8.0]])
        low = truth.copy()
        low[:, 1] -= 1.0
        metrics = peak_metrics(truth, low)
        assert metrics["peak_bias_N"] == pytest.approx(-1.0)
        assert metrics["peak_rmse_N"] == pytest.approx(1.0)

    def test_the_peak_curvature_is_negative_at_a_maximum(self):
        grid = np.linspace(0.0, 20.0, 201)
        curve = -((grid - 10.0) ** 2) + 100.0
        curvature = peak_curvature(curve.reshape(1, -1), grid)
        assert curvature.shape == (1,)
        assert curvature[0] == pytest.approx(-2.0, rel=1.0e-6)
