"""The propagation stage of build spec section 13, tested layer by layer.

The module is deliberately built out of pure functions on arrays, so most of what matters here
can be tested without a fitted surrogate and without an artifact store: the two layers, the
quantile standard error, the limit state counting, and the conservative bound are all
functions of numbers. That is what makes the central property of build spec 13.1 testable
through the public interface rather than by reaching into the stage: a zero predictive standard
deviation, passed in as an argument, must produce an epistemic layer with no spread at all.

The parts that genuinely need a Gaussian process fit one on synthetic data rather than reading
the artifact store, so they run on the full stack CI job where no artifacts exist. They carry
``pytest.importorskip`` for torch and gpytorch at the point of use and the ``fullstack`` marker,
which is the light stack lesson of docs/DEFECT_LOG.md applied before it could bite a fourth
time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufem.calibrate import jackknife_plus_intervals
from ufem.config import FEATURE_ORDER, load_config
from ufem.propagate import (
    BAND_ALPHA,
    LIMIT_STATES,
    QUANTILE_LEVELS,
    RESOLVABLE_PF_FLOOR,
    LimitState,
    binomial_standard_error,
    build_reliability_table,
    curve_envelope,
    density_curve,
    draw_inputs,
    epistemic_draws,
    failure_mask,
    jackknife_plus_rows,
    kernel_density,
    limit_state_result,
    posterior_pieces,
    predict_mean_and_variance,
    quantile_standard_error,
    quantile_summary,
    roughness_sentence,
    subsample_indices,
)


@pytest.fixture(scope="module")
def config(repo_root):
    return load_config(repo_root)


def _seed(entropy: int = 12345) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy)


class TestTheInputSample:
    def test_the_marginals_reproduce_the_declared_moments(self, config):
        """Binding law 2 end to end: the sample's moments are the config's, not a copy."""
        design = draw_inputs(config, 200000, _seed())
        model = config.probabilistic_model
        for position, name in enumerate(FEATURE_ORDER):
            column = design[:, position]
            variable = model.variables[name]
            declared_mean = getattr(variable, "mean", None)
            if declared_mean is None:
                declared_mean = variable.mu
                declared_std = variable.sigma
            else:
                declared_std = declared_mean * variable.cov
            assert column.mean() == pytest.approx(declared_mean, rel=0.01)
            assert column.std(ddof=1) == pytest.approx(declared_std, rel=0.02)

    def test_the_sample_depends_only_on_the_spawned_seed(self, config):
        first = draw_inputs(config, 5000, _seed())
        assert np.array_equal(first, draw_inputs(config, 5000, _seed()))
        other = draw_inputs(config, 5000, _seed(12346))
        assert not np.array_equal(first, other), (
            "two different entropies produced the same input sample, which is what a hard "
            "coded seed inside the sampler would look like"
        )

    def test_a_shorter_sample_is_the_prefix_of_a_longer_one(self, config):
        """One generator per column, drawn in order, so the sample size is not a seed.

        This is what makes a reduced rerun comparable with the full one: shrinking the sample
        removes draws from the end rather than producing a different sample.
        """
        long_sample = draw_inputs(config, 4000, _seed())
        short_sample = draw_inputs(config, 1000, _seed())
        assert np.array_equal(long_sample[:1000], short_sample)

    def test_the_columns_are_independent_of_each_other(self, config):
        design = draw_inputs(config, 100000, _seed())
        correlation = np.corrcoef(design, rowvar=False)
        off_diagonal = correlation[~np.eye(len(FEATURE_ORDER), dtype=bool)]
        assert np.max(np.abs(off_diagonal)) < 0.02

    def test_an_empty_sample_raises_rather_than_returning_an_empty_array(self, config):
        with pytest.raises(ValueError):
            draw_inputs(config, 0, _seed())

    def test_the_subsample_is_seeded_sorted_and_without_replacement(self):
        first = subsample_indices(10000, 500, _seed())
        assert first.size == 500
        assert np.array_equal(first, np.sort(first))
        assert np.unique(first).size == 500
        assert np.array_equal(first, subsample_indices(10000, 500, _seed()))
        assert not np.array_equal(first, subsample_indices(10000, 500, _seed(999)))

    def test_a_subsample_larger_than_the_sample_raises(self):
        with pytest.raises(ValueError):
            subsample_indices(100, 101, _seed())


class TestTheTwoLayers:
    """Build spec 13.1: the layers are separate, and the separation is a property."""

    def test_a_zero_predictive_variance_gives_an_epistemic_layer_with_no_spread(self):
        """The headline property, reachable through the public signature.

        No monkeypatching and nothing private: the epistemic layer takes the predictive
        standard deviation as an argument, so forcing it to zero is a call rather than a
        surgery. Every draw is then the mean exactly, which is what a surrogate claiming no
        uncertainty should propagate. A layer that returned a small nonzero spread here would
        be manufacturing uncertainty, which is binding law 1.
        """
        mean = np.linspace(30000.0, 45000.0, 500)
        draws = epistemic_draws(mean, np.zeros_like(mean), 64, _seed())
        assert draws.shape == (500, 64)
        assert np.array_equal(draws, np.repeat(mean[:, None], 64, axis=1))
        # Peak to peak rather than a standard deviation: the sample standard deviation of 64
        # identical values of order 3e4 comes out at 7e-12 through the two pass cancellation,
        # which is a property of the estimator and not of the layer. The spread being exactly
        # zero is a statement about the draws themselves.
        assert float(np.ptp(draws, axis=1).max()) == 0.0

    def test_the_aleatory_layer_ignores_the_draw_count_and_the_epistemic_layer_does_not(self):
        """The second half of the same statement, in the direction that catches a mix up.

        The aleatory summary is a function of the mean predictions alone, so it cannot move
        when the posterior draw count changes. The predictive summary is a function of the
        draws, so it must. A stage that had accidentally summarized the wrong array would fail
        one of these two assertions whichever way round the mistake went.
        """
        rng = np.random.default_rng(2)
        mean = 38000.0 + 3000.0 * rng.standard_normal(4000)
        sigma = np.full_like(mean, 1800.0)
        aleatory = quantile_summary(mean)
        few = quantile_summary(epistemic_draws(mean, sigma, 8, _seed()).ravel())
        many = quantile_summary(epistemic_draws(mean, sigma, 128, _seed()).ravel())
        assert aleatory == quantile_summary(mean)
        assert few["std"] > aleatory["std"]
        assert few["quantiles"] != many["quantiles"]
        assert many["std"] == pytest.approx(
            math.sqrt(aleatory["std"] ** 2 + 1800.0**2), rel=0.05
        )

    def test_the_epistemic_layer_is_seeded(self):
        mean = np.linspace(0.0, 1.0, 200)
        sigma = np.full_like(mean, 0.1)
        first = epistemic_draws(mean, sigma, 16, _seed())
        assert np.array_equal(first, epistemic_draws(mean, sigma, 16, _seed()))
        assert not np.array_equal(first, epistemic_draws(mean, sigma, 16, _seed(7)))

    def test_a_negative_standard_deviation_raises(self):
        with pytest.raises(ValueError):
            epistemic_draws(np.zeros(3), np.array([1.0, -1.0, 1.0]), 4, _seed())

    def test_mismatched_shapes_raise_rather_than_broadcast(self):
        with pytest.raises(ValueError):
            epistemic_draws(np.zeros(5), np.ones(4), 4, _seed())


class TestTheQuantileStandardError:
    def test_the_formula_matches_the_closed_form_for_a_normal_median(self):
        """A case with an exact answer: the median of a standard normal sample.

        The asymptotic standard error is ``sqrt(p(1-p)/n) / f(q_p)``, and at the median of a
        standard normal that density is ``1 / sqrt(2 pi)``, so the whole expression collapses
        to ``sqrt(pi / (2 n))``. The density is an argument to the function, so this test
        checks the formula and not an estimator.
        """
        n = 10000
        density = 1.0 / math.sqrt(2.0 * math.pi)
        assert quantile_standard_error(0.5, n, density) == pytest.approx(
            math.sqrt(math.pi / (2.0 * n)), rel=1.0e-12
        )

    def test_the_formula_matches_the_closed_form_in_a_tail(self):
        """The 5th percentile of a standard normal, where the density is not the modal one."""
        from scipy.stats import norm as standard_normal

        n = 40000
        level = 0.05
        quantile = standard_normal.ppf(level)
        density = standard_normal.pdf(quantile)
        expected = math.sqrt(level * (1.0 - level) / n) / density
        assert quantile_standard_error(level, n, density) == pytest.approx(expected, rel=1.0e-12)

    def test_the_predicted_error_matches_the_spread_of_repeated_samples(self):
        """The formula against the thing it predicts, measured rather than derived twice."""
        rng = np.random.default_rng(17)
        n, replicates = 4000, 400
        medians = np.median(rng.standard_normal((replicates, n)), axis=1)
        predicted = quantile_standard_error(0.5, n, 1.0 / math.sqrt(2.0 * math.pi))
        assert float(medians.std(ddof=1)) == pytest.approx(predicted, rel=0.10)

    def test_a_zero_density_refuses_to_report_a_standard_error(self):
        with pytest.raises(ValueError):
            quantile_standard_error(0.5, 100, 0.0)

    def test_an_impossible_level_raises(self):
        with pytest.raises(ValueError):
            quantile_standard_error(1.0, 100, 0.4)

    def test_the_summary_reports_every_requested_level_with_an_error(self):
        rng = np.random.default_rng(19)
        summary = quantile_summary(rng.standard_normal(20000))
        assert summary["levels"] == list(QUANTILE_LEVELS)
        assert len(summary["standard_error"]) == len(QUANTILE_LEVELS)
        assert all(value > 0.0 for value in summary["standard_error"])
        # The tails are harder to resolve than the middle, which is the whole point of
        # reporting the error per quantile rather than one number for the distribution.
        assert summary["standard_error"][0] > summary["standard_error"][3]

    def test_the_density_estimate_integrates_to_one(self):
        rng = np.random.default_rng(21)
        sample = rng.standard_normal(20000)
        grid = np.linspace(-6.0, 6.0, 2001)
        assert float(np.trapezoid(kernel_density(sample, grid), grid)) == pytest.approx(
            1.0, abs=0.01
        )

    def test_the_density_curve_covers_the_bulk_of_the_sample(self):
        rng = np.random.default_rng(23)
        grid, values = density_curve(rng.standard_normal(20000))
        assert grid.size == values.size
        assert float(values.min()) >= 0.0
        assert grid[0] < -2.0 < 2.0 < grid[-1]


class TestTheJackknifePlusFastPath:
    def test_it_agrees_exactly_with_the_calibration_stage_implementation(self):
        """The calibration stage stays the reference; this path only has to be faster.

        Both implementations take the same quantile of the same ensemble, so agreement is
        exact rather than approximate: they select the same order statistic, and selecting is
        not arithmetic.
        """
        rng = np.random.default_rng(29)
        n_models, n_query = 198, 400
        cross_mean = rng.normal(38000.0, 900.0, size=(n_query, n_models))
        cross_sigma = rng.uniform(600.0, 2200.0, size=(n_query, n_models))
        scores = np.abs(rng.standard_normal(n_models))
        fast_lower, fast_upper = jackknife_plus_rows(
            cross_mean, cross_sigma, scores, BAND_ALPHA
        )
        slow_lower, slow_upper = jackknife_plus_intervals(
            cross_mean, cross_sigma, scores, BAND_ALPHA
        )
        assert np.array_equal(fast_lower, slow_lower)
        assert np.array_equal(fast_upper, slow_upper)

    def test_the_interval_is_ordered_and_widens_with_the_scores(self):
        rng = np.random.default_rng(31)
        cross_mean = rng.normal(0.0, 1.0, size=(50, 60))
        cross_sigma = np.full((50, 60), 1.0)
        scores = np.abs(rng.standard_normal(60))
        lower, upper = jackknife_plus_rows(cross_mean, cross_sigma, scores, BAND_ALPHA)
        assert np.all(upper >= lower)
        wider_lower, wider_upper = jackknife_plus_rows(
            cross_mean, cross_sigma, 2.0 * scores, BAND_ALPHA
        )
        assert np.all(wider_upper - wider_lower >= upper - lower)

    def test_too_few_models_for_the_requested_level_raise(self):
        with pytest.raises(ValueError):
            jackknife_plus_rows(np.zeros((3, 5)), np.ones((3, 5)), np.ones(5), 0.01)

    def test_a_non_finite_cross_prediction_raises_rather_than_shrinking_the_ensemble(self):
        cross_mean = np.zeros((3, 40))
        cross_mean[0, 0] = np.nan
        with pytest.raises(ValueError):
            jackknife_plus_rows(cross_mean, np.ones((3, 40)), np.ones(40), BAND_ALPHA)


class TestTheLimitStates:
    """Build spec 13.2, on synthetic distributions whose failure probability is known."""

    @staticmethod
    def _uniform_case(n: int = 200000):
        """Predictions uniform on the unit interval, with a narrow interval around each."""
        rng = np.random.default_rng(37)
        mean = rng.uniform(0.0, 1.0, n)
        half_width = 0.05
        return mean, mean - half_width, mean + half_width

    def test_the_counted_probability_matches_the_known_one(self):
        """Uniform predictions on the unit interval: P(y < t) is exactly t."""
        mean, lower, upper = self._uniform_case()
        state = LimitState("synthetic", "P_max_N", "below", "synthetic", "synthetic", "test only")
        result = limit_state_result(
            state,
            0.30,
            mean,
            lower,
            upper,
            mean[:, None],
            mean,
            np.full_like(mean, 1.0e-9),
            np.ones(mean.size, dtype=bool),
        )
        assert result["pf_point"] == pytest.approx(0.30, abs=4.0 * result["pf_standard_error"])
        assert result["pf_standard_error"] == pytest.approx(
            math.sqrt(0.3 * 0.7 / mean.size), rel=0.02
        )
        assert result["pf_wilson_low"] < result["pf_point"] < result["pf_wilson_high"]

    def test_the_above_direction_is_the_complement_of_the_below_direction(self):
        mean, _lower, _upper = self._uniform_case()
        below = failure_mask(mean, 0.30, "below")
        above = failure_mask(mean, 0.30, "above")
        assert np.array_equal(below, ~above)

    def test_an_unknown_direction_raises(self):
        with pytest.raises(ValueError):
            failure_mask(np.zeros(3), 0.5, "sideways")

    def test_the_conservative_bound_is_never_below_the_point_estimate(self):
        """Build spec 13.2: a bound that could fall under its own estimate is not a bound.

        Checked across every threshold and both directions on the same synthetic sample, and
        with an interval deliberately offset from the mean so the union in the counting rule
        is exercised rather than assumed away.
        """
        rng = np.random.default_rng(41)
        mean = rng.normal(0.0, 1.0, 20000)
        lower = mean - rng.uniform(0.05, 0.6, mean.size)
        upper = mean + rng.uniform(0.05, 0.6, mean.size)
        for direction in ("below", "above"):
            for threshold in np.linspace(-2.5, 2.5, 21):
                state = LimitState(
                    "synthetic", "P_max_N", direction, "synthetic", "synthetic", "test only"
                )
                result = limit_state_result(
                    state,
                    float(threshold),
                    mean,
                    lower,
                    upper,
                    mean[:, None],
                    mean,
                    np.full_like(mean, 0.2),
                    np.ones(mean.size, dtype=bool),
                )
                assert result["pf_conservative"] >= result["pf_point"], (direction, threshold)

    def test_the_sampled_predictive_probability_matches_its_closed_form(self):
        """The epistemic layer's failure probability against the Gaussian integral of it."""
        rng = np.random.default_rng(43)
        mean = rng.normal(0.0, 1.0, 4000)
        sigma = np.full_like(mean, 0.5)
        draws = epistemic_draws(mean, sigma, 256, _seed())
        state = LimitState("synthetic", "P_max_N", "below", "synthetic", "synthetic", "test only")
        result = limit_state_result(
            state, 0.0, mean, mean - 1.0, mean + 1.0, draws, mean, sigma,
            np.ones(mean.size, dtype=bool),
        )
        assert result["pf_predictive"] == pytest.approx(
            result["pf_predictive_closed_form"], abs=0.01
        )

    def test_a_probability_under_the_floor_is_marked_unresolvable(self):
        mean = np.full(100000, 1.0)
        mean[:3] = -1.0
        state = LimitState("synthetic", "P_max_N", "below", "synthetic", "synthetic", "test only")
        result = limit_state_result(
            state, 0.0, mean, mean - 0.1, mean + 0.1, mean[:, None], mean,
            np.full_like(mean, 0.1), np.ones(mean.size, dtype=bool),
        )
        assert result["pf_point"] == 3.0e-5
        assert result["pf_point"] < RESOLVABLE_PF_FLOOR
        assert not result["resolvable"]

    def test_the_binomial_standard_error_is_the_closed_form(self):
        assert binomial_standard_error(0.25, 400) == pytest.approx(math.sqrt(0.25 * 0.75 / 400))
        assert binomial_standard_error(0.0, 400) == 0.0
        with pytest.raises(ValueError):
            binomial_standard_error(1.5, 400)


class TestTheDeclaredContract:
    def test_every_configured_limit_state_has_exactly_one_declaration(self, config):
        """A threshold in the config with no declaration here would never be evaluated."""
        configured = set(config.pipeline.limit_states.model_dump())
        declared = {state.config_field for state in LIMIT_STATES}
        assert declared == configured

    def test_every_limit_state_names_a_quantity_of_interest_and_a_direction(self):
        from ufem.surrogate import SCALAR_QOI

        for state in LIMIT_STATES:
            assert state.target in SCALAR_QOI
            assert state.direction in ("below", "above")
            assert len(state.justification) > 80

    def test_the_band_level_is_one_the_calibration_stage_measured(self, config):
        assert BAND_ALPHA in set(config.pipeline.conformal.alphas)

    def test_the_roughness_sentence_carries_the_measured_share(self):
        """The P6 caveat is generated from the P6 measurement, never remembered."""
        sentence = roughness_sentence(
            {"roughness_ratio": 0.393, "n_pairs": 20, "median_abs_difference": 1370.0}
        )
        assert "39 percent" in sentence
        assert "20 closest" in sentence
        assert "rough" in sentence


class TestTheValidityDomainMass:
    """Build spec 9.4 and 13.1: the fraction of Monte Carlo mass outside the domain."""

    def test_points_straddling_the_domain_are_classified_on_the_right_side(
        self, repo_root, config
    ):
        from ufem.validity import (
            ValidityDomainUnavailable,
            in_validity_domain,
            load_validity_domain,
        )

        try:
            domain = load_validity_domain(repo_root, config)
        except ValidityDomainUnavailable:
            pytest.skip("the audit stage has not run for this config hash")
        low = domain.bounds[:, 0]
        high = domain.bounds[:, 1]
        centre = 0.5 * (low + high)
        span = high - low
        query = np.vstack(
            [
                centre,
                centre + 2.0 * span,
                centre - 2.0 * span,
                np.array([centre[0], centre[1], high[2] + span[2]]),
            ]
        )
        inside = in_validity_domain(query, repo_root, config)
        assert not inside[1] and not inside[2] and not inside[3], (
            "a point outside the executed design box was reported inside the validity domain"
        )
        assert inside.dtype == np.dtype(bool)
        assert float((~inside).mean()) >= 0.75


@pytest.mark.fullstack
class TestTheAlgebraAgainstTheLibrary:
    """The fast matrix path must equal what GPyTorch computes, or it is not the same model."""

    @staticmethod
    def _fitted():
        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")
        from ufem.surrogate import GPSettings, Standardizer, fit_gp

        rng = np.random.default_rng(53)
        design = rng.uniform(-1.5, 1.5, size=(40, 3))
        response = np.sin(design[:, 0]) + 0.4 * design[:, 1] ** 2 - 0.3 * design[:, 2]
        settings = GPSettings(
            nu=2.5,
            ard=True,
            lengthscale_bounds=(0.11, 10.0),
            restarts=2,
            noise_prior_median_variance=0.1,
            noise_prior_log_scale=1.5,
            max_iterations=40,
        )
        gp, _log = fit_gp(design, response, "synthetic", settings, _seed(61))
        return gp, Standardizer(mean=np.zeros(1), scale=np.ones(1)), rng

    def test_the_batched_mean_and_variance_equal_the_library_prediction(self):
        gp, standardizer, rng = self._fitted()
        query = rng.uniform(-1.5, 1.5, size=(500, 3))
        pieces = posterior_pieces(gp)
        mean, variance = predict_mean_and_variance(
            gp, standardizer, query, pieces, chunk=64
        )
        reference_mean, reference_variance = gp.predict(query)
        np.testing.assert_allclose(mean, reference_mean, rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(variance, reference_variance, rtol=0.0, atol=1.0e-9)

    def test_the_chunk_size_changes_the_prediction_only_at_round_off(self):
        """The same answer, not the same bytes, and the reason is worth recording.

        The pairwise distances inside the kernel are computed by ``torch.cdist``, which
        switches between a direct evaluation and a matrix multiply formulation depending on the
        shapes it is handed. The two are algebraically identical and numerically are not, so a
        300 row block and eight 37 row blocks land about 3e-11 apart on values of order one.
        That is a library level blocking choice, the same class of effect the P6 chunk size
        test records for the pathwise sampler, and it is why the chunk sizes are module
        constants rather than arguments the caller picks: they are part of the artifact
        contract the bitwise determinism gate of build spec 17.2 holds under.
        """
        gp, standardizer, rng = self._fitted()
        query = rng.uniform(-1.5, 1.5, size=(300, 3))
        pieces = posterior_pieces(gp)
        whole = predict_mean_and_variance(gp, standardizer, query, pieces, chunk=300)
        pieces_of_it = predict_mean_and_variance(gp, standardizer, query, pieces, chunk=37)
        np.testing.assert_allclose(pieces_of_it[0], whole[0], rtol=0.0, atol=1.0e-8)
        np.testing.assert_allclose(pieces_of_it[1], whole[1], rtol=0.0, atol=1.0e-8)
        # And bitwise identical at a fixed chunk size, which is what the stage actually uses.
        repeated = predict_mean_and_variance(gp, standardizer, query, pieces, chunk=300)
        assert np.array_equal(repeated[0], whole[0])
        assert np.array_equal(repeated[1], whole[1])

    def test_the_prior_variance_is_the_kernel_diagonal(self):
        """The stationarity the batched variance formula relies on, asserted not assumed."""
        gp, _standardizer, rng = self._fitted()
        query = rng.uniform(-1.5, 1.5, size=(16, 3))
        diagonal = np.diag(gp.cross_covariance(gp.train_x))
        np.testing.assert_allclose(
            diagonal, np.full(gp.train_x.shape[0], gp.prior_variance()), rtol=0.0, atol=1.0e-12
        )
        assert gp.cross_covariance(query).shape == (16, gp.train_x.shape[0])

    def test_the_query_point_cross_predictions_reduce_to_the_training_point_ones(self):
        """The band at a query point uses the same identity the calibration uses at a design
        point, so evaluated at the design it has to give the calibration's own matrices back."""
        gp, _standardizer, _rng = self._fitted()
        pieces = posterior_pieces(gp)
        block = gp.cross_covariance(gp.train_x)
        projected = block @ pieces.inverse
        mean = pieces.constant_mean + block @ pieces.weights
        full = pieces.prior_variance + pieces.noise - np.einsum("ij,ij->i", projected, block)
        variance = full[:, None] + projected**2 / pieces.inverse_diagonal[None, :]
        reference_mean, reference_variance = gp.leave_one_out_cross_predictions()
        np.testing.assert_allclose(mean, reference_mean, rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(variance, reference_variance, rtol=0.0, atol=1.0e-9)


@pytest.mark.fullstack
class TestTheStageProducts:
    """What the stage wrote, read back. Skips when the pipeline has not run."""

    @staticmethod
    def _payload(repo_root, config):
        import json

        from ufem.config import config_hash
        from ufem.manifest import stage_dir
        from ufem.propagate import PROPAGATION_JSON, STAGE_NAME

        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
        )
        path = directory / PROPAGATION_JSON
        if not path.is_file():
            pytest.skip("the propagate stage has not run for this config hash")
        return json.loads(path.read_text(encoding="utf-8")), directory

    def test_the_reliability_table_states_the_out_of_domain_mass_on_every_row(
        self, repo_root, config
    ):
        payload, _directory = self._payload(repo_root, config)
        table = build_reliability_table(payload)
        share = f"{100.0 * payload['validity']['out_of_domain_fraction']:.1f}"
        assert table.count(share) == len(payload["limit_states"])
        for record in payload["limit_states"]:
            assert record["short_label"] in table

    def test_every_conservative_bound_is_at_or_above_its_point_estimate(self, repo_root, config):
        payload, _directory = self._payload(repo_root, config)
        for record in payload["limit_states"]:
            assert record["pf_conservative"] >= record["pf_point"], record["config_field"]

    def test_the_characteristic_value_agrees_with_the_configured_threshold(
        self, repo_root, config
    ):
        """Build spec 13.2: the committed threshold is the measured characteristic value.

        This is the gate on the revision taken in P7. If the propagated distribution moves far
        enough that the configured threshold is no longer its 5th percentile, this fails and
        the config is revised deliberately with a dated decision, rather than the limit state
        quietly becoming an arbitrary number again.
        """
        payload, _directory = self._payload(repo_root, config)
        characteristic = payload["characteristic_value"]
        assert characteristic["agrees"], (
            f"the configured peak load threshold {characteristic['configured_N']:.1f} N is "
            f"{100.0 * characteristic['relative_gap']:+.2f} percent from the measured "
            f"characteristic value {characteristic['value_N']:.1f} N"
        )

    def test_the_floor_and_the_roughness_caveat_reach_the_written_summary(
        self, repo_root, config
    ):
        from ufem.propagate import PROPAGATION_MD

        payload, directory = self._payload(repo_root, config)
        text = (directory / PROPAGATION_MD).read_text(encoding="utf-8")
        assert f"{RESOLVABLE_PF_FLOOR:.0e}" in text
        assert payload["roughness_caveat"] in text
        assert "outside the validity domain" in text

    def test_the_curve_envelope_is_ordered_at_every_displacement(self, repo_root, config):
        import pandas as pd

        from ufem.propagate import CURVE_ENVELOPE_PARQUET

        _payload, directory = self._payload(repo_root, config)
        frame = pd.read_parquet(directory / CURVE_ENVELOPE_PARQUET)
        for prefix in ("force_N", "damage"):
            columns = [f"{prefix}_p{level:02d}" for level in (5, 25, 50, 75, 95)]
            values = frame[columns].to_numpy(dtype=float)
            assert np.all(np.diff(values, axis=1) >= -1.0e-9), prefix

    def test_the_quantiles_are_increasing_in_their_level_for_every_target(
        self, repo_root, config
    ):
        import pandas as pd

        from ufem.propagate import QUANTILES_PARQUET

        _payload, directory = self._payload(repo_root, config)
        frame = pd.read_parquet(directory / QUANTILES_PARQUET)
        for _target, group in frame.groupby("target"):
            ordered = group.sort_values("level")
            assert np.all(np.diff(ordered["aleatory"].to_numpy(dtype=float)) > 0.0)


@pytest.mark.fullstack
class TestTheFailureRegionIsShadedOnTheFailingSide:
    """Regression test for the defect logged on 2026-08-30.

    The first version of this figure inferred which side of the threshold was the failure
    region from where the threshold fell relative to the median, which is backwards by
    construction: a below type limit state puts its threshold in the lower tail, exactly the
    case the heuristic read as an above type. All three panels shaded the safe side, and
    nothing but looking at the rendered figure would have caught it, which is why this test
    reads the shaded polygon rather than the code path.

    Guarded at the point of use rather than at module import: matplotlib is not in the light
    stack the fast CI jobs install.
    """

    @staticmethod
    def _frame():
        import pandas as pd

        values = np.linspace(0.0, 10.0, 64)
        return pd.DataFrame(
            {
                "target": "P_max_N",
                "value": values,
                "aleatory_density": np.exp(-((values - 5.0) ** 2) / 4.0),
                "predictive_density": np.exp(-((values - 5.0) ** 2) / 9.0),
            }
        )

    def _shaded_extent(self, direction: str, threshold: float):
        pytest.importorskip("matplotlib")
        from matplotlib import pyplot as plt

        from ufem.plotting.propagation import qoi_densities

        figure = qoi_densities(
            self._frame(),
            ["P_max_N"],
            {"P_max_N": threshold},
            {"P_max_N": direction},
            {"P_max_N": "Peak load"},
            {"P_max_N": 1.0},
            {"P_max_N": "kN"},
            0.4,
        )
        try:
            axis = figure.axes[0]
            # Two filled regions per panel, added in this order: the aleatory density and then
            # the failure region. The second is the one under test.
            vertices = np.vstack([path.vertices for path in axis.collections[1].get_paths()])
            return float(vertices[:, 0].min()), float(vertices[:, 0].max())
        finally:
            plt.close(figure)

    def test_a_below_limit_state_shades_only_under_its_threshold(self):
        low, high = self._shaded_extent("below", 3.0)
        assert high <= 3.0 + 1.0e-9
        assert low < 3.0

    def test_an_above_limit_state_shades_only_over_its_threshold(self):
        low, high = self._shaded_extent("above", 7.0)
        assert low >= 7.0 - 1.0e-9
        assert high > 7.0

    def test_an_unknown_direction_raises_rather_than_guessing(self):
        with pytest.raises(ValueError):
            self._shaded_extent("sideways", 5.0)


@pytest.mark.fullstack
class TestTheReadmeAgreesWithTheArtifact:
    """Ground rule 10: a README claim that disagrees with the manifest is a CI failure.

    The same gate the P5 row carries, applied to the P7 row. The status table is the one place
    in the README that quotes a number before the P10 injection, and a reliability number is
    exactly the kind that ages badly: the point estimate, its standard error, its bound and the
    out of domain mass all move together when the pipeline changes, and a row quoting three of
    the four correctly would be worse than one quoting none.
    """

    def test_the_status_row_quotes_the_measured_reliability_numbers(self, repo_root, config):
        from ufem.config import config_hash
        from ufem.manifest import load_manifest, stage_dir
        from ufem.propagate import STAGE_NAME

        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip("the propagate stage has not run for this config hash")
        extra = load_manifest(directory)["extra"]
        peak = extra["limit_states"]["peak_load_below_N"]

        rows = [
            line
            for line in (repo_root / "README.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| P7 |")
        ]
        assert len(rows) == 1, "the README status table must carry exactly one P7 row."
        row = rows[0]
        for value in (
            str(int(extra["n_samples"])),
            f"{peak['pf_point']:.4f}",
            f"{peak['pf_standard_error']:.5f}",
            f"{peak['pf_conservative']:.4f}",
            f"{100.0 * extra['out_of_domain_fraction']:.1f} percent",
            f"{extra['characteristic_value']['configured_N'] / 1000.0:.1f} kN",
        ):
            assert value in row, (
                f"the README P7 status row does not quote {value!r}, which is what the "
                f"propagate manifest at {directory} records. Ground rule 10: fix the README, "
                "not this test."
            )


@pytest.mark.fullstack
def test_the_curve_envelope_orders_its_percentiles_on_a_synthetic_family():
    """The envelope function itself, without the stage, on a family it cannot reorder."""
    pytest.importorskip("torch")

    class _Stub:
        def __init__(self):
            self.u_grid = np.linspace(0.0, 20.0, 21)

        def predict_curve(self, design):
            from ufem.surrogate import CurvePrediction

            scale = np.asarray(design, dtype=float)[:, :1]
            shape = np.sin(np.pi * self.u_grid / 20.0)[None, :]
            return CurvePrediction(
                u_grid=self.u_grid,
                force_mean=scale * shape,
                force_variance=np.ones((scale.size, self.u_grid.size)),
                damage_mean=scale * shape / 100.0,
                damage_variance=np.ones((scale.size, self.u_grid.size)),
            )

    design = np.column_stack([np.linspace(1.0, 2.0, 50), np.zeros(50), np.zeros(50)])
    frame = curve_envelope(_Stub(), design)
    columns = [f"force_N_p{level:02d}" for level in (5, 25, 50, 75, 95)]
    values = frame[columns].to_numpy(dtype=float)
    assert np.all(np.diff(values, axis=1) >= -1.0e-12)
    assert int(frame["n_curves"].iloc[0]) == 50
