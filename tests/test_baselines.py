"""Contract and property tests for the four mandatory baselines (build spec 10.5).

These need no torch and no artifact store, so they run in the fast CI job. That matters:
the baselines are the measuring stick, and a measuring stick that is only checked when the
heavy dependencies are installed is a measuring stick nobody checks.

What is pinned here is what makes the comparison fair rather than what makes the baselines
good. Each one has to be exactly what build spec 10.5 names, with no tuned quantity anywhere
in it, and each has to recover the function it is capable of representing exactly, so that a
failure in the harness cannot be mistaken for a weak baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufem.baselines import (
    CHAOS_DEGREE,
    ClimatologyRegressor,
    CurvePredictor,
    LinearRegressor,
    MeanCurveModel,
    NearestNeighborCurveModel,
    NearestNeighborRegressor,
    QuadraticChaosRegressor,
    Regressor,
    build_baseline_regressors,
    hermite_multi_indices,
    inverse_distance_weights,
    probabilists_hermite,
)


@pytest.fixture
def design():
    rng = np.random.default_rng(4242)
    return rng.uniform(-2.0, 2.0, size=(60, 3))


class TestTheInterfaceIsShared:
    def test_every_baseline_satisfies_the_regressor_protocol(self):
        for model in build_baseline_regressors(3):
            assert isinstance(model, Regressor), f"{model.name} is not a Regressor"

    def test_the_curve_models_satisfy_the_curve_protocol(self):
        X = np.random.default_rng(0).normal(size=(4, 3))
        curves = np.random.default_rng(1).normal(size=(4, 11))
        for model in (
            MeanCurveModel.fit(X, curves, curves),
            NearestNeighborCurveModel.fit(X, curves, curves, 3),
        ):
            assert isinstance(model, CurvePredictor), f"{model.name} is not a CurvePredictor"

    def test_the_four_baselines_are_the_four_the_spec_names(self):
        names = [model.name for model in build_baseline_regressors(3)]
        assert names == ["climatology", "linear", "quadratic_chaos", "nearest_neighbour"]


class TestClimatology:
    def test_it_predicts_the_training_mean_and_nothing_else(self, design):
        Y = design[:, :1] * 3.0 + 1.0
        model = ClimatologyRegressor().fit(design, Y)
        predicted = model.predict(np.zeros((5, 3)))
        assert predicted.shape == (5, 1)
        np.testing.assert_allclose(
            predicted, np.tile(Y.mean(axis=0), (5, 1)), rtol=0, atol=1e-12
        )

    def test_predicting_before_fitting_raises_rather_than_guessing(self):
        with pytest.raises(RuntimeError):
            ClimatologyRegressor().predict(np.zeros((1, 3)))


class TestLinear:
    def test_it_recovers_a_linear_function_exactly(self, design):
        coefficients = np.array([2.0, -1.5, 0.25])
        Y = (design @ coefficients + 7.0).reshape(-1, 1)
        model = LinearRegressor().fit(design, Y)
        query = np.array([[0.1, -0.3, 1.4], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(
            model.predict(query).ravel(), query @ coefficients + 7.0, rtol=1e-10, atol=1e-8
        )

    def test_it_is_invariant_to_an_affine_rescaling_of_the_features(self, design):
        Y = (design @ np.array([1.0, 2.0, -0.5])).reshape(-1, 1)
        shift, scale = np.array([10.0, -4.0, 200.0]), np.array([0.5, 3.0, 20.0])
        query = np.array([[0.2, 0.4, -1.1]])
        plain = LinearRegressor().fit(design, Y).predict(query)
        rescaled = (
            LinearRegressor()
            .fit(design * scale + shift, Y)
            .predict(query * scale + shift)
        )
        np.testing.assert_allclose(plain, rescaled, rtol=1e-8, atol=1e-9)


class TestQuadraticChaos:
    def test_the_term_count_is_the_full_quadratic_expansion_in_three_inputs(self):
        indices = hermite_multi_indices(3, CHAOS_DEGREE)
        assert len(indices) == 10, (
            "the full quadratic expansion over three inputs has (3 + 2) choose 2 = 10 terms. "
            "Build spec 10.5 says 15, which is the count for four inputs; the feature "
            "contract has three because E is derived."
        )
        assert indices[0] == (0, 0, 0)
        assert all(sum(index) <= CHAOS_DEGREE for index in indices)
        assert len(set(indices)) == len(indices)

    def test_the_hermite_polynomials_are_the_probabilists_family(self):
        x = np.linspace(-2.0, 2.0, 9)
        np.testing.assert_allclose(probabilists_hermite(x, 0), np.ones_like(x))
        np.testing.assert_allclose(probabilists_hermite(x, 1), x)
        np.testing.assert_allclose(probabilists_hermite(x, 2), x**2 - 1.0)

    def test_a_higher_order_is_refused_rather_than_silently_approximated(self):
        with pytest.raises(ValueError, match="orders 0 to 2"):
            probabilists_hermite(np.zeros(3), 3)

    def test_it_recovers_a_quadratic_function_exactly(self, design):
        def truth(x):
            return (
                1.0
                + 2.0 * x[:, 0]
                - 0.5 * x[:, 1]
                + 0.3 * x[:, 0] ** 2
                + 0.7 * x[:, 0] * x[:, 2]
                - 0.2 * x[:, 2] ** 2
            )

        model = QuadraticChaosRegressor().fit(design, truth(design).reshape(-1, 1))
        query = np.array([[0.5, -0.5, 1.0], [-1.0, 0.25, 0.75]])
        np.testing.assert_allclose(
            model.predict(query).ravel(), truth(query), rtol=1e-8, atol=1e-8
        )

    def test_an_underdetermined_fit_raises_rather_than_picking_a_solution(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="underdetermined"):
            QuadraticChaosRegressor().fit(rng.normal(size=(6, 3)), rng.normal(size=(6, 1)))


class TestNearestNeighbour:
    def test_the_weights_sum_to_one_and_favour_the_closer_point(self):
        train = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        order, weights = inverse_distance_weights(np.array([[0.2, 0.0, 0.0]]), train, 3)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        assert order[0, 0] == 0
        assert weights[0, 0] > weights[0, 1] > weights[0, 2]

    def test_a_query_on_a_training_point_returns_that_point_rather_than_an_infinity(self):
        train = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        _order, weights = inverse_distance_weights(train[:1], train, 3)
        assert np.all(np.isfinite(weights))
        np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])

    def test_it_reproduces_the_training_targets_at_the_training_inputs(self, design):
        Y = np.column_stack([design[:, 0], design[:, 1] ** 2])
        model = NearestNeighborRegressor(n_neighbors=3).fit(design, Y)
        np.testing.assert_allclose(model.predict(design), Y, rtol=1e-10, atol=1e-10)

    def test_too_few_training_rows_raises(self):
        with pytest.raises(ValueError, match="at least 3 training rows"):
            inverse_distance_weights(np.zeros((1, 3)), np.zeros((2, 3)), 3)

    def test_the_curve_model_averages_curves_not_scores(self, design):
        grid = np.linspace(0.0, 1.0, 21)
        force = np.array([np.sin(grid * (1.0 + row / 60.0)) for row in range(60)])
        model = NearestNeighborCurveModel.fit(design, force, force, 3)
        predicted, damage = model.predict_curves(design[:4])
        np.testing.assert_allclose(predicted, force[:4], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(damage, force[:4], rtol=1e-10, atol=1e-10)


class TestNoTuning:
    def test_no_baseline_exposes_a_quantity_fitted_on_the_data(self):
        """Ground rule for build spec 10.5: none of these has a tuned hyperparameter.

        The neighbour count is declared in the config and the polynomial degree in the build
        spec. Anything else that varied with the data would make the baseline a competitor
        that had been given a search budget the comparison never accounted for.
        """
        model = NearestNeighborRegressor(n_neighbors=3)
        assert model.n_neighbors == 3
        chaos = QuadraticChaosRegressor()
        assert chaos.degree == CHAOS_DEGREE
        rng = np.random.default_rng(2)
        X, Y = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
        chaos.fit(X, Y)
        assert chaos.degree == CHAOS_DEGREE
        assert chaos.n_terms == 10
