"""Contract and property tests for the Gaussian process surrogate (build spec 16.1).

The oracle problem is real here: nobody knows the true curve at an unseen input, so almost
nothing in this file compares a prediction against a known answer. What it pins instead are
the properties a correct implementation has to have whatever the answer is.

- A Gaussian process with a small noise reproduces its training targets at its training
  inputs, to within that noise. This is the interpolation property of build spec 16.1, and it
  is the one check that catches a design matrix silently transposed or a target silently
  shuffled.
- Standardizing the features makes the prediction invariant to an affine rescaling of the raw
  frame. A model that failed this would be reporting the units it was handed rather than the
  response.
- Permuting the rows of the training set changes nothing. A model that failed this would have
  learned something about the order the simulations happened to be extracted in.
- The restart policy is deterministic: the same entropy gives the same fit, bit for bit, and
  a different entropy gives a genuinely different set of starting points.

The fast tests build tiny synthetic problems and need only torch. The tests marked fullstack
read the artifact store and skip cleanly when the pipeline has not run.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from ufem.config import FEATURE_ORDER, config_hash, load_config  # noqa: E402
from ufem.manifest import stage_dir  # noqa: E402
from ufem.surrogate import (  # noqa: E402
    GP_PARAMETER_NAMES,
    LANDMARK_QOI,
    SCALAR_QOI,
    GPSettings,
    Standardizer,
    SurrogateModel,
    configure_torch,
    fit_gp,
    srsf_curve,
    srsf_tangent,
)
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE  # noqa: E402

pytestmark = pytest.mark.fullstack


def smooth_target(X: np.ndarray) -> np.ndarray:
    """A smooth analytic function of three inputs, used as a stand in for a response."""
    x = np.asarray(X, dtype=float)
    return (
        2.0 * np.sin(0.8 * x[:, 0])
        + 0.5 * x[:, 1] ** 2
        - 0.7 * x[:, 0] * x[:, 2]
        + 0.3 * x[:, 2]
    )


@pytest.fixture(scope="module")
def settings(repo_root) -> GPSettings:
    configure_torch()
    return GPSettings.from_config(load_config(repo_root))


@pytest.fixture(scope="module")
def toy(settings):
    """A 40 point design in three dimensions with a smooth, noiseless target."""
    rng = np.random.default_rng(20260830)
    X = rng.uniform(-1.5, 1.5, size=(40, 3))
    y = smooth_target(X)
    return X, (y - y.mean()) / y.std(ddof=1)


class TestInterpolation:
    def test_the_process_reproduces_its_training_targets_within_the_fitted_noise(
        self, toy, settings
    ):
        """Build spec 16.1: a GP interpolates its training points.

        Exact interpolation is only the zero noise limit, so the assertion is against the
        noise the fit actually chose: the residual at a training input must be small compared
        with the standard deviation the model itself reports there. Asserting exact equality
        would be asserting that the noise is zero, which this project deliberately does not
        allow anyone to assume.
        """
        X, y = toy
        gp, _log = fit_gp(X, y, "toy", settings, np.random.SeedSequence(1))
        mean, variance = gp.predict(X, include_noise=False)
        residual = np.abs(mean - y)
        assert np.max(residual) < 4.0 * np.sqrt(gp.noise() + variance.max()), (
            f"the largest training residual is {np.max(residual):.4g} against a fitted noise "
            f"standard deviation of {np.sqrt(gp.noise()):.4g}; the process is not passing "
            "through its own training data."
        )
        assert np.max(residual) < 0.2 * float(np.std(y))

    def test_a_noiseless_smooth_target_is_fitted_with_a_small_noise(self, toy, settings):
        """The hyperprior is a center, not a floor: a clean target pulls the noise below it."""
        X, y = toy
        gp, _log = fit_gp(X, y, "toy", settings, np.random.SeedSequence(2))
        assert gp.noise() < settings.noise_prior_median_variance, (
            f"the fitted noise is {gp.noise():.3e} against a prior median of "
            f"{settings.noise_prior_median_variance:.3e}. On a noiseless target the data "
            "should pull the noise below the center; if it cannot, the prior is acting as a "
            "floor, which ground rule 4 forbids."
        )


class TestInvariances:
    def test_an_affine_rescaling_of_the_features_leaves_predictions_unchanged(
        self, toy, settings
    ):
        """Standardization is what makes this true, so it is a test of the standardizer.

        Every feature column is shifted and stretched by a different amount and the whole fit
        is repeated. Because both fits standardize by their own training statistics, the
        standardized designs are identical and so is everything downstream.
        """
        X, y = toy
        shift = np.array([12.0, -3.0, 250.0])
        scale = np.array([0.25, 40.0, 7.0])
        query = np.array([[0.4, -0.9, 1.1], [-1.2, 0.3, 0.0]])

        first = Standardizer.fit(X)
        gp_a, _ = fit_gp(first.transform(X), y, "a", settings, np.random.SeedSequence(3))
        mean_a, var_a = gp_a.predict(first.transform(query))

        rescaled = X * scale + shift
        second = Standardizer.fit(rescaled)
        gp_b, _ = fit_gp(
            second.transform(rescaled), y, "b", settings, np.random.SeedSequence(3)
        )
        mean_b, var_b = gp_b.predict(second.transform(query * scale + shift))

        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(var_a, var_b, rtol=1e-8, atol=1e-10)

    def test_permuting_the_training_rows_leaves_predictions_unchanged(self, toy, settings):
        """A model that failed this would have learned the extraction order of the campaign."""
        X, y = toy
        query = np.array([[0.2, 0.7, -0.4]])
        gp_a, _ = fit_gp(X, y, "a", settings, np.random.SeedSequence(4))
        order = np.random.default_rng(7).permutation(X.shape[0])
        gp_b, _ = fit_gp(X[order], y[order], "b", settings, np.random.SeedSequence(4))
        mean_a, var_a = gp_a.predict(query)
        mean_b, var_b = gp_b.predict(query)
        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(var_a, var_b, rtol=1e-6, atol=1e-8)


class TestRestartPolicy:
    def test_the_same_entropy_gives_the_same_fit(self, toy, settings):
        X, y = toy
        first, log_a = fit_gp(X, y, "a", settings, np.random.SeedSequence(11))
        second, log_b = fit_gp(X, y, "b", settings, np.random.SeedSequence(11))
        np.testing.assert_array_equal(first.parameters, second.parameters)
        assert [record["init_noise"] for record in log_a] == [
            record["init_noise"] for record in log_b
        ]

    def test_a_different_entropy_gives_different_starting_points(self, toy, settings):
        """Guards a hard coded seed: the draws must actually depend on the entropy."""
        X, y = toy
        _first, log_a = fit_gp(X, y, "a", settings, np.random.SeedSequence(11))
        _second, log_b = fit_gp(X, y, "b", settings, np.random.SeedSequence(12))
        assert [record["init_noise"] for record in log_a] != [
            record["init_noise"] for record in log_b
        ]

    def test_every_restart_is_logged_with_its_outcome(self, toy, settings):
        X, y = toy
        _gp, log = fit_gp(X, y, "a", settings, np.random.SeedSequence(13))
        assert len(log) == settings.restarts
        for record in log:
            assert record["status"] in {"converged", "iteration_limit", "failed"}
            if record["status"] != "failed":
                assert np.isfinite(record["marginal_log_likelihood"])

    def test_the_best_restart_is_the_one_that_is_kept(self, toy, settings):
        X, y = toy
        gp, log = fit_gp(X, y, "a", settings, np.random.SeedSequence(14))
        best = max(
            record["marginal_log_likelihood"]
            for record in log
            if record["status"] != "failed"
        )
        assert gp.marginal_log_likelihood == pytest.approx(best, rel=0, abs=1e-12)


class TestSquareRootSlopeRepresentation:
    def test_a_reconstructed_monotone_map_is_monotone_by_construction(self):
        """The reason the displacement block is not a linear principal component basis."""
        stations = np.linspace(0.0, 1.0, 201)
        rng = np.random.default_rng(5)
        family = np.empty((30, stations.size))
        for row in range(30):
            power = float(np.exp(rng.uniform(-0.6, 0.6)))
            family[row] = stations**power
        tangent, mean_psi = srsf_tangent(family)
        for row in range(30):
            rebuilt = srsf_curve(mean_psi, tangent[row], stations)
            assert np.all(np.diff(rebuilt) > 0.0)
            assert rebuilt[0] == pytest.approx(0.0, abs=1e-12)
            assert rebuilt[-1] == pytest.approx(1.0, abs=1e-12)

    def test_the_round_trip_recovers_the_family_it_came_from(self):
        stations = np.linspace(0.0, 1.0, 201)
        rng = np.random.default_rng(6)
        family = np.empty((20, stations.size))
        for row in range(20):
            family[row] = stations ** float(np.exp(rng.uniform(-0.4, 0.4)))
        tangent, mean_psi = srsf_tangent(family)
        errors = [
            np.abs(srsf_curve(mean_psi, tangent[row], stations) - family[row]).max()
            for row in range(20)
        ]
        assert max(errors) < 5e-3, (
            f"the square root slope round trip loses {max(errors):.3e}, which is above the "
            "discretization floor this representation was measured at."
        )


@pytest.fixture(scope="module")
def fitted(repo_root):
    """The production surrogate, or a skip when the stage has not run."""
    config = load_config(repo_root)
    digest = config_hash(config)
    artifact_root = repo_root / config.pipeline.paths.artifact_root
    if not (stage_dir(artifact_root, SURROGATE_STAGE, digest) / "surrogate.json").is_file():
        pytest.skip("the surrogate stage has not run for this config hash")
    return SurrogateModel.load(artifact_root, digest)


class TestTheFittedArtifact:
    def test_the_feature_contract_is_the_pinned_one(self, fitted):
        assert tuple(fitted.metadata["feature_order"]) == FEATURE_ORDER
        assert "E_MPa" not in fitted.metadata["feature_order"]

    def test_every_scalar_qoi_and_landmark_has_its_own_process(self, fitted):
        for name in (*SCALAR_QOI, *LANDMARK_QOI):
            assert name in fitted.models, f"no Gaussian process for {name}"
        assert set(fitted.scalar_targets) == set(SCALAR_QOI) | set(LANDMARK_QOI)

    def test_the_state_dict_carries_the_pinned_parameter_blocks(self, fitted):
        assert tuple(fitted.metadata["gp_parameter_names"]) == GP_PARAMETER_NAMES
        for model in fitted.models.values():
            assert model.parameters.shape == (6,)

    def test_the_lengthscale_lower_bound_is_not_below_the_design_site_spacing(self, fitted):
        """Build spec 10.3 justifies the bound by the minimum site spacing, so check it.

        A lengthscale shorter than the closest pair of design points describes a correlation
        the design cannot observe, and a kernel allowed to go there will use it to interpolate
        scatter. That is the documented failure of build spec 5.2, and this is the assertion
        that the configured bound is the one the spec's justification implies.
        """
        train_x = next(iter(fitted.models.values())).train_x
        distances = np.linalg.norm(train_x[:, None, :] - train_x[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        spacing = float(distances.min())
        low, _high = fitted.settings.lengthscale_bounds
        assert low >= 0.9 * spacing, (
            f"the lengthscale lower bound is {low} but the minimum nearest neighbour distance "
            f"in the standardized design is {spacing:.4f}. Build spec 10.3 sets the bound from "
            "the site spacing; a bound below it lets the kernel interpolate scatter."
        )

    def test_predictions_at_the_training_design_are_finite_and_in_range(self, fitted):
        query = np.array(
            [fitted.metadata["feature_standardization"]["mean"]], dtype=float
        )
        prediction = fitted.predict_curve(query)
        assert np.all(np.isfinite(prediction.force_mean))
        assert np.all(prediction.force_variance > 0.0)
        assert prediction.force_mean.shape == (1, len(fitted.metadata["u_grid_mm"]))
        peak = float(prediction.force_mean.max())
        assert 20000.0 < peak < 60000.0, (
            f"the surrogate predicts a peak load of {peak:.0f} N at the design centre, where "
            "the campaign measured 38.1 kN on average. That is not a plausible prediction."
        )

    def test_the_scalar_qoi_come_from_their_own_processes_not_from_the_curve(self, fitted):
        """Build spec 10.4: reading a QoI off the reconstructed curve is a diagnostic only.

        The two must differ, because the curve is a truncated reconstruction and the scalar
        process is fitted on the exact measurement. If they agreed to machine precision the
        scalar processes would be dead code.
        """
        query = np.array([fitted.metadata["feature_standardization"]["mean"]], dtype=float)
        from_curve = float(fitted.predict_curve(query).force_mean.max())
        from_process = float(fitted.predict_qoi(query)["P_max_N"][0][0])
        assert from_curve != from_process
        assert abs(from_curve - from_process) / from_process < 0.25

    def test_the_curve_variance_grows_where_the_family_spreads(self, fitted):
        """The softening branch carries most of the campaign's variance, so the band must too."""
        query = np.array([fitted.metadata["feature_standardization"]["mean"]], dtype=float)
        prediction = fitted.predict_curve(query)
        u_grid = np.array(fitted.metadata["u_grid_mm"], dtype=float)
        early = prediction.force_std()[0][int(np.argmin(np.abs(u_grid - 2.0)))]
        late = prediction.force_std()[0][int(np.argmin(np.abs(u_grid - 15.0)))]
        assert late > early

    def test_posterior_draws_are_reproducible_and_spread(self, fitted):
        query = np.array([fitted.metadata["feature_standardization"]["mean"]], dtype=float)
        first = fitted.draw_curves(query, 8, np.random.SeedSequence(99))
        second = fitted.draw_curves(query, 8, np.random.SeedSequence(99))
        np.testing.assert_array_equal(first, second)
        third = fitted.draw_curves(query, 8, np.random.SeedSequence(100))
        assert not np.array_equal(first, third)
        assert first.std(axis=1).mean() > 0.0

    def test_the_fit_budget_of_build_spec_10_3_was_met(self, repo_root):
        from ufem.manifest import load_manifest

        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            SURROGATE_STAGE,
            config_hash(config),
        )
        extra = load_manifest(directory)["extra"]
        assert extra["fit_budget_met"], (
            f"the Gaussian process fits took {extra['gp_fit_wall_time_s']:.1f} s against the "
            f"{extra['fit_budget_s']:.0f} s budget of build spec 10.3. The spec says to stop "
            "and look rather than to widen the budget."
        )

    def test_the_stage_registration_agrees_with_the_reduce_stage(self, repo_root):
        from ufem.manifest import load_manifest

        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            SURROGATE_STAGE,
            config_hash(config),
        )
        agreement = load_manifest(directory)["extra"]["registration_agreement"]
        assert agreement["loading_max_abs_deviation"] <= agreement["tolerance"]
        assert agreement["mean_max_abs_deviation_N"] <= agreement["tolerance"]


class TestSettingsContract:
    def test_the_gp_settings_come_from_the_config_and_nowhere_else(self, repo_root):
        config = load_config(repo_root)
        settings = GPSettings.from_config(config)
        assert settings.nu == config.pipeline.kernel.nu
        assert settings.restarts == config.pipeline.kernel.restarts
        assert (
            settings.noise_prior_median_variance
            == config.pipeline.surrogate.noise_prior_median_variance
        )

    def test_a_zero_restart_setting_is_rejected_by_the_config(self, repo_root):
        """A restart policy of nothing would silently become a single arbitrary start."""
        from pydantic import ValidationError

        from ufem.config import KernelSettings

        declared = load_config(repo_root).pipeline.kernel.model_dump()
        with pytest.raises(ValidationError):
            KernelSettings(**{**declared, "restarts": 0})
        with pytest.raises(ValidationError):
            KernelSettings(**{**declared, "lengthscale_bounds": (10.0, 0.05)})
