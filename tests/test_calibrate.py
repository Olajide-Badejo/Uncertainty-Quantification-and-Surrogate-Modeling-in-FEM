"""The calibration stage: the algebra, the properties, the gate, and the ban on styling.

Build spec 11 and 16.1. The oracle problem is at its worst here, because nobody knows the
true predictive distribution at an unseen input, so almost nothing in this file is an example
based test. What it pins instead is: closed form algebra against brute force refits, the
properties a conformal interval must have whatever the data is, determinism of the measured
scaling, and that the gate raises on inputs constructed to be miscalibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufem.calibrate import (
    CURVE_SIGNALS,
    GATE_ALPHA,
    GATE_PIT_OUTER_MASS_MAX,
    SIGNAL_DAMAGE,
    SIGNAL_FORCE,
    CalibrationGateFailed,
    climatology_leave_one_out,
    crps_gaussian,
    enforce_gate,
    evaluate_gate,
    informative_abscissae,
    jackknife_plus_intervals,
    leave_one_out_scaling_factors,
    negative_log_predictive_density,
    pit_by_abscissa,
    pit_outer_mass,
    predictive_variance_adequacy,
    variance_scaling_factor,
    wilson_interval,
)

pytestmark = pytest.mark.fullstack


class TestTheMeasurements:
    def test_the_variance_scaling_factor_is_the_root_mean_square_standardized_residual(self):
        z = np.array([1.0, -2.0, 3.0, -4.0])
        assert variance_scaling_factor(z) == pytest.approx(np.sqrt(30.0 / 4.0))

    def test_scaling_by_the_factor_sets_the_adequacy_to_zero(self):
        rng = np.random.default_rng(11)
        z = 2.4 * rng.standard_normal(500)
        tau = variance_scaling_factor(z)
        assert predictive_variance_adequacy(z / tau) == pytest.approx(0.0, abs=1e-12)
        assert predictive_variance_adequacy(z) == pytest.approx(2.0 * np.log(tau), abs=1e-12)

    def test_the_adequacy_sign_says_which_way_the_model_is_wrong(self):
        """Positive is overconfident, negative is underconfident. The sign is the diagnosis."""
        assert predictive_variance_adequacy(np.full(50, 2.0)) > 0.0
        assert predictive_variance_adequacy(np.full(50, 0.5)) < 0.0

    def test_the_scaling_is_deterministic_and_depends_only_on_the_residuals(self):
        rng = np.random.default_rng(4)
        z = rng.standard_normal(300)
        first = variance_scaling_factor(z)
        assert variance_scaling_factor(z) == first
        assert variance_scaling_factor(z[::-1]) == pytest.approx(first, rel=0.0, abs=1e-15)
        assert variance_scaling_factor(rng.permutation(z)) == pytest.approx(first, abs=1e-12)

    def test_the_leave_one_out_factors_match_an_explicit_recomputation(self):
        rng = np.random.default_rng(5)
        z = rng.standard_normal(40) * 1.7
        factors = leave_one_out_scaling_factors(z)
        for index in range(z.size):
            expected = variance_scaling_factor(np.delete(z, index))
            assert factors[index] == pytest.approx(expected, rel=1e-12)

    def test_the_wilson_interval_brackets_the_proportion_and_stays_inside_zero_one(self):
        low, high = wilson_interval(179, 198)
        assert low < 179 / 198 < high
        assert 0.0 < low and high < 1.0
        # A perfect count still gets an interval with an upper end below one.
        low, high = wilson_interval(198, 198)
        assert high < 1.0000001 and low < 1.0

    def test_the_gaussian_crps_matches_a_numerical_integration(self):
        from scipy import integrate, stats

        truth, mu, sigma = 1.3, 0.4, 0.9
        closed = float(crps_gaussian(np.array([truth]), np.array([mu]), np.array([sigma]))[0])
        # Integrated in two pieces split at the observation: the indicator is a step there,
        # and a rule that straddles the discontinuity converges too slowly to catch a real
        # error in the closed form (a 20001 point Simpson still disagrees in the fourth
        # decimal, which is larger than the mistakes worth catching).

        def integrand(x: float) -> float:
            return (stats.norm.cdf(x, mu, sigma) - float(x >= truth)) ** 2

        numerical = sum(
            integrate.quad(integrand, low, high, limit=200)[0]
            for low, high in ((mu - 40 * sigma, truth), (truth, mu + 40 * sigma))
        )
        assert closed == pytest.approx(numerical, rel=1e-10)

    def test_the_nlpd_of_a_standard_normal_residual_is_its_entropy(self):
        rng = np.random.default_rng(8)
        z = rng.standard_normal(200000)
        measured = negative_log_predictive_density(z, np.zeros_like(z), np.ones_like(z))
        assert measured == pytest.approx(0.5 * np.log(2 * np.pi) + 0.5, abs=0.01)

    def test_the_climatology_leaves_the_point_out(self):
        values = np.array([1.0, 2.0, 3.0, 10.0])
        mean, sigma = climatology_leave_one_out(values)
        assert mean[3] == pytest.approx(2.0)
        assert sigma[3] == pytest.approx(np.std(np.array([1.0, 2.0, 3.0]), ddof=1))


class TestTheIntervalProperties:
    """What a conformal interval must satisfy whatever the data happens to be."""

    @staticmethod
    def _ensemble(n=60, seed=1):
        rng = np.random.default_rng(seed)
        mean = rng.normal(size=(n, n)) * 0.05 + 3.0
        np.fill_diagonal(mean, np.nan)
        sigma = np.full((n, n), 0.4) + rng.uniform(0.0, 0.05, size=(n, n))
        scores = np.abs(rng.standard_normal(n))
        return mean, sigma, scores

    def test_a_ninety_five_percent_interval_is_wider_than_a_ninety_percent_one(self):
        mean, sigma, scores = self._ensemble()
        low90, high90 = jackknife_plus_intervals(mean, sigma, scores, 0.10)
        low95, high95 = jackknife_plus_intervals(mean, sigma, scores, 0.05)
        assert np.all(high95 - low95 >= high90 - low90)
        assert np.mean(high95 - low95) > np.mean(high90 - low90)

    def test_the_interval_brackets_its_own_centre(self):
        mean, sigma, scores = self._ensemble(seed=6)
        lower, upper = jackknife_plus_intervals(mean, sigma, scores, 0.10)
        centre = np.nanmedian(mean, axis=1)
        assert np.all(lower <= centre) and np.all(centre <= upper)

    def test_a_level_the_sample_cannot_support_raises(self):
        mean, sigma, scores = self._ensemble(n=8)
        with pytest.raises(ValueError, match="does not exist at this sample size"):
            jackknife_plus_intervals(mean, sigma, scores, 0.01)

    def test_larger_scores_give_wider_intervals_and_nothing_shrinks_them(self):
        """Ground rule 4 from the other side: the interval is a function of the scores alone."""
        mean, sigma, scores = self._ensemble(seed=9)
        lower, upper = jackknife_plus_intervals(mean, sigma, scores, 0.10)
        wider_low, wider_high = jackknife_plus_intervals(mean, sigma, 1.5 * scores, 0.10)
        assert np.all(wider_high - wider_low >= upper - lower)


class TestTheDomainRule:
    def test_a_constant_abscissa_is_excluded_and_the_rest_are_kept(self):
        family = np.tile(np.linspace(1.0, 2.0, 5), (7, 1))
        family[:, 2] += np.arange(7) * 0.1
        mask = informative_abscissae(family)
        assert mask.tolist() == [False, False, True, False, False]

    def test_a_wholly_constant_family_raises_rather_than_returning_nothing(self):
        with pytest.raises(ValueError, match="nothing"):
            informative_abscissae(np.ones((4, 6)))


class TestThePitDiagnostics:
    def test_a_calibrated_residual_field_gives_a_flat_histogram(self):
        rng = np.random.default_rng(2)
        z = rng.standard_normal((4000, 6))
        histogram = pit_by_abscissa(z)
        assert histogram.shape == (10, 6)
        np.testing.assert_allclose(histogram.sum(axis=0), np.ones(6), atol=1e-12)
        assert np.abs(histogram - 0.1).max() < 0.02

    def test_an_overconfident_field_puts_mass_in_the_outer_deciles(self):
        rng = np.random.default_rng(3)
        grid = np.linspace(0.0, 20.0, 21)
        peak = np.full(500, 2.0)
        calibrated = rng.standard_normal((500, grid.size))
        assert pit_outer_mass(calibrated, grid, peak)["outer_mass"] == pytest.approx(0.2, abs=0.02)
        overconfident = 2.0 * calibrated
        measured = pit_outer_mass(overconfident, grid, peak)["outer_mass"]
        assert measured > GATE_PIT_OUTER_MASS_MAX, (
            "a predictive standard deviation half of what it should be must trip the gate "
            f"threshold, but the statistic measured {measured:.3f}."
        )

    def test_a_family_with_no_post_peak_abscissa_raises(self):
        with pytest.raises(ValueError, match="softening branch"):
            pit_outer_mass(np.zeros((3, 4)), np.linspace(0.0, 1.0, 4), np.full(3, 5.0))


def _gate_inputs(coverage: float = 0.904, outer_mass: float = 0.11):
    """Minimal scalar, functional and PIT records shaped as ``evaluate_gate`` reads them."""
    key = f"{GATE_ALPHA:g}"
    low, high = wilson_interval(int(round(coverage * 198)), 198)
    band = {"empirical": coverage, "wilson_low": low, "wilson_high": high}
    scalar = {name: {"jackknife_plus": {key: dict(band)}} for name in ("P_max_N", "u_peak_mm")}
    functional = {signal: {"bands": {key: dict(band)}} for signal in CURVE_SIGNALS}
    pit = {
        SIGNAL_FORCE: {"after": {"outer_mass": outer_mass}},
        SIGNAL_DAMAGE: {"after": {"outer_mass": outer_mass}},
    }
    return scalar, functional, pit, ["P_max_N", "u_peak_mm"]


class TestTheGate:
    def test_a_calibrated_model_passes(self):
        gate = evaluate_gate(*_gate_inputs())
        assert gate["passed"] and gate["failing"] == []

    def test_an_undercovering_band_fails_and_is_named(self):
        gate = evaluate_gate(*_gate_inputs(coverage=0.70))
        assert not gate["passed"]
        assert any("simultaneous" in name for name in gate["failing"])
        assert any("jackknife+" in name for name in gate["failing"])

    def test_an_overcovering_band_also_fails(self):
        """A 90 percent interval that covers 99 percent of the time is not calibrated either."""
        gate = evaluate_gate(*_gate_inputs(coverage=0.99))
        assert not gate["passed"]

    def test_a_u_shaped_pit_on_the_softening_branch_fails(self):
        gate = evaluate_gate(*_gate_inputs(outer_mass=0.52))
        assert not gate["passed"]
        assert any("PIT" in name for name in gate["failing"])

    def test_the_stage_raises_on_a_failing_gate_rather_than_returning(self, tmp_path):
        """The failure path build spec 11.5 requires: nonzero exit, named diagnostic.

        ``enforce_gate`` is the production call the stage makes between writing its
        diagnostics and writing its manifest, so this exercises the real failure path rather
        than a copy of it. ``ufem.runner.run_stage`` turns the exception into a nonzero exit,
        and because it is raised before ``write_manifest``, a failed calibration leaves no
        manifest and therefore can never be served as a cache hit.
        """
        gate = evaluate_gate(*_gate_inputs(coverage=0.60, outer_mass=0.61))
        with pytest.raises(CalibrationGateFailed, match="never band styling") as raised:
            enforce_gate(gate, tmp_path)
        message = str(raised.value)
        assert "simultaneous" in message and "PIT" in message
        assert str(tmp_path) in message

    def test_a_passing_gate_lets_the_stage_continue(self, tmp_path):
        assert enforce_gate(evaluate_gate(*_gate_inputs()), tmp_path) is None


class TestNoStylingIsPossible:
    """Ground rule 4, as a property of the module rather than as an intention."""

    def test_the_module_contains_no_banned_identifier_and_no_variance_floor(self):
        from pathlib import Path

        import ufem.calibrate as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in ("AMPLIFY", "noise_level"):
            assert banned not in source
        for floor in ("np.maximum(sigma", "np.clip(variance", "np.clip(sigma", "+ 1e-"):
            assert floor not in source, f"{floor!r} looks like a floor under a variance."

    def test_the_conformal_module_never_touches_the_modulation_it_is_given(self):
        from ufem.conformal_functional import simultaneous_band, sup_norm_scores

        rng = np.random.default_rng(12)
        modulation = 1.0 + rng.uniform(0.0, 1.0, size=(5, 30))
        original = modulation.copy()
        truth = rng.standard_normal((5, 30))
        mean = np.zeros((5, 30))
        sup_norm_scores(truth, mean, modulation)
        simultaneous_band(mean, modulation, 2.0)
        np.testing.assert_array_equal(modulation, original)


class TestTheReadmeAgreesWithTheArtifact:
    """Ground rule 10: a README claim that disagrees with the manifest is a CI failure.

    Until P10 this read the README's phase status table, which was the only place in the
    document carrying a number. P10 removed that table and injected the coverage claim into a
    named marker pair instead, so the check now reads the injected sentence. What it asserts is
    unchanged and deliberately independent of `scripts/readme_inject.py`: the numbers come out
    of the calibrate stage's own artifact and are looked for verbatim in the page, so a
    generator that read the wrong key fails here even though it would pass its own byte
    comparison. `tests/test_readme_consistency.py` owns the staleness half of the gate.
    """

    def test_the_readme_quotes_the_measured_calibration_numbers(self, repo_root):
        import json
        import re

        from ufem.calibrate import CALIBRATION_JSON, SIGNAL_FORCE
        from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
        from ufem.config import config_hash, load_config
        from ufem.manifest import load_manifest, stage_dir

        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            CALIBRATE_STAGE,
            config_hash(config),
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip("the calibrate stage has not run for this config hash")
        extra = load_manifest(directory)["extra"]
        calibration = json.loads((directory / CALIBRATION_JSON).read_text(encoding="utf-8"))
        band = calibration["functional"][SIGNAL_FORCE]["bands"][f"{GATE_ALPHA:g}"]

        # Whitespace collapsed: the injector wraps its prose, so a claim can carry a line break
        # inside it, and the wrapping is presentation of the source rather than content.
        readme = re.sub(
            r"\s+", " ", (repo_root / "README.md").read_text(encoding="utf-8")
        )
        for value in (
            f"{extra['functional_coverage'][SIGNAL_FORCE]:.4f}",
            f"[{band['wilson_low']:.4f}, {band['wilson_high']:.4f}]",
            f"{1.0 - GATE_ALPHA:.2f}",
        ):
            assert value in readme, (
                f"the README does not quote {value!r}, which is what the calibrate manifest at "
                f"{directory} records. Ground rule 10: rerun scripts/readme_inject.py, do not "
                "edit this test."
            )
        assert extra["gate"]["passed"], (
            "the calibration gate no longer passes, and the README says it does. Both have to "
            "be revisited rather than one of them."
        )
        assert "gate of build spec 11.5 passed" in readme
