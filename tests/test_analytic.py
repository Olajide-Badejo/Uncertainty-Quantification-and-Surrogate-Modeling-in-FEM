"""The analytic mechanics model of build spec 13.4, tested against its own equations.

This module has something the rest of the project mostly does not: an oracle. Every quantity
here follows from equilibrium and plane sections, so a test can solve the same equation a
different way and demand the same answer, rather than pinning whatever the implementation
happened to produce. Three kinds of check appear below:

* **equilibrium**, solved again numerically and compared against the closed form selection;
* **invariance**, where a quantity that must not depend on a choice is computed under two
  choices (moments about two different points, a vectorized call against a loop);
* **regime**, where a section is constructed to sit in each of the three branches of the
  neutral axis solution, and the branch that was taken is the one the mechanics requires.

No torch, no artifact store: this runs on the light stack CI job.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from ufem.analytic import (
    DEFAULT_BEAM,
    MODEL_ERROR_FRACTION,
    Beam,
    SectionAssumptionViolated,
    compression_steel_stress_MPa,
    cracked_second_moment_mm4,
    cracking_load_N,
    cross_check,
    empirical_log_elasticities,
    hogging_capacity,
    log_elasticities,
    neutral_axis_depth_mm,
    peak_load_N,
    tensile_strength_MPa,
    tip_stiffness_N_per_mm,
)

#: The mean point of the probabilistic model, used as the reference section. The values are
#: arguments to a mechanics function here, not a declaration of a distribution.
MEAN_STRENGTH = 28.0
MEAN_BOTTOM = 27.0
MEAN_TOP = 223.0


def _equilibrium_residual(x: float, fcm: float, c_bottom: float, beam: Beam) -> float:
    """Horizontal equilibrium of the hogging section, written out independently."""
    concrete = beam.block_depth_factor * x * beam.thickness_mm * beam.block_stress_factor * fcm
    strain = beam.concrete_ultimate_strain * (x - c_bottom) / x
    stress = float(
        np.clip(beam.steel_modulus_MPa * strain, -beam.steel_yield_MPa, beam.steel_yield_MPa)
    )
    return concrete + beam.area_bottom_mm2 * stress - beam.area_top_mm2 * beam.steel_yield_MPa


class TestTheNeutralAxis:
    def test_the_closed_form_root_satisfies_equilibrium_solved_numerically(self):
        """The oracle: a bracketed numerical root of the same equation, to 1e-10 relative."""
        for fcm in (20.0, 28.0, 38.0):
            for c_bottom in (16.0, 27.0, 39.0):
                closed = float(neutral_axis_depth_mm(fcm, c_bottom))
                bracketed = brentq(
                    _equilibrium_residual,
                    1.0e-6,
                    0.99 * MEAN_TOP,
                    args=(fcm, c_bottom, DEFAULT_BEAM),
                    xtol=1.0e-13,
                    rtol=1.0e-15,
                )
                assert closed == pytest.approx(bracketed, rel=1.0e-10), (fcm, c_bottom)

    def test_the_three_regimes_are_all_reached_and_all_correct(self):
        """Bottom layer elastic, yielded in compression, yielded in tension.

        The regime is set by where the bottom layer sits relative to the neutral axis, so a
        very shallow cover puts it deep in the compression zone and a very deep one puts it
        below the axis entirely. Each case is checked against the numerical root and against
        the stress the mechanics requires, so a selection that happened to land on the right
        number for the wrong reason would still fail.
        """
        yield_strain = DEFAULT_BEAM.steel_yield_strain
        cases = {"compression": 2.0, "elastic": 27.0, "tension": 160.0}
        seen = set()
        for label, c_bottom in cases.items():
            x = float(neutral_axis_depth_mm(MEAN_STRENGTH, c_bottom))
            stress = float(compression_steel_stress_MPa(x, c_bottom))
            strain = DEFAULT_BEAM.concrete_ultimate_strain * (x - c_bottom) / x
            if strain > yield_strain:
                seen.add("compression")
            elif strain < -yield_strain:
                seen.add("tension")
            else:
                seen.add("elastic")
            bracketed = brentq(
                _equilibrium_residual,
                1.0e-6,
                0.99 * MEAN_TOP,
                args=(MEAN_STRENGTH, c_bottom, DEFAULT_BEAM),
                xtol=1.0e-13,
                rtol=1.0e-15,
            )
            assert x == pytest.approx(bracketed, rel=1.0e-10), label
            assert abs(stress) <= DEFAULT_BEAM.steel_yield_MPa + 1.0e-9
        assert seen == {"compression", "elastic", "tension"}

    def test_the_neutral_axis_deepens_as_the_concrete_weakens(self):
        weak = float(neutral_axis_depth_mm(20.0, MEAN_BOTTOM))
        strong = float(neutral_axis_depth_mm(38.0, MEAN_BOTTOM))
        assert weak > strong

    def test_a_vectorized_call_equals_the_loop_over_its_rows(self):
        strengths = np.array([21.0, 28.0, 35.0])
        covers = np.array([18.0, 27.0, 36.0])
        together = neutral_axis_depth_mm(strengths, covers)
        apart = np.array(
            [float(neutral_axis_depth_mm(f, c)) for f, c in zip(strengths, covers)]
        )
        np.testing.assert_allclose(together, apart, rtol=0.0, atol=1.0e-12)


class TestTheCapacity:
    def test_the_moment_does_not_depend_on_the_point_it_is_taken_about(self):
        """With the section in equilibrium, the resultant moment is point independent.

        The implementation takes moments about the tension steel. Here they are taken about the
        soffit instead, using the same three forces, and the two must agree. This is the check
        that the lever arms are right, which no comparison against a remembered number is.
        """
        state = hogging_capacity(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP)
        x = float(state.neutral_axis_mm)
        stress = float(state.bottom_steel_stress_MPa)
        beam = DEFAULT_BEAM
        concrete_force = (
            beam.block_depth_factor * x * beam.thickness_mm * beam.block_stress_factor
            * MEAN_STRENGTH
        )
        steel_force = beam.area_bottom_mm2 * stress
        tension_force = beam.area_top_mm2 * beam.steel_yield_MPa
        assert concrete_force + steel_force == pytest.approx(tension_force, rel=1.0e-12)
        about_soffit = (
            tension_force * MEAN_TOP
            - concrete_force * beam.block_depth_factor * x / 2.0
            - steel_force * MEAN_BOTTOM
        )
        assert float(state.moment_Nmm) == pytest.approx(about_soffit, rel=1.0e-10)

    def test_the_peak_load_is_the_capacity_over_the_lever_arm(self):
        state = hogging_capacity(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP)
        expected = float(state.moment_Nmm) / DEFAULT_BEAM.overhang_mm
        assert float(peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP)) == pytest.approx(expected)

    def test_the_capacity_is_monotone_in_every_input_in_the_direction_mechanics_requires(self):
        base = float(peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP))
        assert float(peak_load_N(MEAN_STRENGTH + 4.0, MEAN_BOTTOM, MEAN_TOP)) > base
        assert float(peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP + 5.0)) > base
        assert float(peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM + 5.0, MEAN_TOP)) < base

    def test_an_over_reinforced_section_raises_instead_of_returning_a_capacity(self):
        """The block formula assumes the tension steel yields, so it must refuse when it does
        not."""
        with pytest.raises(SectionAssumptionViolated):
            peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, 45.0)

    def test_a_section_with_no_compression_zone_left_raises(self):
        thin = replace(DEFAULT_BEAM, thickness_mm=6.0)
        with pytest.raises(SectionAssumptionViolated):
            peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP, thin)

    def test_the_elasticities_say_the_capacity_is_governed_by_the_effective_depth(self):
        """The model's own structure, pinned as a property rather than as three digits.

        A section whose tension steel yields carries its force whatever the concrete strength
        is, so the strength enters only through the depth of the stress block and its
        elasticity is small. The top cover is the effective depth, so its elasticity is near
        one. This is the shape of the analytic model that the cross check of build spec 13.4
        compares against the campaign, and it is a consequence of the mechanics rather than a
        fitted outcome.
        """
        elasticities = log_elasticities(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP)
        assert elasticities["Fcm_MPa"] < 0.1
        assert 0.9 < elasticities["c_nom_top_mm"] < 1.3
        assert abs(elasticities["c_nom_bottom_mm"]) < 0.1


class TestTheElasticQuantities:
    def test_the_tip_stiffness_reduces_to_the_textbook_bending_formula(self):
        """Removing the shear flexibility must leave ``3 E I / (a**2 (L + a))`` exactly."""
        beam = replace(DEFAULT_BEAM, shear_correction=1.0e12)
        modulus, inertia = 30000.0, DEFAULT_BEAM.gross_second_moment_mm4
        expected = (
            3.0 * modulus * inertia / (beam.overhang_mm**2 * (beam.span_mm + beam.overhang_mm))
        )
        assert float(tip_stiffness_N_per_mm(modulus, inertia, beam)) == pytest.approx(
            expected, rel=1.0e-9
        )

    def test_shear_flexibility_can_only_soften_the_member(self):
        modulus, inertia = 30000.0, DEFAULT_BEAM.gross_second_moment_mm4
        with_shear = float(tip_stiffness_N_per_mm(modulus, inertia))
        without = float(
            tip_stiffness_N_per_mm(modulus, inertia, replace(DEFAULT_BEAM, shear_correction=1.0e12))
        )
        assert with_shear < without

    def test_the_gross_second_moment_is_the_rectangle_formula(self):
        assert DEFAULT_BEAM.gross_second_moment_mm4 == pytest.approx(
            150.0 * 250.0**3 / 12.0, rel=1.0e-12
        )

    def test_the_cracked_section_is_softer_than_the_gross_one_and_balances_its_own_areas(self):
        modulus = 30000.0
        cracked = float(cracked_second_moment_mm4(modulus, MEAN_BOTTOM, MEAN_TOP))
        assert 0.0 < cracked < DEFAULT_BEAM.gross_second_moment_mm4
        # Recover the compression depth from the returned second moment by solving the same
        # transformed area balance independently, then confirm the two agree.
        ratio = DEFAULT_BEAM.steel_modulus_MPa / modulus

        def balance(depth: float) -> float:
            return (
                DEFAULT_BEAM.thickness_mm * depth**2 / 2.0
                + (ratio - 1.0) * DEFAULT_BEAM.area_bottom_mm2 * (depth - MEAN_BOTTOM)
                - ratio * DEFAULT_BEAM.area_top_mm2 * (MEAN_TOP - depth)
            )

        depth = brentq(balance, 1.0e-6, MEAN_TOP - 1.0e-9, xtol=1.0e-12, rtol=1.0e-15)
        expected = (
            DEFAULT_BEAM.thickness_mm * depth**3 / 3.0
            + (ratio - 1.0) * DEFAULT_BEAM.area_bottom_mm2 * (depth - MEAN_BOTTOM) ** 2
            + ratio * DEFAULT_BEAM.area_top_mm2 * (MEAN_TOP - depth) ** 2
        )
        assert cracked == pytest.approx(expected, rel=1.0e-9)

    def test_the_tensile_strength_is_the_eurocode_expression(self):
        assert float(tensile_strength_MPa(28.0)) == pytest.approx(
            0.3 * 20.0 ** (2.0 / 3.0), rel=1.0e-12
        )

    def test_the_tensile_strength_refuses_a_strength_the_expression_cannot_take(self):
        with pytest.raises(SectionAssumptionViolated):
            tensile_strength_MPa(7.0)

    def test_cracking_happens_far_below_the_capacity(self):
        """A member that cracked near its capacity would not be a softening beam at all."""
        assert float(cracking_load_N(MEAN_STRENGTH)) < 0.2 * float(
            peak_load_N(MEAN_STRENGTH, MEAN_BOTTOM, MEAN_TOP)
        )


class TestTheCrossCheck:
    def test_a_distribution_brackets_itself_on_both_verdicts(self):
        rng = np.random.default_rng(3)
        sample = 38000.0 + 3000.0 * rng.standard_normal(20000)
        verdict = cross_check(sample, sample.copy())
        assert verdict["central_tendency_brackets"]
        assert verdict["dispersion_brackets"]
        assert verdict["median_ratio"] == pytest.approx(1.0, abs=1.0e-12)

    def test_a_shifted_distribution_fails_the_central_tendency_verdict(self):
        rng = np.random.default_rng(4)
        sample = 38000.0 + 3000.0 * rng.standard_normal(20000)
        verdict = cross_check(sample, 1.5 * sample)
        assert not verdict["central_tendency_brackets"]
        assert verdict["median_ratio"] == pytest.approx(1.5, rel=1.0e-6)

    def test_a_too_narrow_distribution_fails_the_dispersion_verdict_alone(self):
        """The failure mode this campaign actually produced, constructed on purpose."""
        rng = np.random.default_rng(5)
        wide = 38000.0 + 6000.0 * rng.standard_normal(40000)
        narrow = 38000.0 + 300.0 * rng.standard_normal(40000)
        verdict = cross_check(wide, narrow)
        assert verdict["central_tendency_brackets"]
        assert not verdict["dispersion_brackets"]
        assert verdict["dispersion_ratio"] < 0.2
        assert verdict["quantile_ratio"]["p05"] > 1.0 + MODEL_ERROR_FRACTION

    def test_the_stated_model_error_is_a_constant_and_not_an_argument_default_nobody_chose(self):
        assert MODEL_ERROR_FRACTION == 0.15
        rng = np.random.default_rng(6)
        sample = 38000.0 + 3000.0 * rng.standard_normal(5000)
        assert cross_check(sample, sample)["model_error"] == MODEL_ERROR_FRACTION


class TestTheEmpiricalElasticities:
    def test_a_known_power_law_is_recovered_exactly(self):
        rng = np.random.default_rng(7)
        design = np.column_stack(
            [
                rng.uniform(20.0, 38.0, 400),
                rng.uniform(18.0, 36.0, 400),
                rng.uniform(210.0, 236.0, 400),
            ]
        )
        response = 3.0 * design[:, 0] ** 0.7 * design[:, 1] ** -0.2 * design[:, 2] ** 1.1
        measured = empirical_log_elasticities(
            design, response, ("Fcm_MPa", "c_nom_bottom_mm", "c_nom_top_mm")
        )
        assert measured["elasticities"]["Fcm_MPa"] == pytest.approx(0.7, abs=1.0e-9)
        assert measured["elasticities"]["c_nom_bottom_mm"] == pytest.approx(-0.2, abs=1.0e-9)
        assert measured["elasticities"]["c_nom_top_mm"] == pytest.approx(1.1, abs=1.0e-9)
        assert measured["r2"] == pytest.approx(1.0, abs=1.0e-12)

    def test_a_mismatched_design_and_response_raise_rather_than_broadcast(self):
        with pytest.raises(ValueError):
            empirical_log_elasticities(np.zeros((4, 3)), np.ones(5), ("a", "b", "c"))


def test_the_analytic_model_is_deterministic():
    """No generator anywhere in it: the same inputs give bit identical output twice."""
    first = peak_load_N(
        np.array([21.0, 28.0, 35.0]), np.array([20.0, 27.0, 34.0]), np.array([214.0, 223.0, 232.0])
    )
    second = peak_load_N(
        np.array([21.0, 28.0, 35.0]), np.array([20.0, 27.0, 34.0]), np.array([214.0, 223.0, 232.0])
    )
    assert np.array_equal(first, second)
    assert math.isfinite(float(first[0]))
