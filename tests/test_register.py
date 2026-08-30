"""Registration contracts, properties, and the P3 gate of build spec 22.

The oracle problem bites hardest here: nobody knows the true warp between two FE curves, so
the landmark tests are built on synthetic curves whose landmarks are analytic, and the tests
against the real family assert properties (monotonicity, endpoint conditions, recovery)
rather than values nobody can independently verify.

The banned construction of build spec 10.1, per curve normalizers, gets its own test. It is
the kind of defect that improves every reconstruction metric while destroying exactly the
amplitude information the surrogate exists to predict, so a test that only checked
reconstruction quality would rate it an improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ufem.config import load_config
from ufem.register import (
    KNEE_WINDOW_MIN_MM,
    N_ARCLENGTH_POINTS,
    cracking_knee,
    extract_landmarks,
    gamma_monotonicity_report,
    invert_warp,
    normalized_arclength,
    post_peak_level_crossing,
    recover_unregistered,
    resample_on_arclength,
    smoothed_second_difference,
)

SETTINGS = settings(deadline=None, derandomize=True, max_examples=40)


#: Half the smoothing kernel's width in mm, on the 0.1 mm synthetic grid. The moving average
#: of half width 3 spreads a sharp curvature feature symmetrically, so the estimator reports
#: the knee up to about this far below where it was planted. That is a known, bounded
#: property of the estimator rather than an error, and the tolerances below allow for it.
KNEE_SMOOTHING_OFFSET_MM = 0.3


def synthetic_curve(
    n: int = 201,
    u_max: float = 20.0,
    u_knee: float = 2.0,
    u_peak: float = 10.0,
    peak: float = 40000.0,
    stiffness_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """A piecewise curve with landmarks that are known by construction, in (mm, N).

    A stiff elastic branch to ``u_knee``, a softer branch that rises to ``peak`` at
    ``u_peak``, then a linear descent. The knee is a drop in stiffness by
    ``stiffness_ratio``, which is what a first cracking knee physically is and what makes its
    curvature sharply negative and well localized.

    Two earlier versions of this fixture were wrong in instructive ways, so the shape is
    documented rather than left to be rediscovered. The first gave the post knee arc a
    steeper initial slope than the elastic branch, which bends the curve upward at the knee:
    a positive curvature, correctly not identified as a knee. The second made the arc tangent
    continuous but parabolic, which spreads a weak constant curvature over the whole arc and
    leaves the estimator choosing among ties. A real cracking knee is neither: it is a sharp,
    localized loss of stiffness, and the real campaign's curves show exactly that, running
    linear to four significant digits and then breaking within two grid points.
    """
    if not 0.0 < stiffness_ratio < 1.0:
        raise ValueError(f"the stiffness ratio must lie in (0, 1), got {stiffness_ratio}.")
    u = np.linspace(0.0, u_max, n)
    # peak = k0 u_knee + k0 ratio (u_peak - u_knee) fixes k0 for a requested peak.
    k0 = peak / (u_knee + stiffness_ratio * (u_peak - u_knee))
    force = np.where(u <= u_knee, k0 * u, k0 * u_knee + k0 * stiffness_ratio * (u - u_knee))
    tail = u > u_peak
    force[tail] = peak - 0.6 * peak * (u[tail] - u_peak) / (u_max - u_peak)
    return u, force


class TestTheKneeEstimator:
    def test_it_finds_the_analytic_knee_of_a_synthetic_curve(self):
        """The one point where the synthetic curve's slope changes discontinuously."""
        u, force = synthetic_curve(u_knee=2.0, u_peak=10.0)
        u_knee, _p_knee = cracking_knee(u, force, 10.0)
        assert u_knee == pytest.approx(2.0, abs=KNEE_SMOOTHING_OFFSET_MM + 0.05)

    @pytest.mark.parametrize("expected", [1.5, 2.0, 3.0, 4.0])
    def test_it_tracks_the_knee_wherever_it_is_placed(self, expected):
        """The landmark must follow the knee, not sit at a fixed point of the window."""
        u, force = synthetic_curve(u_knee=expected, u_peak=12.0)
        u_knee, _ = cracking_knee(u, force, 12.0)
        assert u_knee == pytest.approx(expected, abs=KNEE_SMOOTHING_OFFSET_MM + 0.05)

    def test_the_smoothing_offset_is_small_and_consistent(self):
        """The bias is systematic rather than noisy, which is what makes it tolerable.

        Measured on this fixture the estimator reports the knee 0.1 to 0.2 mm *late*, never
        early. I had guessed the opposite sign when writing this test: a symmetric moving
        average does not imply a symmetric response, because the second difference of the
        smoothed signal peaks just past a break in slope. A landmark carrying a consistent
        sub grid bias is usable, since it shifts every curve the same way; one whose error
        wandered in sign between curves would not be, so the consistency is the property
        worth pinning, not the sign I assumed.
        """
        offsets = []
        for planted in (1.5, 2.0, 3.0, 4.0):
            u, force = synthetic_curve(u_knee=planted, u_peak=12.0)
            offsets.append(cracking_knee(u, force, 12.0)[0] - planted)
        assert all(0.0 <= value <= KNEE_SMOOTHING_OFFSET_MM for value in offsets)
        assert max(offsets) - min(offsets) <= 0.15

    def test_it_refuses_a_window_too_short_to_hold_a_second_difference(self):
        u, force = synthetic_curve()
        with pytest.raises(ValueError, match="too few for a second difference"):
            cracking_knee(u, force, KNEE_WINDOW_MIN_MM * 1.05)

    def test_it_refuses_a_peak_inside_the_excluded_head(self):
        u, force = synthetic_curve()
        with pytest.raises(ValueError, match="pre peak window"):
            cracking_knee(u, force, KNEE_WINDOW_MIN_MM * 0.5)

    def test_the_second_difference_keeps_the_length_of_its_input(self):
        """A shortened return would silently shift every index the caller derives from it."""
        values = np.linspace(0.0, 1.0, 51) ** 2
        assert smoothed_second_difference(values, 3).shape == values.shape


class TestThePostPeakLandmark:
    def test_it_finds_the_crossing_on_a_curve_that_softens(self):
        u, force = synthetic_curve(u_peak=10.0, peak=40000.0)
        peak_index = int(np.argmax(force))
        u_85, p_85, reached = post_peak_level_crossing(u, force, peak_index)
        assert reached
        assert p_85 == pytest.approx(0.85 * 40000.0, rel=1e-9)
        assert 10.0 < u_85 < 20.0

    def test_a_curve_that_never_softens_reports_a_missing_landmark(self):
        """Ground rule 8: the honest answer is NaN with a flag, never the stroke end."""
        u = np.linspace(0.0, 20.0, 201)
        force = 40000.0 * (1.0 - 0.01 * u / 20.0)
        u_85, p_85, reached = post_peak_level_crossing(u, force, 0)
        assert not reached
        assert np.isnan(u_85) and np.isnan(p_85)

    def test_it_scans_the_whole_branch_not_only_the_final_value(self):
        """A curve that dips below the level and recovers still has a real crossing.

        33 curves in the real campaign do exactly this, so a rule that tested the end point
        alone would discard 33 genuine landmarks.
        """
        u = np.linspace(0.0, 20.0, 201)
        force = np.full_like(u, 40000.0)
        force[100:120] = 30000.0  # dips to 0.75 of peak, then recovers
        u_85, _p, reached = post_peak_level_crossing(u, force, 0)
        assert reached
        assert 9.0 < u_85 < 12.0


class TestArcLength:
    def test_the_normalizers_must_be_scalars(self):
        """Build spec 10.1 bans per curve normalizers; an array is the shape they arrive in."""
        u, force = synthetic_curve()
        with pytest.raises(TypeError, match="must be scalars from config"):
            normalized_arclength(u, force, np.full(u.size, 20.0), 40000.0)

    def test_the_config_normalizers_are_scalars_and_global(self, repo_root):
        """The banned construction, checked at its source rather than at its use."""
        normalizers = load_config(repo_root).pipeline.normalizers
        assert isinstance(normalizers.P0_N, float)
        assert isinstance(normalizers.u0_mm, float)
        assert normalizers.P0_N == 40000.0
        assert normalizers.u0_mm == 20.0

    def test_arc_length_starts_at_zero_and_increases(self):
        u, force = synthetic_curve()
        s = normalized_arclength(u, force, 20.0, 40000.0)
        assert s[0] == 0.0
        assert np.all(np.diff(s) > 0.0)

    @SETTINGS
    @given(
        u_knee=st.floats(min_value=1.0, max_value=4.0),
        u_peak=st.floats(min_value=8.0, max_value=14.0),
    )
    def test_resampling_an_arc_length_parameterization_is_idempotent(self, u_knee, u_peak):
        """Property: resampling an already arc length parameterized curve changes nothing.

        The second pass has nothing left to do, so any drift is the resampler introducing
        error rather than removing distortion.
        """
        u, force = synthetic_curve(u_knee=u_knee, u_peak=u_peak)
        s = normalized_arclength(u, force, 20.0, 40000.0)
        once = resample_on_arclength(force, s)
        stations = np.linspace(0.0, 1.0, N_ARCLENGTH_POINTS)
        twice = resample_on_arclength(once, stations * float(s[-1]))
        np.testing.assert_allclose(twice, once, atol=1e-9, rtol=0.0)

    def test_a_stalled_curve_is_a_stop_condition(self):
        s = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="not strictly increasing"):
            resample_on_arclength(np.arange(4.0), s)


class TestWarpAlgebra:
    @SETTINGS
    @given(power=st.floats(min_value=0.4, max_value=2.5))
    def test_inverting_a_warp_twice_returns_it(self, power):
        """Property, on synthetic warps: inversion is an involution up to interpolation."""
        s = np.linspace(0.0, 1.0, 201)
        gamma = s**power
        back = invert_warp(invert_warp(gamma, s), s)
        np.testing.assert_allclose(back, gamma, atol=2e-3, rtol=0.0)

    @SETTINGS
    @given(power=st.floats(min_value=0.5, max_value=2.0))
    def test_recovery_undoes_a_synthetic_warp(self, power):
        """The composition direction, pinned on a case where the answer is known.

        Getting this backwards is silent: it returns a plausible curve that is not the input.
        """
        s = np.linspace(0.0, 1.0, 401)
        gamma = s**power
        _u, force = synthetic_curve(n=401)
        warped = np.interp(gamma, s, force)
        recovered = recover_unregistered(warped, gamma, s)
        assert np.abs(recovered - force).max() / force.max() < 0.02

    def test_monotonicity_report_counts_a_planted_violation(self):
        gamma = np.tile(np.linspace(0.0, 1.0, 51), (3, 1))
        gamma[1, 20] = gamma[1, 19] - 0.01
        report = gamma_monotonicity_report(gamma)
        assert report["n_decreasing_increments"] == 1
        assert report["min_increment"] < 0.0


class TestLandmarkExtraction:
    def test_it_returns_the_full_schedule_for_a_synthetic_curve(self):
        u, force = synthetic_curve(u_knee=2.0, u_peak=10.0, peak=40000.0)
        landmarks = extract_landmarks(u, force)
        assert landmarks["u_knee_mm"] == pytest.approx(2.0, abs=0.3)
        assert landmarks["u_peak_mm"] == pytest.approx(10.0, abs=0.1)
        assert landmarks["P_peak_N"] == pytest.approx(40000.0, rel=1e-6)
        assert landmarks["u_end_mm"] == pytest.approx(20.0)
        assert landmarks["u85_reached"] is True

    def test_a_flat_curve_is_rejected(self):
        u = np.linspace(0.0, 20.0, 201)
        with pytest.raises(ValueError, match="not a loaded curve"):
            extract_landmarks(u, np.zeros_like(u))


@pytest.mark.fullstack
class TestAgainstTheArtifactStore:
    """The P3 gate of build spec 22, measured on the real registered family."""

    @staticmethod
    @pytest.fixture(scope="class")
    def artifacts(repo_root):
        from ufem.config import config_hash
        from ufem.manifest import stage_dir
        from ufem.register import (
            AMPLITUDE_PARQUET,
            AMPLITUDE_UNREGISTERED_PARQUET,
            LANDMARKS_PARQUET,
            STAGE_NAME,
            WARP_PARQUET,
            curve_matrix,
        )

        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip(f"the register stage has not run for this config hash: {directory}")
        _jobs, registered = curve_matrix(pd.read_parquet(directory / AMPLITUDE_PARQUET))
        _j2, raw = curve_matrix(pd.read_parquet(directory / AMPLITUDE_UNREGISTERED_PARQUET))
        _j3, gamma = curve_matrix(pd.read_parquet(directory / WARP_PARQUET))
        return {
            "registered": registered,
            "raw": raw,
            "gamma": gamma,
            "landmarks": pd.read_parquet(directory / LANDMARKS_PARQUET),
        }

    def test_every_warp_is_monotone_with_fixed_endpoints(self, artifacts):
        """The build spec 22 gate, at the 1e-8 tolerance the phase brief names."""
        report = gamma_monotonicity_report(artifacts["gamma"], tolerance=1e-8)
        assert report["n_decreasing_increments"] == 0
        assert report["max_abs_start_error"] <= 1e-8
        assert report["max_abs_end_error"] <= 1e-8

    def test_the_registered_family_recovers_the_input_through_the_warps(self, artifacts):
        """Recovery at the measured tolerance, not an aspirational one.

        The residual is discretization: composing functions sampled at 201 stations resamples
        through regions the warp compresses. The measured median is 1.6 percent and the worst
        curve, the one with the steepest warp, is 14.7 percent. The bounds here are set above
        the measurement so a real regression is visible without the test firing on the floor.
        """
        registered, gamma, raw = (
            artifacts["registered"], artifacts["gamma"], artifacts["raw"]
        )
        errors = np.array(
            [
                np.abs(recover_unregistered(registered[row], gamma[row]) - raw[row]).max()
                / raw[row].max()
                for row in range(registered.shape[0])
            ]
        )
        assert np.median(errors) < 0.03
        assert np.percentile(errors, 90) < 0.06
        assert errors.max() < 0.20

    def test_registration_preserves_amplitude_while_removing_spread(self, artifacts):
        """Registration must take out phase, not amplitude.

        A step that shrank the peak would be destroying the quantity the surrogate exists to
        predict, which is the failure mode per curve normalizers cause.
        """
        registered, raw = artifacts["registered"], artifacts["raw"]
        assert registered.std(axis=0).mean() < raw.std(axis=0).mean()
        assert registered.max(axis=1).mean() == pytest.approx(
            raw.max(axis=1).mean(), rel=0.02
        )

    def test_no_landmark_pins_against_the_knee_window_edge(self, artifacts):
        """The defect the first knee window had: 135 of 198 curves on the boundary."""
        knees = artifacts["landmarks"]["u_knee_mm"].to_numpy(dtype=float)
        assert int((knees <= KNEE_WINDOW_MIN_MM + 0.1).sum()) == 0

    def test_the_missing_landmark_is_recorded_rather_than_invented(self, artifacts):
        """A curve that never softens carries NaN and a False flag, not the stroke end."""
        landmarks = artifacts["landmarks"]
        missing = ~landmarks["u85_reached"].to_numpy(dtype=bool)
        assert int(missing.sum()) >= 1
        assert bool(np.isnan(landmarks.loc[missing, "u_85_mm"]).all())
        assert not np.isnan(landmarks.loc[~missing, "u_85_mm"]).any()
