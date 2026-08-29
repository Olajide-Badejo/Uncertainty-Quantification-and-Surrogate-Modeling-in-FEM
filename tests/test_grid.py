"""Unit tests for the grid stage: interpolation, the QoI extractors, and the banned column.

Every numeric assertion here has a closed form answer. A synthetic linear curve has an
exactly known stiffness; a triangle has an exactly known peak, energy and residual. That is
the point: the extractors are checked against arithmetic, not against their own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ufem.config import FEATURE_ORDER, config_hash, load_config
from ufem.grid import (
    BANNED_QOI_COLUMNS,
    DAMAGE_SATURATION,
    QOI_COLUMNS,
    QOI_PARQUET,
    RF2_GRID_PARQUET,
    STAGE_NAME,
    damage_half_saturation,
    displacement_grid,
    extract_qoi,
    headline_stats,
    initial_stiffness,
    interpolate_onto_grid,
    monotone_curve,
)
from ufem.manifest import stage_dir

ATOL = 1e-9

#: The common grid every test measures against, in mm.
GRID_MM = np.linspace(0.0, 20.0, 201)


def triangle_force(u_mm: np.ndarray, p_max: float, u_peak: float, p_end: float) -> np.ndarray:
    """A triangular load displacement curve in N: linear up to the peak, linear down to 20 mm.

    Every QoI of build spec 9.5 has a closed form on this shape, which is what makes it a
    usable oracle. Absorbed energy is the area of the two triangles under the polyline.
    """
    u = np.asarray(u_mm, dtype=float)
    rising = p_max * u / u_peak
    falling = p_max + (p_end - p_max) * (u - u_peak) / (20.0 - u_peak)
    return np.where(u <= u_peak, rising, falling)


class TestMonotoneCurve:
    def test_an_already_increasing_curve_is_unchanged(self):
        x, y = monotone_curve(np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]))
        assert x.tolist() == [0.0, 1.0, 2.0]
        assert y.tolist() == [10.0, 20.0, 30.0]

    def test_an_unsorted_curve_is_sorted_by_abscissa(self):
        x, y = monotone_curve(np.array([2.0, 0.0, 1.0]), np.array([30.0, 10.0, 20.0]))
        assert x.tolist() == [0.0, 1.0, 2.0]
        assert y.tolist() == [10.0, 20.0, 30.0]

    def test_a_repeated_abscissa_keeps_the_first_in_solver_order(self):
        x, y = monotone_curve(np.array([0.0, 1.0, 1.0, 2.0]), np.array([0.0, 11.0, 99.0, 22.0]))
        assert x.tolist() == [0.0, 1.0, 2.0]
        assert y.tolist() == [0.0, 11.0, 22.0]

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="matching shapes"):
            monotone_curve(np.array([0.0, 1.0]), np.array([0.0]))


class TestInitialStiffness:
    def test_a_linear_curve_recovers_its_slope_exactly(self):
        """P = k u with k = 13500 N/mm: the through origin fit is exact by construction."""
        k_true = 13500.0
        u = np.linspace(0.0, 20.0, 2001)
        assert_allclose(initial_stiffness(u, k_true * u, u_peak_mm=12.0), k_true, atol=ATOL)

    @pytest.mark.parametrize("k_true", [1.0, 9501.528333291784, 17330.53630271611])
    def test_the_slope_is_recovered_across_the_campaign_range(self, k_true):
        u = np.linspace(0.0, 20.0, 4001)
        assert_allclose(initial_stiffness(u, k_true * u, u_peak_mm=11.0), k_true, atol=ATOL)

    def test_only_the_window_below_a_tenth_of_the_peak_is_used(self):
        """Corrupting the curve above the window must not move the answer."""
        k_true = 12000.0
        u = np.linspace(0.0, 20.0, 2001)
        force = k_true * u
        force[u > 1.5] = -1e6
        assert_allclose(initial_stiffness(u, force, u_peak_mm=12.0), k_true, atol=ATOL)

    def test_the_origin_is_excluded_so_a_zero_row_cannot_bias_the_fit(self):
        k_true = 5000.0
        u = np.array([0.0, 0.2, 0.4, 0.6])
        assert_allclose(initial_stiffness(u, k_true * u, u_peak_mm=10.0), k_true, atol=ATOL)

    def test_a_window_with_one_point_raises_rather_than_falling_back(self):
        """Build spec 5.8: the v1 code silently fell back to the first three points."""
        u = np.array([0.0, 0.5, 5.0, 10.0])
        with pytest.raises(ValueError, match="fewer than the two"):
            initial_stiffness(u, 1000.0 * u, u_peak_mm=6.0)

    def test_a_nonpositive_peak_displacement_raises(self):
        u = np.linspace(0.0, 20.0, 201)
        with pytest.raises(ValueError, match="positive displacement at peak"):
            initial_stiffness(u, 1000.0 * u, u_peak_mm=0.0)


class TestDamageHalfSaturation:
    def test_a_ramp_reaching_the_threshold_at_a_node_returns_that_node(self):
        u = np.array([0.0, 5.0, 10.0, 15.0])
        damage = np.array([0.0, 0.1, 0.5 * DAMAGE_SATURATION, DAMAGE_SATURATION])
        assert_allclose(damage_half_saturation(u, damage), 10.0, atol=ATOL)

    def test_the_crossing_is_interpolated_between_the_bracketing_samples(self):
        """Halfway between two samples that straddle the threshold symmetrically."""
        half = 0.5 * DAMAGE_SATURATION
        u = np.array([0.0, 8.0, 9.0, 20.0])
        damage = np.array([0.0, half - 0.05, half + 0.05, DAMAGE_SATURATION])
        assert_allclose(damage_half_saturation(u, damage), 8.5, atol=ATOL)

    def test_a_linear_damage_ramp_gives_the_analytic_crossing(self):
        u = np.linspace(0.0, 20.0, 2001)
        damage = DAMAGE_SATURATION * u / 20.0
        assert_allclose(damage_half_saturation(u, damage), 10.0, atol=1e-6)

    def test_a_curve_that_never_saturates_raises(self):
        u = np.linspace(0.0, 20.0, 21)
        with pytest.raises(ValueError, match="never reaches half saturation"):
            damage_half_saturation(u, np.full_like(u, 0.1))


class TestExtractQoiOnATriangle:
    """Analytic oracle: peak 40000 N at 10 mm, falling to 8000 N at 20 mm."""

    P_MAX = 40000.0
    U_PEAK = 10.0
    P_END = 8000.0

    @pytest.fixture
    def qoi(self):
        force = triangle_force(GRID_MM, self.P_MAX, self.U_PEAK, self.P_END)
        damage = DAMAGE_SATURATION * GRID_MM / 20.0
        return extract_qoi(GRID_MM, force, damage)

    def test_peak_load(self, qoi):
        assert_allclose(qoi["P_max_N"], self.P_MAX, atol=ATOL)

    def test_displacement_at_peak(self, qoi):
        assert_allclose(qoi["u_peak_mm"], self.U_PEAK, atol=ATOL)

    def test_initial_stiffness_is_the_rising_slope(self, qoi):
        assert_allclose(qoi["k0_N_per_mm"], self.P_MAX / self.U_PEAK, atol=ATOL)

    def test_absorbed_energy_is_the_area_of_the_two_triangles(self, qoi):
        """0.5 * 10 * 40000 rising, plus 0.5 * 10 * (40000 + 8000) falling, in N mm."""
        expected = 0.5 * self.U_PEAK * self.P_MAX + 0.5 * (20.0 - self.U_PEAK) * (
            self.P_MAX + self.P_END
        )
        assert_allclose(qoi["E_abs_Nmm"], expected, atol=ATOL)

    def test_residual_load_is_the_value_at_20_mm(self, qoi):
        assert_allclose(qoi["P_residual_N"], self.P_END, atol=ATOL)

    def test_softening_ratio_is_residual_over_peak(self, qoi):
        assert_allclose(qoi["softening_ratio"], self.P_END / self.P_MAX, atol=ATOL)

    def test_damage_at_10mm_is_the_midpoint_of_a_linear_ramp(self, qoi):
        assert_allclose(qoi["damage_at_10mm"], 0.5 * DAMAGE_SATURATION, atol=ATOL)

    def test_damage_half_saturation_is_the_midpoint_displacement(self, qoi):
        assert_allclose(qoi["u_damage_half_sat_mm"], 10.0, atol=1e-9)

    def test_the_extractor_returns_exactly_the_scheduled_scalars(self, qoi):
        scheduled = set(QOI_COLUMNS) - {"job", "sample_id", *FEATURE_ORDER}
        assert set(qoi) == scheduled

    def test_a_flat_zero_curve_raises_rather_than_reporting_a_zero_peak(self):
        damage = DAMAGE_SATURATION * GRID_MM / 20.0
        with pytest.raises(ValueError, match="not a loaded curve"):
            extract_qoi(GRID_MM, np.zeros_like(GRID_MM), damage)


class TestInterpolateOntoGrid:
    def _frame(self, jobs, u_values, values, column="RF2"):
        rows = []
        for job in jobs:
            rows.append(
                pd.DataFrame(
                    {
                        "job": pd.Series([job] * len(u_values), dtype="string"),
                        "U2": np.asarray(u_values, dtype=float),
                        column: np.asarray(values, dtype=float),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_a_linear_curve_interpolates_to_itself(self):
        frame = self._frame(["sample_000"], [0.0, 20.0], [0.0, 2000.0])
        jobs, matrix = interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")
        assert jobs == ["sample_000"]
        assert_allclose(matrix[0], 100.0 * GRID_MM, atol=ATOL)

    def test_jobs_come_back_in_sorted_order(self):
        frame = self._frame(["sample_007", "sample_001"], [0.0, 20.0], [0.0, 100.0])
        jobs, _ = interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")
        assert jobs == ["sample_001", "sample_007"]

    def test_the_negative_solver_sign_is_handled_by_the_magnitude(self):
        frame = self._frame(["sample_000"], [0.0, -20.0], [0.0, 2000.0])
        _, matrix = interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")
        assert_allclose(matrix[0], 100.0 * GRID_MM, atol=ATOL)

    def test_a_curve_that_stops_short_raises_rather_than_being_flat_filled(self):
        """np.interp would silently hold the last value; that would be an invented tail."""
        frame = self._frame(["sample_000"], [0.0, 15.0], [0.0, 1500.0])
        with pytest.raises(ValueError, match="does not cover the common grid"):
            interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")

    def test_a_curve_starting_above_the_grid_origin_raises(self):
        frame = self._frame(["sample_000"], [2.0, 20.0], [200.0, 2000.0])
        with pytest.raises(ValueError, match="does not cover the common grid"):
            interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")

    def test_a_single_point_curve_raises(self):
        frame = self._frame(["sample_000"], [0.0], [0.0])
        with pytest.raises(ValueError, match="too few"):
            interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")

    def test_zero_nan_on_a_well_formed_batch(self):
        frame = self._frame(["sample_000", "sample_001"], [0.0, 10.0, 20.0], [0.0, 40.0, 8.0])
        _, matrix = interpolate_onto_grid(frame, "RF2", GRID_MM, "synthetic")
        assert int(np.isnan(matrix).sum()) == 0


class TestHeadlineStats:
    def test_the_moments_of_a_known_sample(self):
        stats = headline_stats(np.array([1.0, 2.0, 3.0, 4.0]))
        assert_allclose(stats["mean"], 2.5, atol=ATOL)
        assert_allclose(stats["std"], np.sqrt(5.0 / 3.0), atol=ATOL)
        assert_allclose(stats["cov"], np.sqrt(5.0 / 3.0) / 2.5, atol=ATOL)
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0

    def test_a_single_value_raises_because_a_sample_std_needs_two(self):
        with pytest.raises(ValueError, match="at least two values"):
            headline_stats(np.array([1.0]))


class TestDisplacementGridFromConfig:
    def test_the_grid_is_the_configured_linspace(self, repo_root):
        config = load_config(repo_root)
        grid = displacement_grid(config)
        assert grid.size == config.pipeline.grid.n_points == 201
        assert_allclose(grid[0], 0.0, atol=ATOL)
        assert_allclose(grid[-1], 20.0, atol=ATOL)
        assert np.all(np.diff(grid) > 0.0)


@pytest.fixture(scope="module")
def qoi(repo_root):
    """The real QoI table, or a skip naming the command that produces it."""
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
    )
    path = directory / QOI_PARQUET
    if not path.is_file():
        pytest.skip(f"{path} does not exist; run `ufem run ingest grid` first.")
    return pd.read_parquet(path)


class TestQoiTableContract:
    """Build spec 5.6 and 9.5, enforced on the real artifact."""

    def test_terminal_damage_is_not_a_column(self, qoi):
        """The banned QoI of build spec 5.6: zero variance, saturated at the table cap."""
        offenders = sorted(set(qoi.columns) & set(BANNED_QOI_COLUMNS))
        assert offenders == [], f"banned QoI columns present: {offenders}"

    def test_no_column_name_mentions_a_terminal_or_final_damage(self, qoi):
        lowered = [name.lower() for name in qoi.columns]
        assert not [
            name
            for name in lowered
            if "damage" in name and ("final" in name or "terminal" in name or "max" in name)
        ]

    def test_the_column_set_is_exactly_the_scheduled_one(self, qoi):
        assert tuple(qoi.columns) == QOI_COLUMNS

    def test_the_table_holds_one_row_per_curve_with_zero_nan(self, qoi):
        assert len(qoi) == 198
        assert int(qoi.drop(columns=["job"]).isna().to_numpy().sum()) == 0

    def test_every_scalar_is_physically_signed(self, qoi):
        assert (qoi["P_max_N"] > 0).all()
        assert (qoi["k0_N_per_mm"] > 0).all()
        assert (qoi["E_abs_Nmm"] > 0).all()
        assert (qoi["u_peak_mm"] > 0).all()
        assert (qoi["u_peak_mm"] < 20.0).all()
        assert (qoi["softening_ratio"] > 0).all()
        assert (qoi["softening_ratio"] < 1.0).all()
        assert (qoi["damage_at_10mm"] >= 0).all()
        assert (qoi["damage_at_10mm"] <= DAMAGE_SATURATION + ATOL).all()

    def test_the_design_columns_joined_onto_every_row(self, qoi):
        for name in FEATURE_ORDER:
            assert qoi[name].notna().all()
        assert qoi["sample_id"].is_unique


class TestGridMatricesHaveNoNan:
    def test_both_gridded_signals_are_finite(self, repo_root):
        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
        )
        path = directory / RF2_GRID_PARQUET
        if not path.is_file():
            pytest.skip(f"{path} does not exist; run `ufem run ingest grid` first.")
        matrix = pd.read_parquet(path).drop(columns=["job"]).to_numpy(dtype=float)
        assert matrix.shape == (198, 201)
        assert np.isfinite(matrix).all()
