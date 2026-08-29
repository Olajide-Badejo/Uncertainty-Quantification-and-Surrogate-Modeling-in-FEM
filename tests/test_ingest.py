"""Unit tests for the ingest stage: the dedup filter, the contracts, and the cache.

These run on synthetic frames rather than the 1.87 million row campaign, so a broken filter
fails in milliseconds with a readable diff. The one test that touches the real data is
marked slow and lives at the bottom.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ufem.config import config_hash, load_config
from ufem.ingest import (
    DESIGN_MAX_ABS_CORRELATION,
    EXPECTED_DEDUP_JOBS,
    EXPECTED_DEDUP_ROWS,
    STAGE_NAME,
    deduplicate_by_time,
    job_to_sample_id,
    raw_paths,
    strict_increasing_mask,
    verify_design_independence,
    verify_displacement_control,
)
from ufem.manifest import stage_dir


def _curve(job: str, times: list[float]) -> pd.DataFrame:
    """A synthetic displacement controlled curve: U2 = 20 t mm, force linear in t."""
    time = np.asarray(times, dtype=float)
    return pd.DataFrame(
        {
            "job": pd.Series([job] * len(time), dtype="string"),
            "time": time,
            "U2": 20.0 * time,
            "RF2": 1000.0 * time,
        }
    )


class TestStrictIncreasingMask:
    def test_keeps_everything_when_already_strictly_increasing(self):
        mask = strict_increasing_mask(np.array([0.0, 0.1, 0.2, 0.3]))
        assert mask.tolist() == [True, True, True, True]

    def test_drops_the_repeat_and_keeps_the_first(self):
        mask = strict_increasing_mask(np.array([0.0, 0.1, 0.1, 0.2]))
        assert mask.tolist() == [True, True, False, True]

    def test_drops_a_run_of_three_identical_stamps(self):
        mask = strict_increasing_mask(np.array([0.0, 0.5, 0.5, 0.5, 0.9]))
        assert mask.tolist() == [True, True, False, False, True]

    def test_drops_a_backward_step_from_a_solver_cutback(self):
        mask = strict_increasing_mask(np.array([0.0, 0.4, 0.3, 0.5]))
        assert mask.tolist() == [True, True, False, True]

    def test_empty_input_gives_an_empty_mask(self):
        assert strict_increasing_mask(np.array([])).tolist() == []

    def test_the_survivors_are_strictly_increasing(self):
        values = np.array([0.0, 0.2, 0.2, 0.1, 0.4, 0.4, 0.9])
        kept = values[strict_increasing_mask(values)]
        assert np.all(np.diff(kept) > 0.0)


class TestDeduplicateByTime:
    def test_a_clean_frame_loses_no_rows(self):
        frame = pd.concat(
            [_curve("sample_000", [0.0, 0.1, 0.2]), _curve("sample_001", [0.0, 0.5, 1.0])],
            ignore_index=True,
        )
        cleaned, report = deduplicate_by_time(frame)
        assert len(cleaned) == len(frame)
        assert report["rows_removed"].sum() == 0

    def test_duplicated_stamps_are_removed_and_counted_per_job(self):
        frame = pd.concat(
            [
                _curve("sample_000", [0.0, 0.1, 0.1, 0.2]),
                _curve("sample_001", [0.0, 0.5, 1.0]),
                _curve("sample_002", [0.0, 0.3, 0.3, 0.3, 0.7]),
            ],
            ignore_index=True,
        )
        cleaned, report = deduplicate_by_time(frame)
        removed = dict(zip(report["job"], report["rows_removed"], strict=True))
        assert removed == {"sample_000": 1, "sample_001": 0, "sample_002": 2}
        assert int((report["rows_removed"] > 0).sum()) == 2
        assert int(report["rows_removed"].sum()) == 3
        assert len(cleaned) == len(frame) - 3

    def test_one_job_duplication_does_not_leak_into_the_next(self):
        """Every job restarts at t = 0, so a per job filter is not a global one."""
        frame = pd.concat(
            [_curve("sample_000", [0.0, 0.1, 0.1]), _curve("sample_001", [0.0, 0.1, 0.2])],
            ignore_index=True,
        )
        cleaned, _ = deduplicate_by_time(frame)
        assert len(cleaned[cleaned["job"] == "sample_001"]) == 3

    def test_each_survivor_has_a_strictly_increasing_time_axis(self):
        frame = pd.concat(
            [
                _curve("sample_000", [0.0, 0.2, 0.2, 0.4]),
                _curve("sample_001", [0.0, 0.1, 0.1, 0.1, 0.6]),
            ],
            ignore_index=True,
        )
        cleaned, _ = deduplicate_by_time(frame)
        for _, group in cleaned.groupby("job"):
            assert np.all(np.diff(group["time"].to_numpy(float)) > 0.0)

    def test_the_first_row_of_a_duplicated_pair_survives(self):
        frame = pd.DataFrame(
            {
                "job": pd.Series(["sample_000"] * 3, dtype="string"),
                "time": [0.0, 0.5, 0.5],
                "U2": [0.0, 10.0, 10.0],
                "RF2": [0.0, 111.0, 222.0],
            }
        )
        cleaned, _ = deduplicate_by_time(frame)
        assert cleaned["RF2"].tolist() == [0.0, 111.0]

    def test_a_missing_column_raises_naming_it(self):
        with pytest.raises(KeyError, match="time"):
            deduplicate_by_time(pd.DataFrame({"job": ["sample_000"]}))


class TestDisplacementControl:
    def test_a_conforming_frame_returns_its_measured_deviation(self):
        frame = _curve("sample_000", [0.0, 0.25, 0.5, 1.0])
        assert verify_displacement_control(frame, "synthetic") == pytest.approx(0.0, abs=1e-12)

    def test_the_negative_sign_convention_is_accepted(self):
        """The solver writes U2 downward negative; only the magnitude is contracted."""
        frame = _curve("sample_000", [0.0, 0.5, 1.0])
        frame["U2"] = -frame["U2"]
        assert verify_displacement_control(frame, "synthetic") == pytest.approx(0.0, abs=1e-12)

    def test_a_deviation_just_inside_the_tolerance_passes(self):
        frame = _curve("sample_000", [0.0, 0.5, 1.0])
        frame.loc[1, "U2"] += 9e-4
        assert verify_displacement_control(frame, "synthetic") == pytest.approx(9e-4)

    def test_a_violating_row_raises_and_names_the_job_and_the_deviation(self):
        frame = _curve("sample_000", [0.0, 0.5, 1.0])
        frame.loc[1, "U2"] = 12.0
        with pytest.raises(ValueError, match="displacement control"):
            verify_displacement_control(frame, "synthetic")

    def test_the_diagnostic_reports_the_offending_job(self):
        frame = pd.concat(
            [_curve("sample_000", [0.0, 0.5]), _curve("sample_007", [0.0, 0.5])],
            ignore_index=True,
        )
        frame.loc[3, "U2"] = 3.0
        with pytest.raises(ValueError, match="sample_007"):
            verify_displacement_control(frame, "synthetic")


class TestJobToSampleId:
    def test_a_well_formed_label_maps_to_its_integer(self):
        jobs = pd.Series(["sample_000", "sample_042", "sample_399"], dtype="string")
        assert job_to_sample_id(jobs).tolist() == [0, 42, 399]

    def test_a_malformed_label_raises_rather_than_becoming_a_sentinel(self):
        """The v1 audit returned -1 here inside a bare except, merging every bad row."""
        jobs = pd.Series(["sample_000", "sample_bad"], dtype="string")
        with pytest.raises(ValueError, match="sample_bad"):
            job_to_sample_id(jobs)


class TestDesignIndependence:
    """Build spec 6.1: the realized LHS design cross correlates at |r| <= 0.05."""

    def test_an_exactly_orthogonal_design_passes_with_zero_correlation(self):
        """A full factorial over the three inputs: every pairwise correlation is exactly 0.

        Constructed rather than sampled, so the test asserts the estimator on a case whose
        answer is known in closed form instead of on a lucky draw.
        """
        levels = np.linspace(-1.0, 1.0, 8)
        a, b, c = (axis.ravel() for axis in np.meshgrid(levels, levels, levels, indexing="ij"))
        design = pd.DataFrame(
            {
                "Fcm_MPa": 28.0 + 8.0 * a,
                "c_nom_bottom_mm": 27.0 + 5.0 * b,
                "c_nom_top_mm": 223.0 + 10.0 * c,
            }
        )
        worst = verify_design_independence(design)
        assert worst == pytest.approx(0.0, abs=1e-12)

    def test_the_real_lhs_design_satisfies_the_bound(self, repo_root):
        """Build spec 6.1 measured |r| <= 0.046 on the executed campaign's design."""
        config = load_config(repo_root)
        path = raw_paths(repo_root, config)["design_csv"]
        if not path.is_file():
            pytest.skip(f"{path} is not present.")
        worst = verify_design_independence(pd.read_csv(path))
        assert 0.0 < worst <= DESIGN_MAX_ABS_CORRELATION

    def test_a_collinear_pair_raises(self):
        """Two marginals swept in step: exactly the defect the check exists to catch."""
        levels = np.linspace(-1.0, 1.0, 8)
        a, _, c = (axis.ravel() for axis in np.meshgrid(levels, levels, levels, indexing="ij"))
        design = pd.DataFrame(
            {
                "Fcm_MPa": 28.0 + 8.0 * a,
                "c_nom_bottom_mm": 27.0 + 5.0 * a,
                "c_nom_top_mm": 223.0 + 10.0 * c,
            }
        )
        with pytest.raises(ValueError, match="cross correlation"):
            verify_design_independence(design)


@pytest.mark.slow
class TestAgainstTheRealCampaign:
    """The counts build spec 6.1 pins, measured from the inherited CSVs themselves."""

    def test_the_dedup_counts_are_26_jobs_and_165_rows(self, repo_root):
        config = load_config(repo_root)
        path = raw_paths(repo_root, config)["load_displacement_csv"]
        if not path.is_file():
            pytest.skip(f"{path} is not present; the raw CSVs are not tracked by git.")
        frame = pd.read_csv(path, dtype={"job": "string"})
        frame = frame.sort_values(["job", "time"], kind="mergesort").reset_index(drop=True)
        _, report = deduplicate_by_time(frame)
        assert int((report["rows_removed"] > 0).sum()) == EXPECTED_DEDUP_JOBS
        assert int(report["rows_removed"].sum()) == EXPECTED_DEDUP_ROWS


@pytest.mark.slow
def test_rerunning_ingest_is_a_cache_hit_that_leaves_the_manifest_untouched(repo_root):
    """Build spec 17.1: a stage whose cache key is unchanged is skipped, not redone."""
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(
            f"{manifest_path} does not exist; run `ufem run ingest` before the cache test."
        )
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    before_mtime = manifest_path.stat().st_mtime_ns

    done = subprocess.run(
        [sys.executable, "-m", "ufem.runner", "run", STAGE_NAME],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "[cache hit] ingest" in done.stdout, done.stdout

    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after["written_at"] == before["written_at"]
    assert after["outputs"] == before["outputs"]
    assert manifest_path.stat().st_mtime_ns == before_mtime
