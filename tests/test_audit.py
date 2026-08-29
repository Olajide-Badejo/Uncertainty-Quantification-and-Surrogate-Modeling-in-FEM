"""The Phase P2 gate: the audit stage must reclassify the campaign, not recall it.

Build spec 9.4 and the P2 gate of section 22. Two layers of test live here.

The unit layer drives :func:`ufem.audit.classify_samples` and the statistics helpers on
synthetic frames that are small enough to reason about by hand. Every classification tier
gets a positive case, including ``partial``, which the real campaign never produces: a tier
that is only ever exercised by data that happens not to contain it is an untested branch,
and the whole reason the tier exists is the day a Track B rerun stops short.

The integration layer runs against the artifact store and asserts the two things the P2
gate names: that all 400 rows agree with ``data/audit_reference/sample_validity.csv`` and
that the regenerated quartile failure rates equal the committed ones. Those tests carry the
``fullstack`` marker because they need the stages to have run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ufem.audit import (
    CENSORING_JSON,
    COMPLETION_JSON,
    EXPECTED_STATUS_COUNTS,
    REFERENCE_VALIDITY_CSV,
    STAGE_NAME,
    STATUS_MISSING,
    STATUS_PARTIAL,
    STATUS_VALID,
    VALIDITY_PARQUET,
    WEIGHTING_JSON,
    bootstrap_auc_interval,
    censoring_statistics,
    classify_samples,
    compare_with_reference,
    expected_calibration_error,
    fit_completion_model,
    quantile_failure_table,
    reliability_table,
    status_counts,
    weighted_moments,
)
from ufem.config import FEATURE_ORDER, config_hash, load_config
from ufem.manifest import stage_dir

#: A design row that sits in the middle of every marginal, so a synthetic case is never
#: accidentally testing an edge of the input space at the same time as the classification.
NOMINAL = {"Fcm_MPa": 28.0, "c_nom_bottom_mm": 27.0, "c_nom_top_mm": 223.0}


def _design(sample_ids: list[int]) -> pd.DataFrame:
    """A minimal design frame carrying the feature contract columns."""
    return pd.DataFrame(
        [{"sample_id": sample_id, **NOMINAL} for sample_id in sample_ids]
    )


def _curve(sample_id: int, u_end: float = 20.0, n: int = 200, t_end: float = 1.0):
    """One synthetic displacement controlled curve, complete unless told otherwise."""
    step_time = np.linspace(0.0, t_end, n)
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "job": f"sample_{sample_id:03d}",
            "time": step_time,
            "U2": -np.linspace(0.0, u_end, n),
            "RF2": np.linspace(0.0, 40000.0, n),
        }
    )


class TestClassification:
    def test_a_complete_curve_is_valid(self, repo_root):
        config = load_config(repo_root)
        result = classify_samples(_design([0]), _curve(0), config)
        assert list(result["status"]) == [STATUS_VALID]
        assert result.loc[0, "present"]
        assert "covers" in result.loc[0, "reason"]

    def test_a_sample_with_no_rows_is_missing(self, repo_root):
        config = load_config(repo_root)
        load = _curve(0)
        result = classify_samples(_design([0, 1]), load, config)
        assert dict(zip(result["sample_id"], result["status"])) == {
            0: STATUS_VALID,
            1: STATUS_MISSING,
        }
        assert result.loc[1, "n_points"] == 0
        assert not result.loc[1, "present"]

    def test_a_curve_that_stops_short_of_the_grid_is_partial(self, repo_root):
        """The tier the inherited campaign never exercises, exercised deliberately."""
        config = load_config(repo_root)
        result = classify_samples(_design([0]), _curve(0, u_end=12.0, t_end=0.6), config)
        assert list(result["status"]) == [STATUS_PARTIAL]
        assert "stops at 12" in result.loc[0, "reason"]

    def test_a_curve_with_too_few_increments_is_partial(self, repo_root):
        config = load_config(repo_root)
        n = config.pipeline.audit.min_points - 1
        result = classify_samples(_design([0]), _curve(0, n=n), config)
        assert list(result["status"]) == [STATUS_PARTIAL]
        assert "increments" in result.loc[0, "reason"]

    def test_a_curve_whose_displacement_reverses_is_partial(self, repo_root):
        config = load_config(repo_root)
        load = _curve(0)
        u2 = load["U2"].to_numpy(dtype=float).copy()
        u2[100] = u2[99] * 0.5
        load["U2"] = u2
        result = classify_samples(_design([0]), load, config)
        assert list(result["status"]) == [STATUS_PARTIAL]
        assert "decreases" in result.loc[0, "reason"]

    def test_a_curve_that_stops_short_in_step_time_is_partial(self, repo_root):
        """Full displacement range but an unfinished step is still not a completed run."""
        config = load_config(repo_root)
        result = classify_samples(_design([0]), _curve(0, t_end=0.5), config)
        assert list(result["status"]) == [STATUS_PARTIAL]
        assert "step time" in result.loc[0, "reason"]

    def test_every_design_row_leaves_with_a_status(self, repo_root):
        config = load_config(repo_root)
        design = _design(list(range(12)))
        load = pd.concat([_curve(index) for index in (0, 3, 7)], ignore_index=True)
        result = classify_samples(design, load, config)
        assert len(result) == 12
        assert result["status"].notna().all()
        assert result["reason"].str.len().gt(0).all()
        assert status_counts(result) == {
            STATUS_VALID: 3,
            STATUS_MISSING: 9,
            STATUS_PARTIAL: 0,
        }

    def test_a_missing_design_column_raises_naming_it(self, repo_root):
        config = load_config(repo_root)
        design = _design([0]).drop(columns=["c_nom_top_mm"])
        with pytest.raises(KeyError, match="c_nom_top_mm"):
            classify_samples(design, _curve(0), config)


class TestCensoringStatistics:
    def test_quantile_bins_hold_equal_counts_and_measure_the_planted_rate(self):
        values = np.arange(100, dtype=float)
        failed = values < 25.0
        table = quantile_failure_table(values, failed, 4)
        assert [row["n"] for row in table] == [25, 25, 25, 25]
        assert [row["fail_rate"] for row in table] == [1.0, 0.0, 0.0, 0.0]

    def test_a_mismatched_shape_raises(self):
        with pytest.raises(ValueError, match="matching shapes"):
            quantile_failure_table(np.arange(10.0), np.ones(9, dtype=bool), 4)

    def test_a_planted_dependence_is_found_significant(self, repo_root):
        """A synthetic censoring pattern must be detected, not merely tolerated."""
        config = load_config(repo_root)
        rng = np.random.default_rng(np.random.SeedSequence(20260830))
        n = 400
        frame = pd.DataFrame(
            {
                "sample_id": np.arange(n),
                "Fcm_MPa": rng.normal(28.0, 2.8, n),
                "c_nom_bottom_mm": rng.normal(27.0, 3.0, n),
                "c_nom_top_mm": rng.normal(223.0, 5.0, n),
            }
        )
        # Failure driven entirely by the top cover, which is the real pattern's shape.
        frame["status"] = np.where(
            frame["c_nom_top_mm"] < 223.0, STATUS_MISSING, STATUS_VALID
        )
        result = censoring_statistics(frame, config)
        assert result["by_input"]["c_nom_top_mm"]["significant_at_level"]
        assert result["by_input"]["c_nom_top_mm"]["point_biserial_r"] < -0.5
        assert not result["by_input"]["Fcm_MPa"]["significant_at_level"]

    def test_a_disagreeing_reference_raises_rather_than_being_absorbed(self):
        validity = pd.DataFrame(
            {"sample_id": [0, 1], "status": [STATUS_VALID, STATUS_MISSING]}
        )
        reference = pd.DataFrame(
            {"sample_id": [0, 1], "status": [STATUS_VALID, STATUS_VALID]}
        )
        with pytest.raises(AssertionError, match="disagrees"):
            compare_with_reference(validity, reference)


class TestCompletionModelHelpers:
    def test_weighted_moments_reduce_to_the_unweighted_ones_at_unit_weights(self):
        values = np.array([1.0, 2.0, 4.0, 8.0])
        result = weighted_moments(values, np.ones_like(values))
        assert result["mean"] == pytest.approx(float(values.mean()))
        assert result["std"] == pytest.approx(float(values.std(ddof=1)))

    def test_weighted_moments_rejects_a_non_positive_weight(self):
        with pytest.raises(ValueError, match="positive weights"):
            weighted_moments(np.array([1.0, 2.0]), np.array([1.0, 0.0]))

    def test_a_perfect_score_gives_an_interval_that_contains_one(self):
        rng = np.random.default_rng(np.random.SeedSequence(1))
        y = np.repeat([0, 1], 50)
        probability = y.astype(float) * 0.9 + 0.05
        interval = bootstrap_auc_interval(y, probability, 200, 0.90, rng)
        assert interval["auc_low"] == pytest.approx(1.0)
        assert interval["auc_high"] == pytest.approx(1.0)

    def test_the_reliability_table_covers_the_whole_unit_interval(self):
        rng = np.random.default_rng(np.random.SeedSequence(2))
        probability = rng.uniform(0.0, 1.0, 500)
        y = (rng.uniform(0.0, 1.0, 500) < probability).astype(int)
        table = reliability_table(y, probability, 10)
        assert len(table) == 10
        assert sum(row["n"] for row in table) == 500
        # Probabilities that mean what they say give a small calibration error.
        assert expected_calibration_error(table, 500) < 0.10

    def test_an_empty_bin_is_reported_rather_than_dropped(self):
        y = np.array([0, 1, 0, 1])
        probability = np.array([0.45, 0.55, 0.45, 0.55])
        table = reliability_table(y, probability, 10)
        assert len(table) == 10
        assert [row["n"] for row in table].count(0) == 8
        assert all(row["mean_predicted"] is None for row in table if row["n"] == 0)


@pytest.fixture(scope="module")
def audit_dir(repo_root):
    """The audit stage's artifact directory for the current config, or a skip."""
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
    )
    if not (directory / VALIDITY_PARQUET).is_file():
        pytest.skip(f"{directory} has no audit artifacts; run `ufem run all` before this test.")
    return directory


@pytest.mark.fullstack
class TestAgainstTheArtifactStore:
    """The P2 gate proper: run against what the pipeline actually wrote."""

    def test_the_split_is_the_one_the_campaign_produced(self, audit_dir):
        validity = pd.read_parquet(audit_dir / VALIDITY_PARQUET)
        assert status_counts(validity) == EXPECTED_STATUS_COUNTS

    def test_every_one_of_the_400_rows_matches_the_committed_reference(
        self, repo_root, audit_dir
    ):
        """The gate build spec 22 names: exact agreement, sample for sample."""
        validity = pd.read_parquet(audit_dir / VALIDITY_PARQUET)
        reference = pd.read_csv(
            repo_root / "data" / "audit_reference" / REFERENCE_VALIDITY_CSV
        )
        result = compare_with_reference(validity, reference)
        assert result["n_compared"] == 400
        assert result["n_disagreements"] == 0

    def test_the_quartile_failure_rates_regenerate_exactly(self, repo_root, audit_dir):
        measured = json.loads((audit_dir / CENSORING_JSON).read_text(encoding="utf-8"))
        reference = json.loads(
            (repo_root / "data" / "audit_reference" / "audit_summary.json").read_text(
                encoding="utf-8"
            )
        )
        rates = reference["failure_clustering"]["failure_rate_by_input_quartile"]
        for name in FEATURE_ORDER:
            for row in measured["by_input"][name]["quantile_failure_rates"]:
                expected = rates[name][row["bin"]]
                assert row["n"] == expected["n"]
                assert row["n_failed"] == expected["n_failed"]
                assert row["fail_rate"] == pytest.approx(expected["fail_rate"])
        assert measured["reference_gate"]["n_disagreements"] == 0

    def test_the_censoring_tests_reproduce_the_audits_p_values(self, repo_root, audit_dir):
        """The two effects build spec 5.5 names, regenerated rather than quoted."""
        measured = json.loads((audit_dir / CENSORING_JSON).read_text(encoding="utf-8"))
        reference = json.loads(
            (repo_root / "data" / "audit_reference" / "audit_summary.json").read_text(
                encoding="utf-8"
            )
        )
        biserial = reference["failure_clustering"]["point_biserial_input_vs_failure"]
        for name in FEATURE_ORDER:
            block = measured["by_input"][name]
            assert block["chi2_p_value"] == pytest.approx(
                biserial[name]["chi2_quartile_p"], rel=1e-9
            )
            assert block["point_biserial_r"] == pytest.approx(
                biserial[name]["pearson_vs_failure_flag"], rel=1e-9
            )
            assert block["welch_p_value"] == pytest.approx(
                biserial[name]["welch_t_p"], rel=1e-9
            )

    def test_the_completion_model_reports_a_cross_validated_interval(self, audit_dir):
        report = json.loads((audit_dir / COMPLETION_JSON).read_text(encoding="utf-8"))
        interval = report["cv_roc_auc_interval"]
        assert interval["auc_low"] < report["cv_roc_auc"] < interval["auc_high"]
        assert interval["level"] == pytest.approx(0.90)
        assert interval["n_resamples"] == 1000
        # Binding law 3: the number that matters is the out of sample one, and it must beat
        # the base rate predictor it is compared against.
        assert report["cv_brier_score"] < report["baseline_brier_score"]
        assert report["n_samples"] == 400

    def test_the_importance_weighting_study_covers_every_headline_qoi(self, audit_dir):
        study = json.loads((audit_dir / WEIGHTING_JSON).read_text(encoding="utf-8"))
        assert study["n_weighted_samples"] == EXPECTED_STATUS_COUNTS[STATUS_VALID]
        assert "P_max_N" in study["by_qoi"]
        # Every weight is finite and positive, so no survivor was dropped or made infinite.
        assert study["weight"]["min"] > 0.0
        assert study["weight"]["effective_sample_size"] > 0.0

    def test_refitting_the_completion_model_reproduces_the_recorded_auc(
        self, repo_root, audit_dir
    ):
        """Determinism (build spec 17.2): same seed, same config, same number."""
        config = load_config(repo_root)
        validity = pd.read_parquet(audit_dir / VALIDITY_PARQUET)
        recorded = json.loads((audit_dir / COMPLETION_JSON).read_text(encoding="utf-8"))
        _, report = fit_completion_model(
            validity, config, config.pipeline.seed_entropy
        )
        assert report["kind"] == recorded["kind"]
        assert report["cv_roc_auc"] == pytest.approx(recorded["cv_roc_auc"], rel=1e-12)
        assert report["cv_brier_score"] == pytest.approx(
            recorded["cv_brier_score"], rel=1e-12
        )
        assert report["cv_roc_auc_interval"]["auc_low"] == pytest.approx(
            recorded["cv_roc_auc_interval"]["auc_low"], rel=1e-12
        )
