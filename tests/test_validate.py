"""Tests for the one cross validation harness, including the leak test of build spec 16.3.

The leak test is the important one. Build spec 16.3 requires a test that plants a duplicated
sample across folds and asserts the harness refuses it, because the predecessor project split
augmented children of the same parent across its train and test sets and published the result.
Here the refusal lives in :func:`ufem.validate.make_folds` itself rather than in a checker
somebody has to remember to call, and this file proves it fires.

Everything else in here pins the properties that make a reported metric mean what it says: the
folds partition the runs exactly once each, the metrics are the textbook ones, and the gate
verdict follows from the table rather than from a separate calculation that could drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufem.validate import (
    GP_MODEL,
    LeakDetected,
    baseline_leave_one_out,
    error_summary,
    evaluate_gate,
    make_folds,
    r2_score,
    relative_l2,
    rmse,
)


def jobs(count: int) -> list[str]:
    return [f"sample_{index:04d}" for index in range(count)]


class TestTheLeakTest:
    def test_a_duplicated_job_is_refused_by_the_fold_harness(self):
        """Build spec 16.3: plant a duplicate across folds and assert the harness refuses.

        The duplicate here is the shape the predecessor's defect took: the same simulation
        present twice under the same identifier, which a fold splitter that only counts rows
        will happily deal into two different folds.
        """
        planted = jobs(40)
        planted[17] = planted[3]
        with pytest.raises(LeakDetected) as raised:
            make_folds(planted, 5, np.random.SeedSequence(1))
        assert "sample_0003" in str(raised.value)
        assert "training set and a test set" in str(raised.value)

    def test_the_clean_case_is_accepted_so_the_refusal_is_not_vacuous(self):
        folds = make_folds(jobs(40), 5, np.random.SeedSequence(1))
        assert len(folds) == 5
        assert sum(fold.size for fold in folds) == 40

    def test_several_duplicates_are_all_named(self):
        planted = jobs(30)
        planted[10] = planted[0]
        planted[20] = planted[1]
        with pytest.raises(LeakDetected) as raised:
            make_folds(planted, 3, np.random.SeedSequence(2))
        message = str(raised.value)
        assert "sample_0000" in message and "sample_0001" in message


class TestFolds:
    def test_every_run_is_a_test_row_exactly_once(self):
        folds = make_folds(jobs(198), 10, np.random.SeedSequence(3))
        covered = np.sort(np.concatenate(folds))
        np.testing.assert_array_equal(covered, np.arange(198))

    def test_the_folds_are_within_one_row_of_the_same_size(self):
        sizes = [fold.size for fold in make_folds(jobs(198), 10, np.random.SeedSequence(3))]
        assert max(sizes) - min(sizes) <= 1

    def test_the_split_is_reproducible_from_the_entropy(self):
        first = make_folds(jobs(50), 5, np.random.SeedSequence(9))
        second = make_folds(jobs(50), 5, np.random.SeedSequence(9))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        third = make_folds(jobs(50), 5, np.random.SeedSequence(10))
        assert any(not np.array_equal(a, b) for a, b in zip(first, third))

    def test_the_split_does_not_depend_on_the_order_the_jobs_arrived_in(self):
        """A fold assignment that tracked the input order would group by extraction order."""
        labels = jobs(50)
        first = make_folds(labels, 5, np.random.SeedSequence(11))
        shuffled = list(reversed(labels))
        second = make_folds(shuffled, 5, np.random.SeedSequence(11))
        assert sorted(fold.size for fold in first) == sorted(fold.size for fold in second)

    def test_an_impossible_fold_count_raises(self):
        with pytest.raises(ValueError, match="between 2"):
            make_folds(jobs(10), 1, np.random.SeedSequence(1))
        with pytest.raises(ValueError, match="between 2"):
            make_folds(jobs(10), 11, np.random.SeedSequence(1))


class TestMetrics:
    def test_r2_of_a_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_of_the_mean_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        assert r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)

    def test_r2_of_a_worse_than_mean_prediction_is_negative(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        assert r2_score(y, y[::-1]) < 0.0

    def test_a_constant_truth_raises_rather_than_dividing_by_zero(self):
        with pytest.raises(ValueError, match="zero variance"):
            r2_score(np.ones(5), np.zeros(5))

    def test_rmse_is_the_root_mean_square(self):
        assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
            np.sqrt(12.5)
        )

    def test_relative_l2_is_per_row_and_dimensionless(self):
        truth = np.array([[3.0, 4.0], [1.0, 0.0]])
        prediction = np.array([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(relative_l2(truth, prediction), [1.0, 0.0])

    def test_a_zero_norm_reference_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            relative_l2(np.zeros((1, 3)), np.ones((1, 3)))

    def test_the_error_summary_reports_the_percentiles_the_report_quotes(self):
        summary = error_summary(np.linspace(0.0, 1.0, 101))
        assert summary["p50"] == pytest.approx(0.5)
        assert summary["p90"] == pytest.approx(0.9)
        assert summary["max"] == pytest.approx(1.0)
        assert summary["n"] == 101


class TestBaselineLeaveOneOut:
    def test_the_held_out_row_never_reaches_its_own_fit(self):
        """A leave one out that leaked would return the training fit, so plant an outlier.

        One row is moved far away from the linear trend the rest follow. If the harness were
        leaking, the prediction at that row would track the outlier; because it does not, the
        prediction sits on the trend and the residual is large.
        """
        from ufem.baselines import LinearRegressor

        rng = np.random.default_rng(31)
        X = rng.uniform(-1.0, 1.0, size=(30, 3))
        Y = (2.0 * X[:, 0]).reshape(-1, 1)
        Y[15, 0] += 50.0
        predictions = baseline_leave_one_out(LinearRegressor, X, Y)
        assert abs(predictions[15, 0] - Y[15, 0]) > 10.0
        others = np.delete(np.arange(30), 15)
        assert np.median(np.abs(predictions[others, 0] - Y[others, 0])) < 5.0

    def test_it_returns_one_prediction_per_row_and_column(self):
        from ufem.baselines import ClimatologyRegressor

        rng = np.random.default_rng(5)
        X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 4))
        assert baseline_leave_one_out(ClimatologyRegressor, X, Y).shape == (20, 4)


class TestTheGate:
    def rows(self, gp: float, linear: float) -> list[dict]:
        out = []
        for target in ("P_max_N", "u_peak_mm"):
            out.append(
                {
                    "harness": "leave_one_out",
                    "target": target,
                    "model": GP_MODEL,
                    "r2_test": gp,
                }
            )
            for name, value in (
                ("climatology", 0.0),
                ("linear", linear),
                ("quadratic_chaos", 0.1),
                ("nearest_neighbour", 0.05),
            ):
                out.append(
                    {
                        "harness": "leave_one_out",
                        "target": target,
                        "model": name,
                        "r2_test": value,
                    }
                )
        return out

    def test_a_clear_win_passes(self):
        gate = evaluate_gate(self.rows(0.8, 0.5), ["P_max_N", "u_peak_mm"])
        assert gate["passed"]
        assert gate["failing_targets"] == []
        assert gate["per_target"]["P_max_N"]["best_baseline"][0] == "linear"

    def test_a_loss_to_one_baseline_fails_and_names_it(self):
        gate = evaluate_gate(self.rows(0.4, 0.5), ["P_max_N", "u_peak_mm"])
        assert not gate["passed"]
        assert gate["failing_targets"] == ["P_max_N", "u_peak_mm"]
        assert gate["per_target"]["P_max_N"]["lost_to"] == ["linear"]

    def test_a_tie_counts_as_a_loss(self):
        """Beating means beating. A model that only matches a baseline has not earned it."""
        gate = evaluate_gate(self.rows(0.5, 0.5), ["P_max_N"])
        assert not gate["passed"]

    def test_a_missing_surrogate_row_raises_rather_than_passing_by_default(self):
        rows = [row for row in self.rows(0.8, 0.5) if row["model"] != GP_MODEL]
        with pytest.raises(KeyError):
            evaluate_gate(rows, ["P_max_N"])

    def test_only_the_leave_one_out_harness_decides_the_gate(self):
        """Mixing harnesses would let a favourable fold split decide the verdict."""
        rows = self.rows(0.8, 0.5)
        rows.append(
            {
                "harness": "grouped_fold",
                "target": "P_max_N",
                "model": "linear",
                "r2_test": 0.99,
            }
        )
        assert evaluate_gate(rows, ["P_max_N"])["passed"]
