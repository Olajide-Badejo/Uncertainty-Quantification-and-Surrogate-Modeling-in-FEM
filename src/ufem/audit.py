"""Stage ``audit``: censoring is data (binding law 4, build spec 9.4).

202 of the 400 designed samples produced nothing, and the failures cluster in a known
corner of the input space, so the surviving 198 are a biased subsample. This stage refuses
to treat that as a nuisance. It does four things, in order:

1. **Reclassifies all 400 design rows from the ingest artifacts alone.** Never from a
   literal list. Build spec 5.5 records that the v1 extraction hard coded its 198 sample
   list as an unexplained literal; the classification here is derived from the bytes ingest
   wrote, by criteria that live in ``configs/pipeline.yaml`` rather than in this file, and
   is asserted row for row against ``data/audit_reference/sample_validity.csv``. Both are
   derived from the same raw inputs, so exact agreement is the only acceptable outcome.
2. **Regenerates the censoring statistics.** Failure rate by quartile of each input, a chi
   squared test of independence per input, point biserial correlations, and a Welch test of
   the group means. The quartile rates are gated against the committed audit values.
3. **Fits a completion probability model** P(complete | Fcm, c_bottom, c_top) over all 400
   design rows, cross validated with a bootstrap interval on the ROC AUC, a Brier score,
   and a reliability table. The Gaussian process classifier is primary; the L2 logistic
   regression is the pre authorized fallback of build spec 9.4, taken only when the GPC
   fails a stated guard, and which one shipped is recorded in the manifest.
4. **Stamps a validity domain**, the region where the completion probability clears the
   configured threshold and the design density is non negligible, which every downstream
   stage and the UI must consult through :mod:`ufem.validity`.

Plus the importance weighting sensitivity study of build spec 9.4: inverse probability of
completion weights on the 198 valid jobs, headline QoI statistics recomputed weighted
against unweighted, so the size of the censoring bias is measured rather than asserted.

Units: strength in MPa, covers in mm, force in N, displacement in mm. Probabilities, rates,
and correlations are dimensionless.

RNG discipline (build spec 17.3): one ``SeedSequence`` from the configured entropy, spawned
per consumer, and integer ``random_state`` for every scikit-learn estimator.
"""

from __future__ import annotations

import json
import pickle
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufem.config import FEATURE_ORDER, Config, features
from ufem.grid import QOI_PARQUET as GRID_QOI_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import DESIGN_PARQUET, LOAD_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest

STAGE_NAME = "audit"

#: Output file names inside the stage directory.
VALIDITY_PARQUET = "sample_validity.parquet"
CENSORING_JSON = "censoring_statistics.json"
COMPLETION_JSON = "completion_model.json"
COMPLETION_PICKLE = "completion_model.pkl"
VALIDITY_DOMAIN_JSON = "validity_domain.json"
WEIGHTING_JSON = "importance_weighting.json"

#: The three status tiers of build spec 9.4. ``valid`` is present and complete, ``missing``
#: is absent from the extracted data entirely, ``partial`` is present but incomplete. The
#: 2026-08-28 audit measured 198, 202 and 0; this stage measures them again every run.
STATUS_VALID = "valid"
STATUS_MISSING = "missing"
STATUS_PARTIAL = "partial"
STATUS_ORDER = (STATUS_VALID, STATUS_MISSING, STATUS_PARTIAL)

#: Build spec 6.1 and 5.5, asserted rather than assumed.
EXPECTED_STATUS_COUNTS = {STATUS_VALID: 198, STATUS_MISSING: 202, STATUS_PARTIAL: 0}

#: The committed reference the reclassification is gated against, relative to the repo root.
AUDIT_REFERENCE_DIR = "data/audit_reference"
REFERENCE_VALIDITY_CSV = "sample_validity.csv"
REFERENCE_SUMMARY_JSON = "audit_summary.json"

#: Quartile labels of the censoring tables, in ascending order of the input.
QUARTILE_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")

#: The headline QoIs the importance weighting study recomputes, with their units. Terminal
#: damage is deliberately absent: it is the banned QoI of build spec 5.6.
WEIGHTED_QOI_COLUMNS: dict[str, str] = {
    "P_max_N": "N",
    "u_peak_mm": "mm",
    "k0_N_per_mm": "N/mm",
    "E_abs_Nmm": "N mm",
    "P_residual_N": "N",
    "softening_ratio": "-",
    "u_damage_half_sat_mm": "mm",
    "damage_at_10mm": "-",
}


def _seed_children(seed_entropy: int, n: int) -> list[np.random.Generator]:
    """``n`` independent generators from one root ``SeedSequence`` (build spec 17.3)."""
    root = np.random.SeedSequence(seed_entropy)
    return [np.random.default_rng(child) for child in root.spawn(n)]


def _integer_random_state(seed_entropy: int) -> int:
    """A stable non negative 32 bit integer ``random_state`` for scikit-learn.

    Build spec ground rule 13 demands an integer, never a ``RandomState`` instance, so the
    estimator's own reproducibility does not depend on the state of a shared object. The
    value is derived from the configured entropy, so changing the entropy changes it.
    """
    (generator,) = _seed_children(seed_entropy, 1)
    return int(generator.integers(0, 2**31 - 1))


def classify_samples(design: pd.DataFrame, load: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Classify every design row as valid, missing, or partial from the ingest artifacts.

    ``design`` is the 400 row LHS design and ``load`` the deduplicated load displacement
    table, both as ingest wrote them. Returns one row per design row, in ``sample_id``
    order, carrying the status, the reason, and the measurements the status was decided on,
    so a classification can always be argued with rather than merely believed.

    The criteria come from ``config.pipeline.audit`` and are, for a present sample:

    - the curve covers the whole common displacement grid, within the configured tolerances;
    - it carries no non finite value;
    - the imposed displacement never decreases along the run;
    - it reached the target step time within the configured tolerance;
    - it holds at least the configured minimum number of increments.

    A sample absent from the extracted data is ``missing``. A present sample that fails any
    criterion is ``partial``: present, but not a complete run. There is no fourth tier and
    no silent drop; every one of the 400 rows leaves here with a status and a reason.
    """
    settings = config.pipeline.audit
    grid = config.pipeline.grid
    required_design = {"sample_id", *FEATURE_ORDER}
    missing_columns = sorted(required_design - set(design.columns))
    if missing_columns:
        raise KeyError(
            f"classify_samples needs design columns {sorted(required_design)}; the frame is "
            f"missing {missing_columns} and offers {list(design.columns)}."
        )
    required_load = {"sample_id", "time", "U2"}
    missing_columns = sorted(required_load - set(load.columns))
    if missing_columns:
        raise KeyError(
            f"classify_samples needs load columns {sorted(required_load)}; the frame is "
            f"missing {missing_columns} and offers {list(load.columns)}."
        )

    measured: dict[int, dict[str, Any]] = {}
    for sample_id, group in load.groupby("sample_id", sort=True):
        u_abs = np.abs(group["U2"].to_numpy(dtype=float))
        step_time = group["time"].to_numpy(dtype=float)
        finite = np.isfinite(u_abs) & np.isfinite(step_time)
        measured[int(sample_id)] = {
            "n_points": int(len(group)),
            "n_non_finite": int((~finite).size - int(finite.sum())) if finite.size else 0,
            "u_start_mm": float(u_abs.min()) if u_abs.size else float("nan"),
            "u_end_mm": float(u_abs.max()) if u_abs.size else float("nan"),
            "t_final": float(step_time.max()) if step_time.size else float("nan"),
            "min_u_increment_mm": float(np.diff(u_abs).min()) if u_abs.size > 1 else 0.0,
        }

    records = []
    for row in design.itertuples(index=False):
        sample_id = int(getattr(row, "sample_id"))
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "job": f"sample_{sample_id:03d}",
        }
        record.update({name: float(getattr(row, name)) for name in FEATURE_ORDER})
        stats_row = measured.get(sample_id)
        if stats_row is None:
            record.update(
                {
                    "present": False,
                    "n_points": 0,
                    "n_non_finite": 0,
                    "u_start_mm": float("nan"),
                    "u_end_mm": float("nan"),
                    "t_final": float("nan"),
                    "min_u_increment_mm": float("nan"),
                    "status": STATUS_MISSING,
                    "reason": "absent from the extracted load displacement data",
                }
            )
            records.append(record)
            continue

        record.update({"present": True, **stats_row})
        failures = []
        if stats_row["n_non_finite"] > 0:
            failures.append(f"{stats_row['n_non_finite']} non finite values")
        if stats_row["n_points"] < settings.min_points:
            failures.append(
                f"{stats_row['n_points']} increments, fewer than the "
                f"{settings.min_points} a usable curve needs"
            )
        if stats_row["u_start_mm"] > grid.u_min_mm + settings.u_start_tolerance_mm:
            failures.append(
                f"starts at {stats_row['u_start_mm']:.6g} mm, above the grid start "
                f"{grid.u_min_mm:.6g} mm"
            )
        if stats_row["u_end_mm"] < grid.u_max_mm - settings.u_end_tolerance_mm:
            failures.append(
                f"stops at {stats_row['u_end_mm']:.6g} mm, short of the grid end "
                f"{grid.u_max_mm:.6g} mm"
            )
        if stats_row["min_u_increment_mm"] < -settings.u_monotone_tolerance_mm:
            failures.append(
                f"displacement decreases by {-stats_row['min_u_increment_mm']:.3g} mm, "
                "which displacement control forbids"
            )
        if stats_row["t_final"] < settings.target_step_time - settings.step_time_tolerance:
            failures.append(
                f"ends at step time {stats_row['t_final']:.6g}, short of the target "
                f"{settings.target_step_time:.6g}"
            )
        if failures:
            record["status"] = STATUS_PARTIAL
            record["reason"] = "present but incomplete: " + "; ".join(failures)
        else:
            record["status"] = STATUS_VALID
            record["reason"] = (
                f"covers [{grid.u_min_mm:.6g}, {grid.u_max_mm:.6g}] mm over "
                f"{stats_row['n_points']} increments to step time "
                f"{stats_row['t_final']:.6g}, monotone, zero non finite"
            )
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    unclassified = frame["status"].isna().sum()
    if unclassified:
        raise AssertionError(
            f"{unclassified} of {len(frame)} design rows left classify_samples without a "
            "status. Every row must carry one; an unclassified row is a silent drop."
        )
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def status_counts(validity: pd.DataFrame) -> dict[str, int]:
    """Count each status tier, with every tier present even at zero."""
    counted = validity["status"].value_counts().to_dict()
    return {name: int(counted.get(name, 0)) for name in STATUS_ORDER}


def compare_with_reference(validity: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Compare the reclassification against the committed audit, row for row.

    Both classifications derive from the same raw inputs, so exact agreement is required
    rather than merely expected. A disagreement raises naming the offending samples: build
    spec section 24 says to stop and investigate, not to relax the comparison.
    """
    left = validity[["sample_id", "status"]].set_index("sample_id").sort_index()
    right = reference[["sample_id", "status"]].set_index("sample_id").sort_index()
    if list(left.index) != list(right.index):
        only_left = sorted(set(left.index) - set(right.index))
        only_right = sorted(set(right.index) - set(left.index))
        raise AssertionError(
            "the reclassification and the committed audit cover different sample sets: "
            f"{len(only_left)} only in the pipeline ({only_left[:5]}) and "
            f"{len(only_right)} only in the reference ({only_right[:5]})."
        )
    disagreements = left.index[left["status"].to_numpy() != right["status"].to_numpy()]
    if len(disagreements) > 0:
        examples = [
            f"sample {int(sample_id)}: pipeline {left.loc[sample_id, 'status']!r} against "
            f"reference {right.loc[sample_id, 'status']!r}"
            for sample_id in list(disagreements)[:5]
        ]
        raise AssertionError(
            f"the reclassification disagrees with data/audit_reference/"
            f"{REFERENCE_VALIDITY_CSV} on {len(disagreements)} of {len(left)} samples: "
            + "; ".join(examples)
            + ". Both are derived from the same raw inputs, so this is a real change in the "
            "data or in the criteria, not a tolerance to widen. Build spec section 24: stop "
            "and record the finding before touching either side."
        )
    return {
        "n_compared": int(len(left)),
        "n_disagreements": 0,
        "reference_status_counts": {
            name: int((right["status"] == name).sum()) for name in STATUS_ORDER
        },
    }


def quantile_failure_table(
    values: np.ndarray, failed: np.ndarray, n_bins: int
) -> list[dict[str, Any]]:
    """Failure rate per quantile bin of one input, with the bin edges that defined it.

    Bins are equal count quantile bins of the executed design, which is what makes the rate
    comparable across bins: each holds the same number of designed samples, so a difference
    in rate is a difference in outcome rather than in exposure.
    """
    array = np.asarray(values, dtype=float)
    flags = np.asarray(failed, dtype=bool)
    if array.shape != flags.shape:
        raise ValueError(
            f"quantile_failure_table needs matching shapes, got values {array.shape} and "
            f"failure flags {flags.shape}."
        )
    edges = np.quantile(array, np.linspace(0.0, 1.0, n_bins + 1))
    codes = np.clip(np.searchsorted(edges[1:-1], array, side="right"), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        in_bin = codes == index
        n = int(in_bin.sum())
        if n == 0:
            raise ValueError(
                f"quantile bin {index} of {n_bins} is empty, which means the input has "
                "fewer distinct values than bins and the rate table would be meaningless."
            )
        n_failed = int(flags[in_bin].sum())
        rows.append(
            {
                "bin": QUARTILE_LABELS[index] if n_bins == 4 else f"Q{index + 1}",
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "n": n,
                "n_failed": n_failed,
                "fail_rate": float(n_failed / n),
            }
        )
    return rows


def censoring_statistics(validity: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Regenerate the censoring statistics of build spec 5.5 from the classification.

    Per input: the quantile failure rate table, a chi squared test of independence between
    the quantile bin and the outcome, the point biserial correlation between the input and
    the failure indicator, and a Welch test of the two group means. Every one is measured
    from the 400 row classification, never read from the reference.
    """
    settings = config.pipeline.audit
    failed = (validity["status"] != STATUS_VALID).to_numpy(dtype=bool)
    n_failed = int(failed.sum())
    n_total = int(failed.size)
    per_input: dict[str, Any] = {}
    for name in FEATURE_ORDER:
        values = validity[name].to_numpy(dtype=float)
        table = quantile_failure_table(values, failed, settings.n_quantile_bins)
        contingency = np.array([[row["n_failed"], row["n"] - row["n_failed"]] for row in table])
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency, correction=False)
        # Point biserial against a 0/1 failure indicator is Pearson's r; scipy names it
        # separately because one variable is dichotomous, which is exactly this case.
        biserial = stats.pointbiserialr(failed.astype(float), values)
        welch = stats.ttest_ind(values[failed], values[~failed], equal_var=False)
        mannwhitney = stats.mannwhitneyu(values[failed], values[~failed], alternative="two-sided")
        per_input[name] = {
            "quantile_failure_rates": table,
            "chi2_statistic": float(chi2),
            "chi2_p_value": float(chi2_p),
            "chi2_dof": int(dof),
            "point_biserial_r": float(biserial.statistic),
            "point_biserial_p_value": float(biserial.pvalue),
            "mean_failed": float(values[failed].mean()),
            "mean_valid": float(values[~failed].mean()),
            "welch_t_statistic": float(welch.statistic),
            "welch_p_value": float(welch.pvalue),
            "mannwhitney_p_value": float(mannwhitney.pvalue),
            "significant_at_level": bool(chi2_p < settings.significance_level),
        }
    return {
        "n_designed": n_total,
        "n_failed": n_failed,
        "n_valid": n_total - n_failed,
        "overall_failure_rate": float(n_failed / n_total),
        "n_quantile_bins": int(settings.n_quantile_bins),
        "significance_level": float(settings.significance_level),
        "by_input": per_input,
    }


def check_quartile_rates_against_reference(
    measured: dict[str, Any], reference_summary: dict[str, Any]
) -> dict[str, Any]:
    """Gate the regenerated quartile failure rates against the committed audit values.

    The reference holds the rates for every input, so this compares them exactly. A rate is
    a ratio of two integers over the same 400 designed samples: there is no floating point
    slack to allow for and none is allowed.
    """
    reference_rates = reference_summary["failure_clustering"]["failure_rate_by_input_quartile"]
    compared = 0
    for name, block in measured["by_input"].items():
        if name not in reference_rates:
            raise KeyError(
                f"the committed audit summary carries no quartile failure rates for {name}; "
                f"it offers {sorted(reference_rates)}."
            )
        for row in block["quantile_failure_rates"]:
            expected = reference_rates[name].get(row["bin"])
            if expected is None:
                raise KeyError(
                    f"the committed audit summary has no {row['bin']!r} bin for {name}; it "
                    f"offers {sorted(reference_rates[name])}."
                )
            if int(expected["n"]) != row["n"] or int(expected["n_failed"]) != row["n_failed"]:
                raise AssertionError(
                    f"the regenerated failure rate for {name} {row['bin']} is "
                    f"{row['n_failed']}/{row['n']} but the committed audit recorded "
                    f"{expected['n_failed']}/{expected['n']}. Both count the same 400 "
                    "designed samples, so this is a change in the data or the binning, not "
                    "a rounding difference."
                )
            compared += 1
    return {"n_rates_compared": compared, "n_disagreements": 0}


def _build_completion_estimator(kind: str, config: Config, random_state: int) -> Pipeline:
    """A standardizing pipeline around one completion classifier.

    Standardization is inside the pipeline, not applied to the frame beforehand, so that
    every cross validation fold refits its scaler on its own training half. A scaler fitted
    on all 400 rows and then cross validated would leak the test fold's location and scale
    into the training fold, which is the fold honesty requirement of build spec 16.3.
    """
    settings = config.pipeline.audit.completion_model
    if kind == "gaussian_process":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(len(FEATURE_ORDER)),
            length_scale_bounds=settings.lengthscale_bounds,
            nu=settings.matern_nu,
        )
        estimator: Any = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=settings.restarts,
            random_state=random_state,
        )
    elif kind == "logistic":
        estimator = LogisticRegression(
            penalty="l2",
            C=settings.logistic_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"unknown completion model kind {kind!r}; build spec 9.4 names exactly "
            "'gaussian_process' (primary) and 'logistic' (fallback)."
        )
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def cross_validated_probabilities(
    X: np.ndarray, y: np.ndarray, kind: str, config: Config, random_state: int
) -> np.ndarray:
    """Out of fold completion probabilities from stratified k fold cross validation.

    Build spec binding law 3: out of sample or it did not happen. Every returned probability
    was produced by a model that never saw its own row, and the scaler is refitted inside
    each fold by the pipeline.
    """
    settings = config.pipeline.audit.completion_model
    splitter = StratifiedKFold(
        n_splits=settings.n_folds, shuffle=True, random_state=random_state
    )
    out_of_fold = np.full(y.shape, np.nan, dtype=float)
    for train_index, test_index in splitter.split(X, y):
        model = _build_completion_estimator(kind, config, random_state)
        model.fit(X[train_index], y[train_index])
        out_of_fold[test_index] = model.predict_proba(X[test_index])[:, 1]
    n_missing = int(np.isnan(out_of_fold).sum())
    if n_missing:
        raise AssertionError(
            f"{n_missing} of {y.size} rows received no out of fold prediction. Stratified "
            "k fold must cover every row exactly once; a gap here would silently shrink the "
            "sample the metrics are computed on."
        )
    return out_of_fold


def bootstrap_auc_interval(
    y: np.ndarray, probability: np.ndarray, n_resamples: int, level: float, rng: np.random.Generator
) -> dict[str, float]:
    """Percentile bootstrap interval on the ROC AUC of the out of fold probabilities.

    Resamples pairs, not classes separately, so the interval reflects the sampling variation
    of the whole design. A resample that draws only one class carries no AUC and is skipped,
    with the count of skips reported rather than hidden.
    """
    truth = np.asarray(y, dtype=int)
    scores = np.asarray(probability, dtype=float)
    n = truth.size
    values = []
    n_degenerate = 0
    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        resampled = truth[index]
        if resampled.min() == resampled.max():
            n_degenerate += 1
            continue
        values.append(roc_auc_score(resampled, scores[index]))
    if not values:
        raise ValueError(
            f"all {n_resamples} bootstrap resamples were single class, so no interval can be "
            "formed. That means the outcome is almost constant, which contradicts the "
            "198/202 split this stage measured."
        )
    array = np.asarray(values, dtype=float)
    tail = (1.0 - level) / 2.0
    return {
        "auc_low": float(np.quantile(array, tail)),
        "auc_high": float(np.quantile(array, 1.0 - tail)),
        "level": float(level),
        "n_resamples": int(n_resamples),
        "n_degenerate_resamples": int(n_degenerate),
    }


def reliability_table(
    y: np.ndarray, probability: np.ndarray, n_bins: int
) -> list[dict[str, Any]]:
    """Predicted against empirical completion rate in equal width probability bins.

    Equal width, not equal count, because the question a calibration table answers is
    whether a stated probability means what it says, and the statement is on the probability
    axis. Empty bins are reported as empty rather than dropped, so the table always shows
    where the model does and does not make predictions.
    """
    truth = np.asarray(y, dtype=float)
    scores = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    codes = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        in_bin = codes == index
        n = int(in_bin.sum())
        rows.append(
            {
                "bin_low": float(edges[index]),
                "bin_high": float(edges[index + 1]),
                "n": n,
                "mean_predicted": float(scores[in_bin].mean()) if n else None,
                "empirical_rate": float(truth[in_bin].mean()) if n else None,
            }
        )
    return rows


def expected_calibration_error(table: list[dict[str, Any]], n_total: int) -> float:
    """Sample weighted mean absolute gap between predicted and empirical rate."""
    if n_total <= 0:
        raise ValueError("expected_calibration_error needs a positive sample size.")
    total = 0.0
    for row in table:
        if row["n"] == 0:
            continue
        total += row["n"] * abs(row["mean_predicted"] - row["empirical_rate"])
    return float(total / n_total)


def _describe_fit(model: Pipeline, kind: str) -> dict[str, Any]:
    """The fitted hyperparameters, so the shape of the model is recorded, not just its score.

    For the Gaussian process this is the per feature lengthscale. A lengthscale far above
    the unit scale of the standardized features is the kernel stating that the input has no
    effect, which is the honest reading for the bottom cover here, and recording it is what
    lets the report say so rather than leaving it to a scikit-learn convergence warning.
    """
    estimator = model["clf"]
    if kind == "gaussian_process":
        lengthscale = np.atleast_1d(
            np.asarray(estimator.kernel_.k2.length_scale, dtype=float)
        )
        return {
            "kernel": str(estimator.kernel_),
            "log_marginal_likelihood": float(estimator.log_marginal_likelihood_value_),
            "length_scale_standardized": {
                name: float(lengthscale[index]) for index, name in enumerate(FEATURE_ORDER)
            },
        }
    return {
        "intercept": float(np.ravel(estimator.intercept_)[0]),
        "coefficients_standardized": {
            name: float(np.ravel(estimator.coef_)[index])
            for index, name in enumerate(FEATURE_ORDER)
        },
    }


def fit_completion_model(
    validity: pd.DataFrame, config: Config, seed_entropy: int
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit and evaluate the completion probability model of build spec 9.4.

    Returns the estimator fitted on all 400 design rows and a report holding the cross
    validated metrics of whichever estimator shipped. The primary is tried first; the
    logistic fallback is taken only when the primary fails a configured guard, and the
    report records the guard that failed so the substitution is never silent.
    """
    settings = config.pipeline.audit.completion_model
    random_state = _integer_random_state(seed_entropy)
    bootstrap_rng, _ = _seed_children(seed_entropy, 2)

    X = features(validity)
    y = (validity["status"] == STATUS_VALID).to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"the completion outcome is constant at {int(y[0])}, so no classifier can be "
            "fitted. The campaign measured a 198/202 split, so this is a data change."
        )

    attempts = []
    chosen_kind = None
    chosen_probability = None
    for kind in (settings.primary, settings.fallback):
        probability = cross_validated_probabilities(X, y, kind, config, random_state)
        auc = float(roc_auc_score(y, probability))
        spread = float(probability.max() - probability.min())
        rejections = []
        if auc < settings.min_auc:
            rejections.append(
                f"cross validated AUC {auc:.4f} is below the {settings.min_auc} floor"
            )
        if spread < settings.min_prediction_spread:
            rejections.append(
                f"the predictions span only {spread:.4g}, below the "
                f"{settings.min_prediction_spread} floor, so the model is effectively constant"
            )
        attempts.append(
            {
                "kind": kind,
                "cv_auc": auc,
                "prediction_spread": spread,
                "accepted": not rejections,
                "rejection_reasons": rejections,
            }
        )
        if not rejections:
            chosen_kind = kind
            chosen_probability = probability
            break

    if chosen_kind is None or chosen_probability is None:
        detail = "; ".join(
            f"{item['kind']}: " + ", ".join(item["rejection_reasons"]) for item in attempts
        )
        raise RuntimeError(
            "neither the primary completion model nor its pre authorized fallback cleared "
            f"the guards of configs/pipeline.yaml ({detail}). Build spec 9.4 authorizes one "
            "fallback, not an unbounded search, so this stops here rather than trying a "
            "third estimator that was never specified."
        )

    interval = bootstrap_auc_interval(
        y, chosen_probability, settings.n_bootstrap, settings.interval_level, bootstrap_rng
    )
    table = reliability_table(y, chosen_probability, settings.n_calibration_bins)
    model = _build_completion_estimator(chosen_kind, config, random_state)
    model.fit(X, y)

    report = {
        "fitted_hyperparameters": _describe_fit(model, chosen_kind),
        "kind": chosen_kind,
        "primary": settings.primary,
        "fallback": settings.fallback,
        "fallback_taken": chosen_kind != settings.primary,
        "attempts": attempts,
        "n_samples": int(y.size),
        "n_complete": int(y.sum()),
        "n_incomplete": int(y.size - y.sum()),
        "features": list(FEATURE_ORDER),
        "random_state": random_state,
        "n_folds": int(settings.n_folds),
        "cv_roc_auc": float(roc_auc_score(y, chosen_probability)),
        "cv_roc_auc_interval": interval,
        "cv_brier_score": float(brier_score_loss(y, chosen_probability)),
        # The reference Brier score of a model that always predicts the base rate. A model
        # that does not beat it has learned nothing, which is a result, not a failure to hide.
        "baseline_brier_score": float(brier_score_loss(y, np.full(y.shape, y.mean()))),
        "calibration_table": table,
        "expected_calibration_error": expected_calibration_error(table, int(y.size)),
    }
    return model, report


def _design_bounds(X: np.ndarray, expansion: float) -> dict[str, list[float]]:
    """Per feature min and max of the executed design, expanded by a fraction of the range."""
    low = X.min(axis=0)
    high = X.max(axis=0)
    pad = expansion * (high - low)
    return {
        name: [float(low[index] - pad[index]), float(high[index] + pad[index])]
        for index, name in enumerate(FEATURE_ORDER)
    }


def build_validity_domain(
    model: Pipeline, validity: pd.DataFrame, config: Config, model_sha256: str
) -> dict[str, Any]:
    """The validity domain artifact of build spec 9.4.

    The domain is the intersection of two conditions: the completion model predicts at
    least the configured probability, and the query sits inside the executed design's box,
    which is the design density condition. The second matters as much as the first. A
    classifier extrapolating far outside the design will happily report a high completion
    probability in a region where no simulation was ever attempted, and a probability with
    no data under it is exactly the manufactured confidence binding law 1 forbids.

    The returned record holds the model reference and hash, the threshold, the bounds, and
    the volume fraction measured on a dense grid, so :mod:`ufem.validity` can enforce the
    same contract without refitting anything.
    """
    settings = config.pipeline.audit.validity_domain
    X = features(validity)
    bounds = _design_bounds(X, settings.hull_expansion)

    axes = [
        np.linspace(bounds[name][0], bounds[name][1], settings.grid_resolution)
        for name in FEATURE_ORDER
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    dense = np.column_stack([axis.ravel() for axis in mesh])
    dense_probability = model.predict_proba(dense)[:, 1]
    inside_dense = dense_probability >= settings.completion_threshold

    design_probability = model.predict_proba(X)[:, 1]
    inside_design = design_probability >= settings.completion_threshold
    valid_flag = (validity["status"] == STATUS_VALID).to_numpy(dtype=bool)

    return {
        "definition": (
            "a point is inside the validity domain when the completion model predicts "
            f"P(complete) >= {settings.completion_threshold} and the point lies inside the "
            "box of the executed design, which is the design density condition of build "
            "spec 9.4"
        ),
        "completion_threshold": float(settings.completion_threshold),
        "hull_expansion": float(settings.hull_expansion),
        "feature_order": list(FEATURE_ORDER),
        "design_bounds": bounds,
        "model_file": COMPLETION_PICKLE,
        "model_sha256": model_sha256,
        "grid_resolution": int(settings.grid_resolution),
        "dense_grid_points": int(dense.shape[0]),
        "dense_inside_fraction": float(inside_dense.mean()),
        "design_inside_count": int(inside_design.sum()),
        "design_inside_fraction": float(inside_design.mean()),
        "valid_jobs_inside_count": int((inside_design & valid_flag).sum()),
        "valid_jobs_inside_fraction": float(inside_design[valid_flag].mean()),
        "failed_jobs_inside_fraction": float(inside_design[~valid_flag].mean()),
    }


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Weighted mean, standard deviation and CoV of one QoI.

    The standard deviation uses the reliability weight correction
    ``sum(w) / (sum(w)^2 - sum(w^2))``, which reduces to the familiar ``1/(n-1)`` when every
    weight is one. That is the right correction here because the weights are frequency like
    (each valid job stands in for ``1/P(complete)`` designed samples), not precision like.
    """
    array = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if array.shape != w.shape:
        raise ValueError(
            f"weighted_moments needs matching shapes, got values {array.shape} and "
            f"weights {w.shape}."
        )
    if array.size < 2:
        raise ValueError("weighted_moments needs at least two values for a spread.")
    if np.any(w <= 0.0):
        raise ValueError("weighted_moments needs strictly positive weights.")
    total = float(w.sum())
    mean = float(np.dot(w, array) / total)
    denominator = total**2 - float(np.dot(w, w))
    if denominator <= 0.0:
        raise ValueError(
            "the weights are too concentrated to form an unbiased weighted variance: one "
            "sample carries effectively all the weight."
        )
    variance = total * float(np.dot(w, (array - mean) ** 2)) / denominator
    std = float(np.sqrt(variance))
    if mean == 0.0:
        raise ValueError("weighted_moments cannot form a CoV around a zero mean.")
    return {"mean": mean, "std": std, "cov": std / mean}


def importance_weighting_study(
    qoi: pd.DataFrame, model: Pipeline, config: Config
) -> dict[str, Any]:
    """Inverse probability of completion weighting on the valid jobs (build spec 9.4).

    Each surviving job is reweighted by ``1 / P(complete | its inputs)``, so a job from a
    corner where most runs failed stands in for the designed samples that did not survive.
    If the headline statistics move materially under that reweighting, the censoring is
    biasing the population estimates and that is a finding; if they barely move, that is
    equally a finding and it is the one that licenses the unweighted numbers.
    """
    settings = config.pipeline.audit.importance_weighting
    X = features(qoi)
    probability = model.predict_proba(X)[:, 1]
    clipped = np.clip(probability, settings.min_probability, 1.0)
    n_clipped = int((probability < settings.min_probability).sum())
    weights = 1.0 / clipped
    normalized = weights / weights.mean()

    per_qoi = {}
    for name, unit in WEIGHTED_QOI_COLUMNS.items():
        if name not in qoi.columns:
            raise KeyError(
                f"the QoI table has no column {name!r}; the importance weighting study "
                f"expects {sorted(WEIGHTED_QOI_COLUMNS)} and the frame offers "
                f"{list(qoi.columns)}."
            )
        values = qoi[name].to_numpy(dtype=float)
        unweighted = weighted_moments(values, np.ones_like(values))
        weighted = weighted_moments(values, normalized)
        per_qoi[name] = {
            "unit": unit,
            "unweighted": unweighted,
            "weighted": weighted,
            "mean_shift": weighted["mean"] - unweighted["mean"],
            "mean_shift_relative": (weighted["mean"] - unweighted["mean"]) / unweighted["mean"],
            "cov_shift": weighted["cov"] - unweighted["cov"],
        }
    largest = max(per_qoi.items(), key=lambda item: abs(item[1]["mean_shift_relative"]))
    return {
        "n_weighted_samples": int(len(qoi)),
        "min_probability_clip": float(settings.min_probability),
        "n_probabilities_clipped": n_clipped,
        "completion_probability": {
            "min": float(probability.min()),
            "median": float(np.median(probability)),
            "max": float(probability.max()),
        },
        "weight": {
            "min": float(normalized.min()),
            "median": float(np.median(normalized)),
            "max": float(normalized.max()),
            # Kish's effective sample size: how many equally weighted observations the
            # weighted set is worth. A large gap from the nominal count means the reweighting
            # is leaning hard on a few survivors.
            "effective_sample_size": float(normalized.sum() ** 2 / np.dot(normalized, normalized)),
        },
        "by_qoi": per_qoi,
        "largest_relative_mean_shift": {
            "qoi": largest[0],
            "value": float(largest[1]["mean_shift_relative"]),
        },
    }


def _load_reference(repo_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Read the committed audit reference the stage is gated against."""
    directory = Path(repo_root) / AUDIT_REFERENCE_DIR
    csv_path = directory / REFERENCE_VALIDITY_CSV
    json_path = directory / REFERENCE_SUMMARY_JSON
    for path, role in ((csv_path, "validity classification"), (json_path, "audit summary")):
        if not path.is_file():
            raise FileNotFoundError(
                f"the audit stage needs the committed {role} at {path}, which does not "
                "exist. It is the 2026-08-28 reference the reclassification is gated "
                "against and is committed to the repository."
            )
    reference = pd.read_csv(csv_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    hashes = {
        REFERENCE_VALIDITY_CSV: sha256_file(csv_path),
        REFERENCE_SUMMARY_JSON: sha256_file(json_path),
    }
    return reference, summary, hashes


def _load_ingest(root: Path, config: Config, config_sha256: str) -> tuple[Path, dict[str, str]]:
    """Locate the ingest artifacts this stage depends on, or raise naming the fix."""
    directory = stage_dir(root / config.pipeline.paths.artifact_root, INGEST_STAGE, config_sha256)
    hashes = {}
    for name in (LOAD_PARQUET, DESIGN_PARQUET):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the audit stage needs {path}, which does not exist. Run `ufem run ingest` "
                "first: audit reclassifies the design from the ingest artifacts for this "
                "config hash."
            )
        hashes[name] = sha256_file(path)
    return directory, hashes


def _grid_qoi_path(root: Path, config: Config, config_sha256: str) -> Path:
    """The QoI table the importance weighting study reweights, or a raise naming the fix."""
    path = stage_dir(
        root / config.pipeline.paths.artifact_root, GRID_STAGE, config_sha256
    ) / GRID_QOI_PARQUET
    if not path.is_file():
        raise FileNotFoundError(
            f"the importance weighting study needs {path}, which does not exist. "
            "Run `ufem run grid` before `ufem run audit`, or run `ufem run all`: the study "
            "reweights the QoI table the grid stage extracts from the surviving runs."
        )
    return path


def declared_input_hashes(
    repo_root: Path | str, config: Config, config_sha256: str
) -> dict[str, str]:
    """Hash this stage's declared inputs as they are on disk right now (see ``ufem.runner``).

    Audit reads three things: the ingest artifacts, the committed 2026-08-28 audit reference
    it is gated against, and the grid stage's QoI table. All three are declared here in the
    same order ``run`` records them, so the two can never disagree.
    """
    root = Path(repo_root)
    _ingest_dir, hashes = _load_ingest(root, config, config_sha256)
    _reference, _summary, reference_hashes = _load_reference(root)
    return {
        **hashes,
        **reference_hashes,
        GRID_QOI_PARQUET: sha256_file(_grid_qoi_path(root, config, config_sha256)),
    }


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write one JSON artifact with sorted keys, so a rerun is byte comparable."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the audit stage and return its artifact directory."""
    started = _time.perf_counter()
    root = Path(repo_root)
    ingest_dir, input_hashes = _load_ingest(root, config, config_sha256)
    reference, reference_summary, reference_hashes = _load_reference(root)
    input_hashes = {**input_hashes, **reference_hashes}

    load = pd.read_parquet(ingest_dir / LOAD_PARQUET)
    design = pd.read_parquet(ingest_dir / DESIGN_PARQUET)

    validity = classify_samples(design, load, config)
    counts = status_counts(validity)
    if counts != EXPECTED_STATUS_COUNTS:
        raise AssertionError(
            f"the reclassification measured {counts}, but build spec 5.5 and 6.1 pin "
            f"{EXPECTED_STATUS_COUNTS}. Per build spec section 24, stop and record the "
            "measured split in docs/DESIGN_DECISIONS.md before relaxing this assertion."
        )
    comparison = compare_with_reference(validity, reference)

    censoring = censoring_statistics(validity, config)
    rate_gate = check_quartile_rates_against_reference(censoring, reference_summary)

    model, completion = fit_completion_model(
        validity, config, config.pipeline.seed_entropy
    )

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / COMPLETION_PICKLE
    model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    model_sha256 = sha256_file(model_path)
    completion["model_file"] = COMPLETION_PICKLE
    completion["model_sha256"] = model_sha256

    domain = build_validity_domain(model, validity, config, model_sha256)

    grid_qoi_path = _grid_qoi_path(root, config, config_sha256)
    input_hashes[GRID_QOI_PARQUET] = sha256_file(grid_qoi_path)
    qoi = pd.read_parquet(grid_qoi_path)
    weighting = importance_weighting_study(qoi, model, config)

    validity_path = directory / VALIDITY_PARQUET
    validity.to_parquet(validity_path, engine="pyarrow", compression="zstd", index=False)
    outputs = [
        validity_path,
        model_path,
        _write_json({**censoring, "reference_gate": rate_gate}, directory / CENSORING_JSON),
        _write_json(completion, directory / COMPLETION_JSON),
        _write_json(domain, directory / VALIDITY_DOMAIN_JSON),
        _write_json(weighting, directory / WEIGHTING_JSON),
    ]

    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "status_counts": counts,
        "reference_comparison": comparison,
        "quartile_rate_gate": rate_gate,
        "completion_model_kind": completion["kind"],
        "completion_fallback_taken": completion["fallback_taken"],
        "cv_roc_auc": completion["cv_roc_auc"],
        "cv_roc_auc_interval": [
            completion["cv_roc_auc_interval"]["auc_low"],
            completion["cv_roc_auc_interval"]["auc_high"],
        ],
        "cv_brier_score": completion["cv_brier_score"],
        "expected_calibration_error": completion["expected_calibration_error"],
        "validity_domain_design_inside_fraction": domain["design_inside_fraction"],
        "largest_relative_mean_shift": weighting["largest_relative_mean_shift"],
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=outputs,
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    interval = completion["cv_roc_auc_interval"]
    print(
        f"[audit] {counts[STATUS_VALID]} valid, {counts[STATUS_MISSING]} missing, "
        f"{counts[STATUS_PARTIAL]} partial over {len(validity)} design rows; exact match "
        f"against data/audit_reference/{REFERENCE_VALIDITY_CSV} on "
        f"{comparison['n_compared']} rows, {rate_gate['n_rates_compared']} quartile rates. "
        f"Completion model {completion['kind']}: AUC {completion['cv_roc_auc']:.4f} "
        f"[{interval['auc_low']:.4f}, {interval['auc_high']:.4f}] at "
        f"{interval['level']:.0%}, Brier {completion['cv_brier_score']:.4f}, ECE "
        f"{completion['expected_calibration_error']:.4f}; validity domain holds "
        f"{domain['design_inside_fraction']:.1%} of the design; largest weighted mean shift "
        f"{weighting['largest_relative_mean_shift']['value']:+.2%} on "
        f"{weighting['largest_relative_mean_shift']['qoi']}"
    )
    return directory
