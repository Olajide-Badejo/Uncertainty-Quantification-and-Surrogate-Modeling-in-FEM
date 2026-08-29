"""The Phase P1 gate: the pipeline must reproduce the committed 2026-08-28 audit.

Build spec section 9.3 and the P1 gate of section 22. The audit reference in
``data/audit_reference/`` was produced before this repository existed, by a standalone
script reading the same raw CSVs. If the rebuilt pipeline regenerates it to 1e-9, the
ingest and grid stages carry no silent transformation of the inherited data.

Two comparisons, each against the basis the audit actually used.

1. ``RF2_on_common_U2_grid.npy`` against the grid stage's ``rf2_grid.parquet``, row for row
   in the job order of ``common_grid_sample_ids.csv``. Same basis, so this is exact.
2. The headline statistics in ``audit_summary.json`` against the same quantities recomputed
   from the ingest stage's deduplicated curves. The audit read peak, displacement at peak
   and initial stiffness off the solver's raw adaptive increments, not off the 201 point
   grid, so this comparison uses :func:`ufem.grid.raw_curve_qoi`, which is the pipeline's
   implementation of that same measurement. The difference between the two bases is real
   and is recorded in docs/DESIGN_DECISIONS.md; it is not absorbed by loosening a tolerance.

Which fields are compared, and which are deliberately not:

- Compared from ``statistics_valid``: ``peak_RF2``, ``U2_at_peak`` and ``k0_N_per_mm``, each
  on mean, std, cov, min and max, against the raw increment basis; and
  ``damage_at_U2_10mm`` on the same five, against the grid QoI table, because 10 mm is a
  grid node so the two bases coincide there.
- Compared from ``curve_alignment.recommended_common_U2_grid.RF2_at_u_max``: mean, std, min
  and max of the residual load at 20 mm, against the grid stage's last column. This is the
  audit's own grid basis, so it is exact.
- Compared from ``inventory`` and ``data_quality_flags``: the row counts, the job count, and
  the deduplication counts the ingest stage asserts.
- Not compared: ``damage_final`` (banned as a QoI, build spec 5.6, and constant by
  construction) and ``damage_U2_at_half_max``. The audit thresholded that one on each
  curve's own maximum at the raw sample nearest above the threshold; the pipeline
  interpolates linearly between the bracketing grid points against the fixed 0.947
  saturation of build spec 9.5. Different estimator, deliberately, so there is no 1e-9
  claim to make about it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ufem.config import config_hash, load_config
from ufem.grid import (
    QOI_PARQUET,
    RF2_GRID_PARQUET,
    headline_stats,
    raw_curve_qoi,
)
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import LOAD_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import stage_dir

#: Build spec 16.1: explicit atol, because the softening tail crosses small values and a
#: relative tolerance alone is meaningless near zero.
ATOL = 1e-9

AUDIT_DIR = "data/audit_reference"
STAT_KEYS = ("mean", "std", "cov", "min", "max")


def _artifact_dir(repo_root, stage_name):
    """Locate one stage's artifact directory, skipping the test if it has not been run."""
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, stage_name, config_hash(config)
    )
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"stage {stage_name} has not been run for this config hash: {directory} holds "
            "no manifest. Run `ufem run ingest grid` before the golden gate."
        )
    return directory


@pytest.fixture(scope="module")
def audit_summary(repo_root):
    path = repo_root / AUDIT_DIR / "audit_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"the committed audit reference {path} is missing. The P1 gate compares against "
            "it, so its absence is a failure rather than a skip."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def grid_outputs(repo_root):
    directory = _artifact_dir(repo_root, GRID_STAGE)
    return {
        "rf2": pd.read_parquet(directory / RF2_GRID_PARQUET),
        "qoi": pd.read_parquet(directory / QOI_PARQUET),
    }


@pytest.fixture(scope="module")
def raw_qoi(repo_root):
    """Peak, u at peak and k0 on the raw increments, the audit's own basis."""
    directory = _artifact_dir(repo_root, INGEST_STAGE)
    return raw_curve_qoi(pd.read_parquet(directory / LOAD_PARQUET))


def _sample_ids(jobs) -> list[int]:
    return [int(str(job).rsplit("_", 1)[-1]) for job in jobs]


def test_grid_job_order_matches_the_audit_sample_ids(repo_root, grid_outputs):
    """The comparison below is row wise, so the orders must agree before the values do."""
    reference = pd.read_csv(repo_root / AUDIT_DIR / "common_grid_sample_ids.csv")
    assert _sample_ids(grid_outputs["rf2"]["job"]) == reference["sample_id"].tolist()


def test_common_displacement_grid_matches_the_audit_grid(repo_root):
    """The pipeline's linspace must be the audit's saved abscissa, node for node."""
    config = load_config(repo_root)
    settings = config.pipeline.grid
    pipeline_grid = np.linspace(settings.u_min_mm, settings.u_max_mm, settings.n_points)
    reference = np.load(repo_root / AUDIT_DIR / "common_U2_grid.npy")
    assert_allclose(pipeline_grid, reference, atol=ATOL)


def test_rf2_grid_matrix_matches_the_audit_reference(repo_root, grid_outputs):
    """The P1 gate proper: 198 by 201 reaction forces in N, to 1e-9."""
    reference = np.load(repo_root / AUDIT_DIR / "RF2_on_common_U2_grid.npy")
    produced = grid_outputs["rf2"].drop(columns=["job"]).to_numpy(dtype=float)
    assert produced.shape == reference.shape
    assert_allclose(produced, reference, atol=ATOL)


@pytest.mark.parametrize(
    ("audit_field", "qoi_column"),
    [
        ("peak_RF2", "P_max_N"),
        ("U2_at_peak", "u_peak_mm"),
        ("k0_N_per_mm", "k0_N_per_mm"),
    ],
)
def test_headline_stats_match_on_the_raw_increment_basis(
    audit_summary, raw_qoi, audit_field, qoi_column
):
    """Peak load, displacement at peak and initial stiffness, on the audit's own basis."""
    reference = audit_summary["statistics_valid"][audit_field]
    produced = headline_stats(raw_qoi[qoi_column].to_numpy(dtype=float))
    assert reference["n"] == len(raw_qoi)
    for key in STAT_KEYS:
        assert_allclose(
            produced[key], reference[key], atol=ATOL, err_msg=f"{audit_field}.{key}"
        )


def test_damage_at_10mm_matches_on_the_grid_basis(audit_summary, grid_outputs):
    """10 mm is a grid node, so the grid QoI and the audit's interpolation coincide."""
    reference = audit_summary["statistics_valid"]["damage_at_U2_10mm"]
    produced = headline_stats(grid_outputs["qoi"]["damage_at_10mm"].to_numpy(dtype=float))
    for key in STAT_KEYS:
        assert_allclose(
            produced[key], reference[key], atol=ATOL, err_msg=f"damage_at_U2_10mm.{key}"
        )


def test_residual_load_at_20mm_matches_the_audit_grid_endpoint(audit_summary, grid_outputs):
    """Residual load in N at the last grid node, the audit's own gridded quantity."""
    reference = audit_summary["curve_alignment"]["recommended_common_U2_grid"]["RF2_at_u_max"]
    produced = headline_stats(grid_outputs["qoi"]["P_residual_N"].to_numpy(dtype=float))
    for key in ("mean", "std", "min", "max"):
        assert_allclose(produced[key], reference[key], atol=ATOL, err_msg=f"RF2_at_u_max.{key}")


def test_residual_cov_is_the_audit_value(audit_summary, grid_outputs):
    """The residual load CoV of build spec 6.1, recomputed rather than quoted."""
    reference = audit_summary["curve_alignment"]["recommended_common_U2_grid"]["RF2_at_u_max"]
    produced = headline_stats(grid_outputs["qoi"]["P_residual_N"].to_numpy(dtype=float))
    assert_allclose(produced["cov"], reference["std"] / reference["mean"], atol=ATOL)


def test_inventory_and_dedup_counts_match_the_audit(repo_root, audit_summary):
    """The counts the ingest stage asserts are the counts the audit measured."""
    inventory = audit_summary["inventory"]
    flags = audit_summary["data_quality_flags"]
    manifest = json.loads(
        (_artifact_dir(repo_root, INGEST_STAGE) / "manifest.json").read_text(encoding="utf-8")
    )
    extra = manifest["extra"]
    assert extra["rows_in"]["load_displacement"] == inventory["n_rows_load_displacement"]
    assert extra["rows_in"]["damage_evolution"] == inventory["n_rows_damage"]
    assert extra["n_jobs"] == inventory["n_jobs_load_displacement"]
    assert extra["n_jobs"] == inventory["n_jobs_damage"]
    assert extra["n_design_rows"] == inventory["n_input_samples"]
    assert extra["jobs_deduplicated"]["load_displacement"] == (
        flags["n_jobs_with_duplicate_time_stamps"]
    )
    assert extra["rows_dropped"]["load_displacement"] == flags["n_duplicate_time_rows_total"]


def test_grid_curve_count_and_zero_nan(audit_summary, grid_outputs):
    """198 curves with zero NaN anywhere, the other half of the P1 gate."""
    alignment = audit_summary["curve_alignment"]["recommended_common_U2_grid"]
    rf2 = grid_outputs["rf2"].drop(columns=["job"]).to_numpy(dtype=float)
    assert list(rf2.shape) == alignment["interpolation_executed_shape"]
    assert int(np.isnan(rf2).sum()) == alignment["interpolation_n_nan"] == 0
    assert int(grid_outputs["qoi"].drop(columns=["job"]).isna().to_numpy().sum()) == 0
