"""The committed copies under ``data/processed/`` must be the artifacts they claim to be.

Binding law 5: a committed number is only evidence if it resolves to a manifest. These files
are byte copies of stage outputs, so the check is a digest comparison against the manifest
that recorded them, plus the 5 MB rule of build spec 3.3.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from export_processed import EXPORTS, LIMIT_BYTES, PROCESSED_DIR
from ufem.config import config_hash, load_config
from ufem.grid import QOI_COLUMNS
from ufem.manifest import sha256_file, stage_dir


@pytest.fixture(scope="module")
def processed_dir(repo_root):
    directory = repo_root / PROCESSED_DIR
    if not directory.is_dir():
        pytest.skip(
            f"{directory} does not exist; run scripts/export_processed.py after the stages."
        )
    return directory


def _exported_names() -> list[tuple[str, str]]:
    return [(stage, name) for stage, names in EXPORTS.items() for name in names]


@pytest.mark.parametrize(("stage_name", "file_name"), _exported_names())
def test_every_committed_file_exists_and_is_under_the_size_limit(
    processed_dir, stage_name, file_name
):
    path = processed_dir / file_name
    assert path.is_file(), f"{path} is missing; scripts/export_processed.py writes it."
    assert path.stat().st_size < LIMIT_BYTES


@pytest.mark.parametrize(("stage_name", "file_name"), _exported_names())
def test_every_committed_file_matches_its_recorded_manifest_digest(
    repo_root, processed_dir, stage_name, file_name
):
    """A committed copy that has drifted from its artifact is a traceability break."""
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, stage_name, config_hash(config)
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"{manifest_path} does not exist; run `ufem run {stage_name}` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {record["name"]: record["sha256"] for record in manifest["outputs"]}
    assert file_name in recorded, f"{file_name} is not an output of stage {stage_name}."
    assert sha256_file(processed_dir / file_name) == recorded[file_name]


def test_the_committed_qoi_table_carries_the_scheduled_columns(processed_dir):
    frame = pd.read_parquet(processed_dir / "qoi.parquet")
    assert tuple(frame.columns) == QOI_COLUMNS
    assert len(frame) == 198


def test_the_committed_grids_are_198_curves_on_201_points(processed_dir):
    for name in ("rf2_grid.parquet", "damage_grid.parquet"):
        frame = pd.read_parquet(processed_dir / name)
        assert frame.shape == (198, 202), name
        assert frame.drop(columns=["job"]).notna().to_numpy().all(), name


def test_the_committed_design_is_the_400_row_campaign(processed_dir):
    frame = pd.read_parquet(processed_dir / "design.parquet")
    assert len(frame) == 400
    assert frame["sample_id"].is_unique


def test_the_readme_exists_and_names_the_regeneration_command(processed_dir):
    text = (processed_dir / "README.md").read_text(encoding="utf-8")
    assert "ufem run ingest" in text
    assert "ufem run grid" in text
    assert "SHA-256" in text
