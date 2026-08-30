"""The cache decision, tested against a real stage and a real upstream artifact.

Ground rule 7 and build spec 17.1. The cache key of `ufem.manifest.cache_key` is honest by
construction; what these tests pin is the runner's use of it, which is where the interesting
mistake lives. A cache check that recomputes the key from the hashes the stage recorded last
time compares a number against itself and always agrees, so a changed upstream artifact would
be served stale until someone happened to pass `--force`.

The stage under test is `grid`, chosen because its declared inputs are exactly the three
ingest Parquet files and nothing else. The files here hold arbitrary bytes: the cache decision
hashes them, it does not read them, and using real Parquet would test pyarrow rather than the
runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ufem.config import config_hash, load_config
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import DAMAGE_PARQUET, DESIGN_PARQUET, LOAD_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.runner import is_cache_hit, stage_code_file

INGEST_OUTPUTS = (LOAD_PARQUET, DAMAGE_PARQUET, DESIGN_PARQUET)


@pytest.fixture
def cached_pipeline(tmp_path, repo_root):
    """An ingest directory and a grid directory whose manifest is a valid cache hit.

    Returns the temporary root, the loaded config, its digest, and the two directories, all
    laid out exactly as a real run would leave them.
    """
    config = load_config(repo_root)
    digest = config_hash(config)
    artifact_root = tmp_path / config.pipeline.paths.artifact_root

    ingest_dir = stage_dir(artifact_root, INGEST_STAGE, digest)
    ingest_dir.mkdir(parents=True)
    for index, name in enumerate(INGEST_OUTPUTS):
        (ingest_dir / name).write_bytes(b"upstream artifact %d\n" % index)
    write_manifest(
        stage_dir=ingest_dir,
        stage_name=INGEST_STAGE,
        config_hash=digest,
        input_hashes={"design_csv": "0" * 64},
        outputs=[ingest_dir / name for name in INGEST_OUTPUTS],
        seed_entropy=config.pipeline.seed_entropy,
        extra={"cache_key": "not read by the downstream stage"},
    )

    grid_dir = stage_dir(artifact_root, GRID_STAGE, digest)
    grid_dir.mkdir(parents=True)
    (grid_dir / "rf2_grid.parquet").write_bytes(b"downstream artifact\n")
    inputs = {name: sha256_file(ingest_dir / name) for name in INGEST_OUTPUTS}
    write_manifest(
        stage_dir=grid_dir,
        stage_name=GRID_STAGE,
        config_hash=digest,
        input_hashes=inputs,
        outputs=[grid_dir / "rf2_grid.parquet"],
        seed_entropy=config.pipeline.seed_entropy,
        extra={
            "cache_key": cache_key(
                GRID_STAGE, stage_code_file(GRID_STAGE), digest, inputs
            )
        },
    )
    return tmp_path, config, digest, ingest_dir, grid_dir


def test_an_untouched_stage_is_a_cache_hit(cached_pipeline):
    root, config, digest, _ingest_dir, _grid_dir = cached_pipeline
    assert is_cache_hit(root, config, GRID_STAGE, digest) is True


def test_a_changed_upstream_artifact_is_not_a_cache_hit(cached_pipeline):
    """The regression test of the 2026-08-30 defect: rehash the inputs, do not trust them.

    The downstream manifest still records the old input hashes and still records a cache key
    that agrees with them, which is exactly why recomputing the key from the manifest's own
    record cannot detect this. The upstream file on disk has changed, so the stage must rerun.
    """
    root, config, digest, ingest_dir, _grid_dir = cached_pipeline
    target = ingest_dir / LOAD_PARQUET
    target.write_bytes(target.read_bytes() + b"one more row\n")
    assert is_cache_hit(root, config, GRID_STAGE, digest) is False


def test_a_deleted_upstream_artifact_is_not_a_cache_hit(cached_pipeline):
    root, config, digest, ingest_dir, _grid_dir = cached_pipeline
    (ingest_dir / DESIGN_PARQUET).unlink()
    assert is_cache_hit(root, config, GRID_STAGE, digest) is False


def test_a_tampered_output_is_not_a_cache_hit(cached_pipeline):
    root, config, digest, _ingest_dir, grid_dir = cached_pipeline
    (grid_dir / "rf2_grid.parquet").write_bytes(b"tampered\n")
    assert is_cache_hit(root, config, GRID_STAGE, digest) is False


def test_a_manifest_without_a_recorded_cache_key_is_not_a_cache_hit(cached_pipeline):
    root, config, digest, _ingest_dir, grid_dir = cached_pipeline
    path = grid_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["extra"].pop("cache_key")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert is_cache_hit(root, config, GRID_STAGE, digest) is False


def test_every_implemented_stage_declares_its_inputs():
    """The cache check needs each stage to say what it reads; none may be silent about it.

    A stage that did not expose the declaration would fall back to whatever its manifest
    recorded, which is the defect this module exists to pin. The runner raises instead, and
    this test is the standing check that no implemented stage is in that position.
    """
    import importlib

    from ufem.runner import STAGES

    for stage_name, (module_name, _phase) in STAGES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        declare = getattr(module, "declared_input_hashes", None)
        assert callable(declare), (
            f"stage {stage_name!r} has no declared_input_hashes(), so the runner cannot "
            "verify its inputs without trusting its own manifest."
        )


def test_stage_code_file_points_at_the_implementation():
    path = stage_code_file(GRID_STAGE)
    assert path.name == "grid.py"
    assert path.is_file()
    assert Path(path).parent.name == "ufem"
