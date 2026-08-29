"""Round trip and tamper detection over the content addressed artifact store."""

from __future__ import annotations

import json

import pytest

from ufem.manifest import (
    MANIFEST_NAME,
    cache_key,
    load_manifest,
    package_versions,
    sha256_file,
    stage_dir,
    verify_manifest,
    write_manifest,
)

CONFIG_A = "a" * 64
CONFIG_B = "b" * 64


@pytest.fixture
def written_stage(tmp_path):
    """A stage directory with two outputs and a manifest already written."""
    directory = tmp_path / "grid" / CONFIG_A
    directory.mkdir(parents=True)
    first = directory / "curves.csv"
    first.write_text("job,u_mm,rf2_N\n1,0.0,0.0\n", encoding="utf-8")
    second = directory / "summary.json"
    second.write_text(json.dumps({"n_curves": 198}), encoding="utf-8")
    write_manifest(
        stage_dir=directory,
        stage_name="grid",
        config_hash=CONFIG_A,
        input_hashes={"raw_curves": "c" * 64},
        outputs=[first, second],
        seed_entropy=866105494284971936421390281,
        extra={"n_curves": 198, "wall_time_s": 1.5},
    )
    return directory


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    # SHA-256 of the empty byte string.
    assert sha256_file(path) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not an existing file"):
        sha256_file(tmp_path / "absent.csv")


def test_stage_dir_layout(tmp_path):
    assert stage_dir(tmp_path, "reduce", CONFIG_A) == tmp_path / "reduce" / CONFIG_A


def test_round_trip_write_load_verify(written_stage):
    assert (written_stage / MANIFEST_NAME).is_file()
    manifest = load_manifest(written_stage)
    assert manifest["stage"] == "grid"
    assert manifest["config_sha256"] == CONFIG_A
    assert manifest["inputs"] == {"raw_curves": "c" * 64}
    assert {record["name"] for record in manifest["outputs"]} == {"curves.csv", "summary.json"}
    assert manifest["seed_entropy"] == "866105494284971936421390281"
    assert manifest["packages"]["python"] == package_versions()["python"]
    assert manifest["wall_time_s"] == pytest.approx(1.5)
    assert manifest["extra"]["n_curves"] == 198
    assert "commit" in manifest["git"]
    assert verify_manifest(written_stage) is True


def test_verify_detects_a_tampered_output(written_stage):
    target = written_stage / "curves.csv"
    target.write_text("job,u_mm,rf2_N\n1,0.0,999999.0\n", encoding="utf-8")
    assert verify_manifest(written_stage) is False


def test_verify_detects_a_deleted_output(written_stage):
    (written_stage / "summary.json").unlink()
    assert verify_manifest(written_stage) is False


def test_load_manifest_raises_with_a_named_diagnostic(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no manifest at"):
        load_manifest(empty)


def test_write_manifest_raises_on_a_declared_but_missing_output(tmp_path):
    directory = tmp_path / "surrogate" / CONFIG_A
    directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="declared output"):
        write_manifest(
            stage_dir=directory,
            stage_name="surrogate",
            config_hash=CONFIG_A,
            input_hashes={},
            outputs=[directory / "never_written.pkl"],
            seed_entropy=1,
            extra={},
        )


def test_write_manifest_raises_when_the_stage_dir_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage directory does not exist"):
        write_manifest(
            stage_dir=tmp_path / "missing",
            stage_name="grid",
            config_hash=CONFIG_A,
            input_hashes={},
            outputs=[],
            seed_entropy=1,
            extra={},
        )


def test_cache_key_changes_when_the_config_hash_changes(tmp_path):
    code = tmp_path / "grid.py"
    code.write_text("def run():\n    return 1\n", encoding="utf-8")
    inputs = {"raw": "d" * 64}
    assert cache_key("grid", code, CONFIG_A, inputs) != cache_key("grid", code, CONFIG_B, inputs)


def test_cache_key_changes_when_the_code_changes(tmp_path):
    code = tmp_path / "grid.py"
    code.write_text("def run():\n    return 1\n", encoding="utf-8")
    before = cache_key("grid", code, CONFIG_A, {})
    code.write_text("def run():\n    return 2\n", encoding="utf-8")
    assert cache_key("grid", code, CONFIG_A, {}) != before


def test_cache_key_changes_when_an_input_hash_changes(tmp_path):
    code = tmp_path / "grid.py"
    code.write_text("x = 1\n", encoding="utf-8")
    a = cache_key("grid", code, CONFIG_A, {"raw": "d" * 64})
    b = cache_key("grid", code, CONFIG_A, {"raw": "e" * 64})
    assert a != b


def test_cache_key_is_stable_and_order_independent(tmp_path):
    code = tmp_path / "grid.py"
    code.write_text("x = 1\n", encoding="utf-8")
    forward = cache_key("grid", code, CONFIG_A, {"a": "1" * 64, "b": "2" * 64})
    reversed_order = cache_key("grid", code, CONFIG_A, {"b": "2" * 64, "a": "1" * 64})
    assert forward == reversed_order


def test_cache_key_distinguishes_stages(tmp_path):
    code = tmp_path / "shared.py"
    code.write_text("x = 1\n", encoding="utf-8")
    assert cache_key("grid", code, CONFIG_A, {}) != cache_key("reduce", code, CONFIG_A, {})
