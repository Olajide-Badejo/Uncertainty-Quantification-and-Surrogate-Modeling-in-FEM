"""Content addressed artifact store: about 100 lines, no external service.

Binding law 5: no number appears in the README, the report, or the UI unless it is
reproducible from a committed manifest whose hashes resolve to real files and a real
commit. Every stage is a pure function from (config hash, input hashes) to a directory
holding its outputs and one ``manifest.json``.

Ground rule 8: nothing here falls back silently. A missing directory, a missing manifest,
or an output whose hash no longer matches raises with a named diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"

#: Recorded in every manifest so a result can be tied to the stack that produced it.
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "matplotlib",
    "torch",
    "gpytorch",
    "openturns",
    "SALib",
    "fdasrsf",
)


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes, streamed so large artifacts do not load into memory."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"cannot hash {target}: not an existing file.")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Resolved versions of the core stack, plus the interpreter."""
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_state(repo_root: Path | str) -> dict[str, Any]:
    """Current commit and dirty flag, or a stated reason the state is unavailable."""
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    if commit is None:
        return {"commit": "unavailable", "dirty": None, "branch": "unavailable"}
    return {
        "commit": commit,
        "dirty": bool(status),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unavailable",
    }


def stage_dir(artifact_root: Path | str, stage_name: str, config_hash: str) -> Path:
    """Where one stage's artifacts live: ``<artifact_root>/<stage>/<config hash>``."""
    return Path(artifact_root) / stage_name / config_hash


def cache_key(
    stage_name: str, code_file: Path | str, config_hash: str, input_hashes: dict[str, str]
) -> str:
    """SHA-256 over the stage name, its code file's hash, the config hash, and its inputs.

    A change to any of the four invalidates the stage, which is what lets the runner skip
    work without ever serving a stale artifact.
    """
    payload = json.dumps(
        {
            "stage": stage_name,
            "code_sha256": sha256_file(code_file),
            "config_sha256": config_hash,
            "inputs": dict(sorted(input_hashes.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(
    stage_dir: Path | str,
    stage_name: str,
    config_hash: str,
    input_hashes: dict[str, str],
    outputs: list[Path],
    seed_entropy: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` beside a stage's outputs and return its path."""
    target = Path(stage_dir)
    if not target.is_dir():
        raise FileNotFoundError(
            f"cannot write a manifest into {target}: the stage directory does not exist. "
            f"Stage {stage_name} must create its output directory before recording it."
        )
    output_records = []
    for item in outputs:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(
                f"stage {stage_name} declared output {path}, which does not exist. An "
                "output that was not written is a stage failure, not a manifest warning."
            )
        output_records.append(
            {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "stage": stage_name,
        "config_sha256": config_hash,
        "inputs": dict(sorted(input_hashes.items())),
        "outputs": output_records,
        "seed_entropy": str(seed_entropy),
        "packages": package_versions(),
        "git": git_state(target),
        "hostname": socket.gethostname(),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": float((extra or {}).pop("wall_time_s", 0.0)),
        "extra": extra or {},
    }
    path = target / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(stage_dir: Path | str) -> dict[str, Any]:
    """Read one stage's manifest, raising if it is absent or unparseable."""
    path = Path(stage_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest at {path}. The stage has not run, or its artifact directory was "
            "deleted; rerun the stage rather than treating the absence as a cache miss."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"manifest at {path} is not valid JSON: {err}") from err


def verify_manifest(stage_dir: Path | str) -> bool:
    """Recheck every recorded output hash against what is on disk.

    Returns True only if every declared output still exists with its recorded digest.
    """
    target = Path(stage_dir)
    manifest = load_manifest(target)
    for record in manifest.get("outputs", []):
        path = target / record["name"]
        if not path.is_file():
            return False
        if sha256_file(path) != record["sha256"]:
            return False
    return True
