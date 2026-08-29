"""The pipeline driver: a pure batch CLI.

``ufem run <stage>|all`` and ``ufem doctor``. Ground rule 8 and build spec section 5.8: the
v1 driver blocked on ``input()`` whenever anything failed, so a broken run waited forever
for a human instead of returning nonzero. There is no ``input()`` anywhere in this file and
there never will be. Every failure path exits nonzero with a named diagnostic.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any, Callable

from ufem import __version__
from ufem.config import Config, config_hash, load_config
from ufem.manifest import (
    cache_key,
    load_manifest,
    package_versions,
    stage_dir,
    verify_manifest,
)

#: Stage name to (module under ufem, phase that implements it). Order is execution order.
#:
#: ``audit`` sits after ``grid`` rather than between ``ingest`` and ``grid``, where the P0
#: skeleton first placed it. Its validity reclassification and censoring statistics need
#: only the ingest artifacts, but the importance weighting study of build spec 9.4 reweights
#: the QoI table that ``grid`` extracts, so running audit first would mean either splitting
#: the stage in two or having it recompute the QoI schedule itself. The reordering is
#: recorded in docs/DESIGN_DECISIONS.md.
STAGES: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    [
        ("ingest", ("ufem.ingest", "P1")),
        ("grid", ("ufem.grid", "P1")),
        ("audit", ("ufem.audit", "P2")),
        ("register", ("ufem.register", "P3")),
        ("reduce", ("ufem.reduce", "P3")),
        ("surrogate", ("ufem.surrogate", "P4")),
        ("calibrate", ("ufem.calibrate", "P5")),
        ("validate", ("ufem.validate", "P4")),
        ("sensitivity", ("ufem.sensitivity", "P6")),
        ("propagate", ("ufem.propagate", "P7")),
        ("report", ("ufem.report", "P9")),
    ]
)

DESIGN_DECISIONS = "docs/DESIGN_DECISIONS.md"
BLOCK_BEGIN = "<!-- BEGIN RESOLVED VERSIONS -->"
BLOCK_END = "<!-- END RESOLVED VERSIONS -->"


def repo_root_from_here() -> Path:
    """The repository root, found from this file: ``src/ufem/runner.py`` is three deep."""
    return Path(__file__).resolve().parents[2]


def load_stage(stage_name: str) -> Callable[..., Any]:
    """Import a stage's ``run`` callable lazily, so an unbuilt stage costs nothing."""
    module_name, phase = STAGES[stage_name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name is not None and not module_name.startswith(err.name):
            raise
        raise NotImplementedError(
            f"stage {stage_name!r} is not implemented yet: {module_name} arrives in phase "
            f"{phase}. See docs/BUILD_SPEC.md section 22."
        ) from err
    run = getattr(module, "run", None)
    if run is None:
        raise NotImplementedError(
            f"stage {stage_name!r} has module {module_name} but no run() entry point; "
            f"phase {phase} is responsible for adding it."
        )
    return run


def stage_code_file(stage_name: str) -> Path:
    """Path to a stage's implementation file, hashed into its cache key."""
    module_name, _ = STAGES[stage_name]
    return Path(__file__).with_name(module_name.split(".")[-1] + ".py")


def is_cache_hit(root: Path, config: Config, stage_name: str, digest: str) -> bool:
    """True when the stage directory holds a valid manifest with a matching cache key."""
    artifact_root = root / config.pipeline.paths.artifact_root
    directory = stage_dir(artifact_root, stage_name, digest)
    if not (directory / "manifest.json").is_file():
        return False
    if not verify_manifest(directory):
        return False
    recorded = load_manifest(directory).get("extra", {}).get("cache_key")
    if recorded is None:
        return False
    code_file = stage_code_file(stage_name)
    if not code_file.is_file():
        return False
    expected = cache_key(stage_name, code_file, digest, load_manifest(directory)["inputs"])
    return bool(recorded == expected)


def run_stage(root: Path, config: Config, stage_name: str, force: bool) -> int:
    """Run one stage, honoring the cache. Returns a process exit code."""
    digest = config_hash(config)
    if not force and is_cache_hit(root, config, stage_name, digest):
        print(f"[cache hit] {stage_name}")
        return 0
    started = time.perf_counter()
    try:
        run = load_stage(stage_name)
        run(repo_root=root, config=config, config_sha256=digest)
    except NotImplementedError as err:
        print(f"[not implemented] {stage_name}: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[failed] {stage_name}: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    print(f"[done] {stage_name} in {time.perf_counter() - started:.2f} s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_here()
    config = load_config(root)
    if args.stage == "all":
        selected = list(STAGES)
    elif args.stage in STAGES:
        selected = [args.stage]
    else:
        print(
            f"unknown stage {args.stage!r}. Known stages, in order: "
            f"{', '.join(STAGES)}, or 'all'.",
            file=sys.stderr,
        )
        return 2
    for stage_name in selected:
        code = run_stage(root, config, stage_name, args.force)
        if code != 0:
            return code
    return 0


def torch_report() -> str:
    """Torch build and device availability, or a stated reason it is unknown."""
    try:
        import torch
    except ImportError:
        return "torch not importable"
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {device}"
    return f"torch {torch.__version__}, CUDA build {torch.version.cuda}, no visible GPU (CPU path)"


def version_block(root: Path, config: Config) -> str:
    """The dated resolved version matrix that doctor writes into DESIGN_DECISIONS."""
    versions = package_versions()
    lines = [
        BLOCK_BEGIN,
        "",
        f"### Resolved version matrix, {date.today().isoformat()}",
        "",
        f"Written by `ufem doctor` on {platform.platform()}. Regenerate it, do not edit it.",
        "",
        "| Component | Resolved |",
        "|---|---|",
        f"| ufem | {__version__} |",
    ]
    lines += [f"| {name} | {value} |" for name, value in versions.items()]
    lines += [
        f"| torch device | {torch_report()} |",
        f"| config SHA-256 | `{config_hash(config)}` |",
        "",
        BLOCK_END,
    ]
    return "\n".join(lines)


def write_version_block(root: Path, config: Config) -> Path:
    """Append or replace the resolved version block in docs/DESIGN_DECISIONS.md."""
    path = root / DESIGN_DECISIONS
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Doctor records the resolved matrix there, so the "
            "document must be created before doctor can update it."
        )
    text = path.read_text(encoding="utf-8")
    block = version_block(root, config)
    if BLOCK_BEGIN in text and BLOCK_END in text:
        head, rest = text.split(BLOCK_BEGIN, 1)
        _, tail = rest.split(BLOCK_END, 1)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_here()
    config = load_config(root)
    print(f"ufem {__version__}")
    print(f"repo root      {root}")
    print(f"python         {platform.python_version()} ({platform.platform()})")
    for name, value in package_versions().items():
        if name != "python":
            print(f"{name:<14} {value}")
    print(f"torch device   {torch_report()}")
    print(f"config sha256  {config_hash(config)}")
    print(f"  probabilistic_model.yaml + pipeline.yaml, {len(STAGES)} stages registered")
    written = write_version_block(root, config)
    print(f"wrote resolved version matrix to {written.relative_to(root)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ufem", description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=None, help="repository root (default: inferred)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one stage or the whole pipeline")
    run_parser.add_argument("stage", help=f"one of: {', '.join(STAGES)}, or 'all'")
    run_parser.add_argument("--force", action="store_true", help="rerun even on a cache hit")
    run_parser.set_defaults(func=cmd_run)

    doctor_parser = sub.add_parser("doctor", help="print and record the resolved environment")
    doctor_parser.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as err:
        print(f"ufem: {type(err).__name__}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
