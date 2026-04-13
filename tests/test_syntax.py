"""
tests/test_syntax.py
Lightweight syntax and import smoke tests for CI.
"""

import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "env", "ENV", "__pycache__", "fixes"}


def collect_py_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def compile_check_all() -> tuple[list[Path], list[tuple[Path, str]]]:
    ok: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for pyfile in collect_py_files(REPO_ROOT):
        source = pyfile.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(source, filename=str(pyfile))
            ok.append(pyfile)
        except SyntaxError as exc:
            failed.append((pyfile, str(exc)))

    return ok, failed


if __name__ == "__main__":
    ok, failed = compile_check_all()
    for path, err in failed:
        print(f"SYNTAX ERROR  {path.relative_to(REPO_ROOT)}: {err}", file=sys.stderr)
    for path in ok:
        print(f"OK            {path.relative_to(REPO_ROOT)}")
    print(f"\n{len(ok)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


def test_all_files_parse_without_syntax_errors() -> None:
    _, failed = compile_check_all()
    if failed:
        lines = [f"{p.relative_to(REPO_ROOT)}: {e}" for p, e in failed]
        raise AssertionError(
            f"{len(failed)} file(s) have syntax errors:\n" + "\n".join(lines)
        )


def test_core_imports() -> None:
    required = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "joblib",
        "tqdm",
        "openpyxl",
    ]
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        raise AssertionError(f"Missing import(s): {missing}")


def test_salib_optional_import() -> None:
    try:
        from SALib.analyze import sobol  # noqa: F401
        from SALib.sample import saltelli  # noqa: F401
    except ImportError as exc:
        raise AssertionError(f"SALib import failed: {exc}") from exc


def test_torch_importable() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise AssertionError(f"torch import failed: {exc}") from exc


def test_no_windows_only_paths() -> None:
    bad: list[str] = []
    patterns = ["Path(r\".\\", "Path(r'.\\"]

    for pyfile in collect_py_files(REPO_ROOT):
        if "tests" in pyfile.parts:
            continue
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        if any(pattern in text for pattern in patterns):
            bad.append(str(pyfile.relative_to(REPO_ROOT)))

    if bad:
        raise AssertionError(
            "Windows-only path literals detected (use pathlib with '/'):\n" + "\n".join(bad)
        )
