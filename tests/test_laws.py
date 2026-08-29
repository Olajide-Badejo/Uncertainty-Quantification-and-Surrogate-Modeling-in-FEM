"""The binding law tests (build spec 16.2).

Each of these is a machine checkable version of a rule that v1 broke. They import the lint
logic rather than shelling out, so a failure points at a line instead of an exit code, and
they include planted violation cases so the checks are proven able to fire.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

import check_file_sizes
import dash_lint

BANNED_IN_SRC = ("AMPLIFY", "noise_level")
QUARANTINE_TOKENS = (
    "augmentation_physics_fixed",
    "data/quarantine",
    "data\\quarantine",
    "v1_legacy",
)


@pytest.fixture(scope="module")
def src_files(repo_root):
    files = sorted((repo_root / "src").rglob("*.py"))
    assert files, "no python files found under src/, the law tests would pass vacuously"
    return files


def read(path):
    return path.read_text(encoding="utf-8")


def test_dash_lint_is_clean_on_the_tree(repo_root):
    """Ground rule 3: no em dash or en dash anywhere the linter walks."""
    hits = dash_lint.check_dashes(repo_root) + dash_lint.check_src_laws(repo_root)
    assert hits == [], "dash_lint violations:\n" + "\n".join(hits)


def test_dash_lint_fires_on_a_planted_dash(tmp_path):
    """A check that cannot fail protects nothing, so plant one and watch it fire."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planted.py").write_text(
        "VALUE = 1  # a range of 1 " + chr(0x2013) + " 2\n", encoding="utf-8"
    )
    hits = dash_lint.check_dashes(tmp_path)
    assert len(hits) == 1
    assert "en dash" in hits[0]


def test_dash_lint_skips_the_build_spec(repo_root):
    """docs/BUILD_SPEC.md is the input document and is kept byte identical, so it is exempt."""
    candidates = list(dash_lint.iter_candidates(repo_root))
    assert not any(str(path).replace("\\", "/") == "docs/BUILD_SPEC.md" for path in candidates)


def test_no_banned_identifiers_in_src(src_files, repo_root):
    """Ground rule 4: a memorial to the fabricated uncertainty of build spec 5.1."""
    hits = []
    for path in src_files:
        for number, line in enumerate(read(path).splitlines(), start=1):
            for word in BANNED_IN_SRC:
                if word in line:
                    hits.append(f"{path.relative_to(repo_root)}:{number}: {word}")
    assert hits == [], "banned identifiers in src/:\n" + "\n".join(hits)


def test_no_np_random_seed_in_src(src_files, repo_root):
    """Ground rule 13: RNG comes from a SeedSequence tree, never a global seed."""
    pattern = re.compile(r"np\.random\.seed")
    hits = [
        f"{path.relative_to(repo_root)}:{number}"
        for path in src_files
        for number, line in enumerate(read(path).splitlines(), start=1)
        if pattern.search(line)
    ]
    assert hits == [], "np.random.seed in src/:\n" + "\n".join(hits)


def test_no_bare_except_in_src(src_files, repo_root):
    """Ground rule 8: v1 shipped 18 bare except blocks and validated on training data."""
    pattern = re.compile(r"except\s*:")
    hits = [
        f"{path.relative_to(repo_root)}:{number}"
        for path in src_files
        for number, line in enumerate(read(path).splitlines(), start=1)
        if pattern.search(line)
    ]
    assert hits == [], "bare except in src/:\n" + "\n".join(hits)


def test_no_distribution_construction_outside_config(src_files, repo_root):
    """Binding law 2: only config.py may name a distribution family or a QMC engine."""
    hits = []
    for path in src_files:
        if path.name == "config.py":
            continue
        for number, line in enumerate(read(path).splitlines(), start=1):
            for token in dash_lint.DISTRIBUTION_SUBSTRINGS:
                if token in line:
                    hits.append(f"{path.relative_to(repo_root)}:{number}: {token}")
    assert hits == [], "distribution construction outside config.py:\n" + "\n".join(hits)


def test_src_law_check_fires_on_a_planted_violation(tmp_path):
    """Plant one of each and assert the src sweep reports all four."""
    src = tmp_path / "src" / "ufem"
    src.mkdir(parents=True)
    (src / "planted.py").write_text(
        "import numpy as np\n"
        "AMPLIFY = 1.1\n"
        "noise_level = 0.005\n"
        "np.random.seed(42)\n"
        "from scipy.stats import lognorm\n"
        "try:\n"
        "    pass\n"
        "except:\n"
        "    pass\n",
        encoding="utf-8",
    )
    hits = dash_lint.check_src_laws(tmp_path)
    joined = "\n".join(hits)
    assert "AMPLIFY" in joined
    assert "noise_level" in joined
    assert "np.random.seed" in joined
    assert "lognorm" in joined
    assert "bare except" in joined


def test_src_never_references_quarantined_paths(src_files, repo_root):
    """Build spec 6.3: nothing in quarantine is read by any pipeline stage."""
    hits = []
    for path in src_files:
        for number, line in enumerate(read(path).splitlines(), start=1):
            stripped = line.strip()
            # Comments and docstring prose may name the quarantine; code may not open it.
            if stripped.startswith("#"):
                continue
            for token in QUARANTINE_TOKENS:
                if token in line and ("open(" in line or "read" in line or "Path(" in line):
                    hits.append(f"{path.relative_to(repo_root)}:{number}: {token}")
    assert hits == [], "src/ reaches into a quarantined path:\n" + "\n".join(hits)


def test_no_tracked_file_exceeds_five_mb(repo_root):
    """Build spec 3.3: the repository holds no file over 5 MB."""
    offenders = [
        (relative, os.path.getsize(repo_root / relative))
        for relative in check_file_sizes.tracked_files(repo_root)
        if (repo_root / relative).is_file()
        and os.path.getsize(repo_root / relative) > check_file_sizes.LIMIT_BYTES
    ]
    assert offenders == [], "tracked files over 5 MB:\n" + "\n".join(
        f"{name}: {size / 1024 / 1024:.1f} MB" for name, size in offenders
    )


def test_no_venv_or_interpreter_is_tracked(repo_root):
    """Build spec 3.3: the repository is not a venv."""
    tracked = check_file_sizes.tracked_files(repo_root)
    offenders = [
        name
        for name in tracked
        if name.startswith(".venv/") or name.endswith((".exe", ".dll", ".pyd"))
    ]
    assert offenders == [], "interpreter or venv files are tracked:\n" + "\n".join(offenders)


def test_lint_scripts_exit_zero_on_the_current_tree(repo_root):
    """The scripts CI runs must actually pass here, not just their imported logic."""
    for script in ("dash_lint.py", "check_file_sizes.py"):
        done = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / script), str(repo_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, f"{script} failed:\n{done.stdout}\n{done.stderr}"


def test_no_todo_markers_in_src(src_files, repo_root):
    """Ground rule 6: no TODOs in shipped code."""
    pattern = re.compile(r"\bTODO\b|\bFIXME\b|\bXXX\b")
    hits = [
        f"{path.relative_to(repo_root)}:{number}"
        for path in src_files
        for number, line in enumerate(read(path).splitlines(), start=1)
        if pattern.search(line)
    ]
    assert hits == [], "TODO markers in src/:\n" + "\n".join(hits)


def test_runner_never_calls_input(repo_root):
    """Build spec 5.8: the v1 driver blocked on input() whenever anything failed.

    Parsed rather than grepped, because the docstrings in runner.py name ``input()`` while
    explaining why it is banned, and a text search cannot tell prose from a call.
    """
    tree = ast.parse(read(repo_root / "src" / "ufem" / "runner.py"))
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    ]
    assert calls == [], f"runner.py calls input() at line(s) {calls}"
