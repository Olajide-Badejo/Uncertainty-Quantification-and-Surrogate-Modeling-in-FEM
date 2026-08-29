"""Shared fixtures. The repo root is resolved from this file, never from the cwd."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make the src layout importable when the package is not pip installed, so the suite runs
# the same way from a bare checkout as from an editable install.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SCRIPTS = str(REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
