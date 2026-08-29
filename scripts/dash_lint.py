"""Ground rule 3 and ground rule 4, enforced as a script.

Two sweeps. The first looks for em dashes (U+2014) and en dashes (U+2013) anywhere in the
tree, because they break TeX ligatures in the report and because I decided once that this
project uses commas, colons, and parentheses instead. The second looks inside ``src/`` only,
for the specific defects the v1 autopsy found: the banned identifiers of build spec section
5.1, seeded global RNG, bare except, and any distribution construction outside
``config.py``.

Run it directly. Exit 0 is clean, exit 1 prints every hit as ``file:line: reason``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Written as escapes on purpose: this file must not contain the characters it bans, or it
# would report itself.
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

#: Never walked at all: frozen trees, the venv, generated artifacts, and the build spec,
#: which is exempt because it is the input document and is kept byte identical.
SKIP_DIRS = {
    ".git",
    ".venv",
    "v1_legacy",
    "legacy_salvage",
    "data_audit",
    "data",
    "experiments",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
SKIP_PATHS = {
    Path("docs/BUILD_SPEC.md"),
    Path("report/figures"),
}
SKIP_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".npy",
    ".npz",
    ".parquet",
    ".pptx",
    ".xlsx",
    ".pyc",
    ".ico",
    ".woff",
    ".woff2",
}

#: Ground rule 4: a direct memorial to the fabricated uncertainty of build spec 5.1.
BANNED_IDENTIFIERS = ("AMPLIFY", "noise_level")

#: Binding law 2: only config.py may name a distribution family or a QMC engine.
DISTRIBUTION_SUBSTRINGS = ("lognorm", "scipy.stats.norm", "qmc.LatinHypercube")

RE_RANDOM_SEED = re.compile(r"np\.random\.seed")
RE_BARE_EXCEPT = re.compile(r"except\s*:")


def iter_candidates(root: Path):
    """Yield every text file worth scanning, relative to the repository root."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if any(relative == skip or skip in relative.parents for skip in SKIP_PATHS):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield relative


def read_lines(path: Path) -> list[str] | None:
    """Decode as UTF-8, or return None for anything that is not text."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def check_dashes(root: Path) -> list[str]:
    hits = []
    for relative in iter_candidates(root):
        lines = read_lines(root / relative)
        if lines is None:
            continue
        for number, line in enumerate(lines, start=1):
            if EM_DASH in line:
                hits.append(f"{relative}:{number}: em dash (U+2014), use a comma or parentheses")
            if EN_DASH in line:
                hits.append(f"{relative}:{number}: en dash (U+2013), use 'to' or a hyphen")
    return hits


def check_src_laws(root: Path) -> list[str]:
    """The binding law greps, over ``src/`` only."""
    hits = []
    src = root / "src"
    if not src.is_dir():
        return [f"src/: directory not found under {root}, nothing to check"]
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(root)
        lines = read_lines(path)
        if lines is None:
            continue
        is_config = path.name == "config.py"
        for number, line in enumerate(lines, start=1):
            for word in BANNED_IDENTIFIERS:
                if word in line:
                    hits.append(
                        f"{relative}:{number}: banned identifier {word!r} "
                        "(ground rule 4: uncertainty is computed, never styled)"
                    )
            if RE_RANDOM_SEED.search(line):
                hits.append(
                    f"{relative}:{number}: np.random.seed is banned "
                    "(ground rule 13: RNG comes from a SeedSequence tree)"
                )
            if RE_BARE_EXCEPT.search(line):
                hits.append(
                    f"{relative}:{number}: bare except "
                    "(ground rule 8: silent fallbacks are forbidden)"
                )
            if not is_config:
                for token in DISTRIBUTION_SUBSTRINGS:
                    if token in line:
                        hits.append(
                            f"{relative}:{number}: {token!r} outside config.py "
                            "(binding law 2: the probabilistic model is declared once)"
                        )
    return hits


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    hits = check_dashes(root) + check_src_laws(root)
    if hits:
        for hit in hits:
            print(hit)
        print(f"\ndash_lint: {len(hits)} violation(s) under {root}")
        return 1
    print(f"dash_lint: clean under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
