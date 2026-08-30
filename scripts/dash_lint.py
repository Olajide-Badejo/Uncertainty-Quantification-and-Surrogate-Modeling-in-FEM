"""Ground rule 3, ground rule 4, and binding law 5, enforced as a script.

Three sweeps. The first looks for em dashes (U+2014) and en dashes (U+2013) anywhere in the
tree, because they break TeX ligatures in the report and because I decided once that this
project uses commas, colons, and parentheses instead. The second looks inside ``src/`` only,
for the specific defects the v1 autopsy found: the banned identifiers of build spec section
5.1, seeded global RNG, bare except, and any distribution construction outside
``config.py``. The third is the constants check over ``src/ufem/ui/``, which is binding law 5
made mechanical: the dashboard displays only numbers it read from the artifact store.

Run it directly. Exit 0 is clean, exit 1 prints every hit as ``file:line: reason``.
"""

from __future__ import annotations

import ast
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


#: The dashboard package, relative to the repository root.
UI_PACKAGE = Path("src/ufem/ui")

#: The module inside it that is allowed to declare presentation constants.
UI_LAYOUT_MODULE = "layout.py"

#: Literals that cannot be a measurement whatever they appear next to: an index, an arity, a
#: square, a last element, an empty width. Compared by value, so ``0.0`` and ``1.0`` count too.
UI_STRUCTURAL_LITERALS: tuple[float, ...] = (0, 1, 2, -1)

#: A constant in ``ui/layout.py`` may carry a literal only if its name ends in one of these,
#: each of which names a presentational role: a pixel size, a millisecond budget, a color, an
#: opacity, a padding, a size, a slider resolution, a decimal count, a line width, a panel
#: height, a font size, a dash pattern, a marker, a ratio of the frame, a row or column
#: count, a character count.
#:
#: What is deliberately absent is the point of the list. There is no ``_SCALE``, because a unit
#: conversion is a statement about the quantity; no ``_THRESHOLD`` or ``_LEVEL``, because a
#: limit state and a confidence level are results; no ``_ALPHA``, for the same reason. Those
#: come from the configuration and from the calibration artifact, and the check below is what
#: keeps somebody from adding one here under a presentational sounding name.
UI_PRESENTATION_SUFFIXES: tuple[str, ...] = (
    "PX",
    "MS",
    "COLOR",
    "COLORS",
    "OPACITY",
    "PAD",
    "SIZE",
    "STEPS",
    "DECIMALS",
    "WIDTH",
    "HEIGHT",
    "FONT",
    "DASH",
    "MARKER",
    "RATIO",
    "ROWS",
    "COLS",
    "CHARS",
)

RE_UI_PRESENTATION_NAME = re.compile(
    r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:" + "|".join(UI_PRESENTATION_SUFFIXES) + r")$"
)


def _is_presentation_constant(node: ast.stmt) -> bool:
    """True for a module level assignment to an allowlisted presentation constant name."""
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return False
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    if len(names) != len(targets) or not names:
        return False
    return all(RE_UI_PRESENTATION_NAME.match(name) is not None for name in names)


def check_ui_constants(root: Path) -> list[str]:
    """Binding law 5: no computed constant anywhere in the dashboard package.

    Every number UFEM Lab displays has to have been read from an artifact the pipeline wrote.
    A dashboard is the easiest place in a project to publish a number nobody can regenerate,
    because the number arrives on a screen rather than in a table, so the rule is checked
    rather than asserted.

    A numeric literal under :data:`UI_PACKAGE` is accepted only when it is one of
    :data:`UI_STRUCTURAL_LITERALS`, which cannot carry a measurement, or when it sits inside a
    module level assignment in :data:`UI_LAYOUT_MODULE` whose name matches
    :data:`RE_UI_PRESENTATION_NAME`. Everything else is a hit, and the fix is always the same:
    read the number from the artifact that measured it.

    Parsed rather than grepped. A regular expression cannot tell ``0.9`` in a band level from
    ``0.9`` in an opacity, and it cannot tell either from the digits inside a docstring.
    """
    package = root / UI_PACKAGE
    if not package.is_dir():
        return [
            f"{UI_PACKAGE}: directory not found under {root}. The binding law 5 check would "
            "pass vacuously, which is worse than failing."
        ]
    modules = sorted(package.rglob("*.py"))
    if not modules:
        return [f"{UI_PACKAGE}: no python modules found, so this check protects nothing."]
    hits: list[str] = []
    for path in modules:
        relative = path.relative_to(root)
        source = read_lines(path)
        if source is None:
            continue
        tree = ast.parse("\n".join(source), filename=str(relative))
        allowed: set[int] = set()
        if path.name == UI_LAYOUT_MODULE:
            for statement in tree.body:
                if _is_presentation_constant(statement):
                    end = statement.end_lineno or statement.lineno
                    allowed.update(range(statement.lineno, end + 1))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if any(value == structural for structural in UI_STRUCTURAL_LITERALS):
                continue
            if node.lineno in allowed:
                continue
            hits.append(
                f"{relative}:{node.lineno}: numeric literal {value!r} in the UI package "
                "(binding law 5: the dashboard displays only numbers read from the artifact "
                f"store). Structural literals {list(UI_STRUCTURAL_LITERALS)} are allowed "
                f"anywhere; a presentation constant belongs in {UI_PACKAGE.as_posix()}/"
                f"{UI_LAYOUT_MODULE} under a name ending in one of "
                f"{list(UI_PRESENTATION_SUFFIXES)}; anything else has to come from an artifact."
            )
    return hits


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    hits = check_dashes(root) + check_src_laws(root) + check_ui_constants(root)
    if hits:
        for hit in hits:
            print(hit)
        print(f"\ndash_lint: {len(hits)} violation(s) under {root}")
        return 1
    print(f"dash_lint: clean under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
