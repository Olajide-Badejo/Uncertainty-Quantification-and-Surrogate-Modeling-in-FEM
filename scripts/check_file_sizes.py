"""Build spec section 3.3: no file over 5 MB enters this repository, and none is exempt.

The v1 tree shared one git directory with an interpreter, twelve console script
executables, and 405 MB of solver scratch. The fix is not a better .gitignore, it is a gate:
anything git actually tracks must be under 5 MB, and large inherited data is referenced by
manifest hash at a pinned location instead of copied in.

``docs/media/ufem_lab.gif`` was expected to need an exemption here, because build spec 15.1
requires it to be committed at 12 fps and 960 px and a recording of a live dashboard is not
obviously a small file. It did not, twice: the first capture measured 0.90 MB and the longer
P10 recapture, at a taller frame so no panel is clipped, measures 1.49 MB. Both are well
inside the rule, so no exemption was added. Those measurements are in
docs/ENGINEERING_LOG.md and the decision not to weaken the gate pre emptively is in
docs/DESIGN_DECISIONS.md. If a future capture crosses 5 MB, that is a decision to make then,
in a commit that says so; ``scripts/capture_ui_gif.py`` now checks against this limit rather
than against the looser one in build spec 15.1, so it fails at capture time instead of one
commit later.

Exit 0 is clean, exit 1 lists every offender with its size.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LIMIT_BYTES = 5 * 1024 * 1024


def tracked_files(root: Path) -> list[str]:
    """Everything git tracks, as repo relative paths."""
    done = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {root} with exit {done.returncode}: "
            f"{done.stderr.strip() or 'no stderr'}. This check needs a real git repository."
        )
    return [line for line in done.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    offenders = []
    checked = 0
    for relative in tracked_files(root):
        path = root / relative
        if not path.is_file():
            continue
        checked += 1
        size = os.path.getsize(path)
        if size > LIMIT_BYTES:
            offenders.append((relative, size))
    if offenders:
        for relative, size in sorted(offenders, key=lambda item: -item[1]):
            print(f"{relative}: {size / 1024 / 1024:.1f} MB exceeds the 5 MB limit")
        print(f"\ncheck_file_sizes: {len(offenders)} file(s) over 5 MB")
        return 1
    print(f"check_file_sizes: {checked} tracked files, all under 5 MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
