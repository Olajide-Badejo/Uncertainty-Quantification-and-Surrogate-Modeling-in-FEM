"""Prepare a release: build the PDF, check the tree, and print the command to run.

Build spec 21 and 23. This script does everything a release needs except the release itself. It
verifies that the working tree is clean and on the release branch, that the README injection is
current, that the generated documents are not stale, and that ``latexmk`` builds the report from
the committed fragments; then it prints the ``gh release create`` command with the PDF attached.

It deliberately does not run ``gh``. Tagging and publishing are the one step in this project that
cannot be undone by rerunning a stage, so they stay in a human's hands, and the script's job is
to make sure that when the human runs the command, everything it will attach is current. There
is no ``--yes`` and there is no ``--force``: a check that failed is a release that is not ready.

Exit 0 means ready, and the last line is the command. Exit 1 names every check that failed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

#: Releases are tagged from this branch and no other (build spec 21).
RELEASE_BRANCH = "main"

#: What the release carries. The PDF is gitignored on purpose: it is a large binary no test can
#: diff, so it ships as an asset built from the committed sources rather than as a tracked file.
REPORT_PDF = Path("report") / "main.pdf"

#: The filename the report is attached under. ``gh`` takes the asset name from the file on
#: disk, and ``#`` only sets a display label, so a release built straight from ``main.pdf``
#: would hand a reader a download called ``main.pdf`` and a permanent URL to match. The PDF is
#: therefore staged under this name before the command is printed.
#:
#: ``scripts/readme_inject.py`` imports :func:`report_asset_name` to build the README's direct
#: download link, so the link and the upload cannot drift into two different filenames. That is
#: also what ``tests/test_readme_consistency.py`` asserts.
RELEASE_ASSET_TEMPLATE = "ufem-2.0-report-v{version}.pdf"


def report_asset_name(version: str) -> str:
    """The filename the report PDF is published under for one release."""
    return RELEASE_ASSET_TEMPLATE.format(version=version)


class NotReady(RuntimeError):
    """A release check failed. The message names which one and what to do about it."""


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise NotReady(
            f"git {' '.join(args)} failed with exit {done.returncode}: "
            f"{done.stderr.strip() or 'no stderr'}"
        )
    return done.stdout.strip()


def check_branch(root: Path) -> str:
    """Build spec 21: releases are tagged from main only."""
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != RELEASE_BRANCH:
        raise NotReady(
            f"on branch {branch!r}, and a release is tagged from {RELEASE_BRANCH!r} only. Merge "
            "the phase branch first."
        )
    return branch


def check_clean_tree(root: Path) -> None:
    """A release built from a dirty tree is a release nobody can reproduce."""
    status = _git(root, "status", "--porcelain")
    if status:
        raise NotReady(
            "the working tree is not clean:\n  "
            + "\n  ".join(status.splitlines())
            + "\nCommit or discard before releasing; the manifests record the commit and a "
            "dirty tree makes that record a lie."
        )


def check_generated_documents(root: Path) -> None:
    """The README, the data card and the model card must equal what regenerating produces."""
    import make_data_card
    import make_model_card
    import readme_inject

    failures = []
    for module, label in (
        (readme_inject, "README"),
        (make_data_card, "data card and report fragments"),
        (make_model_card, "model card"),
    ):
        try:
            generated = module.generate(root)
        except (RuntimeError, FileNotFoundError) as err:
            raise NotReady(
                f"the {label} generator could not run: {err}. The pipeline has to have run "
                "for this config hash before a release."
            ) from err
        for relative, content in generated.items():
            path = root / relative
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            if current != content.replace("\r\n", "\n"):
                failures.append(relative)
    if failures:
        raise NotReady(
            "these generated files are stale:\n  "
            + "\n  ".join(sorted(failures))
            + "\nRerun scripts/readme_inject.py, scripts/make_data_card.py and "
            "scripts/make_model_card.py, then commit."
        )


def check_lints(root: Path) -> None:
    """The same gates CI runs, so a release is never the first place they are checked."""
    import check_file_sizes
    import dash_lint

    for module, label in ((dash_lint, "dash_lint"), (check_file_sizes, "check_file_sizes")):
        code = module.main([str(root)])
        if code != 0:
            raise NotReady(f"{label} exited {code}; fix the violations it printed.")


def build_report(root: Path) -> Path:
    """Compile the PDF with latexmk, from the committed fragments and figures."""
    report = root / "report"
    done = subprocess.run(
        ["latexmk", "-pdf", "-halt-on-error", "-interaction=nonstopmode", "main.tex"],
        cwd=report,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        tail = "\n".join(done.stdout.splitlines()[-40:])
        raise NotReady(f"latexmk failed with exit {done.returncode}:\n{tail}")
    pdf = root / REPORT_PDF
    if not pdf.is_file():
        raise NotReady(f"latexmk reported success but {REPORT_PDF} does not exist.")
    return pdf


def release_version(root: Path) -> str:
    """The version to tag, from the one file that declares it."""
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(metadata["project"]["version"])
    if ".dev" in version:
        raise NotReady(
            f"pyproject.toml declares {version!r}, which is a development version. Drop the "
            "dev suffix in a commit of its own before releasing."
        )
    return version


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else REPO_ROOT
    checks = (
        ("branch", lambda: check_branch(root)),
        ("clean tree", lambda: check_clean_tree(root)),
        ("generated documents", lambda: check_generated_documents(root)),
        ("lints", lambda: check_lints(root)),
        ("version", lambda: release_version(root)),
        ("report build", lambda: build_report(root)),
    )
    failures = []
    for label, check in checks:
        try:
            check()
        except NotReady as err:
            failures.append(f"{label}: {err}")
            print(f"[fail] {label}")
        else:
            print(f"[ok]   {label}")
    if failures:
        print("\nnot ready to release:\n")
        for failure in failures:
            print(f"  {failure}\n")
        return 1
    version = release_version(root)
    pdf = root / REPORT_PDF
    asset = REPORT_PDF.with_name(report_asset_name(version))
    shutil.copyfile(pdf, root / asset)
    print(
        f"\nready to release v{version}. The report is staged at {asset.as_posix()}, "
        f"{pdf.stat().st_size / 1024 / 1024:.2f} MB, which is the filename the README's "
        "download link points at."
    )
    print("\nRun this yourself; this script does not tag and does not publish:\n")
    print(
        f'  gh release create v{version} "{asset.as_posix()}#UFEM 2.0 report" '
        f'--title "UFEM 2.0 v{version}" --notes-file docs/RELEASE_CHECKLIST.md'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
