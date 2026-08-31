"""Export the README's result images from the report's own figure functions.

Build spec 21 wants the repository page to show the work, and the temptation in every project
that reaches this point is to draw a second, prettier set of figures for the front page. That is
how a README ends up disagreeing with the document it advertises. This script does the opposite:
it runs ``report/figures_src/make_figures.py`` unchanged, with the raster preview hook of
:func:`ufem.plotting.style.save_figure` pointed at a scratch directory, and then copies the
selected previews into ``docs/media/``. The PNG in the README and the PDF in the report are
therefore the same figure, from the same artifacts, drawn by the same function on the same run.

Six figures, chosen because each one carries a claim the README makes in words:

- the load displacement family, which is the data and the softening;
- the design and its censoring, which is why a validity domain exists at all;
- registration before and after, which is the methodological core;
- the simultaneous conformal band, which is what a measured coverage looks like;
- predicted against actual peak load, which is the quantity the reliability analysis thresholds;
- the propagated curve envelope with its limit states, which is the output.

Everything the figures contain came from the artifact store, so this script needs the pipeline
to have run and says so by name when it has not.

Exit 0 is clean, exit 1 names the failure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MEDIA = "docs/media"

#: The raster resolution for the README images. High enough to stay sharp on a wide screen,
#: low enough that six of them together are a rounding error against the 5 MB file size gate.
DPI = 150

#: Figure stem to the claim it carries, in the order the README shows them. Stems, not paths:
#: the same stem names the PDF in ``report/figures/`` and the PNG in ``docs/media/``.
SELECTED: tuple[tuple[str, str], ...] = (
    ("fig_ld_family", "the completed campaign on the common grid, with its envelope"),
    ("fig_design_censoring", "the executed design, completed against failed"),
    ("fig_registration_before_after", "amplitude separated from phase"),
    ("fig_conformal_band", "held out curves against the simultaneous band"),
    ("fig_pmax_predicted_vs_actual", "predicted against actual peak load, out of sample"),
    ("fig_curve_envelope", "the propagated envelope with its limit states"),
)

#: No single image may exceed this. The gate in ``scripts/check_file_sizes.py`` is 5 MB and
#: applies to every tracked file; a README image that came anywhere near it would be a mistake
#: long before it broke the gate, so this script fails at a much lower bar and says why.
MAX_BYTES = 1_500_000


class MediaFailed(RuntimeError):
    """The figures did not render, or one of the selected figures is not among them."""


def render(scratch: Path) -> None:
    """Run the report's figure generator with the raster preview hook enabled."""
    figures_src = REPO_ROOT / "report" / "figures_src"
    if str(figures_src) not in sys.path:
        sys.path.insert(0, str(figures_src))
    os.environ["UFEM_FIG_PNG"] = str(scratch)
    os.environ["UFEM_FIG_PNG_DPI"] = str(DPI)
    import make_figures

    code = make_figures.main()
    if code != 0:
        raise MediaFailed(
            f"report/figures_src/make_figures.py exited {code}; the README images are copies "
            "of its output and are not generated any other way."
        )


def copy_selected(scratch: Path, root: Path) -> list[tuple[str, int]]:
    """Copy the selected previews into ``docs/media/``, refusing anything missing or large."""
    destination = root / MEDIA
    destination.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []
    for stem, _claim in SELECTED:
        source = scratch / f"{stem}.png"
        if not source.is_file():
            raise MediaFailed(
                f"{stem}.png was not produced at {source}. Either the figure was renamed in "
                "report/figures_src/make_figures.py and this script's selection is stale, or "
                "the run stopped early."
            )
        size = source.stat().st_size
        if size > MAX_BYTES:
            allowed = MAX_BYTES / 1024 / 1024
            raise MediaFailed(
                f"{stem}.png is {size / 1024 / 1024:.2f} MB, above the {allowed:.2f} MB this "
                "script allows for a README image. Lower the resolution rather than raising "
                "the limit."
            )
        shutil.copyfile(source, destination / f"{stem}.png")
        written.append((f"{MEDIA}/{stem}.png", size))
    return written


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="ufem_readme_media_") as scratch:
        directory = Path(scratch)
        render(directory)
        written = copy_selected(directory, root)
    total = 0
    for relative, size in written:
        total += size
        print(f"{relative}: {size / 1024:.0f} KB")
    print(f"make_readme_media: {len(written)} image(s), {total / 1024:.0f} KB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
