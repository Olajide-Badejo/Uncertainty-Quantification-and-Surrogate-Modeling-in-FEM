"""The one style module: every report and UI figure is drawn through this.

Build spec section 8. A house style that lives in a comment at the top of one script is a
house style that the second script ignores. This module is imported by every figure
function, so the palette, the type sizes, and the chrome are the same object rather than the
same intention.

Design decisions, each with its reason:

- **Two identity colors, not a rainbow.** The report distinguishes at most two groups at a
  time (completed against failed, predicted against empirical), so two hues that clear a
  colorblind safe all pairs check are enough, and a third would only invite a figure that
  encodes something it should have faceted instead.
- **Recessive chrome.** Hairline solid gridlines, no top or right spine, tick labels a shade
  lighter than the body ink. The data is the darkest thing on the page.
- **No titles inside the figure.** Captions live in LaTeX, where they can be referenced and
  where they do not compete with the axis labels for the reader's first fixation.
- **Units on every axis label.** A number without a unit is not a measurement.
- **Type sized for the worst realistic case.** Figures are drawn 6.3 inches wide and may be
  scaled to 0.6 of a column; the sizes below keep every glyph legible after that.

Determinism: the PDF creation timestamp is pinned, so regenerating a figure from unchanged
inputs produces unchanged bytes and a figure that did change is visible in a diff.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)

#: Categorical identity colors. Slots 1 and 2 of the reference palette, which clear the all
#: pairs colorblind gate (worst CVD dE 9.2, normal vision dE 24.0, OKLab times 100).
C_SERIES_1 = "#2a78d6"
C_SERIES_2 = "#eb6834"

#: Fill for a pointwise envelope: the same blue, far lighter, so a band never reads as a line.
C_BAND = "#9ec5f4"

#: Ink and chrome, darkest to lightest.
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"

#: Figure width in inches, about one column of an A4 body at the report's margins.
FIG_WIDTH_IN = 6.3

#: Pinned so repeated runs write byte comparable PDFs.
SOURCE_DATE_EPOCH = "1700000000"

_RC_PARAMS: dict[str, Any] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "figure.titlesize": 10.0,
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "xtick.labelcolor": C_INK_2,
    "ytick.labelcolor": C_INK_2,
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.borderaxespad": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_style() -> None:
    """Install the house style into the active matplotlib session."""
    plt.rcParams.update(_RC_PARAMS)
    os.environ.setdefault("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)


def annotation_style() -> dict[str, Any]:
    """Keyword arguments for an in axes annotation box, so every one looks the same."""
    return {
        "fontsize": 9.0,
        "color": C_INK_2,
        "ha": "left",
        "va": "top",
        "bbox": {
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": C_GRID,
            "linewidth": 0.5,
            "alpha": 0.92,
        },
    }


#: Raster preview resolution when one is requested. Overridable through ``UFEM_FIG_PNG_DPI``,
#: which is how ``scripts/make_readme_media.py`` gets the README images out of exactly the same
#: figure functions the report compiles instead of drawing a second, divergent set.
PNG_DPI = 200


def save_figure(
    fig: Any, path: Path | str, png_dir: str | None = None, png_dpi: int | None = None
) -> Path:
    """Write one vector PDF, optionally a raster preview, and close the figure.

    Closing matters in a script that draws a dozen figures: matplotlib holds every open
    figure in memory, and a loop that forgets to close them is the usual way a figure script
    turns into a memory problem on a larger campaign.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="pdf")
    directory = png_dir if png_dir is not None else os.environ.get("UFEM_FIG_PNG")
    if directory:
        preview = Path(directory)
        preview.mkdir(parents=True, exist_ok=True)
        if png_dpi is None:
            png_dpi = int(os.environ.get("UFEM_FIG_PNG_DPI", PNG_DPI))
        fig.savefig(preview / (target.stem + ".png"), dpi=png_dpi)
    plt.close(fig)
    return target
