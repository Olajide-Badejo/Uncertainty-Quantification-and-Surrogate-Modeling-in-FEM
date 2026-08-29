"""Figure code for the report and the UI, sharing one style module.

Build spec section 8 puts every report and UI figure function here behind a single style
module, so a figure cannot drift from the house style by being drawn somewhere else. The
style itself lives in :mod:`ufem.plotting.style`; the figure functions live beside it.

Nothing in this package reads a CSV or fits a model. Figures take arrays and frames that a
stage already produced, so a figure is never the place a number is first computed. That is
what keeps binding law 5 true of the report: every plotted value traces to an artifact.
"""

from __future__ import annotations

from ufem.plotting.style import (
    C_BAND,
    C_GRID,
    C_INK,
    C_INK_2,
    C_MUTED,
    C_SERIES_1,
    C_SERIES_2,
    FIG_WIDTH_IN,
    annotation_style,
    apply_style,
    save_figure,
)

__all__ = [
    "C_BAND",
    "C_GRID",
    "C_INK",
    "C_INK_2",
    "C_MUTED",
    "C_SERIES_1",
    "C_SERIES_2",
    "FIG_WIDTH_IN",
    "annotation_style",
    "apply_style",
    "save_figure",
]
