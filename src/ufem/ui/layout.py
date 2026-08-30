"""Presentation constants for UFEM Lab, and the only place in ``ui/`` allowed to hold one.

Binding law 5 says the UI displays only numbers read from the artifact store. That rule has to
survive contact with a plotting library, which needs a line width, a marker size, a panel
height and a legend offset before it will draw anything, and none of those are measurements of
a beam. So the law is enforced as a shape rather than as an absolute: a numeric literal may
appear in this package only if it is structurally trivial (``0``, ``1``, ``2``, ``-1``: an
index, an arity, a square, a last element) or if it is declared here under a name whose suffix
says which presentational role it plays.

The allowed suffixes are in :data:`ufem.ui.layout.PRESENTATION_SUFFIXES` and the check that
enforces them is ``dash_lint.check_ui_constants``. The suffix list deliberately excludes
anything that could launder a measurement: there is no ``_SCALE``, no ``_THRESHOLD``, no
``_LEVEL`` and no ``_ALPHA``, because a unit conversion, a limit state and a confidence level
are all statements about the model. Those come from
:data:`ufem.propagate.QOI_DISPLAY`, from the configuration, and from the calibration artifact.

Colors are imported from :mod:`ufem.plotting.style` rather than restated, so the dashboard and
the report figures are the same palette rather than two palettes that were once the same.
"""

from __future__ import annotations

from ufem.plotting.style import (
    C_AXIS,
    C_BAND,
    C_GRID,
    C_INK,
    C_INK_2,
    C_MUTED,
    C_SERIES_1,
    C_SERIES_2,
)

#: The suffixes ``dash_lint.check_ui_constants`` accepts on a constant that carries a literal.
#: Named here so the linter and the documentation cannot drift apart.
PRESENTATION_SUFFIXES: tuple[str, ...] = (
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

# -- palette ----------------------------------------------------------------

#: The predicted mean, its band, and the second identity color for a comparison series.
PREDICTION_COLOR = C_SERIES_1
COMPARISON_COLOR = C_SERIES_2
BAND_COLOR = C_BAND

#: Completed against failed in the design scatter. Completed takes the identity blue; failed
#: takes the muted ink rather than a red, because a solver that did not finish is missing data
#: and not an alarm.
COMPLETED_COLOR = C_SERIES_1
FAILED_COLOR = C_MUTED

#: Chrome. The data stays the darkest thing on the panel.
INK_COLOR = C_INK
SUBDUED_COLOR = C_INK_2
MUTED_COLOR = C_MUTED
GRID_COLOR = C_GRID
AXIS_COLOR = C_AXIS
PAPER_COLOR = "#ffffff"

#: A prediction the validity domain does not cover is drawn in this gray, and only in this
#: gray: the curve stays visible so the shape can still be read, and nothing about it says
#: the number under the cursor is trustworthy.
GRAYED_COLOR = "#9a9a9a"
WARNING_COLOR = "#b45309"

#: Fills, from the strongest to the faintest.
BAND_OPACITY = 0.28
ENVELOPE_OPACITY = 0.16
SCATTER_OPACITY = 0.75
GRAYED_OPACITY = 0.35

# -- geometry ---------------------------------------------------------------

CURVE_PANEL_HEIGHT_PX = 340
DAMAGE_PANEL_HEIGHT_PX = 260
MATRIX_PANEL_HEIGHT_PX = 620
SURFACE_PANEL_HEIGHT_PX = 330
DENSITY_PANEL_HEIGHT_PX = 320
OVERLAY_PANEL_HEIGHT_PX = 380
DIAGNOSTIC_PANEL_HEIGHT_PX = 300
SENSITIVITY_PANEL_HEIGHT_PX = 420

MARGIN_PX = 48
TOP_MARGIN_PX = 28
LEGEND_PAD = 0.02

#: Axis padding as a share of the data range, so a curve never touches the frame.
AXIS_PAD_RATIO = 0.05

#: Spacing between the panels of a subplot row, as a share of the figure width.
SUBPLOT_SPACING_RATIO = 0.07

#: Contour spacing on the completion probability surface, as a share of the probability axis.
CONTOUR_SIZE_RATIO = 0.05

LINE_WIDTH = 2.2
THIN_LINE_WIDTH = 1.2
HAIRLINE_WIDTH = 0.8
MARKER_SIZE = 6
SMALL_MARKER_SIZE = 4
SELECTED_MARKER_SIZE = 13

DASH_PATTERN_DASH = "dash"
DASH_PATTERN_DOT = "dot"

TICK_FONT = 11
LABEL_FONT = 12

# -- interaction ------------------------------------------------------------

#: Slider resolution. Three sliders over a bounded design range, and this many stops across
#: it: fine enough that a drag looks continuous, coarse enough that a drag does not queue more
#: recomputations than the latency budget can retire.
SLIDER_STEPS = 200

#: The completion probability surface is evaluated on this many points per axis.
SURFACE_GRID_STEPS = 41

# -- formatting -------------------------------------------------------------

#: How much of a SHA-256 to print where a full digest would swamp the line it sits on. A
#: truncated hash is an identifier for a reader, never for a check: everything that verifies
#: a digest in this project compares the whole thing.
SHORT_HASH_CHARS = 12

#: Significant figures per kind of readout. A displayed digit that the measurement does not
#: support is a fabricated digit, so these are deliberately short.
VALUE_DECIMALS = 3
COARSE_DECIMALS = 1
PROBABILITY_DECIMALS = 4
INPUT_DECIMALS = 2
