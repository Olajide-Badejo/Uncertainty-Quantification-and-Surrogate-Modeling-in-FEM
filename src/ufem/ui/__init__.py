"""UFEM Lab: the local dashboard over the artifact store (build spec section 15).

Five panels, one rule. **Binding law 5: every number this package displays was read from an
artifact the pipeline wrote and a manifest covers.** There is not one computed constant in
here. A dashboard is the easiest place in a project to publish a number nobody can regenerate,
because the number arrives on a screen rather than in a table, so the rule is enforced
mechanically: ``dash_lint.check_ui_constants`` walks every module in this package and rejects
any numeric literal that is neither structurally trivial nor a presentation constant declared
in :mod:`ufem.ui.layout`. The allowlist and its reasoning are documented there and in
``tests/test_ui.py``, which also plants a violation and watches the check fire.

What that means in practice, and it is worth being explicit because the alternative looks
harmless: the predict panel does not know what a calibrated band is. It asks
:mod:`ufem.calibrate` and :mod:`ufem.propagate` for the numbers those stages measured, applies
the surrogate the artifact store holds, and draws the result. The reliability panel does not
know how to count a failure; it calls :func:`ufem.propagate.recompute_limit_state` on rows the
propagate stage persisted. The sensitivity panel does not draw a Sobol bar at all, because the
Q2 gate of build spec 12.1 withheld every index in this campaign, and a bar chart of withheld
indices would be the most convincing lie in the repository.

Module layout, recorded in docs/DESIGN_DECISIONS.md:

* :mod:`ufem.ui.layout` holds every presentation constant, and is the only module allowed to
  carry a numeric literal that is not structural.
* :mod:`ufem.ui.store` loads the artifact store once and hands out what it holds.
* :mod:`ufem.ui.predict` turns three input values into a prediction with its calibrated band,
  its scalar intervals, and its validity verdict.
* :mod:`ufem.ui.figures` builds the Plotly figures, taking data and returning figures.
* :mod:`ufem.ui.app` wires the five panels together and runs the server.
"""

from __future__ import annotations

from ufem.ui.predict import Prediction, ScalarReadout, ValidityVerdict, export_payload, predict
from ufem.ui.store import LabArtifactMissing, LabStore

__all__ = [
    "LabArtifactMissing",
    "LabStore",
    "Prediction",
    "ScalarReadout",
    "ValidityVerdict",
    "export_payload",
    "predict",
]
