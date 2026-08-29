"""The staleness gate: the committed data card must be what the pipeline generates now.

Build spec 19 and binding law 5. A generated document is only evidence while it still equals
what regenerating it would produce. The moment the two diverge, the committed copy is a
snapshot of a pipeline that no longer exists, and every number in it is a hand typed number
that merely used to be true.

So this file regenerates ``docs/DATA_CARD.md`` and every ``report/tables/*.tex`` fragment in
memory and asserts byte identity against what is committed. It never writes; a failure here
means run ``python scripts/make_data_card.py`` and commit the result, which is exactly the
action the failure message names.

The comparison is on the committed text with newlines normalized, because the generator
writes ``\\n`` while git may check the files out with ``\\r\\n`` on Windows. Line endings are
not the drift this gate exists to catch; changed numbers are.
"""

from __future__ import annotations

import pytest
from make_data_card import DATA_CARD, TABLES_DIR, ArtifactMissing, generate

#: Every file the generator owns. A file added to the generator and not to this list would
#: be committed without a staleness gate, so the test asserts the set matches too.
GENERATED = (
    DATA_CARD,
    f"{TABLES_DIR}/macros.tex",
    f"{TABLES_DIR}/design_moments.tex",
    f"{TABLES_DIR}/quartile_failure_rates.tex",
    f"{TABLES_DIR}/calibration.tex",
    f"{TABLES_DIR}/importance_weighting.tex",
)


@pytest.fixture(scope="module")
def regenerated(repo_root):
    try:
        return generate(repo_root)
    except ArtifactMissing as err:
        pytest.skip(f"the pipeline has not run: {err}")


def test_the_generator_owns_exactly_the_files_this_gate_checks(regenerated):
    assert set(regenerated) == set(GENERATED)


@pytest.mark.parametrize("relative", GENERATED)
def test_the_committed_file_is_byte_identical_to_a_fresh_generation(
    repo_root, regenerated, relative
):
    path = repo_root / relative
    assert path.is_file(), (
        f"{relative} is not committed. Run `python scripts/make_data_card.py` and commit it."
    )
    committed = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fresh = regenerated[relative].replace("\r\n", "\n")
    assert committed == fresh, (
        f"{relative} is stale: it differs from what the pipeline generates now. Run "
        "`python scripts/make_data_card.py` and commit the result. Do not edit the file by "
        "hand; every number in it comes from an artifact."
    )


@pytest.mark.parametrize("relative", GENERATED)
def test_no_generated_file_is_empty(regenerated, relative):
    """A generator that silently produced nothing would pass an identity check twice."""
    assert len(regenerated[relative]) > 200


def test_every_report_macro_name_is_unique(regenerated):
    """Two measurements reaching the report through one symbol would overwrite each other."""
    import re

    names = re.findall(
        r"\\newcommand\{\\([A-Za-z]+)\}", regenerated[f"{TABLES_DIR}/macros.tex"]
    )
    assert len(names) == len(set(names)), sorted(
        name for name in names if names.count(name) > 1
    )
    assert len(names) > 40


def test_the_card_carries_no_placeholder_or_unformatted_value(regenerated):
    """A placeholder or a leaked Python repr in the card would be a fabricated number.

    Matched on word boundaries, not as substrings: ``nan`` is a substring of "provenance"
    and ``None`` of nothing useful, and a check that fires on ordinary English is a check
    that gets deleted the first time it is inconvenient.
    """
    import re

    text = regenerated[DATA_CARD]
    for marker in ("TODO", "FIXME", "XXX", "nan", "None", "inf"):
        found = re.search(rf"\b{marker}\b", text)
        assert found is None, (
            f"the generated data card contains the bare token {marker!r} at offset "
            f"{found.start() if found else -1}, which means a measurement did not resolve."
        )
