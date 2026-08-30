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
    f"{TABLES_DIR}/reduction_summary.tex",
    f"{TABLES_DIR}/baselines_table.tex",
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


@pytest.mark.parametrize("relative", GENERATED)
def test_no_generated_file_records_a_wall_time(regenerated, relative):
    """A generated document must not embed a quantity that changes between runs.

    This is a real defect that shipped and was caught: the data card's provenance table
    listed each stage's wall time, read from its manifest. Wall time varies between runs on
    the same machine, so rerunning a stage made the byte comparison above fail even though
    no measurement had changed. A staleness gate that fires on scheduling noise is a gate
    that gets muted, which would then have hidden the drift it exists to catch.

    Wall times still live in every manifest and in the engineering log. They are simply not
    allowed into a file whose whole purpose is to be reproducible byte for byte.
    """
    text = regenerated[relative].lower()
    for phrase in ("wall time", "wall_time", "elapsed", "seconds to run"):
        assert phrase not in text or "deliberately absent" in text, (
            f"{relative} embeds {phrase!r}, which varies between runs and would make the "
            "staleness gate fire on noise."
        )


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


@pytest.mark.fullstack
def test_the_ablation_fragment_matches_the_ablation_artifact(repo_root):
    """The one report fragment the data card does not own must not drift either.

    ``report/tables/ablation_registration.tex`` is written by
    ``scripts/ablation_1_registration.py`` rather than by the card generator, so the byte
    comparison above does not cover it. Without this test it would be the single committed
    number in the report with no gate behind it, which is exactly the gap that let a stale
    figure reach a CI run at the end of P2. Here the fragment is checked against the values
    in the ablation's own JSON artifact rather than regenerated, so the test states the
    relationship that has to hold instead of duplicating the formatting code.
    """
    import json

    from ufem.config import config_hash, load_config
    from ufem.manifest import stage_dir

    config = load_config(repo_root)
    artifact = (
        stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            "ablation_1_registration",
            config_hash(config),
        )
        / "ablation_1_registration.json"
    )
    fragment = repo_root / "report" / "tables" / "ablation_registration.tex"
    if not artifact.is_file() or not fragment.is_file():
        pytest.skip("the registration ablation has not been run for this config hash")

    results = json.loads(artifact.read_text(encoding="utf-8"))
    text = fragment.read_text(encoding="utf-8")
    counts = results["components_at_target"]
    correlations = results["derivative_mode_correlation"]
    assert f"{int(counts['registered'])} & {int(counts['unregistered'])}" in text
    for side in ("registered", "unregistered"):
        assert f"{correlations[side]:.3f}" in text, side
        bias = results["peak_load_bias"][side]["mean_signed_error_N"]
        assert f"{bias:+.1f}" in text, side
