"""The staleness gate for the model card, and the honesty checks on what it says.

Build spec 19 and binding law 5, the same pattern as ``tests/test_data_card.py``: a generated
document is only evidence while it still equals what regenerating it would produce. The moment
the two diverge, the committed copy is a snapshot of a pipeline that no longer exists, and
every number in it is a hand typed number that merely used to be true.

Beyond the byte comparison, this file asserts the three things the model card exists to say
that a byte comparison cannot check: that the known failure modes section is populated, that
the withheld sensitivity indices are described as withheld, and that the card and UFEM Lab's
model card panel are reading the same artifacts rather than two copies of them.
"""

from __future__ import annotations

import re

import pytest
from make_model_card import MODEL_CARD, ArtifactMissing, generate


@pytest.fixture(scope="module")
def regenerated(repo_root):
    try:
        return generate(repo_root)
    except ArtifactMissing as err:
        pytest.skip(f"the pipeline has not run: {err}")


def test_the_generator_owns_exactly_the_model_card(regenerated):
    assert set(regenerated) == {MODEL_CARD}


def test_the_committed_card_is_byte_identical_to_a_fresh_generation(repo_root, regenerated):
    path = repo_root / MODEL_CARD
    assert path.is_file(), (
        f"{MODEL_CARD} is not committed. Run `python scripts/make_model_card.py` and commit it."
    )
    committed = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fresh = regenerated[MODEL_CARD].replace("\r\n", "\n")
    assert committed == fresh, (
        f"{MODEL_CARD} is stale: it differs from what the pipeline generates now. Run "
        "`python scripts/make_model_card.py` and commit the result. Do not edit the file by "
        "hand; every number in it comes from an artifact."
    )


def test_the_card_records_no_wall_time(regenerated):
    """A generated document must not embed a quantity that changes between runs."""
    text = regenerated[MODEL_CARD].lower()
    for phrase in ("wall time", "wall_time", "elapsed", "seconds to run"):
        assert phrase not in text, (
            f"the model card embeds {phrase!r}, which varies between runs and would make the "
            "staleness gate fire on noise rather than on drift."
        )


def test_the_card_carries_no_placeholder_or_unformatted_value(regenerated):
    """A leaked repr in the card would be a measurement that did not resolve."""
    text = regenerated[MODEL_CARD]
    for marker in ("TODO", "FIXME", "XXX", "nan", "None", "inf"):
        found = re.search(rf"\b{marker}\b", text)
        assert found is None, (
            f"the model card contains the bare token {marker!r} at offset {found.start()}, "
            "which means a measurement did not resolve."
        )


def test_the_card_carries_no_latex_fragment(regenerated):
    """The card is markdown. A LaTeX escape hatch here renders as dollar signs and braces."""
    assert "$" not in regenerated[MODEL_CARD]
    assert "\\begin{" not in regenerated[MODEL_CARD]


def test_the_known_failure_modes_section_is_populated(regenerated):
    """The section that names what the model cannot do is the point of the document."""
    text = regenerated[MODEL_CARD]
    assert "## Known failure modes" in text
    section = text.split("## Known failure modes", 1)[1]
    section = section.split("## Intended use", 1)[0]
    for topic in (
        "Design roughness",
        "Withheld sensitivity indices",
        "Censoring",
        "Damage saturation",
        "Fixed model parameters",
    ):
        assert topic in section, topic
    assert len(section) > 1500


def test_the_card_reports_the_sensitivity_indices_as_withheld(repo_root, regenerated):
    """P6 withheld every index. A model card that quoted one would be the whole failure."""
    import json

    from ufem.config import config_hash, load_config
    from ufem.manifest import stage_dir
    from ufem.sensitivity import PUBLICATION_WITHHELD, SENSITIVITY_JSON
    from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE

    config = load_config(repo_root)
    path = (
        stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            SENSITIVITY_STAGE,
            config_hash(config),
        )
        / SENSITIVITY_JSON
    )
    if not path.is_file():
        pytest.skip("the sensitivity stage has not run for this config hash")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["gate_outcome"] == PUBLICATION_WITHHELD, (
        "the gate outcome changed; this test and the model card's wording both have to be "
        "revisited rather than one of them."
    )
    text = regenerated[MODEL_CARD]
    assert "**withheld**" in text
    assert "no Sobol index value and no input ranking is published" in text


def test_the_card_states_every_limit_state_the_pipeline_propagated(repo_root, regenerated):
    import json

    from ufem.config import config_hash, load_config
    from ufem.manifest import stage_dir
    from ufem.propagate import PROPAGATION_JSON
    from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE

    config = load_config(repo_root)
    path = (
        stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            PROPAGATE_STAGE,
            config_hash(config),
        )
        / PROPAGATION_JSON
    )
    if not path.is_file():
        pytest.skip("the propagate stage has not run for this config hash")
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = regenerated[MODEL_CARD]
    for record in payload["limit_states"]:
        assert record["short_label"] in text, record["config_field"]
    assert payload["roughness_caveat"] in text


def test_the_card_and_the_dashboard_read_the_same_stages(repo_root):
    """Two documents built from two lists of artifacts would eventually say two things."""
    pytest.importorskip("torch")
    pytest.importorskip("nicegui")
    import make_model_card

    from ufem.ui.store import REQUIRED_STAGES as UI_STAGES

    card_stages = {stage for stage, _how in make_model_card.REQUIRED_STAGES}
    ui_stages = {stage for stage, _how in UI_STAGES}
    assert card_stages <= ui_stages, sorted(card_stages - ui_stages)


def test_the_committed_card_is_referenced_by_the_documentation_set(repo_root):
    """Build spec 19 lists the model card. A generated file nothing points at goes stale."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "docs/MODEL_CARD.md" in readme


def test_the_card_embeds_no_git_commit(regenerated):
    """A commit changes when a stage is rerun while its numbers do not, so it stays out.

    This is the wall time defect of the data card in its second form: a generated document
    must not carry a quantity that moves without a measurement moving, or its staleness gate
    fires on noise and gets muted, and then it stops catching the drift it exists for. The
    surrogate stage's manifest still records the commit, and the card says where to find it.

    Matched as a bare 40 hex character token, which is what a git object name looks like. The
    config hash and the artifact digests are 64 characters and are deliberately not matched.
    """
    found = re.search(r"\b[0-9a-f]{40}\b", regenerated[MODEL_CARD])
    assert found is None, (
        f"the model card carries what looks like a git commit at offset {found.start()}: "
        f"{found.group()}. It belongs in the manifest, not in a byte gated document."
    )


def test_the_provenance_hashes_are_the_ones_that_move_only_when_the_model_does(
    repo_root, regenerated
):
    from ufem.config import config_hash, load_config
    from ufem.manifest import load_manifest, stage_dir
    from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
    from ufem.surrogate import SURROGATE_JSON

    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, SURROGATE_STAGE, config_hash(config)
    )
    if not (directory / "manifest.json").is_file():
        pytest.skip("the surrogate stage has not run for this config hash")
    manifest = load_manifest(directory)
    digest = {record["name"]: record["sha256"] for record in manifest["outputs"]}[
        SURROGATE_JSON
    ]
    text = regenerated[MODEL_CARD]
    assert config_hash(config) in text
    assert digest in text
