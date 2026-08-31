"""The staleness gate for the README, and the rule that no number lives outside its markers.

Ground rule 10 and binding law 5: a README claim that disagrees with the manifest it came from
is a CI failure, and hand typed numbers are forbidden. Two halves, and neither one is sufficient
alone.

The first half is the data card's and the model card's pattern, unchanged: regenerate the
injection and compare byte for byte, so a README that has drifted from the pipeline is a failing
test rather than a page nobody rechecked. It needs the artifact store, so it skips on a clean
checkout the way every other artifact backed test in this suite does.

The second half is the one a byte comparison cannot do, and it is the half that catches the
realistic failure. An injected block that is correct is worth nothing if somebody writes a
second, staler number into the paragraph above it, because a reader cannot tell the two apart:
they are both just text on the page. So the prose outside the markers is scanned for anything
that looks like a measurement, a digit carrying a unit or a decimal with real precision, and any
hit fails. That check reads the file and needs no artifacts, which means it is the part that
actually runs in CI.

The third group verifies the injected numbers against the artifacts directly, recomputing each
one from the stage JSON rather than trusting the generator that wrote it. A generator that reads
the wrong key would otherwise pass the byte comparison against itself forever.
"""

from __future__ import annotations

import json
import re
import tomllib

import pytest

import readme_inject
from readme_inject import (
    BLOCKS,
    MARKER_BEGIN,
    MARKER_END,
    README,
    ArtifactMissing,
    generate,
)

# ---------------------------------------------------------------------------
# What counts as a numeric claim in prose
# ---------------------------------------------------------------------------

#: A digit carrying a unit or a statistical name. These are the shapes a result takes in this
#: project's prose: a force, a length, a strength, a share, a coefficient of determination, a
#: probability, a duration, a file size.
UNIT_CLAIM = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*"
    r"(?:kN/mm|kN|MPa|mm|N\b|J\b|%|percent|R2|R\^2|MB|KB|GB|ms\b|s\b|"
    r"seconds?|minutes?|hours?|draws|runs|simulations|samples|folds|components|curves)",
    re.IGNORECASE,
)

#: A decimal with the precision of a measurement. Two places or more: a version string has at
#: most one dot pair and is allowlisted below, and nothing else in prose has three decimals by
#: accident.
PRECISE_DECIMAL = re.compile(r"\b\d+\.\d{2,}\b")

#: Matched text that is not a claim about the beam. Kept short on purpose: the way to satisfy
#: this test is to move the number into an injected block, not to widen this tuple.
ALLOWED_MATCHES: tuple[str, ...] = (
    # Semantic versions and the interpreter, both of which the badges block injects anyway.
    "1.0.0",
    "1.1.0",
    "2.0.0",
    "3.14",
)


def _strip_fenced_code(text: str) -> str:
    """Remove fenced blocks. Shell commands and mermaid are not prose and are not claims."""
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def _strip_injected(text: str) -> str:
    """Remove every marker pair and its contents, leaving only the hand written prose."""
    for name in BLOCKS:
        begin = re.escape(MARKER_BEGIN.format(name=name))
        end = re.escape(MARKER_END.format(name=name))
        text = re.sub(rf"{begin}.*?{end}", "", text, flags=re.DOTALL)
    return text


def prose_outside_markers(text: str) -> str:
    """The part of the README a human wrote and a generator does not own."""
    return _strip_fenced_code(_strip_injected(text))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed(repo_root) -> str:
    path = repo_root / README
    assert path.is_file(), f"{README} is not committed at {path}"
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def flat(committed) -> str:
    """The README with every run of whitespace collapsed.

    The injector wraps its prose blocks at a fixed width, so a two token claim like a file size
    can land with a line break inside it. The wrapping is presentation of the source, and a test
    that asserts on content has no business seeing it.
    """
    return re.sub(r"\s+", " ", committed)


@pytest.fixture(scope="module")
def regenerated(repo_root):
    try:
        return generate(repo_root)
    except ArtifactMissing as err:
        pytest.skip(f"the pipeline has not run: {err}")


@pytest.fixture(scope="module")
def sources(repo_root):
    """The artifacts the README quotes, loaded independently of the generator."""
    from ufem.config import config_hash, load_config
    from ufem.manifest import stage_dir

    config = load_config(repo_root)
    digest = config_hash(config)
    artifact_root = repo_root / config.pipeline.paths.artifact_root

    def read(stage: str, filename: str):
        path = stage_dir(artifact_root, stage, digest) / filename
        if not path.is_file():
            pytest.skip(f"the {stage} stage has not run for this config hash")
        return json.loads(path.read_text(encoding="utf-8"))

    from ufem.audit import CENSORING_JSON
    from ufem.audit import STAGE_NAME as AUDIT_STAGE
    from ufem.calibrate import CALIBRATION_JSON
    from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
    from ufem.propagate import PROPAGATION_JSON
    from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE
    from ufem.sensitivity import SENSITIVITY_JSON
    from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE
    from ufem.validate import BASELINES_JSON
    from ufem.validate import STAGE_NAME as VALIDATE_STAGE

    return {
        "censoring": read(AUDIT_STAGE, CENSORING_JSON),
        "baselines": read(VALIDATE_STAGE, BASELINES_JSON),
        "calibration": read(CALIBRATE_STAGE, CALIBRATION_JSON),
        "sensitivity": read(SENSITIVITY_STAGE, SENSITIVITY_JSON),
        "propagation": read(PROPAGATE_STAGE, PROPAGATION_JSON),
    }


# ---------------------------------------------------------------------------
# Group 1: the staleness gate
# ---------------------------------------------------------------------------


def test_the_generator_owns_exactly_the_readme(regenerated):
    assert set(regenerated) == {README}


def test_the_committed_readme_is_byte_identical_to_a_fresh_injection(committed, regenerated):
    fresh = regenerated[README].replace("\r\n", "\n")
    assert committed == fresh, (
        f"{README} is stale: it differs from what `python scripts/readme_inject.py` produces "
        "now. Rerun the injector and commit the result. Do not edit an injected block by hand; "
        "every number in one comes from an artifact."
    )


def test_the_injection_is_idempotent(repo_root, regenerated):
    """Injecting twice must not move anything, or the gate would fire on its own output."""
    from ufem.config import load_config

    once = regenerated[README]
    data = readme_inject.collect(repo_root, load_config(repo_root))
    assert readme_inject.inject(once, data) == once


# ---------------------------------------------------------------------------
# Group 2: the marker contract and the prose rule. No artifacts needed.
# ---------------------------------------------------------------------------


def test_every_named_marker_pair_is_present_exactly_once(committed):
    for name in BLOCKS:
        begin = MARKER_BEGIN.format(name=name)
        end = MARKER_END.format(name=name)
        assert committed.count(begin) == 1, f"{begin} appears {committed.count(begin)} times"
        assert committed.count(end) == 1, f"{end} appears {committed.count(end)} times"
        assert committed.index(begin) < committed.index(end), name


def test_no_injected_block_is_empty(committed):
    """An empty block means the injector never ran, and the page would be silently wrong."""
    for name in BLOCKS:
        begin = MARKER_BEGIN.format(name=name)
        end = MARKER_END.format(name=name)
        body = committed[committed.index(begin) + len(begin) : committed.index(end)]
        assert body.strip(), f"the {name} block is empty; run scripts/readme_inject.py"


def test_the_prose_outside_the_markers_carries_no_numeric_claim(committed):
    """Ground rule 10, enforced where it actually gets broken.

    Every measurement in this README lives inside a marker pair whose contents a generator owns.
    A number in the surrounding prose is by construction a hand typed one: nothing regenerates
    it, nothing checks it, and a reader cannot tell it apart from the injected ones.
    """
    prose = prose_outside_markers(committed)
    hits = []
    for pattern in (UNIT_CLAIM, PRECISE_DECIMAL):
        for found in pattern.finditer(prose):
            matched = found.group().strip()
            if any(allowed in matched for allowed in ALLOWED_MATCHES):
                continue
            line = prose[: found.start()].count("\n") + 1
            context = prose.splitlines()[line - 1].strip()
            hits.append(f"{matched!r} in {context!r}")
    assert not hits, (
        "numeric claims found in README prose outside the injected markers:\n  "
        + "\n  ".join(hits)
        + "\nMove each one into the block that owns it in scripts/readme_inject.py, or "
        "reword the sentence so it does not carry a measurement."
    )


def test_the_readme_has_no_phase_status_table(committed):
    """Phase status is development scaffolding and belongs in the engineering log.

    It was removed at P10 for a reason worth keeping: a table saying which phase is complete
    ages the moment the project ships, and a reader arriving at a release does not need the
    build order to evaluate the result.
    """
    assert "## Status" not in committed
    assert "| Phase | What it delivers | State |" not in committed
    assert "docs/ENGINEERING_LOG.md" in committed


def test_every_media_image_the_readme_shows_is_committed(repo_root, committed):
    referenced = set(re.findall(r"\((docs/media/[^)\s]+)\)", committed))
    assert referenced, "the README references no media at all"
    for relative in sorted(referenced):
        assert (repo_root / relative).is_file(), (
            f"{relative} is referenced by the README but is not in the tree. Run "
            "`python scripts/make_readme_media.py`."
        )


def test_every_repository_document_the_readme_links_is_present(repo_root, committed):
    for relative in sorted(set(re.findall(r"\]\((docs/[A-Z_]+\.md)\)", committed))):
        assert (repo_root / relative).is_file(), relative


def test_every_relative_link_resolves_to_something_in_the_tree(repo_root, committed):
    """A broken relative link is the cheapest kind of broken promise a README can make.

    Wider than the check above on purpose: it catches a path that was moved, a directory
    README that was never written, and a link typed against the working directory rather than
    against the repository root, which is the way these break on GitHub.
    """
    targets = {
        target
        for target in re.findall(r"\]\(([^)\s]+)\)", committed)
        if "://" not in target and not target.startswith("#")
    }
    assert targets, "the README links to nothing in its own tree"
    missing = sorted(target for target in targets if not (repo_root / target).exists())
    assert not missing, f"README links that do not resolve in the tree: {missing}"


def _heading_anchors(text: str) -> set[str]:
    """GitHub's heading slugs: lowercased, punctuation dropped, spaces to hyphens."""
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.*)$", text, flags=re.MULTILINE):
        plain = re.sub(r"[`*_]", "", heading).strip().lower()
        anchors.add(re.sub(r"[^a-z0-9 -]", "", plain).replace(" ", "-"))
    return anchors


def test_every_anchor_link_points_at_a_heading_that_exists(committed):
    """An in page link to a section that was renamed is a link that silently does nothing."""
    anchors = _heading_anchors(committed)
    used = {target.lstrip("#") for target in re.findall(r"\]\((#[^)\s]+)\)", committed)}
    assert used, "the README has no in page navigation at all"
    missing = sorted(target for target in used if target not in anchors)
    assert not missing, (
        f"README anchors with no matching heading: {missing}. Known headings: {sorted(anchors)}"
    )


def test_the_first_screen_carries_the_links_a_reader_arrives_for(committed):
    """The report, the dashboard and the specification, before anything has to be scrolled.

    Measured as a character budget rather than as a line count, because the GIF and the badge
    row are each one long line: everything above this offset renders inside the first screen.
    """
    first_screen = committed[: committed.index("## What this is")]
    for needle in ("releases/download/", "#ufem-lab-the-local-dashboard", "docs/BUILD_SPEC.md"):
        assert needle in first_screen, (
            f"{needle!r} is not reachable from the first screen of the README"
        )


def test_the_readme_reaches_every_destination_a_reader_needs(committed):
    """One click to each place the project actually lives, in an obvious spot."""
    for needle, what in (
        ("/actions/workflows/ci.yml", "the CI run history"),
        ("/actions/workflows/report.yml", "the report build history"),
        ("/releases", "the releases page"),
        ("/releases/tag/v1.0.0", "the frozen predecessor release"),
        ("/issues", "the issue tracker"),
        ("(CONTRIBUTING.md)", "the contributing guide"),
        ("(LICENSE)", "the license"),
        ("(data/quarantine/README.md)", "the quarantine notice"),
        ("(v1_legacy/README.md)", "the frozen predecessor tree"),
        ("(docs/ARCHITECTURE.md)", "the architecture document"),
    ):
        assert needle in committed, f"the README does not link {what} ({needle})"


def test_the_release_asset_link_is_the_file_the_release_script_uploads(repo_root, committed):
    """The download link and the upload command cannot be two different filenames.

    ``gh`` names an asset after the file on disk, so a README pointing at one name while the
    release script attaches another is a permanent 404 that nobody notices until somebody
    clicks it. Both come from :func:`make_release.report_asset_name`, and this is the assertion
    that keeps them there.
    """
    import tomllib as toml

    from make_release import report_asset_name

    metadata = toml.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(metadata["project"]["version"]).split(".dev")[0]
    assert f"/releases/download/v{version}/{report_asset_name(version)}" in committed


def test_the_readme_shows_the_dashboard_gif_first(repo_root, committed):
    """Build spec 15.1: the GIF is at the top of the README, and it is the real one."""
    gif = "docs/media/ufem_lab.gif"
    assert gif in committed
    assert (repo_root / gif).is_file()
    assert committed.index(gif) < committed.index("## What this is")


def test_the_schematic_is_a_mermaid_flowchart_with_the_named_subgraphs(committed):
    begin = MARKER_BEGIN.format(name="schematic")
    end = MARKER_END.format(name="schematic")
    body = committed[committed.index(begin) : committed.index(end)]
    assert "```mermaid" in body
    assert "flowchart TD" in body
    for subgraph in ("inherit", "surro", "calib", "evid", "prop", "prods"):
        assert f"subgraph {subgraph}[" in body, subgraph


def test_the_readme_carries_no_dash_the_lint_bans(committed):
    """Ground rule 3, asserted here as well so a README only change cannot slip it through."""
    assert chr(0x2014) not in committed
    assert chr(0x2013) not in committed


def test_the_badge_row_matches_the_declared_version_and_interpreter(repo_root, flat):
    """The badges are a claim about the project, so they come from pyproject.toml."""
    metadata = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    python = metadata["project"]["requires-python"].lstrip(">=~^ ")
    assert f"python-{python}-blue" in flat
    assert "license-MIT-green" in flat
    assert "/actions/workflow/status/" in flat
    assert "/github/v/release/" in flat


# ---------------------------------------------------------------------------
# Group 3: every injected number against its source, recomputed here
# ---------------------------------------------------------------------------


def test_the_scope_paragraph_matches_the_censoring_artifact(flat, sources):
    censoring = sources["censoring"]
    for count in ("n_designed", "n_valid", "n_failed"):
        assert str(int(censoring[count])) in flat, count


def test_the_results_table_matches_the_baselines_artifact(flat, sources):
    gate = sources["baselines"]["gate"]
    for target in gate["headline_qoi"]:
        record = gate["per_target"][target]
        assert f"{float(record['gp_r2_test']):.4f}" in flat, target
        assert f"{float(record['best_baseline'][1]):.4f}" in flat, target


def test_the_curve_row_names_the_baselines_that_beat_the_surrogate(flat, sources):
    """Definition of done item 4: where the surrogate loses, the README says to what.

    Read off the artifact rather than off the generator, and asserted as a non empty set,
    because the honest reporting only means something while there is something to report.
    """
    rows = [row for row in sources["baselines"]["curve"] if row["signal"] == "force"]
    by_model = {row["model"]: float(row["test"]["p50"]) for row in rows}
    surrogate = by_model["gaussian_process"]
    beaten_by = {name for name, value in by_model.items() if value < surrogate}
    assert beaten_by, (
        "no baseline beats the surrogate on the force curve any more. That is good news and "
        "this test and the README's caveat both have to be revisited rather than one of them."
    )
    labels = {"nearest_neighbour": "nearest neighbour", "quadratic_chaos": "quadratic chaos"}
    for name in beaten_by:
        assert labels.get(name, name) in flat, name
    assert f"{100.0 * surrogate:.2f} percent" in flat


def test_the_coverage_line_matches_the_calibration_gate(flat, sources):
    gate = sources["calibration"]["gate"]
    assert gate["passed"], "the calibration gate no longer passes; the README says it does"
    counted = [check for check in gate["checks"] if check.get("wilson") is not None]
    measured = min(float(check["measured"]) for check in counted)
    assert f"{measured:.4f}" in flat
    assert f"{1.0 - float(gate['alpha']):.2f}" in flat
    low = min(float(check["wilson"][0]) for check in counted)
    high = max(float(check["wilson"][1]) for check in counted)
    assert f"[{low:.4f}, {high:.4f}]" in flat


def test_the_reliability_line_matches_the_propagation_artifact(flat, sources):
    propagation = sources["propagation"]
    limit = next(state for state in propagation["limit_states"] if state["resolvable"])
    assert f"{float(limit['pf_point']):.4f}" in flat
    assert f"{float(limit['pf_standard_error']):.5f}" in flat
    assert f"{float(limit['pf_conservative']):.4f}" in flat
    assert str(int(propagation["context"]["n_samples"])) in flat
    fraction = 100.0 * float(propagation["validity"]["out_of_domain_fraction"])
    assert f"{fraction:.1f} percent" in flat


def test_the_readme_reports_the_sensitivity_indices_as_withheld(flat, sources):
    """P6 withheld every index. A README that quoted one would be the whole failure."""
    from ufem.sensitivity import PUBLICATION_WITHHELD

    sensitivity = sources["sensitivity"]
    assert sensitivity["gate_outcome"] == PUBLICATION_WITHHELD
    assert str(int(sensitivity["publication_counts"]["not_published"])) in flat
    assert "No sensitivity index is published" in flat


def test_the_readme_labels_the_predecessor_numbers_invalid(flat):
    """Definition of done item 12, in the one document a stranger reads first."""
    section = flat.split("## Versioning", 1)[1].split("\n## ", 1)[0]
    assert "v1.0.0" in section
    assert "invalid" in section
    assert "docs/BUILD_SPEC.md" in section


def test_the_test_count_the_readme_publishes_is_the_count_of_this_tree(repo_root, flat):
    counted = readme_inject._test_suite_size(repo_root)
    assert str(counted["functions"]) in flat
    assert str(counted["modules"]) in flat


def test_the_file_size_limit_the_readme_publishes_is_the_one_the_gate_enforces(flat):
    from check_file_sizes import LIMIT_BYTES

    assert f"{LIMIT_BYTES // (1024 * 1024)} MB" in flat


def test_the_binding_laws_are_quoted_from_the_build_specification(repo_root, flat):
    for title in readme_inject._binding_laws(repo_root):
        assert f"**{title}.**" in flat, title
