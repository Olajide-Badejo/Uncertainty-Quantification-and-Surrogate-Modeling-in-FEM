"""Inject every numeric claim in ``README.md`` from the artifact store and the repository.

Binding law 5 and ground rule 10: no number appears in the README unless it is reproducible
from a committed manifest whose hashes resolve to real files, and a README claim that
disagrees with its source is a CI failure rather than a documentation bug. This script is
what makes that mechanical. It reads the artifact store, the project metadata, the build
specification and the test tree, renders one block of markdown per named marker pair, and
splices those blocks into ``README.md`` between the markers. Nothing outside a marker pair is
touched, and nothing inside one is written by hand.

The pattern is the data card's and the model card's, deliberately, down to the staleness gate:
``tests/test_readme_consistency.py`` regenerates the README and compares it byte for byte, so a
README that has drifted from the pipeline is a failing test rather than a page nobody rechecked.
That test also refuses a numeric claim in the prose *outside* the markers, which is the half of
the rule a byte comparison cannot enforce: an injected block that is correct is worth nothing if
somebody types a second, staler number two paragraphs above it.

The markers, and what each one is allowed to say:

``badges``       the shields row. Version and interpreter come from ``pyproject.toml``.
``scope``        what the pipeline was built from: the campaign, the split, the inputs.
``schematic``    the mermaid diagram, with its counts substituted from the artifacts.
``results``      the out of sample table: surrogate against the best baseline, per quantity.
``coverage``     the measured calibration coverage with its Wilson interval.
``reliability``  the headline failure probability with its error bar and its bound.
``evidence``     the ablations, and how many of their committed predictions survived.
``caveats``      the three sentences that say what those numbers do not claim.
``laws``         the five binding laws, with their titles read out of the build specification.
``quickstart``   the regeneration budget.
``versioning``   the two releases and the status of the predecessor's numbers.
``gates``        the file size limit and the size of the test suite.

One quantity here is deliberately coarse. The full pipeline wall time is a sum of stage wall
times from the manifests, and a stage rewrites its wall time every time it is rerun while
reproducing its outputs bitwise: that is exactly the quantity ``docs/DEFECT_LOG.md`` records as
having made the model card stale twice, once as a wall time and once as a git commit. A byte
gated document carries only quantities that change when the numbers change, so the README states
the measured total rounded up to the enclosing five minutes, which moves only when the cost of
the pipeline genuinely moves. The exact per stage wall times stay in ``docs/ENGINEERING_LOG.md``,
which is written by hand and gated by nothing.

Exit 0 is clean, exit 1 names the failure.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any, Callable

from check_file_sizes import LIMIT_BYTES
from ufem.ablation_table import AblationMissing, verdict_summary
from ufem.ablation_table import load_payloads as load_ablation_payloads
from ufem.audit import CENSORING_JSON, VALIDITY_DOMAIN_JSON
from ufem.audit import STAGE_NAME as AUDIT_STAGE
from ufem.calibrate import CALIBRATION_JSON, GATE_ALPHA, SIGNAL_FORCE
from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import Config, config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.propagate import PROPAGATION_JSON, QOI_DISPLAY
from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE
from ufem.sensitivity import SENSITIVITY_JSON
from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.surrogate import SURROGATE_JSON
from ufem.validate import BASELINES_JSON, QOI_LABELS
from ufem.validate import STAGE_NAME as VALIDATE_STAGE

README = "README.md"
BUILD_SPEC = "docs/BUILD_SPEC.md"

#: ``owner/repo`` on GitHub, which is what the shields.io badge paths are built from. It is a
#: location rather than a measurement, so it lives here rather than in an artifact.
GITHUB_SLUG = "Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM"

#: The stages the README reads, with the command that produces each.
REQUIRED_STAGES: tuple[tuple[str, str], ...] = (
    (AUDIT_STAGE, "ufem run audit"),
    (SURROGATE_STAGE, "ufem run surrogate"),
    (VALIDATE_STAGE, "ufem run validate"),
    (CALIBRATE_STAGE, "ufem run calibrate"),
    (SENSITIVITY_STAGE, "ufem run sensitivity"),
    (PROPAGATE_STAGE, "ufem run propagate"),
)

#: Every stage whose wall time counts toward a full regeneration, in pipeline order.
PIPELINE_STAGES: tuple[str, ...] = (
    "ingest",
    "grid",
    AUDIT_STAGE,
    "register",
    "reduce",
    SURROGATE_STAGE,
    VALIDATE_STAGE,
    CALIBRATE_STAGE,
    SENSITIVITY_STAGE,
    PROPAGATE_STAGE,
)

#: Build spec section 2 budgets a full regeneration at half an hour. The README quotes the
#: measured total against it, so the budget is read out of the specification rather than typed.
WALL_TIME_BUDGET_PATTERN = re.compile(r"in under (\d+) minutes", re.IGNORECASE)

#: The wall time claim is rounded up to a multiple of this many minutes. See the module
#: docstring: a byte gated document carries only quantities that move when a measurement moves.
WALL_TIME_BUCKET_MINUTES = 5

#: Display names for the baseline models, because ``nearest_neighbour`` is a key, not English.
BASELINE_LABELS: dict[str, str] = {
    "climatology": "climatology",
    "linear": "linear",
    "nearest_neighbour": "nearest neighbour",
    "quadratic_chaos": "quadratic chaos",
}

MARKER_BEGIN = "<!-- BEGIN INJECTED: {name} -->"
MARKER_END = "<!-- END INJECTED: {name} -->"

#: The README is wrapped at this width so a diff of it is readable. Markdown joins the lines
#: back together, so the wrapping is presentation of the source and nothing else.
LINE_WIDTH = 96

#: Blocks that are one paragraph of prose, and blocks that are a list. Everything else (the
#: badge row, the mermaid diagram, the results table) has line breaks that carry meaning and is
#: spliced exactly as its builder produced it.
PROSE_BLOCKS = frozenset(
    {"scope", "coverage", "reliability", "evidence", "quickstart", "gates"}
)
LIST_BLOCKS = frozenset({"caveats", "laws", "versioning"})

RE_LIST_MARKER = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)")


class ArtifactMissing(RuntimeError):
    """A source this script reads from is absent, and no number will be invented."""


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _read_json(path: Path, role: str, how: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMissing(
            f"cannot inject the README: the {role} at {path} does not exist. Run `{how}`; "
            "this script reads artifacts and never recomputes them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _project_metadata(root: Path) -> dict[str, str]:
    """The version and the interpreter, from the one file that declares them."""
    path = root / "pyproject.toml"
    if not path.is_file():
        raise ArtifactMissing(f"cannot inject the README: no pyproject.toml at {path}.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data["project"]
    version = str(project["version"])
    release = version.split(".dev")[0]
    requires = str(project["requires-python"])
    python = requires.lstrip(">=~^ ")
    return {"version": version, "release": release, "python": python}


def _binding_laws(root: Path) -> list[str]:
    """The five law titles, read out of build spec section 0.2 rather than restated.

    The titles are the bolded lead of each numbered item. Reading them here means a law that is
    reworded in the specification and not in the README is a failing test, which is the only way
    a quoted list stays a quote.
    """
    path = root / BUILD_SPEC
    if not path.is_file():
        raise ArtifactMissing(f"cannot inject the README: no build specification at {path}.")
    text = path.read_text(encoding="utf-8")
    if "### 0.2 The five binding laws" not in text:
        raise ArtifactMissing(
            f"{BUILD_SPEC} has no section 0.2; the README quotes the laws from there."
        )
    section = text.split("### 0.2 The five binding laws", 1)[1].split("\n### ", 1)[0]
    titles = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*", section, flags=re.MULTILINE)
    if len(titles) != 5:
        raise ArtifactMissing(
            f"{BUILD_SPEC} section 0.2 yielded {len(titles)} law titles, not five. The README "
            "quotes that section and will not guess at it."
        )
    return [title.rstrip(".") for title in titles]


def _wall_time_budget_minutes(root: Path) -> int:
    """The regeneration budget of build spec section 23, item 1."""
    text = (root / BUILD_SPEC).read_text(encoding="utf-8")
    section = text.split("## 23. Definition of done", 1)
    if len(section) != 2:
        raise ArtifactMissing(f"{BUILD_SPEC} has no section 23; the budget is quoted from it.")
    found = WALL_TIME_BUDGET_PATTERN.search(section[1])
    if found is None:
        raise ArtifactMissing(
            f"{BUILD_SPEC} section 23 states no regeneration budget in minutes; the README "
            "quotes it and will not invent one."
        )
    return int(found.group(1))


def _test_suite_size(root: Path) -> dict[str, int]:
    """Test functions and test modules, counted by parsing rather than by running.

    Parsed for the same reason ``scripts/dash_lint.py`` parses the UI package: a regular
    expression cannot tell a test function from the word ``test`` in a docstring, and this count
    is a published claim. It counts declarations, not the cases pytest expands them into, and the
    README says so.
    """
    directory = root / "tests"
    if not directory.is_dir():
        raise ArtifactMissing(f"cannot inject the README: no test tree at {directory}.")
    modules = sorted(directory.glob("test_*.py"))
    if not modules:
        raise ArtifactMissing(f"{directory} holds no test modules; the count would be a zero.")
    functions = 0
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_function and node.name.startswith("test_"):
                functions += 1
    return {"functions": functions, "modules": len(modules)}


def collect(root: Path, config: Config) -> dict[str, Any]:
    """Every source the README is built from, loaded once."""
    digest = config_hash(config)
    artifact_root = root / config.pipeline.paths.artifact_root
    directories: dict[str, Path] = {}
    for stage, how in REQUIRED_STAGES:
        directory = stage_dir(artifact_root, stage, digest)
        if not (directory / "manifest.json").is_file():
            raise ArtifactMissing(
                f"the {stage} stage has no manifest at {directory}. Run `{how}` for config "
                f"{digest[:12]} before injecting the README."
            )
        directories[stage] = directory
    wall_times: dict[str, float] = {}
    for stage in PIPELINE_STAGES:
        directory = stage_dir(artifact_root, stage, digest)
        if not (directory / "manifest.json").is_file():
            raise ArtifactMissing(
                f"the {stage} stage has no manifest at {directory}, so the regeneration wall "
                "time cannot be summed. Run `ufem run all`."
            )
        wall_times[stage] = float(load_manifest(directory)["wall_time_s"])
    try:
        ablations = load_ablation_payloads(artifact_root, digest)
    except AblationMissing as err:
        raise ArtifactMissing(str(err)) from err
    return {
        "config_sha256": digest,
        "project": _project_metadata(root),
        "laws": _binding_laws(root),
        "wall_time_budget_min": _wall_time_budget_minutes(root),
        "tests": _test_suite_size(root),
        "wall_times": wall_times,
        "censoring": _read_json(
            directories[AUDIT_STAGE] / CENSORING_JSON, "censoring statistics", "ufem run audit"
        ),
        "domain": _read_json(
            directories[AUDIT_STAGE] / VALIDITY_DOMAIN_JSON, "validity domain", "ufem run audit"
        ),
        "surrogate": _read_json(
            directories[SURROGATE_STAGE] / SURROGATE_JSON, "surrogate record",
            "ufem run surrogate",
        ),
        "baselines": _read_json(
            directories[VALIDATE_STAGE] / BASELINES_JSON, "baselines", "ufem run validate"
        ),
        "calibration": _read_json(
            directories[CALIBRATE_STAGE] / CALIBRATION_JSON, "calibration", "ufem run calibrate"
        ),
        "sensitivity": _read_json(
            directories[SENSITIVITY_STAGE] / SENSITIVITY_JSON, "sensitivity",
            "ufem run sensitivity",
        ),
        "propagation": _read_json(
            directories[PROPAGATE_STAGE] / PROPAGATION_JSON, "propagation", "ufem run propagate"
        ),
        "ablations": verdict_summary(ablations),
        "ablation_payloads": ablations,
    }


# ---------------------------------------------------------------------------
# Derived quantities, each computed once and used by more than one block
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def headline_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per headline scalar: the surrogate, the best baseline, and the verdict."""
    gate = data["baselines"]["gate"]
    rows = []
    for target in gate["headline_qoi"]:
        record = gate["per_target"][target]
        best_model, best_r2 = record["best_baseline"]
        rows.append(
            {
                "target": target,
                "label": QOI_LABELS[target],
                "unit": QOI_DISPLAY[target][0],
                "surrogate": float(record["gp_r2_test"]),
                "baseline": float(best_r2),
                "baseline_model": BASELINE_LABELS.get(best_model, best_model),
                "beats_all": bool(record["beats_all"]),
                "lost_to": [BASELINE_LABELS.get(name, name) for name in record["lost_to"]],
            }
        )
    return rows


def curve_row(data: dict[str, Any], signal: str = SIGNAL_FORCE) -> dict[str, Any]:
    """The whole curve comparison for one signal, which is where the pipeline loses.

    Reported at the median rather than the mean of the per run relative L2, because the mean of
    a ratio over 198 runs is carried by the few runs whose denominator is small, and the report
    quotes the median for the same reason.
    """
    records = [row for row in data["baselines"]["curve"] if row["signal"] == signal]
    if not records:
        raise ArtifactMissing(f"the baselines artifact has no curve rows for the {signal} signal")
    by_model = {row["model"]: float(row["test"]["p50"]) for row in records}
    surrogate = by_model["gaussian_process"]
    rivals = {name: value for name, value in by_model.items() if name != "gaussian_process"}
    beaten_by = sorted(name for name, value in rivals.items() if value < surrogate)
    best_model = min(rivals, key=lambda name: rivals[name])
    return {
        "signal": signal,
        "surrogate": surrogate,
        "baseline": rivals[best_model],
        "baseline_model": BASELINE_LABELS.get(best_model, best_model),
        "beaten_by": [BASELINE_LABELS.get(name, name) for name in beaten_by],
        "n_beaten_by": len(beaten_by),
    }


def coverage_checks(data: dict[str, Any]) -> dict[str, Any]:
    """The calibration gate reduced to the one line a README can carry.

    Every check in the gate measured the same coverage on the same 198 leave one out folds, so
    the line states that value once and says how many criteria it covers. If a future run
    disagrees between criteria, the minimum and the maximum diverge and the sentence changes
    shape rather than quietly reporting the first one.
    """
    gate = data["calibration"]["gate"]
    counted = [check for check in gate["checks"] if check.get("wilson") is not None]
    if not counted:
        raise ArtifactMissing("the calibration gate has no counted coverage criterion")
    measured = [float(check["measured"]) for check in counted]
    lows = [float(check["wilson"][0]) for check in counted]
    highs = [float(check["wilson"][1]) for check in counted]
    return {
        "n_checks": len(gate["checks"]),
        "n_counted": len(counted),
        "identical": max(measured) - min(measured) < 1e-12,
        "measured_min": min(measured),
        "measured_max": max(measured),
        "wilson_low": min(lows),
        "wilson_high": max(highs),
        "nominal": 1.0 - float(gate["alpha"]),
        "passed": bool(gate["passed"]),
    }


def headline_limit_state(data: dict[str, Any]) -> dict[str, Any]:
    """The peak load limit state, which is the one the reliability sentence reports."""
    states = data["propagation"]["limit_states"]
    resolvable = [state for state in states if state["resolvable"]]
    if not resolvable:
        raise ArtifactMissing("no limit state in the propagation artifact is above the floor")
    return resolvable[0]


def regeneration_minutes(data: dict[str, Any]) -> dict[str, Any]:
    """The summed pipeline wall time, and the coarse bucket the README is allowed to quote."""
    total = sum(data["wall_times"].values())
    minutes = total / 60.0
    bucket = int(math.ceil(minutes / WALL_TIME_BUCKET_MINUTES) * WALL_TIME_BUCKET_MINUTES)
    return {"seconds": total, "minutes": minutes, "bucket": bucket}


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------


def block_badges(data: dict[str, Any]) -> str:
    project = data["project"]
    shields = "https://img.shields.io"
    parts = [
        f"[![CI]({shields}/github/actions/workflow/status/{GITHUB_SLUG}/ci.yml"
        f"?branch=main&label=CI)](https://github.com/{GITHUB_SLUG}/actions/workflows/ci.yml)",
        f"[![report]({shields}/github/actions/workflow/status/{GITHUB_SLUG}/report.yml"
        f"?branch=main&label=report)]"
        f"(https://github.com/{GITHUB_SLUG}/actions/workflows/report.yml)",
        f"[![release]({shields}/github/v/release/{GITHUB_SLUG}?label=release)]"
        f"(https://github.com/{GITHUB_SLUG}/releases)",
        f"[![python]({shields}/badge/python-{project['python']}-blue)]"
        "(https://www.python.org/downloads/)",
        f"[![license]({shields}/badge/license-MIT-green)](LICENSE)",
    ]
    return "\n".join(parts)


def block_scope(data: dict[str, Any]) -> str:
    censoring = data["censoring"]
    surrogate = data["surrogate"]
    domain = data["domain"]
    n_inputs = len(surrogate["feature_order"])
    return (
        f"The evidence is a completed Abaqus concrete damaged plasticity campaign: "
        f"{int(censoring['n_designed'])} designed simulations, of which "
        f"{int(censoring['n_valid'])} produced a usable response and "
        f"{int(censoring['n_failed'])} produced nothing at all. The surrogate is fitted on those "
        f"{int(surrogate['n_training_runs'])} runs, over {n_inputs} independent inputs, and it "
        f"predicts the whole load displacement and damage evolution curve rather than a summary "
        f"of it. The failures are not spread evenly across the design, so every product carries "
        f"a machine checked validity domain that covers "
        f"{_fmt(100.0 * float(domain['dense_inside_fraction']), 1)} percent of the design box "
        f"and excludes the corner where the campaign died."
    )


def block_schematic(data: dict[str, Any]) -> str:
    censoring = data["censoring"]
    surrogate = data["surrogate"]
    counts = surrogate["component_counts"]
    calibration = data["calibration"]
    propagation = data["propagation"]
    coverage = coverage_checks(data)
    n_ablations = len(data["ablations"])
    n_grid = int(calibration["context"]["n_grid"])
    n_baselines = len({row["model"] for row in data["baselines"]["scalar"]}) - 1
    n_withheld = int(data["sensitivity"]["publication_counts"]["not_published"])
    limit = headline_limit_state(data)
    lines = [
        "```mermaid",
        "flowchart TD",
        '    subgraph inherit["Data inheritance (ingest, grid, audit)"]',
        f'        RAW["Abaqus campaign CSVs<br/>{int(censoring["n_designed"])} designed runs, '
        'read only"]',
        f'        QGATE["Quality gate and audit<br/>{int(censoring["n_valid"])} valid, '
        f'{int(censoring["n_failed"])} produced nothing"]',
        '        CENS["Censoring model<br/>completion probability per design point"]',
        '        DOM["Validity domain<br/>the censored corner is excluded, not caveated"]',
        "    end",
        '    subgraph surro["Functional surrogate (register, reduce, surrogate)"]',
        f'        GRIDN["Common displacement grid<br/>force and damage on {n_grid} abscissae"]',
        '        SRVF["Landmarks and SRVF registration<br/>amplitude separated from phase"]',
        f'        PCA["Dual functional PCA<br/>{int(counts["amplitude"])} amplitude and '
        f'{int(counts["phase"])} phase components"]',
        '        GPS["Matern Gaussian processes<br/>one per score and per scalar"]',
        "    end",
        '    subgraph calib["Calibrated uncertainty (calibrate)"]',
        '        LOO["Closed form leave one out<br/>every fold refits its own basis"]',
        '        JKP["Jackknife plus intervals<br/>sigma normalized conformal scores"]',
        '        BAND["Simultaneous sup norm bands<br/>the whole curve, not one abscissa"]',
        f'        CGATE["Calibration gate<br/>measured coverage '
        f'{_fmt(coverage["measured_min"], 4)} against nominal '
        f'{_fmt(coverage["nominal"], 2)}"]',
        "    end",
        '    subgraph evid["Evidence (validate, ablations, sensitivity)"]',
        f'        BASE["{n_baselines} dumb baselines<br/>the surrogate reports its losses '
        'too"]',
        f'        ABL["{n_ablations} ablations<br/>every prediction committed before its '
        'measurement"]',
        '        MANU["Manufactured solution<br/>error falls at the expected rate"]',
        f'        QTWO["Sensitivity Q2 gate<br/>{n_withheld} indices withheld, none published"]',
        "    end",
        '    subgraph prop["Propagation (propagate, analytic)"]',
        f'        MC["Monte Carlo<br/>{int(propagation["context"]["n_samples"])} draws through '
        'the calibrated surrogate"]',
        f'        LSTATE["Limit states<br/>{len(propagation["limit_states"])} thresholds, '
        'aleatory and epistemic kept apart"]',
        f'        PF["Failure probability<br/>{_fmt(limit["pf_point"], 4)} point estimate with '
        'an error bar and a bound"]',
        '        XCHK["Analytic cross check<br/>independent mechanics model, not a '
        'transcription"]',
        "    end",
        '    subgraph prods["Products"]',
        '        LAB["UFEM Lab dashboard<br/>five panels, every number read from an artifact"]',
        '        REP["LaTeX report<br/>every figure and table generated by the pipeline"]',
        '        MAN["Manifests and CI<br/>content addressed hashes that resolve to real '
        'files"]',
        "    end",
        "    RAW --> QGATE --> CENS --> DOM",
        "    DOM --> GRIDN --> SRVF --> PCA --> GPS",
        "    GPS --> LOO --> JKP --> CGATE",
        "    LOO --> BAND --> CGATE",
        "    GPS --> BASE --> ABL",
        "    GPS --> MANU",
        "    GPS --> QTWO",
        "    CGATE --> MC --> LSTATE --> PF",
        "    PF --> XCHK",
        "    DOM --> MC",
        "    CGATE --> LAB",
        "    PF --> REP",
        "    ABL --> REP",
        "    QTWO --> REP",
        "    PF --> LAB",
        "    REP --> MAN",
        "    LAB --> MAN",
        "```",
    ]
    return "\n".join(lines)


def block_results(data: dict[str, Any]) -> str:
    rows = headline_rows(data)
    curve = curve_row(data)
    out = [
        "| Out of sample quantity | Surrogate | Best baseline | Verdict |",
        "|---|---|---|---|",
    ]
    for row in rows:
        verdict = (
            "beats every baseline"
            if row["beats_all"]
            else "loses to " + ", ".join(row["lost_to"])
        )
        out.append(
            f"| {row['label']}, leave one out test R2 | {_fmt(row['surrogate'], 4)} | "
            f"{_fmt(row['baseline'], 4)} ({row['baseline_model']}) | {verdict} |"
        )
    beaten = ", ".join(curve["beaten_by"])
    out.append(
        f"| Whole force curve, median relative L2 | "
        f"{_fmt(100.0 * curve['surrogate'], 2)} percent | "
        f"{_fmt(100.0 * curve['baseline'], 2)} percent ({curve['baseline_model']}) | "
        f"loses to {beaten} |"
    )
    return "\n".join(out)


def block_coverage(data: dict[str, Any]) -> str:
    coverage = coverage_checks(data)
    calibration = data["calibration"]
    band = calibration["functional"][SIGNAL_FORCE]["bands"][str(GATE_ALPHA)]
    same = (
        "every one of them measured the same coverage"
        if coverage["identical"]
        else f"coverage ran from {_fmt(coverage['measured_min'], 4)} to "
        f"{_fmt(coverage['measured_max'], 4)} across them"
    )
    return (
        f"**Calibration.** The gate of build spec 11.5 passed on all "
        f"{coverage['n_checks']} criteria. Across the {coverage['n_counted']} counted ones, "
        f"simultaneous sup norm bands on both curve families and jackknife plus intervals on "
        f"every headline scalar, {same}: "
        f"{_fmt(coverage['measured_min'], 4)} against a nominal "
        f"{_fmt(coverage['nominal'], 2)}, with a 95 percent Wilson interval of "
        f"[{_fmt(coverage['wilson_low'], 4)}, {_fmt(coverage['wilson_high'], 4)}] that contains "
        f"the nominal level. The band that achieves it is not free: its median half width on the "
        f"force curves is {_fmt(float(band['median_half_width']) / 1000.0, 2)} kN."
    )


def block_reliability(data: dict[str, Any]) -> str:
    limit = headline_limit_state(data)
    propagation = data["propagation"]
    unit, scale = QOI_DISPLAY[limit["target"]]
    validity = propagation["validity"]
    return (
        f"**Reliability.** A Monte Carlo of "
        f"{int(propagation['context']['n_samples'])} draws through the calibrated surrogate "
        f"puts the probability that the peak load falls below its "
        f"{_fmt(float(limit['threshold']) * scale, 1)} {unit} characteristic value at "
        f"{_fmt(limit['pf_point'], 4)}, binomial standard error "
        f"{_fmt(limit['pf_standard_error'], 5)}, against a surrogate aware conservative bound of "
        f"{_fmt(limit['pf_conservative'], 4)} obtained by counting a failure whenever the "
        f"calibrated interval crosses the threshold. "
        f"{_fmt(100.0 * float(validity['out_of_domain_fraction']), 1)} percent of the sample "
        f"fell outside the validity domain, and no probability below "
        f"{propagation['context']['resolvable_pf_floor']:g} is claimed at "
        f"{int(propagation['context']['n_training_runs'])} training runs."
    )


def block_evidence(data: dict[str, Any]) -> str:
    """What was measured against the pipeline, and how much of it survived contact.

    The count of surviving predictions is exactly the kind of claim that ages badly, because it
    moves the moment any one measurement does, which is why it is injected rather than typed.
    """
    ablations = data["ablations"]
    held = sum(record["n_held"] for record in ablations.values())
    total = sum(record["n_claims"] for record in ablations.values())
    spline = data["ablation_payloads"][4]
    return (
        f"**Evidence.** The design choices were measured rather than assumed: "
        f"{len(ablations)} ablations ran against the production pipeline in the same fold "
        f"harness, each one's prediction committed before its measurement existed so that the "
        f"commit order is the evidence, and {held} of the {total} committed claims held. "
        f"The sharpest single pair of numbers in them is the B-spline rival's peak load bias of "
        f"{float(spline['ablation']['peak']['peak_bias_N']):.0f} N against the shipped "
        f"pipeline's {float(spline['production']['peak']['peak_bias_N']):.0f} N on the same "
        f"folds: the direct curve model reconstructs better and predicts the peak far worse. "
        f"Every prediction, result and verdict is in `docs/ABLATIONS.md`."
    )


def block_caveats(data: dict[str, Any]) -> str:
    sensitivity = data["sensitivity"]
    counts = sensitivity["publication_counts"]
    limit = headline_limit_state(data)
    censoring = data["censoring"]
    curve = curve_row(data)
    ratio = float(limit["pf_conservative"]) / float(limit["pf_point"])
    return "\n".join(
        [
            f"- **No sensitivity index is published.** All {int(counts['not_published'])} sparse "
            f"chaos expansions failed their Q2 gate, so the ranking of the inputs this campaign "
            f"can support is nothing, and that is reported instead of a plausible bar chart.",
            f"- **The reliability bounds are dominated by surrogate error.** The conservative "
            f"bound is {_fmt(ratio, 1)} times the point estimate, which measures the width of "
            f"the calibrated interval rather than the fragility of the beam.",
            f"- **The design is censored.** {int(censoring['n_failed'])} of "
            f"{int(censoring['n_designed'])} runs produced nothing and the failures cluster, so "
            f"the survivors are a biased subsample and every number here is conditional on the "
            f"validity domain.",
            f"- **The whole curve is where the pipeline leaks.** The force curve is "
            f"reconstructed at a lower median relative L2 by {curve['n_beaten_by']} of the "
            f"baselines than by the registered and reduced surrogate, and the table above says "
            f"which.",
        ]
    )


def block_laws(data: dict[str, Any]) -> str:
    glosses = [
        "Every predictive interval is the output of a stated, tested procedure, and every "
        "calibration claim is verified by a coverage measurement with a confidence band.",
        "The input random variables are defined exactly once, in one validated config file, "
        "hashed into every artifact.",
        "Every reported metric is cross validated or held out, with the reduction basis "
        "recomputed inside each fold.",
        "The runs that failed are modeled, and everything downstream carries the domain they "
        "define.",
        "No number appears in the README, the report or the dashboard unless a committed "
        "manifest can regenerate it.",
    ]
    return "\n".join(
        f"{index}. **{title}.** {gloss}"
        for index, (title, gloss) in enumerate(zip(data["laws"], glosses), start=1)
    )


def block_quickstart(data: dict[str, Any]) -> str:
    timing = regeneration_minutes(data)
    return (
        f"A full regeneration from the raw campaign CSVs runs in under {timing['bucket']} "
        f"minutes on the reference machine, inside the {data['wall_time_budget_min']} minute "
        f"budget of build spec section 23. The per stage wall times are in "
        f"`docs/ENGINEERING_LOG.md`; they are deliberately not quoted here, because a document "
        f"gated on byte identity must not carry a quantity that moves without a measurement "
        f"moving."
    )


def block_versioning(data: dict[str, Any]) -> str:
    project = data["project"]
    return "\n".join(
        [
            "- **v1.0.0**, the frozen predecessor, preserved read only in `v1_legacy/`. "
            "**Its published metrics are invalid**, and the evidence is section 5 of "
            "`docs/BUILD_SPEC.md`: the reported uncertainty was manufactured by an "
            "amplification factor and a standard deviation floor, and its own diagnostic "
            "output proved it. Nothing from it is repeated as a result here.",
            f"- **v{project['release']}**, this overhaul, built from scratch on the same "
            f"inherited campaign behind the gates above. The in progress version is "
            f"`{project['version']}`, declared in `pyproject.toml` and reported by "
            f"`ufem doctor`.",
        ]
    )


def block_gates(data: dict[str, Any]) -> str:
    tests = data["tests"]
    limit_mb = LIMIT_BYTES // (1024 * 1024)
    return (
        f"A change lands only when the dash and banned identifier lint passes, no tracked file "
        f"exceeds {limit_mb} MB, `ruff check src tests scripts` is clean, and the suite of "
        f"{tests['functions']} test functions across {tests['modules']} modules passes. Those "
        f"are declarations rather than the cases pytest expands them into. All four gates run "
        f"in CI on every push to `main` and to any `phase/**` branch, and on every pull request."
    )


#: Marker name to builder. The order is the order they appear in the README, which is only a
#: convenience for reading this file; the splice is keyed by name.
BLOCKS: dict[str, Callable[[dict[str, Any]], str]] = {
    "badges": block_badges,
    "scope": block_scope,
    "schematic": block_schematic,
    "results": block_results,
    "coverage": block_coverage,
    "reliability": block_reliability,
    "evidence": block_evidence,
    "caveats": block_caveats,
    "laws": block_laws,
    "quickstart": block_quickstart,
    "versioning": block_versioning,
    "gates": block_gates,
}


# ---------------------------------------------------------------------------
# The splice
# ---------------------------------------------------------------------------


def marker_span(text: str, name: str) -> tuple[int, int]:
    """The character offsets between one marker pair, exclusive of the markers themselves."""
    begin = MARKER_BEGIN.format(name=name)
    end = MARKER_END.format(name=name)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ArtifactMissing(
            f"{README} must contain exactly one {begin} and one {end}; found "
            f"{text.count(begin)} and {text.count(end)}."
        )
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    if stop < start:
        raise ArtifactMissing(f"{README} has {end} before {begin}.")
    return start, stop


def _wrap(body: str, name: str) -> str:
    """Wrap a prose or list block at :data:`LINE_WIDTH`, with a hanging indent on list items."""
    if name in PROSE_BLOCKS:
        return "\n".join(
            textwrap.wrap(
                body, width=LINE_WIDTH, break_long_words=False, break_on_hyphens=False
            )
        )
    if name not in LIST_BLOCKS:
        return body
    out = []
    for line in body.split("\n"):
        marker = RE_LIST_MARKER.match(line)
        indent = " " * len(marker.group(1)) if marker else ""
        out.append(
            "\n".join(
                textwrap.wrap(
                    line,
                    width=LINE_WIDTH,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return "\n".join(out)


def inject(text: str, data: dict[str, Any]) -> str:
    """Splice every block into its marker pair, leaving everything else byte identical."""
    for name, builder in BLOCKS.items():
        start, stop = marker_span(text, name)
        body = _wrap(builder(data).rstrip("\n"), name)
        text = text[:start] + "\n\n" + body + "\n\n" + text[stop:]
    return text


def generate(root: Path) -> dict[str, str]:
    """Build the injected README and return a mapping of repo relative path to content."""
    path = root / README
    if not path.is_file():
        raise ArtifactMissing(f"{README} does not exist at {path}; there is nothing to inject.")
    config = load_config(root)
    data = collect(root, config)
    current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return {README: inject(current, data)}


def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    written = 0
    for relative, content in generate(root).items():
        changed = write_if_changed(root / relative, content)
        written += int(changed)
        print(f"{relative}: {'written' if changed else 'unchanged'} ({len(content)} bytes)")
    print(f"readme_inject: {written} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
