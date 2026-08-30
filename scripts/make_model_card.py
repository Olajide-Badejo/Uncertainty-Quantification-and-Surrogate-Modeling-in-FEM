"""Generate ``docs/MODEL_CARD.md`` from the artifact store, and from nothing else.

Build spec 19 asks the documentation set for a model card: surrogate scope, validity domain,
calibration status, known failure modes. Build spec 15 asks UFEM Lab for the same content as a
panel. This script and :mod:`ufem.ui.app` read the same artifacts, so the card a reader opens
in the repository and the panel a reader opens in the dashboard cannot disagree.

The pattern is the data card's, deliberately. Every number here came out of a Parquet or a
JSON that a stage wrote; there is not one hand typed measurement in this file; a stage that has
not run raises :class:`ArtifactMissing` naming the stage rather than leaving a gap; and
``tests/test_model_card.py`` regenerates the card and asserts byte identity against the
committed copy, so a card that has drifted from the pipeline is a failing test rather than a
document nobody rechecked.

Three things are deliberately absent, all for one reason: a generated document must not carry a
quantity that moves while the measurements it reports stay put, because a staleness gate that
fires on that gets muted, and a muted gate stops catching the drift it exists for.

- No wall time. It varies between runs on one machine, and the data card already learned this.
- No generation date, for the same reason.
- No git commit. Rerunning a stage rewrites its manifest with the current commit while
  reproducing its outputs bitwise, so a commit in the card would move on every determinism
  check. What the card carries instead is the config hash and the digest of the surrogate
  record, which move only when the model does, plus a pointer to the manifest that holds the
  rest of the chain.

Exit 0 is clean, exit 1 names the failure.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ufem.audit import CENSORING_JSON, COMPLETION_JSON, VALIDITY_DOMAIN_JSON
from ufem.audit import STAGE_NAME as AUDIT_STAGE
from ufem.calibrate import CALIBRATION_JSON
from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import FEATURE_ORDER, Config, config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.propagate import PROPAGATION_JSON, QOI_DISPLAY, RESOLVABLE_PF_FLOOR
from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE
from ufem.sensitivity import PUBLICATION_TEX, SENSITIVITY_JSON
from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.surrogate import SURROGATE_JSON
from ufem.validate import BASELINES_JSON, QOI_LABELS
from ufem.validate import STAGE_NAME as VALIDATE_STAGE

MODEL_CARD = "docs/MODEL_CARD.md"

#: The stages the card reads, with the command that produces each.
REQUIRED_STAGES: tuple[tuple[str, str], ...] = (
    (AUDIT_STAGE, "ufem run audit"),
    (SURROGATE_STAGE, "ufem run surrogate"),
    (VALIDATE_STAGE, "ufem run validate"),
    (CALIBRATE_STAGE, "ufem run calibrate"),
    (SENSITIVITY_STAGE, "ufem run sensitivity"),
    (PROPAGATE_STAGE, "ufem run propagate"),
)

#: Display names and units for the three inputs. Names, not numbers.
INPUT_LABELS: dict[str, tuple[str, str]] = {
    "Fcm_MPa": ("Mean compressive strength", "MPa"),
    "c_nom_bottom_mm": ("Bottom cover", "mm"),
    "c_nom_top_mm": ("Top cover", "mm"),
}

#: The committed report figures the card points a reader at, with what each one shows. Paths
#: and captions, no measurements: the numbers in those figures come from the same artifacts.
CALIBRATION_FIGURES: tuple[tuple[str, str], ...] = (
    (
        "report/figures/fig_calibration.pdf",
        "predicted against observed with the deployed intervals",
    ),
    (
        "report/figures/fig_coverage_sweep.pdf",
        "nominal against empirical coverage, before and after",
    ),
    (
        "report/figures/fig_conformal_band.pdf",
        "held out curves inside the simultaneous band",
    ),
    (
        "report/figures/fig_pit_heatmap.pdf",
        "probability integral transform along the displacement axis",
    ),
    (
        "report/figures/fig_crps_skill.pdf",
        "continuous ranked probability score against climatology",
    ),
)


class ArtifactMissing(RuntimeError):
    """A stage this script reads from has not run, and no number will be invented."""


def _read_json(path: Path, role: str, how: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMissing(
            f"cannot build the model card: the {role} at {path} does not exist. Run `{how}`; "
            "this script reads artifacts and never recomputes them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path, role: str, how: str) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactMissing(
            f"cannot build the model card: the {role} at {path} does not exist. Run `{how}`."
        )
    return pd.read_parquet(path)


def collect(root: Path, config: Config) -> dict[str, Any]:
    """Every artifact the card is built from, loaded once."""
    digest = config_hash(config)
    artifact_root = root / config.pipeline.paths.artifact_root
    directories: dict[str, Path] = {}
    for stage, how in REQUIRED_STAGES:
        directory = stage_dir(artifact_root, stage, digest)
        if not (directory / "manifest.json").is_file():
            raise ArtifactMissing(
                f"the {stage} stage has no manifest at {directory}. Run `{how}` for config "
                f"{digest[:12]} before generating the model card."
            )
        directories[stage] = directory
    return {
        "config_sha256": digest,
        "domain": _read_json(
            directories[AUDIT_STAGE] / VALIDITY_DOMAIN_JSON, "validity domain", "ufem run audit"
        ),
        "censoring": _read_json(
            directories[AUDIT_STAGE] / CENSORING_JSON, "censoring statistics", "ufem run audit"
        ),
        "completion": _read_json(
            directories[AUDIT_STAGE] / COMPLETION_JSON, "completion model", "ufem run audit"
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
        "surrogate_manifest": load_manifest(directories[SURROGATE_STAGE]),
    }


def _fmt(value: float, digits: int) -> str:
    """Fixed point at a stated precision, so a regenerated card is byte comparable."""
    return f"{float(value):.{digits}f}"


def _probability(value: float, n_samples: int) -> str:
    """A probability for a markdown table, with the two cases that must not print as a number.

    The same rule as :func:`ufem.propagate.format_probability`, rendered as prose rather than
    as a LaTeX fragment: this document is markdown and a reader would see the dollar signs. A
    count of zero prints as a bound at the resolution of the sample rather than as 0.0000,
    because no draw failing is a statement about the sample and a printed zero invites a
    reader to hear impossibility.
    """
    if value is None or not math.isfinite(float(value)):
        return "not defined"
    value = float(value)
    if value == 0.0:
        exponent = int(math.floor(math.log10(max(int(n_samples), 1))))
        return f"below 1e-{exponent}, the resolution of this sample"
    if value < RESOLVABLE_PF_FLOOR:
        return f"{value:.2e}, below the resolvable floor"
    return f"{value:.4f}"


def _p_value(value: float) -> str:
    """A p value in the form the report reads it: fixed below 0.001, scientific above.

    Not :func:`_probability`. A p value is not a failure probability and has no resolvable
    floor; borrowing that formatter would have attached a sentence about the training set size
    to a chi squared test, which is the kind of caption that gets quoted back later.
    """
    number = float(value)
    if number >= 0.001:
        return f"{number:.4f}"
    return f"{number:.2e}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return (
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def build_model_card(data: dict[str, Any], config: Config) -> str:
    """The whole card, as markdown."""
    out: list[str] = []

    def add(line: str = "") -> None:
        out.append(line)

    domain = data["domain"]
    censoring = data["censoring"]
    completion = data["completion"]
    surrogate = data["surrogate"]
    calibration = data["calibration"]
    sensitivity = data["sensitivity"]
    propagation = data["propagation"]
    manifest = data["surrogate_manifest"]

    add("# Model card: the UFEM 2.0 calibrated functional surrogate")
    add()
    add(
        "Generated by `python scripts/make_model_card.py` from the artifact store. Every "
        "number below was read from a file a pipeline stage wrote under the config hash in "
        "the provenance table; none was typed. A staleness gate in `tests/test_model_card.py` "
        "regenerates this document and compares it byte for byte, so if you are reading it, "
        "it is what the pipeline produces now."
    )
    add()

    # -- provenance ---------------------------------------------------------
    add("## Provenance")
    add()
    digests = {record["name"]: record["sha256"] for record in manifest["outputs"]}
    rows = [
        ["Config SHA-256", f"`{data['config_sha256']}`"],
        ["Surrogate record digest", f"`{digests[SURROGATE_JSON]}`"],
        ["Training runs", str(int(surrogate["n_training_runs"]))],
        ["Registration", surrogate["registration"]],
    ]
    out.extend(_table(["Item", "Value"], rows))
    add()
    add(
        "The commit this model was fitted at, the working tree's state at the time, and the "
        "digest of every other artifact are in "
        "`experiments/results/surrogate/<config hash>/manifest.json` and in the manifest of "
        "each stage upstream of it. They are deliberately not restated here. A commit changes "
        "every time a stage is rerun on a new branch while the numbers it produced do not, "
        "and a staleness gate that fired on that would be a gate that gets muted, which is a "
        "lesson the data card's provenance table already taught this project once. The two "
        "hashes above change only when the model changes, which is the property a card wants."
    )
    add()
    add("Resolved stack, as the surrogate stage recorded it:")
    add()
    out.extend(
        _table(
            ["Component", "Version"],
            [[name, value] for name, value in sorted(manifest["packages"].items())],
        )
    )
    add()

    # -- scope ---------------------------------------------------------------
    add("## What the model predicts, and from what")
    add()
    add(
        "Given three independent inputs, the surrogate predicts the whole load displacement "
        "response and the whole compressive damage evolution on a common displacement grid, "
        "plus a set of scalar quantities of interest. The scalars come from their own "
        "Gaussian processes and are never read off a reconstructed curve, so a peak load and "
        "the curve it belongs to are two predictions rather than one prediction and an "
        "inference from it."
    )
    add()
    bounds = domain["design_bounds"]
    out.extend(
        _table(
            ["Input", "Unit", "Design minimum", "Design maximum"],
            [
                [
                    INPUT_LABELS[name][0],
                    INPUT_LABELS[name][1],
                    _fmt(bounds[name][0], 2),
                    _fmt(bounds[name][1], 2),
                ]
                for name in FEATURE_ORDER
            ],
        )
    )
    add()
    add(
        "The elastic modulus is **not** an input. It is an exact deterministic function of "
        "the compressive strength in the inherited campaign, so treating it as a fourth "
        "independent variable would have made every variance decomposition here meaningless. "
        "It is derived in the one config file that declares the probabilistic model."
    )
    add()
    counts = surrogate["component_counts"]
    add(
        "Representation: the curve family is registered and reduced, and the Gaussian "
        "processes are fitted on the retained scores of each block."
    )
    add()
    out.extend(
        _table(
            ["Block", "Retained components"],
            [[name, str(int(counts[name]))] for name in sorted(counts)],
        )
    )
    add()
    add(
        f"Scalar targets with their own process: {', '.join(sorted(surrogate['scalar_targets']))}."
    )
    add()

    # -- validity domain -----------------------------------------------------
    add("## Validity domain: where this model is allowed to speak")
    add()
    add(
        f"{int(censoring['n_failed'])} of {int(censoring['n_designed'])} designed simulations "
        f"produced nothing, an overall failure rate of "
        f"{_fmt(censoring['overall_failure_rate'], 4)}, and the failures are not spread "
        "evenly across the design. The surviving runs are therefore a biased subsample, and "
        "the model carries a machine checked domain rather than a caveat."
    )
    add()
    add(
        "A query is inside the domain when the fitted completion model gives P(complete) at "
        f"or above {_fmt(domain['completion_threshold'], 2)} **and** the point lies inside the "
        "box of the executed design. The second condition is not redundant: a classifier "
        "asked about a point with no design points under it will answer, and that answer is "
        "an extrapolation of a smooth kernel rather than evidence."
    )
    add()
    out.extend(
        _table(
            ["Measurement", "Value"],
            [
                ["Completion model", completion["kind"]],
                [
                    "Cross validated ROC AUC",
                    _fmt(completion["cv_roc_auc"], 4),
                ],
                [
                    "Cross validated Brier score against the base rate",
                    f"{_fmt(completion['cv_brier_score'], 4)} against "
                    f"{_fmt(completion['baseline_brier_score'], 4)}",
                ],
                [
                    "Expected calibration error",
                    _fmt(completion["expected_calibration_error"], 4),
                ],
                [
                    "Share of the design box inside the domain",
                    _fmt(domain["dense_inside_fraction"], 4),
                ],
                [
                    "Share of the completed runs inside the domain",
                    _fmt(domain["valid_jobs_inside_fraction"], 4),
                ],
                [
                    "Share of the failed runs inside the domain",
                    _fmt(domain["failed_jobs_inside_fraction"], 4),
                ],
            ],
        )
    )
    add()
    add("Where the campaign failed, by input, at the configured significance level:")
    add()
    corner_rows = []
    for name in FEATURE_ORDER:
        record = censoring["by_input"][name]
        worst = max(record["quantile_failure_rates"], key=lambda item: item["fail_rate"])
        corner_rows.append(
            [
                INPUT_LABELS[name][0],
                "yes" if record["significant_at_level"] else "no",
                _p_value(record["chi2_p_value"]),
                f"{worst['bin']}, [{_fmt(worst['low'], 2)}, {_fmt(worst['high'], 2)}] "
                f"{INPUT_LABELS[name][1]}",
                f"{int(worst['n_failed'])} of {int(worst['n'])}",
            ]
        )
    out.extend(
        _table(
            [
                "Input",
                "Associated with failure",
                "Chi squared p value",
                "Worst quartile",
                "Failures there",
            ],
            corner_rows,
        )
    )
    add()
    add(
        "UFEM Lab grays out a prediction whose inputs fall outside this domain and names the "
        "corner it fell into, from this same table."
    )
    add()

    # -- out of sample performance -------------------------------------------
    baselines = data["baselines"]
    gate = baselines["gate"]
    models = sorted({row["model"] for row in baselines["scalar"]})
    add("## Out of sample performance")
    add()
    add(
        "Leave one out test R2 on the headline quantities, surrogate against every baseline. "
        "Train numbers are in the validation artifact and are never reported alone."
    )
    add()
    performance_rows = []
    for target in gate["headline_qoi"]:
        row = [QOI_LABELS[target]]
        for model in models:
            match = [
                item
                for item in baselines["scalar"]
                if item["target"] == target and item["model"] == model
            ]
            row.append(_fmt(match[0]["r2_test"], 4) if match else "not run")
        performance_rows.append(row)
    out.extend(_table(["Quantity"] + models, performance_rows))
    add()
    add(
        f"Baseline gate passed: **{bool(gate['passed'])}**"
        + (
            "."
            if not gate["failing_targets"]
            else f", failing on {', '.join(gate['failing_targets'])}."
        )
    )
    add()

    # -- calibration ----------------------------------------------------------
    calibration_gate = calibration["gate"]
    add("## Calibration status")
    add()
    add(
        "Predictive intervals are not asserted here, they are measured. Scalar intervals are "
        "jackknife+ with sigma normalized scores; curve bands are simultaneous sup norm bands "
        "over the displacement grid. Both are checked by a leave one out coverage measurement "
        "with a binomial confidence band, and the gate below is what blocks the propagated "
        "numbers when the measurement disagrees with the nominal level."
    )
    add()
    add(f"Calibration gate passed: **{bool(calibration_gate['passed'])}**.")
    add()
    gate_rows = []
    for check in calibration_gate["checks"]:
        interval = check.get("wilson")
        gate_rows.append(
            [
                check["criterion"],
                _fmt(check["measured"], 4),
                f"[{_fmt(interval[0], 4)}, {_fmt(interval[1], 4)}]"
                if interval is not None
                else "not a counted proportion",
                str(bool(check["passed"])),
            ]
        )
    out.extend(
        _table(["Criterion", "Measured", "95 percent Wilson interval", "Passed"], gate_rows)
    )
    add()
    add(
        "The measured out of fold variance scaling factor per scalar target. A factor near "
        "one means the fitted process was already saying the right width; a factor far from "
        "one is the model being corrected, and it is applied rather than hidden."
    )
    add()
    out.extend(
        _table(
            ["Target", "Variance scaling factor", "Predictive variance adequacy, before"],
            [
                [
                    name,
                    _fmt(record["variance_scaling_factor"], 4),
                    _fmt(record["pva_before"], 4),
                ]
                for name, record in sorted(calibration["scalar"].items())
            ],
        )
    )
    add()
    add("Functional bands, per signal, at the levels whose coverage was measured:")
    add()
    band_rows = []
    for signal, record in sorted(calibration["functional"].items()):
        for alpha, band in sorted(record["bands"].items()):
            band_rows.append(
                [
                    signal,
                    _fmt(band["nominal"], 2),
                    _fmt(band["band_scale"], 4),
                    _fmt(record["variance_scaling_factor"], 4),
                    _fmt(band["empirical"], 4),
                    f"[{_fmt(band['wilson_low'], 4)}, {_fmt(band['wilson_high'], 4)}]",
                ]
            )
    out.extend(
        _table(
            [
                "Signal",
                "Nominal",
                "Band scale",
                "Variance scaling",
                "Empirical coverage",
                "Wilson interval",
            ],
            band_rows,
        )
    )
    add()
    for signal, record in sorted(calibration["functional"].items()):
        add(
            f"- The {signal} band is defined on "
            f"{int(record['n_grid']) - int(record['domain']['n_excluded'])} of "
            f"{int(record['domain']['n_abscissae'])} abscissae, excluding "
            f"{int(record['domain']['n_excluded'])} where every observed run takes the same "
            "value. No variance is floored to manufacture calibration information there."
        )
    add()
    add("Calibration figures, all generated from these same artifacts:")
    add()
    for path, caption in CALIBRATION_FIGURES:
        add(f"- `{path}`: {caption}")
    add()

    # -- reliability ----------------------------------------------------------
    context = propagation["context"]
    validity = propagation["validity"]
    add("## Propagated behavior and reliability")
    add()
    add(
        f"{int(context['n_samples'])} Monte Carlo draws of the three inputs through the "
        "calibrated surrogate, with the aleatory and epistemic layers kept apart and never "
        "added. Every failure probability carries a Monte Carlo standard error, a surrogate "
        "aware conservative bound obtained by counting a failure whenever the calibrated "
        "interval crosses the threshold, and the share of the sample that fell outside the "
        "validity domain."
    )
    add()
    out.extend(
        _table(
            [
                "Limit state",
                "Threshold",
                "Pf",
                "Binomial SE",
                "Conservative bound",
                "Pf inside the domain",
            ],
            [
                [
                    record["short_label"],
                    _fmt(record["threshold"], 3),
                    _probability(record["pf_point"], int(record["n_samples"])),
                    _fmt(record["pf_standard_error"], 5),
                    _probability(record["pf_conservative"], int(record["n_samples"])),
                    _probability(record["pf_inside_domain"], int(record["n_inside_domain"])),
                ]
                for record in propagation["limit_states"]
            ],
        )
    )
    add()
    add(
        f"Out of domain mass fraction: {_fmt(validity['out_of_domain_fraction'], 4)}. "
        f"Resolvable failure probability floor: {context['resolvable_pf_floor']:g}. That floor "
        f"is imposed by the {int(context['n_training_runs'])} training runs and not by the "
        "Monte Carlo sample size; no probability below it is claimed, whatever the sample "
        "makes it possible to print."
    )
    add()
    add("Propagated quantities, in the units the pipeline carries them in:")
    add()
    out.extend(
        _table(
            ["Quantity", "Unit", "Aleatory mean", "5th percentile", "95th percentile"],
            [
                [
                    record["label"],
                    QOI_DISPLAY[name][0],
                    _fmt(record["aleatory"]["mean"] * QOI_DISPLAY[name][1], 4),
                    _fmt(
                        record["aleatory"]["quantiles"][record["aleatory"]["levels"].index(0.05)]
                        * QOI_DISPLAY[name][1],
                        4,
                    ),
                    _fmt(
                        record["aleatory"]["quantiles"][record["aleatory"]["levels"].index(0.95)]
                        * QOI_DISPLAY[name][1],
                        4,
                    ),
                ]
                for name, record in sorted(propagation["targets"].items())
                if name in context["reported_targets"]
            ],
        )
    )
    add()

    # -- failure modes --------------------------------------------------------
    add("## Known failure modes")
    add()
    add(
        "This section is the point of the document. Each entry is a measurement, and each one "
        "names something this model cannot do."
    )
    add()
    add(f"1. **Design roughness.** {propagation['roughness_caveat']}")
    add()
    counts_by_level = sensitivity["publication_counts"]
    add(
        f"2. **Withheld sensitivity indices.** All "
        f"{len(sensitivity['context']['targets'])} sparse chaos expansions failed the "
        f"publication gate: {counts_by_level['values']} reached the value publication "
        f"threshold of {sensitivity['context']['q2_publish_values']:g} and "
        f"{counts_by_level['rankings']} the ranking threshold of "
        f"{sensitivity['context']['q2_publish_rankings']:g}. The overall outcome is "
        f"**{PUBLICATION_TEX[sensitivity['gate_outcome']]}**, so no Sobol index value and no "
        "input ranking is published from this campaign. The Gaussian process posterior cross "
        "check exists and is reported as indicative only; it agreed with the chaos "
        f"expansions on {sensitivity['agreement']['n_agree']} of "
        f"{sensitivity['agreement']['n_rows']} assessed rows, which is itself a statement "
        "about how little either family pins down."
    )
    add()
    add(
        f"3. **Censoring.** {int(censoring['n_failed'])} of "
        f"{int(censoring['n_designed'])} designed runs produced nothing and the failures "
        "cluster, as the validity domain table above records. Everything downstream carries "
        "the domain; nothing downstream corrects the bias, because a correction would need "
        "runs that do not exist."
    )
    add()
    saturation = propagation["targets"].get("damage_at_10mm")
    if saturation is not None:
        add(
            "4. **Damage saturation, and the surrogate walking past it.** The compressive "
            "damage scalar is capped by the material table in every run of the inherited "
            "campaign, so a threshold near that cap asks when saturation is reached rather "
            "than whether it is. The Gaussian process does not know about the cap: its "
            f"propagated aleatory distribution has a mean of "
            f"{_fmt(saturation['aleatory']['mean'], 4)} and reaches "
            f"{_fmt(saturation['aleatory']['max'], 4)}, above the value the material table "
            "allows. That is a property of an unconstrained regression on a saturating "
            "response, it is visible rather than clipped away, and it is a reason to read the "
            "damage limit state as a screening number."
        )
        add()
    add(
        "5. **Fixed model parameters.** Dilation angle, eccentricity, viscosity and fracture "
        "energy were frozen across the inherited campaign, so nothing here quantifies their "
        "contribution to the response. That is the largest single gap in this model's "
        "coverage of its own uncertainty, and closing it needs new solver runs."
    )
    add()
    add(
        "6. **In sample overlays.** The dashboard's dataset panel draws a finite element run "
        "against the surrogate's prediction at the same inputs. Those runs are training runs, "
        "so that view is an in sample comparison and the panel says so. The out of sample "
        "version is the calibration stage's held out band examples."
    )
    add()

    add("## Intended use, and use this model is not fit for")
    add()
    add(
        "Fit for: exploring how the response of this beam family varies with strength and "
        "cover inside the executed design, with an interval whose coverage was measured; "
        "screening limit states whose failure probability is above the resolvable floor; and "
        "reproducing every number in the report and the dashboard from the manifests."
    )
    add()
    add(
        "Not fit for: design of a member outside the input box or inside the censored corner; "
        "any statement about the fixed material parameters; failure probabilities below the "
        "floor above; and any claim about a different geometry, since the beam geometry is "
        "fixed across the whole campaign."
    )
    add()
    return "\n".join(out)


def write_if_changed(path: Path, content: str) -> bool:
    """Write only when the content differs, with newlines forced to ``\\n``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def generate(root: Path) -> dict[str, str]:
    """Build the card and return a mapping of repo relative path to content."""
    config = load_config(root)
    return {MODEL_CARD: build_model_card(collect(root, config), config)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    written = 0
    for relative, content in generate(root).items():
        changed = write_if_changed(root / relative, content)
        written += int(changed)
        print(f"{relative}: {'written' if changed else 'unchanged'} ({len(content)} bytes)")
    print(f"make_model_card: {written} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
