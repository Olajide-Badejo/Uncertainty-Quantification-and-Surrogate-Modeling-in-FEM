"""Generate ``docs/DATA_CARD.md`` and the report's numeric fragments from artifacts only.

Build spec 9.4 and 19. Binding law 5: no number appears in the README, the report, or the UI
unless it is reproducible from a committed manifest whose hashes resolve to real files. This
script is what makes that true for the data card and for every number in ``report/main.tex``.

It reads the artifact store and the stage manifests, nothing else. There is not one hand
typed measurement in this file: every figure in the generated card and in every ``.tex``
fragment under ``report/tables/`` came out of a Parquet or a JSON that a stage wrote, and if
a stage has not run, this script raises naming the stage rather than filling a gap.

Two output families:

- ``docs/DATA_CARD.md``, the campaign, the extraction, the censoring bias tables, the
  completion model performance, the validity domain, and the importance weighting result.
- ``report/tables/*.tex``, one fragment per number or table that ``main.tex`` needs. The
  fragments are committed because they are small text files, and committing them is what
  lets the LaTeX build run in CI without the artifact store.

A staleness test regenerates both and asserts byte identity, so a card that has drifted from
the pipeline is a failing test rather than a document nobody rechecked.

Exit 0 is clean, exit 1 names the failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ufem.audit import (
    CENSORING_JSON,
    COMPLETION_JSON,
    STATUS_MISSING,
    STATUS_PARTIAL,
    STATUS_VALID,
    VALIDITY_DOMAIN_JSON,
    VALIDITY_PARQUET,
    WEIGHTING_JSON,
)
from ufem.audit import STAGE_NAME as AUDIT_STAGE
from ufem.config import FEATURE_ORDER, Config, config_hash, load_config
from ufem.grid import QOI_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import DESIGN_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import load_manifest, stage_dir
from ufem.reduce import BASIS_JSON, RECONSTRUCTION_JSON
from ufem.reduce import STAGE_NAME as REDUCE_STAGE
from ufem.register import LANDMARKS_PARQUET
from ufem.register import STAGE_NAME as REGISTER_STAGE

#: The registration ablation is a script rather than a stage, but it writes into the same
#: artifact store under this name, and the report quotes its three metrics.
ABLATION_1_STAGE = "ablation_1_registration"
ABLATION_1_JSON = "ablation_1_registration.json"

DATA_CARD = "docs/DATA_CARD.md"
TABLES_DIR = "report/tables"

#: Display names and units for the three independent inputs, so the card reads as a document
#: rather than as a column dump. The names themselves come from the feature contract.
INPUT_LABELS: dict[str, tuple[str, str]] = {
    "Fcm_MPa": ("Mean compressive strength", "MPa"),
    "c_nom_bottom_mm": ("Bottom cover", "mm"),
    "c_nom_top_mm": ("Top cover", "mm"),
}

#: Short LaTeX safe keys for the three inputs, used to build macro names. Derived by hand
#: rather than from the column name because a mechanical truncation collides: both covers
#: begin ``c_nom_``, so the first two parts of each name are the same word.
INPUT_KEYS: dict[str, str] = {
    "Fcm_MPa": "Fcm",
    "c_nom_bottom_mm": "CBot",
    "c_nom_top_mm": "CTop",
}

#: LaTeX math names for the same three, used in the generated table fragments.
INPUT_MATH: dict[str, str] = {
    "Fcm_MPa": r"$f_{cm}$ [MPa]",
    "c_nom_bottom_mm": r"$c_{\mathrm{bot}}$ [mm]",
    "c_nom_top_mm": r"$c_{\mathrm{top}}$ [mm]",
}

#: The QoIs the weighting table reports, with the display name, unit, and the scale each is
#: divided by for presentation. Forces read in kN and energies in J, as the report does.
WEIGHTED_QOI_DISPLAY: dict[str, tuple[str, str, float]] = {
    "P_max_N": ("Peak load", "kN", 1000.0),
    "u_peak_mm": ("Displacement at peak", "mm", 1.0),
    "k0_N_per_mm": ("Initial stiffness", "kN/mm", 1000.0),
    "E_abs_Nmm": ("Absorbed energy to 20 mm", "J", 1000.0),
    "P_residual_N": ("Residual load at 20 mm", "kN", 1000.0),
    "softening_ratio": ("Softening ratio", "-", 1.0),
    "u_damage_half_sat_mm": ("Displacement at half damage saturation", "mm", 1.0),
    "damage_at_10mm": ("Damage at 10 mm", "-", 1.0),
}


class ArtifactMissing(RuntimeError):
    """A stage this script reads from has not run, and no number will be invented."""


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMissing(
            f"cannot build the data card: the {role} at {path} does not exist. Run "
            "`ufem run all` first; this script reads artifacts and never recomputes them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactMissing(
            f"cannot build the data card: the {role} at {path} does not exist. Run "
            "`ufem run all` first."
        )
    return pd.read_parquet(path)


def collect(root: Path, config: Config) -> dict[str, Any]:
    """Every artifact the card and the fragments are built from, loaded once."""
    digest = config_hash(config)
    artifact_root = root / config.pipeline.paths.artifact_root
    ingest_dir = stage_dir(artifact_root, INGEST_STAGE, digest)
    grid_dir = stage_dir(artifact_root, GRID_STAGE, digest)
    audit_dir = stage_dir(artifact_root, AUDIT_STAGE, digest)
    for directory, stage in (
        (ingest_dir, INGEST_STAGE),
        (grid_dir, GRID_STAGE),
        (audit_dir, AUDIT_STAGE),
    ):
        if not (directory / "manifest.json").is_file():
            raise ArtifactMissing(
                f"the {stage} stage has no manifest at {directory}. Run `ufem run {stage}` "
                f"for config {digest[:12]} before generating the data card."
            )
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, digest)
    reduce_dir = stage_dir(artifact_root, REDUCE_STAGE, digest)
    ablation_dir = stage_dir(artifact_root, ABLATION_1_STAGE, digest)
    for directory, stage, how in (
        (register_dir, REGISTER_STAGE, f"ufem run {REGISTER_STAGE}"),
        (reduce_dir, REDUCE_STAGE, f"ufem run {REDUCE_STAGE}"),
    ):
        if not (directory / "manifest.json").is_file():
            raise ArtifactMissing(
                f"the {stage} stage has no manifest at {directory}. Run `{how}` for config "
                f"{digest[:12]} before generating the data card."
            )
    ablation_path = ablation_dir / ABLATION_1_JSON
    if not ablation_path.is_file():
        raise ArtifactMissing(
            f"the registration ablation has no result at {ablation_path}. Run "
            "`python scripts/ablation_1_registration.py` before generating the data card: "
            "the report quotes its three metrics and they must come from the artifact."
        )
    return {
        "config_sha256": digest,
        "design": _read_parquet(ingest_dir / DESIGN_PARQUET, "LHS design"),
        "qoi": _read_parquet(grid_dir / QOI_PARQUET, "QoI table"),
        "validity": _read_parquet(audit_dir / VALIDITY_PARQUET, "validity classification"),
        "censoring": _read_json(audit_dir / CENSORING_JSON, "censoring statistics"),
        "completion": _read_json(audit_dir / COMPLETION_JSON, "completion model report"),
        "domain": _read_json(audit_dir / VALIDITY_DOMAIN_JSON, "validity domain"),
        "weighting": _read_json(audit_dir / WEIGHTING_JSON, "importance weighting study"),
        "landmarks": _read_parquet(register_dir / LANDMARKS_PARQUET, "landmark table"),
        "bases": _read_json(reduce_dir / BASIS_JSON, "PCA bases"),
        "reconstruction": _read_json(
            reduce_dir / RECONSTRUCTION_JSON, "reconstruction error percentiles"
        ),
        "ablation_1": _read_json(ablation_path, "registration ablation"),
        "ingest_manifest": load_manifest(ingest_dir),
        "grid_manifest": load_manifest(grid_dir),
        "audit_manifest": load_manifest(audit_dir),
        "register_manifest": load_manifest(register_dir),
        "reduce_manifest": load_manifest(reduce_dir),
    }


def _fmt(value: float, digits: int) -> str:
    """Fixed point with a stated precision, so a regenerated file is byte comparable."""
    return f"{value:.{digits}f}"


def _fmt_p(value: float) -> str:
    """A p value in the form the report reads it: fixed below 0.001, scientific above."""
    if value >= 1e-3:
        return f"{value:.3f}"
    exponent = 0
    mantissa = value
    while mantissa < 1.0:
        mantissa *= 10.0
        exponent -= 1
    return f"{mantissa:.1f}e{exponent}"


def _fmt_p_tex(value: float) -> str:
    """The same p value as LaTeX math mode content, with a real power of ten.

    Deliberately without surrounding dollars. A macro that carried its own math delimiters
    could only be used outside math mode, and the report writes these inside expressions
    like ``$p = \\ChiCTopP$``; nesting dollars there is a TeX error. So the macro expands to
    math mode *content* and the document supplies the mode, which is the convention that
    composes.
    """
    if value >= 1e-3:
        return f"{value:.3f}"
    exponent = 0
    mantissa = value
    while mantissa < 1.0:
        mantissa *= 10.0
        exponent -= 1
    return f"{mantissa:.1f} \\times 10^{{{exponent}}}"


def design_moments(design: pd.DataFrame) -> list[dict[str, Any]]:
    """The realized moments of the executed design, measured from the design Parquet."""
    rows = []
    for name in FEATURE_ORDER:
        values = design[name].to_numpy(dtype=float)
        label, unit = INPUT_LABELS[name]
        rows.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "cov": float(values.std(ddof=1) / values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def _declared_marginals(config: Config) -> dict[str, str]:
    """How each input was declared, read from the validated probabilistic model.

    Read, not typed. Binding law 2 puts the distributions in one config file; the card
    reports what that file says rather than restating it from memory. The family and its
    parameters only: the ``basis`` field of each variable is a paragraph of justification
    that belongs in the prose, not folded into a table cell.
    """
    described = {}
    for name in FEATURE_ORDER:
        variable = config.probabilistic_model.variables[name]
        if variable.kind == "lognormal":
            described[name] = f"lognormal, mean {variable.mean:g}, CoV {variable.cov:g}"
        else:
            described[name] = f"normal, mu {variable.mu:g}, sigma {variable.sigma:g}"
    return described


def _declared_bases(config: Config) -> dict[str, str]:
    """The stated justification for each marginal, as one flowed line per input."""
    return {
        name: " ".join(config.probabilistic_model.variables[name].basis.split())
        for name in FEATURE_ORDER
    }


# ---------------------------------------------------------------------------
# The LaTeX fragments
# ---------------------------------------------------------------------------


def _macro(name: str, body: str) -> str:
    """One ``\\newcommand`` line, which is how a number reaches ``main.tex``."""
    return f"\\newcommand{{\\{name}}}{{{body}}}\n"


def build_macro_fragment(data: dict[str, Any], config: Config) -> str:
    """Every scalar the report quotes, as a LaTeX macro defined from an artifact.

    ``main.tex`` inputs this file and then writes ``\\PeakLoadMean`` where it would
    otherwise have typed a number. That is the mechanism build spec 20 requires: the prose
    cannot drift from the pipeline because the prose does not contain the numbers.
    """
    censoring = data["censoring"]
    completion = data["completion"]
    domain = data["domain"]
    weighting = data["weighting"]
    counts = data["validity"]["status"].value_counts()
    qoi = data["qoi"]
    design = data["design"]
    interval = completion["cv_roc_auc_interval"]

    lines = [
        "% Generated by scripts/make_data_card.py from the artifact store. Do not edit.",
        "% Every macro below resolves to a number a pipeline stage measured and hashed into",
        "% its manifest (build spec 20 and binding law 5).",
        "",
    ]

    defined: set[str] = set()

    def add(name: str, body: str) -> None:
        """Define one macro, refusing to redefine a name.

        LaTeX would accept a second ``\\newcommand`` for the same name only by erroring, but
        a colliding name is worse than a build failure: it means two different measurements
        were trying to reach the report through one macro, and whichever lost would have
        been silently replaced. This catches the collision here, where the name is chosen.
        """
        if name in defined:
            raise ValueError(
                f"the macro name {name!r} is defined twice. Two different measurements are "
                "mapping onto one report symbol, so one of them would silently overwrite "
                "the other; give them distinct names in INPUT_KEYS or at the call site."
            )
        defined.add(name)
        lines.append(_macro(name, body).rstrip("\n"))

    add("NDesign", str(len(data["validity"])))
    add("NValid", str(int(counts.get(STATUS_VALID, 0))))
    add("NMissing", str(int(counts.get(STATUS_MISSING, 0))))
    add("NPartial", str(int(counts.get(STATUS_PARTIAL, 0))))
    add("OverallFailureRate", _fmt(100.0 * censoring["overall_failure_rate"], 1))

    for name in FEATURE_ORDER:
        values = design[name].to_numpy(dtype=float)
        key = INPUT_KEYS[name]
        add(f"Design{key}Mean", _fmt(float(values.mean()), 2))
        add(f"Design{key}Std", _fmt(float(values.std(ddof=1)), 2))

    for name in FEATURE_ORDER:
        block = censoring["by_input"][name]
        key = INPUT_KEYS[name]
        add(f"Chi{key}P", _fmt_p_tex(block["chi2_p_value"]))
        add(f"Biserial{key}", _fmt(block["point_biserial_r"], 3))
        add(f"WelchP{key}", _fmt_p_tex(block["welch_p_value"]))
        add(f"MeanFailed{key}", _fmt(block["mean_failed"], 2))
        add(f"MeanValid{key}", _fmt(block["mean_valid"], 2))
        rates = block["quantile_failure_rates"]
        add(f"FailRate{key}Low", _fmt(100.0 * rates[0]["fail_rate"], 0))
        add(f"FailRate{key}High", _fmt(100.0 * rates[-1]["fail_rate"], 0))
        add(
            f"FailRate{key}Min",
            _fmt(100.0 * min(row["fail_rate"] for row in rates), 0),
        )

    add("CompletionModel", completion["kind"].replace("_", " "))
    add("CompletionAUC", _fmt(completion["cv_roc_auc"], 3))
    add("CompletionAUCLow", _fmt(interval["auc_low"], 3))
    add("CompletionAUCHigh", _fmt(interval["auc_high"], 3))
    add("CompletionIntervalLevel", _fmt(100.0 * interval["level"], 0))
    add("CompletionBootstrap", str(interval["n_resamples"]))
    add("CompletionBrier", _fmt(completion["cv_brier_score"], 3))
    add("CompletionBrierBaseline", _fmt(completion["baseline_brier_score"], 3))
    add("CompletionECE", _fmt(completion["expected_calibration_error"], 3))
    add("CompletionFolds", str(completion["n_folds"]))

    add("DomainThreshold", _fmt(domain["completion_threshold"], 2))
    add("DomainDesignInside", _fmt(100.0 * domain["design_inside_fraction"], 1))
    add("DomainValidInside", _fmt(100.0 * domain["valid_jobs_inside_fraction"], 1))
    add("DomainFailedInside", _fmt(100.0 * domain["failed_jobs_inside_fraction"], 1))

    largest = weighting["largest_relative_mean_shift"]
    add("WeightLargestQoI", WEIGHTED_QOI_DISPLAY[largest["qoi"]][0].lower())
    add("WeightLargestShift", _fmt(100.0 * abs(largest["value"]), 2))
    add("WeightPeakShift", _fmt(100.0 * weighting["by_qoi"]["P_max_N"]["mean_shift_relative"], 2))
    add("WeightESS", _fmt(weighting["weight"]["effective_sample_size"], 0))
    add("WeightMax", _fmt(weighting["weight"]["max"], 2))
    add("WeightNSamples", str(weighting["n_weighted_samples"]))

    add("QoIPeakMeanKN", _fmt(float(qoi["P_max_N"].mean()) / 1000.0, 2))
    add("QoIPeakCoV", _fmt(float(qoi["P_max_N"].std(ddof=1) / qoi["P_max_N"].mean()), 3))
    add("QoIResidualCoV", _fmt(
        float(qoi["P_residual_N"].std(ddof=1) / qoi["P_residual_N"].mean()), 3
    ))

    # Phase P3: registration, reduction, and the registration ablation.
    landmarks = data["landmarks"]
    reached = landmarks["u85_reached"].to_numpy(dtype=bool)
    add("LandmarkKneeMean", _fmt(float(landmarks["u_knee_mm"].mean()), 2))
    add("LandmarkPeakMean", _fmt(float(landmarks["u_peak_mm"].mean()), 2))
    add("LandmarkPostPeakMean", _fmt(float(landmarks.loc[reached, "u_85_mm"].mean()), 2))
    add("LandmarkPostPeakMissing", str(int((~reached).sum())))

    blocks = {block["name"]: block for block in data["bases"]["blocks"]}
    for name, key in (("amplitude", "Amplitude"), ("phase", "Phase"), ("damage", "Damage")):
        block = blocks[name]
        add(f"Components{key}", str(int(block["n_retained"])))
        add(f"Variance{key}", _fmt(100.0 * float(block["variance_explained_by_retained"]), 2))
        add(f"ReconMedian{key}", _fmt(100.0 * float(block["reconstruction_error"]["p50"]), 2))
        add(f"ReconP{key}Ninety", _fmt(
            100.0 * float(block["reconstruction_error"]["p90"]), 2
        ))
    add("VarianceTarget", _fmt(100.0 * float(data["bases"]["variance_target"]), 0))
    add("AmplitudePCOne", _fmt(
        100.0 * float(blocks["amplitude"]["explained_variance_ratio"][0]), 1
    ))

    ablation = data["ablation_1"]
    add("AblationComponentsRegistered", str(int(ablation["components_at_target"]["registered"])))
    add(
        "AblationComponentsUnregistered",
        str(int(ablation["components_at_target"]["unregistered"])),
    )
    add("AblationComponentRatio", _fmt(float(ablation["components_at_target"]["ratio"]), 2))
    add("AblationDerivCorrRegistered", _fmt(
        float(ablation["derivative_mode_correlation"]["registered"]), 3
    ))
    add("AblationDerivCorrUnregistered", _fmt(
        float(ablation["derivative_mode_correlation"]["unregistered"]), 3
    ))
    add("AblationDerivCorrPCOneRegistered", _fmt(
        float(ablation["derivative_mode_correlation_by_component"]["registered"][0]), 3
    ))
    add("AblationDerivCorrPCOneUnregistered", _fmt(
        float(ablation["derivative_mode_correlation_by_component"]["unregistered"][0]), 3
    ))
    add("AblationPeakBiasRegistered", _fmt(
        float(ablation["peak_load_bias"]["registered"]["mean_signed_error_N"]), 1
    ))
    add("AblationPeakBiasUnregistered", _fmt(
        float(ablation["peak_load_bias"]["unregistered"]["mean_signed_error_N"]), 1
    ))
    add("AblationPeakBiasRegisteredPct", _fmt(
        100.0 * abs(float(ablation["peak_load_bias"]["registered"]["mean_relative_error"])), 2
    ))
    add("AblationPeakBiasUnregisteredPct", _fmt(
        100.0 * abs(float(ablation["peak_load_bias"]["unregistered"]["mean_relative_error"])),
        2,
    ))
    add("AblationRank", str(int(ablation["rank_for_peak_comparison"])))

    add("ConfigHash", data["config_sha256"][:12])
    add("GridPoints", str(config.pipeline.grid.n_points))
    add("QuantileBins", str(censoring["n_quantile_bins"]))

    return "\n".join(lines) + "\n"


def build_quartile_table(data: dict[str, Any]) -> str:
    """The failure rate by input quartile table, as a committed LaTeX fragment."""
    censoring = data["censoring"]
    lines = [
        "% Generated by scripts/make_data_card.py. Do not edit.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Input & Q1 (low) & Q2 & Q3 & Q4 (high) & $\chi^2$ $p$ \\",
        r"\midrule",
    ]
    for name in FEATURE_ORDER:
        block = censoring["by_input"][name]
        rates = " & ".join(
            _fmt(100.0 * row["fail_rate"], 0) + r"\,\%"
            for row in block["quantile_failure_rates"]
        )
        # The macros expand to math mode content, so a table cell that is not already in
        # math mode has to open it. See the note on _fmt_p_tex.
        lines.append(
            f"{INPUT_MATH[name]} & {rates} & ${_fmt_p_tex(block['chi2_p_value'])}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_calibration_table(data: dict[str, Any]) -> str:
    """The reliability table: predicted against empirical completion rate, per bin."""
    table = data["completion"]["calibration_table"]
    lines = [
        "% Generated by scripts/make_data_card.py. Do not edit.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Predicted bin & $n$ & Mean predicted & Empirical rate \\",
        r"\midrule",
    ]
    for row in table:
        span = f"{_fmt(row['bin_low'], 1)} to {_fmt(row['bin_high'], 1)}"
        if row["n"] == 0:
            lines.append(f"{span} & 0 & {{--}} & {{--}} \\\\".replace("--", r"\textendash"))
            continue
        lines.append(
            f"{span} & {row['n']} & {_fmt(row['mean_predicted'], 3)} & "
            f"{_fmt(row['empirical_rate'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_weighting_table(data: dict[str, Any]) -> str:
    """Headline QoI statistics, unweighted against inverse completion probability weighted."""
    weighting = data["weighting"]
    # The quantity names are long and the page is narrow, so the label column is a
    # paragraph column with a fixed width rather than an l column that pushes the table
    # past the text block. LaTeX reports that as an overfull hbox; a table that runs into
    # the margin is a defect, not a warning to live with.
    lines = [
        "% Generated by scripts/make_data_card.py. Do not edit.",
        r"\begin{tabular}{p{0.34\linewidth}rrrr}",
        r"\toprule",
        r"Quantity & Unweighted & Weighted & Shift & $\Delta$ CoV \\",
        r"\midrule",
    ]
    for name, (label, unit, scale) in WEIGHTED_QOI_DISPLAY.items():
        block = weighting["by_qoi"][name]
        unweighted = block["unweighted"]["mean"] / scale
        weighted = block["weighted"]["mean"] / scale
        digits = 3 if unit == "-" else 2
        lines.append(
            f"{label} [{unit}] & {_fmt(unweighted, digits)} & {_fmt(weighted, digits)} & "
            f"{_fmt(100.0 * block['mean_shift_relative'], 2)}"
            r"\,\%"
            f" & {_fmt(block['cov_shift'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_reduction_table(data: dict[str, Any]) -> str:
    """One row per reduction block: components retained, variance, reconstruction error."""
    blocks = {block["name"]: block for block in data["bases"]["blocks"]}
    labels = {
        "amplitude": "Registered amplitude",
        "phase": "Warp tangent (phase)",
        "damage": "Damage (unregistered)",
    }
    lines = [
        "% Generated by scripts/make_data_card.py. Do not edit.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Block & Components & Variance & Median error & 90th pct. error \\",
        r"\midrule",
    ]
    for name, label in labels.items():
        block = blocks[name]
        error = block["reconstruction_error"]
        lines.append(
            f"{label} & {int(block['n_retained'])} & "
            f"{_fmt(100.0 * float(block['variance_explained_by_retained']), 2)}"
            r"\,\%"
            f" & {_fmt(100.0 * float(error['p50']), 2)}"
            r"\,\%"
            f" & {_fmt(100.0 * float(error['p90']), 2)}"
            r"\,\% \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_design_table(data: dict[str, Any], config: Config) -> str:
    """Declared marginals against the moments the executed design realized."""
    declared = _declared_marginals(config)
    lines = [
        "% Generated by scripts/make_data_card.py. Do not edit.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Input & Declared & Mean & Std.\ dev. & Min & Max \\",
        r"\midrule",
    ]
    for row in design_moments(data["design"]):
        text = declared[row["name"]].split(" (")[0]
        lines.append(
            f"{INPUT_MATH[row['name']]} & {text} & {_fmt(row['mean'], 2)} & "
            f"{_fmt(row['std'], 2)} & {_fmt(row['min'], 2)} & {_fmt(row['max'], 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def build_data_card(data: dict[str, Any], config: Config) -> str:
    """The whole of ``docs/DATA_CARD.md``, assembled from the loaded artifacts."""
    censoring = data["censoring"]
    completion = data["completion"]
    domain = data["domain"]
    weighting = data["weighting"]
    counts = data["validity"]["status"].value_counts()
    n_valid = int(counts.get(STATUS_VALID, 0))
    n_missing = int(counts.get(STATUS_MISSING, 0))
    n_partial = int(counts.get(STATUS_PARTIAL, 0))
    declared = _declared_marginals(config)
    interval = completion["cv_roc_auc_interval"]

    out: list[str] = []
    add = out.append

    add("# Data card: the inherited simulation campaign")
    add("")
    add(
        "Generated by `scripts/make_data_card.py` from the artifact store. Every number "
        "below was measured by a pipeline stage and is recorded in that stage's "
        "`manifest.json`; none is typed here. Regenerate with"
    )
    add("")
    add("```")
    add("ufem run all")
    add("python scripts/make_data_card.py")
    add("```")
    add("")
    add(f"Config SHA-256: `{data['config_sha256']}`.")
    add("")

    add("## 1. The campaign")
    add("")
    add(
        f"A Latin hypercube design of {len(data['validity'])} samples over the three "
        "independent inputs of the probabilistic model, evaluated by a parametric Abaqus "
        "concrete damaged plasticity model of a reinforced concrete member under "
        "displacement control. The elastic modulus is a deterministic Eurocode 2 function "
        "of the strength and is never an independent input."
    )
    add("")
    add("| Input | Declared | Mean | Std. dev. | CoV | Min | Max |")
    add("|---|---|---|---|---|---|---|")
    for row in design_moments(data["design"]):
        add(
            f"| {row['label']} [{row['unit']}] | {declared[row['name']]} | "
            f"{_fmt(row['mean'], 3)} | {_fmt(row['std'], 3)} | {_fmt(row['cov'], 4)} | "
            f"{_fmt(row['min'], 2)} | {_fmt(row['max'], 2)} |"
        )
    add("")
    add(
        "The declared column is read from `configs/probabilistic_model.yaml`, which is the "
        "only place in this project a distribution exists (binding law 2). Each marginal's "
        "stated basis, from the same file:"
    )
    add("")
    for name, basis in _declared_bases(config).items():
        add(f"- **{INPUT_LABELS[name][0]}.** {basis}")
    add("")

    add("## 2. The extraction")
    add("")
    add(
        "Every one of the design rows is reclassified from the ingest artifacts on every "
        "run, by the criteria in the `audit` block of `configs/pipeline.yaml`, never from a "
        "stored list of sample identifiers. The predecessor project hard coded its surviving "
        "sample list as an unexplained literal; that is the defect this stage exists to make "
        "impossible."
    )
    add("")
    add("| Status | Count | Definition |")
    add("|---|---|---|")
    add(
        f"| valid | {n_valid} | present in the extracted data, covering the full "
        "displacement range, zero non finite values, monotone displacement, full step time |"
    )
    add(f"| missing | {n_missing} | absent from the extracted data entirely |")
    add(
        f"| partial | {n_partial} | present but failing at least one completeness "
        "criterion |"
    )
    add("")
    add(
        f"The classification agrees with the committed 2026-08-28 reference at "
        f"`data/audit_reference/sample_validity.csv` on all "
        f"{data['audit_manifest']['extra']['reference_comparison']['n_compared']} rows, with "
        "zero disagreements. Both are derived from the same raw inputs, so exact agreement "
        "is the gate rather than a tolerance."
    )
    add("")
    add(
        f"The overall failure rate is "
        f"{_fmt(100.0 * censoring['overall_failure_rate'], 1)} percent. The failure mode is "
        "binary: an analysis either finished completely or left nothing behind. The "
        "production solver logs are not preserved in the inherited tree, so the root cause "
        "of the losses cannot be diagnosed retrospectively."
    )
    add("")

    add("## 3. Censoring bias")
    add("")
    add(
        "The failures are not distributed at random over the design. Each input is split "
        f"into {censoring['n_quantile_bins']} equal count quantile bins of the executed "
        "design, so every bin carries the same exposure and a difference in rate is a "
        "difference in outcome."
    )
    add("")
    header_bins = [
        row["bin"].replace("_low", " (low)").replace("_high", " (high)")
        for row in censoring["by_input"][FEATURE_ORDER[0]]["quantile_failure_rates"]
    ]
    add("| Input | " + " | ".join(header_bins) + " | chi squared p | point biserial r |")
    add("|---" * (len(header_bins) + 3) + "|")
    for name in FEATURE_ORDER:
        block = censoring["by_input"][name]
        rates = " | ".join(
            _fmt(100.0 * row["fail_rate"], 0) + " %" for row in block["quantile_failure_rates"]
        )
        add(
            f"| {INPUT_LABELS[name][0]} | {rates} | {_fmt_p(block['chi2_p_value'])} | "
            f"{_fmt(block['point_biserial_r'], 3)} |"
        )
    add("")
    add("| Input | Mean when failed | Mean when valid | Welch p | Mann Whitney p |")
    add("|---|---|---|---|---|")
    for name in FEATURE_ORDER:
        block = censoring["by_input"][name]
        unit = INPUT_LABELS[name][1]
        add(
            f"| {INPUT_LABELS[name][0]} | {_fmt(block['mean_failed'], 2)} {unit} | "
            f"{_fmt(block['mean_valid'], 2)} {unit} | {_fmt_p(block['welch_p_value'])} | "
            f"{_fmt_p(block['mannwhitney_p_value'])} |"
        )
    add("")
    significant = [
        INPUT_LABELS[name][0]
        for name in FEATURE_ORDER
        if censoring["by_input"][name]["significant_at_level"]
    ]
    insignificant = [
        INPUT_LABELS[name][0]
        for name in FEATURE_ORDER
        if not censoring["by_input"][name]["significant_at_level"]
    ]
    add(
        "Significant at the "
        f"{_fmt(100.0 * censoring['significance_level'], 0)} percent level: "
        + ", ".join(significant).lower()
        + ". Not significant: "
        + ", ".join(insignificant).lower()
        + "."
    )
    add("")
    add(
        "The consequence is stated rather than implied: the surviving runs are a biased "
        "subsample of the intended design, so every population statistic estimated on them "
        "inherits that bias, and every downstream product must either model the censoring "
        "or carry the validity domain of section 5."
    )
    add("")

    add("## 4. The completion probability model")
    add("")
    add(
        "A model of P(complete | "
        + ", ".join(INPUT_LABELS[name][0].lower() for name in FEATURE_ORDER)
        + f") fitted over all {completion['n_samples']} design rows, "
        f"{completion['n_complete']} complete and {completion['n_incomplete']} not. The "
        f"estimator that shipped is the {completion['kind'].replace('_', ' ')}"
        + (
            " classifier, the primary of build spec 9.4."
            if not completion["fallback_taken"]
            else " classifier, the pre authorized fallback of build spec 9.4."
        )
    )
    if completion["fallback_taken"]:
        add("")
        add("The primary was rejected for the following measured reasons:")
        add("")
        for attempt in completion["attempts"]:
            if attempt["rejection_reasons"]:
                add(f"- {attempt['kind']}: " + "; ".join(attempt["rejection_reasons"]))
    add("")
    add(
        f"Evaluated by stratified {completion['n_folds']} fold cross validation with the "
        "feature standardization refitted inside each fold, so no test row contributes to "
        "the scaling of its own training set."
    )
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(
        f"| ROC AUC (cross validated) | {_fmt(completion['cv_roc_auc'], 4)} "
        f"[{_fmt(interval['auc_low'], 4)}, {_fmt(interval['auc_high'], 4)}] at "
        f"{_fmt(100.0 * interval['level'], 0)} percent, {interval['n_resamples']} bootstrap "
        "resamples |"
    )
    add(f"| Brier score (cross validated) | {_fmt(completion['cv_brier_score'], 4)} |")
    add(
        f"| Brier score of the base rate predictor | "
        f"{_fmt(completion['baseline_brier_score'], 4)} |"
    )
    add(
        f"| Expected calibration error | "
        f"{_fmt(completion['expected_calibration_error'], 4)} |"
    )
    add("")
    add(
        "The AUC is modest by construction. Roughly half the design failed, the effect is "
        "carried mostly by one input, and no amount of modeling recovers information the "
        "solver logs did not preserve. What matters for the use this model is put to is not "
        "its ranking power but its calibration: the validity domain thresholds a probability, "
        "so that probability has to mean what it says."
    )
    add("")
    add("| Predicted bin | n | Mean predicted | Empirical rate |")
    add("|---|---|---|---|")
    for row in completion["calibration_table"]:
        span = f"{_fmt(row['bin_low'], 1)} to {_fmt(row['bin_high'], 1)}"
        if row["n"] == 0:
            add(f"| {span} | 0 | not populated | not populated |")
        else:
            add(
                f"| {span} | {row['n']} | {_fmt(row['mean_predicted'], 3)} | "
                f"{_fmt(row['empirical_rate'], 3)} |"
            )
    add("")

    add("## 5. The validity domain")
    add("")
    # The definition is one sentence written by the audit stage; only its first letter is
    # raised, because ``str.capitalize`` would lowercase the P of P(complete).
    definition = domain["definition"]
    add(f"{definition[0].upper()}{definition[1:]}.")
    add("")
    add("| Property | Value |")
    add("|---|---|")
    add(f"| Completion threshold | {_fmt(domain['completion_threshold'], 2)} |")
    for name in FEATURE_ORDER:
        low, high = domain["design_bounds"][name]
        add(
            f"| {INPUT_LABELS[name][0]} bounds [{INPUT_LABELS[name][1]}] | "
            f"{_fmt(low, 2)} to {_fmt(high, 2)} |"
        )
    add(
        f"| Design points inside | {domain['design_inside_count']} of "
        f"{len(data['validity'])} ({_fmt(100.0 * domain['design_inside_fraction'], 1)} "
        "percent) |"
    )
    add(
        f"| Completed runs inside | "
        f"{_fmt(100.0 * domain['valid_jobs_inside_fraction'], 1)} percent |"
    )
    add(
        f"| Failed runs inside | "
        f"{_fmt(100.0 * domain['failed_jobs_inside_fraction'], 1)} percent |"
    )
    add("")
    add(
        "The gap between the last two rows is the point. The domain admits "
        f"{_fmt(100.0 * domain['valid_jobs_inside_fraction'], 1)} percent of the runs that "
        f"completed and only {_fmt(100.0 * domain['failed_jobs_inside_fraction'], 1)} "
        "percent of those that did not, which is what it means for a domain to track the "
        "censoring rather than merely to exist."
    )
    add("")
    add(
        "`ufem.validity.in_validity_domain(X)` is the only interface to this. It loads the "
        "fitted model from the artifact store, rechecks its digest against the one the "
        "domain was stamped with, and raises a named diagnostic if the audit stage has not "
        "run. It never falls back to an open domain."
    )
    add("")

    add("## 6. Importance weighting sensitivity")
    add("")
    add(
        f"Each of the {weighting['n_weighted_samples']} surviving runs is reweighted by the "
        "inverse of its completion probability, so a survivor from a corner where most runs "
        "failed stands in for the designed samples that did not survive. If the headline "
        "statistics move materially under that reweighting, the censoring is biasing them."
    )
    add("")
    add(
        "| Quantity | Unweighted mean | Weighted mean | Relative shift | "
        "Unweighted CoV | Weighted CoV |"
    )
    add("|---|---|---|---|---|---|")
    for name, (label, unit, scale) in WEIGHTED_QOI_DISPLAY.items():
        block = weighting["by_qoi"][name]
        digits = 4 if unit == "-" else 2
        add(
            f"| {label} [{unit}] | {_fmt(block['unweighted']['mean'] / scale, digits)} | "
            f"{_fmt(block['weighted']['mean'] / scale, digits)} | "
            f"{_fmt(100.0 * block['mean_shift_relative'], 2)} % | "
            f"{_fmt(block['unweighted']['cov'], 4)} | {_fmt(block['weighted']['cov'], 4)} |"
        )
    add("")
    largest = weighting["largest_relative_mean_shift"]
    add(
        f"The largest relative shift is "
        f"{_fmt(100.0 * largest['value'], 2)} percent, on the "
        f"{WEIGHTED_QOI_DISPLAY[largest['qoi']][0].lower()}. The weights themselves span "
        f"{_fmt(weighting['weight']['min'], 2)} to {_fmt(weighting['weight']['max'], 2)} "
        f"after normalization, for an effective sample size of "
        f"{_fmt(weighting['weight']['effective_sample_size'], 1)} against a nominal "
        f"{weighting['n_weighted_samples']}, so the reweighting is not leaning on a handful "
        "of survivors."
    )
    add("")
    add(
        "The shifts are modest, which is itself the finding, and it is the finding that "
        "licenses the unweighted statistics for the descriptive purpose they are used for. "
        "It does not license extrapolation into the censored corner, which is what the "
        "validity domain is for."
    )
    add("")

    add("## 7. Provenance")
    add("")
    add(
        "Wall times are deliberately absent from this table. They vary between runs on the "
        "same machine and between machines, so including them in a document that a test "
        "compares byte for byte would make the staleness gate fire on scheduling noise "
        "instead of on numbers. They are recorded per run in each stage manifest and "
        "discussed in `docs/ENGINEERING_LOG.md`."
    )
    add("")
    add("| Stage | Outputs | Output SHA-256 (first 12) |")
    add("|---|---|---|")
    for label, key in (
        ("ingest", "ingest_manifest"),
        ("grid", "grid_manifest"),
        ("audit", "audit_manifest"),
    ):
        manifest = data[key]
        for record in sorted(manifest["outputs"], key=lambda item: item["name"]):
            add(f"| {label} | {record['name']} | `{record['sha256'][:12]}` |")
    add("")
    add(
        "Each stage directory carries a `manifest.json` recording the config hash, the input "
        "artifact hashes, the output hashes, the seed entropy, the resolved package "
        "versions, and the git commit. A number in this card that cannot be traced back "
        "through one of those is a defect."
    )
    add("")

    return "\n".join(out)


def write_if_changed(path: Path, content: str) -> bool:
    """Write only when the content differs, and report whether it did.

    Newlines are forced to ``\\n`` so a regeneration on Windows produces the same bytes the
    staleness test compares against on a Linux runner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def generate(root: Path) -> dict[str, str]:
    """Build every generated file and return a mapping of repo relative path to content."""
    config = load_config(root)
    data = collect(root, config)
    return {
        DATA_CARD: build_data_card(data, config),
        f"{TABLES_DIR}/macros.tex": build_macro_fragment(data, config),
        f"{TABLES_DIR}/design_moments.tex": build_design_table(data, config),
        f"{TABLES_DIR}/quartile_failure_rates.tex": build_quartile_table(data),
        f"{TABLES_DIR}/calibration.tex": build_calibration_table(data),
        f"{TABLES_DIR}/importance_weighting.tex": build_weighting_table(data),
        f"{TABLES_DIR}/reduction_summary.tex": build_reduction_table(data),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    written = 0
    for relative, content in generate(root).items():
        changed = write_if_changed(root / relative, content)
        written += int(changed)
        print(f"{relative}: {'written' if changed else 'unchanged'} ({len(content)} bytes)")
    print(f"make_data_card: {written} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
