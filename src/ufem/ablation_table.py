"""The ablations table of build spec 10.6, and the verdict logic behind it.

One row per ablation, one verdict per row, and every verdict decided by code rather than by
prose. That is the point of this module: the predictions were committed to
``docs/ABLATIONS.md`` before any of the five scripts ran, and the thresholds those predictions
named are transcribed here as named constants, so whether a prediction held is an evaluation of
the committed claim against the measured artifact instead of a sentence somebody wrote after
looking at the number.

Every threshold below carries the date its prediction was committed. Changing one after the
fact would be moving the goalposts, and the git history of this file is what makes that
visible.

The measurements themselves come from the five ablation scripts' JSON artifacts and are never
recomputed here. This module reads, compares, and formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ufem.manifest import stage_dir


class AblationMissing(FileNotFoundError):
    """An ablation has not been run for this config hash."""


@dataclass(frozen=True)
class AblationSource:
    """Where one ablation's artifact lives and what the table calls it."""

    number: int
    stage: str
    filename: str
    title: str
    script: str


ABLATIONS: tuple[AblationSource, ...] = (
    AblationSource(
        1,
        "ablation_1_registration",
        "ablation_1_registration.json",
        "Registration before reduction",
        "scripts/ablation_1_registration.py",
    ),
    AblationSource(
        2,
        "ablation_2_autoencoder",
        "ablation_2_autoencoder.json",
        "Autoencoder plus GP",
        "scripts/ablation_2_autoencoder.py",
    ),
    AblationSource(
        3,
        "ablation_3_deep_ensemble",
        "ablation_3_deep_ensemble.json",
        "Five member deep ensemble",
        "scripts/ablation_3_deep_ensemble.py",
    ),
    AblationSource(
        4,
        "ablation_4_bspline",
        "ablation_4_bspline.json",
        "B-spline coefficients",
        "scripts/ablation_4_bspline.py",
    ),
    AblationSource(
        5,
        "ablation_5_design",
        "ablation_5_design.json",
        "Sobol against thinning",
        "scripts/ablation_5_design.py",
    ),
)

#: Ablation 1, committed 2026-08-30: the unregistered representation needs 2 to 3 times the
#: components, its second component correlates with the mean derivative above 0.7 and clearly
#: above the registered side, and its peak reconstruction bias is negative and larger.
A1_COMPONENT_RATIO_RANGE: tuple[float, float] = (2.0, 3.0)
A1_DERIVATIVE_CORRELATION_FLOOR = 0.7

#: Ablation 2, committed 2026-08-31: worse force curve error, a peak load R2 below 0.6 off the
#: decoded curve, and worse damage curve error.
A2_PEAK_R2_CEILING = 0.6

#: Ablation 3, committed 2026-08-31: worse pointwise RMSE, worse NLPD, pointwise coverage below
#: nominal and further below it than the production surrogate's, and worse curve error.
A3_NOMINAL_COVERAGE = 0.9

#: Ablation 4, committed 2026-08-31: within 10 percent relative of the production curve error,
#: a negative peak bias larger in magnitude than the production one, and a blunter peak with a
#: larger curvature error.
A4_COMPETITIVE_TOLERANCE = 0.10

#: Ablation 5, committed 2026-08-31: the Sobol guided selection wins at n = 64, by under 10
#: percent relative, and the advantage shrinks by n = 128.
A5_SMALLEST_N = 64
A5_MIDDLE_N = 128
A5_ADVANTAGE_CEILING = 0.10


def load_payloads(artifact_root: Path | str, config_sha256: str) -> dict[int, dict[str, Any]]:
    """Read all five ablation artifacts, raising by name for the first one that is absent."""
    payloads: dict[int, dict[str, Any]] = {}
    for source in ABLATIONS:
        path = stage_dir(artifact_root, source.stage, config_sha256) / source.filename
        if not path.is_file():
            raise AblationMissing(
                f"ablation {source.number} has no result at {path}. Run "
                f"`python {source.script}` before building the ablations table: the report "
                "quotes its numbers and they must come from the artifact."
            )
        payloads[source.number] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


# ---------------------------------------------------------------------------
# The verdicts, one function per ablation, each evaluating its committed claims
# ---------------------------------------------------------------------------


def verdict_1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    counts = payload["components_at_target"]
    correlations = payload["derivative_mode_correlation"]
    bias = payload["peak_load_bias"]
    ratio = float(counts["ratio"])
    registered_bias = float(bias["registered"]["mean_signed_error_N"])
    unregistered_bias = float(bias["unregistered"]["mean_signed_error_N"])
    low, high = A1_COMPONENT_RATIO_RANGE
    return [
        {
            "claim": "unregistered needs 2 to 3 times the components",
            "held": bool(low <= ratio <= high),
        },
        {
            "claim": "unregistered PC2 correlates with the mean derivative above 0.7 and above "
            "the registered side",
            "held": bool(
                float(correlations["unregistered"]) > A1_DERIVATIVE_CORRELATION_FLOOR
                and float(correlations["unregistered"]) > float(correlations["registered"])
            ),
        },
        {
            "claim": "both peak biases negative, unregistered larger in magnitude",
            "held": bool(
                registered_bias < 0.0
                and unregistered_bias < 0.0
                and abs(unregistered_bias) > abs(registered_bias)
            ),
        },
    ]


def verdict_2(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ablation, production = payload["ablation"], payload["production"]
    return [
        {
            "claim": "worse out of sample force curve error than the production pipeline",
            "held": bool(
                ablation["force"]["relative_l2_p50"] > production["force"]["relative_l2_p50"]
            ),
        },
        {
            "claim": f"peak load R2 off the decoded curve below {A2_PEAK_R2_CEILING}",
            "held": bool(ablation["peak"]["peak_r2"] < A2_PEAK_R2_CEILING),
        },
        {
            "claim": "worse out of sample damage curve error than the production reduction",
            "held": bool(
                ablation["damage"]["relative_l2_p50"] > production["damage"]["relative_l2_p50"]
            ),
        },
    ]


def verdict_3(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ablation = payload["ablation"]["force"]
    production = payload["production"]["force"]
    return [
        {
            "claim": "worse pointwise RMSE",
            "held": bool(ablation["rmse"] > production["rmse"]),
        },
        {
            "claim": "worse negative log predictive density",
            "held": bool(ablation["nlpd"] > production["nlpd"]),
        },
        {
            "claim": "pointwise coverage below nominal and further below it than production",
            "held": bool(
                ablation["coverage"] < A3_NOMINAL_COVERAGE
                and ablation["coverage"] < production["coverage"]
            ),
        },
        {
            "claim": "worse median relative curve error",
            "held": bool(ablation["relative_l2_p50"] > production["relative_l2_p50"]),
        },
    ]


def verdict_4(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ablation, production = payload["ablation"], payload["production"]
    production_error = float(production["force"]["relative_l2_p50"])
    ablation_error = float(ablation["force"]["relative_l2_p50"])
    ablation_bias = float(ablation["peak"]["peak_bias_N"])
    production_bias = float(production["peak"]["peak_bias_N"])
    truth_magnitude = float(payload["truth_curvature_mean_magnitude"])
    return [
        {
            "claim": (
                f"median curve error within {A4_COMPETITIVE_TOLERANCE:.0%} relative of the "
                "production pipeline"
            ),
            "held": bool(
                abs(ablation_error - production_error) / production_error
                <= A4_COMPETITIVE_TOLERANCE
            ),
        },
        {
            "claim": "peak load under predicted, and by more than the production pipeline",
            "held": bool(ablation_bias < 0.0 and abs(ablation_bias) > abs(production_bias)),
        },
        {
            "claim": "blunter peak: smaller curvature magnitude than the truth and a larger "
            "curvature error than production",
            "held": bool(
                float(ablation["curvature"]["mean_magnitude"]) < truth_magnitude
                and float(ablation["curvature"]["mean_absolute_error"])
                > float(production["curvature"]["mean_absolute_error"])
            ),
        },
    ]


def verdict_5(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload["summary"]
    smallest = summary[str(A5_SMALLEST_N)]
    middle = summary[str(A5_MIDDLE_N)]
    advantage = float(smallest["sobol_advantage_relative"])
    return [
        {
            "claim": f"Sobol guided selection beats random subsets at n = {A5_SMALLEST_N}",
            "held": bool(advantage > 0.0),
        },
        {
            "claim": f"the advantage is under {A5_ADVANTAGE_CEILING:.0%} relative",
            "held": bool(abs(advantage) < A5_ADVANTAGE_CEILING),
        },
        {
            "claim": f"the advantage shrinks by n = {A5_MIDDLE_N}",
            "held": bool(
                abs(float(middle["sobol_advantage_relative"])) < abs(advantage)
            ),
        },
    ]


VERDICTS = {1: verdict_1, 2: verdict_2, 3: verdict_3, 4: verdict_4, 5: verdict_5}


def verdict_summary(payloads: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Every ablation's committed claims with the measured verdict on each."""
    out: dict[int, dict[str, Any]] = {}
    for source in ABLATIONS:
        claims = VERDICTS[source.number](payloads[source.number])
        held = sum(1 for claim in claims if claim["held"])
        out[source.number] = {
            "title": source.title,
            "claims": claims,
            "n_claims": len(claims),
            "n_held": held,
            "all_held": held == len(claims),
        }
    return out


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

#: The one headline metric each row reports, as (label, production value, ablation value), all
#: read from the artifacts. The row's verdict column carries how many of its committed claims
#: survived, because a single ablation makes more than one claim and collapsing that to one
#: word would hide which half failed.
def _row_values(number: int, payload: dict[str, Any]) -> tuple[str, str, str]:
    if number == 1:
        counts = payload["components_at_target"]
        return (
            "Components at 99\\,\\% of variance",
            str(int(counts["registered"])),
            str(int(counts["unregistered"])),
        )
    if number == 5:
        smallest = payload["summary"][str(A5_SMALLEST_N)]
        return (
            f"Leave one out RMSE at $n = {A5_SMALLEST_N}$ [N]",
            f"{smallest['random_subset']['mean']:.0f}",
            f"{smallest['sobol_guided']['mean']:.0f}",
        )
    return (
        "Median curve $L_2$ [\\%]",
        f"{100.0 * payload['production']['force']['relative_l2_p50']:.2f}",
        f"{100.0 * payload['ablation']['force']['relative_l2_p50']:.2f}",
    )


#: The predicted direction, as committed, in the few words a table cell holds. The full text is
#: in docs/ABLATIONS.md and the report points there.
PREDICTED_DIRECTION: dict[int, str] = {
    1: "Registration wins on components and peak bias",
    2: "Loses to the production pipeline",
    3: "Loses on RMSE, density, coverage, curve",
    4: "Competitive pointwise, blunter peak",
    5: "Sobol ahead at small $n$, gone by 198",
}


def build_ablations_table(payloads: dict[int, dict[str, Any]]) -> str:
    """The report fragment: one row per ablation, with its verdict decided by code.

    A ``tabular`` body only, so ``main.tex`` supplies the caption and the label and this stays
    a pure data fragment with no formatting opinions of its own.
    """
    verdicts = verdict_summary(payloads)
    lines = [
        "% Generated by scripts/make_data_card.py through ufem.ablation_table. Do not edit.",
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "Ablation & Headline metric & Production & Ablation & Claims held \\\\",
        "\\midrule",
    ]
    for source in ABLATIONS:
        label, production, ablation = _row_values(source.number, payloads[source.number])
        record = verdicts[source.number]
        lines.append(
            f"{source.number}. {source.title} & {label} & {production} & {ablation} & "
            f"{record['n_held']} of {record['n_claims']} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def build_ablation_macros(payloads: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Every ablation number the report prose quotes, as macro name to formatted body.

    Ablation 1's macros are older than this module and are defined by the data card generator
    itself; everything here belongs to ablations 2 through 5.
    """
    two, three, four, five = (payloads[number] for number in (2, 3, 4, 5))
    smallest = five["summary"][str(A5_SMALLEST_N)]
    middle = five["summary"][str(A5_MIDDLE_N)]
    largest = five["summary"][str(max(five["sample_sizes"]))]
    verdicts = verdict_summary(payloads)
    macros = {
        # Ablation 2
        "AblationTwoForceLTwo": f"{100.0 * two['ablation']['force']['relative_l2_p50']:.2f}",
        "AblationTwoDamageLTwo": f"{100.0 * two['ablation']['damage']['relative_l2_p50']:.2f}",
        "AblationTwoProductionDamageLTwo": (
            f"{100.0 * two['production']['damage']['relative_l2_p50']:.2f}"
        ),
        "AblationTwoPeakRSq": f"{two['ablation']['peak']['peak_r2']:.3f}",
        "AblationTwoProductionCurvePeakRSq": f"{two['production']['peak']['peak_r2']:.3f}",
        "AblationTwoProductionScalarPeakRSq": (
            f"{two['production_scalar_peak_r2']['grouped_fold']:.3f}"
        ),
        "AblationTwoForceLatent": str(int(two["architecture"]["force_latent"])),
        "AblationTwoDamageLatent": str(int(two["architecture"]["damage_latent"])),
        "AblationTwoEpochs": str(int(two["architecture"]["epochs"])),
        "AblationTwoClaimsHeld": str(verdicts[2]["n_held"]),
        # Ablation 3
        "AblationThreeForceLTwo": f"{100.0 * three['ablation']['force']['relative_l2_p50']:.2f}",
        "AblationThreeRMSE": f"{three['ablation']['force']['rmse']:.0f}",
        "AblationThreeProductionRMSE": f"{three['production']['force']['rmse']:.0f}",
        "AblationThreeNLPD": f"{three['ablation']['force']['nlpd']:.3f}",
        "AblationThreeProductionNLPD": f"{three['production']['force']['nlpd']:.3f}",
        "AblationThreeCoverage": f"{three['ablation']['force']['coverage']:.3f}",
        "AblationThreeProductionCoverage": f"{three['production']['force']['coverage']:.3f}",
        "AblationThreeMembers": str(int(three["architecture"]["n_members"])),
        "AblationThreeClaimsHeld": str(verdicts[3]["n_held"]),
        # Ablation 4
        "AblationFourForceLTwo": f"{100.0 * four['ablation']['force']['relative_l2_p50']:.2f}",
        "AblationFourPeakBias": f"{four['ablation']['peak']['peak_bias_N']:+.1f}",
        "AblationFourProductionPeakBias": f"{four['production']['peak']['peak_bias_N']:+.1f}",
        "AblationFourCurvatureError": (
            f"{four['ablation']['curvature']['mean_absolute_error']:.1f}"
        ),
        "AblationFourProductionCurvatureError": (
            f"{four['production']['curvature']['mean_absolute_error']:.1f}"
        ),
        "AblationFourTruthCurvature": f"{four['truth_curvature_mean_magnitude']:.1f}",
        "AblationFourBasisFunctions": str(int(four["basis"]["n_basis"])),
        "AblationFourClaimsHeld": str(verdicts[4]["n_held"]),
        # Ablation 5
        "AblationFiveRandomSmall": f"{smallest['random_subset']['mean']:.0f}",
        "AblationFiveSobolSmall": f"{smallest['sobol_guided']['mean']:.0f}",
        "AblationFiveSpreadSmall": f"{smallest['repetition_spread_N']:.0f}",
        "AblationFiveAdvantageSmall": f"{100.0 * smallest['sobol_advantage_relative']:+.2f}",
        "AblationFiveAdvantageMiddle": f"{100.0 * middle['sobol_advantage_relative']:+.2f}",
        "AblationFiveRandomFull": f"{largest['random_subset']['mean']:.0f}",
        "AblationFiveRepetitions": str(int(five["n_repetitions"])),
        "AblationFiveClaimsHeld": str(verdicts[5]["n_held"]),
    }
    return macros
