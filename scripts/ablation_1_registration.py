"""Ablation 1 of build spec 10.6.1: does registration before reduction actually pay?

The prediction this script tests was committed to ``docs/ABLATIONS.md`` before the script
existed (ground rule 12), and the commit order in ``git log`` is the evidence. Three metrics,
all model free, all measured on the same 198 curves:

1. **Components at 99 percent variance**, registered against unregistered.
2. **The spurious derivative mode**, ``|corr(PC2 loading, d mean/du)|`` on each side. Linear
   PCA cannot express "the same shape, shifted", so on a family with phase variation it
   approximates the shift with a derivative shaped mode, because f(u - d) is about
   f(u) - d f'(u) to first order.
3. **Peak load bias** of a rank k reconstruction, k being the registered component count, so
   both sides are compared at the same budget.

No surrogate, no Gaussian process, no cross validation: the claim under test is about the
representation, so the measurement is about the representation.

Run it from the repository root after `ufem run reduce`:

    .venv/Scripts/python scripts/ablation_1_registration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ufem.config import config_hash, load_config  # noqa: E402
from ufem.grid import RF2_GRID_PARQUET, displacement_grid  # noqa: E402
from ufem.grid import STAGE_NAME as GRID_STAGE  # noqa: E402
from ufem.manifest import stage_dir  # noqa: E402
from ufem.reduce import fit_basis  # noqa: E402
from ufem.register import (  # noqa: E402
    AMPLITUDE_PARQUET,
    WARP_PARQUET,
    curve_matrix,
    recover_unregistered,
)
from ufem.register import STAGE_NAME as REGISTER_STAGE  # noqa: E402

OUT_JSON = "ablation_1_registration.json"
OUT_TEX = REPO_ROOT / "report" / "tables" / "ablation_registration.tex"


def derivative_correlation(basis, mean_curve: np.ndarray, component: int = 1) -> float:
    """``|corr(PC_component loading, d mean/du)|``, the spurious derivative mode check.

    ``component`` is zero based, so the default of 1 is the second principal component, which
    is where the shift mode is expected: PC1 carries the overall amplitude and PC2 is the
    leading correction, which is exactly the slot a phase artifact occupies.
    """
    loading = np.asarray(basis.components[component], dtype=float)
    derivative = np.gradient(np.asarray(mean_curve, dtype=float))
    return float(abs(np.corrcoef(loading, derivative)[0, 1]))


def peak_bias(original: np.ndarray, rebuilt: np.ndarray) -> dict[str, float]:
    """Mean signed error of the reconstructed peak load, in N and as a fraction.

    Signed, deliberately: an absolute error would hide the direction, and the direction is
    the whole claim. Negative means the reconstruction under predicts the peak.
    """
    true_peak = np.asarray(original, dtype=float).max(axis=1)
    rebuilt_peak = np.asarray(rebuilt, dtype=float).max(axis=1)
    signed = rebuilt_peak - true_peak
    return {
        "mean_signed_error_N": float(signed.mean()),
        "mean_relative_error": float((signed / true_peak).mean()),
        "median_signed_error_N": float(np.median(signed)),
    }


def main() -> int:
    config = load_config(REPO_ROOT)
    digest = config_hash(config)
    artifact_root = REPO_ROOT / config.pipeline.paths.artifact_root
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, digest)
    grid_dir = stage_dir(artifact_root, GRID_STAGE, digest)
    for directory, stage in ((register_dir, REGISTER_STAGE), (grid_dir, GRID_STAGE)):
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(
                f"the {stage} stage has no artifacts at {directory}. Run `ufem run all` "
                "before this ablation: it measures artifacts and never recomputes them."
            )

    target = config.pipeline.pca.variance_target
    u_grid = displacement_grid(config)

    # The registered side: amplitude functions on the arc length grid, as the pipeline ships.
    jobs, registered = curve_matrix(pd.read_parquet(register_dir / AMPLITUDE_PARQUET))
    _warp_jobs, gamma = curve_matrix(pd.read_parquet(register_dir / WARP_PARQUET))
    registered_basis = fit_basis(registered, "registered", target)

    # The unregistered side: the curves exactly as they sit on the displacement grid, which
    # is what a pipeline without a registration step would reduce.
    grid_jobs, curves = curve_matrix(pd.read_parquet(grid_dir / RF2_GRID_PARQUET))
    if grid_jobs != jobs:
        raise AssertionError(
            "the gridded and registered artifacts carry different job orders, so the two "
            "sides of this ablation would not be the same curves."
        )
    unregistered_basis = fit_basis(curves, "unregistered", target)

    k = registered_basis.n_retained
    unregistered_rebuilt = unregistered_basis.reconstruct(
        unregistered_basis.project(curves, k)
    )

    # The registered side is reconstructed at the same rank and then carried back through its
    # own warps, so the peak is compared on the displacement grid in both cases. Comparing a
    # registered reconstruction against registered truth would flatter it by never asking the
    # phase to be reproduced at all.
    registered_rebuilt_amplitude = registered_basis.reconstruct(
        registered_basis.project(registered, k)
    )
    registered_rebuilt = np.vstack(
        [
            recover_unregistered(registered_rebuilt_amplitude[row], gamma[row])
            for row in range(len(jobs))
        ]
    )
    registered_reference = np.vstack(
        [recover_unregistered(registered[row], gamma[row]) for row in range(len(jobs))]
    )

    results = {
        "config_sha256": digest,
        "n_curves": len(jobs),
        "variance_target": float(target),
        "rank_for_peak_comparison": int(k),
        "components_at_target": {
            "registered": int(registered_basis.n_retained),
            "unregistered": int(unregistered_basis.n_retained),
            "ratio": float(unregistered_basis.n_retained / registered_basis.n_retained),
        },
        "derivative_mode_correlation": {
            "registered": derivative_correlation(registered_basis, registered.mean(axis=0)),
            "unregistered": derivative_correlation(unregistered_basis, curves.mean(axis=0)),
        },
        # Reported because the PC2 prediction was refuted and the obvious next question is
        # where the derivative structure actually went. This sweep is post hoc and is
        # labeled as such wherever it appears: it did not come with a committed prediction,
        # so it is a diagnostic, not evidence on the same footing as the three metrics above.
        "derivative_mode_correlation_by_component": {
            "registered": [
                derivative_correlation(registered_basis, registered.mean(axis=0), index)
                for index in range(6)
            ],
            "unregistered": [
                derivative_correlation(unregistered_basis, curves.mean(axis=0), index)
                for index in range(6)
            ],
        },
        "peak_load_bias": {
            "registered": peak_bias(registered_reference, registered_rebuilt),
            "unregistered": peak_bias(curves, unregistered_rebuilt),
        },
        "u_grid_span_mm": [float(u_grid[0]), float(u_grid[-1])],
    }

    out_dir = stage_dir(artifact_root, "ablation_1_registration", digest)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_JSON).write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    counts = results["components_at_target"]
    corr = results["derivative_mode_correlation"]
    bias = results["peak_load_bias"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(
        "\n".join(
            [
                "% Generated by scripts/ablation_1_registration.py. Do not edit.",
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Metric & Registered & Unregistered \\\\",
                "\\midrule",
                f"Components at {float(target) * 100:.0f}\\,\\% variance & "
                f"{counts['registered']} & {counts['unregistered']} \\\\",  # type: ignore[index]
                "$|\\mathrm{corr}(\\mathrm{PC}_2, \\mathrm{d}\\bar{f}/\\mathrm{d}u)|$ & "
                f"{corr['registered']:.3f} & {corr['unregistered']:.3f} \\\\",  # type: ignore[index]
                f"Peak load bias at rank {k} [N] & "
                f"{bias['registered']['mean_signed_error_N']:+.1f} & "  # type: ignore[index]
                f"{bias['unregistered']['mean_signed_error_N']:+.1f} \\\\",  # type: ignore[index]
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Ablation 1, registration, on {len(jobs)} curves at {target:.0%} variance")
    print(
        f"  components      registered {counts['registered']}, "  # type: ignore[index]
        f"unregistered {counts['unregistered']}, "  # type: ignore[index]
        f"ratio {counts['ratio']:.2f}x"  # type: ignore[index]
    )
    print(
        f"  |corr(PC2, dmean/du)|  registered {corr['registered']:.3f}, "  # type: ignore[index]
        f"unregistered {corr['unregistered']:.3f}"  # type: ignore[index]
    )
    print(
        f"  peak bias at rank {k}   registered "
        f"{bias['registered']['mean_signed_error_N']:+.1f} N, "  # type: ignore[index]
        f"unregistered {bias['unregistered']['mean_signed_error_N']:+.1f} N"  # type: ignore[index]
    )
    print(f"\nWrote {out_dir / OUT_JSON}\n      {OUT_TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
