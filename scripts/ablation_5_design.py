"""Ablation 5 of build spec 10.6.5: does a Sobol guided design beat a thinned one at small n?

The prediction this script tests was committed to ``docs/ABLATIONS.md`` before the script
existed (ground rule 12), and the commit order in ``git log`` is the evidence.

No new finite element run is available, so this is not a rerun of the campaign under a different
design. It is a subsampling study on the 198 runs that exist, and the honest statement of what
it measures goes in the report as well as here: how much the space filling quality of the
retained points moves the surrogate's error at a fixed budget, not what a Sobol campaign would
have produced.

At each n in {64, 128, 198}, subsets of that size are selected out of the 198 two ways, ten
seeded repetitions each:

- **Random subsets**, drawn without replacement. The stand in for thinning a Latin hypercube
  design, and the weak point of the study: a random subset of a Latin hypercube sample is not
  itself a Latin hypercube sample, it only inherits the parent's stratification in expectation.
- **Sobol guided selection.** A scrambled Sobol sequence of n points over the box spanned by
  the executed design, each point claiming the nearest not yet taken real design point in
  standardized coordinates, greedily in sequence order. The result is the subset of real runs
  that most nearly realizes a Sobol design.

Each subset gets the production peak load Gaussian process, the same kernel, bounds, restarts
and fitted noise the shipped surrogate uses, scored by the closed form leave one out of Dubrule
1983 at the fitted hyperparameters, which is the same estimator the P4 scalar harness reports.

At n = 198 both selections must return the whole population, so the two are the same set by
construction and the difference is exactly zero. That is arithmetic rather than convergence and
the script asserts it rather than presenting it as a measurement.

Run it from the repository root after `ufem run all`:

    .venv/Scripts/python scripts/ablation_5_design.py
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ufem.ablation_reference import load_curve_data  # noqa: E402
from ufem.config import config_hash, load_config  # noqa: E402
from ufem.manifest import sha256_file, stage_dir, write_manifest  # noqa: E402
from ufem.surrogate import GPSettings, Standardizer, configure_torch, fit_all  # noqa: E402
from ufem.validate import r2_score, rmse  # noqa: E402

STAGE_NAME = "ablation_5_design"
OUT_JSON = "ablation_5_design.json"

#: Build spec 10.6.5 names these three budgets.
SAMPLE_SIZES: tuple[int, int, int] = (64, 128, 198)

#: Repetitions per budget per method. Ten is small, and the spread across them is reported
#: beside every mean precisely so a reader can see whether a difference clears it.
N_REPETITIONS = 10

#: The target. Peak load is the quantity the reliability analysis thresholds and the one the
#: audit measured the strongest input relation on, so it is the scalar whose design sensitivity
#: matters most.
TARGET = "P_max_N"

#: Spawn key for this ablation's seed tree (ground rule 13).
SPAWN_KEY: tuple[int, int] = (9, 5)

METHOD_RANDOM = "random_subset"
METHOD_SOBOL = "sobol_guided"


def random_subset(n_total: int, n_take: int, seed_sequence: np.random.SeedSequence) -> np.ndarray:
    """``n_take`` distinct row indices drawn uniformly without replacement, sorted."""
    if not 1 <= n_take <= n_total:
        raise ValueError(f"cannot take {n_take} of {n_total} runs.")
    rng = np.random.default_rng(seed_sequence)
    return np.sort(rng.choice(n_total, size=n_take, replace=False))


def sobol_guided_subset(
    design: np.ndarray, n_take: int, seed_sequence: np.random.SeedSequence
) -> np.ndarray:
    """The ``n_take`` real design points that most nearly realize a scrambled Sobol design.

    The Sobol points are generated over the box the executed design spans, then each claims the
    nearest not yet taken real point in standardized coordinates, greedily in sequence order.
    Greedy rather than an optimal assignment on purpose: an optimal transport would be a
    different study, and the greedy rule is what a campaign planner with a fixed budget and a
    sequence to follow would actually do.

    The mapping is the honest weak point and is recorded as such: where the design is censored,
    a Sobol point can be far from every real run and still claim one.
    """
    points = np.asarray(design, dtype=float)
    n_total = points.shape[0]
    if not 1 <= n_take <= n_total:
        raise ValueError(f"cannot take {n_take} of {n_total} runs.")
    from scipy.stats import qmc

    standardizer = Standardizer.fit(points)
    standardized = standardizer.transform(points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    engine = qmc.Sobol(d=points.shape[1], scramble=True, rng=np.random.default_rng(seed_sequence))
    targets = standardizer.transform(qmc.scale(engine.random(n_take), lower, upper))

    taken: list[int] = []
    available = np.ones(n_total, dtype=bool)
    for row in range(n_take):
        distances = np.linalg.norm(standardized - targets[row], axis=1)
        distances[~available] = np.inf
        chosen = int(np.argmin(distances))
        if not np.isfinite(distances[chosen]):
            raise AssertionError(
                "the greedy claim ran out of unclaimed design points before the Sobol "
                "sequence ended, which cannot happen while n_take is at most the population."
            )
        taken.append(chosen)
        available[chosen] = False
    return np.sort(np.asarray(taken, dtype=int))


def leave_one_out_error(
    X: np.ndarray, y: np.ndarray, settings: GPSettings, seed_sequence: np.random.SeedSequence
) -> dict[str, float]:
    """Fit the production peak load process on a subset and score its closed form leave one out."""
    models, _feature_standardizer, target_standardizers, _log = fit_all(
        X, {TARGET: y}, settings, seed_sequence
    )
    mean, _variance = models[TARGET].leave_one_out()
    standardizer = target_standardizers[TARGET]
    predicted = standardizer.inverse_mean(mean.reshape(-1, 1)).ravel()
    return {
        "loo_rmse_N": rmse(y, predicted),
        "loo_r2": r2_score(y, predicted),
        "n": int(y.size),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n_repetitions": int(array.size),
    }


def main() -> int:
    started = _time.perf_counter()
    configure_torch()
    config = load_config(REPO_ROOT)
    digest = config_hash(config)
    data = load_curve_data(REPO_ROOT, config, digest)
    settings = GPSettings.from_config(config)
    y = data.qoi[TARGET].to_numpy(dtype=float)
    n_total = data.n_runs

    root = np.random.SeedSequence(config.pipeline.seed_entropy, spawn_key=SPAWN_KEY)
    size_seeds = dict(zip(SAMPLE_SIZES, root.spawn(len(SAMPLE_SIZES))))

    records: list[dict[str, Any]] = []
    degenerate: dict[str, bool] = {}
    for size in SAMPLE_SIZES:
        method_seeds = dict(zip((METHOD_RANDOM, METHOD_SOBOL), size_seeds[size].spawn(2)))
        selections: dict[str, list[np.ndarray]] = {}
        for method, seed in method_seeds.items():
            repetition_seeds = seed.spawn(N_REPETITIONS)
            selector = random_subset if method == METHOD_RANDOM else None
            chosen: list[np.ndarray] = []
            for repetition in range(N_REPETITIONS):
                draw, fit_seed = repetition_seeds[repetition].spawn(2)
                if selector is None:
                    index = sobol_guided_subset(data.X, size, draw)
                else:
                    index = selector(n_total, size, draw)
                chosen.append(index)
                score = leave_one_out_error(data.X[index], y[index], settings, fit_seed)
                records.append(
                    {"n": size, "method": method, "repetition": repetition, **score}
                )
            selections[method] = chosen
        degenerate[str(size)] = bool(
            size == n_total
            and all(
                np.array_equal(selections[method][repetition], np.arange(n_total))
                for method in selections
                for repetition in range(N_REPETITIONS)
            )
        )
        means = {
            method: float(
                np.mean(
                    [
                        record["loo_rmse_N"]
                        for record in records
                        if record["n"] == size and record["method"] == method
                    ]
                )
            )
            for method in (METHOD_RANDOM, METHOD_SOBOL)
        }
        print(
            f"[ablation 5] n = {size}: "
            + ", ".join(f"{method} LOO RMSE {value:.1f} N" for method, value in means.items())
        )

    if not degenerate[str(n_total)]:
        raise AssertionError(
            f"at n = {n_total} both selections must return the whole population, and they did "
            "not. Either the selection is dropping runs or the population is not what the "
            "study thinks it is."
        )

    summary: dict[str, Any] = {}
    for size in SAMPLE_SIZES:
        block: dict[str, Any] = {}
        for method in (METHOD_RANDOM, METHOD_SOBOL):
            values = [
                float(r["loo_rmse_N"])
                for r in records
                if r["n"] == size and r["method"] == method
            ]
            block[method] = summarize(values)
            block[f"{method}_r2"] = summarize(
                [
                    float(r["loo_r2"])
                    for r in records
                    if r["n"] == size and r["method"] == method
                ]
            )
        random_mean = block[METHOD_RANDOM]["mean"]
        sobol_mean = block[METHOD_SOBOL]["mean"]
        block["sobol_advantage_N"] = float(random_mean - sobol_mean)
        block["sobol_advantage_relative"] = float((random_mean - sobol_mean) / random_mean)
        block["repetition_spread_N"] = float(
            max(block[METHOD_RANDOM]["std"], block[METHOD_SOBOL]["std"])
        )
        block["advantage_clears_spread"] = bool(
            abs(block["sobol_advantage_N"]) > block["repetition_spread_N"]
        )
        block["selections_identical"] = degenerate[str(size)]
        summary[str(size)] = block

    results: dict[str, Any] = {
        "config_sha256": digest,
        "n_runs": n_total,
        "target": TARGET,
        "sample_sizes": list(SAMPLE_SIZES),
        "n_repetitions": N_REPETITIONS,
        "summary": summary,
        "records": records,
        "selection": {
            "random": "uniform without replacement over the 198 valid runs",
            "sobol": (
                "scrambled Sobol sequence over the box the executed design spans, each point "
                "claiming the nearest unclaimed real run in standardized coordinates"
            ),
            "caveat": (
                "this measures design sensitivity on an inherited censored campaign, not a "
                "rerun under a different design"
            ),
        },
        "wall_time_s": _time.perf_counter() - started,
    }

    artifact_root = REPO_ROOT / config.pipeline.paths.artifact_root
    directory = stage_dir(artifact_root, STAGE_NAME, digest)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUT_JSON
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    grid_dir = stage_dir(artifact_root, "grid", digest)
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=digest,
        input_hashes={"grid/qoi.parquet": sha256_file(grid_dir / "qoi.parquet")},
        outputs=[path],
        seed_entropy=config.pipeline.seed_entropy,
        extra={
            "wall_time_s": results["wall_time_s"],
            "spawn_key": list(SPAWN_KEY),
            "sample_sizes": list(SAMPLE_SIZES),
            "n_repetitions": N_REPETITIONS,
            "summary": summary,
        },
    )

    print(f"\nAblation 5, design subsampling, target {TARGET}, {N_REPETITIONS} repetitions")
    for size in SAMPLE_SIZES:
        block = summary[str(size)]
        print(
            f"  n = {size:3d}  random {block[METHOD_RANDOM]['mean']:7.1f} +/- "
            f"{block[METHOD_RANDOM]['std']:5.1f} N   sobol "
            f"{block[METHOD_SOBOL]['mean']:7.1f} +/- {block[METHOD_SOBOL]['std']:5.1f} N   "
            f"advantage {block['sobol_advantage_relative'] * 100:+.2f} percent"
            + ("  (identical sets by construction)" if block["selections_identical"] else "")
        )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
