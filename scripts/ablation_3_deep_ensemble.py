"""Ablation 3 of build spec 10.6.3: a five member deep ensemble against the surrogate.

The prediction this script tests was committed to ``docs/ABLATIONS.md`` before the script
existed (ground rule 12), and the commit order in ``git log`` is the evidence.

Five multilayer perceptrons, each mapping the three standardized inputs straight to the 201
point load displacement curve, each with a Gaussian negative log likelihood head so it predicts
a variance per station as well as a mean. The ensemble prediction is the mixture of the five
Gaussians: the mean of the member means, and a total variance that is the mean of the member
variances plus the variance of the member means. That decomposition is what makes a deep
ensemble a predictive distribution rather than five point predictions with a spread beside them,
and it is what lets this ablation be scored on the same three uncertainty metrics as the
production surrogate.

The four metrics, all out of fold on the same ten grouped folds the P4 harness used, with the
production side recomputed on those folds by ``ufem.ablation_reference``:

1. pointwise root mean square error in N,
2. Gaussian negative log predictive density per station in nats,
3. empirical pointwise 90 percent coverage,
4. the per curve relative L2 the rest of the project reports.

No variance is floored, amplified, or clipped anywhere in this file (ground rule 4). The member
heads parameterize the log variance and the network is free to move it; if a predicted variance
came back non positive the script raises rather than repairing it, because a model that claims
certainty is a model to fix.

Run it from the repository root after `ufem run all`:

    .venv/Scripts/python scripts/ablation_3_deep_ensemble.py
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

from ufem.ablation_reference import (  # noqa: E402
    curve_metrics,
    load_curve_data,
    load_or_compute_reference,
    peak_metrics,
    production_folds,
    scored_stations,
)
from ufem.config import config_hash, load_config  # noqa: E402
from ufem.manifest import sha256_file, stage_dir, write_manifest  # noqa: E402
from ufem.surrogate import Standardizer, configure_torch  # noqa: E402

STAGE_NAME = "ablation_3_deep_ensemble"
OUT_JSON = "ablation_3_deep_ensemble.json"

#: Build spec 10.6.3 says five members, so five it is.
N_MEMBERS = 5

#: Two hidden layers of 64. Wider than the problem needs and narrower than the literature's
#: defaults, which is the honest middle for 178 training curves: the ablation is asking whether
#: this model class loses at this sample size, so crippling it would answer a different
#: question, and a 512 wide network would answer the same one more slowly.
HIDDEN_WIDTHS: tuple[int, int] = (64, 64)

EPOCHS = 3000
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4

#: Coverage level for both sides.
COVERAGE_LEVEL = 0.9

#: Spawn key for this ablation's seed tree (ground rule 13).
SPAWN_KEY: tuple[int, int] = (9, 3)


def _torch_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(np.random.default_rng(seed_sequence).integers(0, 2**62))


def build_member(n_outputs: int) -> Any:
    """One ensemble member: three inputs to a mean and a log variance per station.

    The head is the log variance rather than the variance, so positivity is a property of the
    parameterization instead of a constraint applied afterwards. Nothing is added to it and
    nothing bounds it from below: build spec ground rule 4 forbids a floor on a variance, and
    the check after training is a stop condition rather than a repair.
    """
    from torch import nn

    first, second = HIDDEN_WIDTHS
    return nn.Sequential(
        nn.Linear(3, first),
        nn.Tanh(),
        nn.Linear(first, second),
        nn.Tanh(),
        nn.Linear(second, 2 * n_outputs),
    )


def train_member(
    X: np.ndarray, Y: np.ndarray, seed_sequence: np.random.SeedSequence
) -> tuple[Any, dict[str, float]]:
    """Fit one member by full batch Adam on the Gaussian negative log likelihood."""
    import torch

    torch.manual_seed(_torch_seed(seed_sequence))
    inputs = torch.tensor(np.asarray(X, dtype=float))
    targets = torch.tensor(np.asarray(Y, dtype=float))
    n_outputs = targets.shape[1]
    model = build_member(n_outputs)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first = last = 0.0
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(inputs)
        mean, log_variance = output[:, :n_outputs], output[:, n_outputs:]
        loss = torch.mean(
            0.5 * (log_variance + (targets - mean) ** 2 * torch.exp(-log_variance))
        )
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        if epoch == 0:
            first = last
    model.eval()
    return model, {
        "initial_loss": first,
        "final_loss": last,
        "n_parameters": float(sum(p.numel() for p in model.parameters())),
    }


def member_prediction(model: Any, X: np.ndarray, n_outputs: int) -> tuple[np.ndarray, np.ndarray]:
    """One member's mean and variance in standardized curve units."""
    import torch

    with torch.no_grad():
        output = model(torch.tensor(np.asarray(X, dtype=float))).numpy().astype(float)
    return output[:, :n_outputs], np.exp(output[:, n_outputs:])


def ensemble_mixture(
    means: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The Gaussian mixture moments of an ensemble, given member means and variances.

    ``means`` and ``variances`` are ``(n_members, n_rows, n_points)``. The mixture mean is the
    average member mean and the mixture variance is the average member variance plus the
    variance of the member means, which is the law of total variance rather than a choice.
    """
    mu = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    if mu.shape != var.shape or mu.ndim != 3:
        raise ValueError(
            f"an ensemble mixture needs matching 3D arrays, got {mu.shape} and {var.shape}."
        )
    mixture_mean = mu.mean(axis=0)
    mixture_variance = var.mean(axis=0) + mu.var(axis=0)
    return mixture_mean, mixture_variance


def main() -> int:
    started = _time.perf_counter()
    configure_torch()
    config = load_config(REPO_ROOT)
    digest = config_hash(config)
    data = load_curve_data(REPO_ROOT, config, digest)
    reference = load_or_compute_reference(REPO_ROOT, config, digest)
    if reference.jobs != data.jobs:
        raise AssertionError(
            "the production reference and the gridded curves carry different job orders."
        )
    folds = production_folds(data.jobs, config)
    root = np.random.SeedSequence(config.pipeline.seed_entropy, spawn_key=SPAWN_KEY)
    fold_seeds = root.spawn(len(folds))

    n_runs, n_grid = data.force.shape
    ensemble_mean = np.full((n_runs, n_grid), np.nan)
    ensemble_variance = np.full((n_runs, n_grid), np.nan)
    single_member_mean = np.full((n_runs, n_grid), np.nan)
    training_seconds = 0.0
    training_log: list[dict[str, Any]] = []

    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(n_runs), test_index, assume_unique=False)
        feature_standardizer = Standardizer.fit(data.X[train_index])
        X_train = feature_standardizer.transform(data.X[train_index])
        X_test = feature_standardizer.transform(data.X[test_index])
        force_train = data.force[train_index]
        mean_curve = force_train.mean(axis=0)
        scale = float(force_train.std())
        if not scale > 0.0:
            raise ValueError(f"the training force family has zero spread in fold {index}.")
        Y_train = (force_train - mean_curve) / scale

        member_seeds = fold_seeds[index].spawn(N_MEMBERS)
        means = np.empty((N_MEMBERS, test_index.size, n_grid))
        variances = np.empty_like(means)
        train_started = _time.perf_counter()
        for member in range(N_MEMBERS):
            model, log = train_member(X_train, Y_train, member_seeds[member])
            mean, variance = member_prediction(model, X_test, n_grid)
            means[member] = mean
            variances[member] = variance
            training_log.append({"fold": index, "member": member, **log})
        training_seconds += _time.perf_counter() - train_started

        mixture_mean, mixture_variance = ensemble_mixture(means, variances)
        ensemble_mean[test_index] = mixture_mean * scale + mean_curve
        ensemble_variance[test_index] = mixture_variance * scale**2
        single_member_mean[test_index] = means[0] * scale + mean_curve
        print(
            f"[ablation 3] fold {index + 1}/{len(folds)}: {N_MEMBERS} members trained, "
            f"median predictive standard deviation "
            f"{np.median(np.sqrt(mixture_variance)) * scale:.1f} N"
        )

    if np.any(~np.isfinite(ensemble_mean)) or np.any(~np.isfinite(ensemble_variance)):
        raise AssertionError(
            "the folds left runs without an ensemble prediction, so the pooled metrics would "
            "be an average over an unknown sample."
        )
    if np.any(ensemble_variance <= 0.0):
        raise ValueError(
            "the ensemble returned a non positive predictive variance, so its log density is "
            "undefined. Ground rule 4 forbids flooring it; the model is the thing to fix."
        )

    stations = scored_stations(data.force, reference.force_variance, ensemble_variance)
    ablation = {
        "force": curve_metrics(
            data.force, ensemble_mean, ensemble_variance, COVERAGE_LEVEL, stations
        ),
        "peak": peak_metrics(data.force, ensemble_mean),
        "single_member_force": curve_metrics(data.force, single_member_mean, None),
    }
    production = {
        "force": curve_metrics(
            data.force,
            reference.force_mean,
            reference.force_variance,
            COVERAGE_LEVEL,
            stations,
        ),
        "peak": peak_metrics(data.force, reference.force_mean),
    }
    results: dict[str, Any] = {
        "config_sha256": digest,
        "n_runs": n_runs,
        "n_folds": len(folds),
        "architecture": {
            "n_members": N_MEMBERS,
            "hidden_widths": list(HIDDEN_WIDTHS),
            "activation": "tanh",
            "head": "mean and log variance per station, Gaussian negative log likelihood",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam, full batch",
            "device": "cpu",
        },
        "ablation": ablation,
        "production": production,
        "coverage_level": COVERAGE_LEVEL,
        "production_variance_note": (
            "the production pointwise variance is the linear amplitude propagation of build "
            "spec 10.4 plus the truncation residual, and it deliberately excludes the phase "
            "and displacement uncertainty, which enter the curve nonlinearly and are "
            "propagated by sampling in the calibration stage; both sides here are therefore "
            "uncalibrated predictive distributions"
        ),
        "training_wall_time_s": training_seconds,
        "training_log": training_log,
        "wall_time_s": _time.perf_counter() - started,
    }

    artifact_root = REPO_ROOT / config.pipeline.paths.artifact_root
    directory = stage_dir(artifact_root, STAGE_NAME, digest)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUT_JSON
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    reference_dir = stage_dir(artifact_root, "ablation_reference", digest)
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=digest,
        input_hashes={
            "ablation_reference/reference.json": sha256_file(reference_dir / "reference.json")
        },
        outputs=[path],
        seed_entropy=config.pipeline.seed_entropy,
        extra={
            "wall_time_s": results["wall_time_s"],
            "training_wall_time_s": training_seconds,
            "spawn_key": list(SPAWN_KEY),
            "architecture": results["architecture"],
            "ablation_force": ablation["force"],
            "production_force": production["force"],
        },
    )

    print(f"\nAblation 3, deep ensemble, {n_runs} runs over {len(folds)} grouped folds")
    for label, block in (("ablation", ablation), ("production", production)):
        force = block["force"]
        print(
            f"  {label:10s} RMSE {force['rmse']:8.1f} N   NLPD {force['nlpd']:7.3f}   "
            f"coverage {force['coverage']:.3f}   median relative L2 "
            f"{force['relative_l2_p50'] * 100:.2f} percent"
        )
    print(
        f"  one member alone reaches median relative L2 "
        f"{ablation['single_member_force']['relative_l2_p50'] * 100:.2f} percent"
    )
    print(f"  training {training_seconds:.1f} s of {results['wall_time_s']:.1f} s total")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
