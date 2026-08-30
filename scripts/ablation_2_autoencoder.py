"""Ablation 2 of build spec 10.6.2: the predecessor's autoencoder architecture, done properly.

The prediction this script tests was committed to ``docs/ABLATIONS.md`` before the script
existed (ground rule 12), and the commit order in ``git log`` is the evidence.

Two small autoencoders on the raw gridded curves, one for force and one for damage, then the
same Gaussian process machinery the production surrogate uses on the latent codes. Everything
the predecessor got wrong about this architecture is corrected rather than reproduced:

- **Group aware folds.** The same ten grouped folds the P4 validation harness used, from the
  same seed, so the two sides are scored on identical held out sets. The predecessor split
  augmented children of one simulation across train and test.
- **A real fitted noise model.** The latent processes are the production ``fit_all`` with the
  configured Matern 5/2 kernel, the configured lengthscale bounds and the fitted noise under
  its LogNormal hyperprior. The predecessor fitted kernels with the noise pinned near zero.
- **The feature contract.** Three inputs, never the strength and the modulus as two features.
- **Everything refitted inside the fold.** The autoencoders, the curve standardization, the
  latent processes and the feature standardization all see the training 90 percent only.

The damage decoder is the salvaged idea of build spec 6.4 item 7: softplus increments through a
cumulative sum, so a decoded damage curve is non decreasing by construction. The terminal
renormalization the salvaged file applied afterwards is deliberately not carried over, because
dividing each decoded curve by its own last value forces every curve to end at the same height
and destroys the amplitude the surrogate exists to predict.

CPU only, float64, deterministic algorithms: build spec 17.2's production policy, which the
neural ablations of build spec 3 were allowed to relax on a GPU. This machine's torch is the
CPU build, so the relaxation does not arise and the wall times below are CPU wall times.

Run it from the repository root after `ufem run all`:

    .venv/Scripts/python scripts/ablation_2_autoencoder.py
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
from ufem.surrogate import GPSettings, configure_torch, fit_all  # noqa: E402
from ufem.validate import BASELINES_JSON, GP_MODEL  # noqa: E402
from ufem.validate import STAGE_NAME as VALIDATE_STAGE  # noqa: E402

STAGE_NAME = "ablation_2_autoencoder"
OUT_JSON = "ablation_2_autoencoder.json"

#: The architecture, stated rather than tuned. A fold trains on 178 curves of 201 points, which
#: is not a training set for a wide network, so the encoder narrows in two steps and the decoder
#: mirrors it. Hyperbolic tangent rather than the rectified linear unit of the salvaged file:
#: these curves are smooth, a piecewise linear decoder puts kinks in a reconstruction at exactly
#: the resolution the peak is read at, and at this width the vanishing gradient the rectifier
#: exists to avoid does not arise.
HIDDEN_WIDTHS: tuple[int, int] = (64, 32)

#: Latent widths. The force autoencoder's 8 coordinates sit between the 5 amplitude components
#: the production basis retains and the 23 score quantities it predicts to rebuild one force
#: curve, so the ablation is not handicapped on budget; the damage autoencoder's 4 sit below the
#: 11 the production damage basis retains, on a family that is close to degenerate.
FORCE_LATENT = 8
DAMAGE_LATENT = 4

#: Full batch Adam. Full batch because 178 curves is one batch on any machine, and because a
#: shuffled minibatch order is one more thing that has to be seeded to be reproducible.
EPOCHS = 3000
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-5

#: The timebox of the phase brief. If the autoencoder training alone crosses this, the run
#: records the overrun rather than silently costing the phase its budget.
TRAINING_BUDGET_S = 600.0

#: Coverage level for the metric block. The autoencoder offers no predictive variance, so no
#: coverage is reported for it; the level exists because the shared metric helper takes one for
#: the production side.
COVERAGE_LEVEL = 0.9

#: Spawn key for this ablation's seed tree, so its draws are reproducible and cannot collide
#: with the production run's spawned children (ground rule 13).
SPAWN_KEY: tuple[int, int] = (9, 2)


def _torch_seed(seed_sequence: np.random.SeedSequence) -> int:
    """A 63 bit seed for torch's global generator, taken from the project's seed tree."""
    return int(np.random.default_rng(seed_sequence).integers(0, 2**62))


def build_autoencoder(n_points: int, latent: int, monotone: bool) -> tuple[Any, Any]:
    """One autoencoder: a narrowing encoder and its mirror, optionally monotone by construction.

    ``monotone`` swaps the final linear layer of the decoder for the salvaged construction:
    softplus increments passed through a cumulative sum along the station axis. A cumulative
    sum of positive numbers is non decreasing whatever the weights are, which is why the
    property needs no penalty term and cannot be traded away by the optimizer. No terminal
    renormalization follows it, deliberately: see the module docstring.
    """
    import torch
    from torch import nn

    class MonotoneHead(nn.Module):
        def forward(self, increments):
            return torch.cumsum(nn.functional.softplus(increments), dim=-1)

    wide, narrow = HIDDEN_WIDTHS
    encoder = nn.Sequential(
        nn.Linear(n_points, wide),
        nn.Tanh(),
        nn.Linear(wide, narrow),
        nn.Tanh(),
        nn.Linear(narrow, latent),
    )
    layers: list[Any] = [
        nn.Linear(latent, narrow),
        nn.Tanh(),
        nn.Linear(narrow, wide),
        nn.Tanh(),
        nn.Linear(wide, n_points),
    ]
    if monotone:
        layers.append(MonotoneHead())
    decoder = nn.Sequential(*layers)
    return encoder, decoder


def train_autoencoder(
    curves: np.ndarray, latent: int, monotone: bool, seed_sequence: np.random.SeedSequence
) -> tuple[Any, Any, dict[str, float]]:
    """Fit one autoencoder on a training family and return its encoder, decoder and its log."""
    import torch

    torch.manual_seed(_torch_seed(seed_sequence))
    data = torch.tensor(np.asarray(curves, dtype=float))
    encoder, decoder = build_autoencoder(data.shape[1], latent, monotone)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    first = last = 0.0
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = torch.mean((decoder(encoder(data)) - data) ** 2)
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        if epoch == 0:
            first = last
    encoder.eval()
    decoder.eval()
    return (
        encoder,
        decoder,
        {
            "initial_loss": first,
            "final_loss": last,
            "epochs": float(EPOCHS),
            "n_parameters": float(sum(p.numel() for p in parameters)),
        },
    )


def encode(encoder: Any, curves: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return encoder(torch.tensor(np.asarray(curves, dtype=float))).numpy().astype(float)


def decode(decoder: Any, codes: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return decoder(torch.tensor(np.asarray(codes, dtype=float))).numpy().astype(float)


def latent_gp_predictions(
    codes: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> np.ndarray:
    """One production Gaussian process per latent coordinate, predicted at the held out rows."""
    targets = {f"latent_{index + 1}": codes[:, index] for index in range(codes.shape[1])}
    models, feature_standardizer, target_standardizers, _log = fit_all(
        X_train, targets, settings, seed_sequence
    )
    standardized = feature_standardizer.transform(X_test)
    out = np.empty((standardized.shape[0], codes.shape[1]))
    for index, name in enumerate(targets):
        mean, _variance = models[name].predict(standardized)
        out[:, index] = target_standardizers[name].inverse_mean(mean.reshape(-1, 1)).ravel()
    return out


def _production_scalar_r2(repo_root: Path, config: Any, digest: str) -> dict[str, float]:
    """The peak load R2 the validate stage measured for the production scalar process."""
    path = (
        stage_dir(repo_root / config.pipeline.paths.artifact_root, VALIDATE_STAGE, digest)
        / BASELINES_JSON
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["harness"]): float(row["r2_test"])
        for row in payload["scalar"]
        if str(row["model"]) == GP_MODEL and str(row["target"]) == "P_max_N"
    }


def main() -> int:
    started = _time.perf_counter()
    configure_torch()
    config = load_config(REPO_ROOT)
    digest = config_hash(config)
    data = load_curve_data(REPO_ROOT, config, digest)
    reference = load_or_compute_reference(REPO_ROOT, config, digest)
    if reference.jobs != data.jobs:
        raise AssertionError(
            "the production reference and the gridded curves carry different job orders, so "
            "the two sides of this ablation would not be the same runs."
        )
    settings = GPSettings.from_config(config)
    folds = production_folds(data.jobs, config)

    root = np.random.SeedSequence(config.pipeline.seed_entropy, spawn_key=SPAWN_KEY)
    fold_seeds = root.spawn(len(folds))

    n_runs, n_grid = data.force.shape
    force_prediction = np.full((n_runs, n_grid), np.nan)
    damage_prediction = np.full((n_runs, n_grid), np.nan)
    training_seconds = 0.0
    training_log: list[dict[str, Any]] = []

    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(n_runs), test_index, assume_unique=False)
        children = fold_seeds[index].spawn(4)
        force_train = data.force[train_index]
        damage_train = data.damage[train_index]

        # The force family is centered on its own training mean curve and divided by one
        # global scale rather than by a per station one: a per station scale would rescale the
        # softening tail against the peak and change which part of the curve the loss cares
        # about, which is a modeling decision disguised as preprocessing.
        mean_curve = force_train.mean(axis=0)
        scale = float(force_train.std())
        if not scale > 0.0:
            raise ValueError("the training force family has zero spread in fold " f"{index}.")
        standardized_force = (force_train - mean_curve) / scale

        train_started = _time.perf_counter()
        force_encoder, force_decoder, force_log = train_autoencoder(
            standardized_force, FORCE_LATENT, monotone=False, seed_sequence=children[0]
        )
        # The damage autoencoder works on raw damage values, not on a centered family: the
        # monotone decoder is monotone in the coordinate it outputs, and subtracting a mean
        # curve first would make it monotone in the wrong one.
        damage_encoder, damage_decoder, damage_log = train_autoencoder(
            damage_train, DAMAGE_LATENT, monotone=True, seed_sequence=children[1]
        )
        training_seconds += _time.perf_counter() - train_started

        force_codes = encode(force_encoder, standardized_force)
        damage_codes = encode(damage_encoder, damage_train)
        predicted_force_codes = latent_gp_predictions(
            force_codes, data.X[train_index], data.X[test_index], settings, children[2]
        )
        predicted_damage_codes = latent_gp_predictions(
            damage_codes, data.X[train_index], data.X[test_index], settings, children[3]
        )
        force_prediction[test_index] = decode(force_decoder, predicted_force_codes) * scale + (
            mean_curve
        )
        damage_prediction[test_index] = decode(damage_decoder, predicted_damage_codes)

        training_log.append({"fold": index, "force": force_log, "damage": damage_log})
        print(
            f"[ablation 2] fold {index + 1}/{len(folds)}: force reconstruction loss "
            f"{force_log['final_loss']:.3e}, damage {damage_log['final_loss']:.3e}"
        )

    if np.any(~np.isfinite(force_prediction)) or np.any(~np.isfinite(damage_prediction)):
        raise AssertionError(
            "the folds left runs without an autoencoder prediction, so the pooled metrics "
            "would be an average over an unknown sample."
        )

    monotone_violation = float(np.min(np.diff(damage_prediction, axis=1)))
    ablation = {
        "force": curve_metrics(data.force, force_prediction, None),
        "damage": curve_metrics(data.damage, damage_prediction, None),
        "peak": peak_metrics(data.force, force_prediction),
    }
    force_stations = scored_stations(data.force, reference.force_variance)
    damage_stations = scored_stations(data.damage, reference.damage_variance)
    production = {
        "force": curve_metrics(
            data.force,
            reference.force_mean,
            reference.force_variance,
            COVERAGE_LEVEL,
            force_stations,
        ),
        "damage": curve_metrics(
            data.damage,
            reference.damage_mean,
            reference.damage_variance,
            COVERAGE_LEVEL,
            damage_stations,
        ),
        "peak": peak_metrics(data.force, reference.force_mean),
    }
    scalar_r2 = _production_scalar_r2(REPO_ROOT, config, digest)

    results: dict[str, Any] = {
        "config_sha256": digest,
        "n_runs": n_runs,
        "n_folds": len(folds),
        "architecture": {
            "hidden_widths": list(HIDDEN_WIDTHS),
            "force_latent": FORCE_LATENT,
            "damage_latent": DAMAGE_LATENT,
            "activation": "tanh",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam, full batch",
            "damage_decoder": "softplus increments through a cumulative sum, no renormalization",
            "device": "cpu",
        },
        "ablation": ablation,
        "production": production,
        "production_scalar_peak_r2": scalar_r2,
        "damage_monotone_min_increment": monotone_violation,
        "training_wall_time_s": training_seconds,
        "training_budget_s": TRAINING_BUDGET_S,
        "training_within_budget": bool(training_seconds <= TRAINING_BUDGET_S),
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
            "force_relative_l2_p50": ablation["force"]["relative_l2_p50"],
            "production_force_relative_l2_p50": production["force"]["relative_l2_p50"],
        },
    )

    print(f"\nAblation 2, autoencoder plus GP, {n_runs} runs over {len(folds)} grouped folds")
    for signal in ("force", "damage"):
        print(
            f"  {signal:7s} median relative L2  ablation "
            f"{ablation[signal]['relative_l2_p50'] * 100:.2f} percent, production "
            f"{production[signal]['relative_l2_p50'] * 100:.2f} percent"
        )
    print(
        f"  peak load R2 off the curve   ablation {ablation['peak']['peak_r2']:.4f}, "
        f"production {production['peak']['peak_r2']:.4f}, production scalar process "
        f"{scalar_r2.get('grouped_fold', float('nan')):.4f}"
    )
    print(
        f"  decoded damage smallest increment {monotone_violation:.3e} "
        "(non negative means the decoder held its monotonicity)"
    )
    print(
        f"  training {training_seconds:.1f} s of {results['wall_time_s']:.1f} s total, "
        f"budget {TRAINING_BUDGET_S:.0f} s"
    )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
