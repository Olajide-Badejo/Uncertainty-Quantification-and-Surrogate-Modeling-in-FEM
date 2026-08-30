"""Stage ``reduce``: functional PCA on the registered amplitude, the phase, and the damage.

Build spec 10.2. Three separate reductions, because the three blocks live in different
spaces and mixing them would let one block's units dictate another's component count:

1. **Registered amplitude**, the output of the SRVF registration, in N against arc length.
2. **Warp tangent vectors**, the phase, already in the linear tangent space at the Karcher
   mean rather than on the constrained monotone manifold the warps themselves occupy.
3. **Damage curves**, raw on the displacement grid, deliberately unregistered. Build spec
   10.2 expects this family to be very low rank as it stands, and the stage records the
   measured count rather than assuming one.

Each block retains components to the configured explained variance target (99 percent), and
the count is a measurement recorded in the manifest, not a constant. The production basis is
fit on all 198 runs and stored with its mean, loadings, explained variance ratios, per job
scores, and reconstruction error percentiles. Build spec 10.2 also requires the basis to be
recomputed inside every cross validation fold; that is the validation stage's obligation at
P4, and this stage exists to provide the production basis and the honest reconstruction
error of it.

The solver is :func:`numpy.linalg.svd` on the centered matrix with ``full_matrices=False``,
not scikit-learn. Build spec 10.2 permits either, and the reason to hand roll about fifteen
lines here is control: the sign convention, the centering, and the truncation are all
explicit and testable rather than being defaults that could change under a library upgrade.
A deterministic full SVD also sidesteps the randomized solver entirely, which matters for the
bitwise reproducibility of build spec 17.2.

Units: the amplitude block carries force in N against dimensionless arc length, so its mean
and loadings are in N. The tangent and damage blocks are dimensionless. Explained variance
ratios and all reconstruction errors relative to the block norm are dimensionless.

RNG discipline (build spec 17.3): a full SVD is deterministic and this stage draws no random
numbers. The seed entropy is recorded in the manifest so the artifact chain stays uniform.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ufem.config import Config
from ufem.grid import DAMAGE_GRID_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.register import (
    AMPLITUDE_PARQUET,
    WARP_TANGENT_PARQUET,
    curve_matrix,
)
from ufem.register import STAGE_NAME as REGISTER_STAGE

STAGE_NAME = "reduce"

#: Output file names inside the stage directory.
BASIS_JSON = "pca_bases.json"
SCORES_PARQUET = "scores.parquet"
RECONSTRUCTION_JSON = "reconstruction_error.json"
TEX_FRAGMENT = "reduction_summary.tex"

#: The three reduction blocks, in the order they are reported.
BLOCK_AMPLITUDE = "amplitude"
BLOCK_PHASE = "phase"
BLOCK_DAMAGE = "damage"
BLOCKS: tuple[str, str, str] = (BLOCK_AMPLITUDE, BLOCK_PHASE, BLOCK_DAMAGE)

#: Percentiles of the per curve reconstruction error reported per block.
ERROR_PERCENTILES: tuple[int, int, int] = (50, 90, 99)


@dataclass(frozen=True)
class Basis:
    """One block's fitted PCA basis.

    ``mean`` is the column mean of the training matrix, ``components`` the full set of right
    singular vectors as rows, and ``explained_variance_ratio`` the per component share of the
    total variance. ``n_retained`` is how many components reach the configured target.
    """

    name: str
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    n_retained: int

    @property
    def n_features(self) -> int:
        return int(self.mean.size)

    def project(self, matrix: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Scores of ``matrix`` on this basis, ``(n_rows, n_components)``."""
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError(
                f"basis {self.name!r} expects rows of {self.n_features} features, got "
                f"shape {data.shape}."
            )
        k = self.components.shape[0] if n_components is None else int(n_components)
        return (data - self.mean) @ self.components[:k].T

    def reconstruct(self, scores: np.ndarray) -> np.ndarray:
        """Rebuild rows from scores, using as many components as the scores carry."""
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"reconstruct needs 2D scores, got shape {values.shape}.")
        k = values.shape[1]
        if k > self.components.shape[0]:
            raise ValueError(
                f"basis {self.name!r} holds {self.components.shape[0]} components but was "
                f"asked to reconstruct from {k}."
            )
        return values @ self.components[:k] + self.mean


def fit_basis(matrix: np.ndarray, name: str, variance_target: float) -> Basis:
    """Fit a PCA basis by full SVD of the centered matrix.

    The sign convention is pinned deliberately: each component is flipped so that its entry
    of largest magnitude is positive. An SVD determines its factors only up to a simultaneous
    sign flip of a singular vector pair, so without a convention the loadings and the scores
    can both change sign between library versions or platforms while describing exactly the
    same subspace. That would break the bitwise reproducibility gate and, worse, would flip
    the sign of any correlation a later stage reports against a score.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"fit_basis needs a 2D matrix, got shape {data.shape}.")
    n_rows = data.shape[0]
    if n_rows < 2:
        raise ValueError(
            f"block {name!r} has {n_rows} rows, too few to estimate a covariance structure."
        )
    if not 0.0 < variance_target <= 1.0:
        raise ValueError(
            f"the variance target must lie in (0, 1], got {variance_target}."
        )
    mean = data.mean(axis=0)
    centered = data - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    for row in range(vt.shape[0]):
        if vt[row, int(np.argmax(np.abs(vt[row])))] < 0.0:
            vt[row] = -vt[row]
    variance = singular**2 / (n_rows - 1)
    total = float(variance.sum())
    if total <= 0.0:
        raise ValueError(
            f"block {name!r} has zero total variance: every row is identical, so there is "
            "nothing for a basis to describe."
        )
    ratio = variance / total
    n_retained = int(np.searchsorted(np.cumsum(ratio), variance_target) + 1)
    n_retained = min(n_retained, int(vt.shape[0]))
    return Basis(
        name=name,
        mean=mean,
        components=vt,
        explained_variance=variance,
        explained_variance_ratio=ratio,
        n_retained=n_retained,
    )


def reconstruction_errors(
    matrix: np.ndarray, basis: Basis, n_components: int
) -> np.ndarray:
    """Per row relative L2 reconstruction error at a given truncation, dimensionless.

    The denominator is the row's own norm about the block mean rather than its raw norm, so
    the error is measured against the variation the basis is actually asked to describe. A
    raw norm denominator would flatter every block whose mean is large compared with its
    spread, which is exactly the case for these curve families.
    """
    data = np.asarray(matrix, dtype=float)
    scores = basis.project(data, n_components)
    rebuilt = basis.reconstruct(scores)
    residual = np.linalg.norm(data - rebuilt, axis=1)
    reference = np.linalg.norm(data - basis.mean, axis=1)
    if np.any(reference <= 0.0):
        raise ValueError(
            f"block {basis.name!r} holds a row identical to the block mean, so a relative "
            "reconstruction error is undefined for it."
        )
    return residual / reference


def error_percentiles(errors: np.ndarray) -> dict[str, float]:
    """The reported percentiles of a block's reconstruction error."""
    array = np.asarray(errors, dtype=float)
    return {f"p{value}": float(np.percentile(array, value)) for value in ERROR_PERCENTILES}


def _block_record(basis: Basis, errors: np.ndarray, n_rows: int) -> dict[str, object]:
    """Everything about one fitted block that goes into the JSON artifact."""
    retained = basis.n_retained
    return {
        "name": basis.name,
        "n_rows": int(n_rows),
        "n_features": basis.n_features,
        "n_retained": int(retained),
        "variance_explained_by_retained": float(
            basis.explained_variance_ratio[:retained].sum()
        ),
        "explained_variance_ratio": [float(v) for v in basis.explained_variance_ratio[:20]],
        "mean": [float(v) for v in basis.mean],
        "components": [[float(v) for v in row] for row in basis.components[:retained]],
        "reconstruction_error": error_percentiles(errors),
    }


def _tex_fragment(records: list[dict[str, object]], variance_target: float) -> str:
    """The report table fragment: one row per block, no hand typed numbers.

    Written as a ``tabular`` body only, so ``main.tex`` supplies the caption and the label
    and this file stays a pure data fragment.
    """
    lines = [
        "% Generated by ufem.reduce. Do not edit: regenerate with `ufem run reduce`.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Block & Curves & Points & Components & Variance & Median error \\\\",
        "\\midrule",
    ]
    for record in records:
        lines.append(
            f"{str(record['name']).capitalize()} & {record['n_rows']} & "
            f"{record['n_features']} & {record['n_retained']} & "
            f"{float(record['variance_explained_by_retained']) * 100:.2f}\\,\\% & "
            f"{float(record['reconstruction_error']['p50']) * 100:.2f}\\,\\% \\\\"  # type: ignore[index]
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% variance target {variance_target}",
        "",
    ]
    return "\n".join(lines)


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, dict[str, str]]:
    """Locate the register and grid artifacts this stage depends on."""
    artifact_root = root / config.pipeline.paths.artifact_root
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, config_sha256)
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    hashes = {}
    for directory, name, stage in (
        (register_dir, AMPLITUDE_PARQUET, REGISTER_STAGE),
        (register_dir, WARP_TANGENT_PARQUET, REGISTER_STAGE),
        (grid_dir, DAMAGE_GRID_PARQUET, GRID_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the reduce stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first: reduction consumes that stage's artifacts."
            )
        hashes[name] = sha256_file(path)
    return register_dir, grid_dir, hashes


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the reduce stage and return its artifact directory."""
    started = _time.perf_counter()
    root = Path(repo_root)
    register_dir, grid_dir, input_hashes = _load_inputs(root, config, config_sha256)

    amplitude_jobs, amplitude = curve_matrix(pd.read_parquet(register_dir / AMPLITUDE_PARQUET))
    phase_jobs, phase = curve_matrix(pd.read_parquet(register_dir / WARP_TANGENT_PARQUET))
    damage_jobs, damage = curve_matrix(pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET))
    if not amplitude_jobs == phase_jobs == damage_jobs:
        raise AssertionError(
            "the three reduction blocks carry different job orders, which would silently "
            "misalign every score row against its design point."
        )

    target = config.pipeline.pca.variance_target
    records: list[dict[str, object]] = []
    score_frame = pd.DataFrame({"job": pd.Series(amplitude_jobs, dtype="string")})
    errors_by_block: dict[str, np.ndarray] = {}
    for name, matrix in (
        (BLOCK_AMPLITUDE, amplitude),
        (BLOCK_PHASE, phase),
        (BLOCK_DAMAGE, damage),
    ):
        basis = fit_basis(matrix, name, target)
        errors = reconstruction_errors(matrix, basis, basis.n_retained)
        errors_by_block[name] = errors
        records.append(_block_record(basis, errors, matrix.shape[0]))
        scores = basis.project(matrix, basis.n_retained)
        for index in range(scores.shape[1]):
            score_frame[f"{name}_pc{index + 1}"] = scores[:, index]

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)

    basis_path = directory / BASIS_JSON
    basis_path.write_text(
        json.dumps(
            {"variance_target": float(target), "blocks": records}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    scores_path = directory / SCORES_PARQUET
    score_frame.to_parquet(scores_path, engine="pyarrow", compression="zstd", index=False)
    reconstruction_path = directory / RECONSTRUCTION_JSON
    reconstruction_path.write_text(
        json.dumps(
            {
                record["name"]: {
                    "n_retained": record["n_retained"],
                    **record["reconstruction_error"],  # type: ignore[dict-item]
                }
                for record in records
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tex_path = directory / TEX_FRAGMENT
    tex_path.write_text(_tex_fragment(records, float(target)), encoding="utf-8", newline="\n")

    counts = {str(record["name"]): int(record["n_retained"]) for record in records}  # type: ignore[arg-type]
    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "variance_target": float(target),
        "n_curves": len(amplitude_jobs),
        "components_retained": counts,
        "solver": "numpy.linalg.svd(centered, full_matrices=False)",
        "reconstruction_error": {
            str(record["name"]): record["reconstruction_error"] for record in records
        },
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=[basis_path, scores_path, reconstruction_path, tex_path],
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    summary = ", ".join(
        f"{name} {counts[name]} ({errors_by_block[name].mean() * 100:.2f}% mean error)"
        for name in BLOCKS
    )
    print(
        f"[reduce] {len(amplitude_jobs)} curves, components to "
        f"{target:.0%} variance: {summary}"
    )
    return directory
