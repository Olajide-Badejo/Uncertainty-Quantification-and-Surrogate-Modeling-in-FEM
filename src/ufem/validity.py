"""The validity domain contract: where the surrogate is allowed to speak.

Binding law 4, build spec 9.4. 202 of the 400 designed simulations produced nothing and the
failures cluster in the low top cover, high strength corner, so a model trained on the 198
survivors has a region where it is interpolating between real runs and a region where it is
extrapolating into a corner the solver itself could not complete. This module is the single
place that distinction is decided, so the surrogate stage, the propagation stage, and the UI
all consult one answer rather than each inventing their own.

The domain is the intersection of two conditions, both stamped into
``validity_domain.json`` by the audit stage:

1. the fitted completion model predicts ``P(complete) >= threshold``;
2. the query lies inside the box of the executed design.

The second condition is not redundant. A classifier asked about a point far outside the
design will answer, and the answer will be an extrapolation of a smooth kernel rather than
evidence. A high completion probability with no design points under it is precisely the
manufactured confidence of binding law 1.

Ground rule 8: nothing here falls back. If the audit stage has not run, or its artifacts do
not match the hashes it recorded, every function raises with a named diagnostic and the
command that fixes it. A validity check that silently answered True when it could not load
its model would be worse than no check at all.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ufem.audit import STAGE_NAME as AUDIT_STAGE
from ufem.audit import VALIDITY_DOMAIN_JSON
from ufem.config import FEATURE_ORDER, Config, config_hash, load_config
from ufem.manifest import sha256_file, stage_dir


class ValidityDomainUnavailable(RuntimeError):
    """The validity domain artifact cannot be loaded, and no default will be invented."""


@dataclass(frozen=True)
class ValidityDomain:
    """The loaded domain: the completion model, its threshold, and the design box."""

    model: Any
    threshold: float
    bounds: np.ndarray
    feature_order: tuple[str, ...]
    record: dict[str, Any]

    def completion_probability(self, X: np.ndarray) -> np.ndarray:
        """P(complete) for each row of an ``(n, 3)`` feature matrix."""
        return np.asarray(self.model.predict_proba(_as_matrix(X))[:, 1], dtype=float)

    def inside_design_box(self, X: np.ndarray) -> np.ndarray:
        """True where every feature lies within the executed design's box."""
        matrix = _as_matrix(X)
        return np.all(
            (matrix >= self.bounds[:, 0]) & (matrix <= self.bounds[:, 1]), axis=1
        )


def _as_matrix(X: Any) -> np.ndarray:
    """Coerce a frame or array to an ``(n, 3)`` float matrix in the feature order."""
    if hasattr(X, "columns"):
        from ufem.config import features

        return features(X)
    matrix = np.atleast_2d(np.asarray(X, dtype=float))
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"the validity domain takes an (n, {len(FEATURE_ORDER)}) matrix whose columns "
            f"are {list(FEATURE_ORDER)} in that order; got shape {matrix.shape}."
        )
    return matrix


def audit_stage_dir(repo_root: Path | str, config: Config | None = None) -> Path:
    """Where the audit stage wrote its artifacts for the current config."""
    root = Path(repo_root)
    resolved = config if config is not None else load_config(root)
    return stage_dir(
        root / resolved.pipeline.paths.artifact_root, AUDIT_STAGE, config_hash(resolved)
    )


def load_validity_domain(
    repo_root: Path | str, config: Config | None = None
) -> ValidityDomain:
    """Load the domain artifact, or raise naming exactly what is missing.

    The model's digest is rechecked against the one ``validity_domain.json`` recorded. A
    pickle that no longer hashes to its manifest entry is not the model the metrics were
    measured on, and using it anyway would break the chain of custody of binding law 5.
    """
    root = Path(repo_root)
    directory = audit_stage_dir(root, config)
    record_path = directory / VALIDITY_DOMAIN_JSON
    if not record_path.is_file():
        raise ValidityDomainUnavailable(
            f"no validity domain at {record_path}. The audit stage has not run for this "
            "config hash, so there is no measured region where the surrogate can be "
            "trusted. Run `ufem run audit` (which needs `ufem run ingest` and "
            "`ufem run grid` first). This does not fall back to an open domain: build spec "
            "binding law 4 requires every downstream product to carry a real one."
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))

    declared = tuple(record["feature_order"])
    if declared != FEATURE_ORDER:
        raise ValidityDomainUnavailable(
            f"the validity domain at {record_path} was built on features {declared}, but "
            f"the pinned feature contract is {FEATURE_ORDER}. The artifact predates a "
            "change to the feature order; rerun the audit stage rather than reordering the "
            "columns to fit."
        )

    model_path = directory / record["model_file"]
    if not model_path.is_file():
        raise ValidityDomainUnavailable(
            f"the validity domain at {record_path} references the completion model "
            f"{record['model_file']}, which is not present at {model_path}. Rerun "
            "`ufem run audit`; the domain is not usable without the model that defines it."
        )
    digest = sha256_file(model_path)
    if digest != record["model_sha256"]:
        raise ValidityDomainUnavailable(
            f"the completion model at {model_path} hashes to {digest} but the validity "
            f"domain recorded {record['model_sha256']}. The artifact has changed since the "
            "domain was stamped, so the threshold no longer describes this model. Rerun "
            "`ufem run audit --force`."
        )
    model = pickle.loads(model_path.read_bytes())

    bounds = np.array(
        [record["design_bounds"][name] for name in FEATURE_ORDER], dtype=float
    )
    return ValidityDomain(
        model=model,
        threshold=float(record["completion_threshold"]),
        bounds=bounds,
        feature_order=FEATURE_ORDER,
        record=record,
    )


@lru_cache(maxsize=4)
def _cached_domain(repo_root: str, config_digest: str) -> ValidityDomain:
    """Cache by (root, config hash) so repeated UI queries do not reload the pickle.

    The config hash is part of the key, so a config edit produces a different key rather
    than serving a domain fitted under different settings.
    """
    del config_digest
    return load_validity_domain(Path(repo_root))


def in_validity_domain(
    X: Any, repo_root: Path | str | None = None, config: Config | None = None
) -> np.ndarray:
    """Boolean array: is each query point inside the measured validity domain?

    ``X`` is an ``(n, 3)`` array or a frame carrying the three feature columns, in the
    contract order of :data:`ufem.config.FEATURE_ORDER`. A point is inside only when the
    completion model clears the stamped threshold and the point lies inside the executed
    design's box.

    Raises :class:`ValidityDomainUnavailable` when the audit stage has not run. That is the
    whole point of the contract: a downstream stage that cannot establish where it is
    allowed to predict must stop, not guess.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root_from_here()
    if config is not None:
        domain = load_validity_domain(root, config)
    else:
        resolved = load_config(root)
        domain = _cached_domain(str(root.resolve()), config_hash(resolved))
    matrix = _as_matrix(X)
    return np.asarray(
        (domain.completion_probability(matrix) >= domain.threshold)
        & domain.inside_design_box(matrix),
        dtype=bool,
    )


def _repo_root_from_here() -> Path:
    """The repository root, found from this file: ``src/ufem/validity.py`` is three deep."""
    return Path(__file__).resolve().parents[2]
