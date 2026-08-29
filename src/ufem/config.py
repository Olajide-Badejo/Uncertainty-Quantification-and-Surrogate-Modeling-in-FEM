"""The single source of truth for the configuration and the probabilistic model.

Binding law 2: the input random variables, their distributions, parameters, and couplings
are defined exactly once, in ``configs/probabilistic_model.yaml``, validated here, and
hashed into every artifact. This module is the only place in ``src/`` allowed to name a
distribution family or build a frozen ``scipy.stats`` object. Every other stage asks for
one through :func:`input_distributions`.

Units: strength in MPa, cover and displacement in mm, force in N.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

PROBABILISTIC_MODEL_FILE = "configs/probabilistic_model.yaml"
PIPELINE_FILE = "configs/pipeline.yaml"

#: The feature contract of build spec section 9.2, pinned in code as well as in YAML so a
#: config edit that reorders the columns is caught rather than silently honored.
FEATURE_ORDER: tuple[str, str, str] = (
    "Fcm_MPa",
    "c_nom_bottom_mm",
    "c_nom_top_mm",
)

PositiveFloat = Annotated[float, Field(gt=0.0)]


class _Frozen(BaseModel):
    """Base for every config model: immutable, and unknown keys are an error.

    Rejecting extras matters more than it looks. A silently ignored misspelled key is how a
    config drifts away from what the pipeline actually ran.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class LognormalVariable(_Frozen):
    """A lognormal input, parameterized by the mean and CoV of the variable itself."""

    kind: Literal["lognormal"]
    mean: PositiveFloat
    cov: PositiveFloat
    basis: str

    def log_params(self) -> tuple[float, float]:
        """Return ``(mu_ln, sigma_ln)``, the parameters of the underlying normal.

        For a lognormal with mean ``m`` and coefficient of variation ``v``:
        ``sigma_ln = sqrt(ln(1 + v**2))`` and ``mu_ln = ln(m) - sigma_ln**2 / 2``.
        """
        sigma_ln = math.sqrt(math.log(1.0 + self.cov**2))
        mu_ln = math.log(self.mean) - sigma_ln**2 / 2.0
        return mu_ln, sigma_ln


class NormalVariable(_Frozen):
    """A normal input, parameterized directly by mu and sigma."""

    kind: Literal["normal"]
    mu: float
    sigma: PositiveFloat
    basis: str


InputVariable = Annotated[LognormalVariable | NormalVariable, Field(discriminator="kind")]


class DerivedVariable(_Frozen):
    """A deterministic function of the independent inputs, never a surrogate feature."""

    expression: str
    basis: str


class SectionGeometry(_Frozen):
    depth_mm: PositiveFloat


class ProbabilisticModel(_Frozen):
    schema_version: int
    variables: dict[str, InputVariable]
    derived: dict[str, DerivedVariable]
    feature_order: list[str]
    section: SectionGeometry

    @model_validator(mode="after")
    def _check_contract(self) -> ProbabilisticModel:
        missing = [name for name in self.feature_order if name not in self.variables]
        if missing:
            raise ValueError(
                f"feature_order names variables that are not declared: {missing}. "
                f"Declared variables are {sorted(self.variables)}."
            )
        if "E_MPa" in self.feature_order:
            raise ValueError(
                "E_MPa is a derived variable and must never be a surrogate feature: it is "
                "the Eurocode 2 function of Fcm_MPa to within 2e-15, so feeding both makes "
                "ARD lengthscales and Sobol indices unidentifiable (build spec 5.4)."
            )
        if tuple(self.feature_order) != FEATURE_ORDER:
            raise ValueError(
                f"feature_order is {tuple(self.feature_order)} but the pinned contract is "
                f"{FEATURE_ORDER}. Column order is part of the artifact contract, so "
                "changing it is a deliberate code change, not a config edit."
            )
        if "E_MPa" not in self.derived:
            raise ValueError("derived.E_MPa is required: the modulus must come from one place.")
        self._check_geometry()
        return self

    def _check_geometry(self) -> None:
        """Reject a section the covers cannot physically fit inside."""
        depth = self.section.depth_mm
        for name in ("c_nom_bottom_mm", "c_nom_top_mm"):
            var = self.variables.get(name)
            if var is None:
                raise ValueError(f"{name} must be declared: it is part of the feature contract.")
            if not isinstance(var, NormalVariable):
                raise ValueError(f"{name} is declared as {var.kind}, expected normal.")
            if not 0.0 < var.mu < depth:
                raise ValueError(
                    f"{name} has mu {var.mu} mm, which is not inside the section depth "
                    f"{depth} mm. A cover outside the section is not a geometry."
                )
        bottom = self.variables["c_nom_bottom_mm"]
        top = self.variables["c_nom_top_mm"]
        assert isinstance(bottom, NormalVariable)
        assert isinstance(top, NormalVariable)
        if bottom.mu >= top.mu:
            raise ValueError(
                f"c_nom_bottom_mm mu {bottom.mu} mm is not below c_nom_top_mm mu {top.mu} mm. "
                "Both are measured from the soffit, so the bottom layer must sit lower."
            )


class GridSettings(_Frozen):
    u_min_mm: float
    u_max_mm: float
    n_points: int = Field(ge=2)

    @model_validator(mode="after")
    def _check_span(self) -> GridSettings:
        if self.u_max_mm <= self.u_min_mm:
            raise ValueError(
                f"grid spans [{self.u_min_mm}, {self.u_max_mm}] mm, which is not increasing."
            )
        return self


class CompletionModelSettings(_Frozen):
    """The completion probability classifier of build spec 9.4.

    ``primary`` is the estimator the stage tries first; ``fallback`` is the pre authorized
    substitute of spec 9.4, taken only when the primary fails one of the guards below. The
    guards are thresholds, not opinions, so which estimator shipped is a measured outcome
    recorded in the stage manifest rather than a silent choice.
    """

    primary: Literal["gaussian_process"]
    fallback: Literal["logistic"]
    matern_nu: PositiveFloat
    lengthscale_bounds: tuple[PositiveFloat, PositiveFloat]
    restarts: int = Field(ge=0)
    n_folds: int = Field(ge=2)
    n_bootstrap: int = Field(ge=1)
    interval_level: float = Field(gt=0.0, lt=1.0)
    n_calibration_bins: int = Field(ge=2)
    min_auc: float = Field(ge=0.0, le=1.0)
    min_prediction_spread: float = Field(ge=0.0, le=1.0)
    logistic_C: PositiveFloat

    @model_validator(mode="after")
    def _check_bounds(self) -> CompletionModelSettings:
        low, high = self.lengthscale_bounds
        if high <= low:
            raise ValueError(f"lengthscale_bounds {self.lengthscale_bounds} is not increasing.")
        return self


class ValidityDomainSettings(_Frozen):
    """Where the completion model says the data can be trusted (build spec 9.4)."""

    completion_threshold: float = Field(gt=0.0, lt=1.0)
    hull_expansion: float = Field(ge=0.0)
    grid_resolution: int = Field(ge=2)


class ImportanceWeightingSettings(_Frozen):
    """Inverse probability of completion weights, with the clip that keeps them finite."""

    min_probability: float = Field(gt=0.0, lt=1.0)


class AuditSettings(_Frozen):
    """Validity classification and censoring thresholds of build spec 9.4.

    None of these is the 198 sample list. The classification is derived from the ingest
    artifacts on every run; these are only the criteria it is derived by, which is the whole
    difference between a measurement and the hard coded literal of build spec 5.5.
    """

    u_start_tolerance_mm: PositiveFloat
    u_end_tolerance_mm: PositiveFloat
    target_step_time: PositiveFloat
    step_time_tolerance: float = Field(gt=0.0, lt=1.0)
    min_points: int = Field(ge=2)
    u_monotone_tolerance_mm: float = Field(ge=0.0)
    n_quantile_bins: int = Field(ge=2)
    significance_level: float = Field(gt=0.0, lt=1.0)
    completion_model: CompletionModelSettings
    validity_domain: ValidityDomainSettings
    importance_weighting: ImportanceWeightingSettings


class Normalizers(_Frozen):
    P0_N: PositiveFloat
    u0_mm: PositiveFloat


class PcaSettings(_Frozen):
    variance_target: float = Field(gt=0.0, le=1.0)


class KernelSettings(_Frozen):
    family: Literal["matern"]
    nu: float
    ard: bool
    lengthscale_bounds: tuple[PositiveFloat, PositiveFloat]
    restarts: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> KernelSettings:
        low, high = self.lengthscale_bounds
        if high <= low:
            raise ValueError(f"lengthscale_bounds {self.lengthscale_bounds} is not increasing.")
        return self


class ConformalSettings(_Frozen):
    alphas: list[float]
    K_posterior_draws: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_alphas(self) -> ConformalSettings:
        bad = [a for a in self.alphas if not 0.0 < a < 1.0]
        if bad:
            raise ValueError(f"conformal alphas must lie in (0, 1), got {bad}.")
        return self


class McSettings(_Frozen):
    n_samples: int = Field(ge=1)


class LimitStates(_Frozen):
    peak_load_below_N: PositiveFloat
    residual_ratio_below: float = Field(gt=0.0, lt=1.0)
    damage_at_10mm_above: float = Field(gt=0.0, lt=1.0)


class Paths(_Frozen):
    raw_data: str
    artifact_root: str


class PipelineConfig(_Frozen):
    schema_version: int
    grid: GridSettings
    audit: AuditSettings
    normalizers: Normalizers
    pca: PcaSettings
    kernel: KernelSettings
    conformal: ConformalSettings
    mc: McSettings
    limit_states: LimitStates
    seed_entropy: int = Field(ge=0)
    paths: Paths


class Config(_Frozen):
    """The fully resolved configuration. Its SHA-256 is the run identity."""

    probabilistic_model: ProbabilisticModel
    pipeline: PipelineConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}. Expected it relative to the repository root; "
            "run ufem from the repo root or pass an explicit root."
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} did not parse to a mapping, got {type(loaded)}.")
    return loaded


def load_config(repo_root: Path | str) -> Config:
    """Load and validate both YAML files into one frozen :class:`Config`."""
    root = Path(repo_root)
    return Config(
        probabilistic_model=ProbabilisticModel(**_read_yaml(root / PROBABILISTIC_MODEL_FILE)),
        pipeline=PipelineConfig(**_read_yaml(root / PIPELINE_FILE)),
    )


def config_hash(config: Config) -> str:
    """SHA-256 over the canonical JSON of the fully resolved config.

    Canonical means sorted keys, no insignificant whitespace, UTF-8. Two loads of the same
    files give the same digest; changing any parameter changes it.
    """
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_E(fcm_MPa: np.ndarray | float) -> np.ndarray | float:
    """Eurocode 2 secant modulus in MPa from mean compressive strength in MPa.

    ``E = 22000 * (Fcm / 10) ** 0.3``. This is the expression declared in
    ``configs/probabilistic_model.yaml``; it is reproduced here so callers get it as a
    function rather than by evaluating config strings.
    """
    return 22000.0 * (np.asarray(fcm_MPa, dtype=float) / 10.0) ** 0.3


def features(df: Any) -> np.ndarray:
    """The only constructor of the surrogate design matrix.

    Returns an ``(n, 3)`` float array whose columns are ``Fcm_MPa``, ``c_nom_bottom_mm``,
    ``c_nom_top_mm``, in that order. E is never a column. Raises ``KeyError`` naming the
    missing columns rather than reordering, guessing, or filling.
    """
    required = FEATURE_ORDER
    available = list(getattr(df, "columns", []))
    missing = [name for name in required if name not in available]
    if missing:
        raise KeyError(
            f"features(df) is missing required columns {missing}. The feature contract is "
            f"exactly {list(required)} in that order; the frame offers {available}."
        )
    return np.column_stack([np.asarray(df[name], dtype=float) for name in required])


def input_distributions(config: Config) -> dict[str, Any]:
    """Frozen ``scipy.stats`` distributions for the independent inputs, built from YAML.

    Returned in ``feature_order``. The lognormal is parameterized so that the returned
    object has exactly the declared mean and CoV: ``s = sigma_ln`` and
    ``scale = exp(mu_ln)``.
    """
    model = config.probabilistic_model
    out: dict[str, Any] = {}
    for name in model.feature_order:
        var = model.variables[name]
        if isinstance(var, LognormalVariable):
            mu_ln, sigma_ln = var.log_params()
            out[name] = stats.lognorm(s=sigma_ln, scale=math.exp(mu_ln))
        else:
            out[name] = stats.norm(loc=var.mu, scale=var.sigma)
    return out
