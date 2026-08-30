"""Stage ``surrogate``: one exact Gaussian process per retained score and per scalar QoI.

Build spec 10.3 and 10.4. The reduction of phase P3 turned 198 load displacement curves into
a handful of coordinates apiece; this stage learns the map from the three independent inputs
to those coordinates, and back out to a curve with a predictive variance that was propagated
rather than chosen.

Four things in here are decisions rather than transcriptions of the spec, and each is
recorded in ``docs/DESIGN_DECISIONS.md`` as well as here:

1. **The phase block is capped at a few components.** The reduce stage retained 63 phase
   components to reach 99 percent of the phase variance. Fitting 63 independent Gaussian
   processes on 198 samples, for components that individually carry well under one percent of
   a block, is noise chasing with extra steps. The stage fits the phase components needed to
   reach ``surrogate.phase_variance_target`` of the phase block variance, capped at
   ``surrogate.phase_max_components``, and carries the phase variance it leaves behind in the
   reconstruction residual variance that build spec 10.4 already requires. What the cap costs
   is measured, not assumed, and it is in the manifest.

2. **A fourth reduction block, the displacement coordinate.** The register stage
   reparameterized every curve by arc length in the normalized load displacement plane, so
   the amplitude functions it produced are force against arc length, not force against
   displacement. A curve in the physical plane is a parametric pair, and predicting only the
   force half does not determine it. Two cheaper routes were tried first and both were
   rejected on measurement rather than on taste. Recovering the displacement half by
   quadrature from the arc length identity reproduces the training curves to a median relative
   L2 error of 10.3 percent and overshoots the 20 mm stroke by 7 percent, because the polyline
   that defines the arc length is not the polyline through the resampled stations; constraining
   the quadrature to land on 20 mm exactly fixes the overshoot and then stalls, leaving 36
   zero length steps per curve and only 37 of 198 curves usable. So the displacement
   coordinate is carried as its own block. It is monotone, and a linear principal component
   basis on a monotone function is not monotone: at rank 7 it produced a decreasing
   reconstruction for 125 of the 198 training curves, which ``numpy.interp`` would have turned
   into a plausible looking curve rather than an error. It is therefore represented the way
   P3 represents the warps, through psi = sqrt of the derivative into the tangent space at the
   Karcher mean, where monotonicity is structural.

3. **The noise is fitted under a hyperprior, and the hyperprior is a center, not a floor.**
   GPyTorch's default Gaussian likelihood carries a hard ``GreaterThan(1e-4)`` constraint on
   the noise, which is exactly the sigma floor that ground rule 4 bans, so it is replaced by a
   plain positivity transform and a LogNormal hyperprior. Where that prior is centered turned
   out to matter more than anything else in this stage. Centered at the solver's numerical
   resolution, which is what build spec 10.3 suggests, 36 of the 45 targets converged with a
   nugget of 1.8e-06 and every lengthscale pinned at its lower bound: the interpolate the
   scatter failure of build spec 5.2, reproduced. The center is now the share of a
   standardized target's variance the three inputs are not expected to explain, which is what
   this variance actually is, and the fit then leaves it by a factor of 25 in both directions
   across the target set. The full measurement is in docs/DESIGN_DECISIONS.md.

4. **Hyperparameters are optimized by L-BFGS-B on the raw parameters, not by fixed step
   Adam.** Adam at a fixed iteration budget stopped short of the optimum here (marginal
   likelihood 1.199 against 0.916 nats per point on the peak load) and cost three times the
   wall clock. Both optimizers are deterministic; only one of them converges inside the
   budget of build spec 10.3.

Units: the amplitude block is force in N against dimensionless arc length, the displacement
block is mm against the same, the damage block is dimensionless against displacement in mm,
and the scalar QoIs carry the units of the grid stage's QoI schedule. Features enter as
(MPa, mm, mm) and are standardized by the training statistics stored in the artifact.

Determinism (build spec 17.2): single threaded torch with deterministic algorithms on, float64
throughout, and every restart initialization drawn from a ``SeedSequence`` spawned off the
configured entropy. Nothing in this stage touches a torch random number generator. The gate is
a forced double run producing bitwise identical artifacts, asserted by
``tests/test_p4_determinism.py``.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.config import FEATURE_ORDER, Config, features
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET, displacement_grid
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.reduce import BASIS_JSON as REDUCE_BASIS_JSON
from ufem.reduce import STAGE_NAME as REDUCE_STAGE
from ufem.reduce import Basis, fit_basis
from ufem.register import (
    LANDMARKS_PARQUET,
    N_ARCLENGTH_POINTS,
    curve_matrix,
    normalized_arclength,
    recover_unregistered,
    resample_on_arclength,
    srsf_register,
    warp_tangent_vectors,
)
from ufem.register import STAGE_NAME as REGISTER_STAGE

STAGE_NAME = "surrogate"

#: Output file names inside the stage directory.
#:
#: Every array artifact is a plain ``.npy``, and the state dict is a float64 matrix rather
#: than a ``torch.save`` archive. That is a deliberate choice for the bitwise gate of build
#: spec 17.2, and it was measured rather than assumed: writing the same tensor twice a second
#: apart, ``numpy.save`` produced identical bytes and ``torch.save`` did not, because its zip
#: container records a local timestamp per entry. A ``.npy`` header holds only the dtype, the
#: shape and the order, so identical numbers give identical files.
GP_STATE_NPY = "gp_state.npy"
TRAINING_INPUTS_NPY = "training_inputs.npy"
TRAINING_TARGETS_NPY = "training_targets.npy"
BASIS_AMPLITUDE_NPY = "basis_amplitude.npy"
BASIS_PHASE_NPY = "basis_phase.npy"
BASIS_DAMAGE_NPY = "basis_damage.npy"
BASIS_DISPLACEMENT_NPY = "basis_displacement.npy"
SRSF_REFERENCE_NPY = "srsf_references.npy"
TRUNCATION_VARIANCE_NPY = "truncation_variance.npy"
SURROGATE_JSON = "surrogate.json"
RESTART_LOG_JSON = "restart_log.json"

#: The reduction blocks this stage predicts, in report order.
BLOCK_AMPLITUDE = "amplitude"
BLOCK_PHASE = "phase"
BLOCK_DAMAGE = "damage"
BLOCK_DISPLACEMENT = "displacement"
CURVE_BLOCKS: tuple[str, str, str, str] = (
    BLOCK_AMPLITUDE,
    BLOCK_PHASE,
    BLOCK_DAMAGE,
    BLOCK_DISPLACEMENT,
)

#: Scalar QoIs taken from the grid stage's table, in its own order.
SCALAR_QOI: tuple[str, ...] = (
    "P_max_N",
    "u_peak_mm",
    "k0_N_per_mm",
    "E_abs_Nmm",
    "P_residual_N",
    "softening_ratio",
    "u_damage_half_sat_mm",
    "damage_at_10mm",
)

#: Landmark quantities promoted to surrogate targets. Build spec 10.1 calls the landmarks
#: quantities of interest in their own right, and these three are complete over all 198 runs.
#:
#: ``u_85_mm`` is deliberately absent. One curve never falls to 85 percent of its peak, so
#: that column carries a NaN, and a Gaussian process fitted on the other 197 would be a model
#: trained on a silently different sample than every other target in this artifact. The
#: missing landmark is data (build spec 10.1 and the P3 decision record), not a gap to fill.
LANDMARK_QOI: tuple[str, ...] = ("u_knee_mm", "P_knee_N", "arclength_total")

#: The parameter blocks of one fitted GP, in the order they are flattened into ``gp_state``.
#: Pinned here so the artifact can be read back without a torch model in hand.
GP_PARAMETER_NAMES: tuple[str, ...] = (
    "likelihood.noise_covar.raw_noise",
    "mean_module.raw_constant",
    "covar_module.raw_outputscale",
    "covar_module.base_kernel.raw_lengthscale",
)

#: Restart initializations. The first two are fixed and deliberately bracket the interesting
#: corner of the likelihood surface: a short lengthscale with a large noise, and a long
#: lengthscale with a tiny one. The rest are drawn log uniform from the ranges below.
#:
#: The bracket is not decoration. With every restart started at a tiny noise, the optimizer
#: converged on the peak load to lengthscales of about 0.3, a noise of 1.1e-05 and a leave one
#: out R2 of 0.38, against 0.73 at the true optimum, which a restart started from a noise of
#: 0.3 finds immediately. That is the interpolate the scatter failure of build spec 5.2
#: reproduced by an initialization, and randomizing the noise as well as the lengthscale is
#: what kills it.
FIXED_RESTARTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 1.0e-1, 1.0),
    (3.0, 1.0e-3, 1.0),
)
RESTART_LENGTHSCALE_RANGE: tuple[float, float] = (0.1, 8.0)
RESTART_NOISE_RANGE: tuple[float, float] = (1.0e-6, 1.0)
RESTART_OUTPUTSCALE_RANGE: tuple[float, float] = (0.2, 5.0)

#: L-BFGS-B convergence tolerances, measured rather than guessed. Against ``ftol = 1e-12`` and
#: ``gtol = 1e-8``, these move the best marginal log likelihood of the worst of the 45 targets
#: by 7.3e-08 nats per point and the hyperparameters by at most 0.3 percent, with a median of
#: 0.013 percent, and they take 51.9 seconds of fitting instead of 67.2. The tighter setting
#: buys nothing a prediction can see and costs the build spec 10.3 budget, so it is not used.
OPTIMIZER_FTOL = 1.0e-9
OPTIMIZER_GTOL = 1.0e-5

#: The smallest positive value the softplus transformed parameters may take, and the box
#: bound L-BFGS-B is given so it never evaluates a point outside it.
#:
#: This is a domain guard on a transform, not a floor on an estimate, and the difference is
#: worth being precise about because ground rule 4 bans the second. GPyTorch's ``Positive``
#: constraint is a softplus, which underflows to exactly 0.0 for a raw value below about -745
#: in float64; the LogNormal hyperprior's log density at 0 is minus infinity, so the objective
#: stops being defined there and the optimizer crashes rather than converging. This bound sits
#: at 1e-12 in standardized variance units, six orders of magnitude below the smallest noise
#: any target in this campaign fits, and a test asserts that no fitted value ever sits near it.
#: Nothing here is ever added to, multiplied into, or clipped onto a reported variance.
TRANSFORM_DOMAIN_FLOOR = 1.0e-12

#: Percentiles reported for every error distribution in this stage.
ERROR_PERCENTILES: tuple[int, int, int] = (50, 90, 99)


def configure_torch() -> None:
    """Install the determinism policy of build spec 17.2 into the torch session.

    Single threaded so no reduction order depends on scheduling, deterministic algorithms on
    rather than warn only, and float64 everywhere because a surrogate whose Cholesky runs in
    float32 has no business reporting a variance to three digits.
    """
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Standardizer:
    """Column means and sample standard deviations, stored so predictions can be inverted."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardizer:
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Standardizer.fit needs a 2D matrix, got shape {data.shape}.")
        if data.shape[0] < 2:
            raise ValueError(
                f"Standardizer.fit needs at least two rows for a sample standard deviation, "
                f"got {data.shape[0]}."
            )
        scale = data.std(axis=0, ddof=1)
        degenerate = np.flatnonzero(scale <= 0.0)
        if degenerate.size:
            raise ValueError(
                f"columns {degenerate.tolist()} have zero spread, so standardizing them would "
                "divide by zero. A constant column is not a feature or a target."
            )
        return cls(mean=data.mean(axis=0), scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        data = np.atleast_2d(np.asarray(matrix, dtype=float))
        if data.shape[1] != self.mean.size:
            raise ValueError(
                f"this standardizer was fitted on {self.mean.size} columns but was given "
                f"{data.shape[1]}."
            )
        return (data - self.mean) / self.scale

    def inverse_mean(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float) * self.scale + self.mean

    def inverse_variance(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float) * self.scale**2


@dataclass(frozen=True)
class GPSettings:
    """Everything the kernel and the optimizer need, read from the validated config."""

    nu: float
    ard: bool
    lengthscale_bounds: tuple[float, float]
    restarts: int
    noise_prior_median_variance: float
    noise_prior_log_scale: float
    max_iterations: int

    @classmethod
    def from_config(cls, config: Config) -> GPSettings:
        kernel = config.pipeline.kernel
        surrogate = config.pipeline.surrogate
        return cls(
            nu=float(kernel.nu),
            ard=bool(kernel.ard),
            lengthscale_bounds=(
                float(kernel.lengthscale_bounds[0]),
                float(kernel.lengthscale_bounds[1]),
            ),
            restarts=int(kernel.restarts),
            noise_prior_median_variance=float(surrogate.noise_prior_median_variance),
            noise_prior_log_scale=float(surrogate.noise_prior_log_scale),
            max_iterations=int(surrogate.optimizer_max_iterations),
        )


def _build_model(x: Any, y: Any, settings: GPSettings) -> tuple[Any, Any, Any]:
    """Assemble the GPyTorch model, likelihood, and marginal likelihood for one target.

    Constant mean, Matern 5/2 with automatic relevance determination over the three
    standardized features, and a homoscedastic Gaussian likelihood whose noise is fitted under
    a LogNormal hyperprior. The noise constraint is ``Positive()``, a softplus transform with
    no lower bound, replacing GPyTorch's default ``GreaterThan(1e-4)``: that default is a hard
    floor on a variance, which is precisely what ground rule 4 forbids in a production path.
    """
    import gpytorch

    class _ExactGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=settings.nu,
                    ard_num_dims=train_x.shape[1] if settings.ard else None,
                    lengthscale_constraint=gpytorch.constraints.Interval(
                        *settings.lengthscale_bounds
                    ),
                )
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    prior = gpytorch.priors.LogNormalPrior(
        float(np.log(settings.noise_prior_median_variance)), settings.noise_prior_log_scale
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=prior, noise_constraint=gpytorch.constraints.Positive()
    )
    model = _ExactGP(x, y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def _flat_parameters(model: Any) -> np.ndarray:
    """The raw (unconstrained) parameters of one model as a flat float64 vector."""
    return np.concatenate(
        [parameter.detach().numpy().ravel().astype(float) for parameter in model.parameters()]
    )


def _set_flat_parameters(model: Any, vector: np.ndarray) -> None:
    """Write a flat vector back into a model's raw parameters, shape by shape."""
    import torch

    offset = 0
    for parameter in model.parameters():
        size = int(np.prod(parameter.shape)) if parameter.shape else 1
        chunk = np.asarray(vector[offset : offset + size], dtype=float)
        parameter.data = torch.tensor(chunk).reshape(parameter.shape)
        offset += size
    if offset != len(vector):
        raise ValueError(
            f"the parameter vector carries {len(vector)} values but the model takes {offset}."
        )


def _parameter_bounds(model: Any) -> list[tuple[float | None, float | None]]:
    """Box bounds on the raw parameters, one per scalar, in ``model.parameters()`` order.

    Only the two softplus transformed parameters, the noise and the outputscale, get a bound,
    and only from below, at the raw value whose softplus is
    :data:`TRANSFORM_DOMAIN_FLOOR`. The constant mean is unbounded and the lengthscales are
    bounded already by their own sigmoid transform onto the configured interval, so they need
    nothing here.
    """
    raw_floor = float(np.log(np.expm1(TRANSFORM_DOMAIN_FLOOR)))
    bounds: list[tuple[float | None, float | None]] = []
    for name, parameter in model.named_parameters():
        size = int(np.prod(parameter.shape)) if parameter.shape else 1
        softplus = name.endswith("raw_noise") or name.endswith("raw_outputscale")
        bounds.extend([(raw_floor, None) if softplus else (None, None)] * size)
    return bounds


def _restart_initializations(
    settings: GPSettings, n_features: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, float, float]]:
    """The ``restarts`` starting points: two fixed brackets, then log uniform draws."""
    low, high = settings.lengthscale_bounds
    inits: list[tuple[np.ndarray, float, float]] = []
    for lengthscale, noise, outputscale in FIXED_RESTARTS[: settings.restarts]:
        inits.append(
            (np.full(n_features, float(np.clip(lengthscale, low, high))), noise, outputscale)
        )
    while len(inits) < settings.restarts:
        lengthscale = np.exp(
            rng.uniform(
                np.log(max(RESTART_LENGTHSCALE_RANGE[0], low)),
                np.log(min(RESTART_LENGTHSCALE_RANGE[1], high)),
                size=n_features,
            )
        )
        noise = float(np.exp(rng.uniform(*np.log(RESTART_NOISE_RANGE))))
        outputscale = float(np.exp(rng.uniform(*np.log(RESTART_OUTPUTSCALE_RANGE))))
        inits.append((lengthscale, noise, outputscale))
    return inits


@dataclass(frozen=True)
class FittedGP:
    """One trained Gaussian process, held as its raw parameters plus its training data.

    The raw parameters are the unconstrained values GPyTorch optimizes, so restoring a model
    from them is exact rather than a round trip through a constrained value.
    """

    name: str
    parameters: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    settings: GPSettings
    marginal_log_likelihood: float
    #: Assembled model, built on first use. Assembling a GPyTorch module with its constraints
    #: and its prior costs about 25 ms, which is nothing once and 27 seconds over the 1080
    #: rebuilds a naive accessor per hyperparameter caused when this stage first ran.
    _cache: dict = dataclass_field(default_factory=dict, repr=False, compare=False)

    def _model(self) -> tuple[Any, Any]:
        import torch

        if "model" not in self._cache:
            x = torch.tensor(self.train_x)
            y = torch.tensor(self.train_y)
            model, likelihood, _mll = _build_model(x, y, self.settings)
            _set_flat_parameters(model, self.parameters)
            model.eval()
            likelihood.eval()
            self._cache["model"] = (model, likelihood)
        return self._cache["model"]

    def predict(self, X: np.ndarray, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance at ``X``, in standardized target units.

        ``include_noise`` adds the fitted observation noise, which is what a predictive
        interval for a new run needs. The latent function variance is what a decomposition of
        the curve variance needs, so both are reachable and neither is a default nobody chose.
        """
        import gpytorch
        import torch

        model, likelihood = self._model()
        query = torch.tensor(np.atleast_2d(np.asarray(X, dtype=float)))
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            latent = model(query)
            distribution = likelihood(latent) if include_noise else latent
            return (
                distribution.mean.numpy().astype(float),
                distribution.variance.numpy().astype(float),
            )

    def kernel_matrix(self) -> np.ndarray:
        """The training covariance including the fitted noise, in standardized units."""
        import torch

        model, likelihood = self._model()
        with torch.no_grad():
            x = torch.tensor(self.train_x)
            covariance = model.covar_module(x).to_dense()
            covariance = covariance + torch.eye(x.shape[0]) * likelihood.noise
            return covariance.numpy().astype(float)

    def constant_mean(self) -> float:
        import torch

        model, _likelihood = self._model()
        with torch.no_grad():
            return float(model.mean_module(torch.tensor(self.train_x))[0])

    def noise(self) -> float:
        _model, likelihood = self._model()
        return float(likelihood.noise.detach().numpy().ravel()[0])

    def lengthscales(self) -> np.ndarray:
        model, _likelihood = self._model()
        return (
            model.covar_module.base_kernel.lengthscale.detach().numpy().ravel().astype(float)
        )

    def outputscale(self) -> float:
        model, _likelihood = self._model()
        return float(model.covar_module.outputscale.detach().numpy())

    def leave_one_out(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed form leave one out mean and variance at fixed hyperparameters.

        Dubrule 1983, and Rasmussen and Williams section 5.4.2: one inverse of the training
        covariance gives all n folds, because dropping row i from a Gaussian conditional is an
        algebraic operation on the inverse rather than a refit.

        ``mu_-i = y_i - [K^-1 (y - m)]_i / [K^-1]_ii`` and ``var_-i = 1 / [K^-1]_ii``.

        The approximation is stated rather than hidden: the hyperparameters and the constant
        mean are those fitted on all n points, so each fold's model saw its own held out point
        through the hyperparameter fit. Build spec 11.1 calls for exactly this, with a per fold
        refit cross check alongside, and ``ufem.validate`` runs that cross check.
        """
        covariance = self.kernel_matrix()
        inverse = np.linalg.inv(covariance)
        residual = self.train_y - self.constant_mean()
        alpha = inverse @ residual
        diagonal = np.diag(inverse)
        if np.any(diagonal <= 0.0):
            raise ValueError(
                f"target {self.name!r} has a non positive diagonal in the inverse covariance, "
                "so the closed form leave one out is not defined. The covariance is not "
                "positive definite at these hyperparameters."
            )
        return self.train_y - alpha / diagonal, 1.0 / diagonal


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> tuple[FittedGP, list[dict[str, Any]]]:
    """Fit one GP by multi start L-BFGS-B on the marginal likelihood, best start wins.

    Every restart is logged with its initialization, its converged marginal likelihood, its
    hyperparameters, and its iteration count, so the dispersion between restarts is a recorded
    measurement rather than something to trust. A restart whose Cholesky fails is recorded as
    failed and skipped; if every restart fails the function raises, because a surrogate that
    quietly returned the least bad numerical accident would be worse than no surrogate.
    """
    import torch
    from linear_operator.utils.errors import NotPSDError
    from scipy.optimize import minimize

    inputs = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if inputs.ndim != 2 or targets.ndim != 1 or inputs.shape[0] != targets.size:
        raise ValueError(
            f"fit_gp needs an (n, d) design and an (n,) target, got {inputs.shape} and "
            f"{targets.shape} for {name!r}."
        )
    x_tensor = torch.tensor(inputs)
    y_tensor = torch.tensor(targets)
    rng = np.random.default_rng(seed_sequence)
    log: list[dict[str, Any]] = []
    best: FittedGP | None = None

    for index, (lengthscale, noise, outputscale) in enumerate(
        _restart_initializations(settings, inputs.shape[1], rng)
    ):
        model, likelihood, mll = _build_model(x_tensor, y_tensor, settings)
        model.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale).reshape(1, -1)
        model.covar_module.outputscale = torch.tensor(float(outputscale))
        likelihood.noise = torch.tensor(float(noise))
        model.train()
        likelihood.train()

        def objective(vector: np.ndarray, _model=model, _mll=mll) -> tuple[float, np.ndarray]:
            _set_flat_parameters(_model, vector)
            _model.zero_grad()
            loss = -_mll(_model(x_tensor), y_tensor)
            loss.backward()
            gradient = np.concatenate(
                [
                    parameter.grad.detach().numpy().ravel().astype(float)
                    for parameter in _model.parameters()
                ]
            )
            return float(loss.detach()), gradient

        record: dict[str, Any] = {
            "restart": index,
            "init_lengthscale": [float(value) for value in np.atleast_1d(lengthscale)],
            "init_noise": float(noise),
            "init_outputscale": float(outputscale),
        }
        try:
            result = minimize(
                objective,
                _flat_parameters(model),
                jac=True,
                method="L-BFGS-B",
                bounds=_parameter_bounds(model),
                options={
                    "maxiter": settings.max_iterations,
                    "ftol": OPTIMIZER_FTOL,
                    "gtol": OPTIMIZER_GTOL,
                },
            )
        except (NotPSDError, torch.linalg.LinAlgError, RuntimeError) as err:
            record.update({"status": "failed", "reason": f"{type(err).__name__}: {err}"})
            log.append(record)
            continue

        _set_flat_parameters(model, result.x)
        with torch.no_grad():
            value = float(mll(model(x_tensor), y_tensor))
        candidate = FittedGP(
            name=name,
            parameters=np.asarray(result.x, dtype=float),
            train_x=inputs,
            train_y=targets,
            settings=settings,
            marginal_log_likelihood=value,
        )
        with torch.no_grad():
            record.update(
                {
                    "status": "converged" if result.success else "iteration_limit",
                    "marginal_log_likelihood": value,
                    "n_function_evaluations": int(result.nfev),
                    "lengthscale": [
                        float(item)
                        for item in model.covar_module.base_kernel.lengthscale.numpy().ravel()
                    ],
                    "noise": float(likelihood.noise.numpy().ravel()[0]),
                    "outputscale": float(model.covar_module.outputscale.numpy()),
                }
            )
        log.append(record)
        if best is None or value > best.marginal_log_likelihood:
            best = candidate

    if best is None:
        reasons = "; ".join(str(item.get("reason", "unknown")) for item in log)
        raise RuntimeError(
            f"every one of the {settings.restarts} restarts for target {name!r} failed: "
            f"{reasons}. This is a stop condition, not something to work around."
        )
    return best, log


# ---------------------------------------------------------------------------
# The curve representation
# ---------------------------------------------------------------------------

#: How the amplitude family is aligned. ``srvf`` is the production path of build spec 10.1.
#: ``identity`` exists for the manufactured solution test of build spec 16.1, whose synthetic
#: curves are generated already aligned, so a registration would be fitting a warp to nothing
#: while costing the test its wall clock budget. No pipeline stage ever selects it, and the
#: stage manifest records which one ran.
REGISTRATION_SRVF = "srvf"
REGISTRATION_IDENTITY = "identity"


def _components_for_target(ratio: np.ndarray, target: float, cap: int) -> int:
    """Components reaching ``target`` of the explained variance, capped at ``cap``."""
    cumulative = np.cumsum(np.asarray(ratio, dtype=float))
    needed = int(np.searchsorted(cumulative, target) + 1)
    return int(min(max(needed, 1), cap, ratio.size))


def srsf_tangent(family: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a family of monotone functions onto the tangent space at its Karcher mean.

    ``family`` holds one monotone function per row, each rescaled to run from 0 to 1. Returns
    the shooting vectors and the Karcher mean psi, which is what :func:`srsf_curve` needs to
    invert the map. This is ``ufem.register.warp_tangent_vectors`` under a name that says what
    it does for a coordinate that is not a warp.
    """
    return warp_tangent_vectors(np.asarray(family, dtype=float))


def srsf_curve(mean_psi: np.ndarray, tangent: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """Invert :func:`srsf_tangent` for one tangent vector, giving a monotone map onto [0, 1].

    The exponential map at the Karcher mean lands on the unit sphere, a point of which is the
    square root of a derivative, so squaring and integrating gives a non decreasing function
    which is then normalized to the unit interval. Monotonicity is structural here rather than
    checked afterwards, which is the whole reason this representation is used.
    """
    from fdasrsf import geometry
    from scipy.integrate import cumulative_trapezoid

    psi = geometry.exp_map(
        np.asarray(mean_psi, dtype=float), np.asarray(tangent, dtype=float)
    )
    curve = cumulative_trapezoid(psi * psi, np.asarray(stations, dtype=float), initial=0.0)
    span = float(curve.max() - curve.min())
    if not span > 0.0:
        raise ValueError(
            "the reconstructed monotone map is constant, so it is not a reparameterization. "
            "The tangent vector it came from is not in the image of the log map."
        )
    return (curve - curve.min()) / span


@dataclass(frozen=True)
class CurveBasis:
    """The four block representation that turns scores back into a physical curve.

    Blocks: the registered amplitude and the phase from the SRVF registration of build spec
    10.1, the damage family raw on the displacement grid, and the displacement coordinate
    along the arc length stations, which is what makes the amplitude a curve in the physical
    plane rather than a function of a parameter.

    Two of the four are monotone maps of the unit interval, the phase and the displacement
    coordinate, and both live in the square root slope tangent space rather than in the space
    of the functions themselves.
    """

    u_grid: np.ndarray
    stations: np.ndarray
    u0_mm: float
    P0_N: float
    registration: str
    amplitude: Basis
    phase: Basis | None
    damage: Basis
    displacement: Basis
    n_amplitude: int
    n_phase: int
    n_damage: int
    n_displacement: int
    mean_psi_phase: np.ndarray | None
    mean_psi_displacement: np.ndarray
    scores: dict[str, np.ndarray]
    truncation_force_variance: np.ndarray
    truncation_damage_variance: np.ndarray
    reconstruction: dict[str, Any]

    @property
    def u_end_mm(self) -> float:
        return float(self.u_grid[-1])

    @property
    def block_counts(self) -> dict[str, int]:
        return {
            BLOCK_AMPLITUDE: self.n_amplitude,
            BLOCK_PHASE: self.n_phase,
            BLOCK_DAMAGE: self.n_damage,
            BLOCK_DISPLACEMENT: self.n_displacement,
        }

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        u_grid: np.ndarray,
        force: np.ndarray,
        damage: np.ndarray,
        config: Config,
        registration: str = REGISTRATION_SRVF,
    ) -> CurveBasis:
        """Build the representation from a training family of gridded curves."""
        u = np.asarray(u_grid, dtype=float)
        force_matrix = np.asarray(force, dtype=float)
        damage_matrix = np.asarray(damage, dtype=float)
        if force_matrix.shape != damage_matrix.shape:
            raise ValueError(
                f"the force family is {force_matrix.shape} and the damage family "
                f"{damage_matrix.shape}; they describe the same runs and must match."
            )
        if force_matrix.shape[1] != u.size:
            raise ValueError(
                f"the curves carry {force_matrix.shape[1]} points but the displacement grid "
                f"has {u.size}."
            )
        normalizers = config.pipeline.normalizers
        u0, P0 = float(normalizers.u0_mm), float(normalizers.P0_N)
        stations = np.linspace(0.0, 1.0, N_ARCLENGTH_POINTS)

        n_curves = force_matrix.shape[0]
        amplitude_raw = np.empty((n_curves, N_ARCLENGTH_POINTS), dtype=float)
        displacement = np.empty((n_curves, N_ARCLENGTH_POINTS), dtype=float)
        for row in range(n_curves):
            arclength = normalized_arclength(u, force_matrix[row], u0, P0)
            amplitude_raw[row] = resample_on_arclength(force_matrix[row], arclength)
            displacement[row] = resample_on_arclength(u, arclength)

        if registration == REGISTRATION_SRVF:
            registered, gamma = srsf_register(amplitude_raw, stations)
            phase_tangent, phase_reference = srsf_tangent(gamma)
        elif registration == REGISTRATION_IDENTITY:
            registered = amplitude_raw
            phase_tangent, phase_reference = None, None
        else:
            raise ValueError(
                f"registration must be {REGISTRATION_SRVF!r} or {REGISTRATION_IDENTITY!r}, "
                f"got {registration!r}."
            )

        u_end = float(u[-1])
        displacement_tangent, displacement_reference = srsf_tangent(displacement / u_end)

        target = float(config.pipeline.pca.variance_target)
        settings = config.pipeline.surrogate
        amplitude_basis = fit_basis(registered, BLOCK_AMPLITUDE, target)
        damage_basis = fit_basis(damage_matrix, BLOCK_DAMAGE, target)
        displacement_basis = fit_basis(displacement_tangent, BLOCK_DISPLACEMENT, target)
        n_displacement = _components_for_target(
            displacement_basis.explained_variance_ratio,
            float(settings.displacement_variance_target),
            int(settings.displacement_max_components),
        )
        if phase_tangent is None:
            phase_basis, n_phase = None, 0
        else:
            phase_basis = fit_basis(phase_tangent, BLOCK_PHASE, target)
            n_phase = _components_for_target(
                phase_basis.explained_variance_ratio,
                float(settings.phase_variance_target),
                int(settings.phase_max_components),
            )

        scores = {
            BLOCK_AMPLITUDE: amplitude_basis.project(registered, amplitude_basis.n_retained),
            BLOCK_DAMAGE: damage_basis.project(damage_matrix, damage_basis.n_retained),
            BLOCK_DISPLACEMENT: displacement_basis.project(
                displacement_tangent, n_displacement
            ),
        }
        if phase_basis is not None and phase_tangent is not None:
            scores[BLOCK_PHASE] = phase_basis.project(phase_tangent, n_phase)
        else:
            scores[BLOCK_PHASE] = np.zeros((n_curves, 0), dtype=float)

        basis = cls(
            u_grid=u,
            stations=stations,
            u0_mm=u0,
            P0_N=P0,
            registration=registration,
            amplitude=amplitude_basis,
            phase=phase_basis,
            damage=damage_basis,
            displacement=displacement_basis,
            n_amplitude=int(amplitude_basis.n_retained),
            n_phase=int(n_phase),
            n_damage=int(damage_basis.n_retained),
            n_displacement=int(n_displacement),
            mean_psi_phase=phase_reference,
            mean_psi_displacement=displacement_reference,
            scores=scores,
            truncation_force_variance=np.zeros(u.size),
            truncation_damage_variance=np.zeros(u.size),
            reconstruction={},
        )
        return basis._with_truncation(force_matrix, damage_matrix)

    def _with_truncation(
        self, force: np.ndarray, damage: np.ndarray
    ) -> CurveBasis:
        """Measure the truncation residual of build spec 10.4 on the training family.

        The residual is what the retained ranks cannot express, evaluated with the TRUE scores
        of each training curve, so it isolates the truncation from any regression error. It is
        reported as a pointwise mean square rather than a variance about its own mean, because
        a truncation residual has a bias and pretending otherwise would understate it.

        In sample, therefore a lower bound on the honest out of sample truncation. The fold
        harness of ``ufem.validate`` measures the out of sample version, and the report states
        both.
        """
        force_hat = self.reconstruct_force(
            self.scores[BLOCK_AMPLITUDE],
            self.scores[BLOCK_PHASE],
            self.scores[BLOCK_DISPLACEMENT],
        )
        damage_hat = self.reconstruct_damage(self.scores[BLOCK_DAMAGE])
        force_residual = np.asarray(force, dtype=float) - force_hat
        damage_residual = np.asarray(damage, dtype=float) - damage_hat
        relative = np.linalg.norm(force_residual, axis=1) / np.linalg.norm(
            np.asarray(force, dtype=float), axis=1
        )
        relative_damage = np.linalg.norm(damage_residual, axis=1) / np.linalg.norm(
            np.asarray(damage, dtype=float), axis=1
        )
        reconstruction = {
            "force_relative_l2": {
                f"p{value}": float(np.percentile(relative, value))
                for value in ERROR_PERCENTILES
            },
            "damage_relative_l2": {
                f"p{value}": float(np.percentile(relative_damage, value))
                for value in ERROR_PERCENTILES
            },
            "force_relative_l2_mean": float(relative.mean()),
            "damage_relative_l2_mean": float(relative_damage.mean()),
            "basis": (
                "training family, true scores at the retained ranks; in sample and therefore "
                "a lower bound on the out of sample truncation residual"
            ),
        }
        return CurveBasis(
            u_grid=self.u_grid,
            stations=self.stations,
            u0_mm=self.u0_mm,
            P0_N=self.P0_N,
            registration=self.registration,
            amplitude=self.amplitude,
            phase=self.phase,
            damage=self.damage,
            displacement=self.displacement,
            n_amplitude=self.n_amplitude,
            n_phase=self.n_phase,
            n_damage=self.n_damage,
            n_displacement=self.n_displacement,
            mean_psi_phase=self.mean_psi_phase,
            mean_psi_displacement=self.mean_psi_displacement,
            scores=self.scores,
            truncation_force_variance=(force_residual**2).mean(axis=0),
            truncation_damage_variance=(damage_residual**2).mean(axis=0),
            reconstruction=reconstruction,
        )

    # -- reconstruction -----------------------------------------------------

    def warp_from_tangent(self, tangent: np.ndarray) -> np.ndarray:
        """The phase warp implied by a tangent vector, or the identity when unregistered."""
        if self.mean_psi_phase is None:
            return self.stations.copy()
        return srsf_curve(self.mean_psi_phase, tangent, self.stations)

    def displacement_from_tangent(self, tangent: np.ndarray) -> np.ndarray:
        """The displacement coordinate in mm along the arc length stations."""
        return (
            srsf_curve(self.mean_psi_displacement, tangent, self.stations) * self.u_end_mm
        )

    def _amplitude_on_displacement(
        self,
        amplitude_functions: np.ndarray,
        tangent: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        """Carry one or more registered amplitude functions out to the displacement grid.

        For a fixed warp and a fixed displacement coordinate this map is linear in the
        amplitude values, which is what lets the same routine transport both the predicted
        mean curve and the amplitude loadings, and therefore what makes the pointwise variance
        of build spec 10.4 a propagated quantity rather than an assumed one.
        """
        gamma = self.warp_from_tangent(tangent)
        functions = np.atleast_2d(np.asarray(amplitude_functions, dtype=float))
        out = np.empty((functions.shape[0], self.u_grid.size), dtype=float)
        for row in range(functions.shape[0]):
            on_stations = recover_unregistered(functions[row], gamma, self.stations)
            out[row] = np.interp(self.u_grid, displacement, on_stations)
        return out

    def _phase_tangent_from_scores(self, phase_scores: np.ndarray) -> np.ndarray:
        if self.phase is None or self.n_phase == 0:
            return np.zeros(self.stations.size, dtype=float)
        return self.phase.reconstruct(
            np.asarray(phase_scores, dtype=float).reshape(1, -1)[:, : self.n_phase]
        )[0]

    def _displacement_from_scores(self, displacement_scores: np.ndarray) -> np.ndarray:
        tangent = self.displacement.reconstruct(
            np.asarray(displacement_scores, dtype=float).reshape(1, -1)[
                :, : self.n_displacement
            ]
        )[0]
        return _monotone_displacement(self.displacement_from_tangent(tangent))

    def reconstruct_force(
        self,
        amplitude_scores: np.ndarray,
        phase_scores: np.ndarray,
        displacement_scores: np.ndarray,
    ) -> np.ndarray:
        """Physical load displacement curves in N on the common displacement grid."""
        amplitude = np.atleast_2d(np.asarray(amplitude_scores, dtype=float))
        phase = np.atleast_2d(np.asarray(phase_scores, dtype=float))
        displacement = np.atleast_2d(np.asarray(displacement_scores, dtype=float))
        if not amplitude.shape[0] == phase.shape[0] == displacement.shape[0]:
            raise ValueError(
                f"the three score blocks describe different numbers of curves: "
                f"{amplitude.shape[0]}, {phase.shape[0]}, {displacement.shape[0]}."
            )
        registered = self.amplitude.reconstruct(amplitude[:, : self.n_amplitude])
        out = np.empty((amplitude.shape[0], self.u_grid.size), dtype=float)
        for row in range(amplitude.shape[0]):
            out[row] = self._amplitude_on_displacement(
                registered[row],
                self._phase_tangent_from_scores(phase[row]),
                self._displacement_from_scores(displacement[row]),
            )[0]
        return out

    def reconstruct_damage(self, damage_scores: np.ndarray) -> np.ndarray:
        """Damage curves on the common displacement grid, dimensionless."""
        scores = np.atleast_2d(np.asarray(damage_scores, dtype=float))
        return self.damage.reconstruct(scores[:, : self.n_damage])

    def force_loadings_on_displacement(
        self, phase_scores: np.ndarray, displacement_scores: np.ndarray
    ) -> np.ndarray:
        """The retained amplitude loadings carried out to the displacement grid.

        These are the ``phi_k(u)`` of build spec 10.4's ``Var f(u) = sum_k phi_k(u)^2
        sigma_k^2``, expressed in the coordinate the curve is reported in.
        """
        return self._amplitude_on_displacement(
            self.amplitude.components[: self.n_amplitude],
            self._phase_tangent_from_scores(np.asarray(phase_scores, dtype=float).ravel()),
            self._displacement_from_scores(displacement_scores),
        )


def _monotone_displacement(values: np.ndarray) -> np.ndarray:
    """Guard the reconstructed displacement coordinate before it is used as an abscissa.

    A truncated principal component reconstruction of a monotone function is not guaranteed
    monotone, and ``numpy.interp`` against a non increasing abscissa returns plausible
    nonsense rather than raising. The training family reconstructs monotone at every retained
    rank measured here, so this raises rather than repairing: a non monotone displacement
    coordinate means the representation has failed and the caller needs to know, not to be
    handed a curve.
    """
    array = np.asarray(values, dtype=float)
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(
            "the reconstructed displacement coordinate is not strictly increasing, so it "
            "cannot serve as an abscissa. Interpolating against it would return a plausible "
            "curve that is not a prediction of anything."
        )
    return array


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurvePrediction:
    """A predicted response: mean curves with their propagated pointwise variance."""

    u_grid: np.ndarray
    force_mean: np.ndarray
    force_variance: np.ndarray
    damage_mean: np.ndarray
    damage_variance: np.ndarray

    def force_std(self) -> np.ndarray:
        return np.sqrt(self.force_variance)

    def damage_std(self) -> np.ndarray:
        return np.sqrt(self.damage_variance)


@dataclass(frozen=True)
class SurrogateModel:
    """Everything needed to predict, loaded from one stage directory.

    The prediction contract of build spec 10.4: scalar QoIs come from their own Gaussian
    processes and are never read off a reconstructed curve, and the curve comes from the score
    GPs through the reduction basis with its variance propagated through the same linear map.
    """

    feature_standardizer: Standardizer
    target_standardizers: dict[str, Standardizer]
    models: dict[str, FittedGP]
    basis: CurveBasis
    score_targets: dict[str, list[str]]
    scalar_targets: list[str]
    settings: GPSettings
    metadata: dict[str, Any]

    # -- prediction ---------------------------------------------------------

    def _design(self, X: Any) -> np.ndarray:
        matrix = features(X) if hasattr(X, "columns") else np.atleast_2d(
            np.asarray(X, dtype=float)
        )
        if matrix.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"the surrogate takes an (n, {len(FEATURE_ORDER)}) design whose columns are "
                f"{list(FEATURE_ORDER)}; got shape {matrix.shape}."
            )
        return self.feature_standardizer.transform(matrix)

    def predict_target(self, name: str, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of one target in its own units."""
        if name not in self.models:
            raise KeyError(
                f"no Gaussian process for target {name!r}. The artifact carries "
                f"{sorted(self.models)}."
            )
        mean, variance = self.models[name].predict(self._design(X))
        standardizer = self.target_standardizers[name]
        return (
            standardizer.inverse_mean(mean.reshape(-1, 1)).ravel(),
            standardizer.inverse_variance(variance.reshape(-1, 1)).ravel(),
        )

    def predict_scores(self, X: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Score means and variances per reduction block, in the blocks' own units."""
        means: dict[str, np.ndarray] = {}
        variances: dict[str, np.ndarray] = {}
        design = self._design(X)
        for block, names in self.score_targets.items():
            if not names:
                means[block] = np.zeros((design.shape[0], 0))
                variances[block] = np.zeros((design.shape[0], 0))
                continue
            block_mean = np.empty((design.shape[0], len(names)))
            block_variance = np.empty((design.shape[0], len(names)))
            for column, name in enumerate(names):
                mean, variance = self.predict_target(name, X)
                block_mean[:, column] = mean
                block_variance[:, column] = variance
            means[block] = block_mean
            variances[block] = block_variance
        return means, variances

    def predict_qoi(self, X: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Every scalar QoI from its own Gaussian process, never off the curve."""
        return {name: self.predict_target(name, X) for name in self.scalar_targets}

    def predict_curve(self, X: Any) -> CurvePrediction:
        """The load displacement and damage curves of build spec 10.4, with variance.

        The mean is the reconstruction at the predicted score means. The force variance is the
        linear propagation ``sum_k phi_k(u)^2 sigma_k^2`` over the amplitude scores, with the
        amplitude loadings carried out to the displacement grid through the same warp and the
        same displacement coordinate the mean used, plus the truncation residual variance the
        representation cannot express. The damage variance is the same propagation on the
        damage block, which needs no warp because that family is not registered.

        Phase and displacement uncertainty enter the curve nonlinearly, so they are not in
        this variance; :meth:`draw_curves` propagates them by sampling, which is what build
        spec 10.4 prescribes for the bands.
        """
        means, variances = self.predict_scores(X)
        force_mean = self.basis.reconstruct_force(
            means[BLOCK_AMPLITUDE], means[BLOCK_PHASE], means[BLOCK_DISPLACEMENT]
        )
        damage_mean = self.basis.reconstruct_damage(means[BLOCK_DAMAGE])
        n_rows = force_mean.shape[0]
        force_variance = np.empty_like(force_mean)
        for row in range(n_rows):
            loadings = self.basis.force_loadings_on_displacement(
                means[BLOCK_PHASE][row], means[BLOCK_DISPLACEMENT][row]
            )
            force_variance[row] = (
                loadings**2 * variances[BLOCK_AMPLITUDE][row][:, None]
            ).sum(axis=0) + self.basis.truncation_force_variance
        damage_loadings = self.basis.damage.components[: self.basis.n_damage]
        damage_variance = (
            variances[BLOCK_DAMAGE] @ (damage_loadings**2)
        ) + self.basis.truncation_damage_variance
        return CurvePrediction(
            u_grid=self.basis.u_grid,
            force_mean=force_mean,
            force_variance=force_variance,
            damage_mean=damage_mean,
            damage_variance=damage_variance,
        )

    def draw_curves(
        self, X: Any, n_draws: int, seed_sequence: np.random.SeedSequence
    ) -> np.ndarray:
        """``n_draws`` posterior realizations of the load displacement curve per input row.

        Build spec 10.4: the warp uncertainty is propagated by sampling scores rather than by
        a linearization, because the composition of an amplitude with a warp is not linear in
        the warp. Returns an ``(n_rows, n_draws, n_grid)`` array in N.

        A draw whose reconstructed displacement coordinate is not monotone is a draw the
        representation cannot turn into a curve; it is counted and reraised rather than being
        dropped, because a silently thinned sample would bias the band it feeds.
        """
        means, variances = self.predict_scores(X)
        rng = np.random.default_rng(seed_sequence)
        n_rows = means[BLOCK_AMPLITUDE].shape[0]
        out = np.empty((n_rows, int(n_draws), self.basis.u_grid.size), dtype=float)
        for row in range(n_rows):
            for draw in range(int(n_draws)):
                sampled = {}
                for block in (BLOCK_AMPLITUDE, BLOCK_PHASE, BLOCK_DISPLACEMENT):
                    mean = means[block][row]
                    std = np.sqrt(variances[block][row])
                    sampled[block] = mean + std * rng.standard_normal(mean.size)
                out[row, draw] = self.basis.reconstruct_force(
                    sampled[BLOCK_AMPLITUDE],
                    sampled[BLOCK_PHASE],
                    sampled[BLOCK_DISPLACEMENT],
                )[0]
        return out

    # -- persistence --------------------------------------------------------

    @classmethod
    def load(cls, artifact_root: Path | str, config_sha256: str) -> SurrogateModel:
        """Load the fitted surrogate from its stage directory, or raise naming the fix."""
        directory = stage_dir(artifact_root, STAGE_NAME, config_sha256)
        record_path = directory / SURROGATE_JSON
        if not record_path.is_file():
            raise FileNotFoundError(
                f"no surrogate artifact at {record_path}. Run `ufem run surrogate` for this "
                "config hash; nothing here falls back to an unfitted model."
            )
        configure_torch()
        record = json.loads(record_path.read_text(encoding="utf-8"))
        state = np.load(directory / GP_STATE_NPY)
        train_x = np.load(directory / TRAINING_INPUTS_NPY)
        train_y = np.load(directory / TRAINING_TARGETS_NPY)

        feature_standardizer = Standardizer(
            mean=np.array(record["feature_standardization"]["mean"], dtype=float),
            scale=np.array(record["feature_standardization"]["scale"], dtype=float),
        )
        settings = GPSettings(**record["gp_settings"])
        names = list(record["target_order"])
        target_standardizers = {
            name: Standardizer(
                mean=np.array([record["target_standardization"][name]["mean"]], dtype=float),
                scale=np.array([record["target_standardization"][name]["scale"]], dtype=float),
            )
            for name in names
        }
        standardized_x = feature_standardizer.transform(train_x)
        models = {}
        for index, name in enumerate(names):
            standardized_y = target_standardizers[name].transform(
                train_y[:, index].reshape(-1, 1)
            ).ravel()
            models[name] = FittedGP(
                name=name,
                parameters=state[index],
                train_x=standardized_x,
                train_y=standardized_y,
                settings=settings,
                marginal_log_likelihood=float(record["marginal_log_likelihood"][name]),
            )

        basis = _load_basis(directory, record)
        return cls(
            feature_standardizer=feature_standardizer,
            target_standardizers=target_standardizers,
            models=models,
            basis=basis,
            score_targets={
                block: list(record["score_targets"][block]) for block in CURVE_BLOCKS
            },
            scalar_targets=list(record["scalar_targets"]),
            settings=settings,
            metadata=record,
        )


def _basis_array(basis: Basis, n_retained: int) -> np.ndarray:
    """One block as a single array: the mean in row 0, the retained loadings after it."""
    return np.vstack([basis.mean.reshape(1, -1), basis.components[:n_retained]])


def _basis_from_array(array: np.ndarray, name: str, n_retained: int) -> Basis:
    """Rebuild a :class:`ufem.reduce.Basis` from the stored mean and loadings.

    The explained variance is not stored back: it played its part in choosing the rank, and a
    loaded basis is only ever asked to project and reconstruct. Storing a number the loaded
    object cannot verify would invite it to be quoted somewhere.
    """
    components = np.asarray(array[1:], dtype=float)
    return Basis(
        name=name,
        mean=np.asarray(array[0], dtype=float),
        components=components,
        explained_variance=np.full(components.shape[0], np.nan),
        explained_variance_ratio=np.full(components.shape[0], np.nan),
        n_retained=int(n_retained),
    )


def _load_basis(directory: Path, record: dict[str, Any]) -> CurveBasis:
    """Rebuild the curve representation from the stored blocks."""
    counts = record["component_counts"]
    truncation = np.load(directory / TRUNCATION_VARIANCE_NPY)
    phase_array = np.load(directory / BASIS_PHASE_NPY)
    phase = (
        _basis_from_array(phase_array, BLOCK_PHASE, int(counts[BLOCK_PHASE]))
        if int(counts[BLOCK_PHASE]) > 0
        else None
    )
    references = np.load(directory / SRSF_REFERENCE_NPY)
    return CurveBasis(
        u_grid=np.array(record["u_grid_mm"], dtype=float),
        stations=np.linspace(0.0, 1.0, N_ARCLENGTH_POINTS),
        u0_mm=float(record["normalizers"]["u0_mm"]),
        P0_N=float(record["normalizers"]["P0_N"]),
        registration=str(record["registration"]),
        amplitude=_basis_from_array(
            np.load(directory / BASIS_AMPLITUDE_NPY),
            BLOCK_AMPLITUDE,
            int(counts[BLOCK_AMPLITUDE]),
        ),
        phase=phase,
        damage=_basis_from_array(
            np.load(directory / BASIS_DAMAGE_NPY), BLOCK_DAMAGE, int(counts[BLOCK_DAMAGE])
        ),
        displacement=_basis_from_array(
            np.load(directory / BASIS_DISPLACEMENT_NPY),
            BLOCK_DISPLACEMENT,
            int(counts[BLOCK_DISPLACEMENT]),
        ),
        n_amplitude=int(counts[BLOCK_AMPLITUDE]),
        n_phase=int(counts[BLOCK_PHASE]),
        n_damage=int(counts[BLOCK_DAMAGE]),
        n_displacement=int(counts[BLOCK_DISPLACEMENT]),
        mean_psi_phase=references[0] if int(counts[BLOCK_PHASE]) > 0 else None,
        mean_psi_displacement=references[1],
        scores={},
        truncation_force_variance=truncation[0],
        truncation_damage_variance=truncation[1],
        reconstruction=record["reconstruction"],
    )


# ---------------------------------------------------------------------------
# Fitting the whole target set
# ---------------------------------------------------------------------------


def score_target_names(basis: CurveBasis) -> dict[str, list[str]]:
    """Target names for the retained scores, one per block and component."""
    return {
        block: [f"{block}_pc{index + 1}" for index in range(basis.block_counts[block])]
        for block in CURVE_BLOCKS
    }


def fit_all(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
) -> tuple[dict[str, FittedGP], Standardizer, dict[str, Standardizer], list[dict[str, Any]]]:
    """Fit one GP per named target on a shared standardized design.

    Every target gets its own child of the stage's ``SeedSequence``, indexed by its position
    in the sorted name list rather than by iteration order, so adding a target does not
    silently reshuffle the restart draws of the ones already there.
    """
    design = np.asarray(X, dtype=float)
    feature_standardizer = Standardizer.fit(design)
    standardized_x = feature_standardizer.transform(design)
    names = list(targets)
    children = seed_sequence.spawn(len(names))
    models: dict[str, FittedGP] = {}
    standardizers: dict[str, Standardizer] = {}
    log: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        column = np.asarray(targets[name], dtype=float).reshape(-1, 1)
        standardizer = Standardizer.fit(column)
        standardizers[name] = standardizer
        y = standardizer.transform(column).ravel()
        model, restarts = fit_gp(standardized_x, y, name, settings, children[index])
        models[name] = model
        for record in restarts:
            log.append({"target": name, **record})
    return models, feature_standardizer, standardizers, log


def collect_targets(
    basis: CurveBasis, qoi: pd.DataFrame, landmarks: pd.DataFrame | None
) -> dict[str, np.ndarray]:
    """Every surrogate target as a column, in the artifact's pinned order."""
    names = score_target_names(basis)
    targets: dict[str, np.ndarray] = {}
    for block in CURVE_BLOCKS:
        scores = basis.scores[block]
        for index, name in enumerate(names[block]):
            targets[name] = scores[:, index]
    for name in SCALAR_QOI:
        if name not in qoi.columns:
            raise KeyError(
                f"the QoI table has no column {name!r}. The surrogate's scalar target list is "
                f"the QoI schedule of build spec 9.5; the table offers {list(qoi.columns)}."
            )
        targets[name] = qoi[name].to_numpy(dtype=float)
    if landmarks is not None:
        for name in LANDMARK_QOI:
            column = landmarks[name].to_numpy(dtype=float)
            if np.any(~np.isfinite(column)):
                raise ValueError(
                    f"landmark target {name!r} carries {int((~np.isfinite(column)).sum())} "
                    "non finite values. A GP fitted on the remainder would be trained on a "
                    "different sample than every other target in this artifact."
                )
            targets[name] = column
    return targets


#: How far the stage's own registration may differ from the P3 artifact before it is a defect.
#: Both run the same fdasrsf call on the same curves, so the honest expectation is exact
#: agreement and this is a float64 round off allowance, not a tolerance to hide behind.
REGISTRATION_AGREEMENT_TOLERANCE = 1.0e-9


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Locate the grid, register, and reduce artifacts this stage depends on."""
    artifact_root = root / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, config_sha256)
    reduce_dir = stage_dir(artifact_root, REDUCE_STAGE, config_sha256)
    hashes: dict[str, str] = {}
    for directory, name, stage in (
        (grid_dir, RF2_GRID_PARQUET, GRID_STAGE),
        (grid_dir, DAMAGE_GRID_PARQUET, GRID_STAGE),
        (grid_dir, QOI_PARQUET, GRID_STAGE),
        (register_dir, LANDMARKS_PARQUET, REGISTER_STAGE),
        (reduce_dir, REDUCE_BASIS_JSON, REDUCE_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the surrogate stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first."
            )
        hashes[name] = sha256_file(path)
    return grid_dir, register_dir, reduce_dir, hashes


def _registration_agreement(basis: CurveBasis, reduce_dir: Path) -> dict[str, float]:
    """Check this stage's own registration against the one the reduce stage published.

    The stage repeats the arc length reparameterization, the SRVF registration and the
    amplitude principal component fit rather than reading P3's artifacts, and the reason is
    fold honesty: the cross validation of build spec 16.3 has to refit all of that inside
    every fold, and a production path that took a shortcut the fold path cannot take would be
    two code paths whose agreement nobody checks. So they are one code path, and this is the
    check that it lands where P3 landed. The measured deviation goes in the manifest.
    """
    published = json.loads((reduce_dir / REDUCE_BASIS_JSON).read_text(encoding="utf-8"))
    block = next(
        item for item in published["blocks"] if item["name"] == BLOCK_AMPLITUDE
    )
    mean_deviation = float(
        np.abs(np.array(block["mean"], dtype=float) - basis.amplitude.mean).max()
    )
    rank = min(int(block["n_retained"]), basis.n_amplitude)
    loading_deviation = float(
        np.abs(
            np.array(block["components"][:rank], dtype=float)
            - basis.amplitude.components[:rank]
        ).max()
    )
    worst = max(mean_deviation, loading_deviation)
    if worst > REGISTRATION_AGREEMENT_TOLERANCE:
        raise AssertionError(
            f"the surrogate stage's own registration and amplitude basis differ from the "
            f"reduce stage's published basis by {worst:.3e}, beyond the "
            f"{REGISTRATION_AGREEMENT_TOLERANCE:g} round off allowance. The two are meant to "
            "be the same computation on the same curves, so a difference is a defect in one "
            "of them rather than a tolerance to widen."
        )
    return {
        "mean_max_abs_deviation_N": mean_deviation,
        "loading_max_abs_deviation": loading_deviation,
        "n_components_compared": rank,
        "tolerance": REGISTRATION_AGREEMENT_TOLERANCE,
    }


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the surrogate stage and return its artifact directory."""
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    grid_dir, register_dir, reduce_dir, input_hashes = _load_inputs(
        root, config, config_sha256
    )

    force_frame = pd.read_parquet(grid_dir / RF2_GRID_PARQUET)
    damage_frame = pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET)
    qoi = pd.read_parquet(grid_dir / QOI_PARQUET)
    landmarks = pd.read_parquet(register_dir / LANDMARKS_PARQUET)

    jobs, force = curve_matrix(force_frame)
    damage_jobs, damage = curve_matrix(damage_frame)
    if jobs != damage_jobs or list(qoi["job"].astype(str)) != jobs:
        raise AssertionError(
            "the gridded curves, the damage curves and the QoI table carry different job "
            "orders, which would misalign every target against its design point."
        )
    if list(landmarks["job"].astype(str)) != jobs:
        raise AssertionError(
            "the landmark table carries a different job order than the gridded curves."
        )

    u_grid = displacement_grid(config)
    basis_started = _time.perf_counter()
    basis = CurveBasis.fit(u_grid, force, damage, config)
    basis_seconds = _time.perf_counter() - basis_started
    agreement = _registration_agreement(basis, reduce_dir)

    X = features(qoi)
    targets = collect_targets(basis, qoi, landmarks)
    settings = GPSettings.from_config(config)
    seed_sequence = np.random.SeedSequence(config.pipeline.seed_entropy).spawn(1)[0]

    fit_started = _time.perf_counter()
    models, feature_standardizer, target_standardizers, restart_log = fit_all(
        X, targets, settings, seed_sequence
    )
    fit_seconds = _time.perf_counter() - fit_started

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)

    names = list(targets)
    state = np.vstack([models[name].parameters.reshape(1, -1) for name in names])
    target_matrix = np.column_stack([targets[name] for name in names])
    score_names = score_target_names(basis)
    scalar_names = [
        name for name in names if name not in {n for block in score_names.values() for n in block}
    ]

    phase_report = _capped_block_report(
        basis.phase,
        basis.n_phase,
        float(config.pipeline.surrogate.phase_variance_target),
        int(config.pipeline.surrogate.phase_max_components),
    )
    displacement_report = _capped_block_report(
        basis.displacement,
        basis.n_displacement,
        float(config.pipeline.surrogate.displacement_variance_target),
        int(config.pipeline.surrogate.displacement_max_components),
    )
    record: dict[str, Any] = {
        "config_sha256": config_sha256,
        "n_training_runs": len(jobs),
        "jobs": jobs,
        "feature_order": list(FEATURE_ORDER),
        "feature_standardization": {
            "mean": [float(value) for value in feature_standardizer.mean],
            "scale": [float(value) for value in feature_standardizer.scale],
        },
        "target_order": names,
        "target_standardization": {
            name: {
                "mean": float(target_standardizers[name].mean[0]),
                "scale": float(target_standardizers[name].scale[0]),
            }
            for name in names
        },
        "gp_parameter_names": list(GP_PARAMETER_NAMES),
        "gp_settings": {
            "nu": settings.nu,
            "ard": settings.ard,
            "lengthscale_bounds": list(settings.lengthscale_bounds),
            "restarts": settings.restarts,
            "noise_prior_median_variance": settings.noise_prior_median_variance,
            "noise_prior_log_scale": settings.noise_prior_log_scale,
            "max_iterations": settings.max_iterations,
        },
        "score_targets": {block: score_names[block] for block in CURVE_BLOCKS},
        "scalar_targets": scalar_names,
        "component_counts": basis.block_counts,
        "phase_block": phase_report,
        "displacement_block": displacement_report,
        "registration": basis.registration,
        "registration_agreement": agreement,
        "normalizers": {"u0_mm": basis.u0_mm, "P0_N": basis.P0_N},
        "u_grid_mm": [float(value) for value in u_grid],
        "reconstruction": basis.reconstruction,
        "marginal_log_likelihood": {
            name: float(models[name].marginal_log_likelihood) for name in names
        },
        "fitted_hyperparameters": {
            name: {
                "lengthscale": [float(value) for value in models[name].lengthscales()],
                "noise": models[name].noise(),
                "outputscale": models[name].outputscale(),
            }
            for name in names
        },
    }

    outputs = []
    for array, filename in (
        (state, GP_STATE_NPY),
        (X, TRAINING_INPUTS_NPY),
        (target_matrix, TRAINING_TARGETS_NPY),
        (_basis_array(basis.amplitude, basis.n_amplitude), BASIS_AMPLITUDE_NPY),
        (_basis_array(basis.damage, basis.n_damage), BASIS_DAMAGE_NPY),
        (_basis_array(basis.displacement, basis.n_displacement), BASIS_DISPLACEMENT_NPY),
        (
            np.vstack(
                [basis.truncation_force_variance, basis.truncation_damage_variance]
            ),
            TRUNCATION_VARIANCE_NPY,
        ),
    ):
        path = directory / filename
        np.save(path, np.ascontiguousarray(np.asarray(array, dtype=float)))
        outputs.append(path)
    phase_path = directory / BASIS_PHASE_NPY
    np.save(
        phase_path,
        np.ascontiguousarray(
            _basis_array(basis.phase, basis.n_phase)
            if basis.phase is not None
            else np.zeros((1, N_ARCLENGTH_POINTS))
        ),
    )
    outputs.append(phase_path)
    reference_path = directory / SRSF_REFERENCE_NPY
    np.save(
        reference_path,
        np.ascontiguousarray(
            np.vstack(
                [
                    basis.mean_psi_phase
                    if basis.mean_psi_phase is not None
                    else np.zeros(N_ARCLENGTH_POINTS),
                    basis.mean_psi_displacement,
                ]
            ).astype(float)
        ),
    )
    outputs.append(reference_path)

    record_path = directory / SURROGATE_JSON
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    outputs.append(record_path)
    restart_path = directory / RESTART_LOG_JSON
    restart_path.write_text(
        json.dumps(restart_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs.append(restart_path)

    dispersion = _restart_dispersion(restart_log)
    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "gp_fit_wall_time_s": fit_seconds,
        "representation_wall_time_s": basis_seconds,
        "n_targets": len(names),
        "n_gp_fits": len(names) * settings.restarts,
        "n_training_runs": len(jobs),
        "component_counts": basis.block_counts,
        "phase_block": phase_report,
        "displacement_block": displacement_report,
        "registration": basis.registration,
        "registration_agreement": agreement,
        "reconstruction": basis.reconstruction,
        "restart_dispersion": dispersion,
        "noise_prior_median_variance": settings.noise_prior_median_variance,
        "fitted_noise": {name: models[name].noise() for name in names},
        "optimizer": "scipy.optimize.minimize(method='L-BFGS-B', jac=True) on the raw parameters",
        "fit_budget_s": 60.0,
        "fit_budget_met": bool(fit_seconds < 60.0),
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=sorted(outputs),
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    counts = basis.block_counts
    print(
        f"[surrogate] {len(names)} Gaussian processes over {len(jobs)} runs "
        f"({counts[BLOCK_AMPLITUDE]} amplitude, {counts[BLOCK_PHASE]} phase, "
        f"{counts[BLOCK_DAMAGE]} damage, {counts[BLOCK_DISPLACEMENT]} displacement scores, "
        f"{len(scalar_names)} scalars); {settings.restarts} restarts each, "
        f"{fit_seconds:.1f} s of fitting against the 60 s budget; median reconstruction "
        f"error {basis.reconstruction['force_relative_l2']['p50'] * 100:.2f} percent"
    )
    return directory


def _capped_block_report(
    basis: Basis | None, n_fitted: int, variance_target: float, cap: int
) -> dict[str, Any]:
    """What a cap on a block's component count actually did, as numbers.

    Reports the rank the 99 percent reduction target asked for, the rank this stage fitted,
    the rank the block's own variance target would have needed, and the share of the block
    variance the fitted rank carries. The point is that the cap is visible as a measured cost
    in the manifest rather than as a sentence in a docstring.
    """
    if basis is None:
        return {
            "n_retained_at_pca_target": 0,
            "n_fitted": 0,
            "variance_target": variance_target,
            "cap": cap,
            "n_needed_for_target": 0,
            "variance_carried_by_fitted": 0.0,
        }
    cumulative = np.cumsum(basis.explained_variance_ratio)
    return {
        "n_retained_at_pca_target": int(basis.n_retained),
        "n_fitted": int(n_fitted),
        "variance_target": variance_target,
        "cap": cap,
        "n_needed_for_target": int(np.searchsorted(cumulative, variance_target) + 1),
        "variance_carried_by_fitted": float(cumulative[n_fitted - 1]) if n_fitted else 0.0,
    }


def _restart_dispersion(log: list[dict[str, Any]]) -> dict[str, float]:
    """How far the restarts of a target spread in marginal likelihood, over all targets.

    A dispersion of zero would say the surface has one basin and the restart policy is
    insurance; a large one says the restarts are doing real work. Either way it is a number in
    the manifest rather than an opinion in a docstring.
    """
    by_target: dict[str, list[float]] = {}
    for record in log:
        if "marginal_log_likelihood" in record:
            by_target.setdefault(str(record["target"]), []).append(
                float(record["marginal_log_likelihood"])
            )
    spreads = [max(values) - min(values) for values in by_target.values() if len(values) > 1]
    if not spreads:
        return {"n_targets": 0, "median": 0.0, "max": 0.0, "n_failed_restarts": 0}
    return {
        "n_targets": len(spreads),
        "median": float(np.median(spreads)),
        "max": float(np.max(spreads)),
        "n_failed_restarts": int(
            sum(1 for record in log if record.get("status") == "failed")
        ),
    }
