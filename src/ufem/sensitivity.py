"""Stage ``sensitivity``: global sensitivity three ways, and they have to agree.

Build spec section 12. The predecessor project's sensitivity stage is the cautionary case
this whole file is written against: it declared a third, different set of input distributions,
amplified the variance by 1.5, injected two percent noise, and published first order indices
summing to 0.23 against total indices near 0.88 for three independent inputs. That signature,
a first order sum far below the total sum with no interaction mechanism to explain it, is what
a noise dominated surface looks like, and it means the numbers described the noise rather than
the beam.

Three constructions run here, on the same 198 training points and the same probabilistic
model, and the point of running three is that two independent surrogate families agreeing is
the strongest validation available without new finite element runs:

1. **Sparse polynomial chaos** (build spec 12.1), the primary. LARS selection over a degree 5
   hyperbolically truncated basis, corrected leave one out as the selection criterion, one
   expansion per scalar quantity of interest and per retained principal component score. The
   first order, total and interaction indices then come analytically from the coefficients,
   with no sampling error at all. Whether they are published is decided by a Q2 gate, below.
2. **Gaussian process posterior Sobol** (build spec 12.2), the cross check. Conditional
   realizations are drawn from each fitted posterior and a Saltelli estimate is computed on
   each one, so every index comes out as a distribution rather than as a number. Realizations
   and not the posterior mean: the mean of a Gaussian process is smoother than any of its
   draws, and its indices are biased toward whichever input dominates.
3. **Functional indices** (build spec 12.3), pointwise along the response and aggregated over
   the amplitude components.

**Why the ordinary Sobol machinery is valid here.** Because the inputs are genuinely
independent after the reparameterization of build spec 9.1. Had the elastic modulus stayed a
feature alongside the strength it is an exact function of, every Saltelli estimate would have
been structurally wrong, because the estimator resamples columns independently and there is no
such thing as an independent resample of two variables that are the same variable. That is not
a caveat about precision, it is a statement that the numbers would have meant nothing.

**The trust gate, and why a gate at all.** A Sobol index computed from a metamodel is a
statement about the metamodel, and it inherits whatever the metamodel got wrong. Build spec
12.1 therefore ties publication to the corrected leave one out Q2 of each expansion, on the
thresholds of the Blatman and Sudret line of work: at or above 0.95 the index values are
published, between 0.80 and 0.95 only the rankings are, with the values carried as indicative,
and below 0.80 nothing is published for that target. The thresholds are in this file rather
than in the configuration for the same reason the calibration gate's are: a threshold that can
be edited without a code change is a threshold that will be.

**The algebra of the functional indices** (build spec 12.3), because it is the one derivation
here that is not a library call.

The registered amplitude family is represented as ``f(s; x) = m(s) + sum_k phi_k(s) c_k(x)``,
where ``s`` indexes the arc length stations, ``m`` is the block mean, ``phi_k`` are the
orthonormal principal component loadings, and ``c_k`` are the scores, which are the random
part: they and only they depend on the inputs ``x``. Each score carries its own chaos
expansion ``c_k(x) = sum_a A[k, a] Psi_a(x)`` over the orthonormal chaos basis, with
``Psi_0 = 1``. Substituting and exchanging the sums,

    f(s; x) = m(s) + sum_a B[a, s] Psi_a(x),    B[a, s] = sum_k A[k, a] phi_k(s),

which says the field itself has a chaos expansion at every station, whose coefficients are the
loadings applied to the score coefficients. One matrix product, ``B = A @ phi``, and the whole
pointwise decomposition follows from orthonormality of the ``Psi``:

    V(s)   = sum_{a != 0} B[a, s]^2                       (pointwise variance)
    V_u(s) = sum_{a in A_u} B[a, s]^2                     (partial variance of input set u)

where ``A_u`` is the set of multi indices whose support is exactly ``u``. Then
``S_i(s) = V_{i}(s) / V(s)`` and ``T_i(s) = sum_{u containing i} V_u(s) / V(s)``.

Two things are worth being explicit about. The construction is exact given the score
expansions; it does not assume the scores are independent, only that they share the same input
distribution, and the cross terms between components are handled by squaring the sum rather
than summing the squares. And because the loadings are orthonormal, summing ``V_u(s)`` over
the stations gives exactly ``sum_k V_u^{(k)}``, the eigenvalue weighted aggregate of Lamboni,
Monod and Makowski 2011. So the pointwise picture and the aggregated table are two views of
one decomposition rather than two calculations that happen to be near each other, and a test
asserts that identity rather than trusting this paragraph.

**Registration comes first, and it is not a detail.** The pointwise indices are computed on
the registered amplitude functions. On unregistered curves the same arithmetic would largely
measure phase: at a fixed displacement near the peak, the dominant source of variation across
runs is whether that run has reached its peak yet, so an index computed there would report
which input moves the peak location rather than which input sets the load. The abscissa
reported alongside is the Karcher mean displacement coordinate, so the axis is physical, but
it is a mean axis and the report says so.

Units: indices are dimensionless by construction. The functional abscissa is displacement in
mm; the amplitude loadings are in N and the damage loadings dimensionless, but both cancel in
a ratio of variances.

Determinism (build spec 17.2 and 17.3): every random draw here comes from a ``SeedSequence``
spawned off the configured entropy, one child per target, and the Sobol design and the SALib
bootstrap take integer seeds derived from those children. The polynomial chaos path draws no
random numbers at all. The gate is a forced double run producing bitwise identical artifacts,
asserted by ``tests/test_p6_determinism.py``.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.calibrate import CALIBRATION_JSON, informative_abscissae
from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import (
    FEATURE_ORDER,
    Config,
    features,
    openturns_input_distribution,
    salib_problem,
)
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest
from ufem.register import AMPLITUDE_PARQUET, curve_matrix
from ufem.register import STAGE_NAME as REGISTER_STAGE
from ufem.surrogate import (
    BLOCK_AMPLITUDE,
    BLOCK_DAMAGE,
    GP_STATE_NPY,
    SCALAR_QOI,
    SURROGATE_JSON,
    TRAINING_TARGETS_NPY,
    FittedGP,
    SurrogateModel,
    configure_torch,
)
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.validate import QOI_LABELS

STAGE_NAME = "sensitivity"

#: Output file names inside the stage directory.
SENSITIVITY_JSON = "sensitivity.json"
PCE_INDICES_PARQUET = "pce_indices.parquet"
GP_INDICES_PARQUET = "gp_sobol_indices.parquet"
FUNCTIONAL_INDICES_PARQUET = "functional_indices.parquet"
AGREEMENT_PARQUET = "pce_gp_agreement.parquet"
SOBOL_TEX = "sobol_indices.tex"
AGGREGATED_TEX = "sensitivity_aggregated.tex"
GATE_TEX = "sensitivity_gate.tex"
SENSITIVITY_MD = "sensitivity_summary.md"

#: The trust gate of build spec 12.1, on the corrected leave one out Q2 of each expansion.
#:
#: These are constants of the specification, not settings. Q2 at or above 0.95 means the
#: expansion reproduces out of sample variation well enough that the variance decomposition it
#: implies is a statement about the beam; between 0.80 and 0.95 the ordering of the inputs
#: survives but the third digit of an index does not, so the ranking publishes and the values
#: are labeled indicative; below 0.80 the expansion is not a description of anything and
#: nothing derived from it is published. They live here rather than in ``pipeline.yaml``
#: because a gate whose threshold is a config field is a gate that gets moved when it fails,
#: which is the same argument the calibration gate of build spec 11.5 is written under.
Q2_PUBLISH_VALUES = 0.95
Q2_PUBLISH_RANKINGS = 0.80

PUBLICATION_VALUES = "values"
PUBLICATION_RANKINGS = "rankings"
PUBLICATION_WITHHELD = "not_published"

#: Quantiles of the Gaussian process posterior index distributions, the 90 percent interval
#: build spec 12.2 asks for, reported with the median.
POSTERIOR_QUANTILES: tuple[float, float, float] = (5.0, 50.0, 95.0)

#: How many query points the pathwise sampler evaluates per chunk. The full design at
#: ``2**15`` is 163840 rows, and at 4096 features a chunk this size keeps the transient
#: feature matrix near 130 MB, well inside the 12 GB working ceiling of build spec section 3.
PATHWISE_CHUNK = 4096

#: Matern smoothness the pathwise prior draws its spectral frequencies for. It is read back
#: from the fitted kernel settings rather than assumed, and this is the value asserted against.
PATHWISE_NU = 2.5

#: Display names for the three inputs, used in the generated fragments and figures.
INPUT_LABELS: dict[str, str] = {
    "Fcm_MPa": "Concrete strength",
    "c_nom_bottom_mm": "Bottom cover",
    "c_nom_top_mm": "Top cover",
}
INPUT_MATH: dict[str, str] = {
    "Fcm_MPa": r"$f_{cm}$",
    "c_nom_bottom_mm": r"$c_{\mathrm{bot}}$",
    "c_nom_top_mm": r"$c_{\mathrm{top}}$",
}


def target_label(name: str) -> str:
    """A LaTeX safe display name for a target, because every raw name carries underscores."""
    if name in QOI_LABELS:
        return QOI_LABELS[name]
    block, _, component = name.rpartition("_pc")
    if block and component.isdigit():
        return f"{block.capitalize()} PC{component}"
    return name.replace("_", " ")

#: The two functional blocks indices are computed along, with the abscissa each is reported on.
FUNCTIONAL_BLOCKS: tuple[str, str] = (BLOCK_AMPLITUDE, BLOCK_DAMAGE)


class SensitivityInputMissing(FileNotFoundError):
    """An upstream artifact this stage reads is absent, and nothing is inferred from that."""


# ---------------------------------------------------------------------------
# The trust gate
# ---------------------------------------------------------------------------


def publication_level(q2: float) -> str:
    """Map a corrected leave one out Q2 onto what build spec 12.1 allows publishing.

    A non finite Q2 is treated as a failed fit and withheld, because the alternative is
    publishing an index whose validity measurement did not evaluate.
    """
    value = float(q2)
    if not np.isfinite(value):
        return PUBLICATION_WITHHELD
    if value >= Q2_PUBLISH_VALUES:
        return PUBLICATION_VALUES
    if value >= Q2_PUBLISH_RANKINGS:
        return PUBLICATION_RANKINGS
    return PUBLICATION_WITHHELD


#: Publication levels ordered from strongest to weakest, so a block can take the weakest of
#: its components rather than an average of them.
PUBLICATION_ORDER: tuple[str, str, str] = (
    PUBLICATION_VALUES,
    PUBLICATION_RANKINGS,
    PUBLICATION_WITHHELD,
)


def weakest_publication_level(levels: list[str]) -> str:
    """The level a block of components publishes at: the weakest any component reached.

    An aggregate of expansions is not more trustworthy than its worst part. Taking the
    weakest is the conservative reading and the only one that cannot be used to smuggle a
    withheld component's contribution into a published aggregate.
    """
    if not levels:
        return PUBLICATION_WITHHELD
    return max(levels, key=PUBLICATION_ORDER.index)


def explainable_variance_ceiling(gp: FittedGP) -> float:
    """The share of a target's variance any smooth metamodel could hope to explain.

    The Gaussian process of build spec 10.3 fits a homoscedastic nugget alongside the kernel,
    and in standardized target units the two together account for the whole variance. The
    nugget is what the fit concluded the three inputs do not determine at the resolution this
    design offers, so ``outputscale / (outputscale + nugget)`` is an estimate, from an
    independent model family, of the ceiling on the corrected leave one out Q2 of build spec
    12.1.

    This is a diagnostic and not a second gate. Nothing is published because of it and no
    threshold moves because of it. It exists so that a target failing the Q2 gate can be told
    apart from an expansion that is simply too coarse: a Q2 near the ceiling says the response
    itself is not reproducible from these three inputs at this design density, and a Q2 far
    below it says the expansion is the problem.
    """
    outputscale = float(gp.outputscale())
    nugget = float(gp.noise())
    total = outputscale + nugget
    if not total > 0.0:
        raise ValueError(
            f"target {gp.name!r} has a fitted prior variance of {total}, so an explainable "
            "share is 0/0."
        )
    return float(outputscale / total)


def design_roughness(
    standardized_design: np.ndarray, values: np.ndarray, closest_fraction: float = 0.10
) -> dict[str, float]:
    """How much the response moves between design points that are almost the same point.

    For every training point, find its nearest neighbour in the standardized input space,
    then look at the closest ``closest_fraction`` of those pairs and report the median
    absolute response difference as a share of the response's own standard deviation.

    This is a model free measurement and it is the one number that separates the two
    explanations for a low Q2. If nearly coincident designs give nearly equal responses, a
    smooth surrogate is possible and a poor one is the surrogate's fault. If they do not, the
    response varies on a scale finer than the design resolves, and no smooth metamodel of any
    family will certify against it, whatever basis it is given.
    """
    from scipy.spatial import cKDTree

    design = np.atleast_2d(np.asarray(standardized_design, dtype=float))
    response = np.asarray(values, dtype=float).ravel()
    if design.shape[0] != response.size:
        raise ValueError(
            f"the design has {design.shape[0]} rows and the response {response.size} values."
        )
    distances, neighbours = cKDTree(design).query(design, k=2)
    nearest = neighbours[:, 1]
    separation = distances[:, 1]
    difference = np.abs(response - response[nearest])
    cutoff = float(np.quantile(separation, float(closest_fraction)))
    close = separation <= cutoff
    spread = float(np.std(response, ddof=1))
    if not spread > 0.0:
        raise ValueError("a constant response has no roughness to measure.")
    return {
        "closest_fraction": float(closest_fraction),
        "n_pairs": int(close.sum()),
        "separation_cutoff": cutoff,
        "median_separation": float(np.median(separation)),
        "median_abs_difference": float(np.median(difference[close])),
        "roughness_ratio": float(np.median(difference[close]) / spread),
    }


# ---------------------------------------------------------------------------
# Sparse polynomial chaos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosBasis:
    """The candidate basis of build spec 12.1, built against the config's own measure."""

    distribution: Any
    enumerate_function: Any
    factory: Any
    size: int
    total_degree: int
    hyperbolic_q: float

    def multi_index(self, term: int) -> np.ndarray:
        """The exponent vector of one candidate term, as a length ``d`` integer array."""
        return np.array(list(self.enumerate_function(int(term))), dtype=int)


def chaos_basis(config: Config) -> ChaosBasis:
    """Assemble the degree ``p``, hyperbolic ``q`` orthonormal basis of build spec 12.1.

    The univariate families are chosen by OpenTURNS to be orthonormal with respect to each
    declared marginal, so the expansion is orthogonal under the actual probabilistic model
    rather than under a standard normal that happens to be nearby. That is what makes the
    variance decomposition of :func:`sobol_from_coefficients` a sum of squares.
    """
    import openturns as ot

    settings = config.pipeline.sensitivity
    distribution = openturns_input_distribution(config)
    dimension = distribution.getDimension()
    if dimension != len(FEATURE_ORDER):
        raise ValueError(
            f"the chaos basis was built on {dimension} inputs but the feature contract names "
            f"{len(FEATURE_ORDER)}: {list(FEATURE_ORDER)}."
        )
    enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(
        dimension, float(settings.pce_hyperbolic_q)
    )
    factory = ot.OrthogonalProductPolynomialFactory(
        [
            ot.StandardDistributionPolynomialFactory(distribution.getMarginal(index))
            for index in range(dimension)
        ],
        enumerate_function,
    )
    size = int(enumerate_function.getStrataCumulatedCardinal(int(settings.pce_total_degree)))
    return ChaosBasis(
        distribution=distribution,
        enumerate_function=enumerate_function,
        factory=factory,
        size=size,
        total_degree=int(settings.pce_total_degree),
        hyperbolic_q=float(settings.pce_hyperbolic_q),
    )


@dataclass(frozen=True)
class PceFit:
    """One fitted sparse expansion with its validity measurement and its indices."""

    name: str
    terms: np.ndarray
    multi_indices: np.ndarray
    coefficients: np.ndarray
    q2_corrected: float
    q2_plain: float
    correction_factor: float
    leave_one_out_error: float
    sample_variance: float
    total_variance: float
    publication: str
    first_order: np.ndarray
    total_order: np.ndarray
    interactions: dict[str, float]

    @property
    def n_terms(self) -> int:
        return int(self.coefficients.size)

    @property
    def interaction_share(self) -> float:
        """One minus the sum of first order indices: everything not a main effect."""
        return float(1.0 - self.first_order.sum())


def design_matrix(basis: ChaosBasis, terms: np.ndarray, standard_sample: Any) -> np.ndarray:
    """Evaluate the selected basis terms on an already transformed sample."""
    import openturns as ot

    sample = (
        standard_sample
        if isinstance(standard_sample, ot.Sample)
        else ot.Sample(np.atleast_2d(np.asarray(standard_sample, dtype=float)))
    )
    return np.column_stack(
        [np.array(basis.factory.build(int(term))(sample)).ravel() for term in terms]
    )


def corrected_leave_one_out(
    psi: np.ndarray, y: np.ndarray, coefficients: np.ndarray
) -> dict[str, float]:
    """The corrected leave one out error and Q2 of Blatman and Sudret 2011.

    With the sparse basis held fixed at what LARS selected, the least squares leave one out
    residual has the closed form ``(y_i - yhat_i) / (1 - h_ii)``, where ``h_ii`` is the
    leverage of point ``i`` in the selected basis, so all ``n`` folds cost one inverse. The
    correction factor

        ``T(P, n) = n / (n - P) * (1 + trace((Psi^T Psi)^-1))``

    accounts for the finite sample size relative to the number of retained terms, and it is
    what keeps a nearly interpolating expansion from reporting a Q2 of one.

    The approximation, stated rather than hidden: the basis selection itself is not repeated
    inside each fold, so a term chosen partly because of point ``i`` is still available to the
    fold that drops it. That is the standard practice in this literature and it makes the
    measurement optimistic, which is the direction that matters for a publication gate, so the
    gate is read as a necessary condition and the agreement with the Gaussian process indices
    is what makes it a sufficient one.
    """
    matrix = np.asarray(psi, dtype=float)
    target = np.asarray(y, dtype=float).ravel()
    n, n_terms = matrix.shape
    if target.size != n:
        raise ValueError(
            f"the design matrix has {n} rows and the target {target.size} values."
        )
    if n_terms >= n:
        raise ValueError(
            f"the expansion retained {n_terms} terms on {n} points, so the leave one out is "
            "not defined. A saturated basis is a failed fit, not a perfect one."
        )
    gram = matrix.T @ matrix
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition):
        raise ValueError(
            "the selected chaos basis has a singular information matrix, so its leave one out "
            "error is not defined. The selection returned collinear terms."
        )
    inverse = np.linalg.inv(gram)
    prediction = matrix @ np.asarray(coefficients, dtype=float).ravel()
    residual = target - prediction
    leverage = np.einsum("ij,jk,ik->i", matrix, inverse, matrix)
    if np.any(leverage >= 1.0):
        raise ValueError(
            "a training point has unit leverage in the selected basis, so its leave one out "
            "prediction is undefined. Nothing is clipped here (ground rule 4)."
        )
    loo = float(np.mean((residual / (1.0 - leverage)) ** 2))
    correction = float(n / (n - n_terms) * (1.0 + float(np.trace(inverse))))
    variance = float(np.var(target, ddof=1))
    if not variance > 0.0:
        raise ValueError(
            "the target has zero sample variance, so a relative leave one out error is 0/0. "
            "A constant quantity is not a sensitivity target."
        )
    return {
        "leave_one_out_error": loo,
        "correction_factor": correction,
        "sample_variance": variance,
        "q2_plain": float(1.0 - loo / variance),
        "q2_corrected": float(1.0 - correction * loo / variance),
        "gram_condition": condition,
    }


def sobol_from_coefficients(
    multi_indices: np.ndarray, coefficients: np.ndarray
) -> dict[str, Any]:
    """First order, total and interaction Sobol indices, analytically from the coefficients.

    The chaos basis is orthonormal, so the response variance is the sum of the squared non
    constant coefficients and the partial variance of an input set is the sum over the terms
    whose exponent vector is supported on exactly that set. No sampling, no estimator, no
    Monte Carlo error: this is the decomposition itself rather than an estimate of it.
    """
    exponents = np.atleast_2d(np.asarray(multi_indices, dtype=int))
    values = np.asarray(coefficients, dtype=float).ravel()
    if exponents.shape[0] != values.size:
        raise ValueError(
            f"{exponents.shape[0]} multi indices carry {values.size} coefficients."
        )
    dimension = exponents.shape[1]
    active = exponents.sum(axis=1) > 0
    total_variance = float((values[active] ** 2).sum())
    first = np.zeros(dimension)
    total = np.zeros(dimension)
    groups: dict[tuple[int, ...], float] = {}
    for row in np.flatnonzero(active):
        support = tuple(int(index) for index in np.flatnonzero(exponents[row]))
        share = float(values[row] ** 2)
        groups[support] = groups.get(support, 0.0) + share
        if len(support) == 1:
            first[support[0]] += share
        for index in support:
            total[index] += share
    if not total_variance > 0.0:
        return {
            "total_variance": 0.0,
            "first_order": np.zeros(dimension),
            "total_order": np.zeros(dimension),
            "groups": {},
        }
    return {
        "total_variance": total_variance,
        "first_order": first / total_variance,
        "total_order": total / total_variance,
        "groups": {
            "|".join(FEATURE_ORDER[index] for index in support): value / total_variance
            for support, value in sorted(groups.items())
        },
    }


def fit_pce(X: np.ndarray, y: np.ndarray, name: str, basis: ChaosBasis) -> PceFit:
    """Fit one sparse expansion by LARS with corrected leave one out selection.

    Build spec 12.1: ``FunctionalChaosAlgorithm`` with a fixed candidate basis and a least
    squares strategy whose model selection factory is LARS scored by corrected leave one out.
    The Q2 that gates publication is then recomputed here from the selected basis rather than
    read back from the library, because OpenTURNS 1.27 refuses to run its own fast cross
    validation on an expansion that involved model selection, and it is right to refuse: the
    honest closed form is the one written out in :func:`corrected_leave_one_out`, with its
    approximation stated.
    """
    import openturns as ot

    inputs = np.atleast_2d(np.asarray(X, dtype=float))
    target = np.asarray(y, dtype=float).ravel()
    if inputs.shape[0] != target.size:
        raise ValueError(
            f"fit_pce got {inputs.shape[0]} design rows and {target.size} responses for "
            f"target {name!r}."
        )
    input_sample = ot.Sample(inputs)
    output_sample = ot.Sample(target.reshape(-1, 1))
    algorithm = ot.FunctionalChaosAlgorithm(
        input_sample,
        output_sample,
        basis.distribution,
        ot.FixedStrategy(basis.factory, basis.size),
        ot.LeastSquaresStrategy(
            ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        ),
    )
    algorithm.run()
    result = algorithm.getResult()
    terms = np.array([int(term) for term in result.getIndices()], dtype=int)
    coefficients = np.array(result.getCoefficients(), dtype=float).ravel()
    if terms.size != coefficients.size:
        raise ValueError(
            f"target {name!r} produced {terms.size} selected terms and "
            f"{coefficients.size} coefficients."
        )
    standard = result.getTransformation()(input_sample)
    psi = design_matrix(basis, terms, standard)
    validity = corrected_leave_one_out(psi, target, coefficients)
    exponents = np.vstack([basis.multi_index(term) for term in terms])
    indices = sobol_from_coefficients(exponents, coefficients)
    return PceFit(
        name=name,
        terms=terms,
        multi_indices=exponents,
        coefficients=coefficients,
        q2_corrected=validity["q2_corrected"],
        q2_plain=validity["q2_plain"],
        correction_factor=validity["correction_factor"],
        leave_one_out_error=validity["leave_one_out_error"],
        sample_variance=validity["sample_variance"],
        total_variance=float(indices["total_variance"]),
        publication=publication_level(validity["q2_corrected"]),
        first_order=np.asarray(indices["first_order"], dtype=float),
        total_order=np.asarray(indices["total_order"], dtype=float),
        interactions={
            key: float(value)
            for key, value in indices["groups"].items()
            if "|" in key
        },
    )


# ---------------------------------------------------------------------------
# Gaussian process posterior Sobol distributions
# ---------------------------------------------------------------------------


def matern52_ard(A: np.ndarray, B: np.ndarray, lengthscales: np.ndarray, outputscale: float):
    """The Matern 5/2 ARD covariance of build spec 10.3, in NumPy.

    Reimplemented rather than called through GPyTorch because the pathwise sampler evaluates
    it on 163840 query points and the torch round trip dominates at that size. A test asserts
    it reproduces the fitted GPyTorch kernel to 1e-12, so this is a second implementation that
    is checked against the first rather than a second implementation nobody compares.
    """
    left = np.atleast_2d(np.asarray(A, dtype=float))
    right = np.atleast_2d(np.asarray(B, dtype=float))
    scale = np.asarray(lengthscales, dtype=float).ravel()
    difference = (left[:, None, :] - right[None, :, :]) / scale
    radius = np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))
    scaled = np.sqrt(5.0) * radius
    return float(outputscale) * (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)


@dataclass(frozen=True)
class PathwiseSampler:
    """Conditional realizations of a fitted Gaussian process, evaluable anywhere.

    Build spec 12.2 asks for 200 conditional realizations per posterior, each evaluated on a
    Saltelli design of ``(d + 2) * 2**15 = 163840`` points. A joint draw at that many points is
    a Cholesky of a 163840 by 163840 covariance, which is 215 terabytes of matrix and is not a
    budgeting question. The realizations therefore come from the decoupled, or pathwise,
    construction of Wilson and co authors 2020, which is Matheron's update rule written as a
    function rather than as a vector:

        ``f_post(x) = f_prior(x) + k(x, X) (K + s^2 I)^-1 (y - f_prior(X) - e)``

    with ``e`` a draw of the observation noise. The update term on the right is exact and costs
    one inverse of the 198 by 198 training covariance, which is already needed. Only the prior
    path is approximated, by a random Fourier feature expansion of the Matern 5/2 spectral
    density: the frequencies are drawn from the multivariate Student t with ``2 nu = 5`` degrees
    of freedom and scale ``diag(1 / l_d^2)`` that is that density, and the feature map pairs a
    cosine with a sine per frequency so the approximation is unbiased in the kernel.

    Keeping the update exact is the point of the construction and not an implementation detail.
    A plain Fourier feature model would show variance starvation, collapsing the posterior
    spread near the training data, which for an index whose whole purpose is to carry the
    surrogate's uncertainty would be the wrong failure in the wrong direction.

    What the approximation costs is measured rather than argued: the stage reports the maximum
    absolute deviation between the feature kernel and the exact kernel on the training design,
    and the test suite compares indices from this sampler against indices from an exact joint
    draw on a design small enough to factorize.
    """

    train_x: np.ndarray
    lengthscales: np.ndarray
    outputscale: float
    noise: float
    constant_mean: float
    frequencies: np.ndarray
    weights: np.ndarray
    alpha: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.frequencies.shape[0] * 2)

    @property
    def n_realizations(self) -> int:
        return int(self.weights.shape[1])

    def prior_features(self, query: np.ndarray) -> np.ndarray:
        """The random Fourier feature map, scaled so its inner product is the kernel."""
        projection = np.asarray(query, dtype=float) @ self.frequencies.T
        return np.sqrt(2.0 * self.outputscale / self.n_features) * np.concatenate(
            [np.cos(projection), np.sin(projection)], axis=1
        )

    def kernel_deviation(self) -> float:
        """Largest absolute gap between the feature kernel and the exact one, on the design."""
        approximate = self.prior_features(self.train_x) @ self.prior_features(self.train_x).T
        exact = matern52_ard(
            self.train_x, self.train_x, self.lengthscales, self.outputscale
        )
        return float(np.abs(approximate - exact).max())

    def __call__(self, query: np.ndarray, chunk: int = PATHWISE_CHUNK) -> np.ndarray:
        """Every realization evaluated at ``query``, as an ``(n_query, n_realizations)`` array."""
        points = np.atleast_2d(np.asarray(query, dtype=float))
        out = np.empty((points.shape[0], self.n_realizations), dtype=float)
        for start in range(0, points.shape[0], int(chunk)):
            stop = min(start + int(chunk), points.shape[0])
            block = points[start:stop]
            cross = matern52_ard(
                block, self.train_x, self.lengthscales, self.outputscale
            )
            out[start:stop] = (
                self.constant_mean
                + self.prior_features(block) @ self.weights
                + cross @ self.alpha
            )
        return out


def pathwise_sampler(
    gp: FittedGP,
    n_realizations: int,
    n_features: int,
    seed_sequence: np.random.SeedSequence,
) -> PathwiseSampler:
    """Build the conditional path sampler of :class:`PathwiseSampler` for one fitted process."""
    if int(n_features) % 2 != 0:
        raise ValueError(
            f"the Fourier feature count must be even so cosines and sines pair, got "
            f"{n_features}."
        )
    if abs(float(gp.settings.nu) - PATHWISE_NU) > 1.0e-12:
        raise ValueError(
            f"the pathwise prior draws frequencies from the Matern {PATHWISE_NU} spectral "
            f"density, but target {gp.name!r} was fitted with nu = {gp.settings.nu}."
        )
    rng = np.random.default_rng(seed_sequence)
    train_x = np.asarray(gp.train_x, dtype=float)
    lengthscales = gp.lengthscales()
    outputscale = gp.outputscale()
    noise = gp.noise()
    mean = gp.constant_mean()
    n_points, dimension = train_x.shape
    half = int(n_features) // 2
    normal = rng.standard_normal((half, dimension))
    chi_square = rng.chisquare(2.0 * PATHWISE_NU, size=half)
    frequencies = (normal / lengthscales) * np.sqrt(2.0 * PATHWISE_NU / chi_square)[:, None]
    weights = rng.standard_normal((int(n_features), int(n_realizations)))
    sampler = PathwiseSampler(
        train_x=train_x,
        lengthscales=lengthscales,
        outputscale=outputscale,
        noise=noise,
        constant_mean=mean,
        frequencies=frequencies,
        weights=weights,
        alpha=np.zeros((n_points, int(n_realizations))),
    )
    prior_at_train = sampler.prior_features(train_x) @ weights
    perturbation = rng.standard_normal((n_points, int(n_realizations))) * np.sqrt(noise)
    residual = (
        (np.asarray(gp.train_y, dtype=float) - mean)[:, None] - prior_at_train - perturbation
    )
    covariance = matern52_ard(train_x, train_x, lengthscales, outputscale) + np.eye(
        n_points
    ) * noise
    alpha = np.linalg.solve(covariance, residual)
    return PathwiseSampler(
        train_x=train_x,
        lengthscales=lengthscales,
        outputscale=outputscale,
        noise=noise,
        constant_mean=mean,
        frequencies=frequencies,
        weights=weights,
        alpha=alpha,
    )


def _seed_int(seed_sequence: np.random.SeedSequence) -> int:
    """A 32 bit integer seed for a library that only takes one, drawn from the spawned tree.

    SALib and the SciPy Sobol engine underneath it accept an integer and nothing richer, so
    the entropy has to be narrowed somewhere. It is narrowed here, from a child of the run's
    own ``SeedSequence``, rather than by a literal in a call site (ground rule 13).
    """
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def saltelli_design(config: Config, seed_sequence: np.random.SeedSequence) -> np.ndarray:
    """The scrambled Sobol design of build spec 12.2, in raw feature units.

    Imported as ``from SALib.sample import sobol``. The bare ``SALib.sample.sobol`` attribute
    path raises ``AttributeError`` on SALib 1.5.2 unless the submodule has already been
    imported, which is the packaging gotcha build spec 12.2 warns about and
    ``tests/test_sensitivity.py`` pins; ``SALib.sample.saltelli`` still exists but is
    deprecated and is not used.
    """
    from SALib.sample import sobol as sobol_sample

    settings = config.pipeline.sensitivity
    return np.asarray(
        sobol_sample.sample(
            salib_problem(config),
            2 ** int(settings.gp_sobol_log2_samples),
            calc_second_order=False,
            scramble=True,
            seed=_seed_int(seed_sequence),
        ),
        dtype=float,
    )


def saltelli_indices(
    config: Config, responses: np.ndarray, seed_sequence: np.random.SeedSequence
) -> dict[str, np.ndarray]:
    """Saltelli first order and total indices for every realization, one analysis each.

    Returns four ``(n_realizations, d)`` arrays: the indices themselves and the SALib bootstrap
    confidence half widths. Those half widths cover the Monte Carlo error of one realization's
    estimate and nothing else; they are not the surrogate's uncertainty, which is the spread
    across realizations, and every caption that shows them says so.
    """
    from SALib.analyze import sobol as sobol_analyze

    problem = salib_problem(config)
    settings = config.pipeline.sensitivity
    values = np.asarray(responses, dtype=float)
    n_realizations = values.shape[1]
    children = seed_sequence.spawn(n_realizations)
    first = np.empty((n_realizations, problem["num_vars"]))
    total = np.empty_like(first)
    first_conf = np.empty_like(first)
    total_conf = np.empty_like(first)
    for column in range(n_realizations):
        analysis = sobol_analyze.analyze(
            problem,
            np.ascontiguousarray(values[:, column]),
            calc_second_order=False,
            num_resamples=int(settings.gp_bootstrap_resamples),
            print_to_console=False,
            seed=_seed_int(children[column]),
        )
        first[column] = np.asarray(analysis["S1"], dtype=float)
        total[column] = np.asarray(analysis["ST"], dtype=float)
        first_conf[column] = np.asarray(analysis["S1_conf"], dtype=float)
        total_conf[column] = np.asarray(analysis["ST_conf"], dtype=float)
    return {
        "first_order": first,
        "total_order": total,
        "first_order_conf": first_conf,
        "total_order_conf": total_conf,
    }


def gp_posterior_sobol(
    gp: FittedGP,
    design: np.ndarray,
    config: Config,
    seed_sequence: np.random.SeedSequence,
) -> dict[str, Any]:
    """The build spec 12.2 cross check for one target: an index distribution, not an index."""
    settings = config.pipeline.sensitivity
    path_seed, analysis_seed = seed_sequence.spawn(2)
    sampler = pathwise_sampler(
        gp,
        int(settings.gp_realizations),
        int(settings.gp_fourier_features),
        path_seed,
    )
    responses = sampler(np.asarray(design, dtype=float))
    indices = saltelli_indices(config, responses, analysis_seed)
    indices["kernel_max_abs_deviation"] = sampler.kernel_deviation()
    indices["outputscale"] = float(sampler.outputscale)
    return indices


# ---------------------------------------------------------------------------
# Functional and aggregated indices
# ---------------------------------------------------------------------------


def coefficient_matrix(fits: list[PceFit], n_candidate_terms: int) -> np.ndarray:
    """Pack the per score sparse coefficient vectors into one dense ``(n_terms, k)`` matrix.

    Different scores select different sparse subsets, so they are unified here on the shared
    candidate index set. A term no expansion selected stays zero, which is exactly what it
    means for that term to have been dropped.
    """
    matrix = np.zeros((int(n_candidate_terms), len(fits)), dtype=float)
    for column, fit in enumerate(fits):
        if fit.terms.size and int(fit.terms.max()) >= int(n_candidate_terms):
            raise ValueError(
                f"expansion {fit.name!r} selected term {int(fit.terms.max())} from a candidate "
                f"basis of {n_candidate_terms}, so the two were not built the same way."
            )
        matrix[fit.terms, column] = fit.coefficients
    return matrix


def functional_decomposition(
    basis: ChaosBasis,
    fits: list[PceFit],
    loadings: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Pointwise first order and total indices along a functional block.

    The algebra is the one derived in this module's docstring: with ``A`` the score
    coefficients and ``phi`` the loadings, ``B = A @ phi`` is the chaos expansion of the field
    at every station, and the pointwise variance decomposition follows from orthonormality.

    ``mask`` names the stations where the observed family actually varies. The rest are 0/0,
    for the reason build spec 11.2 already ran into on the damage saturation, and they are
    excluded rather than floored.
    """
    phi = np.atleast_2d(np.asarray(loadings, dtype=float))
    if phi.shape[0] != len(fits):
        raise ValueError(
            f"{len(fits)} score expansions were given {phi.shape[0]} loadings; they describe "
            "the same components and must match."
        )
    coefficients = coefficient_matrix(fits, basis.size)
    field = coefficients @ phi
    exponents = np.vstack([basis.multi_index(term) for term in range(basis.size)])
    active = exponents.sum(axis=1) > 0
    contribution = field**2
    total_variance = contribution[active].sum(axis=0)
    dimension = len(FEATURE_ORDER)
    first = np.zeros((dimension, phi.shape[1]))
    total = np.zeros((dimension, phi.shape[1]))
    for row in np.flatnonzero(active):
        support = np.flatnonzero(exponents[row])
        if support.size == 1:
            first[support[0]] += contribution[row]
        total[support] += contribution[row]
    usable = np.asarray(mask, dtype=bool) & (total_variance > 0.0)
    first_index = np.full_like(first, np.nan)
    total_index = np.full_like(total, np.nan)
    first_index[:, usable] = first[:, usable] / total_variance[usable]
    total_index[:, usable] = total[:, usable] / total_variance[usable]
    return {
        "variance": total_variance,
        "first_order": first_index,
        "total_order": total_index,
        "partial_first": first,
        "partial_total": total,
        "usable": usable,
    }


def aggregated_indices(fits: list[PceFit], eigenvalues: np.ndarray) -> dict[str, Any]:
    """Generalized sensitivity indices over a block, weighted two ways.

    Lamboni, Monod and Makowski 2011 aggregate a multivariate response by weighting each
    component's index with that component's eigenvalue. Two weightings are reported because
    they answer slightly different questions and their gap is itself a measurement:

    - ``chaos``: the weight is the variance each expansion explains. This is the weighting for
      which the aggregate equals the integral of the pointwise partial variance over the
      stations, because the loadings are orthonormal, so the aggregated table and the pointwise
      figure are two views of one decomposition.
    - ``eigenvalue``: the weight is the component's own empirical variance, which is the
      literal Lamboni construction. It differs from the first by exactly the share of each
      component the expansion failed to explain, so a large gap is a statement about Q2 rather
      than about the physics.
    """
    dimension = len(FEATURE_ORDER)
    chaos_weights = np.array([fit.total_variance for fit in fits], dtype=float)
    empirical = np.asarray(eigenvalues, dtype=float).ravel()[: len(fits)]
    out: dict[str, Any] = {}
    for label, weights in (("chaos", chaos_weights), ("eigenvalue", empirical)):
        denominator = float(weights.sum())
        if not denominator > 0.0:
            raise ValueError(
                f"the {label} weights over this block sum to {denominator}, so a generalized "
                "index is 0/0."
            )
        first = np.zeros(dimension)
        total = np.zeros(dimension)
        for weight, fit in zip(weights, fits):
            first += float(weight) * fit.first_order
            total += float(weight) * fit.total_order
        out[label] = {
            "weights": [float(value) for value in weights],
            "first_order": first / denominator,
            "total_order": total / denominator,
        }
    return out


# ---------------------------------------------------------------------------
# Agreement between the two constructions
# ---------------------------------------------------------------------------


def agreement_rows(
    target: str,
    fit: PceFit,
    posterior: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """One row per input and index kind, saying whether the two constructions agree.

    Build spec 12.2's acceptance criterion is that the two agree within their uncertainties.
    The chaos index has no interval in this construction, because it is analytic rather than
    estimated, so the test is containment: the chaos value sits inside the Gaussian process
    posterior 90 percent interval. Where it does not, the gap to the nearest endpoint is
    recorded so the discrepancy can be investigated in writing rather than averaged away.
    """
    rows: list[dict[str, Any]] = []
    for kind, chaos_values, posterior_values in (
        ("first_order", fit.first_order, posterior["first_order"]),
        ("total_order", fit.total_order, posterior["total_order"]),
    ):
        low, median, high = np.percentile(posterior_values, POSTERIOR_QUANTILES, axis=0)
        for index, name in enumerate(FEATURE_ORDER):
            value = float(chaos_values[index])
            bottom, top = float(low[index]), float(high[index])
            inside = bool(bottom <= value <= top)
            gap = 0.0 if inside else float(min(abs(value - bottom), abs(value - top)))
            rows.append(
                {
                    "target": target,
                    "input": name,
                    "kind": kind,
                    "pce": value,
                    "gp_median": float(median[index]),
                    "gp_low": bottom,
                    "gp_high": top,
                    "agrees": inside,
                    "gap": gap,
                    "publication_level": fit.publication,
                    "q2_corrected": fit.q2_corrected,
                }
            )
    return rows


def ranking(values: np.ndarray) -> list[str]:
    """Input names ordered by decreasing index, which is what a gated target may publish."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    return [FEATURE_ORDER[int(index)] for index in order]


# ---------------------------------------------------------------------------
# Generated fragments
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "{--}"
    if abs(value) < 0.5 * 10.0 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


#: How a publication level reads in a table.
PUBLICATION_TEX: dict[str, str] = {
    PUBLICATION_VALUES: "values",
    PUBLICATION_RANKINGS: "indicative",
    PUBLICATION_WITHHELD: "withheld",
}


def build_sobol_table(records: dict[str, Any], targets: list[str]) -> str:
    """Chaos indices against the Gaussian process posterior, per target and per input."""
    lines = [
        "% Generated by the sensitivity stage (ufem.sensitivity). Do not edit.",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Quantity & Input & $Q^2$ & $S_i$ & $T_i$ & GP median $S_i$ [90\,\%] & Publish \\",
        r"\midrule",
    ]
    for target in targets:
        record = records[target]
        for position, name in enumerate(FEATURE_ORDER):
            chaos = record["pce"]
            posterior = record.get("gp")
            interval = "{--}"
            if posterior is not None:
                interval = (
                    f"{_fmt(posterior['first_order']['median'][position])} "
                    f"[{_fmt(posterior['first_order']['low'][position], 2)}, "
                    f"{_fmt(posterior['first_order']['high'][position], 2)}]"
                )
            first = target_label(target) if position == 0 else ""
            q2 = _fmt(chaos["q2_corrected"]) if position == 0 else ""
            publish = PUBLICATION_TEX[chaos["publication_level"]] if position == 0 else ""
            withheld = chaos["publication_level"] == PUBLICATION_WITHHELD
            values = (
                ("{--}", "{--}")
                if withheld
                else (
                    _fmt(chaos["first_order"][position]),
                    _fmt(chaos["total_order"][position]),
                )
            )
            lines.append(
                f"{first} & {INPUT_MATH[name]} & {q2} & {values[0]} & {values[1]} & "
                f"{'{--}' if withheld else interval} & {publish} \\\\"
            )
        lines.append(r"\addlinespace")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


#: What a withheld cell prints as. A dash is not a small number, and a table that showed the
#: value in grey would still be a table a reader could quote from.
WITHHELD_CELL = "{--}"


def _index_cells(values: list[float], level: str) -> str:
    """Six index cells, or six dashes when the gate withholds this row."""
    if level == PUBLICATION_WITHHELD:
        return " & ".join([WITHHELD_CELL] * len(values))
    return " & ".join(_fmt(value) for value in values)


def build_aggregated_table(payload: dict[str, Any]) -> str:
    """Generalized indices per block, with the per component indices build spec 12.3 wants.

    Every row carries the gate. A block whose weakest component is withheld prints dashes
    rather than numbers, because an aggregate is not more trustworthy than its worst part and
    a greyed out number is still a number a reader can quote.
    """
    lines = [
        "% Generated by the sensitivity stage (ufem.sensitivity). Do not edit.",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Block & Weighting & \multicolumn{3}{c}{First order $S_i$} & "
        r"\multicolumn{3}{c}{Total $T_i$} & Publish \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & $f_{cm}$ & $c_{\mathrm{bot}}$ & $c_{\mathrm{top}}$ & "
        r"$f_{cm}$ & $c_{\mathrm{bot}}$ & $c_{\mathrm{top}}$ & \\",
        r"\midrule",
    ]
    for block in FUNCTIONAL_BLOCKS:
        block_record = payload["functional"][block]
        level = block_record["publication_level"]
        record = block_record["aggregated"]
        for position, weighting in enumerate(("chaos", "eigenvalue")):
            name = block.capitalize() if position == 0 else ""
            cells = _index_cells(
                list(record[weighting]["first_order"]) + list(record[weighting]["total_order"]),
                level,
            )
            publish = PUBLICATION_TEX[level] if position == 0 else ""
            lines.append(f"{name} & {weighting} & {cells} & {publish} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1]
    lines.append(r"\midrule")
    for block in FUNCTIONAL_BLOCKS:
        for component in payload["functional"][block]["components"][:3]:
            cells = _index_cells(
                list(component["first_order"]) + list(component["total_order"]),
                component["publication_level"],
            )
            lines.append(
                f"{target_label(component['name'])} & component & {cells} & "
                f"{PUBLICATION_TEX[component['publication_level']]} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_gate_table(records: dict[str, Any], targets: list[str]) -> str:
    """The phase's own result: what every expansion measured and what it may publish.

    This table is the one that ships whatever the outcome, because the Q2 values, the
    explainable ceiling the Gaussian process nugget implies, and the design roughness are
    measurements of the campaign rather than statements about the beam, so no gate withholds
    them.
    """
    lines = [
        "% Generated by the sensitivity stage (ufem.sensitivity). Do not edit.",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Target & Terms & $Q^2$ & Ceiling & Roughness & Publish \\",
        r"\midrule",
    ]
    for target in targets:
        record = records[target]["pce"]
        lines.append(
            f"{target_label(target)} & {record['n_terms']} & "
            f"{_fmt(record['q2_corrected'])} & "
            f"{_fmt(record['explainable_variance_ceiling'])} & "
            f"{_fmt(record['design_roughness']['roughness_ratio'])} & "
            f"{PUBLICATION_TEX[record['publication_level']]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_markdown_summary(payload: dict[str, Any]) -> str:
    """The stage's own readable summary, regenerated on every run."""
    out: list[str] = []
    add = out.append
    add("# Global sensitivity summary")
    add("")
    add(
        f"Generated by the `sensitivity` stage for config "
        f"`{payload['config_sha256'][:12]}`, over {payload['context']['n_runs']} runs."
    )
    add("")
    add("## The Q2 gate of build spec 12.1")
    add("")
    add(
        "| Target | corrected LOO Q2 | terms | explainable ceiling | roughness | "
        "publication level |"
    )
    add("|---|---|---|---|---|---|")
    for target in payload["context"]["targets"]:
        record = payload["targets"][target]["pce"]
        add(
            f"| {target} | {record['q2_corrected']:.4f} | {record['n_terms']} | "
            f"{record['explainable_variance_ceiling']:.3f} | "
            f"{record['design_roughness']['roughness_ratio']:.3f} | "
            f"{record['publication_level']} |"
        )
    add("")
    add(
        "The ceiling is the share of each target's variance the fitted Gaussian process "
        "nugget says the three inputs determine at this design density; the roughness is the "
        "median absolute response difference between the closest tenth of design pairs, as a "
        "share of the response standard deviation. Both are diagnostics, not gates."
    )
    add("")
    add("## Chaos indices against the Gaussian process posterior")
    add("")
    add("| Target | Input | S_i (PCE) | GP median | GP 90 percent | agrees |")
    add("|---|---|---|---|---|---|")
    for target in payload["context"]["headline"]:
        record = payload["targets"][target]
        posterior = record.get("gp")
        for position, name in enumerate(FEATURE_ORDER):
            interval = (
                f"[{posterior['first_order']['low'][position]:.3f}, "
                f"{posterior['first_order']['high'][position]:.3f}]"
                if posterior is not None
                else "not computed"
            )
            median = (
                f"{posterior['first_order']['median'][position]:.3f}"
                if posterior is not None
                else "not computed"
            )
            agrees = [
                row
                for row in payload["agreement"]["rows"]
                if row["target"] == target
                and row["input"] == name
                and row["kind"] == "first_order"
            ]
            verdict = "yes" if agrees and agrees[0]["agrees"] else "no"
            add(
                f"| {target} | {name} | {record['pce']['first_order'][position]:.3f} | "
                f"{median} | {interval} | {verdict} |"
            )
    add("")
    add(
        f"Agreement over all assessed rows: {payload['agreement']['n_agree']} of "
        f"{payload['agreement']['n_rows']}."
    )
    add("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _load_inputs(
    root: Path, config: Config, config_sha256: str
) -> tuple[Path, Path, Path, dict[str, str]]:
    artifact_root = root / config.pipeline.paths.artifact_root
    grid_dir = stage_dir(artifact_root, GRID_STAGE, config_sha256)
    register_dir = stage_dir(artifact_root, REGISTER_STAGE, config_sha256)
    surrogate_dir = stage_dir(artifact_root, SURROGATE_STAGE, config_sha256)
    calibrate_dir = stage_dir(artifact_root, CALIBRATE_STAGE, config_sha256)
    hashes: dict[str, str] = {}
    for directory, name, stage in (
        (grid_dir, RF2_GRID_PARQUET, GRID_STAGE),
        (grid_dir, DAMAGE_GRID_PARQUET, GRID_STAGE),
        (grid_dir, QOI_PARQUET, GRID_STAGE),
        (register_dir, AMPLITUDE_PARQUET, REGISTER_STAGE),
        (surrogate_dir, SURROGATE_JSON, SURROGATE_STAGE),
        (surrogate_dir, GP_STATE_NPY, SURROGATE_STAGE),
        (surrogate_dir, TRAINING_TARGETS_NPY, SURROGATE_STAGE),
        (calibrate_dir, CALIBRATION_JSON, CALIBRATE_STAGE),
    ):
        path = directory / name
        if not path.is_file():
            raise SensitivityInputMissing(
                f"the sensitivity stage needs {path}, which does not exist. Run "
                f"`ufem run {stage}` first."
            )
        hashes[name] = sha256_file(path)
    return grid_dir, register_dir, surrogate_dir, hashes


def declared_input_hashes(
    repo_root: Path | str, config: Config, config_sha256: str
) -> dict[str, str]:
    """Hash this stage's declared inputs as they are on disk right now (see ``ufem.runner``)."""
    return _load_inputs(Path(repo_root), config, config_sha256)[-1]


def _calibration_gate_passed(root: Path, config: Config, config_sha256: str) -> dict[str, Any]:
    """Refuse to publish a sensitivity index behind a calibration gate that did not pass.

    Build spec 11.5 blocks the propagated numbers of section 13 on this gate. Sensitivity is
    not propagation, and the chaos expansion here is fitted on the training responses rather
    than on the surrogate, so this stage is not strictly blocked. It reads the verdict anyway
    and records it, because the Gaussian process cross check of build spec 12.2 draws from
    exactly the posteriors the gate is a statement about, and an index distribution from an
    uncalibrated posterior is a width nobody has checked.
    """
    directory = stage_dir(
        root / config.pipeline.paths.artifact_root, CALIBRATE_STAGE, config_sha256
    )
    payload = json.loads((directory / CALIBRATION_JSON).read_text(encoding="utf-8"))
    gate = payload["gate"]
    if not gate.get("passed", False):
        raise AssertionError(
            "the calibration gate of build spec 11.5 did not pass for this config hash, so "
            f"the posteriors this stage draws from are not calibrated. Failing checks: "
            f"{gate.get('failing')}. Fix the model, not this stage."
        )
    return {"passed": True, "failing": list(gate.get("failing", []))}


def sensitivity_targets(surrogate: SurrogateModel) -> list[str]:
    """Every target build spec 12.1 fits an expansion for, in report order.

    The scalar quantities of interest of build spec 9.5, then the retained registered
    amplitude scores, then the damage scores. The phase and displacement blocks are absent on
    purpose: they carry the reparameterization rather than the response, so an index on them
    answers which input moves the abscissa, which is a question about the representation and
    not about the beam.
    """
    names = list(SCALAR_QOI)
    names += list(surrogate.score_targets[BLOCK_AMPLITUDE])
    names += list(surrogate.score_targets[BLOCK_DAMAGE])
    return names


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the sensitivity stage and return its artifact directory."""
    started = _time.perf_counter()
    configure_torch()
    root = Path(repo_root)
    grid_dir, register_dir, surrogate_dir, input_hashes = _load_inputs(
        root, config, config_sha256
    )
    artifact_root = root / config.pipeline.paths.artifact_root
    calibration_gate = _calibration_gate_passed(root, config, config_sha256)

    qoi = pd.read_parquet(grid_dir / QOI_PARQUET)
    force_frame = pd.read_parquet(grid_dir / RF2_GRID_PARQUET)
    damage_frame = pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET)
    amplitude_frame = pd.read_parquet(register_dir / AMPLITUDE_PARQUET)
    jobs, _force = curve_matrix(force_frame)
    damage_jobs, damage = curve_matrix(damage_frame)
    amplitude_jobs, amplitude = curve_matrix(amplitude_frame)
    if list(qoi["job"].astype(str)) != jobs or amplitude_jobs != jobs or damage_jobs != jobs:
        raise AssertionError(
            "the QoI table, the gridded curves and the registered amplitude family carry "
            "different job orders, so no row wise comparison between them is valid."
        )

    surrogate = SurrogateModel.load(artifact_root, config_sha256)
    settings = config.pipeline.sensitivity
    basis = chaos_basis(config)
    X = features(qoi)
    standardized_design = surrogate.feature_standardizer.transform(X)
    targets = sensitivity_targets(surrogate)
    # The responses are read back from the surrogate stage's own target matrix rather than
    # recomputed here. The scores in particular exist only as a product of the reduction, and
    # a second projection in this stage would be a second chance for them to differ.
    target_matrix = np.load(surrogate_dir / TRAINING_TARGETS_NPY)
    stored = list(surrogate.metadata["target_order"])
    missing = [name for name in targets if name not in stored]
    if missing:
        raise KeyError(
            f"the surrogate artifact carries no training column for {missing}. The "
            "sensitivity target list must be a subset of what the surrogate stage fitted."
        )
    training = {name: target_matrix[:, stored.index(name)] for name in targets}

    pce_started = _time.perf_counter()
    fits = {name: fit_pce(X, training[name], name, basis) for name in targets}
    pce_seconds = _time.perf_counter() - pce_started

    seed_root = np.random.SeedSequence(config.pipeline.seed_entropy)
    design_seed, target_seed = seed_root.spawn(2)
    design_raw = saltelli_design(config, design_seed)
    design = surrogate.feature_standardizer.transform(design_raw)
    children = dict(zip(targets, target_seed.spawn(len(targets))))

    gp_started = _time.perf_counter()
    posteriors: dict[str, dict[str, Any]] = {}
    for name in targets:
        posteriors[name] = gp_posterior_sobol(
            surrogate.models[name], design, config, children[name]
        )
    gp_seconds = _time.perf_counter() - gp_started

    # ---- assembled records -------------------------------------------------
    records: dict[str, Any] = {}
    pce_rows: list[dict[str, Any]] = []
    gp_rows: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    for name in targets:
        fit = fits[name]
        posterior = posteriors[name]
        summary: dict[str, Any] = {}
        for kind in ("first_order", "total_order"):
            low, median, high = np.percentile(
                posterior[kind], POSTERIOR_QUANTILES, axis=0
            )
            summary[kind] = {
                "low": [float(value) for value in low],
                "median": [float(value) for value in median],
                "high": [float(value) for value in high],
                "salib_conf_median": [
                    float(value)
                    for value in np.median(posterior[f"{kind}_conf"], axis=0)
                ],
            }
        ceiling = explainable_variance_ceiling(surrogate.models[name])
        roughness = design_roughness(standardized_design, training[name])
        records[name] = {
            "pce": {
                "q2_corrected": fit.q2_corrected,
                "q2_plain": fit.q2_plain,
                "correction_factor": fit.correction_factor,
                "n_terms": fit.n_terms,
                "publication_level": fit.publication,
                "explainable_variance_ceiling": ceiling,
                "q2_share_of_ceiling": (
                    float(fit.q2_corrected / ceiling) if ceiling > 0.0 else float("nan")
                ),
                "design_roughness": roughness,
                "first_order": [float(value) for value in fit.first_order],
                "total_order": [float(value) for value in fit.total_order],
                "interaction_share": fit.interaction_share,
                "interactions": fit.interactions,
                "ranking": ranking(fit.first_order),
                "total_ranking": ranking(fit.total_order),
            },
            "gp": {
                **summary,
                "kernel_max_abs_deviation": float(
                    posterior["kernel_max_abs_deviation"]
                ),
                "outputscale": float(posterior["outputscale"]),
                "ranking": ranking(np.array(summary["first_order"]["median"])),
            },
        }
        for position, feature in enumerate(FEATURE_ORDER):
            pce_rows.append(
                {
                    "target": name,
                    "input": feature,
                    "first_order": float(fit.first_order[position]),
                    "total_order": float(fit.total_order[position]),
                    "q2_corrected": fit.q2_corrected,
                    "n_terms": fit.n_terms,
                    "publication_level": fit.publication,
                    "explainable_variance_ceiling": ceiling,
                    "roughness_ratio": roughness["roughness_ratio"],
                }
            )
            for kind in ("first_order", "total_order"):
                gp_rows.append(
                    {
                        "target": name,
                        "input": feature,
                        "kind": kind,
                        "median": summary[kind]["median"][position],
                        "low": summary[kind]["low"][position],
                        "high": summary[kind]["high"][position],
                        "salib_conf_median": summary[kind]["salib_conf_median"][position],
                    }
                )
        agreement += agreement_rows(name, fit, posterior)

    # ---- functional indices ------------------------------------------------
    functional_started = _time.perf_counter()
    station_displacement = surrogate.basis.displacement_from_tangent(
        np.zeros(surrogate.basis.stations.size)
    )
    functional: dict[str, Any] = {}
    functional_rows: list[pd.DataFrame] = []
    block_setup = {
        BLOCK_AMPLITUDE: (
            surrogate.score_targets[BLOCK_AMPLITUDE],
            surrogate.basis.amplitude.components[: surrogate.basis.n_amplitude],
            station_displacement,
            informative_abscissae(amplitude),
        ),
        BLOCK_DAMAGE: (
            surrogate.score_targets[BLOCK_DAMAGE],
            surrogate.basis.damage.components[: surrogate.basis.n_damage],
            surrogate.basis.u_grid,
            informative_abscissae(damage),
        ),
    }
    for block, (names, loadings, abscissa, mask) in block_setup.items():
        block_fits = [fits[name] for name in names]
        decomposition = functional_decomposition(basis, block_fits, loadings, mask)
        eigenvalues = np.array(
            [np.var(training[name], ddof=1) for name in names], dtype=float
        )
        aggregate = aggregated_indices(block_fits, eigenvalues)
        block_level = weakest_publication_level([fit.publication for fit in block_fits])
        functional[block] = {
            "publication_level": block_level,
            "component_levels": {fit.name: fit.publication for fit in block_fits},
            "n_components": len(names),
            "n_stations": int(loadings.shape[1]),
            "n_usable_stations": int(decomposition["usable"].sum()),
            "abscissa_mm": [float(value) for value in abscissa],
            "abscissa_basis": (
                "Karcher mean displacement coordinate along the arc length stations; the "
                "indices are computed on the registered amplitude functions, so the axis is a "
                "mean physical displacement rather than any one run's"
                if block == BLOCK_AMPLITUDE
                else "the common 201 point displacement grid; the damage family is not "
                "registered, so this axis is the physical one directly"
            ),
            "aggregated": {
                weighting: {
                    "weights": aggregate[weighting]["weights"],
                    "first_order": [
                        float(value) for value in aggregate[weighting]["first_order"]
                    ],
                    "total_order": [
                        float(value) for value in aggregate[weighting]["total_order"]
                    ],
                }
                for weighting in ("chaos", "eigenvalue")
            },
            "components": [
                {
                    "name": name,
                    "q2_corrected": fits[name].q2_corrected,
                    "publication_level": fits[name].publication,
                    "first_order": [float(value) for value in fits[name].first_order],
                    "total_order": [float(value) for value in fits[name].total_order],
                }
                for name in names
            ],
            "identity_check": float(
                np.abs(
                    decomposition["partial_first"].sum(axis=1)
                    - np.array(
                        [
                            sum(
                                fit.first_order[index] * fit.total_variance
                                for fit in block_fits
                            )
                            for index in range(len(FEATURE_ORDER))
                        ]
                    )
                ).max()
            ),
        }
        frame = pd.DataFrame(
            {
                "block": block,
                "publication_level": block_level,
                "station": np.arange(loadings.shape[1]),
                "u_mm": np.asarray(abscissa, dtype=float),
                "variance": decomposition["variance"],
                "usable": decomposition["usable"],
            }
        )
        for position, feature in enumerate(FEATURE_ORDER):
            frame[f"S_{feature}"] = decomposition["first_order"][position]
            frame[f"T_{feature}"] = decomposition["total_order"][position]
        functional_rows.append(frame)
    functional_seconds = _time.perf_counter() - functional_started

    headline = [
        name
        for name in config.pipeline.validation.headline_qoi
        if name in records
    ]
    n_agree = int(sum(1 for row in agreement if row["agrees"]))
    payload = {
        "config_sha256": config_sha256,
        "context": {
            "n_runs": len(jobs),
            "targets": targets,
            "headline": headline,
            "feature_order": list(FEATURE_ORDER),
            "pce_total_degree": basis.total_degree,
            "pce_hyperbolic_q": basis.hyperbolic_q,
            "pce_candidate_terms": basis.size,
            "gp_realizations": int(settings.gp_realizations),
            "gp_sobol_samples": 2 ** int(settings.gp_sobol_log2_samples),
            "gp_design_rows": int(design.shape[0]),
            "gp_fourier_features": int(settings.gp_fourier_features),
            "gp_bootstrap_resamples": int(settings.gp_bootstrap_resamples),
            "q2_publish_values": Q2_PUBLISH_VALUES,
            "q2_publish_rankings": Q2_PUBLISH_RANKINGS,
            "posterior_quantiles": list(POSTERIOR_QUANTILES),
            "calibration_gate": calibration_gate,
        },
        "targets": records,
        "functional": functional,
        "agreement": {
            "n_rows": len(agreement),
            "n_agree": n_agree,
            "criterion": (
                "the analytic chaos index lies inside the Gaussian process posterior 90 "
                "percent interval; the chaos index carries no interval of its own because it "
                "is a decomposition rather than an estimate"
            ),
            "rows": agreement,
        },
        "publication_counts": {
            level: int(
                sum(
                    1
                    for name in targets
                    if records[name]["pce"]["publication_level"] == level
                )
            )
            for level in (PUBLICATION_VALUES, PUBLICATION_RANKINGS, PUBLICATION_WITHHELD)
        },
        "gate_outcome": weakest_publication_level(
            [records[name]["pce"]["publication_level"] for name in targets]
        ),
        "diagnostic_note": (
            "the explainable variance ceiling and the design roughness are model free or "
            "independently modeled measurements of the campaign, not of the beam, so no "
            "publication gate applies to them; they exist to say whether a failed Q2 is the "
            "expansion's fault or the campaign's"
        ),
    }

    directory = stage_dir(artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, text in (
        (SENSITIVITY_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (SOBOL_TEX, build_sobol_table(records, headline)),
        (AGGREGATED_TEX, build_aggregated_table(payload)),
        (GATE_TEX, build_gate_table(records, targets)),
        (SENSITIVITY_MD, build_markdown_summary(payload)),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8", newline="\n")
        outputs.append(path)
    for frame, name in (
        (pd.DataFrame(pce_rows), PCE_INDICES_PARQUET),
        (pd.DataFrame(gp_rows), GP_INDICES_PARQUET),
        (pd.concat(functional_rows, ignore_index=True), FUNCTIONAL_INDICES_PARQUET),
        (pd.DataFrame(agreement), AGREEMENT_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "pce_wall_time_s": pce_seconds,
        "gp_sobol_wall_time_s": gp_seconds,
        "functional_wall_time_s": functional_seconds,
        "n_runs": len(jobs),
        "n_targets": len(targets),
        "publication_counts": payload["publication_counts"],
        "q2_by_target": {
            name: records[name]["pce"]["q2_corrected"] for name in targets
        },
        "agreement": {
            "n_rows": len(agreement),
            "n_agree": n_agree,
        },
        "gp_sobol_samples": payload["context"]["gp_sobol_samples"],
        "gp_realizations": payload["context"]["gp_realizations"],
        "kernel_max_abs_deviation": {
            name: records[name]["gp"]["kernel_max_abs_deviation"] for name in targets
        },
        "calibration_gate": calibration_gate,
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
    counts = payload["publication_counts"]
    print(
        f"[sensitivity] {len(targets)} expansions over {len(jobs)} runs "
        f"({counts[PUBLICATION_VALUES]} publish values, {counts[PUBLICATION_RANKINGS]} "
        f"rankings only, {counts[PUBLICATION_WITHHELD]} withheld); chaos in "
        f"{pce_seconds:.1f} s, {settings.gp_realizations} posterior realizations on a "
        f"{payload['context']['gp_design_rows']} point Saltelli design in {gp_seconds:.1f} s; "
        f"chaos and posterior agree on {n_agree} of {len(agreement)} index rows"
    )
    return directory
