"""The four baselines of build spec 10.5, under one interface with the surrogate.

Binding law 3, and the sentence in build spec 10.5 that this module exists to make
enforceable: *a surrogate that does not beat the stated dumb baselines is reported as
failing, not tuned until the test set is memorized.* The predecessor's single feature linear
fit on the concrete strength already reached about 0.49 on the peak load, and every published
Gaussian process number in that project was worse than that without anyone noticing, because
nobody ever ran the two through the same harness.

So these four are not decoration and they are not an appendix. They are the measuring stick,
and the rules they are held to are deliberately strict in their own favor:

- **No tuning.** Not one of them has a hyperparameter fitted on the data. The neighbour count
  and the polynomial degree are declared in the config and in the build spec respectively, and
  nothing here looks at an error before choosing anything.
- **One interface.** Every model, the Gaussian process included, is a
  :class:`Regressor` with ``fit`` and ``predict``, or a :class:`CurvePredictor` with
  ``predict_curves``. ``ufem.validate`` calls them through those two protocols and cannot tell
  them apart, which is what makes the comparison a comparison.
- **Same folds, same basis, same standardization.** The harness hands every model the same
  training rows and the same in fold reduction basis. A baseline that got the production basis
  while the surrogate got a refitted one would be a rigged race in the surrogate's favour.

The four, in the order build spec 10.5 lists them:

a. :class:`ClimatologyRegressor` and :class:`MeanCurveModel`, the training mean. The floor
   nothing may fall below, and the reference a skill score is measured against.
b. :class:`LinearRegressor`, ordinary least squares on the three standardized features.
c. :class:`QuadraticChaosRegressor`, the full quadratic polynomial chaos expansion.
d. :class:`NearestNeighborRegressor` and :class:`NearestNeighborCurveModel`, an inverse
   distance weighted average over the three nearest training runs.

Units: every model here is unit agnostic. It is handed a design matrix in
(MPa, mm, mm) and a target matrix in whatever units the targets carry, and returns predictions
in those same target units. The standardization of the features is internal to each model and
is fitted on the training rows it was given, never on the rows it is asked about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: Build spec 10.5 calls the quadratic chaos expansion "15 terms at d = 3". That count is for
#: four inputs: the number of multi indices of total degree at most 2 in d dimensions is
#: (d + 2) choose 2, which is 15 at d = 4 and 10 at d = 3. The feature contract of build spec
#: 9.2 has three inputs, because E is derived rather than independent, so the expansion here
#: has 10 terms. The discrepancy is arithmetic in the spec, not a reduction of the baseline,
#: and it is recorded in docs/DESIGN_DECISIONS.md as build spec section 24 requires.
CHAOS_DEGREE = 2


@runtime_checkable
class Regressor(Protocol):
    """A model from an ``(n, d)`` design to an ``(n, k)`` target matrix."""

    name: str

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Regressor: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class CurvePredictor(Protocol):
    """A model that predicts whole curves on the common displacement grid."""

    name: str

    def predict_curves(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def _as_design(X: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(X, dtype=float))
    if matrix.ndim != 2:
        raise ValueError(f"a design matrix must be 2D, got shape {matrix.shape}.")
    return matrix


def _as_targets(Y: np.ndarray) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"a target matrix must be 1D or 2D, got shape {matrix.shape}.")
    return matrix


@dataclass
class _FeatureScaling:
    """Training mean and standard deviation of each feature column.

    Every model in this module standardizes with these before it does anything else, for the
    same reason the surrogate does: a distance, a polynomial, and a lengthscale all depend on
    the units of the columns, and a comparison in which one model saw millimetres and another
    saw standardized units would measure the units.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> _FeatureScaling:
        matrix = _as_design(X)
        scale = matrix.std(axis=0, ddof=1)
        if np.any(scale <= 0.0):
            raise ValueError(
                "a feature column has zero spread in the training rows, so it cannot be "
                "standardized. A constant column is not a feature."
            )
        return cls(mean=matrix.mean(axis=0), scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (_as_design(X) - self.mean) / self.scale


class ClimatologyRegressor:
    """Baseline (a): predict the training mean of every target, whatever the input.

    Its out of sample R2 is zero by construction up to the finite sample correction, which is
    exactly why it belongs in the table: a model with a negative R2 is worse than knowing
    nothing, and without this row in front of it that fact is easy to talk around.
    """

    name = "climatology"

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ClimatologyRegressor:
        del X
        self._mean = _as_targets(Y).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("climatology.predict called before fit.")
        return np.tile(self._mean, (_as_design(X).shape[0], 1))


class LinearRegressor:
    """Baseline (b): ordinary least squares with an intercept on the standardized features."""

    name = "linear"

    def __init__(self) -> None:
        self._scaling: _FeatureScaling | None = None
        self._coefficients: np.ndarray | None = None

    def _basis(self, X: np.ndarray) -> np.ndarray:
        if self._scaling is None:
            raise RuntimeError("linear.predict called before fit.")
        standardized = self._scaling.transform(X)
        return np.column_stack([np.ones(standardized.shape[0]), standardized])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> LinearRegressor:
        self._scaling = _FeatureScaling.fit(X)
        design = self._basis(X)
        self._coefficients = np.linalg.lstsq(design, _as_targets(Y), rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("linear.predict called before fit.")
        return self._basis(X) @ self._coefficients


def hermite_multi_indices(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """Every multi index of total degree at most ``degree`` over ``n_features`` variables.

    Returned in graded lexicographic order, so the basis column order is a stated convention
    rather than whatever a recursion happened to produce.
    """
    if n_features < 1 or degree < 0:
        raise ValueError(
            f"hermite_multi_indices needs at least one feature and a non negative degree, "
            f"got {n_features} and {degree}."
        )
    indices: list[tuple[int, ...]] = []

    def walk(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == n_features:
            indices.append(prefix)
            return
        for power in range(remaining + 1):
            walk((*prefix, power), remaining - power)

    walk((), degree)
    return sorted(indices, key=lambda item: (sum(item), tuple(-value for value in item)))


def probabilists_hermite(x: np.ndarray, order: int) -> np.ndarray:
    """The probabilists' Hermite polynomial ``He_n`` evaluated elementwise.

    Only the orders the expansion needs are written out. ``He_0 = 1``, ``He_1 = x``,
    ``He_2 = x^2 - 1``: orthogonal under the standard normal weight, which is the family a
    polynomial chaos expansion in standardized Gaussian inputs is built from.
    """
    values = np.asarray(x, dtype=float)
    if order == 0:
        return np.ones_like(values)
    if order == 1:
        return values
    if order == 2:
        return values**2 - 1.0
    raise ValueError(
        f"probabilists_hermite is written for orders 0 to 2, got {order}. Build spec 10.5 "
        "specifies the full quadratic expansion; a higher degree needs the recursion and a "
        "decision about truncation, not a silent extension here."
    )


class QuadraticChaosRegressor:
    """Baseline (c): the full quadratic polynomial chaos expansion, fitted by least squares.

    The basis is the tensor product of probabilists' Hermite polynomials over the standardized
    features, truncated at total degree two: 10 terms for the three input feature contract.
    Coefficients come from ordinary least squares, which is the projection build spec 10.5
    asks for, and nothing about the basis or the fit is selected on the data.
    """

    name = "quadratic_chaos"

    def __init__(self, degree: int = CHAOS_DEGREE) -> None:
        self.degree = int(degree)
        self._scaling: _FeatureScaling | None = None
        self._indices: list[tuple[int, ...]] | None = None
        self._coefficients: np.ndarray | None = None

    @property
    def n_terms(self) -> int:
        if self._indices is None:
            raise RuntimeError("quadratic_chaos.n_terms is only known after fit.")
        return len(self._indices)

    def _basis(self, X: np.ndarray) -> np.ndarray:
        if self._scaling is None or self._indices is None:
            raise RuntimeError("quadratic_chaos.predict called before fit.")
        standardized = self._scaling.transform(X)
        columns = []
        for multi_index in self._indices:
            term = np.ones(standardized.shape[0])
            for column, order in enumerate(multi_index):
                term = term * probabilists_hermite(standardized[:, column], order)
            columns.append(term)
        return np.column_stack(columns)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> QuadraticChaosRegressor:
        self._scaling = _FeatureScaling.fit(X)
        self._indices = hermite_multi_indices(_as_design(X).shape[1], self.degree)
        design = self._basis(X)
        targets = _as_targets(Y)
        if design.shape[0] < design.shape[1]:
            raise ValueError(
                f"the quadratic chaos expansion has {design.shape[1]} terms but only "
                f"{design.shape[0]} training rows. An underdetermined least squares fit would "
                "return one of infinitely many solutions without saying so."
            )
        self._coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("quadratic_chaos.predict called before fit.")
        return self._basis(X) @ self._coefficients


def inverse_distance_weights(
    query: np.ndarray, train: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour indices and inverse distance weights for each query row.

    A query that coincides with a training point gets all of its weight on the coincident
    points rather than a division by zero. That case is not hypothetical in a leave one out
    harness, where a duplicated design row would put a training point exactly on the query,
    and answering it with an infinity would be a silent failure rather than a loud one.
    """
    queries = _as_design(query)
    training = _as_design(train)
    if training.shape[0] < n_neighbors:
        raise ValueError(
            f"the inverse distance baseline needs at least {n_neighbors} training rows, got "
            f"{training.shape[0]}."
        )
    distances = np.linalg.norm(queries[:, None, :] - training[None, :, :], axis=2)
    order = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
    nearest = np.take_along_axis(distances, order, axis=1)
    weights = np.empty_like(nearest)
    for row in range(nearest.shape[0]):
        exact = nearest[row] <= 0.0
        if np.any(exact):
            weights[row] = exact.astype(float)
        else:
            weights[row] = 1.0 / nearest[row]
        weights[row] = weights[row] / weights[row].sum()
    return order, weights


class NearestNeighborRegressor:
    """Baseline (d) on scores and scalars: inverse distance weighted nearest neighbours."""

    name = "nearest_neighbour"

    def __init__(self, n_neighbors: int = 3) -> None:
        self.n_neighbors = int(n_neighbors)
        self._scaling: _FeatureScaling | None = None
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> NearestNeighborRegressor:
        self._scaling = _FeatureScaling.fit(X)
        self._train_x = self._scaling.transform(X)
        self._train_y = _as_targets(Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._scaling is None or self._train_x is None or self._train_y is None:
            raise RuntimeError("nearest_neighbour.predict called before fit.")
        order, weights = inverse_distance_weights(
            self._scaling.transform(X), self._train_x, self.n_neighbors
        )
        return np.einsum("ij,ijk->ik", weights, self._train_y[order])


@dataclass
class MeanCurveModel:
    """Baseline (a) at curve level: the training mean curve, whatever the input.

    Build spec 10.5 names this climatology, and it is the reference the report's skill scores
    are measured against.
    """

    force: np.ndarray
    damage: np.ndarray
    name: str = "climatology"

    @classmethod
    def fit(cls, X: np.ndarray, force: np.ndarray, damage: np.ndarray) -> MeanCurveModel:
        del X
        return cls(
            force=np.asarray(force, dtype=float).mean(axis=0),
            damage=np.asarray(damage, dtype=float).mean(axis=0),
        )

    def predict_curves(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = _as_design(X).shape[0]
        return np.tile(self.force, (rows, 1)), np.tile(self.damage, (rows, 1))


@dataclass
class NearestNeighborCurveModel:
    """Baseline (d) at curve level: an inverse distance weighted average of whole curves.

    Deliberately an average of curves rather than of scores. Build spec 10.5 specifies a
    "3 nearest neighbor inverse distance curve average", and averaging curves is the version
    of that which owes nothing to the reduction: it is the baseline a reader who distrusts the
    whole functional principal component apparatus would propose, so it has to be run in the
    plane the apparatus is trying to beat it in.
    """

    scaling: _FeatureScaling
    train_x: np.ndarray
    force: np.ndarray
    damage: np.ndarray
    n_neighbors: int
    name: str = "nearest_neighbour"

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        force: np.ndarray,
        damage: np.ndarray,
        n_neighbors: int = 3,
    ) -> NearestNeighborCurveModel:
        scaling = _FeatureScaling.fit(X)
        return cls(
            scaling=scaling,
            train_x=scaling.transform(X),
            force=np.asarray(force, dtype=float),
            damage=np.asarray(damage, dtype=float),
            n_neighbors=int(n_neighbors),
        )

    def predict_curves(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order, weights = inverse_distance_weights(
            self.scaling.transform(X), self.train_x, self.n_neighbors
        )
        return (
            np.einsum("ij,ijk->ik", weights, self.force[order]),
            np.einsum("ij,ijk->ik", weights, self.damage[order]),
        )


def build_baseline_regressors(n_neighbors: int) -> list[Regressor]:
    """The three baselines that predict scores and scalars, in report order.

    Climatology is here as well as at curve level because a scalar QoI needs its own mean
    prediction; the curve level version of it lives in :class:`MeanCurveModel`.
    """
    return [
        ClimatologyRegressor(),
        LinearRegressor(),
        QuadraticChaosRegressor(),
        NearestNeighborRegressor(n_neighbors=n_neighbors),
    ]
