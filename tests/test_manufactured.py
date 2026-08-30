"""The manufactured solution test of build spec 16.1, the strongest evidence in the project.

Everywhere else the oracle problem bites: nobody knows the true load displacement curve at an
unseen input, so no test can compare a prediction against the truth. Here the truth is
constructed. A family of softening curves is generated from an analytic function of the three
standardized inputs, with a rising branch times an exponential tail whose peak load, peak
location and initial stiffness are explicit smooth functions of those inputs, plus a small
seeded perturbation so the problem is not noiseless. The real pipeline is then run on it, the
same code the production stage runs, and the held out error is compared against a threshold
and against itself at three sample sizes.

What passing means: the machinery can learn a response it is capable of representing, the
error falls as the sample grows, and it falls to a stated level at the size of the real
campaign. What failing would mean: something between the design matrix and the reconstructed
curve is wrong in a way no amount of staring at real data would reveal, because on real data
every prediction is defensible.

**Registration is deliberately skipped.** The synthetic curves are generated already aligned
in the arc length parameter, so there is no phase variation for an elastic registration to
find, and running one would cost the test its wall clock budget to fit a warp to nothing. The
curve representation is therefore built with ``registration="identity"``, which is the only
place in the repository that setting is used, and the phase block is empty by construction.
Everything else is the production path: the same arc length reparameterization, the same
displacement block in the square root slope tangent space, the same principal component
reduction, the same Gaussian processes with the same restart policy, and the same fold
harness.

Marked slow. Measured at about 100 seconds on the target machine for the three sample sizes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from ufem.config import load_config  # noqa: E402
from ufem.surrogate import (  # noqa: E402
    BLOCK_AMPLITUDE,
    BLOCK_DAMAGE,
    BLOCK_DISPLACEMENT,
    BLOCK_PHASE,
    REGISTRATION_IDENTITY,
    CurveBasis,
    GPSettings,
    Standardizer,
    configure_torch,
    fit_all,
)
from ufem.validate import make_folds, relative_l2  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.fullstack]

#: Sample sizes of build spec 16.1, ending at the size of the real campaign.
SAMPLE_SIZES: tuple[int, int, int] = (64, 128, 198)

#: The error the manufactured family must reach at the campaign size, as a relative L2 norm
#: on the physical curve. Stated in advance: the truncation floor of the representation on
#: this family is around 1 percent, the response is smooth in three inputs, and a surrogate
#: that cannot get inside 10 percent of a curve it has 198 clean samples of is broken.
THRESHOLD_AT_FULL_SIZE = 0.10

#: Relative amplitude of the seeded perturbation, as a fraction of the peak load. Small enough
#: that the response is learnable and large enough that the problem is not an interpolation
#: exercise. The perturbation is observation noise: it enters the training curves and is absent
#: from the exact family the predictions are scored against.
PERTURBATION = 0.02

#: How much of the n = 64 error must be gone by n = 198. Stated in advance rather than fitted
#: to the outcome: at three times the sample a smooth response in three inputs should lose a
#: clear fraction of its error, and a decrease too small to see would satisfy monotonicity
#: while meaning nothing.
IMPROVEMENT_FACTOR = 0.8


def manufactured_family(
    X: np.ndarray, u_grid: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A softening family whose every feature is a smooth analytic function of the inputs.

    Returns the observed family, the damage family, and the exact family the observations
    are a perturbation of. The error is scored against the exact one, which is the whole point
    of a manufactured solution: the perturbation is observation noise that no surrogate can or
    should reproduce, and scoring against it would put a floor under the error that hides
    whether the method is converging at all.

    The shape is a rising branch times an exponential tail,

        P(u) = P_peak * (1 - exp(-u / a)) * exp(-((u - u_peak) / w)^2 / 2)  for u > u_peak,

    written so that the peak load, the displacement at peak, the rise constant and the
    softening width are each an explicit smooth function of the standardized inputs. The
    damage curve is a saturating exponential with its own smooth rate, which mirrors the
    monotone saturating family the real campaign produced.

    Units follow the project: force in N, displacement in mm, damage dimensionless.
    """
    standardized = Standardizer.fit(X).transform(X)
    f, b, t = standardized[:, 0], standardized[:, 1], standardized[:, 2]
    peak = 38000.0 * (
        1.0
        + 0.09 * f
        + 0.02 * t
        - 0.015 * b
        + 0.05 * f * t
        + 0.06 * np.sin(2.4 * f)
        + 0.04 * np.cos(2.0 * t)
        + 0.03 * np.sin(1.9 * b)
    )
    u_peak = 11.0 * (
        1.0
        + 0.10 * f
        + 0.06 * t
        - 0.03 * f**2
        + 0.07 * np.sin(2.1 * t)
        + 0.05 * f * b
    )
    rise = 1.4 * (1.0 + 0.10 * t - 0.05 * b + 0.08 * np.cos(2.3 * f))
    width = 9.0 * (1.0 + 0.12 * f - 0.05 * t + 0.10 * np.sin(1.8 * b))
    rate = 0.45 * (1.0 + 0.15 * f + 0.08 * t + 0.10 * np.sin(2.2 * b))

    u = np.asarray(u_grid, dtype=float)
    force = np.empty((X.shape[0], u.size), dtype=float)
    exact = np.empty_like(force)
    damage = np.empty_like(force)
    for row in range(X.shape[0]):
        rising = 1.0 - np.exp(-u / rise[row])
        tail = np.where(
            u <= u_peak[row],
            1.0,
            np.exp(-0.5 * ((u - u_peak[row]) / width[row]) ** 2),
        )
        exact[row] = peak[row] * rising * tail
        force[row] = exact[row] + PERTURBATION * peak[row] * rng.standard_normal(u.size) * (
            u / u.max()
        )
        damage[row] = 0.947 * (1.0 - np.exp(-rate[row] * u))
    force[:, 0] = 0.0
    exact[:, 0] = 0.0
    return force, damage, exact


def manufactured_design(n: int, rng: np.random.Generator) -> np.ndarray:
    """A design in the units of the feature contract: strength in MPa, covers in mm."""
    return np.column_stack(
        [
            rng.uniform(20.0, 38.0, size=n),
            rng.uniform(19.0, 35.0, size=n),
            rng.uniform(210.0, 236.0, size=n),
        ]
    )


def fold_error(
    X: np.ndarray,
    force: np.ndarray,
    damage: np.ndarray,
    exact: np.ndarray,
    u_grid: np.ndarray,
    config,
    settings: GPSettings,
    seed_sequence: np.random.SeedSequence,
    n_folds: int,
) -> np.ndarray:
    """Held out relative L2 error against the exact family, through the production path."""
    labels = [f"manufactured_{index:04d}" for index in range(X.shape[0])]
    folds = make_folds(labels, n_folds, seed_sequence)
    children = seed_sequence.spawn(len(folds))
    errors = []
    for index, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(X.shape[0]), test_index)
        basis = CurveBasis.fit(
            u_grid,
            force[train_index],
            damage[train_index],
            config,
            registration=REGISTRATION_IDENTITY,
        )
        assert basis.n_phase == 0, "identity registration must leave the phase block empty"
        targets: dict[str, np.ndarray] = {}
        order: list[tuple[str, int]] = []
        for block in (BLOCK_AMPLITUDE, BLOCK_DAMAGE, BLOCK_DISPLACEMENT):
            scores = basis.scores[block]
            for column in range(scores.shape[1]):
                name = f"{block}_pc{column + 1}"
                targets[name] = scores[:, column]
                order.append((block, column))
        models, feature_standardizer, target_standardizers, _log = fit_all(
            X[train_index], targets, settings, children[index]
        )
        standardized = feature_standardizer.transform(X[test_index])
        blocks = {
            block: np.zeros((test_index.size, basis.block_counts[block]))
            for block in (BLOCK_AMPLITUDE, BLOCK_PHASE, BLOCK_DAMAGE, BLOCK_DISPLACEMENT)
        }
        for name, (block, column) in zip(targets, order):
            mean, _variance = models[name].predict(standardized)
            blocks[block][:, column] = (
                target_standardizers[name].inverse_mean(mean.reshape(-1, 1)).ravel()
            )
        predicted = basis.reconstruct_force(
            blocks[BLOCK_AMPLITUDE], blocks[BLOCK_PHASE], blocks[BLOCK_DISPLACEMENT]
        )
        errors.append(relative_l2(exact[test_index], predicted))
    return np.concatenate(errors)


@pytest.fixture(scope="module")
def manufactured(repo_root):
    """Fit and validate the manufactured family at each sample size, once for the module."""
    configure_torch()
    config = load_config(repo_root)
    settings = GPSettings.from_config(config)
    u_grid = np.linspace(
        config.pipeline.grid.u_min_mm,
        config.pipeline.grid.u_max_mm,
        config.pipeline.grid.n_points,
    )
    started = time.perf_counter()
    results = {}
    for size in SAMPLE_SIZES:
        root = np.random.SeedSequence(20260830 + size)
        design_seed, curve_seed, fold_seed = root.spawn(3)
        X = manufactured_design(size, np.random.default_rng(design_seed))
        force, damage, exact = manufactured_family(
            X, u_grid, np.random.default_rng(curve_seed)
        )
        results[size] = fold_error(
            X, force, damage, exact, u_grid, config, settings, fold_seed, n_folds=4
        )
    results["wall_time_s"] = time.perf_counter() - started
    return results


class TestManufacturedSolution:
    def test_the_error_falls_monotonically_as_the_sample_grows(self, manufactured):
        """Build spec 16.1: the error must decrease at the expected rate from 64 to 198.

        Compared on the median rather than the mean, because a single badly extrapolated
        design point in a small sample would otherwise decide the comparison. The claim under
        test is about the typical curve.
        """
        medians = [float(np.median(manufactured[size])) for size in SAMPLE_SIZES]
        assert medians == sorted(medians, reverse=True), (
            "the held out error did not fall monotonically with the sample size: "
            + ", ".join(
                f"n={size}: {value:.4f}" for size, value in zip(SAMPLE_SIZES, medians)
            )
            + ". A surrogate whose error does not improve with more data is not learning."
        )

    def test_the_error_at_the_campaign_size_is_below_the_stated_threshold(
        self, manufactured
    ):
        median = float(np.median(manufactured[SAMPLE_SIZES[-1]]))
        assert median < THRESHOLD_AT_FULL_SIZE, (
            f"at n = {SAMPLE_SIZES[-1]} the median held out relative L2 error is "
            f"{median:.4f}, above the stated threshold of {THRESHOLD_AT_FULL_SIZE}. On a "
            "family generated from a smooth analytic function of the three inputs, that is a "
            "defect in the pipeline rather than a property of the data."
        )

    def test_the_improvement_from_the_smallest_to_the_largest_sample_is_material(
        self, manufactured
    ):
        """A decrease of a fraction of a percent would satisfy monotonicity and mean nothing."""
        first = float(np.median(manufactured[SAMPLE_SIZES[0]]))
        last = float(np.median(manufactured[SAMPLE_SIZES[-1]]))
        assert last < IMPROVEMENT_FACTOR * first, (
            f"the median error fell only from {first:.4f} to {last:.4f} between n = "
            f"{SAMPLE_SIZES[0]} and n = {SAMPLE_SIZES[-1]}, which is not the improvement a "
            "learnable response should show."
        )

    def test_every_held_out_curve_got_a_finite_prediction(self, manufactured):
        for size in SAMPLE_SIZES:
            errors = manufactured[size]
            assert errors.size == size
            assert np.all(np.isfinite(errors))

    def test_the_test_stays_inside_its_wall_clock_budget(self, manufactured):
        """The budget is part of the design: a slow test is a test that gets skipped."""
        assert manufactured["wall_time_s"] < 300.0, (
            f"the manufactured solution test took {manufactured['wall_time_s']:.0f} s against "
            "a 300 s budget. Skipping the registration on synthetic data is what keeps it "
            "affordable; if it has grown past this, find out what changed."
        )
