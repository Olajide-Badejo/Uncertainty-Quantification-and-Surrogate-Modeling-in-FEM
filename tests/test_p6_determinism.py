"""Build spec 17.2 and 16.1: the sensitivity stage reproduces its artifacts bitwise.

Third of the determinism siblings, after ``test_p4_determinism.py`` and
``test_p5_determinism.py``, and the one with the most surface to get wrong. The polynomial
chaos half is deterministic linear algebra and cannot drift. The Gaussian process half draws
4096 spectral frequencies, 4096 by 200 prior weights and 198 by 200 noise perturbations per
target, then hands an integer seed to a scrambled Sobol engine and another to SALib's
bootstrap, for every one of 24 targets. That is five places a global random number generator
could have been reached for instead of the spawned ``SeedSequence`` tree, and this is the test
that would say so.

Marked slow because a forced rerun pays the whole Saltelli budget, about eleven minutes.

If it ever fails, measure the actual reproducibility and record it in
docs/DESIGN_DECISIONS.md with a tolerance gate per build spec 17.2. Do not delete the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufem.config import config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.runner import run_stage
from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE


def _digests(repo_root, stage_name: str) -> dict[str, str]:
    config = load_config(repo_root)
    directory = stage_dir(
        repo_root / config.pipeline.paths.artifact_root, stage_name, config_hash(config)
    )
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"the {stage_name} stage has not run for this config hash: {directory}")
    return {
        record["name"]: record["sha256"] for record in load_manifest(directory)["outputs"]
    }


class TestTheStochasticPartsAreSeeded:
    """The cheap half of the gate: the two random draws, without rerunning the stage."""

    def test_the_saltelli_design_depends_only_on_the_spawned_seed(self, repo_root):
        from ufem.sensitivity import saltelli_design

        config = load_config(repo_root)
        entropy = config.pipeline.seed_entropy
        first = saltelli_design(config, np.random.SeedSequence(entropy).spawn(2)[0])
        second = saltelli_design(config, np.random.SeedSequence(entropy).spawn(2)[0])
        assert np.array_equal(first, second)
        other = saltelli_design(config, np.random.SeedSequence(entropy + 1).spawn(2)[0])
        assert not np.array_equal(first, other), (
            "two different entropies produced the same Sobol design, which is what a hard "
            "coded seed inside the sampler would look like"
        )

    def test_the_pathwise_draws_depend_only_on_the_spawned_seed(self):
        from ufem.sensitivity import pathwise_sampler
        from ufem.surrogate import GPSettings

        settings = GPSettings(
            nu=2.5,
            ard=True,
            lengthscale_bounds=(0.11, 10.0),
            restarts=1,
            noise_prior_median_variance=0.1,
            noise_prior_log_scale=1.5,
            max_iterations=10,
        )
        rng = np.random.default_rng(0)
        design = rng.uniform(-1.5, 1.5, size=(30, 3))
        response = np.cos(design[:, 0]) + 0.3 * design[:, 1]

        class _Stub:
            name = "stub"

            def __init__(self):
                self.train_x = design
                self.train_y = response
                self.settings = settings

            def lengthscales(self):
                return np.array([1.0, 2.0, 0.8])

            def outputscale(self):
                return 1.1

            def noise(self):
                return 0.08

            def constant_mean(self):
                return -0.2

        query = rng.uniform(-1.5, 1.5, size=(64, 3))
        first = pathwise_sampler(_Stub(), 16, 256, np.random.SeedSequence(4242))(query)
        second = pathwise_sampler(_Stub(), 16, 256, np.random.SeedSequence(4242))(query)
        assert np.array_equal(first, second)
        other = pathwise_sampler(_Stub(), 16, 256, np.random.SeedSequence(4243))(query)
        assert not np.array_equal(first, other)

    def test_the_chunk_size_does_not_change_a_realization(self):
        """A realization is a function, so evaluating it in pieces gives the same answer.

        The same answer, not the same bytes. BLAS picks its blocking from the shape of the
        matrices it is handed, so a 300 by 512 product and eight 37 by 512 products sum their
        inner dimension in different orders and land 3e-15 apart on a value of order one. That
        is float64 associativity and not a seeding defect, so what is asserted here is
        agreement at round off. The consequence is recorded rather than hidden: the chunk size
        is a module constant, and it is part of the artifact contract that the bitwise gate of
        build spec 17.2 holds under. Changing it changes the last bit of the stage's outputs.
        """
        from ufem.sensitivity import pathwise_sampler
        from ufem.surrogate import GPSettings

        settings = GPSettings(
            nu=2.5,
            ard=True,
            lengthscale_bounds=(0.11, 10.0),
            restarts=1,
            noise_prior_median_variance=0.1,
            noise_prior_log_scale=1.5,
            max_iterations=10,
        )
        rng = np.random.default_rng(6)
        design = rng.uniform(-1.5, 1.5, size=(25, 3))

        class _Stub:
            name = "stub"

            def __init__(self):
                self.train_x = design
                self.train_y = rng.standard_normal(25)
                self.settings = settings

            def lengthscales(self):
                return np.array([1.0, 1.0, 1.0])

            def outputscale(self):
                return 1.0

            def noise(self):
                return 0.1

            def constant_mean(self):
                return 0.0

        sampler = pathwise_sampler(_Stub(), 8, 512, np.random.SeedSequence(7))
        query = rng.uniform(-1.5, 1.5, size=(300, 3))
        whole = sampler(query, chunk=300)
        pieces = sampler(query, chunk=37)
        np.testing.assert_allclose(pieces, whole, rtol=0.0, atol=1e-12)
        # And bitwise identical at a fixed chunk size, which is what the stage actually uses.
        assert np.array_equal(whole, sampler(query, chunk=300))


@pytest.mark.slow
@pytest.mark.fullstack
class TestSensitivityIsBitwiseReproducible:
    def test_a_forced_rerun_reproduces_every_output_digest(self, repo_root):
        before = _digests(repo_root, SENSITIVITY_STAGE)
        config = load_config(repo_root)
        assert run_stage(repo_root, config, SENSITIVITY_STAGE, force=True) == 0
        after = _digests(repo_root, SENSITIVITY_STAGE)
        assert set(before) == set(after)
        differing = sorted(name for name in before if before[name] != after[name])
        assert not differing, (
            f"the sensitivity stage did not reproduce {differing} bitwise. Measure the actual "
            "reproducibility and record it in docs/DESIGN_DECISIONS.md with a tolerance gate, "
            "per build spec 17.2; do not delete this test."
        )

    def test_the_gate_outcome_is_recorded_in_the_manifest(self, repo_root):
        """What the stage published has to be readable from the artifact, not from a console."""
        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            SENSITIVITY_STAGE,
            config_hash(config),
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip("the sensitivity stage has not run for this config hash")
        extra = load_manifest(directory)["extra"]
        counts = extra["publication_counts"]
        assert sum(counts.values()) == extra["n_targets"]
        assert set(extra["q2_by_target"]) == set(extra["kernel_max_abs_deviation"])
        assert extra["calibration_gate"]["passed"] is True
