"""Build spec 17.2 and 16.1: the propagate stage reproduces its artifacts bitwise.

Fourth of the determinism siblings, after the P4, P5 and P6 modules. The surface it guards is
different from theirs: this stage draws 300000 uniforms for the input sample, one subsample of
20000 indices, one more of 2000, and 64 posterior draws at each of 20000 points for each of
eleven targets. That is fourteen places a global generator could have been reached for instead
of the spawned ``SeedSequence`` tree, and one of them, the subsample, would have produced a
plausible looking artifact that simply described a different subset every run.

The light stack guard is at module import, not inside the tests. That is the lesson of the
three DEFECT_LOG entries whose common shape is a gate verified only on a machine that happened
to have the full stack: the fast CI jobs install neither torch nor gpytorch, and a module that
imports the stage without saying so fails at collection rather than skipping.

If it ever fails, measure the actual reproducibility and record it in docs/DESIGN_DECISIONS.md
with a tolerance gate per build spec 17.2. Do not delete the test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from ufem.config import config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE
from ufem.propagate import draw_inputs, subsample_indices
from ufem.runner import run_stage


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
    """The cheap half of the gate: every draw, without rerunning the stage."""

    def test_the_input_sample_depends_only_on_the_spawned_seed(self, repo_root):
        config = load_config(repo_root)
        entropy = config.pipeline.seed_entropy
        first = draw_inputs(config, 2000, np.random.SeedSequence(entropy).spawn(4)[0])
        second = draw_inputs(config, 2000, np.random.SeedSequence(entropy).spawn(4)[0])
        assert np.array_equal(first, second)
        other = draw_inputs(config, 2000, np.random.SeedSequence(entropy + 1).spawn(4)[0])
        assert not np.array_equal(first, other), (
            "two different entropies produced the same input sample, which is what a hard "
            "coded seed inside the sampler would look like"
        )

    def test_the_subsamples_depend_only_on_their_own_spawned_seeds(self, repo_root):
        """The two subsamples come from different children, so they must not coincide.

        If the epistemic subsample and the curve subsample were drawn from the same child, the
        2000 curves in the fan would be a prefix of the 20000 rows the epistemic layer used,
        and the stage would be reporting two views of the same rows as two products.
        """
        config = load_config(repo_root)
        children = np.random.SeedSequence(config.pipeline.seed_entropy).spawn(4)
        epistemic = subsample_indices(100000, 2000, children[1])
        curve = subsample_indices(100000, 2000, children[3])
        assert np.array_equal(epistemic, subsample_indices(100000, 2000, children[1]))
        assert not np.array_equal(epistemic, curve)


@pytest.mark.slow
@pytest.mark.fullstack
class TestPropagationIsBitwiseReproducible:
    def test_a_forced_rerun_reproduces_every_output_digest(self, repo_root):
        before = _digests(repo_root, PROPAGATE_STAGE)
        config = load_config(repo_root)
        assert run_stage(repo_root, config, PROPAGATE_STAGE, force=True) == 0
        after = _digests(repo_root, PROPAGATE_STAGE)
        assert set(before) == set(after)
        differing = sorted(name for name in before if before[name] != after[name])
        assert not differing, (
            f"the propagate stage did not reproduce {differing} bitwise. Measure the actual "
            "reproducibility and record it in docs/DESIGN_DECISIONS.md with a tolerance gate, "
            "per build spec 17.2; do not delete this test."
        )

    def test_the_manifest_carries_what_a_reader_has_to_see_beside_a_failure_probability(
        self, repo_root
    ):
        """Build spec 13: no probability without its bound, its floor, and its domain mass."""
        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            PROPAGATE_STAGE,
            config_hash(config),
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip("the propagate stage has not run for this config hash")
        extra = load_manifest(directory)["extra"]
        assert extra["calibration_gate"]["passed"] is True
        assert 0.0 <= extra["out_of_domain_fraction"] <= 1.0
        assert extra["resolvable_pf_floor"] == 1.0e-4
        assert extra["roughness_ratio"] > 0.0
        for record in extra["limit_states"].values():
            assert record["pf_conservative"] >= record["pf_point"]
            assert record["pf_standard_error"] >= 0.0
        assert set(extra["analytic_cross_check"]) == {
            "median_ratio",
            "central_tendency_brackets",
            "dispersion_brackets",
            "model_error",
        }
