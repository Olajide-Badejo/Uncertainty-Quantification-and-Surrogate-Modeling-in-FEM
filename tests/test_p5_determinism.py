"""Build spec 17.2 and 16.1: the calibrate stage reproduces its artifacts bitwise.

Sibling of ``test_p4_determinism.py``, and the reason it is worth its own file is that the
calibration stage has two independent stochastic surfaces where the surrogate stage has one.
The closed form leave one out and the conformal quantiles are pure linear algebra over the
fitted artifact and cannot drift. The 10 fold CV+ cross check refits 11 Gaussian processes
inside each of 10 folds, drawing every restart initialization from the same spawned
``SeedSequence`` positions ``ufem.validate`` uses, and the modulation cross check draws 256
posterior realizations per curve. If either had reached for a global random number generator,
this is the test that would say so.

Marked slow because the forced rerun pays the full fold refit, about two minutes.

If it ever fails, measure the actual reproducibility and record it in
docs/DESIGN_DECISIONS.md with a tolerance gate per build spec 17.2. Do not delete the test.
"""

from __future__ import annotations

import pytest

from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
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


@pytest.mark.slow
@pytest.mark.fullstack
class TestCalibrateIsBitwiseReproducible:
    def test_a_forced_rerun_reproduces_every_output_digest(self, repo_root):
        before = _digests(repo_root, CALIBRATE_STAGE)
        config = load_config(repo_root)
        assert run_stage(repo_root, config, CALIBRATE_STAGE, force=True) == 0
        after = _digests(repo_root, CALIBRATE_STAGE)
        assert set(before) == set(after)
        differing = sorted(name for name in before if before[name] != after[name])
        assert not differing, (
            f"the calibrate stage did not reproduce {differing} bitwise. Measure the actual "
            "reproducibility and record it in docs/DESIGN_DECISIONS.md with a tolerance gate, "
            "per build spec 17.2; do not delete this test."
        )

    def test_the_gate_verdict_is_recorded_in_the_manifest(self, repo_root):
        """A gate that passed has to be readable from the artifact, not only from a console."""
        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root,
            CALIBRATE_STAGE,
            config_hash(config),
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip("the calibrate stage has not run for this config hash")
        gate = load_manifest(directory)["extra"]["gate"]
        assert gate["passed"] is True and gate["failing"] == []
        # A manifest only exists when the gate passed: the stage raises before writing one
        # otherwise, which is what keeps a failed calibration from becoming a cache hit.
        assert all(check["passed"] for check in gate["checks"])
