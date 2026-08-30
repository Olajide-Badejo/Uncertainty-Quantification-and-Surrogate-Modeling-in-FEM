"""Build spec 17.2 and 16.1: the surrogate stage reproduces its artifacts bitwise.

Single threaded torch with deterministic algorithms on, float64 throughout, and every restart
initialization drawn from a ``SeedSequence`` spawned off the configured entropy: nothing in
``ufem.surrogate`` touches a torch random number generator. The stage also recomputes its own
SRVF registration and reduction, which is the P3 machinery this test's sibling,
``test_p3_determinism.py``, already measures deterministic on this machine.

The test reruns the stage and compares the recorded output digests, forced so the cache cannot
serve the first run's artifacts back as a trivial pass. It is marked slow because a forced
rerun costs the full 45 target Gaussian process fit, about a minute against the build spec 10.3
budget, and marked fullstack because it needs torch, gpytorch and fdasrsf plus the upstream
artifacts.

If this test ever fails, the correct response is to measure the actual reproducibility and
record it in docs/DESIGN_DECISIONS.md with a tolerance gate, not to delete the test. Build spec
17.2 allows a downgrade to statistically reproducible where it is stated; it does not allow the
question to go unasked.
"""

from __future__ import annotations

import pytest

from ufem.config import config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.runner import run_stage
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE


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
class TestSurrogateIsBitwiseReproducible:
    def test_a_forced_rerun_reproduces_every_output_digest(self, repo_root):
        """Every ``.npy`` and ``.json`` artifact, gp_state.npy included, matches byte for byte."""
        before = _digests(repo_root, SURROGATE_STAGE)
        config = load_config(repo_root)
        assert run_stage(repo_root, config, SURROGATE_STAGE, force=True) == 0
        after = _digests(repo_root, SURROGATE_STAGE)
        assert set(before) == set(after)
        differing = sorted(name for name in before if before[name] != after[name])
        assert not differing, (
            f"the surrogate stage did not reproduce {differing} bitwise. Measure the actual "
            "reproducibility and record it in docs/DESIGN_DECISIONS.md with a tolerance gate, "
            "per build spec 17.2; do not delete this test."
        )
