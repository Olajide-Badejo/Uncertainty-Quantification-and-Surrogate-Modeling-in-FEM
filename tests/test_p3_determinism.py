"""Build spec 17.2 and 16.1: the P3 stages reproduce their artifacts bitwise.

The registration stage is the one place in the pipeline where a third party library does the
numerical heavy lifting, so its determinism is a measurement rather than an assumption.
fdasrsf's ``srsf_align`` accepts a ``parallel`` flag, and the parallel path splits the family
across workers, which makes the association order of the Karcher mean depend on scheduling.
The stage calls it with ``parallel=False`` for exactly this reason.

The test reruns the stage and compares the recorded output digests. It is marked slow because
a rerun costs about 13 seconds of SRVF, and marked fullstack because it needs fdasrsf and the
upstream artifacts.

If this test ever fails, the correct response is to measure the actual reproducibility and
record it in docs/DESIGN_DECISIONS.md with a tolerance gate, not to delete the test. Build
spec 17.2 allows a downgrade to statistically reproducible where it is stated; it does not
allow the question to go unasked.
"""

from __future__ import annotations

import pytest

from ufem.config import config_hash, load_config
from ufem.manifest import load_manifest, stage_dir
from ufem.register import STAGE_NAME as REGISTER_STAGE
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
class TestRegistrationIsBitwiseReproducible:
    def test_a_forced_rerun_reproduces_every_output_digest(self, repo_root):
        """The SRVF path is deterministic on this machine, measured rather than assumed."""
        before = _digests(repo_root, REGISTER_STAGE)
        config = load_config(repo_root)
        assert run_stage(repo_root, config, REGISTER_STAGE, force=True) == 0
        after = _digests(repo_root, REGISTER_STAGE)
        assert set(before) == set(after)
        differing = sorted(name for name in before if before[name] != after[name])
        assert not differing, (
            f"the register stage did not reproduce {differing} bitwise. Measure the actual "
            "reproducibility and record it in docs/DESIGN_DECISIONS.md with a tolerance "
            "gate, per build spec 17.2; do not delete this test."
        )
