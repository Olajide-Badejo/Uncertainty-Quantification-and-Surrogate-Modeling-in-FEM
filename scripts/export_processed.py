"""Copy the small grid stage products into ``data/processed/`` for self containment.

Build spec 3.3: nothing over 5 MB enters git. The full resolution per signal Parquets stay
in the gitignored artifact store (the load displacement table alone is about 19 MB); what is
committed is the gridded 198 by 201 matrices, the scalar QoI table, and the 400 row design,
which together come to about 1 MB and let a fresh clone read the pipeline's output without
the raw CSVs.

Every file written here is a byte copy of a stage output whose SHA-256 is recorded in the
corresponding manifest, so a committed copy that drifts from the artifact store is
detectable rather than merely unlikely. Refuses to write anything at or above the limit.

Exit 0 is clean, exit 1 names the failure.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ufem.config import config_hash, load_config
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import DESIGN_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import sha256_file, stage_dir

PROCESSED_DIR = "data/processed"
LIMIT_BYTES = 5 * 1024 * 1024

#: Stage name to the outputs of that stage which are small enough to commit.
EXPORTS: dict[str, tuple[str, ...]] = {
    GRID_STAGE: (RF2_GRID_PARQUET, DAMAGE_GRID_PARQUET, QOI_PARQUET),
    INGEST_STAGE: (DESIGN_PARQUET,),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    config = load_config(root)
    digest = config_hash(config)
    artifact_root = root / config.pipeline.paths.artifact_root
    destination = root / PROCESSED_DIR
    destination.mkdir(parents=True, exist_ok=True)

    total = 0
    for stage_name, names in EXPORTS.items():
        directory = stage_dir(artifact_root, stage_name, digest)
        for name in names:
            source = directory / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"cannot export {name}: {source} does not exist. Run "
                    f"`ufem run {stage_name}` for config {digest[:12]} first."
                )
            size = source.stat().st_size
            if size >= LIMIT_BYTES:
                raise ValueError(
                    f"{source} is {size / 1024 / 1024:.1f} MB, at or above the 5 MB limit of "
                    "build spec 3.3. It stays in the artifact store and is referenced by "
                    "manifest hash instead of being committed."
                )
            target = destination / name
            shutil.copyfile(source, target)
            total += size
            print(f"{PROCESSED_DIR}/{name}: {size / 1024:.0f} KB, sha256 {sha256_file(target)}")
    print(f"export_processed: {total / 1024 / 1024:.2f} MB written to {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
