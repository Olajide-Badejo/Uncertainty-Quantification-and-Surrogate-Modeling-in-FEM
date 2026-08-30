"""Stage ``ingest``: the two raw CSVs and the LHS design into typed, deduplicated Parquet.

Build spec section 9.3. This is the only stage that reads the inherited CSVs. It enforces
dtypes, sorts by ``(job, time)``, applies the strict increasing time filter that the solver
cutbacks make necessary, verifies the displacement control law, and writes one Parquet per
signal into the artifact store with the raw SHA-256 digests recorded in the manifest.

Units: ``time`` is the dimensionless Abaqus step time in [0, 1], ``U2`` is displacement in
mm, ``RF2`` is reaction force in N, ``DAMAGEC_max`` is the dimensionless compression damage
scalar in [0, 1], ``Fcm_MPa`` is mean compressive strength in MPa, and both covers are in mm.

Ground rule 8: every check here raises with a named diagnostic rather than repairing the
data quietly. The v1 extraction hard coded its 198 sample list as an unexplained literal;
this stage derives every count from the bytes it read and asserts the ones the audit pinned.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.config import FEATURE_ORDER, Config
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest

STAGE_NAME = "ingest"

#: The inherited file names under ``config.pipeline.paths.raw_data``.
LOAD_CSV = "load_displacement_full.csv"
DAMAGE_CSV = "damage_evolution_full.csv"
DESIGN_CSV = "uq_lhs_samples_training.csv"

#: Output file names inside the stage directory.
LOAD_PARQUET = "load_displacement.parquet"
DAMAGE_PARQUET = "damage_evolution.parquet"
DESIGN_PARQUET = "design.parquet"

#: Typed schema of each raw table. ``job`` stays a string; ``sample_id`` is derived.
LOAD_DTYPES = {"time": "float64", "U2": "float64", "RF2": "float64"}
DAMAGE_DTYPES = {"time": "float64", "U2": "float64", "DAMAGEC_max": "float64"}

#: Pinned by the 2026-08-28 audit and by build spec sections 6.1 and 9.3. Measured again on
#: every run: 26 jobs carry duplicated time stamps from solver cutbacks, 165 rows in total.
EXPECTED_DEDUP_JOBS = 26
EXPECTED_DEDUP_ROWS = 165

#: Build spec 6.1: every valid run is displacement controlled at U2 = 20 t mm exactly.
DISPLACEMENT_RATE_MM = 20.0
U2_TOLERANCE_MM = 1e-3

#: Build spec 6.1 and 5.5: 198 of the 400 designed samples produced data, in both signals.
EXPECTED_JOBS = 198
EXPECTED_DESIGN_ROWS = 400

#: Build spec 6.1: the realized design's cross correlations are all under this in magnitude.
DESIGN_MAX_ABS_CORRELATION = 0.05

#: The design carries E_MPa and the seed alongside the three independent inputs. E is never
#: a feature (build spec 9.2); it is carried through so the audit stage can re prove the
#: Eurocode 2 collinearity from the committed Parquet rather than from the CSV.
DESIGN_COLUMNS = ("sample_id", *FEATURE_ORDER, "E_MPa", "seed")


def raw_paths(repo_root: Path, config: Config) -> dict[str, Path]:
    """The three inherited inputs, resolved against the configured raw data directory."""
    raw_dir = Path(repo_root) / config.pipeline.paths.raw_data
    return {
        "load_displacement_csv": raw_dir / LOAD_CSV,
        "damage_evolution_csv": raw_dir / DAMAGE_CSV,
        "design_csv": raw_dir / DESIGN_CSV,
    }


def _require_file(path: Path, role: str) -> Path:
    """Ground rule 8: a missing input is a named failure, never an empty result set."""
    if not path.is_file():
        raise FileNotFoundError(
            f"ingest cannot read the {role}: {path} does not exist. It is expected under "
            f"the configured paths.raw_data directory; the inherited CSVs are staged in "
            "legacy_salvage/data and are deliberately not tracked by git (5 MB rule)."
        )
    return path


def job_to_sample_id(job: "pd.Series[Any]") -> "pd.Series[Any]":
    """Integer sample id from a ``sample_NNN`` job label.

    The design table is keyed by ``sample_id`` and the extracted curves by ``job``, so this
    is the one place the two namespaces are joined. A label that does not end in digits
    raises rather than becoming a sentinel: the v1 audit's ``jobid_to_int`` returned -1 on
    failure inside a bare except, which silently merged every malformed row onto one key.
    """
    text = job.astype("string")
    suffix = text.str.rsplit("_", n=1).str[-1]
    bad = text[~suffix.str.fullmatch(r"\d+").fillna(False)]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} job labels do not end in an integer sample id, for example "
            f"{sorted(set(bad.tolist()))[:5]}. The extracted data is expected to use the "
            "sample_NNN convention that keys the LHS design."
        )
    return suffix.astype("int64")


def strict_increasing_mask(values: np.ndarray) -> np.ndarray:
    """Keep the first row of each run of non increasing values.

    Given an array already sorted ascending, returns a boolean mask that keeps a row only
    when it strictly exceeds the last kept value. On these curves that means the first of
    each duplicated time stamp survives and the solver cutback repeats are dropped, which
    is what ``np.interp`` needs: a strictly increasing abscissa.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros(0, dtype=bool)
    return np.concatenate(([True], np.diff(array) > 0.0))


def deduplicate_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the strict increasing time filter per job.

    ``frame`` must already be sorted by ``(job, time)``. Returns the filtered frame and a
    per job report with the rows removed, so the count that build spec 9.3 asserts is
    measured from the data rather than assumed.
    """
    required = {"job", "time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"deduplicate_by_time needs columns {sorted(required)}; the frame is missing "
            f"{missing} and offers {list(frame.columns)}."
        )
    keep = np.zeros(len(frame), dtype=bool)
    removed_per_job: dict[Any, int] = {}
    times = frame["time"].to_numpy(dtype=float)
    codes, uniques = pd.factorize(frame["job"], sort=False)
    for position, job in enumerate(uniques):
        rows = np.flatnonzero(codes == position)
        mask = strict_increasing_mask(times[rows])
        keep[rows] = mask
        removed_per_job[job] = int((~mask).sum())
    report = pd.DataFrame(
        {"job": list(removed_per_job), "rows_removed": list(removed_per_job.values())}
    )
    return frame.loc[keep].reset_index(drop=True), report


def verify_displacement_control(frame: pd.DataFrame, source: str) -> float:
    """Assert U2 = 20 t mm on every row and return the measured maximum deviation in mm.

    The campaign was run under displacement control with a 20 mm ramp over unit step time.
    A row that breaks the relation means the curve is not what the grid stage assumes, so
    the deviation is a hard error rather than a logged warning.
    """
    u2 = frame["U2"].to_numpy(dtype=float)
    step_time = frame["time"].to_numpy(dtype=float)
    deviation = np.abs(np.abs(u2) - DISPLACEMENT_RATE_MM * step_time)
    worst = float(deviation.max()) if deviation.size else 0.0
    if worst > U2_TOLERANCE_MM:
        offender = int(np.argmax(deviation))
        raise ValueError(
            f"{source} violates displacement control U2 = {DISPLACEMENT_RATE_MM} t mm: the "
            f"worst row deviates by {worst:.6g} mm, above the {U2_TOLERANCE_MM} mm "
            f"tolerance, at row {offender} (job {frame['job'].iloc[offender]!r}, "
            f"time {step_time[offender]:.6g}, U2 {u2[offender]:.6g} mm)."
        )
    return worst


def _read_signal(path: Path, dtypes: dict[str, str], role: str) -> pd.DataFrame:
    """Read one raw signal CSV with its dtypes enforced and its job column typed."""
    _require_file(path, role)
    frame = pd.read_csv(path, dtype={"job": "string", **dtypes})
    expected = ["job", *dtypes]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}; ingest expects exactly {expected} and "
            f"the file offers {list(frame.columns)}."
        )
    frame = frame[expected].copy()
    n_nan = int(frame[list(dtypes)].isna().to_numpy().sum())
    if n_nan:
        raise ValueError(
            f"{path} carries {n_nan} non finite values in {list(dtypes)}. The audit measured "
            "zero NaN across all 198 jobs, so any is a change in the inherited data and "
            "must be diagnosed rather than dropped."
        )
    frame["sample_id"] = job_to_sample_id(frame["job"])
    return frame.sort_values(["job", "time"], kind="mergesort").reset_index(drop=True)


def read_design(path: Path) -> pd.DataFrame:
    """Read and validate the 400 row LHS design.

    Checks the column set, the row count, and that the three independent inputs are as
    close to uncorrelated as the audit measured. Build spec 6.1 pins |r| <= 0.05.
    """
    _require_file(path, "LHS design CSV")
    frame = pd.read_csv(path)
    missing = [name for name in DESIGN_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing design columns {missing}; ingest expects "
            f"{list(DESIGN_COLUMNS)} and the file offers {list(frame.columns)}."
        )
    frame = frame[list(DESIGN_COLUMNS)].copy()
    if len(frame) != EXPECTED_DESIGN_ROWS:
        raise ValueError(
            f"{path} holds {len(frame)} rows, expected the {EXPECTED_DESIGN_ROWS} row LHS "
            "design of build spec 6.1. A different design is a different campaign."
        )
    duplicated = int(frame["sample_id"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{path} repeats {duplicated} sample_id values. Every design row must be a "
            "distinct sample; colliding ids are the v1 augmentation defect (build spec 5.7)."
        )
    frame = frame.astype(
        {
            "sample_id": "int64",
            "Fcm_MPa": "float64",
            "c_nom_bottom_mm": "float64",
            "c_nom_top_mm": "float64",
            "E_MPa": "float64",
            "seed": "int64",
        }
    )
    worst = verify_design_independence(frame)
    frame.attrs["max_abs_cross_correlation"] = worst
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def verify_design_independence(frame: pd.DataFrame) -> float:
    """Return the largest absolute off diagonal Pearson correlation of the three inputs.

    E_MPa is deliberately excluded: it is the exact Eurocode 2 function of Fcm (build spec
    5.4), so correlating it with Fcm measures the derivation, not the design.
    """
    matrix = frame[list(FEATURE_ORDER)].corr(method="pearson").to_numpy(dtype=float)
    off_diagonal = matrix[~np.eye(len(FEATURE_ORDER), dtype=bool)]
    worst = float(np.abs(off_diagonal).max())
    if worst > DESIGN_MAX_ABS_CORRELATION:
        raise ValueError(
            f"the LHS design's largest absolute cross correlation between "
            f"{list(FEATURE_ORDER)} is {worst:.4f}, above the {DESIGN_MAX_ABS_CORRELATION} "
            "bound of build spec 6.1. Correlated inputs invalidate the Saltelli estimators "
            "of the sensitivity stage, so this is a stop condition, not a note."
        )
    return worst


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write one Parquet with zstd compression, the pinned artifact format."""
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def declared_input_hashes(
    repo_root: Path | str, config: Config, config_sha256: str
) -> dict[str, str]:
    """Hash this stage's declared inputs as they are on disk right now (see ``ufem.runner``).

    The head of the pipeline reads the inherited CSVs rather than another stage's artifacts,
    so ``config_sha256`` plays no part here; it is in the signature because the runner calls
    every stage's declaration the same way.
    """
    del config_sha256
    paths = raw_paths(Path(repo_root), config)
    return {name: sha256_file(_require_file(path, name)) for name, path in paths.items()}


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the ingest stage and return its artifact directory.

    Reads the two raw signal CSVs and the LHS design, enforces the contracts of build spec
    9.3, and writes three Parquet files plus ``manifest.json`` into
    ``<artifact_root>/ingest/<config hash>``.
    """
    started = _time.perf_counter()
    root = Path(repo_root)
    paths = raw_paths(root, config)
    raw_hashes = declared_input_hashes(root, config, config_sha256)

    load = _read_signal(paths["load_displacement_csv"], LOAD_DTYPES, "load displacement CSV")
    damage = _read_signal(paths["damage_evolution_csv"], DAMAGE_DTYPES, "damage evolution CSV")
    design = read_design(paths["design_csv"])

    rows_in = {"load_displacement": len(load), "damage_evolution": len(damage)}

    load_clean, load_report = deduplicate_by_time(load)
    damage_clean, damage_report = deduplicate_by_time(damage)

    dedup_jobs = int((load_report["rows_removed"] > 0).sum())
    dedup_rows = int(load_report["rows_removed"].sum())
    damage_dedup_jobs = int((damage_report["rows_removed"] > 0).sum())
    damage_dedup_rows = int(damage_report["rows_removed"].sum())

    if dedup_jobs != EXPECTED_DEDUP_JOBS or dedup_rows != EXPECTED_DEDUP_ROWS:
        raise AssertionError(
            f"the load displacement table needed deduplication on {dedup_jobs} jobs removing "
            f"{dedup_rows} rows, but build spec 6.1 and the 2026-08-28 audit pin "
            f"{EXPECTED_DEDUP_JOBS} jobs and {EXPECTED_DEDUP_ROWS} rows. The inherited data "
            "has changed. Per build spec section 24, stop and record the measured values in "
            "docs/DESIGN_DECISIONS.md with a date before relaxing this assertion."
        )

    worst_load = verify_displacement_control(load_clean, "load_displacement_full.csv")
    worst_damage = verify_displacement_control(damage_clean, "damage_evolution_full.csv")

    load_jobs = set(load_clean["job"].unique())
    damage_jobs = set(damage_clean["job"].unique())
    if len(load_jobs) != EXPECTED_JOBS or len(damage_jobs) != EXPECTED_JOBS:
        raise AssertionError(
            f"expected {EXPECTED_JOBS} unique jobs in both signals (build spec 6.1), "
            f"measured {len(load_jobs)} in load displacement and {len(damage_jobs)} in "
            "damage evolution."
        )
    if load_jobs != damage_jobs:
        only_load = sorted(load_jobs - damage_jobs)
        only_damage = sorted(damage_jobs - load_jobs)
        raise AssertionError(
            "the two signals carry different job sets, which the audit measured as "
            f"identical: {len(only_load)} only in load displacement ({only_load[:5]}) and "
            f"{len(only_damage)} only in damage evolution ({only_damage[:5]})."
        )
    unknown = sorted(set(load_clean["sample_id"].unique()) - set(design["sample_id"]))
    if unknown:
        raise AssertionError(
            f"{len(unknown)} extracted jobs are absent from the LHS design, for example "
            f"{unknown[:5]}. Every simulated job must trace back to a design row (binding "
            "law 5); ids outside the design are the v1 augmentation collision of spec 5.7."
        )

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        _write_parquet(load_clean, directory / LOAD_PARQUET),
        _write_parquet(damage_clean, directory / DAMAGE_PARQUET),
        _write_parquet(design, directory / DESIGN_PARQUET),
    ]

    extra = {
        "cache_key": cache_key(
            STAGE_NAME, Path(__file__), config_sha256, raw_hashes
        ),
        "wall_time_s": _time.perf_counter() - started,
        "rows_in": rows_in,
        "rows_out": {
            "load_displacement": len(load_clean),
            "damage_evolution": len(damage_clean),
        },
        "rows_dropped": {
            "load_displacement": dedup_rows,
            "damage_evolution": damage_dedup_rows,
        },
        "jobs_deduplicated": {
            "load_displacement": dedup_jobs,
            "damage_evolution": damage_dedup_jobs,
        },
        "n_jobs": len(load_jobs),
        "n_design_rows": len(design),
        "max_abs_U2_minus_20t_mm": {
            "load_displacement": worst_load,
            "damage_evolution": worst_damage,
        },
        "design_max_abs_cross_correlation": float(
            design.attrs["max_abs_cross_correlation"]
        ),
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=raw_hashes,
        outputs=outputs,
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    print(
        f"[ingest] {rows_in['load_displacement']} load rows in, {dedup_rows} dropped over "
        f"{dedup_jobs} jobs; {rows_in['damage_evolution']} damage rows in, "
        f"{damage_dedup_rows} dropped over {damage_dedup_jobs} jobs; "
        f"{len(load_jobs)} jobs, {len(design)} design rows; "
        f"max |U2 - 20t| = {max(worst_load, worst_damage):.3e} mm"
    )
    return directory
