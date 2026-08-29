"""Stage ``grid``: both signals onto the common displacement grid, plus the scalar QoIs.

Build spec sections 9.3 and 9.5. Reads the ingest stage's Parquet artifacts, interpolates
RF2 and DAMAGEC_max onto ``u = linspace(0, 20, 201)`` mm with :func:`numpy.interp`, and
extracts the scalar quantities of interest that the surrogate will predict.

Interpolation is NumPy, not torch, deliberately: torch's linear interpolation raises under
deterministic mode (build spec 9.3). Each curve's abscissa is its own U2 column, sorted and
filtered to strictly increasing before interpolation, which is exactly what the 2026-08-28
audit did and what the golden gate of ``tests/test_golden_audit.py`` compares against.

Units: displacement u and U2 in mm, force RF2 and P in N, absorbed energy in N mm, initial
stiffness in N/mm, strength Fcm in MPa, covers in mm. DAMAGEC_max and the softening ratio
are dimensionless.

Terminal damage is deliberately absent from the QoI table. It saturates at the concrete
damaged plasticity table cap of 0.947 for every completed run, so it has zero variance and
is useless as a target (build spec 5.6); the ban is enforced by a test.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from ufem.config import FEATURE_ORDER, Config
from ufem.ingest import (
    DAMAGE_PARQUET,
    DESIGN_PARQUET,
    EXPECTED_JOBS,
    LOAD_PARQUET,
)
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest

STAGE_NAME = "grid"

#: Output file names inside the stage directory.
RF2_GRID_PARQUET = "rf2_grid.parquet"
DAMAGE_GRID_PARQUET = "damage_grid.parquet"
QOI_PARQUET = "qoi.parquet"

#: Build spec 5.6: the compression damage table cap every completed run reaches.
DAMAGE_SATURATION = 0.9470000267028807

#: Build spec 9.5: the initial stiffness window is the first tenth of the displacement at
#: peak, fitted by least squares through the origin.
STIFFNESS_WINDOW_FRACTION = 0.10

#: Build spec 9.5: damage is reported at this fixed displacement in mm, where it still
#: carries real variance (CoV 0.16), unlike its saturated terminal value.
DAMAGE_PROBE_MM = 10.0

#: The exact column set of ``qoi.parquet``. Pinned here so a test can assert both that
#: nothing is missing and that terminal damage never appears.
QOI_COLUMNS: tuple[str, ...] = (
    "job",
    "sample_id",
    *FEATURE_ORDER,
    "P_max_N",
    "u_peak_mm",
    "k0_N_per_mm",
    "E_abs_Nmm",
    "P_residual_N",
    "softening_ratio",
    "u_damage_half_sat_mm",
    "damage_at_10mm",
)

#: Column names banned from the QoI table, with the reason each is banned.
BANNED_QOI_COLUMNS: dict[str, str] = {
    "damage_final": "terminal damage saturates at the CDP table cap (build spec 5.6)",
    "damage_terminal": "terminal damage saturates at the CDP table cap (build spec 5.6)",
    "terminal_damage": "terminal damage saturates at the CDP table cap (build spec 5.6)",
    "damage_max": "the per curve damage maximum is the same saturated cap for every run",
    "E_MPa": "E is derived from Fcm and is never a feature (build spec 9.2)",
}


def displacement_grid(config: Config) -> np.ndarray:
    """The common displacement grid in mm, from the pipeline config."""
    settings = config.pipeline.grid
    return np.linspace(settings.u_min_mm, settings.u_max_mm, settings.n_points)


def monotone_curve(abscissa: np.ndarray, ordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by abscissa and keep the first point of each strictly increasing run.

    ``np.interp`` requires an increasing abscissa and silently returns nonsense otherwise,
    so this is applied to every curve before interpolation. The sort is stable, so among
    tied abscissa values the earlier row in solver order survives.
    """
    x = np.asarray(abscissa, dtype=float)
    y = np.asarray(ordinate, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"monotone_curve needs matching shapes, got abscissa {x.shape} and "
            f"ordinate {y.shape}."
        )
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    if x.size == 0:
        return x, y
    keep = np.concatenate(([True], np.diff(x) > 0.0))
    return x[keep], y[keep]


def interpolate_onto_grid(
    frame: pd.DataFrame, value_column: str, grid_mm: np.ndarray, source: str
) -> tuple[list[str], np.ndarray]:
    """Interpolate one signal onto the common grid, one row per job.

    Returns the job labels in sorted order and the ``(n_jobs, n_grid)`` matrix. Each curve
    must span the grid: an extrapolated tail would be an invented number, so a curve that
    stops short raises rather than being flat filled the way ``np.interp`` would.
    """
    jobs = sorted(frame["job"].unique())
    matrix = np.empty((len(jobs), grid_mm.size), dtype=float)
    grouped = {job: group for job, group in frame.groupby("job", sort=False)}
    for row, job in enumerate(jobs):
        group = grouped[job]
        x, y = monotone_curve(
            np.abs(group["U2"].to_numpy(dtype=float)), group[value_column].to_numpy(dtype=float)
        )
        if x.size < 2:
            raise ValueError(
                f"{source} job {job!r} has {x.size} distinct displacement values, too few "
                "to interpolate. Ingest should have rejected a curve this short."
            )
        if x[0] > grid_mm[0] + 1e-9 or x[-1] < grid_mm[-1] - 1e-9:
            raise ValueError(
                f"{source} job {job!r} spans [{x[0]:.6g}, {x[-1]:.6g}] mm, which does not "
                f"cover the common grid [{grid_mm[0]:.6g}, {grid_mm[-1]:.6g}] mm. "
                "Interpolating past the data would invent the tail, so this is a stop "
                "condition rather than an extrapolation."
            )
        matrix[row] = np.interp(grid_mm, x, y)
    n_nan = int(np.isnan(matrix).sum())
    if n_nan:
        raise ValueError(
            f"{source} produced {n_nan} NaN on the common grid. The audit measured zero "
            "across all 198 curves; a NaN here is a defect, not a missing value."
        )
    return jobs, matrix


def initial_stiffness(u_mm: np.ndarray, force_N: np.ndarray, u_peak_mm: float) -> float:
    """Least squares slope through the origin over ``0 < u <= 0.1 * u_peak``, in N/mm.

    The estimator is ``sum(u * P) / sum(u * u)``, which is the exact least squares solution
    of ``P = k0 * u`` with no intercept. Build spec 9.5 fixes the window; a window holding
    fewer than two points means the curve is too coarse to carry a stiffness and raises.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    if not u_peak_mm > 0.0:
        raise ValueError(
            f"initial_stiffness needs a positive displacement at peak, got {u_peak_mm}."
        )
    window = (u > 0.0) & (u <= STIFFNESS_WINDOW_FRACTION * u_peak_mm)
    if int(window.sum()) < 2:
        raise ValueError(
            f"the initial stiffness window u in (0, {STIFFNESS_WINDOW_FRACTION * u_peak_mm:.4g}] "
            f"mm holds {int(window.sum())} points, fewer than the two a through origin fit "
            "needs. The v1 audit fell back to the first three points here; falling back "
            "silently is what ground rule 8 forbids."
        )
    u_w, f_w = u[window], force[window]
    denominator = float(np.dot(u_w, u_w))
    if denominator <= 0.0:
        raise ValueError("the initial stiffness window has zero displacement spread.")
    return float(np.dot(u_w, f_w) / denominator)


def damage_half_saturation(
    u_mm: np.ndarray, damage: np.ndarray, saturation: float = DAMAGE_SATURATION
) -> float:
    """Displacement in mm where damage first reaches half its saturation value.

    Build spec 9.5 and 5.6: the terminal value is the same table cap for every run, so the
    informative scalar is where the curve gets to half of it. Linear interpolation between
    the bracketing samples, so the answer does not quantize onto the solver's increments.
    """
    u = np.asarray(u_mm, dtype=float)
    d = np.asarray(damage, dtype=float)
    threshold = 0.5 * saturation
    reached = np.flatnonzero(d >= threshold)
    if reached.size == 0:
        raise ValueError(
            f"damage never reaches half saturation ({threshold:.6g}); the curve tops out at "
            f"{float(d.max()) if d.size else float('nan'):.6g}. Every completed run in the "
            "inherited campaign saturates, so this is a change in the data."
        )
    index = int(reached[0])
    if index == 0:
        return float(u[0])
    d_low, d_high = d[index - 1], d[index]
    if d_high == d_low:
        return float(u[index])
    fraction = (threshold - d_low) / (d_high - d_low)
    return float(u[index - 1] + fraction * (u[index] - u[index - 1]))


def extract_qoi(
    u_mm: np.ndarray, force_N: np.ndarray, damage: np.ndarray
) -> dict[str, float]:
    """The scalar QoI schedule of build spec 9.5 for one curve on the common grid.

    ``u_mm`` is the common displacement grid in mm, ``force_N`` the interpolated reaction in
    N, ``damage`` the interpolated compression damage scalar on the same grid.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    peak_index = int(np.argmax(force))
    p_max = float(force[peak_index])
    u_peak = float(u[peak_index])
    p_residual = float(force[-1])
    if p_max <= 0.0:
        raise ValueError(f"peak load is {p_max} N, which is not a loaded curve.")
    return {
        "P_max_N": p_max,
        "u_peak_mm": u_peak,
        "k0_N_per_mm": initial_stiffness(u, force, u_peak),
        "E_abs_Nmm": float(np.trapezoid(force, u)),
        "P_residual_N": p_residual,
        "softening_ratio": p_residual / p_max,
        "u_damage_half_sat_mm": damage_half_saturation(u, damage),
        "damage_at_10mm": float(np.interp(DAMAGE_PROBE_MM, u, damage)),
    }


def raw_curve_qoi(load: pd.DataFrame) -> pd.DataFrame:
    """Peak, displacement at peak and initial stiffness measured on the raw increments.

    This is the basis the 2026-08-28 audit used for its headline statistics, and it is not
    the same basis as the QoI table: the audit read the peak off the solver's own adaptive
    increments, while :func:`extract_qoi` reads it off the 201 point common grid. A resample
    can only lower a maximum, so the two differ by about 20 N in the mean, and the initial
    stiffness differs far more because the audit's window held thousands of raw points where
    the grid offers two. Both are correct for what they measure; the golden gate of
    ``tests/test_golden_audit.py`` compares each against its own basis, and the reasoning is
    recorded in docs/DESIGN_DECISIONS.md.

    Returns one row per job with ``P_max_N`` [N], ``u_peak_mm`` [mm] and ``k0_N_per_mm``
    [N/mm], sorted by job label.
    """
    records = []
    for job, group in load.groupby("job", sort=True):
        u = np.abs(group["U2"].to_numpy(dtype=float))
        force = group["RF2"].to_numpy(dtype=float)
        peak_index = int(np.argmax(force))
        u_peak = float(u[peak_index])
        records.append(
            {
                "job": str(job),
                "P_max_N": float(force[peak_index]),
                "u_peak_mm": u_peak,
                "k0_N_per_mm": initial_stiffness(u, force, u_peak),
            }
        )
    return pd.DataFrame.from_records(records)


def headline_stats(values: np.ndarray) -> dict[str, float]:
    """Mean, sample standard deviation, CoV, min and max of one QoI over the campaign.

    The sample standard deviation uses ``ddof=1``, matching the audit and every reported
    CoV in build spec 6.1.
    """
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError(
            f"headline_stats needs at least two values for a sample standard deviation, "
            f"got {array.size}."
        )
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    if mean == 0.0:
        raise ValueError("headline_stats cannot form a CoV around a zero mean.")
    return {
        "mean": mean,
        "std": std,
        "cov": std / mean,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _matrix_frame(jobs: list[str], matrix: np.ndarray, grid_mm: np.ndarray) -> pd.DataFrame:
    """One row per job: the job label then one column per grid point, named ``u_<mm>``."""
    columns = [f"u_{value:.2f}" for value in grid_mm]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "job", pd.Series(jobs, dtype="string"))
    return frame


def _load_ingest(root: Path, config: Config, config_sha256: str) -> tuple[Path, dict[str, str]]:
    """Locate the ingest artifacts this stage depends on, or raise naming the fix."""
    directory = stage_dir(
        root / config.pipeline.paths.artifact_root, INGEST_STAGE, config_sha256
    )
    hashes = {}
    for name in (LOAD_PARQUET, DAMAGE_PARQUET, DESIGN_PARQUET):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the grid stage needs {path}, which does not exist. Run `ufem run ingest` "
                "first: grid depends on the ingest artifacts for this config hash."
            )
        hashes[name] = sha256_file(path)
    return directory, hashes


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the grid stage and return its artifact directory."""
    started = _time.perf_counter()
    root = Path(repo_root)
    ingest_dir, input_hashes = _load_ingest(root, config, config_sha256)

    load = pd.read_parquet(ingest_dir / LOAD_PARQUET)
    damage = pd.read_parquet(ingest_dir / DAMAGE_PARQUET)
    design = pd.read_parquet(ingest_dir / DESIGN_PARQUET)

    grid_mm = displacement_grid(config)
    load_jobs, rf2 = interpolate_onto_grid(load, "RF2", grid_mm, "load displacement")
    damage_jobs, damage_matrix = interpolate_onto_grid(
        damage, "DAMAGEC_max", grid_mm, "damage evolution"
    )
    if load_jobs != damage_jobs:
        raise AssertionError(
            "the two gridded signals carry different job orders, which makes any row wise "
            "join between them wrong."
        )
    if len(load_jobs) != EXPECTED_JOBS:
        raise AssertionError(
            f"the grid stage produced {len(load_jobs)} curves, expected {EXPECTED_JOBS} "
            "(build spec 6.1)."
        )

    design_by_id = design.set_index("sample_id")
    records = []
    for row, job in enumerate(load_jobs):
        sample_id = int(str(job).rsplit("_", 1)[-1])
        if sample_id not in design_by_id.index:
            raise KeyError(
                f"job {job!r} maps to sample_id {sample_id}, which is absent from the LHS "
                "design. Ingest asserts this cannot happen, so reaching it means the "
                "artifacts are inconsistent."
            )
        inputs = design_by_id.loc[sample_id]
        record = {"job": str(job), "sample_id": sample_id}
        record.update({name: float(inputs[name]) for name in FEATURE_ORDER})
        record.update(extract_qoi(grid_mm, rf2[row], damage_matrix[row]))
        records.append(record)
    qoi = pd.DataFrame.from_records(records, columns=list(QOI_COLUMNS))
    qoi["job"] = qoi["job"].astype("string")

    banned = sorted(set(qoi.columns) & set(BANNED_QOI_COLUMNS))
    if banned:
        raise AssertionError(
            "the QoI table carries banned columns "
            + "; ".join(f"{name} ({BANNED_QOI_COLUMNS[name]})" for name in banned)
        )
    n_nan = int(qoi.drop(columns=["job"]).isna().to_numpy().sum())
    if n_nan:
        raise ValueError(
            f"the QoI table holds {n_nan} NaN. Build spec 9.3 gates this stage on zero NaN "
            "anywhere, so a missing scalar is a stage failure."
        )

    rf2_frame = _matrix_frame(load_jobs, rf2, grid_mm)
    damage_frame = _matrix_frame(damage_jobs, damage_matrix, grid_mm)

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for frame, name in (
        (rf2_frame, RF2_GRID_PARQUET),
        (damage_frame, DAMAGE_GRID_PARQUET),
        (qoi, QOI_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    extra = {
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "n_curves": len(load_jobs),
        "n_grid_points": int(grid_mm.size),
        "grid_u_min_mm": float(grid_mm[0]),
        "grid_u_max_mm": float(grid_mm[-1]),
        "n_nan_rf2_grid": int(np.isnan(rf2).sum()),
        "n_nan_damage_grid": int(np.isnan(damage_matrix).sum()),
        "n_nan_qoi": n_nan,
        "peak_load_mean_N": float(qoi["P_max_N"].mean()),
        "peak_load_cov": float(qoi["P_max_N"].std(ddof=1) / qoi["P_max_N"].mean()),
        "u_peak_mean_mm": float(qoi["u_peak_mm"].mean()),
        "k0_mean_N_per_mm": float(qoi["k0_N_per_mm"].mean()),
        "residual_load_cov": float(
            qoi["P_residual_N"].std(ddof=1) / qoi["P_residual_N"].mean()
        ),
    }
    write_manifest(
        stage_dir=directory,
        stage_name=STAGE_NAME,
        config_hash=config_sha256,
        input_hashes=input_hashes,
        outputs=outputs,
        seed_entropy=config.pipeline.seed_entropy,
        extra=extra,
    )
    print(
        f"[grid] {len(load_jobs)} curves on {grid_mm.size} points over "
        f"[{grid_mm[0]:.1f}, {grid_mm[-1]:.1f}] mm, zero NaN; peak load mean "
        f"{extra['peak_load_mean_N'] / 1000.0:.2f} kN, CoV {extra['peak_load_cov']:.4f}"
    )
    return directory
