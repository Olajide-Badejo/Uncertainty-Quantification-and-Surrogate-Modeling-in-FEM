"""Stage ``register``: landmarks, arc length reparameterization, elastic SRVF registration.

Build spec 10.1, the methodological core of the project. Softening curves vary in both
amplitude and phase: the displacement at peak has a coefficient of variation of 0.176 over
this family, so the peak of one run sits where another run is still hardening. Principal
component analysis has no way to express "the same shape, shifted", so applied to
unregistered curves it manufactures derivative shaped modes, needs more components, and
smears the peak of the reconstructed mean. Separating amplitude from phase before reduction
is what this stage exists to do, and the ablation of build spec 10.6.1 is what proves it
mattered rather than asserting it.

The four steps, in order:

1. **Landmarks** per curve: the first cracking knee, the peak, 85 percent of peak on the
   descending branch, and the end. These are quantities of interest in their own right and
   they are persisted as a table for later phases.
2. **Arc length reparameterization** with the FIXED global normalizers P0 and u0 from
   ``configs/pipeline.yaml``, never per curve normalizers. A per curve normalizer would
   divide out exactly the amplitude information the surrogate has to predict, which is why
   build spec 10.1 forbids it and why a test asserts the normalizers are scalars read from
   config.
3. **Elastic registration** via the square root slope framework, ``fdasrsf.fdawarp``, giving
   registered amplitude functions and warping functions gamma.
4. **Phase representation**: the warps mapped through psi = sqrt(gamma dot) to the tangent
   space at the Karcher mean, because gamma lives on a constrained monotone space where a
   linear method like PCA does not belong.

Damage curves are deliberately NOT registered. They are near degenerate monotone saturating
curves on the same grid, every one of them ending at the same table cap, so there is no phase
variation worth separating and build spec 10.2 expects them to be very low rank raw. They
pass through to the reduction stage untouched, and that pass through is a stated choice
rather than an oversight.

Units: displacement u and all landmark abscissae in mm, force P and all landmark ordinates
in N. Arc length s, the registered amplitude functions, the warping functions gamma, and the
tangent vectors are dimensionless. The normalizers P0 [N] and u0 [mm] carry the units out.

RNG discipline (build spec 17.3): this stage is deterministic and draws no random numbers.
The seed entropy is still recorded in its manifest so the artifact chain is uniform.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from ufem.config import Config
from ufem.grid import DAMAGE_GRID_PARQUET, RF2_GRID_PARQUET, displacement_grid
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.manifest import cache_key, sha256_file, stage_dir, write_manifest

STAGE_NAME = "register"

#: Output file names inside the stage directory.
LANDMARKS_PARQUET = "landmarks.parquet"
AMPLITUDE_PARQUET = "amplitude_registered.parquet"
AMPLITUDE_UNREGISTERED_PARQUET = "amplitude_unregistered.parquet"
WARP_PARQUET = "warp_gamma.parquet"
WARP_TANGENT_PARQUET = "warp_tangent.parquet"
ARCLENGTH_JSON = "arclength.json"

#: The common arc length grid, matched to the displacement grid's resolution so the
#: registered family carries the same number of samples as the input family.
N_ARCLENGTH_POINTS = 201

#: Build spec 10.1: the knee is sought on the pre peak segment. The upper bound keeps the
#: estimator away from the rounded peak, whose curvature would otherwise win and turn this
#: landmark into a second, worse measurement of the peak.
#:
#: The lower bound is 0.2 mm rather than the 0.5 mm I first wrote, and the reason is worth
#: recording because the first choice produced a landmark that looked plausible and was
#: wrong. These curves are exactly linear over the first few grid points (the reaction rises
#: by 1815.2, 1815.0, 1814.7, 1814.5 N per 0.1 mm step on the first curve, constant to four
#: digits) and then break sharply between 0.6 and 0.8 mm. The first cracking knee of this
#: beam is genuinely that early. A 0.5 mm bound cut into the knee instead of excluding noise,
#: so the curvature minimum landed on the window edge for 135 of 198 curves and the landmark
#: was measuring my window rather than the data. 0.2 mm still excludes the first two grid
#: points, where a second difference of an interpolated curve has nothing real to say, while
#: leaving the actual break well inside the window.
KNEE_WINDOW_MIN_MM = 0.2
KNEE_WINDOW_PEAK_FRACTION = 0.8

#: Half width, in grid points, of the moving average applied before the second difference.
#: Three points either side over a 0.1 mm grid smooths across 0.6 mm, enough to suppress the
#: point to point noise of the interpolation without flattening a real knee.
KNEE_SMOOTH_HALF_WIDTH = 3

#: Build spec 10.1: the post peak landmark is where the curve falls to this fraction of peak.
POST_PEAK_LEVEL = 0.85

#: Columns of ``landmarks.parquet``, pinned so a test can assert the schema.
LANDMARK_COLUMNS: tuple[str, ...] = (
    "job",
    "u_knee_mm",
    "P_knee_N",
    "u_peak_mm",
    "P_peak_N",
    "u_85_mm",
    "P_85_N",
    "u85_reached",
    "u_end_mm",
    "P_end_N",
    "arclength_total",
)


def _value_columns(frame: pd.DataFrame) -> list[str]:
    """The grid columns of a gridded artifact, in stored order."""
    return [name for name in frame.columns if name != "job"]


def curve_matrix(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Split a gridded artifact into its job labels and its ``(n_jobs, n_grid)`` matrix."""
    columns = _value_columns(frame)
    if not columns:
        raise ValueError(
            "a gridded artifact must carry at least one value column beside 'job'; this "
            "frame carries none, so it is not the artifact the grid stage writes."
        )
    return [str(job) for job in frame["job"]], frame[columns].to_numpy(dtype=float)


def smoothed_second_difference(values: np.ndarray, half_width: int) -> np.ndarray:
    """Second difference of a moving averaged signal, same length as the input.

    The estimator of build spec 10.1's cracking knee. A moving average of half width
    ``half_width`` runs first, because the raw second difference of an interpolated curve is
    dominated by the interpolation's own point to point noise; the second difference then
    runs on the smoothed signal via :func:`numpy.gradient` applied twice, which keeps the
    result the same length as the input rather than shortening it by two and silently
    shifting every index the caller then uses.

    Endpoints of the moving average are handled by shrinking the window rather than by
    padding, so no value outside the data is invented.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"smoothed_second_difference needs a 1D signal, got {array.shape}.")
    if half_width < 0:
        raise ValueError(f"half_width must be non negative, got {half_width}.")
    n = array.size
    if n < 3:
        raise ValueError(
            f"a second difference needs at least three samples, got {n}."
        )
    smoothed = np.empty(n, dtype=float)
    for index in range(n):
        low = max(0, index - half_width)
        high = min(n, index + half_width + 1)
        smoothed[index] = array[low:high].mean()
    return np.gradient(np.gradient(smoothed))


def cracking_knee(
    u_mm: np.ndarray,
    force_N: np.ndarray,
    u_peak_mm: float,
    half_width: int = KNEE_SMOOTH_HALF_WIDTH,
) -> tuple[float, float]:
    """First cracking knee: maximum curvature of the pre peak segment, in (mm, N).

    Build spec 10.1 defines the landmark as the maximum curvature of the pre peak segment.
    The implementation takes the most negative smoothed second difference over the window
    ``KNEE_WINDOW_MIN_MM < u < KNEE_WINDOW_PEAK_FRACTION * u_peak``, which is the point of
    sharpest downward bending, i.e. where the curve turns over from the elastic line as
    cracking sets in. Most negative rather than largest absolute: an upward bend is not a
    cracking knee, and taking the absolute value would let one be selected.

    The window is the documented part of the estimator. Below 0.5 mm the curve is a straight
    elastic segment whose numerical curvature is interpolation noise; above 0.8 of the
    displacement at peak the rounded maximum itself has the largest curvature on the curve
    and would win every time, which would make the landmark a second, worse measurement of
    the peak rather than a measurement of the knee.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    if u.shape != force.shape:
        raise ValueError(
            f"cracking_knee needs matching shapes, got u {u.shape} and force {force.shape}."
        )
    if not u_peak_mm > KNEE_WINDOW_MIN_MM:
        raise ValueError(
            f"cracking_knee needs a displacement at peak above {KNEE_WINDOW_MIN_MM} mm to "
            f"have a pre peak window at all, got {u_peak_mm} mm."
        )
    window = (u > KNEE_WINDOW_MIN_MM) & (u < KNEE_WINDOW_PEAK_FRACTION * u_peak_mm)
    if int(window.sum()) < 3:
        raise ValueError(
            f"the knee window u in ({KNEE_WINDOW_MIN_MM}, "
            f"{KNEE_WINDOW_PEAK_FRACTION * u_peak_mm:.4g}) mm holds {int(window.sum())} "
            "points, too few for a second difference. Falling back to an endpoint here "
            "would be the silent fallback of ground rule 8."
        )
    curvature = smoothed_second_difference(force, half_width)
    masked = np.where(window, curvature, np.inf)
    index = int(np.argmin(masked))
    return float(u[index]), float(force[index])


def post_peak_level_crossing(
    u_mm: np.ndarray, force_N: np.ndarray, peak_index: int, level: float = POST_PEAK_LEVEL
) -> tuple[float, float, bool]:
    """Where the descending branch first falls to ``level`` times the peak, in (mm, N, flag).

    Returns the interpolated crossing and ``True``, or ``(nan, nan, False)`` when the curve
    never gets there within the stroke.

    That second case is real: 1 of the 198 curves in this campaign never falls to 85 percent
    of its peak anywhere on its descending branch. The honest representation of "this curve
    never softened that far" is a missing landmark with a flag saying so, not the last grid
    point dressed up as a crossing, which would put a fictitious landmark at u = 20 mm into
    any distribution built on the column. That is the kind of manufactured number ground
    rule 8 and binding law 1 exist to prevent. Downstream consumers read ``u85_reached`` and
    decide; nothing here decides for them.

    The scan covers the whole descending branch rather than only the final value, and the
    difference is not cosmetic. 34 curves end above 85 percent of their peak at u = 20 mm,
    but 33 of those dip below it earlier and then recover load, so a rule that tested the end
    point alone would report 34 missing landmarks where only 1 is genuinely missing and would
    discard 33 real crossings.

    Linear interpolation between the bracketing samples, so the answer does not quantize onto
    the 0.1 mm grid.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    if not 0.0 < level < 1.0:
        raise ValueError(f"the post peak level must lie in (0, 1), got {level}.")
    if not 0 <= peak_index < u.size:
        raise ValueError(
            f"peak_index {peak_index} is outside the curve of {u.size} samples."
        )
    threshold = level * float(force[peak_index])
    tail = force[peak_index:]
    below = np.flatnonzero(tail <= threshold)
    if below.size == 0:
        return float("nan"), float("nan"), False
    index = peak_index + int(below[0])
    if index == peak_index:
        return float(u[index]), float(force[index]), True
    f_high, f_low = force[index - 1], force[index]
    if f_high == f_low:
        return float(u[index]), float(force[index]), True
    fraction = (f_high - threshold) / (f_high - f_low)
    return (
        float(u[index - 1] + fraction * (u[index] - u[index - 1])),
        float(threshold),
        True,
    )


def extract_landmarks(u_mm: np.ndarray, force_N: np.ndarray) -> dict[str, float]:
    """The four landmarks of build spec 10.1 for one curve on the common grid.

    ``u_mm`` is the common displacement grid in mm and ``force_N`` the interpolated reaction
    in N. Returns the knee, the peak, the 85 percent post peak crossing with its reached
    flag, and the end.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    peak_index = int(np.argmax(force))
    peak_force = float(force[peak_index])
    if peak_force <= 0.0:
        raise ValueError(f"peak load is {peak_force} N, which is not a loaded curve.")
    u_peak = float(u[peak_index])
    u_knee, p_knee = cracking_knee(u, force, u_peak)
    u_85, p_85, reached = post_peak_level_crossing(u, force, peak_index)
    return {
        "u_knee_mm": u_knee,
        "P_knee_N": p_knee,
        "u_peak_mm": u_peak,
        "P_peak_N": peak_force,
        "u_85_mm": u_85,
        "P_85_N": p_85,
        "u85_reached": reached,
        "u_end_mm": float(u[-1]),
        "P_end_N": float(force[-1]),
    }


def normalized_arclength(
    u_mm: np.ndarray, force_N: np.ndarray, u0_mm: float, P0_N: float
) -> np.ndarray:
    """Cumulative arc length of one curve in the normalized (u/u0, P/P0) plane.

    Build spec 10.1 step 2. The normalizers are the FIXED global constants from
    ``configs/pipeline.yaml``, identical for every curve in the family. That is the whole
    point: dividing each curve by its own peak would make every curve reach 1.0 and destroy
    the amplitude information the surrogate exists to predict, so per curve normalizers are
    forbidden and a test asserts these two are scalars taken from config.

    Returned unnormalized in the sense that the total is the curve's own path length; the
    caller rescales to [0, 1] for the common grid and keeps the total as a scalar feature.
    """
    u = np.asarray(u_mm, dtype=float)
    force = np.asarray(force_N, dtype=float)
    if not np.isscalar(u0_mm) or not np.isscalar(P0_N):
        raise TypeError(
            f"the arc length normalizers must be scalars from config, got u0 {type(u0_mm)} "
            f"and P0 {type(P0_N)}. A per curve normalizer is forbidden by build spec 10.1."
        )
    if not u0_mm > 0.0 or not P0_N > 0.0:
        raise ValueError(f"the normalizers must be positive, got u0 {u0_mm} and P0 {P0_N}.")
    du = np.diff(u / float(u0_mm))
    dp = np.diff(force / float(P0_N))
    steps = np.sqrt(du**2 + dp**2)
    return np.concatenate(([0.0], np.cumsum(steps)))


def resample_on_arclength(
    values: np.ndarray, arclength: np.ndarray, n_points: int = N_ARCLENGTH_POINTS
) -> np.ndarray:
    """Resample one signal onto ``n_points`` equally spaced normalized arc length stations.

    The arc length must be strictly increasing after the first point, which it is for any
    curve that moves at all; a curve that stalls would produce a flat stretch and is a stop
    condition rather than something to interpolate through.
    """
    s = np.asarray(arclength, dtype=float)
    y = np.asarray(values, dtype=float)
    if s.shape != y.shape:
        raise ValueError(
            f"resample_on_arclength needs matching shapes, got arclength {s.shape} and "
            f"values {y.shape}."
        )
    total = float(s[-1])
    if not total > 0.0:
        raise ValueError(
            f"the curve has total arc length {total}, so it does not move and cannot be "
            "reparameterized by arc length."
        )
    if np.any(np.diff(s) <= 0.0):
        raise ValueError(
            "the cumulative arc length is not strictly increasing, which means the curve "
            "stalls somewhere. Interpolating across a stall would invent the stalled "
            "stretch rather than represent it."
        )
    return np.interp(np.linspace(0.0, 1.0, n_points), s / total, y)


def arclength_family(
    u_mm: np.ndarray, curves: np.ndarray, u0_mm: float, P0_N: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reparameterize a whole family by arc length onto the common station grid.

    Returns the ``(n_curves, N_ARCLENGTH_POINTS)`` matrix of force in N against normalized
    arc length, and the ``(n_curves,)`` vector of total arc lengths (dimensionless).
    """
    matrix = np.asarray(curves, dtype=float)
    resampled = np.empty((matrix.shape[0], N_ARCLENGTH_POINTS), dtype=float)
    totals = np.empty(matrix.shape[0], dtype=float)
    for row in range(matrix.shape[0]):
        s = normalized_arclength(u_mm, matrix[row], u0_mm, P0_N)
        totals[row] = float(s[-1])
        resampled[row] = resample_on_arclength(matrix[row], s)
    return resampled, totals


def srsf_register(
    amplitude: np.ndarray, stations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic SRVF registration of a curve family, returning (registered, gamma).

    Build spec 10.1 step 3. The exact call, documented because reproducibility depends on it:
    ``fdasrsf.fdawarp(f, t).srsf_align(parallel=False)`` where ``f`` is the ``(n_stations,
    n_curves)`` matrix that fdasrsf expects, transposed from this project's row per curve
    convention, and ``t`` is the station grid. Every other argument is left at the library's
    default, which is ``method='mean'`` (Karcher mean rather than median), ``omethod='DP2'``
    (dynamic programming), ``center=True``, ``MaxItr=20``, ``lam=0.0``, and ``thresh=0.01``.

    ``parallel=False`` is the one deliberate departure from the defaults and it is set for
    determinism: the parallel path splits the family across workers, and a reduction whose
    association order depends on scheduling is a reduction that does not reproduce bitwise.
    Build spec 17.2 requires the production path to be bitwise reproducible on this machine,
    which is worth more here than the wall time, and the measured cost is recorded in the
    engineering log.

    Returns both matrices back in row per curve orientation.
    """
    from fdasrsf import fdawarp

    f = np.asarray(amplitude, dtype=float)
    t = np.asarray(stations, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"srsf_register needs a 2D family, got shape {f.shape}.")
    if f.shape[1] != t.size:
        raise ValueError(
            f"the family carries {f.shape[1]} samples per curve but the station grid has "
            f"{t.size}. These must match before registration."
        )
    warp = fdawarp(f.T, t)
    warp.srsf_align(parallel=False)
    return np.ascontiguousarray(warp.fn.T), np.ascontiguousarray(warp.gam.T)


def warp_tangent_vectors(gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map warps to the tangent space at the Karcher mean, returning (vectors, mean psi).

    Build spec 10.1 step 4. Warping functions live on a constrained space: every one of them
    is monotone and pinned at both ends, so the set is not a vector space and PCA applied to
    raw gamma would put mass on combinations that are not warps at all. The standard remedy
    is the square root slope transform psi = sqrt(gamma dot), which carries the warps onto the
    positive orthant of the unit Hilbert sphere, followed by the log map at the Karcher mean
    of the psi family, which lands them in a genuine vector space where a linear method is
    meaningful.

    This is ``fdasrsf.utility_functions.SqrtMean``, used rather than hand rolled: it returns
    the Karcher mean psi, the mean warp, the psi family, and the shooting vectors, which is
    precisely the log map image this stage needs. Build spec 10.1 says to use the library
    where the library has it, and it has it.

    Called with ``smooth=False``, against the library's own default of ``smooth=True``. That
    default fits a ``UnivariateSpline(..., s=1e-4)`` to each warp before differentiating it and
    clips the result at zero, which is a denoising step aimed at warps estimated from noisy
    data. Measured against this project's own exact inverse (:func:`ufem.surrogate.srsf_curve`,
    the log map followed by exact quadrature, which applies no smoothing), the smoothed forward
    map is not a matched pair: round tripping 14 seeded synthetic monotone families through
    ``smooth=True`` gave a reconstruction error of 1 to 5 percent, ten to a hundred times this
    representation's measured discretization floor, and 6 of those 14 families never converged
    at all, hitting the library's fixed 500 iteration cap on its Karcher mean gradient descent
    (a real bug there: the iteration counter then indexes one past the end of its own log
    array). The same 14 families under ``smooth=False`` converged every time, with errors of
    1.5 to 4.7e-3, consistent across seeds. The full measurement is in
    ``docs/DESIGN_DECISIONS.md``. This is a defect in how this project was calling the library,
    not a tolerance to widen around it.
    """
    from fdasrsf.utility_functions import SqrtMean

    gam = np.asarray(gamma, dtype=float)
    if gam.ndim != 2:
        raise ValueError(f"warp_tangent_vectors needs a 2D family, got shape {gam.shape}.")
    mu, _gam_mu, _psi, vec = SqrtMean(gam.T, smooth=False)
    return np.ascontiguousarray(np.asarray(vec, dtype=float).T), np.asarray(mu, dtype=float)


def invert_warp(gamma: np.ndarray, stations: np.ndarray | None = None) -> np.ndarray:
    """Numerical inverse of one warping function, on the same station grid.

    A warp is monotone from [0, 1] onto [0, 1], so its inverse is just the interpolation of
    the stations against gamma instead of gamma against the stations.
    """
    gam = np.asarray(gamma, dtype=float)
    if gam.ndim != 1:
        raise ValueError(f"invert_warp needs one warp at a time, got shape {gam.shape}.")
    s = np.linspace(0.0, 1.0, gam.size) if stations is None else np.asarray(stations, float)
    return np.interp(s, gam, s)


def recover_unregistered(
    registered: np.ndarray, gamma: np.ndarray, stations: np.ndarray | None = None
) -> np.ndarray:
    """Undo the registration of one curve: evaluate the registered function at gamma inverse.

    The composition direction is the thing to get right here, and getting it backwards is
    silent rather than loud: it returns a curve of entirely plausible shape that is simply
    not the input. fdasrsf defines the registered function as the original composed with the
    warp, ``fn = f(gamma)``, so recovering ``f`` means evaluating ``fn`` at ``gamma`` inverse,
    not at ``gamma``. Measured on this family, the correct direction recovers the input to a
    median relative sup norm error of 1.6 percent while the reversed one sits at 26 percent,
    which is how the direction was pinned down rather than argued about.

    The residual is discretization, not error: composing two functions sampled at 201
    stations resamples through regions the warp compresses, and the worst curve in this
    family (relative error 14.7 percent) is exactly the one with the steepest warp, whose
    slope varies by a factor of 42.9 across the grid. A test pins the measured percentiles so
    a real regression is distinguishable from this floor.
    """
    reg = np.asarray(registered, dtype=float)
    gam = np.asarray(gamma, dtype=float)
    if reg.shape != gam.shape:
        raise ValueError(
            f"recover_unregistered needs matching shapes, got registered {reg.shape} and "
            f"gamma {gam.shape}."
        )
    s = np.linspace(0.0, 1.0, reg.size) if stations is None else np.asarray(stations, float)
    return np.interp(invert_warp(gam, s), s, reg)


def gamma_monotonicity_report(gamma: np.ndarray, tolerance: float = 1e-8) -> dict[str, float]:
    """Measure the P3 gate of build spec 22: monotone warps with fixed endpoints.

    Returns the measured quantities rather than a bare pass flag, so the manifest records how
    much slack the family actually had rather than only that it cleared the bar.
    """
    gam = np.asarray(gamma, dtype=float)
    diffs = np.diff(gam, axis=1)
    return {
        "n_warps": int(gam.shape[0]),
        "min_increment": float(diffs.min()),
        "n_decreasing_increments": int((diffs < -tolerance).sum()),
        "max_abs_start_error": float(np.abs(gam[:, 0] - 0.0).max()),
        "max_abs_end_error": float(np.abs(gam[:, -1] - 1.0).max()),
        "tolerance": float(tolerance),
    }


def _matrix_frame(jobs: list[str], matrix: np.ndarray, prefix: str) -> pd.DataFrame:
    """One row per job: the job label then one column per station."""
    columns = [f"{prefix}{index:03d}" for index in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "job", pd.Series(jobs, dtype="string"))
    return frame


def _load_grid(root: Path, config: Config, config_sha256: str) -> tuple[Path, dict[str, str]]:
    """Locate the grid artifacts this stage depends on, or raise naming the fix."""
    directory = stage_dir(root / config.pipeline.paths.artifact_root, GRID_STAGE, config_sha256)
    hashes = {}
    for name in (RF2_GRID_PARQUET, DAMAGE_GRID_PARQUET):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"the register stage needs {path}, which does not exist. Run "
                "`ufem run grid` first: registration operates on the gridded curves."
            )
        hashes[name] = sha256_file(path)
    return directory, hashes


def run(repo_root: Path | str, config: Config, config_sha256: str) -> Path:
    """Execute the register stage and return its artifact directory."""
    started = _time.perf_counter()
    root = Path(repo_root)
    grid_dir, input_hashes = _load_grid(root, config, config_sha256)

    rf2_frame = pd.read_parquet(grid_dir / RF2_GRID_PARQUET)
    jobs, curves = curve_matrix(rf2_frame)
    u_grid = displacement_grid(config)
    if curves.shape[1] != u_grid.size:
        raise AssertionError(
            f"the gridded curves carry {curves.shape[1]} points but the configured "
            f"displacement grid has {u_grid.size}."
        )

    records = []
    for row, job in enumerate(jobs):
        record: dict[str, object] = {"job": job}
        record.update(extract_landmarks(u_grid, curves[row]))
        records.append(record)

    normalizers = config.pipeline.normalizers
    amplitude_raw, totals = arclength_family(
        u_grid, curves, normalizers.u0_mm, normalizers.P0_N
    )
    for row, record in enumerate(records):
        record["arclength_total"] = float(totals[row])
    landmarks = pd.DataFrame.from_records(records, columns=list(LANDMARK_COLUMNS))
    landmarks["job"] = landmarks["job"].astype("string")

    stations = np.linspace(0.0, 1.0, N_ARCLENGTH_POINTS)
    registered, gamma = srsf_register(amplitude_raw, stations)
    tangent, mean_psi = warp_tangent_vectors(gamma)

    monotonicity = gamma_monotonicity_report(gamma)
    if monotonicity["n_decreasing_increments"] > 0:
        raise AssertionError(
            f"{monotonicity['n_decreasing_increments']} warp increments are negative beyond "
            f"the {monotonicity['tolerance']:g} tolerance. Build spec 22 gates phase P3 on "
            "monotone warps: a non monotone warp is not a reparameterization."
        )
    for edge, measured in (
        ("gamma(0) = 0", monotonicity["max_abs_start_error"]),
        ("gamma(1) = 1", monotonicity["max_abs_end_error"]),
    ):
        if measured > monotonicity["tolerance"]:
            raise AssertionError(
                f"the warp endpoint condition {edge} is violated by {measured:.3e}, beyond "
                f"the {monotonicity['tolerance']:g} tolerance of the build spec 22 gate."
            )

    damage_frame = pd.read_parquet(grid_dir / DAMAGE_GRID_PARQUET)
    damage_jobs, _damage = curve_matrix(damage_frame)
    if damage_jobs != jobs:
        raise AssertionError(
            "the gridded force and damage artifacts carry different job orders, which makes "
            "any row wise join between them wrong."
        )

    directory = stage_dir(root / config.pipeline.paths.artifact_root, STAGE_NAME, config_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for frame, name in (
        (landmarks, LANDMARKS_PARQUET),
        (_matrix_frame(jobs, registered, "s"), AMPLITUDE_PARQUET),
        (_matrix_frame(jobs, amplitude_raw, "s"), AMPLITUDE_UNREGISTERED_PARQUET),
        (_matrix_frame(jobs, gamma, "g"), WARP_PARQUET),
        (_matrix_frame(jobs, tangent, "v"), WARP_TANGENT_PARQUET),
    ):
        path = directory / name
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        outputs.append(path)

    recovery = np.array(
        [
            np.abs(recover_unregistered(registered[row], gamma[row]) - amplitude_raw[row]).max()
            / amplitude_raw[row].max()
            for row in range(len(jobs))
        ]
    )

    reached = landmarks["u85_reached"].to_numpy(dtype=bool)
    extra = {
        "recovery_relative_sup_error": {
            "p50": float(np.percentile(recovery, 50)),
            "p90": float(np.percentile(recovery, 90)),
            "p99": float(np.percentile(recovery, 99)),
            "max": float(recovery.max()),
        },
        "cache_key": cache_key(STAGE_NAME, Path(__file__), config_sha256, input_hashes),
        "wall_time_s": _time.perf_counter() - started,
        "n_curves": len(jobs),
        "n_arclength_points": N_ARCLENGTH_POINTS,
        "normalizer_P0_N": float(normalizers.P0_N),
        "normalizer_u0_mm": float(normalizers.u0_mm),
        "srsf_call": "fdasrsf.fdawarp(f, t).srsf_align(parallel=False)",
        "gamma_monotonicity": monotonicity,
        "mean_psi_norm": float(np.linalg.norm(mean_psi)),
        "arclength_total_mean": float(totals.mean()),
        "arclength_total_min": float(totals.min()),
        "arclength_total_max": float(totals.max()),
        "n_curves_reaching_85pct": int(reached.sum()),
        "n_curves_not_reaching_85pct": int((~reached).sum()),
        "u_knee_mean_mm": float(landmarks["u_knee_mm"].mean()),
        "u_peak_mean_mm": float(landmarks["u_peak_mm"].mean()),
        "u_85_mean_mm": float(landmarks.loc[reached, "u_85_mm"].mean()),
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
        f"[register] {len(jobs)} curves on {N_ARCLENGTH_POINTS} arc length stations, "
        f"normalizers P0 {normalizers.P0_N / 1000.0:.0f} kN and u0 {normalizers.u0_mm:.0f} mm; "
        f"warps monotone (min increment {monotonicity['min_increment']:.3e}, endpoint error "
        f"{max(monotonicity['max_abs_start_error'], monotonicity['max_abs_end_error']):.3e}); "
        f"knee at {extra['u_knee_mean_mm']:.2f} mm and peak at "
        f"{extra['u_peak_mean_mm']:.2f} mm on average; "
        f"{extra['n_curves_not_reaching_85pct']} curves never reach 85 percent of peak"
    )
    return directory
