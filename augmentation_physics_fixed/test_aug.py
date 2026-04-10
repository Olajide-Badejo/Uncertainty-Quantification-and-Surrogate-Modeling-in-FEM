# for processed_inputs_2_aug.csv

#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# SETTINGS (keep simple, safer defaults)
# ------------------------------------------------------------
RANDOM_SEED = 42
N_POINTS    = 200          # points per curve after resampling
N_AUG       = 4            # augmentation rounds per original
ALPHA_MIN   = 0.05         # small mixing (your original idea)
ALPHA_MAX   = 0.30

INPUT_JITTER_STD  = 0.04   # 2% (safer than 13%)
OUTPUT_NOISE_STD  = 0.005   # 1% additive noise scaled by peak

# --- IMPORTANT: add Eurocode E from fc (do NOT treat E as independent input) ---
# Eurocode-style formula you requested:
#   E = 22000 * (fc/10)^0.3
# We'll compute this as a derived column named "E" and include it in outputs.
PHYSICAL_COLS = ["fc", "E", "c_nom_top_mm", "c_nom_bottom_mm"]  # keep what matters + derived E

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(r".\01_samplying\processed_inputs_4.csv")
LOAD_CSV  = ROOT / "03_postprocess/results_4/load_displacement_full.csv"
CRACK_CSV = ROOT / "03_postprocess/results_4/crack_evolution_full.csv"

OUT_DIR   = Path(r".\augmentation_physics_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_INPUT = OUT_DIR / "processed_inputs_2_aug.csv"
OUT_LOAD  = OUT_DIR / "load_displacement_full_aug.csv"
OUT_CRACK = OUT_DIR / "crack_evolution_full_aug.csv"
OUT_PLOT  = OUT_DIR / "augmentation_validation.png"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def eurocode_E_from_fc(fc):
    """
    Eurocode-style modulus formula requested by you:
        E = 22000*(fc/10)^0.3
    fc expected in MPa. Result is in MPa.
    """
    fc = float(fc)
    fc = max(fc, 1e-6)  # safety
    return 22000.0 * (fc / 10.0) ** 0.3

def ensure_job_format(df_inputs: pd.DataFrame) -> pd.DataFrame:
    """Create job string sample_000 from sample_id and compute derived E from fc."""
    df = df_inputs.copy()
    if "sample_id" not in df.columns:
        raise ValueError("INPUT_CSV must contain 'sample_id' column.")
    if "fc" not in df.columns:
        raise ValueError("INPUT_CSV must contain 'fc' column to compute Eurocode E.")

    df["job"] = df["sample_id"].apply(lambda i: f"sample_{int(i):03d}")

    # Always compute E from fc (override any existing E to keep consistency)
    df["E"] = df["fc"].apply(eurocode_E_from_fc)
    return df

def guess_x_col(df: pd.DataFrame):
    """Try to identify displacement column. Prefer common Abaqus naming."""
    lower = {c.lower(): c for c in df.columns}
    for cand in ["u2", "displacement", "disp", "u"]:
        if cand in lower:
            return lower[cand]
    # fallback: first numeric non-job column
    for c in df.columns:
        if c.lower() in ["job", "job_aug", "job_temp", "sample_id_aug"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not detect x(displacement) column.")

def guess_y_cols(df: pd.DataFrame, x_col: str):
    """All numeric columns except job-ish and x_col."""
    y = []
    for c in df.columns:
        if c in [x_col, "job", "job_aug", "job_temp", "sample_id_aug"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            y.append(c)
    if not y:
        raise ValueError("No numeric y-columns found.")
    return y

def arc_length_param(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute monotone path parameter s in [0,1] using cumulative distance
    in (x, Y1, Y2, ...) space, preserving original order (no sorting).
    """
    Z = np.column_stack([x, Y])
    dZ = np.diff(Z, axis=0)
    ds = np.sqrt((dZ * dZ).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 0:
        return np.linspace(0.0, 1.0, len(x))
    return s / s[-1]

def resample_curve(df: pd.DataFrame, job: str, x_col: str, y_cols: list, n_points: int):
    """
    Resample one job's curve onto common s-grid using arc-length parameter.
    Returns dict of arrays for x and each y.
    """
    d = df[df["job"] == job].copy()
    if d.empty:
        raise ValueError(f"No rows found for job={job} in dataframe.")
    # preserve existing order; DO NOT sort (important for snapback)
    x = d[x_col].to_numpy(dtype=float)
    Y = np.column_stack([d[c].to_numpy(dtype=float) for c in y_cols])

    s = arc_length_param(x, Y)
    s_new = np.linspace(0.0, 1.0, n_points)

    out = {}
    fx = interp1d(s, x, kind="linear", bounds_error=False, fill_value=(x[0], x[-1]))
    out[x_col] = fx(s_new)

    for j, c in enumerate(y_cols):
        y = Y[:, j]
        fy = interp1d(s, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
        out[c] = fy(s_new)

    return out

def mix_arrays(A: dict, B: dict, alpha: float):
    """C = A + alpha*(B-A) for all keys (same keys)."""
    C = {}
    for k in A.keys():
        C[k] = A[k] + alpha * (B[k] - A[k])
    return C

def add_output_noise(C: dict, rng, y_cols: list, noise_std: float):
    """Add small additive noise to y columns scaled by each column peak."""
    C2 = {k: v.copy() for k, v in C.items()}
    for c in y_cols:
        peak = np.max(np.abs(C2[c])) + 1e-12
        C2[c] = C2[c] + rng.normal(0.0, noise_std * peak, size=C2[c].shape)
    return C2

def jitter_inputs(row_a: pd.Series, row_b: pd.Series, alpha: float, rng):
    """
    Mix inputs + small jitter. E is NOT jittered independently:
    - We mix/jitter fc and covers
    - Then recompute E from fc using Eurocode formula
    """
    r = row_a.copy()

    # --- mix/jitter fc and covers ---
    for col in ["fc", "c_nom_top_mm", "c_nom_bottom_mm"]:
        if col not in row_a.index:
            raise ValueError(f"Missing input column {col} in INPUT_CSV.")
        v = row_a[col] + alpha * (row_b[col] - row_a[col])
        v = v * rng.normal(1.0, INPUT_JITTER_STD)
        v = max(float(v), 1e-6)
        r[col] = v

    # --- recompute E from fc (Eurocode) ---
    r["E"] = eurocode_E_from_fc(r["fc"])
    return r

def curves_to_long_df(job: str, arrays: dict):
    """Convert resampled arrays into long dataframe with 'job' + columns."""
    n = len(next(iter(arrays.values())))
    out = {"job": [job] * n}
    for k, v in arrays.items():
        out[k] = v
    return pd.DataFrame(out)

def plot_validation(df_load_aug, df_crack_aug, load_x, load_y, crack_x, crack_y, n_orig=5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # plot originals
    for i in range(n_orig):
        job = f"sample_{i:03d}"
        dl = df_load_aug[df_load_aug["job_aug"] == job]
        dc = df_crack_aug[df_crack_aug["job_aug"] == job]
        if not dl.empty:
            ax1.plot(dl[load_x], dl[load_y], alpha=0.7, label="Original" if i == 0 else None)
        if not dc.empty:
            ax2.plot(dc[crack_x], dc[crack_y], alpha=0.7)

    # plot some augmented (take last 5 ids)
    last_jobs = sorted(df_load_aug["job_aug"].unique())[-5:]
    for i, job in enumerate(last_jobs):
        dl = df_load_aug[df_load_aug["job_aug"] == job]
        dc = df_crack_aug[df_crack_aug["job_aug"] == job]
        ax1.plot(dl[load_x], dl[load_y], "--", alpha=0.9, label="Augmented" if i == 0 else None)
        ax2.plot(dc[crack_x], dc[crack_y], "--", alpha=0.9)

    ax1.set_title("Load–Displacement (original vs augmented)")
    ax1.set_xlabel(load_x); ax1.set_ylabel(load_y); ax1.legend()

    ax2.set_title("Crack Evolution (original vs augmented)")
    ax2.set_xlabel(crack_x); ax2.set_ylabel(crack_y)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    print(f"✓ Validation plot saved: {OUT_PLOT}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df_in   = pd.read_csv(INPUT_CSV)
    df_load = pd.read_csv(LOAD_CSV)
    df_crk  = pd.read_csv(CRACK_CSV)

    # Ensure job column exists everywhere + compute Eurocode E
    df_in = ensure_job_format(df_in)

    if "job" not in df_load.columns or "job" not in df_crk.columns:
        raise ValueError("LOAD_CSV and CRACK_CSV must contain a 'job' column matching sample_000 naming.")

    jobs = sorted(df_in["job"].unique())
    n_orig = len(jobs)

    # Detect columns
    load_x = guess_x_col(df_load)
    load_y_cols = guess_y_cols(df_load, load_x)

    crack_x = guess_x_col(df_crk)
    crack_y_cols = guess_y_cols(df_crk, crack_x)

    # Pre-resample all original curves (fast + consistent)
    load_resampled = {}
    crack_resampled = {}
    for j in jobs:
        load_resampled[j]  = resample_curve(df_load, j, load_x, load_y_cols, N_POINTS)
        crack_resampled[j] = resample_curve(df_crk,  j, crack_x, crack_y_cols, N_POINTS)

    # Containers
    input_rows = [df_in.copy()]
    load_rows  = [df_load.copy()]
    crack_rows = [df_crk.copy()]

    # Build augmented samples
    for k in range(1, N_AUG + 1):
        new_in_list = []
        new_load_list = []
        new_crack_list = []

        for ja in jobs:
            jb = rng.choice([j for j in jobs if j != ja])
            alpha = float(rng.uniform(ALPHA_MIN, ALPHA_MAX))

            # ----- inputs (mix fc/covers, then recompute E from fc)
            row_a = df_in[df_in["job"] == ja].iloc[0]
            row_b = df_in[df_in["job"] == jb].iloc[0]
            new_row = jitter_inputs(row_a, row_b, alpha, rng)
            temp_id = f"{ja}_aug{k}"
            new_row["job"] = temp_id
            new_in_list.append(pd.DataFrame([new_row]))

            # ----- curves (mix in s-space)
            LA = load_resampled[ja]
            LB = load_resampled[jb]
            LC = mix_arrays(LA, LB, alpha)
            LC = add_output_noise(LC, rng, load_y_cols, OUTPUT_NOISE_STD)
            new_load_list.append(curves_to_long_df(temp_id, LC))

            CA = crack_resampled[ja]
            CB = crack_resampled[jb]
            CC = mix_arrays(CA, CB, alpha)
            CC = add_output_noise(CC, rng, crack_y_cols, OUTPUT_NOISE_STD)
            new_crack_list.append(curves_to_long_df(temp_id, CC))

        input_rows.append(pd.concat(new_in_list, ignore_index=True))
        load_rows.append(pd.concat(new_load_list, ignore_index=True))
        crack_rows.append(pd.concat(new_crack_list, ignore_index=True))

    # Final concatenation
    df_in_f   = pd.concat(input_rows, ignore_index=True)
    df_load_f = pd.concat(load_rows, ignore_index=True)
    df_crk_f  = pd.concat(crack_rows, ignore_index=True)

    # Re-index jobs to sample_000..sample_N
    unique_jobs = list(df_in_f["job"].unique())
    job_map = {name: f"sample_{i:03d}" for i, name in enumerate(unique_jobs)}
    id_map  = {name: i for i, name in enumerate(unique_jobs)}

    df_in_f["sample_id_aug"] = df_in_f["job"].map(id_map)
    df_in_f["job_aug"] = df_in_f["job"].map(job_map)

    df_load_f["job_aug"] = df_load_f["job"].map(job_map)
    df_crk_f["job_aug"]  = df_crk_f["job"].map(job_map)

    # Save
    df_in_f[["sample_id_aug", "job_aug"] + PHYSICAL_COLS].to_csv(OUT_INPUT, index=False)
    df_load_f.drop(columns=["job"], errors="ignore").to_csv(OUT_LOAD, index=False)
    df_crk_f.drop(columns=["job"], errors="ignore").to_csv(OUT_CRACK, index=False)

    # Validation plot (pick first y-col for each)
    load_y = load_y_cols[0]
    crack_y = crack_y_cols[0]
    plot_validation(df_load_f, df_crk_f, load_x, load_y, crack_x, crack_y, n_orig=min(5, n_orig))

    print(f"✓ Done. Original: {n_orig}, total after aug: {len(unique_jobs)}")
    print(f"✓ Saved:\n  {OUT_INPUT}\n  {OUT_LOAD}\n  {OUT_CRACK}\n  {OUT_PLOT}")

if __name__ == "__main__":
    main()
