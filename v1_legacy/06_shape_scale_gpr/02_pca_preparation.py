"""
===============================================================================
STEP 4c.2: SHAPE-SCALE PCA PREPARATION
===============================================================================
Purpose: Perform shape-scale decomposition and PCA on training data

Shape-scale decomposition:
    - Force curves: Normalize by max value (scale), then PCA on shapes
    - Damage curves: Direct PCA (no scaling needed, already [0,1])

IMPORTANT: Uses DAMAGET (tension damage) from ABAQUS, NOT crack_metric

Training strategy:
    - Fit PCA on TRAINING data only
    - Transform validation and test using training PCA basis
    - Prevents data leakage

Output:
    - PCA models (pca_force.joblib, pca_damage.joblib)
    - Displacement grids (u_force.npy, u_damage.npy)
    - PCA scores for train/val/test
===============================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from joblib import dump

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "06_shape_scale_gpr"

# Input files
LOAD_CSV = BASE / r"augmentation_physics_fixed\load_displacement_full_aug.csv"
DAMAGE_CSV = BASE / r"augmentation_physics_fixed\crack_evolution_full_aug.csv"
UQ_CSV = BASE / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"

# Split files
SPLIT_DIR = CLEAN / "split"

# Output directory
OUT_DIR = CLEAN / "output_pca_shapes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid sizes
N_FORCE_GRID = 400
N_DAMAGE_GRID = 400

# PCA components (will be adjusted if not enough samples)
K_FORCE = 10   # Force shape modes
K_DAMAGE = 7   # Damage modes

# Damage variable to use
# Options: 'DAMAGET' (tension damage) or 'DAMAGEC' (compression damage)
DAMAGE_VAR = 'DAMAGEC_max'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def interp_to_grid(u_raw, y_raw, u_grid):
    """
    Interpolate curve onto common grid.
    
    Parameters:
        u_raw: Raw displacement values
        y_raw: Raw output values (force or damage)
        u_grid: Target grid
        
    Returns:
        Interpolated values on u_grid
    """
    # Sort by displacement
    order = np.argsort(u_raw)
    u_raw = u_raw[order]
    y_raw = y_raw[order]
    
    # Clip grid to data range
    u_clipped = np.clip(u_grid, u_raw[0], u_raw[-1])
    
    # Interpolate
    return np.interp(u_clipped, u_raw, y_raw)


def process_force_curves(job_list, df_load, u_grid):
    """
    Process force curves: interpolate, normalize, extract scale.
    
    Returns:
        Y_raw: Raw force curves (n_samples × n_grid)
        Y_norm: Normalized force curves
        scales: Scale factors (n_samples,)
    """
    Y_raw = []
    Y_norm = []
    scales = []
    
    for job in job_list:
        # Get job data
        dfj = df_load[df_load["job_aug"] == job]
        
        if dfj.empty:
            print(f"  ⚠ WARNING: No force data for {job}, skipping")
            continue
        
        # Extract displacement and force
        u = dfj["U2"].abs().to_numpy(float)
        f = dfj["RF2"].to_numpy(float)
        
        # Interpolate onto grid
        f_grid = interp_to_grid(u, f, u_grid)
        
        # Compute scale (max absolute value)
        scale = np.max(np.abs(f_grid))
        if scale == 0.0:
            scale = 1.0  # Avoid division by zero
        
        # Normalize
        f_norm = f_grid / scale
        
        Y_raw.append(f_grid)
        Y_norm.append(f_norm)
        scales.append(scale)
    
    return np.vstack(Y_raw), np.vstack(Y_norm), np.array(scales)


def process_damage_curves(job_list, df_damage, u_grid, damage_var):
    """
    Process damage curves: interpolate to common grid.
    
    No normalization needed - damage is already in [0,1].
    
    Returns:
        Y_damage: Damage curves (n_samples × n_grid)
    """
    Y_damage = []
    
    for job in job_list:
        # Get job data
        dfj = df_damage[df_damage["job_aug"] == job]
        
        if dfj.empty:
            print(f"  ⚠ WARNING: No damage data for {job}, skipping")
            continue
        
        # Extract displacement and damage
        u = dfj["U2"].abs().to_numpy(float)
        
        if damage_var not in dfj.columns:
            print(f"  ⚠ WARNING: {damage_var} not found for {job}, skipping")
            continue
        
        d = dfj[damage_var].to_numpy(float)
        
        # Interpolate onto grid
        d_grid = interp_to_grid(u, d, u_grid)
        
        Y_damage.append(d_grid)
    
    return np.vstack(Y_damage)


# ============================================================
# MAIN
# ============================================================

def main():
    
    print("=" * 70)
    print("SHAPE-SCALE PCA PREPARATION")
    print("=" * 70)
    print(f"Damage variable: {DAMAGE_VAR}")
    print("=" * 70)
    
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("\nLoading data...")
    
    df_load = pd.read_csv(LOAD_CSV)
    df_damage = pd.read_csv(DAMAGE_CSV)
    df_uq = pd.read_csv(UQ_CSV)
    
    # Add job names if needed
    if "job_aug" not in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
    df_uq = df_uq.set_index("job_aug")
    
    print(f"  ✓ Load data: {len(df_load)} records")
    print(f"  ✓ Damage data: {len(df_damage)} records")
    print(f"  ✓ UQ inputs: {len(df_uq)} samples")
    
    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------
    print("\nLoading train/val/test splits...")
    
    train_jobs = np.loadtxt(SPLIT_DIR / "train_jobs.txt", dtype=str)
    val_jobs = np.loadtxt(SPLIT_DIR / "val_jobs.txt", dtype=str)
    test_jobs = np.loadtxt(SPLIT_DIR / "test_jobs.txt", dtype=str)
    
    print(f"  ✓ Train: {len(train_jobs)}")
    print(f"  ✓ Val: {len(val_jobs)}")
    print(f"  ✓ Test: {len(test_jobs)}")
    
    # --------------------------------------------------------
    # Build displacement grids
    # --------------------------------------------------------
    print("\nBuilding displacement grids...")
    
    u_force_max = df_load["U2"].abs().max()
    u_damage_max = df_damage["U2"].abs().max()
    
    u_force = np.linspace(0.0, u_force_max, N_FORCE_GRID)
    u_damage = np.linspace(0.0, u_damage_max, N_DAMAGE_GRID)
    
    print(f"  ✓ Force grid: {N_FORCE_GRID} points, max = {u_force_max:.4f}")
    print(f"  ✓ Damage grid: {N_DAMAGE_GRID} points, max = {u_damage_max:.4f}")
    
    # --------------------------------------------------------
    # Process training set
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("PROCESSING TRAINING SET")
    print("-" * 70)
    
    print("\nProcessing force curves...")
    Y_force_train, Y_force_norm_train, scales_force_train = process_force_curves(
        train_jobs, df_load, u_force
    )
    print(f"  ✓ Shape: {Y_force_train.shape}")
    print(f"  ✓ Scale range: [{scales_force_train.min():.2f}, {scales_force_train.max():.2f}]")
    
    print("\nProcessing damage curves...")
    Y_damage_train = process_damage_curves(
        train_jobs, df_damage, u_damage, DAMAGE_VAR
    )
    print(f"  ✓ Shape: {Y_damage_train.shape}")
    
    # --------------------------------------------------------
    # Process validation set
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("PROCESSING VALIDATION SET")
    print("-" * 70)
    
    print("\nProcessing force curves...")
    Y_force_val, Y_force_norm_val, scales_force_val = process_force_curves(
        val_jobs, df_load, u_force
    )
    print(f"  ✓ Shape: {Y_force_val.shape}")
    
    print("\nProcessing damage curves...")
    Y_damage_val = process_damage_curves(
        val_jobs, df_damage, u_damage, DAMAGE_VAR
    )
    print(f"  ✓ Shape: {Y_damage_val.shape}")
    
    # --------------------------------------------------------
    # Process test set
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("PROCESSING TEST SET")
    print("-" * 70)
    
    print("\nProcessing force curves...")
    Y_force_test, Y_force_norm_test, scales_force_test = process_force_curves(
        test_jobs, df_load, u_force
    )
    print(f"  ✓ Shape: {Y_force_test.shape}")
    
    print("\nProcessing damage curves...")
    Y_damage_test = process_damage_curves(
        test_jobs, df_damage, u_damage, DAMAGE_VAR
    )
    print(f"  ✓ Shape: {Y_damage_test.shape}")
    
    # --------------------------------------------------------
    # FIT PCA ON TRAINING DATA ONLY
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("FITTING PCA (TRAINING DATA ONLY)")
    print("-" * 70)
    
    # Adjust k_force if needed
    k_force = min(K_FORCE, Y_force_norm_train.shape[0], Y_force_norm_train.shape[1])
    print(f"\nForce shape PCA: {k_force} components")
    pca_force = PCA(n_components=k_force)
    scores_force_train = pca_force.fit_transform(Y_force_norm_train)
    
    explained_var_force = pca_force.explained_variance_ratio_
    cumsum_var_force = np.cumsum(explained_var_force)
    print(f"  ✓ Explained variance: {cumsum_var_force[-1]*100:.2f}%")
    print(f"  ✓ First 5 components: {cumsum_var_force[:5]*100}")
    
    # Adjust k_damage if needed
    k_damage = min(K_DAMAGE, Y_damage_train.shape[0], Y_damage_train.shape[1])
    print(f"\nDamage PCA: {k_damage} components")
    pca_damage = PCA(n_components=k_damage)
    scores_damage_train = pca_damage.fit_transform(Y_damage_train)
    
    explained_var_damage = pca_damage.explained_variance_ratio_
    cumsum_var_damage = np.cumsum(explained_var_damage)
    print(f"  ✓ Explained variance: {cumsum_var_damage[-1]*100:.2f}%")
    print(f"  ✓ First 5 components: {cumsum_var_damage[:5]*100}")
    
    # --------------------------------------------------------
    # TRANSFORM VAL/TEST USING TRAINING PCA
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("TRANSFORMING VALIDATION & TEST SETS")
    print("-" * 70)
    
    scores_force_val = pca_force.transform(Y_force_norm_val)
    scores_force_test = pca_force.transform(Y_force_norm_test)
    
    scores_damage_val = pca_damage.transform(Y_damage_val)
    scores_damage_test = pca_damage.transform(Y_damage_test)
    
    print("  ✓ Validation force scores:", scores_force_val.shape)
    print("  ✓ Test force scores:", scores_force_test.shape)
    print("  ✓ Validation damage scores:", scores_damage_val.shape)
    print("  ✓ Test damage scores:", scores_damage_test.shape)
    
    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("SAVING PCA MODELS AND DATA")
    print("-" * 70)
    
    # Save PCA models
    dump(pca_force, OUT_DIR / "pca_force.joblib")
    dump(pca_damage, OUT_DIR / "pca_damage.joblib")
    print("  ✓ Saved PCA models")
    
    # Save grids
    np.save(OUT_DIR / "u_force.npy", u_force)
    np.save(OUT_DIR / "u_damage.npy", u_damage)
    print("  ✓ Saved displacement grids")
    
    # Save training data
    np.savez(
        OUT_DIR / "pca_shapes_data_train.npz",
        jobs=train_jobs,
        scores_force=scores_force_train,
        scores_damage=scores_damage_train,
        scales_force=scales_force_train,
    )
    
    # Save validation data
    np.savez(
        OUT_DIR / "pca_shapes_data_val.npz",
        jobs=val_jobs,
        scores_force=scores_force_val,
        scores_damage=scores_damage_val,
        scales_force=scales_force_val,
    )
    
    # Save test data
    np.savez(
        OUT_DIR / "pca_shapes_data_test.npz",
        jobs=test_jobs,
        scores_force=scores_force_test,
        scores_damage=scores_damage_test,
        scales_force=scales_force_test,
    )
    
    print("  ✓ Saved PCA scores for train/val/test")
    
    # Save metadata
    meta = {
        "k_force": int(k_force),
        "k_damage": int(k_damage),
        "n_force_grid": int(N_FORCE_GRID),
        "n_damage_grid": int(N_DAMAGE_GRID),
        "damage_variable": DAMAGE_VAR,
        "explained_variance_force": explained_var_force.tolist(),
        "explained_variance_damage": explained_var_damage.tolist(),
        "cumulative_variance_force": cumsum_var_force.tolist(),
        "cumulative_variance_damage": cumsum_var_damage.tolist(),
        "description": "Shape+scale PCA (training set only). Val/test transformed using training PCA.",
    }
    
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("  ✓ Saved metadata")
    
    print("\n" + "=" * 70)
    print("PCA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()