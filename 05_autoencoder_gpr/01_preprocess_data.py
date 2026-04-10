#!/usr/bin/env python3
"""
AE + GPR Pipeline - Step 1: Data Preprocessing
===============================================
Load FEM force-displacement and compression damage data from CSV files.
Interpolate curves onto common grids, normalize, and create train/val/test split.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def build_curves_from_csv(df_force, df_crack, n_force_points=200, n_crack_points=200):
    """
    Build stacked force and damage curves per job on common displacement grids.
    
    Args:
        df_force: DataFrame with columns [job_aug, U2, RF2]
        df_crack: DataFrame with columns [job_aug, U2, DAMAGEC_max]
        n_force_points: Number of interpolation points for force curves
        n_crack_points: Number of interpolation points for damage curves
    
    Returns:
        jobs: List of job identifiers
        u_force: Common displacement grid for force
        F_all: Force curves (n_jobs × n_force_points)
        u_crack: Common displacement grid for damage
        C_all: Damage curves (n_jobs × n_crack_points)
    """
    jobs = sorted(df_force["job_aug"].unique())
    
    print(f"\nBuilding curves for {len(jobs)} jobs...")

    # Collect raw displacement ranges per job
    uF_min, uF_max = [], []
    uC_min, uC_max = [], []

    for job in jobs:
        dfF_j = df_force[df_force["job_aug"] == job].sort_values("U2")
        dfC_j = df_crack[df_crack["job_aug"] == job].sort_values("U2")

        uF = dfF_j["U2"].to_numpy(float)
        uC = dfC_j["U2"].to_numpy(float)

        uF_min.append(uF.min())
        uF_max.append(uF.max())
        uC_min.append(uC.min())
        uC_max.append(uC.max())

    # Global grids (common to all jobs)
    u_force = np.linspace(min(uF_min), max(uF_max), n_force_points)
    u_crack = np.linspace(min(uC_min), max(uC_max), n_crack_points)

    print(f"  Force displacement range: [{min(uF_min):.4f}, {max(uF_max):.4f}] mm")
    print(f"  Damage displacement range: [{min(uC_min):.4f}, {max(uC_max):.4f}] mm")
    print(f"  Force grid points: {n_force_points}")
    print(f"  Damage grid points: {n_crack_points}")

    F_all = np.zeros((len(jobs), n_force_points), dtype=float)
    C_all = np.zeros((len(jobs), n_crack_points), dtype=float)

    for i, job in enumerate(jobs):
        dfF_j = df_force[df_force["job_aug"] == job].sort_values("U2")
        dfC_j = df_crack[df_crack["job_aug"] == job].sort_values("U2")

        uF = dfF_j["U2"].to_numpy(float)
        F = dfF_j["RF2"].to_numpy(float)

        uC = dfC_j["U2"].to_numpy(float)
        C = dfC_j["DAMAGEC_max"].to_numpy(float)  # Compression damage

        # Interpolate onto common grids
        F_all[i, :] = np.interp(u_force, uF, F)
        C_all[i, :] = np.interp(u_crack, uC, C)

    print(f"✓ Built {len(jobs)} force curves")
    print(f"✓ Built {len(jobs)} damage curves")

    return jobs, u_force, F_all, u_crack, C_all


def normalize_curves(F_all, C_all, eps=1e-8):
    """
    Normalize force curves using a GLOBAL max force (preserves physical variation).
    Normalize damage curves per-curve.
    
    Args:
        F_all: Force curves (n_jobs × n_points)
        C_all: Damage curves (n_jobs × n_points)
        eps: Small value to prevent division by zero
    
    Returns:
        F_norm: Normalized force curves
        C_norm: Normalized damage curves
        global_Fmax: Global max force for denormalization
        C_max: Per-curve damage max for denormalization
    """
    F_norm = F_all.copy()
    C_norm = C_all.copy()

    # -----------------------------
    # GLOBAL MAX FORCE NORMALIZATION
    # -----------------------------
    global_Fmax = np.max(np.abs(F_norm))
    if global_Fmax < eps:
        print("  ⚠️  Warning: Global max force is near zero!")
        global_Fmax = 1.0

    F_norm /= global_Fmax

    print(f"\nForce normalization:")
    print(f"  Global max force: {global_Fmax:.3e} N")
    print(f"  Normalized range: [{F_norm.min():.6f}, {F_norm.max():.6f}]")

    # -----------------------------
    # DAMAGE CURVE NORMALIZATION (PER-CURVE)
    # -----------------------------
    C_max = np.max(np.abs(C_norm), axis=1, keepdims=True)
    C_max[C_max < eps] = 1.0
    C_norm /= C_max

    print(f"\nDamage normalization (per-curve):")
    print(f"  Max damage range: [{C_max.min():.6f}, {C_max.max():.6f}]")
    print(f"  Normalized range: [{C_norm.min():.6f}, {C_norm.max():.6f}]")

    return F_norm, C_norm, global_Fmax, C_max.squeeze()


def main():
    base = Path(
        r"."
    )

    print("="*70)
    print("AE + GPR PIPELINE - DATA PREPROCESSING")
    print("="*70)

    # --------------------------------------------------------
    # INPUT CSV PATHS
    # --------------------------------------------------------
    path_force = base / r"augmentation_physics_fixed\load_displacement_full_aug.csv"
    path_crack = base / r"augmentation_physics_fixed\crack_evolution_full_aug.csv"

    print("\nInput files:")
    print(f"  Force data: {path_force.name}")
    print(f"  Damage data: {path_crack.name}")

    if not path_force.exists():
        raise FileNotFoundError(f"Force CSV not found: {path_force}")
    if not path_crack.exists():
        raise FileNotFoundError(f"Damage CSV not found: {path_crack}")

    # Load CSVs
    print("\nLoading CSV files...")
    dfF = pd.read_csv(path_force)
    dfC = pd.read_csv(path_crack)

    print(f"  Force CSV: {len(dfF)} rows")
    print(f"    Columns: {list(dfF.columns)}")
    print(f"  Crack CSV: {len(dfC)} rows")
    print(f"    Columns: {list(dfC.columns)}")

    # Verify required columns
    required_force_cols = ["job_aug", "U2", "RF2"]
    required_crack_cols = ["job_aug", "U2", "DAMAGEC_max"]
    
    for col in required_force_cols:
        if col not in dfF.columns:
            raise ValueError(f"Required column '{col}' not found in force CSV")
    
    for col in required_crack_cols:
        if col not in dfC.columns:
            raise ValueError(f"Required column '{col}' not found in crack CSV")

    # --------------------------------------------------------
    # BUILD CURVES PER JOB
    # --------------------------------------------------------
    print("\n" + "-"*70)
    print("BUILDING CURVES")
    print("-"*70)
    
    jobs, u_force, F_all, u_crack, C_all = build_curves_from_csv(dfF, dfC)
    N = len(jobs)

    print(f"\nCurve statistics:")
    print(f"  Number of jobs: {N}")
    print(f"  u_force shape: {u_force.shape}")
    print(f"  F_all shape: {F_all.shape}")
    print(f"  u_crack shape: {u_crack.shape}")
    print(f"  C_all shape: {C_all.shape}")

    # Quality check
    print(f"\nData quality check:")
    print(f"  Force - Max: {F_all.max():.3e} N, Min: {F_all.min():.3e} N")
    print(f"  Damage - Max: {C_all.max():.6f}, Min: {C_all.min():.6f}")
    
    zero_force = np.sum(F_all.max(axis=1) < 1e-6)
    if zero_force > 0:
        print(f"  ⚠️  Warning: {zero_force} jobs have near-zero force")

    # --------------------------------------------------------
    # NORMALIZE CURVES
    # --------------------------------------------------------
    print("\n" + "-"*70)
    print("NORMALIZING CURVES")
    print("-"*70)
    
    F_norm_all, C_norm_all, global_Fmax, C_max = normalize_curves(F_all, C_all)

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
    # --------------------------------------------------------
    print("\n" + "-"*70)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("-"*70)
    
    rng = np.random.default_rng(seed=42)
    indices = np.arange(N)
    rng.shuffle(indices)

    train_size = int(0.70 * N)
    val_size = int(0.15 * N)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_idx):4d} ({len(train_idx)/N*100:.1f}%)")
    print(f"  Validation: {len(val_idx):4d} ({len(val_idx)/N*100:.1f}%)")
    print(f"  Test:       {len(test_idx):4d} ({len(test_idx)/N*100:.1f}%)")
    print(f"  Total:      {N:4d}")

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------
    out_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-"*70)
    print("SAVING PREPROCESSED DATA")
    print("-"*70)
    print(f"\nOutput directory: {out_dir}")

    # Save grids
    np.save(out_dir / "u_force.npy", u_force)
    np.save(out_dir / "u_crack.npy", u_crack)
    print("\n✓ Saved displacement grids")

    # Save raw curves
    np.save(out_dir / "F_all.npy", F_all)
    np.save(out_dir / "C_all.npy", C_all)
    print("✓ Saved raw curves (F_all, C_all)")

    # Save normalized curves
    np.save(out_dir / "F_norm_all.npy", F_norm_all)
    np.save(out_dir / "C_norm_all.npy", C_norm_all)
    print("✓ Saved normalized curves (F_norm_all, C_norm_all)")

    # Save normalization factors
    np.save(out_dir / "F_global_max.npy", np.array(global_Fmax))
    np.save(out_dir / "C_max.npy", C_max)
    print("✓ Saved normalization factors")

    # Save indices and job names
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "val_indices.npy", val_idx)
    np.save(out_dir / "test_indices.npy", test_idx)
    np.save(out_dir / "jobs.npy", np.array(jobs))
    print("✓ Saved split indices and job names")

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nAll data saved to: {out_dir}")
    print("\nReady for autoencoder training (Step 2)")
    print("="*70)


if __name__ == "__main__":
    main()