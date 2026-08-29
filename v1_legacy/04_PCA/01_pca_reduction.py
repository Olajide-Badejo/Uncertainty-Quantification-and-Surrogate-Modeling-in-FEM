#!/usr/bin/env python3
"""
STEP 2: PCA DIMENSIONALITY REDUCTION
=====================================
Performs PCA on force-displacement and damage evolution curves.
Now uses DAMAGEC (compression damage) as the target variable.

Key features:
- Interpolates curves to common displacement grid
- Applies global normalization
- Fits PCA on TRAINING data only
- Saves PCA models, scores, and metadata

Outputs:
    - pca_outputs.xlsx: PCA scores for all samples
    - pca_force.joblib: Trained PCA model for force
    - pca_damage.joblib: Trained PCA model for damage
    - meta.json: All metadata (grids, scales, train/val/test split)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ============================================================
# UTILITIES
# ============================================================

def _ensure_monotonic_increasing(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by x and drop duplicate x values (keeping the last)."""
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    _, idx_last = np.unique(x_s[::-1], return_index=True)
    idx_last = (len(x_s) - 1) - idx_last
    idx_last = np.sort(idx_last)
    return x_s[idx_last], y_s[idx_last]


def _interp_to_grid(u_raw: np.ndarray, y_raw: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Interpolate y(u) onto a fixed grid u_grid."""
    u_raw, y_raw = _ensure_monotonic_increasing(u_raw, y_raw)
    u_g = np.clip(u_grid, u_raw[0], u_raw[-1])
    return np.interp(u_g, u_raw, y_raw)


def fit_pca(Y: np.ndarray, n_components: int) -> PCA:
    """Fit PCA model."""
    pca = PCA(n_components=n_components)
    pca.fit(Y)
    return pca


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="PCA preprocessing for FEM surrogate modeling")

    # Input/output paths
    ap.add_argument(
        "--load_csv",
        type=str,
        default=r".\augmentation_physics_fixed\load_displacement_full_aug.csv",
    )
    ap.add_argument(
        "--damage_csv",
        type=str,
        default=r".\augmentation_physics_fixed\crack_evolution_full_aug.csv",
    )
    ap.add_argument(
        "--out_xlsx",
        type=str,
        default=r".\04_PCA\01_pca_reduction\pca_outputs.xlsx",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=r".\04_PCA\01_pca_reduction\models",
    )

    # Column names
    ap.add_argument("--u_col", type=str, default="U2")
    ap.add_argument("--force_col", type=str, default="RF2")
    ap.add_argument("--damage_col", type=str, default="DAMAGEC_max")  # Changed from crack_metric

    # Grid settings
    ap.add_argument("--n_grid", type=int, default=400)
    ap.add_argument("--u_max", type=float, default=20.0)

    # PCA settings
    ap.add_argument("--k_force", type=int, default=5)
    ap.add_argument("--k_damage", type=int, default=3)  # Renamed from k_crack

    # Train/val/test split
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: PCA DIMENSIONALITY REDUCTION")
    print("="*60)
    print("\n[1/7] Loading raw data...")
    
    df_load = pd.read_csv(args.load_csv)
    df_damage = pd.read_csv(args.damage_csv)

    if "job_aug" not in df_load.columns or "job_aug" not in df_damage.columns:
        raise ValueError("Both load and damage CSVs must contain a 'job' column.")

    # --------------------------------------------------------
    # COMMON JOBS
    # --------------------------------------------------------
    jobs = sorted(set(df_load["job_aug"]) & set(df_damage["job_aug"]))
    if not jobs:
        raise ValueError("No overlapping jobs between load and damage CSVs.")
    
    print(f"✓ Found {len(jobs)} samples")

    # --------------------------------------------------------
    # BUILD FIXED GRIDS
    # --------------------------------------------------------
    print("\n[2/7] Creating displacement grids...")
    u_grid_force = np.linspace(0.0, args.u_max, args.n_grid)
    u_grid_damage = np.linspace(0.0, args.u_max, args.n_grid)

    # --------------------------------------------------------
    # BUILD MATRICES YF, YD (interpolated to grids)
    # --------------------------------------------------------
    print("\n[3/7] Interpolating curves to common grid...")
    YF, YD = [], []

    for job in jobs:
        # Force curve
        dfj = df_load[df_load["job_aug"] == job]
        u = np.abs(dfj[args.u_col].to_numpy(dtype=float))
        f = dfj[args.force_col].to_numpy(dtype=float)
        f_grid = _interp_to_grid(u, f, u_grid_force)

        # Damage curve
        dfd = df_damage[df_damage["job_aug"] == job]
        ud = np.abs(dfd[args.u_col].to_numpy(dtype=float))
        d = dfd[args.damage_col].to_numpy(dtype=float)
        d_grid = _interp_to_grid(ud, d, u_grid_damage)

        YF.append(f_grid)
        YD.append(d_grid)

    YF = np.vstack(YF)
    YD = np.vstack(YD)

    print(f"✓ Force matrix: {YF.shape}")
    print(f"✓ Damage matrix: {YD.shape}")

    # --------------------------------------------------------
    # GLOBAL NORMALIZATION
    # --------------------------------------------------------
    print("\n[4/7] Applying global normalization...")
    
    global_force_scale = float(np.max(np.abs(YF)))
    global_damage_scale = float(np.max(np.abs(YD)))
    
    if global_force_scale == 0.0:
        global_force_scale = 1.0
    if global_damage_scale == 0.0:
        global_damage_scale = 1.0

    YF_norm = YF / global_force_scale
    YD_norm = YD / global_damage_scale

    print(f"✓ Force scale: {global_force_scale:.2f}")
    print(f"✓ Damage scale: {global_damage_scale:.4f}")

    # --------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # --------------------------------------------------------
    print("\n[5/7] Splitting data (train/val/test)...")
    
    N = len(jobs)
    idx = np.arange(N)
    
    # First split: train vs (val+test)
    idx_tr, idx_tmp = train_test_split(
        idx, 
        train_size=args.train_frac, 
        random_state=args.random_state
    )
    
    # Second split: val vs test
    val_frac_of_remaining = args.val_frac / (1 - args.train_frac)
    idx_va, idx_te = train_test_split(
        idx_tmp, 
        train_size=val_frac_of_remaining, 
        random_state=args.random_state
    )

    print(f"✓ Train: {len(idx_tr)} samples ({len(idx_tr)/N*100:.1f}%)")
    print(f"✓ Val:   {len(idx_va)} samples ({len(idx_va)/N*100:.1f}%)")
    print(f"✓ Test:  {len(idx_te)} samples ({len(idx_te)/N*100:.1f}%)")

    # --------------------------------------------------------
    # PCA FIT ON TRAIN ONLY
    # --------------------------------------------------------
    print("\n[6/7] Fitting PCA on training data...")
    
    k_force = min(args.k_force, YF_norm.shape[0], YF_norm.shape[1])
    k_damage = min(args.k_damage, YD_norm.shape[0], YD_norm.shape[1])

    pca_force = fit_pca(YF_norm[idx_tr], k_force)
    pca_damage = fit_pca(YD_norm[idx_tr], k_damage)

    # Transform all data (train, val, test)
    ZF = pca_force.transform(YF_norm)
    ZD = pca_damage.transform(YD_norm)

    print(f"✓ Force PCA: {k_force} components, "
          f"{pca_force.explained_variance_ratio_.sum()*100:.1f}% variance explained")
    print(f"✓ Damage PCA: {k_damage} components, "
          f"{pca_damage.explained_variance_ratio_.sum()*100:.1f}% variance explained")

    # --------------------------------------------------------
    # SAVE EXCEL OUTPUTS
    # --------------------------------------------------------
    print("\n[7/7] Writing outputs...")
    
    jobs_index = pd.Index(jobs, name="job")

    df_scores_force = pd.DataFrame(
        ZF, 
        index=jobs_index, 
        columns=[f"PC{i+1}" for i in range(ZF.shape[1])]
    )
    df_scores_damage = pd.DataFrame(
        ZD, 
        index=jobs_index, 
        columns=[f"PC{i+1}" for i in range(ZD.shape[1])]
    )

    df_components_force = pd.DataFrame(
        pca_force.components_,
        columns=[f"u_{i}" for i in range(len(u_grid_force))],
    )
    df_components_force.insert(0, "PC", [f"PC{i+1}" for i in range(df_components_force.shape[0])])

    df_components_damage = pd.DataFrame(
        pca_damage.components_,
        columns=[f"u_{i}" for i in range(len(u_grid_damage))],
    )
    df_components_damage.insert(0, "PC", [f"PC{i+1}" for i in range(df_components_damage.shape[0])])

    df_grid_force = pd.DataFrame({"u_grid": u_grid_force})
    df_grid_damage = pd.DataFrame({"u_grid": u_grid_damage})

    df_ev_force = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca_force.explained_variance_ratio_))],
        "explained_variance_ratio": pca_force.explained_variance_ratio_,
        "explained_variance": pca_force.explained_variance_,
        "cumulative_variance": np.cumsum(pca_force.explained_variance_ratio_),
    })

    df_ev_damage = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca_damage.explained_variance_ratio_))],
        "explained_variance_ratio": pca_damage.explained_variance_ratio_,
        "explained_variance": pca_damage.explained_variance_,
        "cumulative_variance": np.cumsum(pca_damage.explained_variance_ratio_),
    })

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_scores_force.to_excel(writer, sheet_name="scores_force")
        df_scores_damage.to_excel(writer, sheet_name="scores_damage")
        df_ev_force.to_excel(writer, sheet_name="explained_force", index=False)
        df_ev_damage.to_excel(writer, sheet_name="explained_damage", index=False)
        df_grid_force.to_excel(writer, sheet_name="grid_force", index=False)
        df_grid_damage.to_excel(writer, sheet_name="grid_damage", index=False)
        df_components_force.to_excel(writer, sheet_name="components_force", index=False)
        df_components_damage.to_excel(writer, sheet_name="components_damage", index=False)

    # --------------------------------------------------------
    # SAVE MODELS + META
    # --------------------------------------------------------
    dump(pca_force, out_dir / "pca_force.joblib")
    dump(pca_damage, out_dir / "pca_damage.joblib")

    meta = {
        "load_csv": str(args.load_csv),
        "damage_csv": str(args.damage_csv),
        "u_col": args.u_col,
        "force_col": args.force_col,
        "damage_col": args.damage_col,
        "n_grid": args.n_grid,
        "u_max": args.u_max,
        "k_force": int(k_force),
        "k_damage": int(k_damage),
        "global_force_scale": float(global_force_scale),
        "global_damage_scale": float(global_damage_scale),
        "jobs": jobs,
        "u_grid_force": u_grid_force.tolist(),
        "u_grid_damage": u_grid_damage.tolist(),
        "train_idx": idx_tr.tolist(),
        "val_idx": idx_va.tolist(),
        "test_idx": idx_te.tolist(),
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✓ PCA scores: {out_xlsx}")
    print(f"✓ PCA models: {out_dir}")
    print("\n" + "="*60)
    print("✓ STEP 2 COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()