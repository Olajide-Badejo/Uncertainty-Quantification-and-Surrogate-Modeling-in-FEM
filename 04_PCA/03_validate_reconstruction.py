#!/usr/bin/env python3
"""
STEP 4: RECONSTRUCTION VALIDATION
==================================
Validates surrogate predictions against original FEM curves.
Computes curve-wise reconstruction errors for test set.

Outputs:
    - reconstruction_metrics.json: Per-sample reconstruction errors
    - Various comparison plots
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from surrogate_model import SurrogateModel


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Validation configuration."""
    
    BASE = Path(
        r"."
    )
    
    PCA_DIR = BASE / "04_PCA/01_pca_reduction" / "models"
    SURROGATE_DIR = BASE / "04_PCA/01_pca_reduction/outputs"
    
    LOAD_CSV = BASE / "augmentation_physics_fixed" / "load_displacement_full_aug.csv"
    DAMAGE_CSV = BASE / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv"
    UQ_CSV = BASE / "augmentation_physics_fixed" / "processed_inputs_2_aug.csv"
    OUT_DIR = BASE / "04_PCA/02_validation"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def interp_to_grid(
    u_raw: np.ndarray, 
    y_raw: np.ndarray, 
    u_grid: np.ndarray
) -> np.ndarray:
    """Interpolate curve to common grid."""
    order = np.argsort(u_raw)
    u_raw = u_raw[order]
    y_raw = y_raw[order]
    
    u_clipped = np.clip(u_grid, u_raw[0], u_raw[-1])
    return np.interp(u_clipped, u_raw, y_raw)


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    """Main validation workflow."""
    
    config = Config()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 4: RECONSTRUCTION VALIDATION")
    print("="*60)
    
    # --------------------------------------------------------
    # [1/7] LOAD SURROGATE MODEL
    # --------------------------------------------------------
    print("\n[1/7] Loading surrogate model...")
    surrogate = SurrogateModel.load(
        pca_dir=config.PCA_DIR,
        surrogate_dir=config.SURROGATE_DIR,
    )
    print("✓ Model loaded")
    
    # --------------------------------------------------------
    # [2/7] LOAD METADATA
    # --------------------------------------------------------
    print("\n[2/7] Loading metadata...")
    meta = json.loads((config.PCA_DIR / "meta.json").read_text())
    
    all_jobs = meta["jobs"]
    test_idx = np.array(meta["test_idx"], dtype=int)
    test_jobs = [all_jobs[i] for i in test_idx]
    
    u_force = np.array(meta["u_grid_force"])
    u_damage = np.array(meta["u_grid_damage"])
    
    print(f"✓ Found {len(test_jobs)} test samples")
    
    # --------------------------------------------------------
    # [3/7] LOAD ORIGINAL FEM CURVES
    # --------------------------------------------------------
    print("\n[3/7] Loading original FEM curves...")
    
    df_load = pd.read_csv(config.LOAD_CSV)
    df_damage = pd.read_csv(config.DAMAGE_CSV)
    
    # Build true curves (interpolated to PCA grids)
    true_force = []
    true_damage = []
    
    for job in test_jobs:
        # Force
        dfj = df_load[df_load["job_aug"] == job]
        u = np.abs(dfj["U2"].to_numpy(float))
        f = dfj["RF2"].to_numpy(float)
        f_grid = interp_to_grid(u, f, u_force)
        true_force.append(f_grid)
        
        # Damage
        dfd = df_damage[df_damage["job_aug"] == job]
        ud = np.abs(dfd["U2"].to_numpy(float))
        d = dfd["DAMAGEC_max"].to_numpy(float)
        d_grid = interp_to_grid(ud, d, u_damage)
        true_damage.append(d_grid)
    
    true_force = np.vstack(true_force)
    true_damage = np.vstack(true_damage)
    
    print(f"✓ Force curves: {true_force.shape}")
    print(f"✓ Damage curves: {true_damage.shape}")
    
    # --------------------------------------------------------
    # [4/7] LOAD UQ INPUTS
    # --------------------------------------------------------
    print("\n[4/7] Loading material parameters...")
    
    df_uq = pd.read_csv(config.UQ_CSV)
    
    if "job_aug" not in df_uq.columns:
        if "sample_id" in df_uq.columns:
            df_uq["job_aug"] = df_uq["sample_id"].apply(
                lambda i: f"sample_{int(i):03d}"
            )
        else:
            raise RuntimeError("UQ CSV must contain 'job_aug' or 'sample_id'.")
    
    df_uq = df_uq.set_index("job_aug")
    print("✓ Parameters loaded")
    
    # --------------------------------------------------------
    # [5/7] PREDICT SURROGATE CURVES
    # --------------------------------------------------------
    print("\n[5/7] Predicting surrogate curves for test set...")
    
    pred_force = []
    pred_damage = []
    
    for job in test_jobs:
        row = df_uq.loc[job]
        
        F, D = surrogate.predict_curves(
            fc=row["fc"],
            E=row["E"],
            cbot=row["c_nom_bottom_mm"],
            ctop=row["c_nom_top_mm"],
            return_uncertainty=False,
        )
        
        pred_force.append(F)
        pred_damage.append(D)
    
    pred_force = np.vstack(pred_force)
    pred_damage = np.vstack(pred_damage)
    
    print("✓ Predictions complete")
    
    # --------------------------------------------------------
    # [6/7] COMPUTE METRICS
    # --------------------------------------------------------
    print("\n[6/7] Computing reconstruction metrics...")
    
    force_rmse = np.array([rmse(true_force[i], pred_force[i]) for i in range(len(test_jobs))])
    damage_rmse = np.array([rmse(true_damage[i], pred_damage[i]) for i in range(len(test_jobs))])
    
    force_r2 = np.array([r2_score(true_force[i], pred_force[i]) for i in range(len(test_jobs))])
    damage_r2 = np.array([r2_score(true_damage[i], pred_damage[i]) for i in range(len(test_jobs))])
    
    # Save per-sample metrics
    metrics_df = pd.DataFrame({
        "job": test_jobs,
        "force_rmse": force_rmse,
        "force_r2": force_r2,
        "damage_rmse": damage_rmse,
        "damage_r2": damage_r2,
    })
    
    metrics_df.to_csv(config.OUT_DIR / "reconstruction_metrics.csv", index=False)
    
    # Summary statistics
    summary = {
        "force": {
            "rmse_mean": float(force_rmse.mean()),
            "rmse_std": float(force_rmse.std()),
            "rmse_max": float(force_rmse.max()),
            "r2_mean": float(force_r2.mean()),
            "r2_min": float(force_r2.min()),
        },
        "damage": {
            "rmse_mean": float(damage_rmse.mean()),
            "rmse_std": float(damage_rmse.std()),
            "rmse_max": float(damage_rmse.max()),
            "r2_mean": float(damage_r2.mean()),
            "r2_min": float(damage_r2.min()),
        }
    }
    
    (config.OUT_DIR / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    print(f"✓ Force RMSE: {summary['force']['rmse_mean']:.2f} ± {summary['force']['rmse_std']:.2f}")
    print(f"✓ Damage RMSE: {summary['damage']['rmse_mean']:.4f} ± {summary['damage']['rmse_std']:.4f}")
    
    # --------------------------------------------------------
    # [7/7] GENERATE PLOTS
    # --------------------------------------------------------
    print("\n[7/7] Generating validation plots...")
    
    # Plot 1: RMSE distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(force_rmse, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(force_rmse.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {force_rmse.mean():.2f}')
    ax1.set_xlabel("RMSE", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Force: Reconstruction Error Distribution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.hist(damage_rmse, bins=30, color="coral", alpha=0.7, edgecolor="black")
    ax2.axvline(damage_rmse.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {damage_rmse.mean():.4f}')
    ax2.set_xlabel("RMSE", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Damage: Reconstruction Error Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "01_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 01_error_distribution.png")
    
    # Plot 2: Example curves (best, median, worst)
    best_idx = np.argmin(force_rmse)
    median_idx = np.argsort(force_rmse)[len(force_rmse)//2]
    worst_idx = np.argmax(force_rmse)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for col, (idx, label) in enumerate([
        (best_idx, f"Best (RMSE={force_rmse[best_idx]:.2f})"),
        (median_idx, f"Median (RMSE={force_rmse[median_idx]:.2f})"),
        (worst_idx, f"Worst (RMSE={force_rmse[worst_idx]:.2f})")
    ]):
        # Force
        axes[0, col].plot(u_force, true_force[idx], 'k-', lw=2, label="FEM True")
        axes[0, col].plot(u_force, pred_force[idx], 'r--', lw=2, label="Surrogate")
        axes[0, col].set_xlabel("Displacement (mm)", fontsize=10)
        axes[0, col].set_ylabel("Force (N)", fontsize=10)
        axes[0, col].set_title(f"Force: {label}", fontsize=11, fontweight="bold")
        axes[0, col].legend(fontsize=9)
        axes[0, col].grid(True, alpha=0.3)
        
        # Damage
        axes[1, col].plot(u_damage, true_damage[idx], 'k-', lw=2, label="FEM True")
        axes[1, col].plot(u_damage, pred_damage[idx], 'r--', lw=2, label="Surrogate")
        axes[1, col].set_xlabel("Displacement (mm)", fontsize=10)
        axes[1, col].set_ylabel("Damage", fontsize=10)
        axes[1, col].set_title(f"Damage: {test_jobs[idx]} (RMSE={damage_rmse[idx]:.4f})", 
                               fontsize=11, fontweight="bold")
        axes[1, col].legend(fontsize=9)
        axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "02_example_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 02_example_curves.png")
    
    # Plot 3: R² scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(force_r2, bins=30, color="forestgreen", alpha=0.7, edgecolor="black")
    ax1.axvline(force_r2.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {force_r2.mean():.3f}')
    ax1.set_xlabel("R² Score", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Force: R² Distribution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.hist(damage_r2, bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax2.axvline(damage_r2.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {damage_r2.mean():.3f}')
    ax2.set_xlabel("R² Score", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Damage: R² Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "03_r2_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 03_r2_distribution.png")
    
    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("VALIDATION SUMMARY (Test Set Only)")
    print("="*60)
    print(f"\nForce Curves (n={len(test_jobs)}):")
    print(f"  RMSE: {summary['force']['rmse_mean']:.2f} ± {summary['force']['rmse_std']:.2f} N")
    print(f"  R²:   {summary['force']['r2_mean']:.3f} (min: {summary['force']['r2_min']:.3f})")
    
    print(f"\nDamage Curves (n={len(test_jobs)}):")
    print(f"  RMSE: {summary['damage']['rmse_mean']:.4f} ± {summary['damage']['rmse_std']:.4f}")
    print(f"  R²:   {summary['damage']['r2_mean']:.3f} (min: {summary['damage']['r2_min']:.3f})")
    
    print("\n" + "="*60)
    print("✓ STEP 4 COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()