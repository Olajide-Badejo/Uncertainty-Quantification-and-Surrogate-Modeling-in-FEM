#!/usr/bin/env python3
"""
STEP 3: SURROGATE MODEL TRAINING & EVALUATION
==============================================
Trains GPR surrogates mapping material parameters → PCA scores.
Includes comprehensive evaluation with training/validation/test metrics and plots.

Key features:
- Trains separate GPR for each PCA component
- Evaluates on train, validation, and test sets
- Generates learning curves and error plots
- Saves all models and evaluation metrics

Outputs:
    - force_gpr_models.joblib: List of trained GPR models for force
    - damage_gpr_models.joblib: List of trained GPR models for damage
    - input_scaler.joblib: StandardScaler for input features
    - training_results.json: All metrics and evaluation results
    - Various PNG plots for visualization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Training configuration."""
    ROOT = Path(__file__).resolve().parents[1]

    # Input paths
    UQ_CSV = ROOT / "augmentation_physics_fixed" / "processed_inputs_2_aug.csv"
    PCA_XLSX = ROOT / "04_PCA" / "01_pca_reduction" / "pca_outputs.xlsx"
    PCA_DIR = ROOT / "04_PCA" / "01_pca_reduction" / "models"

    # Output paths
    OUT_DIR = ROOT / "04_PCA" / "01_pca_reduction" / "outputs"
    
    # Model configuration
    FEATURES = ["fc", "E", "c_nom_bottom_mm", "c_nom_top_mm"]
    RANDOM_STATE = 42
    
    # GPR hyperparameters
    N_RESTARTS_FORCE = 10
    N_RESTARTS_DAMAGE = 5


# ============================================================
# GPR FACTORY FUNCTIONS
# ============================================================

def make_gpr_force(n_features: int, config: Config) -> GaussianProcessRegressor:
    """Create GPR model for force PCA scores."""
    kernel = (
        ConstantKernel(0.5, (1e-3, 2.0))
        * RBF(
            length_scale=np.ones(n_features) * 2.5,
            length_scale_bounds=(1.0, 50.0)
        )
        + WhiteKernel(
            noise_level=4e-2,
            noise_level_bounds=(5e-3, 1e1)
        )
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=config.N_RESTARTS_FORCE,
        alpha=0.0,
        random_state=config.RANDOM_STATE,
    )


def make_gpr_damage(n_features: int, config: Config) -> GaussianProcessRegressor:
    """Create GPR model for damage PCA scores."""
    kernel = (
        ConstantKernel(0.5, (1e-3, 2.0))
        * RBF(
            length_scale=np.ones(n_features) * 2.0,
            length_scale_bounds=(0.5, 30.0)
        )
        + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-3, 1e1)
        )
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=config.N_RESTARTS_DAMAGE,
        alpha=0.0,
        random_state=config.RANDOM_STATE,
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_gpr_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    gpr_factory,
    target_name: str,
) -> Tuple[List[GaussianProcessRegressor], Dict]:
    """
    Train GPR models for each PCA component.
    
    Returns:
        models: List of trained GPR models
        metrics: Dictionary of evaluation metrics
    """
    
    n_components = Y_train.shape[1]
    models = []
    
    # Storage for metrics
    metrics = {
        "train_rmse": [],
        "val_rmse": [],
        "test_rmse": [],
        "train_mae": [],
        "val_mae": [],
        "test_mae": [],
        "train_r2": [],
        "val_r2": [],
        "test_r2": [],
        "train_pred": np.zeros_like(Y_train),
        "val_pred": np.zeros_like(Y_val),
        "test_pred": np.zeros_like(Y_test),
    }
    
    print(f"\nTraining {target_name} GPR models ({n_components} components)...")
    
    for i in range(n_components):
        print(f"  Component {i+1}/{n_components}...", end=" ")
        
        # Get training targets for this component
        y_train = Y_train[:, i]
        y_val = Y_val[:, i]
        y_test = Y_test[:, i]
        
        # Create and train model
        gpr = gpr_factory(X_train.shape[1])
        gpr.fit(X_train, y_train)
        models.append(gpr)
        
        # Predictions
        train_pred = gpr.predict(X_train)
        val_pred = gpr.predict(X_val)
        test_pred = gpr.predict(X_test)
        
        metrics["train_pred"][:, i] = train_pred
        metrics["val_pred"][:, i] = val_pred
        metrics["test_pred"][:, i] = test_pred
        
        # Metrics
        metrics["train_rmse"].append(np.sqrt(mean_squared_error(y_train, train_pred)))
        metrics["val_rmse"].append(np.sqrt(mean_squared_error(y_val, val_pred)))
        metrics["test_rmse"].append(np.sqrt(mean_squared_error(y_test, test_pred)))
        
        metrics["train_mae"].append(mean_absolute_error(y_train, train_pred))
        metrics["val_mae"].append(mean_absolute_error(y_val, val_pred))
        metrics["test_mae"].append(mean_absolute_error(y_test, test_pred))
        
        metrics["train_r2"].append(r2_score(y_train, train_pred))
        metrics["val_r2"].append(r2_score(y_val, val_pred))
        metrics["test_r2"].append(r2_score(y_test, test_pred))
        
        print(f"✓ Val RMSE: {metrics['val_rmse'][-1]:.4f}")
    
    # Convert lists to arrays
    for key in ["train_rmse", "val_rmse", "test_rmse", 
                "train_mae", "val_mae", "test_mae",
                "train_r2", "val_r2", "test_r2"]:
        metrics[key] = np.array(metrics[key])
    
    return models, metrics


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_training_validation_curves(
    force_metrics: Dict,
    damage_metrics: Dict,
    out_dir: Path,
):
    """Plot training vs validation RMSE for each PC."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Force
    n_force = len(force_metrics["train_rmse"])
    pcs = np.arange(1, n_force + 1)
    
    ax1.plot(pcs, force_metrics["train_rmse"], 'o-', label="Train", linewidth=2, markersize=8)
    ax1.plot(pcs, force_metrics["val_rmse"], 's-', label="Validation", linewidth=2, markersize=8)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title("Force: Training vs Validation Error", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(pcs)
    
    # Damage
    n_damage = len(damage_metrics["train_rmse"])
    pcs = np.arange(1, n_damage + 1)
    
    ax2.plot(pcs, damage_metrics["train_rmse"], 'o-', label="Train", linewidth=2, markersize=8)
    ax2.plot(pcs, damage_metrics["val_rmse"], 's-', label="Validation", linewidth=2, markersize=8)
    ax2.set_xlabel("Principal Component", fontsize=12)
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.set_title("Damage: Training vs Validation Error", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(pcs)
    
    plt.tight_layout()
    plt.savefig(out_dir / "01_training_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 01_training_validation_curves.png")


def plot_test_performance(
    force_metrics: Dict,
    damage_metrics: Dict,
    out_dir: Path,
):
    """Plot test set performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Force RMSE
    pcs = np.arange(1, len(force_metrics["test_rmse"]) + 1)
    axes[0, 0].bar(pcs, force_metrics["test_rmse"], color="steelblue", alpha=0.7)
    axes[0, 0].set_xlabel("Principal Component", fontsize=11)
    axes[0, 0].set_ylabel("RMSE", fontsize=11)
    axes[0, 0].set_title("Force: Test RMSE by Component", fontsize=13, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_xticks(pcs)
    
    # Force R²
    axes[0, 1].bar(pcs, force_metrics["test_r2"], color="forestgreen", alpha=0.7)
    axes[0, 1].set_xlabel("Principal Component", fontsize=11)
    axes[0, 1].set_ylabel("R² Score", fontsize=11)
    axes[0, 1].set_title("Force: Test R² by Component", fontsize=13, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xticks(pcs)
    axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label="0.8 threshold")
    axes[0, 1].legend()
    
    # Damage RMSE
    pcs = np.arange(1, len(damage_metrics["test_rmse"]) + 1)
    axes[1, 0].bar(pcs, damage_metrics["test_rmse"], color="coral", alpha=0.7)
    axes[1, 0].set_xlabel("Principal Component", fontsize=11)
    axes[1, 0].set_ylabel("RMSE", fontsize=11)
    axes[1, 0].set_title("Damage: Test RMSE by Component", fontsize=13, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(pcs)
    
    # Damage R²
    axes[1, 1].bar(pcs, damage_metrics["test_r2"], color="purple", alpha=0.7)
    axes[1, 1].set_xlabel("Principal Component", fontsize=11)
    axes[1, 1].set_ylabel("R² Score", fontsize=11)
    axes[1, 1].set_title("Damage: Test R² by Component", fontsize=13, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticks(pcs)
    axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label="0.8 threshold")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "02_test_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 02_test_performance.png")


def plot_prediction_scatter(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    target_name: str,
    out_dir: Path,
    filename: str,
):
    """Scatter plot of true vs predicted PCA scores."""
    
    n_components = Y_test.shape[1]
    
    fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 4))
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]
        
        axes[i].scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())
        ]
        axes[i].plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label="Perfect prediction")
        
        # R² score
        r2 = r2_score(y_true, y_pred)
        axes[i].text(
            0.05, 0.95, f"R² = {r2:.3f}",
            transform=axes[i].transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        axes[i].set_xlabel("True PCA Score", fontsize=11)
        axes[i].set_ylabel("Predicted PCA Score", fontsize=11)
        axes[i].set_title(f"{target_name} PC{i+1}", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_error_distribution(
    force_metrics: Dict,
    damage_metrics: Dict,
    out_dir: Path,
):
    """Plot error distribution for test set."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Force errors
    force_errors = []
    for i in range(force_metrics["test_pred"].shape[1]):
        errors = force_metrics["test_pred"][:, i] - Y_test_force[:, i]
        force_errors.extend(errors)
    
    axes[0].hist(force_errors, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label="Zero error")
    axes[0].set_xlabel("Prediction Error", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Force: Test Error Distribution", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Damage errors
    damage_errors = []
    for i in range(damage_metrics["test_pred"].shape[1]):
        errors = damage_metrics["test_pred"][:, i] - Y_test_damage[:, i]
        damage_errors.extend(errors)
    
    axes[1].hist(damage_errors, bins=50, color="coral", alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label="Zero error")
    axes[1].set_xlabel("Prediction Error", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Damage: Test Error Distribution", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / "04_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 04_error_distribution.png")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main training workflow."""
    
    config = Config()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 3: SURROGATE MODEL TRAINING")
    print("="*60)
    
    # --------------------------------------------------------
    # [1/8] LOAD UQ INPUTS
    # --------------------------------------------------------
    print("\n[1/8] Loading UQ inputs...")
    df_uq = pd.read_csv(config.UQ_CSV)
    
    if "sample_id" in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id"].apply(lambda i: f"sample_{int(i):03d}")
    elif "sample" in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample"]
    elif "job_aug" not in df_uq.columns:
        raise RuntimeError("UQ CSV must contain 'sample_id', 'sample', or 'job' column.")
    
    for f in config.FEATURES:
        if f not in df_uq.columns:
            raise KeyError(f"Missing feature '{f}' in {config.UQ_CSV}")
    
    X_all = df_uq[["job_aug"] + config.FEATURES].set_index("job_aug")
    print(f"✓ Loaded {len(X_all)} samples")
    
    # --------------------------------------------------------
    # [2/8] LOAD PCA SCORES
    # --------------------------------------------------------
    print("\n[2/8] Loading PCA scores...")
    xls = pd.ExcelFile(config.PCA_XLSX)
    scores_force = pd.read_excel(xls, sheet_name="scores_force", index_col=0)
    scores_damage = pd.read_excel(xls, sheet_name="scores_damage", index_col=0)
    
    # Load metadata for train/val/test split
    meta = json.loads((config.PCA_DIR / "meta.json").read_text())
    
    # --------------------------------------------------------
    # [3/8] ALIGN DATA
    # --------------------------------------------------------
    print("\n[3/8] Aligning data...")
    jobs = sorted(set(X_all.index).intersection(scores_force.index, scores_damage.index))
    if not jobs:
        raise RuntimeError("No overlapping jobs between UQ CSV and PCA scores.")
    
    X_raw = X_all.loc[jobs, config.FEATURES].to_numpy(dtype=float)
    Y_force = scores_force.loc[jobs].to_numpy(dtype=float)
    Y_damage = scores_damage.loc[jobs].to_numpy(dtype=float)
    
    print(f"✓ Aligned {len(jobs)} samples")
    print(f"✓ Force PCA components: {Y_force.shape[1]}")
    print(f"✓ Damage PCA components: {Y_damage.shape[1]}")
    
    # --------------------------------------------------------
    # [4/8] SCALE INPUTS
    # --------------------------------------------------------
    print("\n[4/8] Scaling inputs...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, config.OUT_DIR / "input_scaler.joblib")
    print("✓ Input scaling complete")
    
    # --------------------------------------------------------
    # [5/8] SPLIT DATA
    # --------------------------------------------------------
    print("\n[5/8] Splitting data...")
    
    # Get indices from meta.json
    train_idx = np.array(meta["train_idx"], dtype=int)
    val_idx = np.array(meta["val_idx"], dtype=int)
    test_idx = np.array(meta["test_idx"], dtype=int)
    
    X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
    Y_train_force = Y_force[train_idx]
    Y_val_force = Y_force[val_idx]
    global Y_test_force
    Y_test_force = Y_force[test_idx]
    
    Y_train_damage = Y_damage[train_idx]
    Y_val_damage = Y_damage[val_idx]
    global Y_test_damage
    Y_test_damage = Y_damage[test_idx]
    
    print(f"✓ Train: {len(train_idx)} samples")
    print(f"✓ Val:   {len(val_idx)} samples")
    print(f"✓ Test:  {len(test_idx)} samples")
    
    # --------------------------------------------------------
    # [6/8] TRAIN FORCE MODELS
    # --------------------------------------------------------
    print("\n[6/8] Training force surrogate models...")
    force_models, force_metrics = train_gpr_models(
        X_train, Y_train_force,
        X_val, Y_val_force,
        X_test, Y_test_force,
        lambda nf: make_gpr_force(nf, config),
        "Force",
    )
    joblib.dump(force_models, config.OUT_DIR / "force_gpr_models.joblib")
    
    # --------------------------------------------------------
    # [7/8] TRAIN DAMAGE MODELS
    # --------------------------------------------------------
    print("\n[7/8] Training damage surrogate models...")
    damage_models, damage_metrics = train_gpr_models(
        X_train, Y_train_damage,
        X_val, Y_val_damage,
        X_test, Y_test_damage,
        lambda nf: make_gpr_damage(nf, config),
        "Damage",
    )
    joblib.dump(damage_models, config.OUT_DIR / "damage_gpr_models.joblib")
    
    # --------------------------------------------------------
    # [8/8] SAVE RESULTS & PLOTS
    # --------------------------------------------------------
    print("\n[8/8] Generating evaluation plots...")
    
    # Training/validation curves
    plot_training_validation_curves(force_metrics, damage_metrics, config.OUT_DIR)
    
    # Test performance
    plot_test_performance(force_metrics, damage_metrics, config.OUT_DIR)
    
    # Scatter plots
    plot_prediction_scatter(
        Y_test_force, force_metrics["test_pred"],
        "Force", config.OUT_DIR, "03_force_test_scatter.png"
    )
    plot_prediction_scatter(
        Y_test_damage, damage_metrics["test_pred"],
        "Damage", config.OUT_DIR, "03_damage_test_scatter.png"
    )
    
    # Error distribution
    plot_error_distribution(force_metrics, damage_metrics, config.OUT_DIR)
    
    # Save metrics
    results = {
        "force": {
            "train_rmse_mean": float(force_metrics["train_rmse"].mean()),
            "val_rmse_mean": float(force_metrics["val_rmse"].mean()),
            "test_rmse_mean": float(force_metrics["test_rmse"].mean()),
            "train_r2_mean": float(force_metrics["train_r2"].mean()),
            "val_r2_mean": float(force_metrics["val_r2"].mean()),
            "test_r2_mean": float(force_metrics["test_r2"].mean()),
            "per_component": {
                f"PC{i+1}": {
                    "test_rmse": float(force_metrics["test_rmse"][i]),
                    "test_r2": float(force_metrics["test_r2"][i]),
                }
                for i in range(len(force_metrics["test_rmse"]))
            }
        },
        "damage": {
            "train_rmse_mean": float(damage_metrics["train_rmse"].mean()),
            "val_rmse_mean": float(damage_metrics["val_rmse"].mean()),
            "test_rmse_mean": float(damage_metrics["test_rmse"].mean()),
            "train_r2_mean": float(damage_metrics["train_r2"].mean()),
            "val_r2_mean": float(damage_metrics["val_r2"].mean()),
            "test_r2_mean": float(damage_metrics["test_r2"].mean()),
            "per_component": {
                f"PC{i+1}": {
                    "test_rmse": float(damage_metrics["test_rmse"][i]),
                    "test_r2": float(damage_metrics["test_r2"][i]),
                }
                for i in range(len(damage_metrics["test_rmse"]))
            }
        }
    }
    
    (config.OUT_DIR / "training_results.json").write_text(json.dumps(results, indent=2))
    print("✓ Saved: training_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("\nForce Surrogate:")
    print(f"  Train RMSE: {results['force']['train_rmse_mean']:.4f}")
    print(f"  Val RMSE:   {results['force']['val_rmse_mean']:.4f}")
    print(f"  Test RMSE:  {results['force']['test_rmse_mean']:.4f}")
    print(f"  Test R²:    {results['force']['test_r2_mean']:.4f}")
    
    print("\nDamage Surrogate:")
    print(f"  Train RMSE: {results['damage']['train_rmse_mean']:.4f}")
    print(f"  Val RMSE:   {results['damage']['val_rmse_mean']:.4f}")
    print(f"  Test RMSE:  {results['damage']['test_rmse_mean']:.4f}")
    print(f"  Test R²:    {results['damage']['test_r2_mean']:.4f}")
    
    print("\n" + "="*60)
    print("✓ STEP 3 COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
