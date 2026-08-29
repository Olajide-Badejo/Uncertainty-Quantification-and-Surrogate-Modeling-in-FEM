"""
===============================================================================
STEP 4c.3: TRAIN GAUSSIAN PROCESS SURROGATES
===============================================================================
Purpose: Train GPR models for shape and scale prediction

Surrogates:
    1. Force shape surrogate: Multi-output GPR predicting PCA scores
    2. Force scale surrogate: Single-output GPR predicting scale factor
    3. Damage surrogate: Multi-output GPR predicting damage PCA scores

Training strategy:
    - Fit on training data
    - Monitor validation performance
    - Report test performance as final unbiased metric

Output:
    - Trained models (gpr_force_shape.joblib, etc.)
    - Loss plots (train vs val, test, learning curves)
    - Performance metrics CSV
===============================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "06_shape_scale_gpr"

# Input directories
PCA_DIR = CLEAN / "output_pca_shapes"
SPLIT_DIR = CLEAN / "split"

# Output directory
OUT_DIR = CLEAN / "output_surrogates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "training_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# UQ inputs
UQ_CSV = BASE / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"

# Kernel choice: 'rbf' or 'matern'
KERNEL_TYPE = 'rbf'

# GPR hyperparameters
ALPHA = 1e-4  # Noise level
NORMALIZE_Y = True

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nrmse(y_true, y_pred):
    """Normalized RMSE (divided by range)"""
    range_y = np.max(y_true) - np.min(y_true)
    if range_y == 0:
        return 0.0
    return rmse(y_true, y_pred) / range_y


def create_kernel(kernel_type='rbf'):
    """
    Create GP kernel.
    
    Options:
        'rbf': Radial Basis Function (smooth)
        'matern': Matérn kernel (more flexible)
    """
    if kernel_type.lower() == 'rbf':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type.lower() == 'matern':
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel


def evaluate_model(model, X, y_true, label=""):
    """
    Evaluate model and return metrics.
    
    Returns:
        dict: Metrics (RMSE, NRMSE, R²)
    """
    y_pred = model.predict(X)
    
    metrics = {
        f'{label}_rmse': rmse(y_true, y_pred),
        f'{label}_nrmse': nrmse(y_true, y_pred),
        f'{label}_r2': r2_score(y_true, y_pred, multioutput='uniform_average') if y_pred.ndim > 1 else r2_score(y_true, y_pred)
    }
    
    return metrics, y_pred


# ============================================================
# MAIN
# ============================================================

def main():
    
    print("=" * 70)
    print("TRAINING GAUSSIAN PROCESS SURROGATES")
    print("=" * 70)
    print(f"Kernel type: {KERNEL_TYPE}")
    print(f"Alpha: {ALPHA}")
    print("=" * 70)
    
    # --------------------------------------------------------
    # Load PCA data
    # --------------------------------------------------------
    print("\nLoading PCA shape+scale data...")
    
    train_data = np.load(PCA_DIR / "pca_shapes_data_train.npz", allow_pickle=True)
    val_data = np.load(PCA_DIR / "pca_shapes_data_val.npz", allow_pickle=True)
    test_data = np.load(PCA_DIR / "pca_shapes_data_test.npz", allow_pickle=True)
    
    train_jobs = train_data["jobs"]
    val_jobs = val_data["jobs"]
    test_jobs = test_data["jobs"]
    
    # Force shape scores
    scores_force_train = train_data["scores_force"]
    scores_force_val = val_data["scores_force"]
    scores_force_test = test_data["scores_force"]
    
    # Force scales
    scales_force_train = train_data["scales_force"]
    scales_force_val = val_data["scales_force"]
    scales_force_test = test_data["scales_force"]
    
    # Damage scores
    scores_damage_train = train_data["scores_damage"]
    scores_damage_val = val_data["scores_damage"]
    scores_damage_test = test_data["scores_damage"]
    
    print(f"  ✓ Train: {len(train_jobs)} samples")
    print(f"  ✓ Val: {len(val_jobs)} samples")
    print(f"  ✓ Test: {len(test_jobs)} samples")
    
    # --------------------------------------------------------
    # Load UQ inputs
    # --------------------------------------------------------
    print("\nLoading UQ inputs...")
    
    df_uq = pd.read_csv(UQ_CSV)
    
    if "job_aug" not in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
    df_uq = df_uq.set_index("job_aug")
    
    # Extract inputs for each split
    input_cols = ["fc", "E", "c_nom_bottom_mm", "c_nom_top_mm"]
    
    df_train = df_uq.loc[train_jobs]
    df_val = df_uq.loc[val_jobs]
    df_test = df_uq.loc[test_jobs]
    
    X_train = df_train[input_cols].to_numpy(float)
    X_val = df_val[input_cols].to_numpy(float)
    X_test = df_test[input_cols].to_numpy(float)
    
    print(f"  ✓ Input features: {input_cols}")
    print(f"  ✓ Input shape: {X_train.shape}")
    
    # --------------------------------------------------------
    # Scale inputs
    # --------------------------------------------------------
    print("\nScaling inputs...")
    
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)
    
    print("  ✓ Input scaling complete")
    
    # --------------------------------------------------------
    # Define kernel
    # --------------------------------------------------------
    kernel = create_kernel(KERNEL_TYPE)
    print(f"\n✓ Created {KERNEL_TYPE.upper()} kernel")
    
    # --------------------------------------------------------
    # Train Force Shape Surrogate
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("TRAINING FORCE SHAPE SURROGATE")
    print("-" * 70)
    
    print(f"Output dimensionality: {scores_force_train.shape[1]}")
    
    gpr_force_shape = MultiOutputRegressor(
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=ALPHA,
            normalize_y=NORMALIZE_Y,
            n_restarts_optimizer=3
        )
    )
    
    gpr_force_shape.fit(Xs_train, scores_force_train)
    print("  ✓ Training complete")
    
    # Evaluate
    metrics_train, _ = evaluate_model(gpr_force_shape, Xs_train, scores_force_train, "train")
    metrics_val, _ = evaluate_model(gpr_force_shape, Xs_val, scores_force_val, "val")
    metrics_test, _ = evaluate_model(gpr_force_shape, Xs_test, scores_force_test, "test")
    
    print(f"\n  Train RMSE: {metrics_train['train_rmse']:.4f}, R²: {metrics_train['train_r2']:.4f}")
    print(f"  Val RMSE:   {metrics_val['val_rmse']:.4f}, R²: {metrics_val['val_r2']:.4f}")
    print(f"  Test RMSE:  {metrics_test['test_rmse']:.4f}, R²: {metrics_test['test_r2']:.4f}")
    
    force_shape_metrics = {**metrics_train, **metrics_val, **metrics_test}
    
    # --------------------------------------------------------
    # Train Force Scale Surrogate
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("TRAINING FORCE SCALE SURROGATE")
    print("-" * 70)
    
    gpr_force_scale = GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA,
        normalize_y=NORMALIZE_Y,
        n_restarts_optimizer=3
    )
    
    gpr_force_scale.fit(Xs_train, scales_force_train)
    print("  ✓ Training complete")
    
    # Evaluate
    metrics_train, _ = evaluate_model(gpr_force_scale, Xs_train, scales_force_train, "train")
    metrics_val, _ = evaluate_model(gpr_force_scale, Xs_val, scales_force_val, "val")
    metrics_test, _ = evaluate_model(gpr_force_scale, Xs_test, scales_force_test, "test")
    
    print(f"\n  Train RMSE: {metrics_train['train_rmse']:.4f}, R²: {metrics_train['train_r2']:.4f}")
    print(f"  Val RMSE:   {metrics_val['val_rmse']:.4f}, R²: {metrics_val['val_r2']:.4f}")
    print(f"  Test RMSE:  {metrics_test['test_rmse']:.4f}, R²: {metrics_test['test_r2']:.4f}")
    
    force_scale_metrics = {**metrics_train, **metrics_val, **metrics_test}
    
    # --------------------------------------------------------
    # Train Damage Surrogate
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("TRAINING DAMAGE SURROGATE")
    print("-" * 70)
    
    print(f"Output dimensionality: {scores_damage_train.shape[1]}")
    
    gpr_damage = MultiOutputRegressor(
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=ALPHA,
            normalize_y=NORMALIZE_Y,
            n_restarts_optimizer=3
        )
    )
    
    gpr_damage.fit(Xs_train, scores_damage_train)
    print("  ✓ Training complete")
    
    # Evaluate
    metrics_train, _ = evaluate_model(gpr_damage, Xs_train, scores_damage_train, "train")
    metrics_val, _ = evaluate_model(gpr_damage, Xs_val, scores_damage_val, "val")
    metrics_test, _ = evaluate_model(gpr_damage, Xs_test, scores_damage_test, "test")
    
    print(f"\n  Train RMSE: {metrics_train['train_rmse']:.4f}, R²: {metrics_train['train_r2']:.4f}")
    print(f"  Val RMSE:   {metrics_val['val_rmse']:.4f}, R²: {metrics_val['val_r2']:.4f}")
    print(f"  Test RMSE:  {metrics_test['test_rmse']:.4f}, R²: {metrics_test['test_r2']:.4f}")
    
    damage_metrics = {**metrics_train, **metrics_val, **metrics_test}
    
    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("SAVING MODELS")
    print("-" * 70)
    
    dump(gpr_force_shape, OUT_DIR / "gpr_force_shape.joblib")
    dump(gpr_force_scale, OUT_DIR / "gpr_force_scale.joblib")
    dump(gpr_damage, OUT_DIR / "gpr_damage.joblib")
    dump(scaler, OUT_DIR / "input_scaler.joblib")
    
    print("  ✓ Saved all models")
    
    # --------------------------------------------------------
    # Create comprehensive plots
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("CREATING LOSS PLOTS")
    print("-" * 70)
    
    # Plot 1: Train vs Val RMSE (bar chart)
    labels = ["Force Shape", "Force Scale", "Damage"]
    
    train_rmses = [
        force_shape_metrics['train_rmse'],
        force_scale_metrics['train_rmse'],
        damage_metrics['train_rmse']
    ]
    
    val_rmses = [
        force_shape_metrics['val_rmse'],
        force_scale_metrics['val_rmse'],
        damage_metrics['val_rmse']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_rmses, width, label='Train RMSE', color='darkred', alpha=0.8)
    ax.bar(x + width/2, val_rmses, width, label='Validation RMSE', color='steelblue', alpha=0.8)
    
    ax.set_ylabel('RMSE')
    ax.set_title('Training vs Validation RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_train_val_rmse.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 01_train_val_rmse.png")
    
    # Plot 2: Test RMSE (bar chart)
    test_rmses = [
        force_shape_metrics['test_rmse'],
        force_scale_metrics['test_rmse'],
        damage_metrics['test_rmse']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, test_rmses, color='darkgreen', alpha=0.8)
    ax.set_ylabel('RMSE')
    ax.set_title('Test Set RMSE (Unbiased Performance)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_test_rmse.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 02_test_rmse.png")
    
    # Plot 3: R² scores (grouped bar chart)
    train_r2 = [
        force_shape_metrics['train_r2'],
        force_scale_metrics['train_r2'],
        damage_metrics['train_r2']
    ]
    
    val_r2 = [
        force_shape_metrics['val_r2'],
        force_scale_metrics['val_r2'],
        damage_metrics['val_r2']
    ]
    
    test_r2 = [
        force_shape_metrics['test_r2'],
        force_scale_metrics['test_r2'],
        damage_metrics['test_r2']
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, train_r2, width, label='Train', color='darkred', alpha=0.8)
    ax.bar(x, val_r2, width, label='Validation', color='steelblue', alpha=0.8)
    ax.bar(x + width, test_r2, width, label='Test', color='darkgreen', alpha=0.8)
    
    ax.set_ylabel('R² Score')
    ax.set_title('R² Scores Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_r2_scores.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 03_r2_scores.png")
    
    # Plot 4: Learning curves (RMSE vs data size)
    # Simulate by subsampling training data
    print("\nGenerating learning curves...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    n_train_full = len(Xs_train)
    
    lc_force_shape_train = []
    lc_force_shape_val = []
    lc_force_scale_train = []
    lc_force_scale_val = []
    lc_damage_train = []
    lc_damage_val = []
    
    for frac in train_sizes:
        n = int(frac * n_train_full)
        if n < 5:  # Need minimum samples
            continue
        
        # Subsample training data
        indices = np.random.choice(n_train_full, n, replace=False)
        Xs_sub = Xs_train[indices]
        
        # Force shape
        y_sub = scores_force_train[indices]
        gpr_temp = MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, alpha=ALPHA, normalize_y=True)
        )
        gpr_temp.fit(Xs_sub, y_sub)
        lc_force_shape_train.append(rmse(gpr_temp.predict(Xs_sub), y_sub))
        lc_force_shape_val.append(rmse(gpr_temp.predict(Xs_val), scores_force_val))
        
        # Force scale
        y_sub = scales_force_train[indices]
        gpr_temp = GaussianProcessRegressor(kernel=kernel, alpha=ALPHA, normalize_y=True)
        gpr_temp.fit(Xs_sub, y_sub)
        lc_force_scale_train.append(rmse(gpr_temp.predict(Xs_sub), y_sub))
        lc_force_scale_val.append(rmse(gpr_temp.predict(Xs_val), scales_force_val))
        
        # Damage
        y_sub = scores_damage_train[indices]
        gpr_temp = MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, alpha=ALPHA, normalize_y=True)
        )
        gpr_temp.fit(Xs_sub, y_sub)
        lc_damage_train.append(rmse(gpr_temp.predict(Xs_sub), y_sub))
        lc_damage_val.append(rmse(gpr_temp.predict(Xs_val), scores_damage_val))
    
    # Plot learning curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Force shape
    axes[0].plot(train_sizes[:len(lc_force_shape_train)], lc_force_shape_train, 
                 'o-', label='Train', color='darkred')
    axes[0].plot(train_sizes[:len(lc_force_shape_val)], lc_force_shape_val, 
                 's-', label='Validation', color='steelblue')
    axes[0].set_xlabel('Training Set Size (fraction)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Learning Curve: Force Shape')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Force scale
    axes[1].plot(train_sizes[:len(lc_force_scale_train)], lc_force_scale_train, 
                 'o-', label='Train', color='darkred')
    axes[1].plot(train_sizes[:len(lc_force_scale_val)], lc_force_scale_val, 
                 's-', label='Validation', color='steelblue')
    axes[1].set_xlabel('Training Set Size (fraction)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Learning Curve: Force Scale')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Damage
    axes[2].plot(train_sizes[:len(lc_damage_train)], lc_damage_train, 
                 'o-', label='Train', color='darkred')
    axes[2].plot(train_sizes[:len(lc_damage_val)], lc_damage_val, 
                 's-', label='Validation', color='steelblue')
    axes[2].set_xlabel('Training Set Size (fraction)')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Learning Curve: Damage')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_learning_curves.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 04_learning_curves.png")
    
    # --------------------------------------------------------
    # Save metrics to CSV
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("SAVING METRICS")
    print("-" * 70)
    
    metrics_df = pd.DataFrame([
        {"surrogate": "force_shape", **force_shape_metrics},
        {"surrogate": "force_scale", **force_scale_metrics},
        {"surrogate": "damage", **damage_metrics}
    ])
    
    metrics_csv = OUT_DIR / "training_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  ✓ Saved: {metrics_csv}")
    
    # Save meta info
    meta = {
        "kernel_type": KERNEL_TYPE,
        "alpha": ALPHA,
        "normalize_y": NORMALIZE_Y,
        "n_train": int(len(train_jobs)),
        "n_val": int(len(val_jobs)),
        "n_test": int(len(test_jobs)),
        "input_features": input_cols,
        "metrics": {
            "force_shape": force_shape_metrics,
            "force_scale": force_scale_metrics,
            "damage": damage_metrics
        }
    }
    
    (OUT_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2))
    
    print("\n" + "=" * 70)
    print("SURROGATE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModels saved to: {OUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()