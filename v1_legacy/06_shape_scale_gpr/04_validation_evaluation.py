"""
===============================================================================
STEP 4c.4: VALIDATION EVALUATION
===============================================================================
Purpose: Comprehensive validation of trained surrogates

Evaluation:
    - Prediction errors (RMSE, NRMSE, R²)
    - Peak value errors
    - Curve reconstruction quality
    - Error distributions
    - Sample-wise analysis

Output:
    - Validation metrics CSV
    - Error distribution plots
    - Prediction quality plots
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from shape_scale_surrogate import ShapeScaleSurrogate

# Set plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "06_shape_scale_gpr"

# Input files
LOAD_CSV = BASE / r"augmentation_physics_fixed\load_displacement_full_aug.csv"
DAMAGE_CSV = BASE / r"augmentation_physics_fixed\crack_evolution_full_aug.csv"
UQ_CSV = BASE / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"

# Validation jobs
VAL_JOBS_FILE = CLEAN / "split" / "val_jobs.txt"

# Output directory
OUT_DIR = CLEAN / "output_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Damage variable
DAMAGE_VAR = 'DAMAGEC_max'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def nrmse(y_true, y_pred):
    """Normalized RMSE"""
    range_y = np.max(y_true) - np.min(y_true)
    if range_y == 0:
        return 0.0
    return rmse(y_true, y_pred) / range_y


def relative_peak_error(y_true, y_pred):
    """Relative error in peak value (%)"""
    peak_true = np.max(np.abs(y_true))
    peak_pred = np.max(np.abs(y_pred))
    if peak_true == 0:
        return 0.0
    return abs(peak_pred - peak_true) / peak_true * 100


# ============================================================
# MAIN
# ============================================================

def main():
    
    print("=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)
    
    # --------------------------------------------------------
    # Load surrogate
    # --------------------------------------------------------
    print("\nLoading surrogate model...")
    model = ShapeScaleSurrogate.load(CLEAN)
    
    info = model.get_info()
    print(f"  ✓ Force modes: {info['n_force_modes']}")
    print(f"  ✓ Damage modes: {info['n_damage_modes']}")
    print(f"  ✓ Force variance explained: {info['force_variance_explained']*100:.2f}%")
    print(f"  ✓ Damage variance explained: {info['damage_variance_explained']*100:.2f}%")
    
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("\nLoading FEM and UQ data...")
    
    df_load = pd.read_csv(LOAD_CSV)
    df_damage = pd.read_csv(DAMAGE_CSV)
    df_uq = pd.read_csv(UQ_CSV)
    
    if "job_aug" not in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
    df_uq = df_uq.set_index("job_aug")
    
    # --------------------------------------------------------
    # Load validation jobs
    # --------------------------------------------------------
    val_jobs = np.loadtxt(VAL_JOBS_FILE, dtype=str)
    print(f"\n✓ Loaded {len(val_jobs)} validation jobs")
    
    # --------------------------------------------------------
    # Evaluate all validation samples
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("EVALUATING VALIDATION SAMPLES")
    print("-" * 70)
    
    results = []
    
    for job in val_jobs:
        
        # Load FEM force curve
        dfj = df_load[df_load["job_aug"] == job]
        if dfj.empty:
            continue
        
        u_fem = dfj["U2"].abs().to_numpy(float)
        f_fem = dfj["RF2"].to_numpy(float)
        
        # Load FEM damage curve
        dfd = df_damage[df_damage["job_aug"] == job]
        if dfd.empty:
            continue
        
        u_damage_fem = dfd["U2"].abs().to_numpy(float)
        
        if DAMAGE_VAR not in dfd.columns:
            continue
        
        d_fem = dfd[DAMAGE_VAR].to_numpy(float)
        
        # Get UQ inputs
        if job not in df_uq.index:
            continue
        
        fc = df_uq.loc[job, "fc"]
        E = df_uq.loc[job, "E"]
        cbot = df_uq.loc[job, "c_nom_bottom_mm"]
        ctop = df_uq.loc[job, "c_nom_top_mm"]
        
        # Predict with surrogate
        u_force, F_pred, u_damage, D_pred = model.predict_curves(fc, E, cbot, ctop)
        
        # Interpolate FEM onto surrogate grids
        f_fem_interp = np.interp(u_force, u_fem, f_fem)
        d_fem_interp = np.interp(u_damage, u_damage_fem, d_fem)
        
        # Compute errors
        rmse_force = rmse(f_fem_interp, F_pred)
        nrmse_force = nrmse(f_fem_interp, F_pred)
        peak_error_force = relative_peak_error(f_fem_interp, F_pred)
        
        rmse_damage = rmse(d_fem_interp, D_pred)
        nrmse_damage = nrmse(d_fem_interp, D_pred)
        
        peak_fem = np.max(f_fem_interp)
        peak_pred = np.max(F_pred)
        
        results.append({
            'job': job,
            'fc': fc,
            'E': E,
            'c_nom_bottom_mm': cbot,
            'c_nom_top_mm': ctop,
            'rmse_force': rmse_force,
            'nrmse_force': nrmse_force,
            'peak_error_force_pct': peak_error_force,
            'peak_fem': peak_fem,
            'peak_pred': peak_pred,
            'rmse_damage': rmse_damage,
            'nrmse_damage': nrmse_damage,
        })
    
    print(f"  ✓ Evaluated {len(results)} samples")
    
    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    df_results = pd.DataFrame(results)
    
    results_csv = OUT_DIR / "validation_metrics.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n✓ Saved metrics to: {results_csv}")
    
    # --------------------------------------------------------
    # Compute statistics
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("VALIDATION STATISTICS")
    print("-" * 70)
    
    print("\nForce prediction:")
    print(f"  RMSE:  Mean = {df_results['rmse_force'].mean():.3f}, "
          f"Median = {df_results['rmse_force'].median():.3f}, "
          f"Std = {df_results['rmse_force'].std():.3f}")
    print(f"  NRMSE: Mean = {df_results['nrmse_force'].mean():.4f}, "
          f"Median = {df_results['nrmse_force'].median():.4f}")
    print(f"  Peak error (%): Mean = {df_results['peak_error_force_pct'].mean():.2f}, "
          f"Median = {df_results['peak_error_force_pct'].median():.2f}")
    
    print("\nDamage prediction:")
    print(f"  RMSE:  Mean = {df_results['rmse_damage'].mean():.4f}, "
          f"Median = {df_results['rmse_damage'].median():.4f}")
    print(f"  NRMSE: Mean = {df_results['nrmse_damage'].mean():.4f}, "
          f"Median = {df_results['nrmse_damage'].median():.4f}")
    
    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("CREATING VALIDATION PLOTS")
    print("-" * 70)
    
    # Plot 1: Error distributions (histograms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df_results['rmse_force'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Force RMSE Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df_results['nrmse_force'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('NRMSE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Force NRMSE Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(df_results['peak_error_force_pct'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Peak Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Force Peak Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(df_results['rmse_damage'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Damage RMSE Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_error_distributions.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 01_error_distributions.png")
    
    # Plot 2: Peak force prediction quality
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df_results['peak_fem'], df_results['peak_pred'], 
               alpha=0.6, edgecolor='k', s=50)
    
    # Perfect prediction line
    lims = [
        min(df_results['peak_fem'].min(), df_results['peak_pred'].min()),
        max(df_results['peak_fem'].max(), df_results['peak_pred'].max())
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('FEM Peak Force [N]')
    ax.set_ylabel('Predicted Peak Force [N]')
    ax.set_title('Peak Force Prediction Quality (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_peak_force_prediction.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 02_peak_force_prediction.png")
    
    # Plot 3: Error vs input parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(df_results['fc'], df_results['rmse_force'], alpha=0.6, edgecolor='k')
    axes[0, 0].set_xlabel('fc [MPa]')
    axes[0, 0].set_ylabel('Force RMSE')
    axes[0, 0].set_title('Error vs Concrete Strength')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(df_results['E'], df_results['rmse_force'], alpha=0.6, edgecolor='k')
    axes[0, 1].set_xlabel('E [MPa]')
    axes[0, 1].set_ylabel('Force RMSE')
    axes[0, 1].set_title('Error vs Young\'s Modulus')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(df_results['c_nom_bottom_mm'], df_results['rmse_force'], alpha=0.6, edgecolor='k')
    axes[1, 0].set_xlabel('Bottom Cover [mm]')
    axes[1, 0].set_ylabel('Force RMSE')
    axes[1, 0].set_title('Error vs Bottom Cover')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df_results['c_nom_top_mm'], df_results['rmse_force'], alpha=0.6, edgecolor='k')
    axes[1, 1].set_xlabel('Top Cover [mm]')
    axes[1, 1].set_ylabel('Force RMSE')
    axes[1, 1].set_title('Error vs Top Cover')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_error_vs_inputs.png", dpi=300)
    plt.close()
    print("  ✓ Saved: 03_error_vs_inputs.png")
    
    # Plot 4: Sample-wise comparison (5 random samples)
    print("\nPlotting sample comparisons...")
    
    np.random.seed(42)
    selected_jobs = np.random.choice(val_jobs, size=min(5, len(val_jobs)), replace=False)
    
    for job in selected_jobs:
        
        # Get data
        dfj = df_load[df_load["job_aug"] == job]
        if dfj.empty:
            continue
        
        u_fem = dfj["U2"].abs().to_numpy(float)
        f_fem = dfj["RF2"].to_numpy(float)
        
        dfd = df_damage[df_damage["job_aug"] == job]
        if dfd.empty or DAMAGE_VAR not in dfd.columns:
            continue
        
        u_damage_fem = dfd["U2"].abs().to_numpy(float)
        d_fem = dfd[DAMAGE_VAR].to_numpy(float)
        
        if job not in df_uq.index:
            continue
        
        fc = df_uq.loc[job, "fc"]
        E = df_uq.loc[job, "E"]
        cbot = df_uq.loc[job, "c_nom_bottom_mm"]
        ctop = df_uq.loc[job, "c_nom_top_mm"]
        
        # Predict
        u_force, F_pred, u_damage, D_pred = model.predict_curves(fc, E, cbot, ctop)
        
        # Plot force comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(u_fem, f_fem, 'b-', linewidth=2, label='FEM')
        axes[0].plot(u_force, F_pred, 'r--', linewidth=2, label='Surrogate')
        axes[0].set_xlabel('Displacement [mm]')
        axes[0].set_ylabel('Reaction Force [N]')
        axes[0].set_title(f'Force-Displacement ({job})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(u_damage_fem, d_fem, 'b-', linewidth=2, label='FEM')
        axes[1].plot(u_damage, D_pred, 'r--', linewidth=2, label='Surrogate')
        axes[1].set_xlabel('Displacement [mm]')
        axes[1].set_ylabel('Tension Damage [-]')
        axes[1].set_title(f'Damage Evolution ({job})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"sample_{job}.png", dpi=200)
        plt.close()
    
    print(f"  ✓ Saved {len(selected_jobs)} sample comparison plots")
    
    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()