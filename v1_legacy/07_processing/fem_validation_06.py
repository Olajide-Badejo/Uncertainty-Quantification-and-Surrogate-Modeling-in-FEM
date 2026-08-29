"""
===============================================================================
STEP 6: SURROGATE VALIDATION AGAINST FEM
===============================================================================
Run small FEM-based Monte Carlo (50-100 samples) to validate surrogate fidelity.

Comparison metrics:
- Mean response curves (FEM vs Surrogate)
- Confidence envelopes (5th-95th percentiles)
- Quantile curves (median, 25th, 75th)
- Statistical agreement tests

Output:
- Validation plots comparing FEM and surrogate
- Statistical comparison report
- Confidence in surrogate for large-scale UQ
===============================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration for FEM validation."""
    
    BASE = Path(__file__).resolve().parents[1]
    
    # Data paths
    LOAD_CSV = BASE / "augmentation_physics_fixed" / "load_displacement_full_aug.csv"
    DAMAGE_CSV = BASE / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv"
    UQ_CSV = BASE / "augmentation_physics_fixed" / "processed_inputs_2_aug.csv"
    
    # Surrogate directory (use best from Step 5)
    SURROGATE_TYPE = "AE+GPR"  # Updated based on Step 5 results
    AE_DIR = BASE / "05_autoencoder_gpr"
    
    # Output directory
    OUT_DIR = BASE / "07_processing" / "07_fem_validation"
    
    # Validation parameters
    N_VALIDATION_SAMPLES = 75  # Number of samples for validation
    DAMAGE_VAR = 'DAMAGEC_max'
    
    # Random seed
    SEED = 42

# ============================================================
# LOAD SURROGATE
# ============================================================

def load_best_surrogate(config: Config):
    """Load the best surrogate from Step 5."""
    
    if config.SURROGATE_TYPE == "AE+GPR":
        import sys
        sys.path.insert(0, str(config.AE_DIR))
        from ae_surrogate_model import ImprovedAESurrogateModel
        
        model = ImprovedAESurrogateModel(str(config.BASE), use_improved=True)
        
        # Load normalization factors
        data_dir = config.AE_DIR / "data_preprocessed"
        global_Fmax = float(np.load(data_dir / "F_global_max.npy"))
        C_max_all = np.load(data_dir / "C_max.npy")
        
        # Store normalization factors with model
        model.global_Fmax = global_Fmax
        model.C_max_all = C_max_all
        
        # Get grids
        model.u_force = np.load(data_dir / "u_force.npy")
        model.u_damage = np.load(data_dir / "u_crack.npy")
        
        # Load job mapping for denormalization
        jobs_all = np.load(data_dir / "jobs.npy")
        model.job_to_idx = {job: idx for idx, job in enumerate(jobs_all)}
        
        return model
    
    elif config.SURROGATE_TYPE == "Shape-Scale PCA+GPR":
        import sys
        sys.path.insert(0, str(config.SHAPE_SCALE_DIR))
        from shape_scale_surrogate import ShapeScaleSurrogate
        
        model = ShapeScaleSurrogate.load(config.SHAPE_SCALE_DIR)
        return model
    
    # Add other surrogate types as needed
    else:
        raise ValueError(f"Surrogate type {config.SURROGATE_TYPE} not implemented")


def predict_unified(model, surrogate_type: str, fc: float, E: float, 
                    cbot: float, ctop: float, job: str = None):
    """Unified prediction interface for different surrogate types."""
    
    if surrogate_type == "AE+GPR":
        # Get normalized predictions
        u_force, F_norm, u_damage, D_norm = model.predict(cbot=cbot, ctop=ctop, fcm=fc)
        
        # Denormalize
        F_pred = F_norm * model.global_Fmax
        
        # Get job index for per-curve damage denormalization
        if hasattr(model, 'job_to_idx') and job and job in model.job_to_idx:
            job_idx = model.job_to_idx[job]
            D_pred = D_norm * model.C_max_all[job_idx]
        else:
            # Use mean if job not found
            D_pred = D_norm * np.mean(model.C_max_all)
        
        return u_force, F_pred, u_damage, D_pred
        
    elif surrogate_type == "Shape-Scale PCA+GPR":
        # Shape-Scale returns actual values directly
        u_force, F_pred, u_damage, D_pred = model.predict_curves(fc, E, cbot, ctop)
        return u_force, F_pred, u_damage, D_pred
    
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")


# ============================================================
# DATA LOADING
# ============================================================

def load_fem_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load FEM data and validation sample list."""
    
    df_load = pd.read_csv(config.LOAD_CSV)
    df_damage = pd.read_csv(config.DAMAGE_CSV)
    df_uq = pd.read_csv(config.UQ_CSV)
    
    if "job_aug" not in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
    df_uq = df_uq.set_index("job_aug")
    
    # Load test set (used for validation)
    test_jobs_file = config.AE_DIR / "split" / "test_jobs.txt"
    if test_jobs_file.exists():
        test_jobs = np.loadtxt(test_jobs_file, dtype=str).tolist()
    else:
        # Fallback: use all available jobs
        test_jobs = sorted(set(df_load["job_aug"].unique()) & set(df_damage["job_aug"].unique()))
    
    # Select random subset for validation
    np.random.seed(config.SEED)
    val_jobs = np.random.choice(test_jobs, 
                                size=min(config.N_VALIDATION_SAMPLES, len(test_jobs)),
                                replace=False).tolist()
    
    return df_load, df_damage, df_uq, val_jobs


# ============================================================
# CURVE EXTRACTION
# ============================================================

def extract_fem_curves(df_load: pd.DataFrame, df_damage: pd.DataFrame, 
                      job: str, damage_var: str, u_grid_force: np.ndarray,
                      u_grid_damage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and interpolate FEM curves onto common grids."""
    
    # Force curve
    dfj = df_load[df_load["job_aug"] == job]
    u_fem = dfj["U2"].abs().to_numpy(float)
    f_fem = dfj["RF2"].to_numpy(float)
    f_fem_interp = np.interp(u_grid_force, u_fem, f_fem)
    
    # Damage curve
    dfd = df_damage[df_damage["job_aug"] == job]
    u_damage_fem = dfd["U2"].abs().to_numpy(float)
    d_fem = dfd[damage_var].to_numpy(float)
    d_fem_interp = np.interp(u_grid_damage, u_damage_fem, d_fem)
    
    return f_fem_interp, d_fem_interp


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def compute_statistical_agreement(fem_curves: np.ndarray, 
                                 surrogate_curves: np.ndarray) -> Dict:
    """Compute statistical agreement metrics between FEM and surrogate ensembles."""
    
    # Mean curves
    fem_mean = np.mean(fem_curves, axis=0)
    surr_mean = np.mean(surrogate_curves, axis=0)
    
    # Standard deviations
    fem_std = np.std(fem_curves, axis=0)
    surr_std = np.std(surrogate_curves, axis=0)
    
    # Percentiles
    fem_p05 = np.percentile(fem_curves, 5, axis=0)
    fem_p95 = np.percentile(fem_curves, 95, axis=0)
    surr_p05 = np.percentile(surrogate_curves, 5, axis=0)
    surr_p95 = np.percentile(surrogate_curves, 95, axis=0)
    
    # Mean error
    mean_error = np.mean(np.abs(fem_mean - surr_mean))
    max_mean_error = np.max(np.abs(fem_mean - surr_mean))
    
    # Confidence band overlap
    overlap = np.mean((surr_p05 <= fem_p95) & (surr_p95 >= fem_p05))
    
    # Kolmogorov-Smirnov test at peak location
    peak_idx = np.argmax(fem_mean)
    ks_stat, ks_p = stats.ks_2samp(fem_curves[:, peak_idx], surrogate_curves[:, peak_idx])
    
    return {
        'mean_error': float(mean_error),
        'max_mean_error': float(max_mean_error),
        'confidence_overlap': float(overlap),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_p),
        'fem_mean': fem_mean,
        'fem_std': fem_std,
        'fem_p05': fem_p05,
        'fem_p95': fem_p95,
        'surr_mean': surr_mean,
        'surr_std': surr_std,
        'surr_p05': surr_p05,
        'surr_p95': surr_p95
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    config = Config()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plots_dir = config.OUT_DIR / "validation_plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SURROGATE VALIDATION AGAINST FEM")
    print("=" * 80)
    print(f"\nSurrogate type: {config.SURROGATE_TYPE}")
    print(f"Validation samples: {config.N_VALIDATION_SAMPLES}")
    
    # --------------------------------------------------------
    # Load surrogate and data
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("LOADING SURROGATE AND DATA")
    print("-" * 80)
    
    model = load_best_surrogate(config)
    print("✓ Surrogate loaded")
    
    df_load, df_damage, df_uq, val_jobs = load_fem_data(config)
    print(f"✓ Selected {len(val_jobs)} validation samples")
    
    # Get grids from model
    if hasattr(model, 'u_force'):
        u_grid_force = model.u_force
        u_grid_damage = model.u_damage
    else:
        # Fallback
        u_grid_force = np.linspace(0, 20, 400)
        u_grid_damage = np.linspace(0, 20, 400)
    
    # --------------------------------------------------------
    # Run Monte Carlo comparison
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("RUNNING MONTE CARLO COMPARISON")
    print("-" * 80)
    
    fem_force_curves = []
    surr_force_curves = []
    fem_damage_curves = []
    surr_damage_curves = []
    
    successful = 0
    failed = []
    
    for idx, job in enumerate(val_jobs):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(val_jobs)}")
        
        try:
            # Get FEM curves
            f_fem, d_fem = extract_fem_curves(
                df_load, df_damage, job, config.DAMAGE_VAR,
                u_grid_force, u_grid_damage
            )
            
            # Get inputs
            if job not in df_uq.index:
                failed.append(job)
                continue
            
            fc = df_uq.loc[job, "fc"]
            E = df_uq.loc[job, "E"]
            cbot = df_uq.loc[job, "c_nom_bottom_mm"]
            ctop = df_uq.loc[job, "c_nom_top_mm"]
            
            # Get surrogate prediction using unified interface
            _, F_surr, _, D_surr = predict_unified(
                model, config.SURROGATE_TYPE, fc, E, cbot, ctop, job=job
            )
            
            fem_force_curves.append(f_fem)
            surr_force_curves.append(F_surr)
            fem_damage_curves.append(d_fem)
            surr_damage_curves.append(D_surr)
            
            successful += 1
            
        except Exception as e:
            print(f"  Warning: Failed on {job}: {e}")
            failed.append(job)
            continue
    
    print(f"\n✓ Successful: {successful}/{len(val_jobs)}")
    if failed:
        print(f"  Failed samples: {len(failed)}")
    
    # Convert to arrays
    fem_force_curves = np.array(fem_force_curves)
    surr_force_curves = np.array(surr_force_curves)
    fem_damage_curves = np.array(fem_damage_curves)
    surr_damage_curves = np.array(surr_damage_curves)
    
    # --------------------------------------------------------
    # Statistical comparison
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("STATISTICAL COMPARISON")
    print("-" * 80)
    
    force_stats = compute_statistical_agreement(fem_force_curves, surr_force_curves)
    damage_stats = compute_statistical_agreement(fem_damage_curves, surr_damage_curves)
    
    print("\nForce curves:")
    print(f"  Mean error: {force_stats['mean_error']:.3f}")
    print(f"  Max mean error: {force_stats['max_mean_error']:.3f}")
    print(f"  Confidence band overlap: {force_stats['confidence_overlap']*100:.1f}%")
    print(f"  KS test p-value: {force_stats['ks_pvalue']:.4f}")
    
    print("\nDamage curves:")
    print(f"  Mean error: {damage_stats['mean_error']:.5f}")
    print(f"  Max mean error: {damage_stats['max_mean_error']:.5f}")
    print(f"  Confidence band overlap: {damage_stats['confidence_overlap']*100:.1f}%")
    print(f"  KS test p-value: {damage_stats['ks_pvalue']:.4f}")
    
    # --------------------------------------------------------
    # Validation plots
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("GENERATING VALIDATION PLOTS")
    print("-" * 80)
    
    # Plot 1: Force curves with confidence bands
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(u_grid_force, force_stats['fem_mean'], 'k-', linewidth=2.5, label='FEM Mean')
    ax.fill_between(u_grid_force, force_stats['fem_p05'], force_stats['fem_p95'],
                     color='gray', alpha=0.3, label='FEM 5-95% CI')
    
    ax.plot(u_grid_force, force_stats['surr_mean'], 'r--', linewidth=2.5, label='Surrogate Mean')
    ax.fill_between(u_grid_force, force_stats['surr_p05'], force_stats['surr_p95'],
                     color='red', alpha=0.2, label='Surrogate 5-95% CI')
    
    ax.set_xlabel('Displacement [mm]', fontsize=12)
    ax.set_ylabel('Force [N]', fontsize=12)
    ax.set_title(f'Force-Displacement Validation (n={successful})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "01_force_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 01_force_validation.png")
    
    # Plot 2: Damage curves with confidence bands
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(u_grid_damage, damage_stats['fem_mean'], 'k-', linewidth=2.5, label='FEM Mean')
    ax.fill_between(u_grid_damage, damage_stats['fem_p05'], damage_stats['fem_p95'],
                     color='gray', alpha=0.3, label='FEM 5-95% CI')
    
    ax.plot(u_grid_damage, damage_stats['surr_mean'], 'r--', linewidth=2.5, label='Surrogate Mean')
    ax.fill_between(u_grid_damage, damage_stats['surr_p05'], damage_stats['surr_p95'],
                     color='red', alpha=0.2, label='Surrogate 5-95% CI')
    
    ax.set_xlabel('Displacement [mm]', fontsize=12)
    ax.set_ylabel('Tension Damage [-]', fontsize=12)
    ax.set_title(f'Damage Evolution Validation (n={successful})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "02_damage_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_damage_validation.png")
    
    # Plot 3: Quantile-Quantile plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Force Q-Q at peak
    peak_idx_force = np.argmax(force_stats['fem_mean'])
    fem_peak_vals = fem_force_curves[:, peak_idx_force]
    surr_peak_vals = surr_force_curves[:, peak_idx_force]
    
    ax1.scatter(np.sort(fem_peak_vals), np.sort(surr_peak_vals), alpha=0.6, s=50)
    lims = [min(fem_peak_vals.min(), surr_peak_vals.min()),
            max(fem_peak_vals.max(), surr_peak_vals.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('FEM Peak Force [N]', fontsize=12)
    ax1.set_ylabel('Surrogate Peak Force [N]', fontsize=12)
    ax1.set_title('Q-Q Plot: Peak Force', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Damage Q-Q at end
    fem_end_vals = fem_damage_curves[:, -1]
    surr_end_vals = surr_damage_curves[:, -1]
    
    ax2.scatter(np.sort(fem_end_vals), np.sort(surr_end_vals), alpha=0.6, s=50, color='purple')
    lims = [min(fem_end_vals.min(), surr_end_vals.min()),
            max(fem_end_vals.max(), surr_end_vals.max())]
    ax2.plot(lims, lims, 'r--', linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('FEM Final Damage [-]', fontsize=12)
    ax2.set_ylabel('Surrogate Final Damage [-]', fontsize=12)
    ax2.set_title('Q-Q Plot: Final Damage', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "03_qq_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_qq_plots.png")
    
    # Plot 4: Error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Force errors
    force_errors = np.mean(np.abs(fem_force_curves - surr_force_curves), axis=1)
    ax1.hist(force_errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(force_errors.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {force_errors.mean():.2f}')
    ax1.set_xlabel('Mean Absolute Error [N]', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Force Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Damage errors
    damage_errors = np.mean(np.abs(fem_damage_curves - surr_damage_curves), axis=1)
    ax2.hist(damage_errors, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(damage_errors.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {damage_errors.mean():.5f}')
    ax2.set_xlabel('Mean Absolute Error [-]', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Damage Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "04_error_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_error_distributions.png")
    
    # --------------------------------------------------------
    # Fidelity assessment
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("FIDELITY ASSESSMENT")
    print("=" * 80)
    
    # Criteria for surrogate validation
    force_criteria = {
        'mean_error < 5% of peak': bool(force_stats['mean_error'] < 0.05 * np.max(force_stats['fem_mean'])),
        'CI overlap > 80%': bool(force_stats['confidence_overlap'] > 0.80),
        'KS test p > 0.05': bool(force_stats['ks_pvalue'] > 0.05)
    }

    damage_criteria = {
        'mean_error < 5% of max': bool(damage_stats['mean_error'] < 0.05 * np.max(damage_stats['fem_mean'])),
        'CI overlap > 80%': bool(damage_stats['confidence_overlap'] > 0.80),
        'KS test p > 0.05': bool(damage_stats['ks_pvalue'] > 0.05)
    }

    
    print("\nForce validation criteria:")
    for criterion, passed in force_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    print("\nDamage validation criteria:")
    for criterion, passed in damage_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    # Overall recommendation
    force_passed = sum(force_criteria.values()) >= 2
    damage_passed = sum(damage_criteria.values()) >= 2
    
    if force_passed and damage_passed:
        recommendation = "APPROVED for large-scale UQ"
        confidence = "HIGH"
    elif force_passed or damage_passed:
        recommendation = "CONDITIONALLY APPROVED - Review failed criteria"
        confidence = "MEDIUM"
    else:
        recommendation = "NOT APPROVED - Significant discrepancies detected"
        confidence = "LOW"
    
    print(f"\n{'=' * 80}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"CONFIDENCE LEVEL: {confidence}")
    print("=" * 80)
    
    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    results = {
        'surrogate_type': config.SURROGATE_TYPE,
        'n_samples': successful,
        'force_statistics': {k: v for k, v in force_stats.items() 
                            if not isinstance(v, np.ndarray)},
        'damage_statistics': {k: v for k, v in damage_stats.items()
                             if not isinstance(v, np.ndarray)},
        'force_criteria': force_criteria,
        'damage_criteria': damage_criteria,
        'recommendation': recommendation,
        'confidence': confidence
    }
    
    with open(config.OUT_DIR / "validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {config.OUT_DIR}")
    print(f"✓ Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()