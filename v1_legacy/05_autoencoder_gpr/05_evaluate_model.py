#!/usr/bin/env python3
"""
AE + GPR Pipeline - Step 5: Comprehensive Model Evaluation
===========================================================
Evaluate the complete AE+GPR surrogate on the test set.
Compute reconstruction metrics, latent smoothness, and create visualizations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import pdist, squareform
import json

from ae_surrogate_model import ImprovedAESurrogateModel

plt.rcParams["font.family"] = "Times New Roman"


def compute_curve_metrics(y_true, y_pred, curve_name=""):
    """Compute comprehensive metrics for curve reconstruction."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Peak error
    peak_true = np.max(np.abs(y_true))
    peak_pred = np.max(np.abs(y_pred))
    peak_error = abs(peak_pred - peak_true)
    peak_relative_error = peak_error / (peak_true + 1e-10)
    
    return {
        f"{curve_name}_rmse": rmse,
        f"{curve_name}_mae": mae,
        f"{curve_name}_r2": r2,
        f"{curve_name}_peak_true": peak_true,
        f"{curve_name}_peak_pred": peak_pred,
        f"{curve_name}_peak_error": peak_error,
        f"{curve_name}_peak_relative_error": peak_relative_error,
    }


def analyze_latent_smoothness(X_test_scaled, Z_test, curve_name=""):
    """
    Analyze latent space smoothness:
    Similar inputs should produce similar latent representations.
    """
    # Compute pairwise distances in input space
    input_distances = squareform(pdist(X_test_scaled, metric='euclidean'))
    
    # Compute pairwise distances in latent space
    latent_distances = squareform(pdist(Z_test, metric='euclidean'))
    
    # Flatten upper triangular (exclude diagonal)
    n = len(X_test_scaled)
    triu_indices = np.triu_indices(n, k=1)
    
    input_dist_flat = input_distances[triu_indices]
    latent_dist_flat = latent_distances[triu_indices]
    
    # Compute correlation (smoothness metric)
    correlation = np.corrcoef(input_dist_flat, latent_dist_flat)[0, 1]
    
    # Compute local smoothness (ratio of close points)
    # For pairs with input distance < threshold, check latent distance
    threshold_percentile = 10  # 10th percentile
    input_threshold = np.percentile(input_dist_flat, threshold_percentile)
    
    close_pairs_mask = input_dist_flat < input_threshold
    close_latent_distances = latent_dist_flat[close_pairs_mask]
    
    latent_smoothness = 1.0 / (1.0 + close_latent_distances.std())
    
    return {
        f"{curve_name}_latent_input_correlation": correlation,
        f"{curve_name}_latent_smoothness": latent_smoothness,
        f"{curve_name}_avg_latent_distance": latent_dist_flat.mean(),
        f"{curve_name}_std_latent_distance": latent_dist_flat.std(),
    }


def plot_reconstruction_comparison(u_grid, y_true_all, y_pred_all, 
                                   curve_name, ylabel, out_dir, n_samples=5):
    """Plot random sample comparisons."""
    n_samples = min(n_samples, len(y_true_all))
    indices = np.random.choice(len(y_true_all), n_samples, replace=False)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(indices, axes):
        ax.plot(u_grid, y_true_all[idx], 'k-', linewidth=2, label='FEM (Test)')
        ax.plot(u_grid, y_pred_all[idx], 'r--', linewidth=2, label='Surrogate')
        ax.set_xlabel('Displacement (mm)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Sample {idx}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.suptitle(f'{curve_name} Reconstruction: Random Test Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"{curve_name.lower().replace(' ', '_')}_samples.png", dpi=300)
    plt.close()


def plot_error_distribution(errors, curve_name, out_dir):
    """Plot distribution of reconstruction errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.6f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.6f}')
    ax.set_xlabel('RMSE per Sample', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{curve_name}: Error Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot(errors, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('RMSE per Sample', fontsize=11)
    ax.set_title(f'{curve_name}: Error Statistics', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{curve_name.lower().replace(' ', '_')}_error_dist.png", dpi=300)
    plt.close()


def plot_latent_smoothness(X_test_scaled, Z_test, curve_name, out_dir):
    """Visualize latent space smoothness via scatter plot."""
    input_distances = squareform(pdist(X_test_scaled, metric='euclidean'))
    latent_distances = squareform(pdist(Z_test, metric='euclidean'))
    
    n = len(X_test_scaled)
    triu_indices = np.triu_indices(n, k=1)
    
    input_dist_flat = input_distances[triu_indices]
    latent_dist_flat = latent_distances[triu_indices]
    
    # Sample points for visualization
    sample_size = min(5000, len(input_dist_flat))
    sample_idx = np.random.choice(len(input_dist_flat), sample_size, replace=False)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(input_dist_flat[sample_idx], latent_dist_flat[sample_idx], 
                alpha=0.3, s=10, c='steelblue')
    
    # Fit and plot trend line
    z = np.polyfit(input_dist_flat[sample_idx], latent_dist_flat[sample_idx], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(input_dist_flat[sample_idx].min(), input_dist_flat[sample_idx].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Linear fit')
    
    corr = np.corrcoef(input_dist_flat, latent_dist_flat)[0, 1]
    plt.xlabel('Input Space Distance', fontsize=12)
    plt.ylabel('Latent Space Distance', fontsize=12)
    plt.title(f'{curve_name}: Latent Space Smoothness\n(Correlation: {corr:.4f})', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{curve_name.lower().replace(' ', '_')}_latent_smoothness.png", dpi=300)
    plt.close()


def main():
    base = Path(
        r"."
    )

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------
    ae_out = base / "05_autoencoder_gpr" / "data_preprocessed"
    data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
    gpr_out = base / "05_autoencoder_gpr" / "output_surrogates"
    eval_dir = base / "05_autoencoder_gpr" / "output_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load split indices
    test_idx = np.load(ae_out / "test_indices.npy")
    print(f"Evaluating on {len(test_idx)} test samples\n")

    # Load ground truth data
    u_force = np.load(data_dir / "u_force.npy")
    u_damage = np.load(data_dir / "u_crack.npy")
    F_norm_all = np.load(data_dir / "F_norm_all.npy")
    C_norm_all = np.load(data_dir / "C_norm_all.npy")
    global_Fmax = float(np.load(data_dir / "F_global_max.npy"))
    C_max_all = np.load(data_dir / "C_max.npy")

    # Load inputs
    uq_csv = base / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"
    df_uq = pd.read_csv(uq_csv)

    # Load test latent vectors and scaled inputs (for smoothness analysis)
    X_test_scaled = np.load(gpr_out / "X_test.npy")
    from sklearn.preprocessing import StandardScaler
    import joblib
    scaler = joblib.load(gpr_out / "input_scaler.joblib")
    X_test_scaled = scaler.transform(X_test_scaled)
    
    Z_force_test = np.load(gpr_out / "Z_force_test.npy")
    Z_damage_test = np.load(gpr_out / "Z_damage_test.npy")

    # --------------------------------------------------------
    # LOAD SURROGATE MODEL
    # --------------------------------------------------------
    print("Loading surrogate model...")
    model = ImprovedAESurrogateModel(str(base))

    # --------------------------------------------------------
    # EVALUATE ON TEST SET
    # --------------------------------------------------------
    print("Running predictions on test set...")
    
    F_true_all = []
    F_pred_all = []
    C_true_all = []
    C_pred_all = []
    
    sample_metrics = []

    for idx in test_idx:
        # Get inputs
        cbot = float(df_uq.iloc[idx]["c_nom_bottom_mm"])
        ctop = float(df_uq.iloc[idx]["c_nom_top_mm"])
        fcm = float(df_uq.iloc[idx]["fc"])

        # Get FEM truth (un-normalized)
        F_true_norm = F_norm_all[idx]
        C_true_norm = C_norm_all[idx]
        F_true = F_true_norm * global_Fmax
        C_true = C_true_norm * C_max_all[idx]

        # Surrogate prediction
        uF, F_pred_norm, uC, C_pred_norm = model.predict(cbot=cbot, ctop=ctop, fcm=fcm)
        F_pred = F_pred_norm * global_Fmax
        C_pred = C_pred_norm * C_max_all[idx]

        F_true_all.append(F_true)
        F_pred_all.append(F_pred)
        C_true_all.append(C_true)
        C_pred_all.append(C_pred)

        # Compute per-sample metrics
        metrics = {
            "index": idx,
            "cbot": cbot,
            "ctop": ctop,
            "fcm": fcm,
        }
        metrics.update(compute_curve_metrics(F_true, F_pred, "force"))
        metrics.update(compute_curve_metrics(C_true, C_pred, "damage"))
        sample_metrics.append(metrics)

    F_true_all = np.array(F_true_all)
    F_pred_all = np.array(F_pred_all)
    C_true_all = np.array(C_true_all)
    C_pred_all = np.array(C_pred_all)

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------
    print("\nComputing aggregate metrics...")
    
    df_metrics = pd.DataFrame(sample_metrics)
    
    aggregate_results = {
        "test_samples": len(test_idx),
        
        "force_metrics": {
            "rmse_mean": float(df_metrics["force_rmse"].mean()),
            "rmse_std": float(df_metrics["force_rmse"].std()),
            "mae_mean": float(df_metrics["force_mae"].mean()),
            "r2_mean": float(df_metrics["force_r2"].mean()),
            "peak_error_mean": float(df_metrics["force_peak_error"].mean()),
            "peak_relative_error_mean": float(df_metrics["force_peak_relative_error"].mean()),
        },
        
        "damage_metrics": {
            "rmse_mean": float(df_metrics["damage_rmse"].mean()),
            "rmse_std": float(df_metrics["damage_rmse"].std()),
            "mae_mean": float(df_metrics["damage_mae"].mean()),
            "r2_mean": float(df_metrics["damage_r2"].mean()),
            "peak_error_mean": float(df_metrics["damage_peak_error"].mean()),
            "peak_relative_error_mean": float(df_metrics["damage_peak_relative_error"].mean()),
        }
    }

    # Latent smoothness analysis
    print("Analyzing latent space smoothness...")
    force_smoothness = analyze_latent_smoothness(X_test_scaled, Z_force_test, "force")
    damage_smoothness = analyze_latent_smoothness(X_test_scaled, Z_damage_test, "damage")
    
    aggregate_results["force_latent_smoothness"] = force_smoothness
    aggregate_results["damage_latent_smoothness"] = damage_smoothness

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    df_metrics.to_csv(eval_dir / "test_sample_metrics.csv", index=False)
    
    with open(eval_dir / "test_aggregate_metrics.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)

    # --------------------------------------------------------
    # GENERATE PLOTS
    # --------------------------------------------------------
    print("Generating evaluation plots...")
    
    # Sample reconstructions
    plot_reconstruction_comparison(
        u_force, F_true_all, F_pred_all,
        "Force Curve", "Force (N)", eval_dir, n_samples=5
    )
    
    plot_reconstruction_comparison(
        u_damage, C_true_all, C_pred_all,
        "Damage Curve", "Compression Damage", eval_dir, n_samples=5
    )

    # Error distributions
    plot_error_distribution(
        df_metrics["force_rmse"].values,
        "Force Curve", eval_dir
    )
    
    plot_error_distribution(
        df_metrics["damage_rmse"].values,
        "Damage Curve", eval_dir
    )

    # Latent smoothness
    plot_latent_smoothness(X_test_scaled, Z_force_test, "Force", eval_dir)
    plot_latent_smoothness(X_test_scaled, Z_damage_test, "Damage", eval_dir)

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"\nTest samples: {aggregate_results['test_samples']}")
    
    print(f"\n{'FORCE CURVE METRICS':^70}")
    print("-"*70)
    fm = aggregate_results['force_metrics']
    print(f"  RMSE:                {fm['rmse_mean']:.6f} ± {fm['rmse_std']:.6f}")
    print(f"  MAE:                 {fm['mae_mean']:.6f}")
    print(f"  R²:                  {fm['r2_mean']:.6f}")
    print(f"  Peak Error:          {fm['peak_error_mean']:.6f} N")
    print(f"  Peak Relative Error: {fm['peak_relative_error_mean']*100:.4f}%")
    
    print(f"\n{'DAMAGE CURVE METRICS':^70}")
    print("-"*70)
    dm = aggregate_results['damage_metrics']
    print(f"  RMSE:                {dm['rmse_mean']:.6f} ± {dm['rmse_std']:.6f}")
    print(f"  MAE:                 {dm['mae_mean']:.6f}")
    print(f"  R²:                  {dm['r2_mean']:.6f}")
    print(f"  Peak Error:          {dm['peak_error_mean']:.6f}")
    print(f"  Peak Relative Error: {dm['peak_relative_error_mean']*100:.4f}%")
    
    print(f"\n{'LATENT SPACE SMOOTHNESS':^70}")
    print("-"*70)
    print(f"  Force Latent-Input Correlation:  {force_smoothness['force_latent_input_correlation']:.6f}")
    print(f"  Damage Latent-Input Correlation: {damage_smoothness['damage_latent_input_correlation']:.6f}")
    
    print("="*70)
    print(f"\nAll results saved to: {eval_dir}")
    print("  - test_sample_metrics.csv")
    print("  - test_aggregate_metrics.json")
    print("  - Various visualization plots\n")


if __name__ == "__main__":
    main()