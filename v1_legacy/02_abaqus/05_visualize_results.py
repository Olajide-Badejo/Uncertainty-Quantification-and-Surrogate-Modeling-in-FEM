"""
===============================================================================
STEP 2.5: VISUALIZE FEM RESULTS
===============================================================================
Purpose: Create comprehensive plots of extracted FEM data

Plots generated:
    1. Load-displacement curves (individual and overlaid)
    2. Damage evolution curves
    3. Statistical distributions of key metrics
    4. Correlation between input parameters and outputs
    5. Outlier identification plots

Output: High-quality plots saved to visualization directory
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set professional style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_ROOT = ROOT / "02_abaqus" / "extracted_data"
LHS_CSV = ROOT / "uq_lhs_samples_training.csv"
VIZ_DIR = ROOT / "02_abaqus" / "fem_visualizations"

VIZ_DIR.mkdir(exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================

def load_all_load_displacement():
    """
    Load all load-displacement curves.
    
    Returns:
        dict: {sample_id: DataFrame}
    """
    data = {}
    
    ld_files = sorted(EXTRACTED_ROOT.glob("sample_*_load_displacement.csv"))
    
    for ld_file in ld_files:
        sample_id = int(ld_file.stem.split("_")[1])
        try:
            df = pd.read_csv(ld_file)
            data[sample_id] = df
        except:
            pass
    
    return data


def load_all_damage():
    """
    Load all damage evolution data.
    
    Returns:
        dict: {sample_id: DataFrame}
    """
    data = {}
    
    damage_files = sorted(EXTRACTED_ROOT.glob("sample_*_damage.csv"))
    
    for damage_file in damage_files:
        sample_id = int(damage_file.stem.split("_")[1])
        try:
            df = pd.read_csv(damage_file)
            data[sample_id] = df
        except:
            pass
    
    return data


def compute_summary_metrics(ld_data, damage_data):
    """
    Compute summary metrics for each sample.
    
    Returns:
        DataFrame with columns: sample_id, max_load, max_disp, 
                                final_damagec, final_damaget, etc.
    """
    metrics = []
    
    for sample_id in ld_data.keys():
        
        ld_df = ld_data[sample_id]
        
        metric = {
            'sample_id': sample_id,
            'max_load': ld_df['reaction_force'].max(),
            'max_displacement': ld_df['displacement'].max(),
            'n_points': len(ld_df)
        }
        
        # Add damage metrics if available
        if sample_id in damage_data:
            damage_df = damage_data[sample_id]
            metric['final_damagec'] = damage_df['damagec_max'].values[-1]
            metric['final_damaget'] = damage_df['damaget_max'].values[-1]
            metric['final_sdeg'] = damage_df['sdeg_max'].values[-1]
        
        metrics.append(metric)
    
    return pd.DataFrame(metrics)


# ============================================================
# PLOT 1: OVERLAY ALL LOAD-DISPLACEMENT CURVES
# ============================================================

def plot_all_load_displacement_curves(ld_data):
    """
    Plot all load-displacement curves on one figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for sample_id, df in ld_data.items():
        ax.plot(df['displacement'], df['reaction_force'], 
                alpha=0.3, linewidth=0.8, label=f'Sample {sample_id}' if sample_id < 5 else '')
    
    ax.set_xlabel('Displacement [mm]')
    ax.set_ylabel('Reaction Force [N]')
    ax.set_title('Load-Displacement Curves (All Samples)')
    ax.grid(True, alpha=0.3)
    
    # Add mean curve
    if len(ld_data) > 0:
        # Interpolate all curves to common displacement grid
        max_disp = max(df['displacement'].max() for df in ld_data.values())
        common_disp = np.linspace(0, max_disp, 200)
        
        interpolated = []
        for df in ld_data.values():
            if len(df) > 5:  # Only use curves with enough points
                rf_interp = np.interp(common_disp, df['displacement'], df['reaction_force'])
                interpolated.append(rf_interp)
        
        if interpolated:
            mean_rf = np.mean(interpolated, axis=0)
            std_rf = np.std(interpolated, axis=0)
            
            ax.plot(common_disp, mean_rf, 'r-', linewidth=2, label='Mean')
            ax.fill_between(common_disp, mean_rf - std_rf, mean_rf + std_rf, 
                            color='red', alpha=0.2, label='±1 std')
    
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '01_all_load_displacement_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 01_all_load_displacement_curves.png")


# ============================================================
# PLOT 2: INDIVIDUAL LOAD-DISPLACEMENT (SELECTED SAMPLES)
# ============================================================

def plot_selected_load_displacement(ld_data, n_samples=12):
    """
    Plot individual load-displacement curves in a grid.
    """
    sample_ids = sorted(list(ld_data.keys()))[:n_samples]
    
    n_rows = 3
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, sample_id in enumerate(sample_ids):
        if i >= len(axes):
            break
        
        df = ld_data[sample_id]
        
        axes[i].plot(df['displacement'], df['reaction_force'], 'b-', linewidth=1.5)
        axes[i].set_title(f'Sample {sample_id:03d}', fontsize=10)
        axes[i].set_xlabel('Displacement [mm]', fontsize=8)
        axes[i].set_ylabel('Reaction Force [N]', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(sample_ids), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Load-Displacement Curves (Selected Samples)', fontsize=14)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '02_selected_load_displacement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 02_selected_load_displacement.png")


# ============================================================
# PLOT 3: DAMAGE EVOLUTION CURVES
# ============================================================

def plot_damage_evolution(damage_data, n_samples=12):
    """
    Plot damage evolution for selected samples.
    """
    sample_ids = sorted(list(damage_data.keys()))[:n_samples]
    
    n_rows = 3
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, sample_id in enumerate(sample_ids):
        if i >= len(axes):
            break
        
        df = damage_data[sample_id]
        
        axes[i].plot(df['time'], df['damagec_max'], 'r-', linewidth=1.2, label='Compression')
        axes[i].plot(df['time'], df['damaget_max'], 'b-', linewidth=1.2, label='Tension')
        axes[i].set_title(f'Sample {sample_id:03d}', fontsize=10)
        axes[i].set_xlabel('Time', fontsize=8)
        axes[i].set_ylabel('Max Damage [-]', fontsize=8)
        axes[i].set_ylim([0, 1])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(fontsize=7)
    
    # Hide unused subplots
    for i in range(len(sample_ids), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Damage Evolution (Selected Samples)', fontsize=14)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '03_damage_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 03_damage_evolution.png")


# ============================================================
# PLOT 4: DISTRIBUTIONS OF KEY METRICS
# ============================================================

def plot_metric_distributions(metrics_df):
    """
    Plot histograms of key output metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('max_load', 'Maximum Load [N]'),
        ('max_displacement', 'Maximum Displacement [mm]'),
        ('final_damagec', 'Final Compression Damage [-]'),
        ('final_damaget', 'Final Tension Damage [-]'),
        ('final_sdeg', 'Final Stiffness Degradation [-]'),
        ('n_points', 'Number of Data Points')
    ]
    
    for i, (metric, label) in enumerate(metrics_to_plot):
        if metric in metrics_df.columns:
            data = metrics_df[metric].dropna()
            
            axes[i].hist(data, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {label}')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean and std
            mean_val = data.mean()
            std_val = data.std()
            axes[i].axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '04_metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 04_metric_distributions.png")


# ============================================================
# PLOT 5: INPUT-OUTPUT CORRELATIONS
# ============================================================

def plot_input_output_correlations(metrics_df, lhs_df):
    """
    Plot correlations between input parameters and output metrics.
    """
    # Merge input and output data
    merged = metrics_df.merge(lhs_df, on='sample_id', how='inner')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    input_params = ['Fcm_MPa', 'c_nom_bottom_mm', 'c_nom_top_mm']
    output_metrics = ['max_load', 'max_displacement']
    
    plot_idx = 0
    for output in output_metrics:
        for input_param in input_params:
            if plot_idx < len(axes) and output in merged.columns and input_param in merged.columns:
                
                x = merged[input_param]
                y = merged[output]
                
                axes[plot_idx].scatter(x, y, alpha=0.6, edgecolor='k', s=30)
                axes[plot_idx].set_xlabel(input_param)
                axes[plot_idx].set_ylabel(output)
                axes[plot_idx].set_title(f'{output} vs {input_param}')
                axes[plot_idx].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(x, y)[0, 1]
                axes[plot_idx].text(0.05, 0.95, f'ρ = {corr:.3f}', 
                                   transform=axes[plot_idx].transAxes, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '05_input_output_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 05_input_output_correlations.png")


# ============================================================
# PLOT 6: OUTLIER IDENTIFICATION
# ============================================================

def plot_outliers(metrics_df):
    """
    Identify and visualize outliers in key metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plots
    metrics_to_plot = ['max_load', 'max_displacement']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics_df.columns:
            data = metrics_df[metric].dropna()
            
            axes[i].boxplot(data, vert=True)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'Outlier Detection: {metric}')
            axes[i].grid(True, alpha=0.3)
            
            # Mark outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            
            axes[i].text(0.5, 0.95, f'{len(outliers)} outliers detected', 
                        transform=axes[i].transAxes, 
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '06_outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: 06_outlier_detection.png")


# ============================================================
# MAIN VISUALIZATION ROUTINE
# ============================================================

def main():
    
    print("=" * 70)
    print("GENERATING FEM RESULT VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    ld_data = load_all_load_displacement()
    damage_data = load_all_damage()
    
    print(f"  ✓ Loaded {len(ld_data)} load-displacement curves")
    print(f"  ✓ Loaded {len(damage_data)} damage datasets")
    
    if len(ld_data) == 0:
        print("\n⚠ No data found! Run extraction first.")
        return
    
    # Compute metrics
    print("\nComputing summary metrics...")
    metrics_df = compute_summary_metrics(ld_data, damage_data)
    print(f"  ✓ Computed metrics for {len(metrics_df)} samples")
    
    # Load LHS data for correlations
    try:
        lhs_df = pd.read_csv(LHS_CSV)
    except:
        lhs_df = pd.DataFrame()
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_all_load_displacement_curves(ld_data)
    plot_selected_load_displacement(ld_data)
    plot_damage_evolution(damage_data)
    plot_metric_distributions(metrics_df)
    
    if not lhs_df.empty:
        plot_input_output_correlations(metrics_df, lhs_df)
    
    plot_outliers(metrics_df)
    
    # Save metrics summary
    metrics_file = VIZ_DIR / 'summary_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  ✓ Saved summary metrics to: {metrics_file}")
    
    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {VIZ_DIR}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
