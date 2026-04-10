# """
# ===============================================================================
# STEP 5: SURROGATE MODEL COMPARISON & SELECTION
# ===============================================================================
# Comprehensive comparison of three surrogate modeling approaches:
# 1. PCA + GPR
# 2. Autoencoder (AE) + GPR  
# 3. Shape-Scale PCA + GPR

# Metrics:
# - Train/Val/Test RMSE
# - Reconstruction error distributions
# - Computational efficiency (prediction time, memory)
# - Physical interpretability scores
# - Peak prediction accuracy
# - R² scores

# Output:
# - Comprehensive comparison report
# - Publication-ready comparison plots
# - Model selection recommendation
# ===============================================================================
# """

# import json
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import Dict, Tuple, List
# import warnings
# warnings.filterwarnings('ignore')

# # Set plotting style
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 10

# # ============================================================
# # CONFIGURATION
# # ============================================================

# class Config:
#     """Configuration for surrogate comparison."""
    
#     BASE = Path(__file__).resolve().parents[1]
    
#     # Data paths
#     LOAD_CSV = BASE / "augmentation_physics_fixed" / "load_displacement_full_aug.csv"
#     DAMAGE_CSV = BASE / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv"
#     UQ_CSV = BASE / "augmentation_physics_fixed" / "processed_inputs_2_aug.csv"
    
#     # Surrogate directories
#     PCA_DIR = BASE / "04_PCA"
#     AE_DIR = BASE / "05_autoencoder_gpr"
#     SHAPE_SCALE_DIR = BASE / "06_shape_scale_gpr"
    
#     # Output directory
#     OUT_DIR = BASE / "07_processing"/ "06_surrogate_comparison"
    
#     # Damage variable to use
#     DAMAGE_VAR = 'DAMAGEC_max'  # or 'DAMAGEC_max' for compression

# # ============================================================
# # SURROGATE LOADERS
# # ============================================================

# def load_pca_surrogate(config: Config):
#     """Load PCA+GPR surrogate."""
#     try:
#         # Import must be relative to where the file exists
#         import sys
#         sys.path.insert(0, str(config.PCA_DIR / "01_pca_reduction"))
#         from surrogate_model import SurrogateModel
        
#         model = SurrogateModel.load(
#             pca_dir=config.PCA_DIR / "01_pca_reduction" / "models",
#             surrogate_dir=config.PCA_DIR / "01_pca_reduction" / "outputs"
#         )
#         return model, "PCA+GPR"
#     except Exception as e:
#         print(f"Warning: Could not load PCA+GPR surrogate: {e}")
#         return None, "PCA+GPR"


# def load_ae_surrogate(config: Config):
#     """Load Autoencoder+GPR surrogate."""
#     try:
#         import sys
#         sys.path.insert(0, str(config.AE_DIR))
#         from ae_surrogate_model import ImprovedAESurrogateModel
        
#         model = ImprovedAESurrogateModel(str(config.BASE), use_improved=True)
#         return model, "AE+GPR"
#     except Exception as e:
#         print(f"Warning: Could not load AE+GPR surrogate: {e}")
#         return None, "AE+GPR"


# def load_shape_scale_surrogate(config: Config):
#     """Load Shape-Scale PCA+GPR surrogate."""
#     try:
#         import sys
#         sys.path.insert(0, str(config.SHAPE_SCALE_DIR))
#         from shape_scale_surrogate import ShapeScaleSurrogate
        
#         model = ShapeScaleSurrogate.load(config.SHAPE_SCALE_DIR)
#         return model, "Shape-Scale PCA+GPR"
#     except Exception as e:
#         print(f"Warning: Could not load Shape-Scale surrogate: {e}")
#         return None, "Shape-Scale PCA+GPR"


# # ============================================================
# # UNIFIED PREDICTION INTERFACE
# # ============================================================

# class UnifiedSurrogate:
#     """Unified interface for all surrogate types."""
    
#     def __init__(self, model, model_type: str):
#         self.model = model
#         self.model_type = model_type
    
#     def predict(self, fc: float, E: float, cbot: float, ctop: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Unified prediction interface.
        
#         Returns:
#             u_force, F_pred, u_damage, D_pred
#         """
#         if self.model_type == "PCA+GPR":
#             # PCA model returns normalized predictions
#             F_pred, D_pred = self.model.predict_curves(
#                 fc=fc, E=E, cbot=cbot, ctop=ctop, return_uncertainty=False
#             )
#             u_force = self.model.u_grid_force
#             u_damage = self.model.u_grid_damage
#             return u_force, F_pred, u_damage, D_pred
            
#         elif self.model_type == "AE+GPR":
#             # AE model returns normalized predictions
#             u_force, F_pred, u_damage, D_pred = self.model.predict(
#                 cbot=cbot, ctop=ctop, fcm=fc
#             )
#             if hasattr(self.model, "force_scaler"):
#                 F_pred = self.model.force_scaler.inverse_transform(F_pred.reshape(-1,1)).ravel()
#             return u_force, F_pred, u_damage, D_pred
            
#         elif self.model_type == "Shape-Scale PCA+GPR":
#             # Shape-scale returns actual values
#             u_force, F_pred, u_damage, D_pred = self.model.predict_curves(
#                 fc=fc, E=E, cbot=cbot, ctop=ctop
#             )
#             return u_force, F_pred, u_damage, D_pred
        
#         else:
#             raise ValueError(f"Unknown model type: {self.model_type}")
    
#     def get_memory_usage(self) -> float:
#         """Estimate memory usage in MB."""
#         import sys
#         return sys.getsizeof(self.model) / 1024 / 1024


# # ============================================================
# # METRICS COMPUTATION
# # ============================================================

# def compute_reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
#     """Compute comprehensive reconstruction metrics."""
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     # Peak metrics
#     peak_true = np.max(np.abs(y_true))
#     peak_pred = np.max(np.abs(y_pred))
#     peak_error = abs(peak_pred - peak_true)
#     peak_rel_error = peak_error / (peak_true + 1e-10) * 100
    
#     # Pointwise errors
#     pointwise_errors = np.abs(y_true - y_pred)
#     max_pointwise = np.max(pointwise_errors)
#     mean_pointwise = np.mean(pointwise_errors)
    
#     return {
#         'rmse': rmse,
#         'mae': mae,
#         'r2': r2,
#         'peak_true': peak_true,
#         'peak_pred': peak_pred,
#         'peak_error': peak_error,
#         'peak_rel_error': peak_rel_error,
#         'max_pointwise_error': max_pointwise,
#         'mean_pointwise_error': mean_pointwise
#     }


# def compute_computational_metrics(surrogate: UnifiedSurrogate, n_samples: int = 100) -> Dict:
#     """Benchmark computational efficiency."""
    
#     # Generate random inputs
#     np.random.seed(42)
#     fc_samples = np.random.uniform(25, 35, n_samples)
#     E_samples = np.random.uniform(30000, 35000, n_samples)
#     cbot_samples = np.random.uniform(20, 30, n_samples)
#     ctop_samples = np.random.uniform(200, 230, n_samples)
    
#     # Warm-up
#     _ = surrogate.predict(fc_samples[0], E_samples[0], cbot_samples[0], ctop_samples[0])
    
#     # Benchmark
#     times = []
#     for i in range(n_samples):
#         t0 = time.perf_counter()
#         _ = surrogate.predict(fc_samples[i], E_samples[i], cbot_samples[i], ctop_samples[i])
#         t1 = time.perf_counter()
#         times.append(t1 - t0)
    
#     times = np.array(times) * 1000  # Convert to ms
    
#     return {
#         'mean_time_ms': float(np.mean(times)),
#         'std_time_ms': float(np.std(times)),
#         'min_time_ms': float(np.min(times)),
#         'max_time_ms': float(np.max(times)),
#         'median_time_ms': float(np.median(times)),
#         'memory_mb': surrogate.get_memory_usage()
#     }


# # ============================================================
# # MAIN COMPARISON
# # ============================================================

# def main():
#     config = Config()
#     config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     plots_dir = config.OUT_DIR / "comparison_plots"
#     plots_dir.mkdir(exist_ok=True)
    
#     print("=" * 80)
#     print("SURROGATE MODEL COMPARISON & SELECTION")
#     print("=" * 80)
    
#     # --------------------------------------------------------
#     # Load all surrogates
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("LOADING SURROGATE MODELS")
#     print("-" * 80)
    
#     surrogates = {}
    
#     model, name = load_pca_surrogate(config)
#     if model is not None:
#         surrogates[name] = UnifiedSurrogate(model, name)
#         print(f"✓ Loaded: {name}")
    
#     model, name = load_ae_surrogate(config)
#     if model is not None:
#         surrogates[name] = UnifiedSurrogate(model, name)
#         print(f"✓ Loaded: {name}")
    
#     model, name = load_shape_scale_surrogate(config)
#     if model is not None:
#         surrogates[name] = UnifiedSurrogate(model, name)
#         print(f"✓ Loaded: {name}")
    
#     if not surrogates:
#         print("ERROR: No surrogates could be loaded!")
#         return
    
#     print(f"\nTotal surrogates loaded: {len(surrogates)}")
    
#     # --------------------------------------------------------
#     # Load test data
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("LOADING TEST DATA")
#     print("-" * 80)
    
#     df_load = pd.read_csv(config.LOAD_CSV)
#     df_damage = pd.read_csv(config.DAMAGE_CSV)
#     df_uq = pd.read_csv(config.UQ_CSV)
    
#     if "job_aug" not in df_uq.columns:
#         df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
#     df_uq = df_uq.set_index("job_aug")
    
#     # Load test indices (use from one of the surrogates)
#     if "Shape-Scale PCA+GPR" in surrogates:
#         test_jobs_file = config.SHAPE_SCALE_DIR / "split" / "test_jobs.txt"
#     elif "PCA+GPR" in surrogates:
#         meta_path = config.PCA_DIR / "01_pca_reduction" / "models" / "meta.json"
#         meta = json.loads(meta_path.read_text())
#         test_idx = np.array(meta["test_idx"])
#         test_jobs = [meta["jobs"][i] for i in test_idx]
#     else:
#         # Use AE test indices
#         test_idx = np.load(config.AE_DIR / "data_preprocessed" / "test_indices.npy")
#         jobs_all = np.load(config.AE_DIR / "data_preprocessed" / "jobs.npy")
#         test_jobs = jobs_all[test_idx].tolist()
    
#     if isinstance(test_jobs_file, Path) and test_jobs_file.exists():
#         test_jobs = np.loadtxt(test_jobs_file, dtype=str).tolist()
    
#     print(f"✓ Loaded {len(test_jobs)} test samples")
    
#     # --------------------------------------------------------
#     # Computational benchmarking
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("COMPUTATIONAL EFFICIENCY BENCHMARKING")
#     print("-" * 80)
    
#     comp_metrics = {}
#     for name, surrogate in surrogates.items():
#         print(f"\nBenchmarking {name}...")
#         metrics = compute_computational_metrics(surrogate, n_samples=50)
#         comp_metrics[name] = metrics
        
#         print(f"  Mean prediction time: {metrics['mean_time_ms']:.3f} ± {metrics['std_time_ms']:.3f} ms")
#         print(f"  Memory usage: {metrics['memory_mb']:.2f} MB")
    
#     # --------------------------------------------------------
#     # Reconstruction accuracy evaluation
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("RECONSTRUCTION ACCURACY EVALUATION")
#     print("-" * 80)
    
#     # Sample subset of test jobs for detailed comparison
#     np.random.seed(42)
#     eval_jobs = np.random.choice(test_jobs, size=min(50, len(test_jobs)), replace=False)
    
#     results = {name: {'force': [], 'damage': []} for name in surrogates.keys()}
    
#     print(f"\nEvaluating on {len(eval_jobs)} test samples...")
    
#     for idx, job in enumerate(eval_jobs):
#         if (idx + 1) % 10 == 0:
#             print(f"  Progress: {idx+1}/{len(eval_jobs)}")
        
#         # Get FEM truth
#         dfj = df_load[df_load["job_aug"] == job]
#         if dfj.empty:
#             continue
        
#         u_fem = dfj["U2"].abs().to_numpy(float)
#         f_fem = dfj["RF2"].to_numpy(float)
        
#         dfd = df_damage[df_damage["job_aug"] == job]
#         if dfd.empty or config.DAMAGE_VAR not in dfd.columns:
#             continue
        
#         u_damage_fem = dfd["U2"].abs().to_numpy(float)
#         d_fem = dfd[config.DAMAGE_VAR].to_numpy(float)
        
#         # Get inputs
#         if job not in df_uq.index:
#             continue
        
#         fc = df_uq.loc[job, "fc"]
#         E = df_uq.loc[job, "E"]
#         cbot = df_uq.loc[job, "c_nom_bottom_mm"]
#         ctop = df_uq.loc[job, "c_nom_top_mm"]
        
#         # Evaluate each surrogate
#         for name, surrogate in surrogates.items():
#             try:
#                 u_force, F_pred, u_damage, D_pred = surrogate.predict(fc, E, cbot, ctop)
                
#                 # Interpolate FEM onto surrogate grids
#                 f_fem_interp = np.interp(u_force, u_fem, f_fem)
#                 d_fem_interp = np.interp(u_damage, u_damage_fem, d_fem)
                
#                 # Compute metrics
#                 force_metrics = compute_reconstruction_metrics(f_fem_interp, F_pred)
#                 damage_metrics = compute_reconstruction_metrics(d_fem_interp, D_pred)
                
#                 results[name]['force'].append(force_metrics)
#                 results[name]['damage'].append(damage_metrics)
                
#             except Exception as e:
#                 print(f"  Warning: {name} failed on {job}: {e}")
#                 continue
    
#     # --------------------------------------------------------
#     # Aggregate statistics
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("AGGREGATE STATISTICS")
#     print("-" * 80)
    
#     summary = {}
    
#     for name in surrogates.keys():
#         force_results = results[name]['force']
#         damage_results = results[name]['damage']
        
#         if not force_results:
#             continue
        
#         summary[name] = {
#             'force': {
#                 'rmse_mean': np.mean([r['rmse'] for r in force_results]),
#                 'rmse_std': np.std([r['rmse'] for r in force_results]),
#                 'r2_mean': np.mean([r['r2'] for r in force_results]),
#                 'peak_rel_error_mean': np.mean([r['peak_rel_error'] for r in force_results]),
#             },
#             'damage': {
#                 'rmse_mean': np.mean([r['rmse'] for r in damage_results]),
#                 'rmse_std': np.std([r['rmse'] for r in damage_results]),
#                 'r2_mean': np.mean([r['r2'] for r in damage_results]),
#                 'peak_rel_error_mean': np.mean([r['peak_rel_error'] for r in damage_results]),
#             },
#             'computational': comp_metrics[name]
#         }
        
#         print(f"\n{name}:")
#         print(f"  Force RMSE: {summary[name]['force']['rmse_mean']:.3f} ± {summary[name]['force']['rmse_std']:.3f}")
#         print(f"  Force R²: {summary[name]['force']['r2_mean']:.4f}")
#         print(f"  Damage RMSE: {summary[name]['damage']['rmse_mean']:.5f} ± {summary[name]['damage']['rmse_std']:.5f}")
#         print(f"  Damage R²: {summary[name]['damage']['r2_mean']:.4f}")
#         print(f"  Prediction time: {comp_metrics[name]['mean_time_ms']:.3f} ms")
    
#     # --------------------------------------------------------
#     # Create comparison plots
#     # --------------------------------------------------------
#     print("\n" + "-" * 80)
#     print("GENERATING COMPARISON PLOTS")
#     print("-" * 80)
    
#     # Plot 1: RMSE Comparison
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     names = list(summary.keys())
#     force_rmse = [summary[n]['force']['rmse_mean'] for n in names]
#     damage_rmse = [summary[n]['damage']['rmse_mean'] for n in names]
    
#     x = np.arange(len(names))
#     width = 0.35
    
#     ax1.bar(x, force_rmse, width, color='steelblue', alpha=0.8)
#     ax1.set_ylabel('RMSE', fontsize=12)
#     ax1.set_title('Force Reconstruction RMSE', fontsize=13, fontweight='bold')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(names, rotation=15, ha='right')
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     ax2.bar(x, damage_rmse, width, color='coral', alpha=0.8)
#     ax2.set_ylabel('RMSE', fontsize=12)
#     ax2.set_title('Damage Reconstruction RMSE', fontsize=13, fontweight='bold')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(names, rotation=15, ha='right')
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(plots_dir / "01_rmse_comparison.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("  ✓ Saved: 01_rmse_comparison.png")
    
#     # Plot 2: R² Comparison
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     force_r2 = [summary[n]['force']['r2_mean'] for n in names]
#     damage_r2 = [summary[n]['damage']['r2_mean'] for n in names]
    
#     ax1.bar(x, force_r2, width, color='forestgreen', alpha=0.8)
#     ax1.set_ylabel('R² Score', fontsize=12)         # Negative R² values were clipped to zero for composite scoring to avoid dominance by failed reconstructions
#     ax1.set_xticklabels(names, rotation=15, ha='right')
#     ax1.set_ylim([0, 1.05])
#     ax1.axhline(y=0.95, color='r', linestyle='--', linewidth=1, alpha=0.7)
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     ax2.bar(x, damage_r2, width, color='purple', alpha=0.8)
#     ax2.set_ylabel('R² Score', fontsize=12)
#     ax2.set_title('Damage R² Score', fontsize=13, fontweight='bold')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(names, rotation=15, ha='right')
#     ax2.set_ylim([0, 1.05])
#     ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=1, alpha=0.7)
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(plots_dir / "02_r2_comparison.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("  ✓ Saved: 02_r2_comparison.png")
    
#     # Plot 3: Computational Efficiency
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     times = [comp_metrics[n]['mean_time_ms'] for n in names]
#     memory = [comp_metrics[n]['memory_mb'] for n in names]
    
#     ax1.bar(x, times, width, color='orange', alpha=0.8)
#     ax1.set_ylabel('Time (ms)', fontsize=12)
#     ax1.set_title('Mean Prediction Time', fontsize=13, fontweight='bold')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(names, rotation=15, ha='right')
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     ax2.bar(x, memory, width, color='teal', alpha=0.8)
#     ax2.set_ylabel('Memory (MB)', fontsize=12)
#     ax2.set_title('Memory Usage', fontsize=13, fontweight='bold')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(names, rotation=15, ha='right')
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(plots_dir / "03_computational_efficiency.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("  ✓ Saved: 03_computational_efficiency.png")
    
#     # Plot 4: Overall Score (weighted)
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Compute weighted scores (higher is better)
#     scores = []
#     for name in names:
#         # Normalize metrics to [0, 1]
#         force_r2_norm = summary[name]['force']['r2_mean']
#         damage_r2_norm = summary[name]['damage']['r2_mean']
#         time_norm = 1.0 / (1.0 + summary[name]['computational']['mean_time_ms'] / 10.0)
        
#         # Weighted score: 40% force, 40% damage, 20% speed
#         score = 0.4 * force_r2_norm + 0.4 * damage_r2_norm + 0.2 * time_norm
#         scores.append(score * 100)
    
#     colors = ['#2E86AB' if s == max(scores) else '#A23B72' for s in scores]
#     bars = ax.bar(x, scores, width=0.6, color=colors, alpha=0.8)
    
#     # Annotate bars
#     for bar, score in zip(bars, scores):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{score:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
#     ax.set_ylabel('Overall Score', fontsize=12)
#     ax.set_title('Overall Surrogate Performance Score\n(40% Force R² + 40% Damage R² + 20% Speed)',
#                  fontsize=13, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(names, rotation=15, ha='right')
#     ax.set_ylim([0, 105])
#     ax.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(plots_dir / "04_overall_score.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("  ✓ Saved: 04_overall_score.png")
    
#     # --------------------------------------------------------
#     # Model selection recommendation
#     # --------------------------------------------------------
#     print("\n" + "=" * 80)
#     print("MODEL SELECTION RECOMMENDATION")
#     print("=" * 80)
    
#     best_idx = np.argmax(scores)
#     best_model = names[best_idx]
    
#     print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
#     print(f"\nOverall Score: {scores[best_idx]:.1f}/100")
#     print(f"\nReason for selection:")
#     print(f"  • Force R²: {summary[best_model]['force']['r2_mean']:.4f}")
#     print(f"  • Damage R²: {summary[best_model]['damage']['r2_mean']:.4f}")
#     print(f"  • Prediction time: {comp_metrics[best_model]['mean_time_ms']:.3f} ms")
#     print(f"  • Memory usage: {comp_metrics[best_model]['memory_mb']:.2f} MB")
    
#     # --------------------------------------------------------
#     # Save complete results
#     # --------------------------------------------------------
#     results_json = {
#         'summary': summary,
#         'recommendation': {
#             'best_model': best_model,
#             'overall_score': float(scores[best_idx]),
#             'reasons': {
#                 'force_r2': float(summary[best_model]['force']['r2_mean']),
#                 'damage_r2': float(summary[best_model]['damage']['r2_mean']),
#                 'pred_time_ms': float(comp_metrics[best_model]['mean_time_ms']),
#                 'memory_mb': float(comp_metrics[best_model]['memory_mb'])
#             }
#         },
#         'all_scores': {name: float(score) for name, score in zip(names, scores)}
#     }
    
#     with open(config.OUT_DIR / "comparison_results.json", 'w') as f:
#         json.dump(results_json, f, indent=2)
    
#     print("\n" + "=" * 80)
#     print("COMPARISON COMPLETE")
#     print("=" * 80)
#     print(f"\nResults saved to: {config.OUT_DIR}")
#     print(f"Plots saved to: {plots_dir}")


# if __name__ == "__main__":
#     main()






























































































"""
===============================================================================
STEP 5: SURROGATE MODEL COMPARISON & SELECTION
===============================================================================
Comprehensive comparison of three surrogate modeling approaches:
1. PCA + GPR
2. Autoencoder (AE) + GPR  
3. Shape-Scale PCA + GPR

Metrics:
- Train/Val/Test RMSE
- Reconstruction error distributions
- Computational efficiency (prediction time, memory)
- Physical interpretability scores
- Peak prediction accuracy
- R² scores

Output:
- Comprehensive comparison report
- Publication-ready comparison plots
- Model selection recommendation
===============================================================================
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration for surrogate comparison."""
    
    BASE = Path(__file__).resolve().parents[1]
    
    # Data paths
    LOAD_CSV = BASE / "augmentation_physics_fixed" / "load_displacement_full_aug.csv"
    DAMAGE_CSV = BASE / "augmentation_physics_fixed" / "crack_evolution_full_aug.csv"
    UQ_CSV = BASE / "augmentation_physics_fixed" / "processed_inputs_2_aug.csv"
    
    # Surrogate directories
    PCA_DIR = BASE / "04_PCA" 
    AE_DIR = BASE / "05_autoencoder_gpr"
    SHAPE_SCALE_DIR = BASE / "06_shape_scale_gpr"
    
    # Output directory
    OUT_DIR = BASE / "07_processing" / "06_surrogate_comparison"
    
    # Damage variable to use
    DAMAGE_VAR = 'DAMAGEC_max'  # or 'DAMAGEC_max' for compression

# ============================================================
# SURROGATE LOADERS
# ============================================================

def load_pca_surrogate(config: Config):
    """Load PCA+GPR surrogate."""
    try:
        # Import must be relative to where the file exists
        import sys
        sys.path.insert(0, str(config.PCA_DIR / "01_pca_reduction"))
        from surrogate_model import SurrogateModel
        
        model = SurrogateModel.load(
            pca_dir=config.PCA_DIR / "01_pca_reduction" / "models",
            surrogate_dir=config.PCA_DIR / "01_pca_reduction" / "outputs"
        )
        return model, "PCA+GPR"
    except Exception as e:
        print(f"Warning: Could not load PCA+GPR surrogate: {e}")
        return None, "PCA+GPR"


def load_ae_surrogate(config: Config):
    """Load Autoencoder+GPR surrogate with normalization factors."""
    try:
        import sys
        sys.path.insert(0, str(config.AE_DIR))
        from ae_surrogate_model import ImprovedAESurrogateModel
        
        model = ImprovedAESurrogateModel(str(config.BASE), use_improved=True)
        
        # Load normalization factors
        data_dir = config.AE_DIR / "data_preprocessed"
        global_Fmax = float(np.load(data_dir / "F_global_max.npy"))
        C_max_all = np.load(data_dir / "C_max.npy")
        
        return (model, global_Fmax, C_max_all), "AE+GPR"
    except Exception as e:
        print(f"Warning: Could not load AE+GPR surrogate: {e}")
        return None, "AE+GPR"


def load_shape_scale_surrogate(config: Config):
    """Load Shape-Scale PCA+GPR surrogate."""
    try:
        import sys
        sys.path.insert(0, str(config.SHAPE_SCALE_DIR))
        from shape_scale_surrogate import ShapeScaleSurrogate
        
        model = ShapeScaleSurrogate.load(config.SHAPE_SCALE_DIR)
        return model, "Shape-Scale PCA+GPR"
    except Exception as e:
        print(f"Warning: Could not load Shape-Scale surrogate: {e}")
        return None, "Shape-Scale PCA+GPR"


# ============================================================
# UNIFIED PREDICTION INTERFACE
# ============================================================

class UnifiedSurrogate:
    """Unified interface for all surrogate types."""
    
    def __init__(self, model, model_type: str, norm_factors=None):
        self.model = model
        self.model_type = model_type
        self.norm_factors = norm_factors  # For AE: (global_Fmax, C_max_all)
    
    def predict(self, fc: float, E: float, cbot: float, ctop: float, 
                job_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unified prediction interface.
        
        Args:
            job_idx: Job index for AE denormalization (uses C_max_all[job_idx])
        
        Returns:
            u_force, F_pred, u_damage, D_pred (all in actual units)
        """
        if self.model_type == "PCA+GPR":
            # PCA model returns normalized predictions
            F_pred, D_pred = self.model.predict_curves(
                fc=fc, E=E, cbot=cbot, ctop=ctop, return_uncertainty=False
            )
            u_force = self.model.u_grid_force
            u_damage = self.model.u_grid_damage
            return u_force, F_pred, u_damage, D_pred
            
        elif self.model_type == "AE+GPR":
            # AE model returns NORMALIZED predictions - need to denormalize
            u_force, F_pred_norm, u_damage, D_pred_norm = self.model.predict(
                cbot=cbot, ctop=ctop, fcm=fc
            )
            
            # Denormalize
            if self.norm_factors is not None:
                global_Fmax, C_max_all = self.norm_factors
                F_pred = F_pred_norm * global_Fmax
                # Use average C_max if job_idx not valid
                if job_idx < len(C_max_all):
                    D_pred = D_pred_norm * C_max_all[job_idx]
                else:
                    D_pred = D_pred_norm * np.mean(C_max_all)
            else:
                # Fallback if no normalization factors
                F_pred = F_pred_norm
                D_pred = D_pred_norm
            
            return u_force, F_pred, u_damage, D_pred
            
        elif self.model_type == "Shape-Scale PCA+GPR":
            # Shape-scale returns actual values
            u_force, F_pred, u_damage, D_pred = self.model.predict_curves(
                fc=fc, E=E, cbot=cbot, ctop=ctop
            )
            return u_force, F_pred, u_damage, D_pred
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        import sys
        if isinstance(self.model, tuple):
            return sys.getsizeof(self.model[0]) / 1024 / 1024
        return sys.getsizeof(self.model) / 1024 / 1024


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute comprehensive reconstruction metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Peak metrics
    peak_true = np.max(np.abs(y_true))
    peak_pred = np.max(np.abs(y_pred))
    peak_error = abs(peak_pred - peak_true)
    peak_rel_error = peak_error / (peak_true + 1e-10) * 100
    
    # Pointwise errors
    pointwise_errors = np.abs(y_true - y_pred)
    max_pointwise = np.max(pointwise_errors)
    mean_pointwise = np.mean(pointwise_errors)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'peak_true': peak_true,
        'peak_pred': peak_pred,
        'peak_error': peak_error,
        'peak_rel_error': peak_rel_error,
        'max_pointwise_error': max_pointwise,
        'mean_pointwise_error': mean_pointwise
    }


def compute_computational_metrics(surrogate: UnifiedSurrogate, n_samples: int = 100) -> Dict:
    """Benchmark computational efficiency."""
    
    # Generate random inputs
    np.random.seed(42)
    fc_samples = np.random.uniform(25, 35, n_samples)
    E_samples = np.random.uniform(30000, 35000, n_samples)
    cbot_samples = np.random.uniform(20, 30, n_samples)
    ctop_samples = np.random.uniform(200, 230, n_samples)
    
    # Warm-up
    _ = surrogate.predict(fc_samples[0], E_samples[0], cbot_samples[0], ctop_samples[0], job_idx=0)
    
    # Benchmark
    times = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        _ = surrogate.predict(fc_samples[i], E_samples[i], cbot_samples[i], ctop_samples[i], job_idx=i%100)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
        'memory_mb': surrogate.get_memory_usage()
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():
    config = Config()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plots_dir = config.OUT_DIR / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SURROGATE MODEL COMPARISON & SELECTION")
    print("=" * 80)
    
    # --------------------------------------------------------
    # Load all surrogates
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("LOADING SURROGATE MODELS")
    print("-" * 80)
    
    surrogates = {}
    
    model, name = load_pca_surrogate(config)
    if model is not None:
        surrogates[name] = UnifiedSurrogate(model, name)
        print(f"✓ Loaded: {name}")
    
    model_data, name = load_ae_surrogate(config)
    if model_data is not None:
        if isinstance(model_data, tuple):
            model, global_Fmax, C_max_all = model_data
            surrogates[name] = UnifiedSurrogate(model, name, norm_factors=(global_Fmax, C_max_all))
        else:
            surrogates[name] = UnifiedSurrogate(model_data, name)
        print(f"✓ Loaded: {name}")
    
    model, name = load_shape_scale_surrogate(config)
    if model is not None:
        surrogates[name] = UnifiedSurrogate(model, name)
        print(f"✓ Loaded: {name}")
    
    if not surrogates:
        print("ERROR: No surrogates could be loaded!")
        return
    
    print(f"\nTotal surrogates loaded: {len(surrogates)}")
    
    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("LOADING TEST DATA")
    print("-" * 80)
    
    df_load = pd.read_csv(config.LOAD_CSV)
    df_damage = pd.read_csv(config.DAMAGE_CSV)
    df_uq = pd.read_csv(config.UQ_CSV)
    
    if "job_aug" not in df_uq.columns:
        df_uq["job_aug"] = df_uq["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")
    
    df_uq = df_uq.set_index("job_aug")
    
    # Load test indices (use from one of the surrogates)
    if "Shape-Scale PCA+GPR" in surrogates:
        test_jobs_file = config.SHAPE_SCALE_DIR / "split" / "test_jobs.txt"
    elif "PCA+GPR" in surrogates:
        meta_path = config.PCA_DIR / "01_pca_reduction" / "models" / "meta.json"
        meta = json.loads(meta_path.read_text())
        test_idx = np.array(meta["test_idx"])
        test_jobs = [meta["jobs"][i] for i in test_idx]
    else:
        # Use AE test indices
        test_idx = np.load(config.AE_DIR / "data_preprocessed" / "test_indices.npy")
        jobs_all = np.load(config.AE_DIR / "data_preprocessed" / "jobs.npy")
        test_jobs = jobs_all[test_idx].tolist()
    
    if isinstance(test_jobs_file, Path) and test_jobs_file.exists():
        test_jobs = np.loadtxt(test_jobs_file, dtype=str).tolist()
    
    print(f"✓ Loaded {len(test_jobs)} test samples")
    
    # --------------------------------------------------------
    # Computational benchmarking
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("COMPUTATIONAL EFFICIENCY BENCHMARKING")
    print("-" * 80)
    
    comp_metrics = {}
    for name, surrogate in surrogates.items():
        print(f"\nBenchmarking {name}...")
        metrics = compute_computational_metrics(surrogate, n_samples=50)
        comp_metrics[name] = metrics
        
        print(f"  Mean prediction time: {metrics['mean_time_ms']:.3f} ± {metrics['std_time_ms']:.3f} ms")
        print(f"  Memory usage: {metrics['memory_mb']:.2f} MB")
    
    # --------------------------------------------------------
    # Reconstruction accuracy evaluation
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("RECONSTRUCTION ACCURACY EVALUATION")
    print("-" * 80)
    
    # Sample subset of test jobs for detailed comparison
    np.random.seed(42)
    eval_jobs = np.random.choice(test_jobs, size=min(50, len(test_jobs)), replace=False)
    
    # For AE: load job list to map job names to indices
    ae_job_to_idx = {}
    if "AE+GPR" in surrogates:
        try:
            jobs_all = np.load(config.AE_DIR / "data_preprocessed" / "jobs.npy")
            ae_job_to_idx = {job: idx for idx, job in enumerate(jobs_all)}
        except:
            print("  Warning: Could not load AE job mapping for denormalization")
    
    results = {name: {'force': [], 'damage': []} for name in surrogates.keys()}
    
    print(f"\nEvaluating on {len(eval_jobs)} test samples...")
    
    for idx, job in enumerate(eval_jobs):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(eval_jobs)}")
        
        # Get FEM truth
        dfj = df_load[df_load["job_aug"] == job]
        if dfj.empty:
            continue
        
        u_fem = dfj["U2"].abs().to_numpy(float)
        f_fem = dfj["RF2"].to_numpy(float)
        
        dfd = df_damage[df_damage["job_aug"] == job]
        if dfd.empty or config.DAMAGE_VAR not in dfd.columns:
            continue
        
        u_damage_fem = dfd["U2"].abs().to_numpy(float)
        d_fem = dfd[config.DAMAGE_VAR].to_numpy(float)
        
        # Get inputs
        if job not in df_uq.index:
            continue
        
        fc = df_uq.loc[job, "fc"]
        E = df_uq.loc[job, "E"]
        cbot = df_uq.loc[job, "c_nom_bottom_mm"]
        ctop = df_uq.loc[job, "c_nom_top_mm"]
        
        # Get job index for AE denormalization
        job_idx = ae_job_to_idx.get(job, 0) if ae_job_to_idx else 0
        
        # Evaluate each surrogate
        for name, surrogate in surrogates.items():
            try:
                u_force, F_pred, u_damage, D_pred = surrogate.predict(
                    fc, E, cbot, ctop, job_idx=job_idx
                )
                
                # Interpolate FEM onto surrogate grids
                f_fem_interp = np.interp(u_force, u_fem, f_fem)
                d_fem_interp = np.interp(u_damage, u_damage_fem, d_fem)
                
                # Compute metrics
                force_metrics = compute_reconstruction_metrics(f_fem_interp, F_pred)
                damage_metrics = compute_reconstruction_metrics(d_fem_interp, D_pred)
                
                results[name]['force'].append(force_metrics)
                results[name]['damage'].append(damage_metrics)
                
            except Exception as e:
                print(f"  Warning: {name} failed on {job}: {e}")
                continue
    
    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)
    
    summary = {}
    
    for name in surrogates.keys():
        force_results = results[name]['force']
        damage_results = results[name]['damage']
        
        if not force_results:
            continue
        
        summary[name] = {
            'force': {
                'rmse_mean': np.mean([r['rmse'] for r in force_results]),
                'rmse_std': np.std([r['rmse'] for r in force_results]),
                'r2_mean': np.mean([r['r2'] for r in force_results]),
                'peak_rel_error_mean': np.mean([r['peak_rel_error'] for r in force_results]),
            },
            'damage': {
                'rmse_mean': np.mean([r['rmse'] for r in damage_results]),
                'rmse_std': np.std([r['rmse'] for r in damage_results]),
                'r2_mean': np.mean([r['r2'] for r in damage_results]),
                'peak_rel_error_mean': np.mean([r['peak_rel_error'] for r in damage_results]),
            },
            'computational': comp_metrics[name]
        }
        
        print(f"\n{name}:")
        print(f"  Force RMSE: {summary[name]['force']['rmse_mean']:.3f} ± {summary[name]['force']['rmse_std']:.3f}")
        print(f"  Force R²: {summary[name]['force']['r2_mean']:.4f}")
        print(f"  Damage RMSE: {summary[name]['damage']['rmse_mean']:.5f} ± {summary[name]['damage']['rmse_std']:.5f}")
        print(f"  Damage R²: {summary[name]['damage']['r2_mean']:.4f}")
        print(f"  Prediction time: {comp_metrics[name]['mean_time_ms']:.3f} ms")
    
    # --------------------------------------------------------
    # Create comparison plots
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("-" * 80)
    
    # Plot 1: RMSE Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(summary.keys())
    force_rmse = [summary[n]['force']['rmse_mean'] for n in names]
    damage_rmse = [summary[n]['damage']['rmse_mean'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x, force_rmse, width, color='steelblue', alpha=0.8)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Force Reconstruction RMSE', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, damage_rmse, width, color='coral', alpha=0.8)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Damage Reconstruction RMSE', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "01_rmse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 01_rmse_comparison.png")
    
    # Plot 2: R² Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    force_r2 = [summary[n]['force']['r2_mean'] for n in names]
    damage_r2 = [summary[n]['damage']['r2_mean'] for n in names]
    
    ax1.bar(x, force_r2, width, color='forestgreen', alpha=0.8)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Force R² Score', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, damage_r2, width, color='purple', alpha=0.8)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Damage R² Score', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "02_r2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_r2_comparison.png")
    
    # Plot 3: Computational Efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    times = [comp_metrics[n]['mean_time_ms'] for n in names]
    memory = [comp_metrics[n]['memory_mb'] for n in names]
    
    ax1.bar(x, times, width, color='orange', alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Mean Prediction Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, memory, width, color='teal', alpha=0.8)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "03_computational_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_computational_efficiency.png")
    
    # Plot 4: Overall Score (weighted)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute weighted scores (higher is better)
    scores = []
    for name in names:
        # Normalize metrics to [0, 1]
        force_r2_norm = summary[name]['force']['r2_mean']
        damage_r2_norm = summary[name]['damage']['r2_mean']
        time_norm = 1.0 / (1.0 + summary[name]['computational']['mean_time_ms'] / 10.0)
        
        # Weighted score: 40% force, 40% damage, 20% speed
        score = 0.4 * force_r2_norm + 0.4 * damage_r2_norm + 0.2 * time_norm
        scores.append(score * 100)
    
    colors = ['#2E86AB' if s == max(scores) else '#A23B72' for s in scores]
    bars = ax.bar(x, scores, width=0.6, color=colors, alpha=0.8)
    
    # Annotate bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Overall Score', fontsize=12)
    ax.set_title('Overall Surrogate Performance Score\n(40% Force R² + 40% Damage R² + 20% Speed)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "04_overall_score.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_overall_score.png")
    
    # --------------------------------------------------------
    # Model selection recommendation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL SELECTION RECOMMENDATION")
    print("=" * 80)
    
    best_idx = np.argmax(scores)
    best_model = names[best_idx]
    
    print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
    print(f"\nOverall Score: {scores[best_idx]:.1f}/100")
    print(f"\nReason for selection:")
    print(f"  • Force R²: {summary[best_model]['force']['r2_mean']:.4f}")
    print(f"  • Damage R²: {summary[best_model]['damage']['r2_mean']:.4f}")
    print(f"  • Prediction time: {comp_metrics[best_model]['mean_time_ms']:.3f} ms")
    print(f"  • Memory usage: {comp_metrics[best_model]['memory_mb']:.2f} MB")
    
    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------
    results_json = {
        'summary': summary,
        'recommendation': {
            'best_model': best_model,
            'overall_score': float(scores[best_idx]),
            'reasons': {
                'force_r2': float(summary[best_model]['force']['r2_mean']),
                'damage_r2': float(summary[best_model]['damage']['r2_mean']),
                'pred_time_ms': float(comp_metrics[best_model]['mean_time_ms']),
                'memory_mb': float(comp_metrics[best_model]['memory_mb'])
            }
        },
        'all_scores': {name: float(score) for name, score in zip(names, scores)}
    }
    
    with open(config.OUT_DIR / "comparison_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.OUT_DIR}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()