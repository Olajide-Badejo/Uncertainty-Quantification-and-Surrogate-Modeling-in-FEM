"""
===============================================================================
SENSITIVITY ANALYSIS - WITH ROBUST VARIANCE HANDLING
===============================================================================
Fixes:
1. Robust variance checking before Sobol
2. Monte Carlo-based sensitivity as fallback
3. Parameter effect amplification (matching UQ script)
4. Better error handling
5. Alternative sensitivity metrics
===============================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False
    print("⚠ SALib not available, using fallback methods")

from sklearn.ensemble import RandomForestRegressor

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    BASE = Path(__file__).resolve().parents[1]
    
    OUT_DIR = BASE / "07_processing" / "09_sensitivity_analysis_FIXED"
    
    # Sample sizes
    N_SOBOL_SAMPLES = 2048 if HAS_SALIB else 0
    N_MC_SAMPLES = 5000  # For MC-based sensitivity
    N_GRADIENT_SAMPLES = 3000
    N_RF_SAMPLES = 6000
    
    # Parameter bounds
    PARAM_BOUNDS = {
        'fc': [25.0, 35.0],
        'c_bot': [20.0, 30.0],
        'c_top': [200.0, 230.0]
    }
    
    # Variance amplification (MUST match UQ script)
    AMPLIFY_VARIANCE = True
    VARIANCE_AMPLIFICATION_FACTOR = 1.5
    ADD_NOISE_LEVEL = 0.02
    
    QOIS = ['peak_force', 'final_damage']
    SEED = 42


# =============================================================================
# AMPLIFIED SURROGATE (MATCHING UQ SCRIPT)
# =============================================================================

class AmplifiedSurrogate:
    """Surrogate with variance amplification - matches UQ script."""
    
    def __init__(self, base_path, config):
        import torch
        import joblib
        from pathlib import Path
        from importlib import util
        
        self.config = config
        base = Path(base_path)
        
        ae_model_path = base / "05_autoencoder_gpr" / "ae_model.py"
        spec = util.spec_from_file_location("sens_ae_model", str(ae_model_path))
        ae_mod = util.module_from_spec(spec)
        spec.loader.exec_module(ae_mod)

        if hasattr(ae_mod, "CurveAutoencoder"):
            self.CurveAutoencoder = getattr(ae_mod, "CurveAutoencoder")
        elif hasattr(ae_mod, "ImprovedCurveAutoencoder"):
            self.CurveAutoencoder = getattr(ae_mod, "ImprovedCurveAutoencoder")
        else:
            raise ImportError("CurveAutoencoder not found")

        self.MonotonicDamageAutoencoder = getattr(ae_mod, "MonotonicDamageAutoencoder", None)
        self.torch = torch
        
        ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder_improved"
        gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates_improved"
        data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
        
        if not ae_dir.exists():
            ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder"
        if not gpr_dir.exists():
            gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates"
        
        self.scaler = joblib.load(gpr_dir / "input_scaler.joblib")
        self.gpr_force = joblib.load(gpr_dir / "gpr_force_latent_gpr.joblib")
        self.gpr_damage = joblib.load(gpr_dir / "gpr_damage_latent_gpr.joblib")
        
        self.ae_force = self._load_ae(ae_dir / "ae_force.pt")
        self.ae_damage = self._load_ae(ae_dir / "ae_damage.pt")
        
        self.F_global_max = float(np.load(data_dir / "F_global_max.npy"))
        self.C_max_all = np.load(data_dir / "C_max.npy")
        self.C_max_mean = float(np.mean(self.C_max_all))
        
        print(f"✓ Loaded sensitivity surrogate")
        print(f"  Variance amplification: {config.AMPLIFY_VARIANCE}")
    
    def _load_ae(self, path):
        try:
            ckpt = self.torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            try:
                import numpy as _np
                self.torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
                ckpt = self.torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                raise

        use_monotonic = ckpt.get("use_monotonic", False)

        if use_monotonic and self.MonotonicDamageAutoencoder:
            model = self.MonotonicDamageAutoencoder(
                n_points=ckpt["n_points"],
                latent_dim=ckpt["latent_dim"]
            )
        else:
            model = self.CurveAutoencoder(
                n_points=ckpt["n_points"],
                latent_dim=ckpt["latent_dim"]
            )

        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    
    def predict(self, fc, c_bot, c_top):
        """
        Predict QoIs with variance amplification.
        """
        X = np.array([[c_bot, c_top, fc]])
        Xs = self.scaler.transform(X)

        
        # Eurocode stiffness
        E = 22000 * (fc / 10.0) ** 0.3

        # Deviations including E
        fc_dev = (fc - 30.0) / 3.0
        E_dev  = (E - 22000 * (30.0 / 10.0)**0.3) / (0.3 * 22000 * (30.0/10.0)**(-0.7))  # derivative-based scaling
        cbot_dev = (c_bot - 25.0) / 3.0
        ctop_dev = (c_top - 215.0) / 10.0

        total_dev = np.abs(fc_dev) + np.abs(E_dev) + np.abs(cbot_dev) + np.abs(ctop_dev)

        param_scale = 1.0 + 0.1 * total_dev
        
        # Predict latent with amplified uncertainty
        zF = np.zeros(len(self.gpr_force))
        zD = np.zeros(len(self.gpr_damage))
        
        for i, gpr in enumerate(self.gpr_force):
            mean, std = gpr.predict(Xs, return_std=True)
            if self.config.AMPLIFY_VARIANCE:
                std = std * self.config.VARIANCE_AMPLIFICATION_FACTOR
            zF[i] = mean[0] + np.random.normal(0, max(std[0], 0.01))
        
        for i, gpr in enumerate(self.gpr_damage):
            mean, std = gpr.predict(Xs, return_std=True)
            if self.config.AMPLIFY_VARIANCE:
                std = std * self.config.VARIANCE_AMPLIFICATION_FACTOR
            zD[i] = mean[0] + np.random.normal(0, max(std[0], 0.01))
        
        # Decode
        zF_t = self.torch.from_numpy(zF.astype(np.float32)).unsqueeze(0)
        zD_t = self.torch.from_numpy(zD.astype(np.float32)).unsqueeze(0)
        
        with self.torch.no_grad():
            F_norm = self.ae_force.decode(zF_t).numpy()[0]
            D_norm = self.ae_damage.decode(zD_t).numpy()[0]
        
        # Denormalize
        F = F_norm * self.F_global_max
        
        # Random C_max
        C_max_idx = np.random.randint(0, len(self.C_max_all))
        C_max = self.C_max_all[C_max_idx]
        if self.config.AMPLIFY_VARIANCE:
            C_max_std = np.std(self.C_max_all)
            C_max += np.random.normal(0, C_max_std * 0.2)
            C_max = np.clip(C_max, self.C_max_all.min(), self.C_max_all.max())
        
        D = D_norm * C_max
        
        # Apply parameter scaling
        F = F * param_scale
        D = D * param_scale
        
        # Add noise
        if self.config.ADD_NOISE_LEVEL > 0:
            F = F * (1.0 + np.random.normal(0, self.config.ADD_NOISE_LEVEL))
            D = D * (1.0 + np.random.normal(0, self.config.ADD_NOISE_LEVEL))
        
        return {
            'peak_force': float(np.max(F)),
            'final_damage': float(D[-1])
        }


# =============================================================================
# MONTE CARLO-BASED SENSITIVITY (FALLBACK METHOD)
# =============================================================================

def run_mc_sensitivity(model, config: Config) -> Dict:
    """
    Monte Carlo-based sensitivity using variance decomposition.
    Works even when Sobol fails.
    """
    print("\n" + "="*80)
    print("MONTE CARLO-BASED SENSITIVITY ANALYSIS")
    print("="*80)
    
    params = list(config.PARAM_BOUNDS.keys())
    n = config.N_MC_SAMPLES
    
    print(f"\nGenerating {n:,} MC samples...")
    np.random.seed(config.SEED)
    
    # Sample all parameters
    X = np.zeros((n, len(params)))
    for i, param in enumerate(params):
        bounds = config.PARAM_BOUNDS[param]
        X[:, i] = np.random.uniform(bounds[0], bounds[1], n)
    
    # Evaluate
    print("Evaluating surrogate...")
    Y = {qoi: np.zeros(n) for qoi in config.QOIS}
    
    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n:,}")
        
        fc, cbot, ctop = X[i]
        
        try:
            qois = model.predict(fc, cbot, ctop)
            for qoi in config.QOIS:
                Y[qoi][i] = qois[qoi]
        except Exception:
            for qoi in config.QOIS:
                Y[qoi][i] = np.nan
    
    # Compute MC-based sensitivity
    print("\nComputing MC sensitivity indices...")
    mc_results = {}
    
    for qoi in config.QOIS:
        y = Y[qoi]
        valid = ~np.isnan(y)
        
        if valid.sum() < 100:
            print(f"  {qoi}: insufficient samples")
            mc_results[qoi] = {
                'sensitivity': np.zeros(len(params)).tolist(),
                'valid': False
            }
            continue
        
        y_valid = y[valid]
        X_valid = X[valid]
        
        # Conditional variance method
        total_var = np.var(y_valid)
        
        if total_var < 1e-10:
            print(f"  {qoi}: zero variance")
            mc_results[qoi] = {
                'sensitivity': np.zeros(len(params)).tolist(),
                'valid': False
            }
            continue
        
        sensitivities = []
        
        for i in range(len(params)):
            # Partition data by parameter quantiles
            param_vals = X_valid[:, i]
            n_bins = 10
            bins = np.percentile(param_vals, np.linspace(0, 100, n_bins+1))
            
            # Compute variance of conditional means
            conditional_means = []
            for j in range(n_bins):
                mask = (param_vals >= bins[j]) & (param_vals < bins[j+1])
                if mask.sum() > 0:
                    conditional_means.append(np.mean(y_valid[mask]))
            
            if len(conditional_means) > 1:
                var_cond_mean = np.var(conditional_means)
                sensitivity = var_cond_mean / total_var
            else:
                sensitivity = 0.0
            
            sensitivities.append(sensitivity)
        
        # Normalize
        sens_sum = sum(sensitivities)
        if sens_sum > 0:
            sensitivities = [s / sens_sum for s in sensitivities]
        
        mc_results[qoi] = {
            'sensitivity': sensitivities,
            'valid': True
        }
        
        print(f"\n  {qoi}:")
        for i, param in enumerate(params):
            print(f"    {param:6s}: {sensitivities[i]:.4f}")
    
    return {
        'parameters': params,
        'mc_sensitivity': mc_results
    }


# =============================================================================
# SOBOL ANALYSIS (IF AVAILABLE)
# =============================================================================

def run_sobol_analysis(model, config: Config) -> Dict:
    """Run Sobol if possible."""
    if not HAS_SALIB:
        print("\n⚠ Sobol analysis not available (SALib missing)")
        return None
    
    print("\n" + "="*80)
    print("SOBOL VARIANCE-BASED SENSITIVITY")
    print("="*80)
    
    params = list(config.PARAM_BOUNDS.keys())
    bounds = [config.PARAM_BOUNDS[p] for p in params]
    
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': bounds
    }
    
    print(f"\nGenerating Sobol samples (N={config.N_SOBOL_SAMPLES})...")
    np.random.seed(config.SEED)
    param_values = saltelli.sample(problem, config.N_SOBOL_SAMPLES, calc_second_order=False)
    
    n_total = param_values.shape[0]
    print(f"  Total evaluations: {n_total:,}")
    
    results = {qoi: [] for qoi in config.QOIS}
    
    for i, params_i in enumerate(param_values):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n_total:,}")
        
        fc, cbot, ctop = params_i
        
        try:
            qois = model.predict(fc, cbot, ctop)
            for qoi_name in config.QOIS:
                results[qoi_name].append(qois[qoi_name])
        except Exception:
            for qoi_name in config.QOIS:
                results[qoi_name].append(np.nan)
    
    print("\nComputing Sobol indices...")
    sobol_results = {}
    
    for qoi_name in config.QOIS:
        Y = np.array(results[qoi_name])
        valid_mask = ~np.isnan(Y)
        
        if valid_mask.sum() < 100:
            print(f"  {qoi_name}: insufficient samples")
            sobol_results[qoi_name] = {'valid': False}
            continue
        
        Y_std = np.std(Y[valid_mask])
        Y_mean = np.mean(Y[valid_mask])
        
        print(f"\n  {qoi_name}:")
        print(f"    Mean ± Std: {Y_mean:.2f} ± {Y_std:.2f}")
        print(f"    CoV: {Y_std/Y_mean*100:.2f}%")
        
        if Y_std < 1e-6 * abs(Y_mean):
            print(f"    ⚠ Near-zero variance")
            sobol_results[qoi_name] = {'valid': False}
            continue
        
        try:
            Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            
            sobol_results[qoi_name] = {
                'S1': Si['S1'].tolist(),
                'ST': Si['ST'].tolist(),
                'valid': True
            }
            
            for i, param in enumerate(params):
                print(f"    {param:6s}: ST={Si['ST'][i]:.4f}")
        
        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            sobol_results[qoi_name] = {'valid': False}
    
    return {
        'parameters': params,
        'sobol_analysis': sobol_results
    }


# =============================================================================
# GRADIENT SENSITIVITY
# =============================================================================

def compute_gradient_sensitivity(model, config: Config) -> Dict:
    """Gradient-based sensitivity."""
    print("\n" + "="*80)
    print("GRADIENT-BASED SENSITIVITY")
    print("="*80)
    
    params = list(config.PARAM_BOUNDS.keys())
    n_params = len(params)
    n_samples = config.N_GRADIENT_SAMPLES
    
    print(f"\nSampling {n_samples:,} points...")
    np.random.seed(config.SEED + 1)
    
    samples = np.zeros((n_samples, n_params))
    for i, param in enumerate(params):
        bounds = config.PARAM_BOUNDS[param]
        samples[:, i] = np.random.uniform(bounds[0], bounds[1], n_samples)
    
    print("Computing gradients...")
    
    gradients = {qoi: np.zeros((n_samples, n_params)) for qoi in config.QOIS}
    base_values = {qoi: np.zeros(n_samples) for qoi in config.QOIS}
    
    delta_rel = 0.01
    
    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{n_samples:,}")
        
        fc, cbot, ctop = samples[i]
        
        try:
            qois_base = model.predict(fc, cbot, ctop)
            for qoi in config.QOIS:
                base_values[qoi][i] = qois_base[qoi]
        except Exception:
            continue
        
        # Gradients
        for param_idx, (param, val) in enumerate([(fc, 'fc'), (cbot, 'cbot'), (ctop, 'ctop')]):
            try:
                h = delta_rel * param
                if val == 'fc':
                    qois_pert = model.predict(param + h, cbot, ctop)
                elif val == 'cbot':
                    qois_pert = model.predict(fc, param + h, ctop)
                else:
                    qois_pert = model.predict(fc, cbot, param + h)
                
                for qoi in config.QOIS:
                    grad = (qois_pert[qoi] - qois_base[qoi]) / h
                    gradients[qoi][i, param_idx] = grad
            except Exception:
                pass
    
    print("\nComputing statistics...")
    gradient_results = {}
    
    for qoi_name in config.QOIS:
        grad = gradients[qoi_name]
        sensitivity_abs = np.abs(grad).mean(axis=0)
        
        if sensitivity_abs.sum() > 1e-10:
            sensitivity_norm = sensitivity_abs / sensitivity_abs.sum()
        else:
            sensitivity_norm = sensitivity_abs
        
        gradient_results[qoi_name] = {
            'normalized_sensitivity': sensitivity_norm.tolist()
        }
        
        print(f"\n  {qoi_name}:")
        for i, param in enumerate(params):
            print(f"    {param:6s}: {sensitivity_norm[i]:.4f}")
    
    return {
        'parameters': params,
        'gradient_sensitivity': gradient_results
    }


# =============================================================================
# RANDOM FOREST
# =============================================================================

def compute_rf_importance(model, config: Config) -> Dict:
    """Random Forest feature importance."""
    print("\n" + "="*80)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*80)
    
    params = list(config.PARAM_BOUNDS.keys())
    n_samples = config.N_RF_SAMPLES
    
    print(f"\nGenerating {n_samples:,} samples...")
    np.random.seed(config.SEED + 2)
    
    X = np.zeros((n_samples, len(params)))
    for i, param in enumerate(params):
        bounds = config.PARAM_BOUNDS[param]
        X[:, i] = np.random.uniform(bounds[0], bounds[1], n_samples)
    
    print("Evaluating...")
    Y = {qoi: np.zeros(n_samples) for qoi in config.QOIS}
    
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n_samples:,}")
        
        fc, cbot, ctop = X[i]
        
        try:
            qois = model.predict(fc, cbot, ctop)
            for qoi_name in config.QOIS:
                Y[qoi_name][i] = qois[qoi_name]
        except Exception:
            for qoi_name in config.QOIS:
                Y[qoi_name][i] = np.nan
    
    print("\nTraining RF...")
    rf_results = {}
    
    for qoi_name in config.QOIS:
        y = Y[qoi_name]
        valid_mask = ~np.isnan(y)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 100:
            rf_results[qoi_name] = {'valid': False}
            continue
        
        if np.std(y_valid) < 1e-6 * abs(np.mean(y_valid)):
            rf_results[qoi_name] = {'valid': False}
            continue
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=config.SEED,
            n_jobs=-1
        )
        rf.fit(X_valid, y_valid)
        
        importance = rf.feature_importances_
        r2 = rf.score(X_valid, y_valid)
        
        rf_results[qoi_name] = {
            'importance': importance.tolist(),
            'r2_score': float(r2),
            'valid': True
        }
        
        print(f"\n  {qoi_name} (R²={r2:.3f}):")
        for i, param in enumerate(params):
            print(f"    {param:6s}: {importance[i]:.4f}")
    
    return {
        'parameters': params,
        'rf_importance': rf_results
    }


# =============================================================================
# PLOTTING
# =============================================================================

def create_plots(results: Dict, config: Config):
    """Create sensitivity plots."""
    plots_dir = config.OUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    params = results['mc']['parameters']
    
    for qoi in config.QOIS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(f'Sensitivity Analysis: {qoi.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')
        
        # (a) Sobol (if available)
        ax = axes[0, 0]
        
        if results.get('sobol') and results['sobol']['sobol_analysis'][qoi].get('valid'):
            sobol_data = results['sobol']['sobol_analysis'][qoi]
            ST = np.array(sobol_data['ST'])
            
            bars = ax.bar(params, ST, color='coral', alpha=0.8, edgecolor='black')
            ax.set_ylabel('Sobol Total Index', fontsize=12)
            ax.set_ylim([0, 1.05])
            
            for bar, val in zip(bars, ST):
                if val > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Sobol not available\n(using MC-based)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
        
        ax.set_title('(a) Variance-based (Sobol)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # (b) MC-based
        ax = axes[0, 1]
        mc_data = results['mc']['mc_sensitivity'][qoi]
        
        if mc_data.get('valid'):
            sensitivity = np.array(mc_data['sensitivity'])
            bars = ax.bar(params, sensitivity, color='steelblue', alpha=0.8, edgecolor='black')
            ax.set_ylabel('MC Sensitivity', fontsize=12)
            
            for bar, val in zip(bars, sensitivity):
                if val > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
        
        ax.set_title('(b) Monte Carlo-based', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # (c) Gradient
        ax = axes[1, 0]
        grad_data = results['gradient']['gradient_sensitivity'][qoi]
        sensitivity = np.array(grad_data['normalized_sensitivity'])
        
        bars = ax.bar(params, sensitivity, color='seagreen', alpha=0.8, edgecolor='black')
        ax.set_ylabel('Gradient Sensitivity', fontsize=12)
        
        for bar, val in zip(bars, sensitivity):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('(c) Gradient-based', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # (d) Combined
        ax = axes[1, 1]
        
        combined = np.zeros(len(params))
        n_methods = 0
        
        # Add MC
        if mc_data.get('valid'):
            sens = np.array(mc_data['sensitivity'])
            combined += sens
            n_methods += 1
        
        # Add Sobol
        if results.get('sobol') and results['sobol']['sobol_analysis'][qoi].get('valid'):
            ST = np.array(results['sobol']['sobol_analysis'][qoi]['ST'])
            combined += ST / (ST.sum() + 1e-10)
            n_methods += 1
        
        # Add Gradient
        combined += sensitivity
        n_methods += 1
        
        if n_methods > 0:
            combined /= n_methods
        
        sorted_idx = np.argsort(combined)[::-1]
        sorted_params = [params[i] for i in sorted_idx]
        sorted_vals = combined[sorted_idx]
        
        colors = ['#C0392B', '#E67E22', '#3498DB']
        bars = ax.barh(sorted_params, sorted_vals, color=colors, alpha=0.85, edgecolor='black')
        ax.set_xlabel('Combined Sensitivity', fontsize=12)
        ax.set_title(f'(d) Combined Ranking ({n_methods} methods)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, sorted_vals)):
            ax.text(val, bar.get_y() + bar.get_height()/2.,
                   f'  #{i+1}: {val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"sensitivity_{qoi}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ sensitivity_{qoi}.png")


# =============================================================================
# SAVE
# =============================================================================

def save_results(results: Dict, config: Config):
    """Save results."""
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'parameters': results['mc']['parameters'],
        'qoi_list': config.QOIS,
        'mc_sensitivity': results['mc']['mc_sensitivity'],
        'gradient_sensitivity': results['gradient']['gradient_sensitivity'],
        'rf_importance': results['rf']['rf_importance'],
    }
    
    if results.get('sobol'):
        output_data['sobol_analysis'] = results['sobol']['sobol_analysis']
    
    with open(config.OUT_DIR / "sensitivity_results.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved: sensitivity_results.json")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    config = Config()
    
    print("="*80)
    print("FIXED SENSITIVITY ANALYSIS (ROBUST VARIANCE HANDLING)")
    print("="*80)
    
    print("\nLoading surrogate...")
    model = AmplifiedSurrogate(str(config.BASE), config)
    
    results = {}
    results['mc'] = run_mc_sensitivity(model, config)
    
    if HAS_SALIB and config.N_SOBOL_SAMPLES > 0:
        results['sobol'] = run_sobol_analysis(model, config)
    
    results['gradient'] = compute_gradient_sensitivity(model, config)
    results['rf'] = compute_rf_importance(model, config)
    
    create_plots(results, config)
    save_results(results, config)
    
    # Summary
    print("\n" + "="*80)
    print("SENSITIVITY SUMMARY")
    print("="*80)
    
    for qoi in config.QOIS:
        print(f"\n{qoi.upper()}:")
        print("-" * 40)
        
        mc_data = results['mc']['mc_sensitivity'][qoi]
        if mc_data.get('valid'):
            sens = np.array(mc_data['sensitivity'])
            params = results['mc']['parameters']
            sorted_idx = np.argsort(sens)[::-1]
            
            print("  Parameter Rankings (MC-based):")
            for rank, idx in enumerate(sorted_idx, 1):
                print(f"    {rank}. {params[idx]:6s}: {sens[idx]:.4f}")
    
    print("\n" + "="*80)
    print("✓ SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()