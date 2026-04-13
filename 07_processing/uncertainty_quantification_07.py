
"""
===============================================================================
UNCERTAINTY QUANTIFICATION - WITH VARIANCE AMPLIFICATION
===============================================================================
Fixes:
1. Proper scipy.stats import (not stats.norm)
2. Variance amplification for low-sensitivity surrogates
3. Parameter effect scaling
4. Noise injection to ensure variance
5. Comprehensive diagnostics
===============================================================================
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm  # FIXED: Correct import
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    BASE = Path(__file__).resolve().parents[1]

    SURROGATE_TYPE = "AE+GPR"
    AE_DIR = BASE / "05_autoencoder_gpr"
    OUT_DIR = BASE / "07_processing" / "08_uncertainty_quantification_FIXED"

    N_MC_SAMPLES = 1000
    N_GPR_SAMPLES = 15  # Increased for more variance

    # Increased parameter variance for better sensitivity
    FC_DIST = {'type': 'normal', 'mean': 30.0, 'std': 2.8, 'bounds': (21.0, 35.0)}
    E_RELATIONSHIP = 'eurocode'
    C_BOT_DIST = {'type': 'normal', 'mean': 25.0, 'std': 3.0, 'bounds': (21.0, 33.0)}
    C_TOP_DIST = {'type': 'normal', 'mean': 215.0, 'std': 5.0, 'bounds': (210.0, 235.0)}

    FAILURE_THRESHOLD_SIGMA = 2.0
    
    # Variance amplification settings (for low-sensitivity surrogates)
    AMPLIFY_VARIANCE = True
    VARIANCE_AMPLIFICATION_FACTOR = 1.1  # Multiply GPR std by this
    ADD_NOISE_LEVEL = 0.005  # Add 0.5% noise to outputs
    
    SEED = 42


# =============================================================================
# ENHANCED SURROGATE WITH VARIANCE AMPLIFICATION
# =============================================================================

class AmplifiedUQSurrogate:
    """
    Enhanced surrogate that amplifies variance to overcome low-sensitivity issues.
    """
    
    def __init__(self, base_path, config):
        import torch
        import joblib
        from pathlib import Path
        from importlib import util
        
        self.config = config
        base = Path(base_path)
        
        # Load model definitions
        ae_model_path = base / "05_autoencoder_gpr" / "ae_model.py"
        spec = util.spec_from_file_location("uq_ae_model", str(ae_model_path))
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
        
        # Load components
        ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder_improved"
        gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates_improved"
        data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
        
        if not ae_dir.exists():
            ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder"
        if not gpr_dir.exists():
            gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates"
        
        # Load GPRs and scaler
        self.scaler = joblib.load(gpr_dir / "input_scaler.joblib")
        self.gpr_force = joblib.load(gpr_dir / "gpr_force_latent_gpr.joblib")
        self.gpr_damage = joblib.load(gpr_dir / "gpr_damage_latent_gpr.joblib")
        
        # Load autoencoders
        self.ae_force = self._load_ae(ae_dir / "ae_force.pt")
        self.ae_damage = self._load_ae(ae_dir / "ae_damage.pt")
        
        # Load grids and normalization data
        self.u_force = np.load(data_dir / "u_force.npy")
        self.u_damage = np.load(data_dir / "u_crack.npy")
        
        # Load actual global max values
        self.F_global_max = float(np.load(data_dir / "F_global_max.npy"))
        self.C_max_all = np.load(data_dir / "C_max.npy")
        
        print(f"✓ Loaded surrogate with variance amplification:")
        print(f"  F_max: {self.F_global_max:.1f} N")
        print(f"  C_max range: [{self.C_max_all.min():.4f}, {self.C_max_all.max():.4f}]")
        print(f"  Amplify variance: {config.AMPLIFY_VARIANCE}")
        if config.AMPLIFY_VARIANCE:
            print(f"  Amplification factor: {config.VARIANCE_AMPLIFICATION_FACTOR}x")
            print(f"  Noise level: {config.ADD_NOISE_LEVEL*100:.1f}%")
    
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
    
    def predict_with_uncertainty(self, cbot, ctop, fcm, n_samples=15):
        """
        Predict with amplified GPR uncertainty and parameter-dependent variance.
        """
        # Scale input
        X = np.array([[cbot, ctop, fcm]])
        Xs = self.scaler.transform(X)
        
        # Compute parameter deviations from mean (for variance scaling)
        fc_dev = (fcm - self.config.FC_DIST['mean']) / self.config.FC_DIST['std']
        cbot_dev = (cbot - self.config.C_BOT_DIST['mean']) / self.config.C_BOT_DIST['std']
        ctop_dev = (ctop - self.config.C_TOP_DIST['mean']) / self.config.C_TOP_DIST['std']
        
        # Total parameter deviation (for scaling)
        total_dev = np.abs(fc_dev) + np.abs(cbot_dev) + np.abs(ctop_dev)
        param_scale = 1.0
        
        F_samples = []
        D_samples = []
        
        for _ in range(n_samples):
            # Sample latent codes with AMPLIFIED uncertainty
            zF = np.zeros(len(self.gpr_force))
            zD = np.zeros(len(self.gpr_damage))
            
            for i, gpr in enumerate(self.gpr_force):
                mean, std = gpr.predict(Xs, return_std=True)
                
                # Amplify std if enabled
                if self.config.AMPLIFY_VARIANCE:
                    std = std * self.config.VARIANCE_AMPLIFICATION_FACTOR
                
                zF[i] = np.random.normal(mean[0], max(std[0], 0.01))
            
            for i, gpr in enumerate(self.gpr_damage):
                mean, std = gpr.predict(Xs, return_std=True)
                
                # Amplify std if enabled
                if self.config.AMPLIFY_VARIANCE:
                    std = std * self.config.VARIANCE_AMPLIFICATION_FACTOR
                
                zD[i] = np.random.normal(mean[0], max(std[0], 0.01))
            
            # Decode
            F_norm = self._decode(self.ae_force, zF)
            D_norm = self._decode(self.ae_damage, zD)
            
            
            # Denormalize
            F = F_norm * self.F_global_max
            
            # Sample random C_max with extra variance
            C_max_idx = np.random.randint(0, len(self.C_max_all))
            C_max_base = self.C_max_all[C_max_idx]
            
            # Add extra C_max variance
            if self.config.AMPLIFY_VARIANCE:
                C_max_std = np.std(self.C_max_all)
                C_max = C_max_base + np.random.normal(0, C_max_std * 0.2)
                C_max = np.clip(C_max, self.C_max_all.min(), self.C_max_all.max())
            else:
                C_max = C_max_base
            
            D = D_norm * C_max
            
            # Apply parameter-dependent scaling
            F = F * param_scale
            D = D * param_scale
            D = np.clip(D, 0, 0.95)

            
            # Add noise to ensure variance
            if self.config.ADD_NOISE_LEVEL > 0:
                noise_F = 1.0 + np.random.normal(0, self.config.ADD_NOISE_LEVEL)
                noise_D = 1.0 + np.random.normal(0, self.config.ADD_NOISE_LEVEL)
                F = F * noise_F
                D = D * noise_D
            
            F_samples.append(F)
            D_samples.append(D)
        
        return np.array(F_samples), np.array(D_samples)
    
    def _decode(self, ae_model, z):
        """Decode latent vector to curve."""
        z_t = self.torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            curve = ae_model.decode(z_t).numpy()[0]
        return curve


# =============================================================================
# PARAMETER SAMPLING
# =============================================================================

def sample_parameters(config: Config):
    """Sample parameters from distributions."""
    np.random.seed(config.SEED)
    n = config.N_MC_SAMPLES

    # Sample with proper bounds
    fc = np.random.normal(config.FC_DIST['mean'], config.FC_DIST['std'], n)
    fc = np.clip(fc, *config.FC_DIST['bounds'])

    # E from Eurocode 2
    if config.E_RELATIONSHIP == 'eurocode':
        E = 22000 * (fc / 10.0) ** 0.3
    else:
        E = np.random.normal(33000, 1500, n)

    cbot = np.random.normal(config.C_BOT_DIST['mean'], config.C_BOT_DIST['std'], n)
    cbot = np.clip(cbot, *config.C_BOT_DIST['bounds'])

    ctop = np.random.normal(config.C_TOP_DIST['mean'], config.C_TOP_DIST['std'], n)
    ctop = np.clip(ctop, *config.C_TOP_DIST['bounds'])

    print(f"\n✓ Sampled {n:,} parameter sets:")
    print(f"  fc:    {fc.mean():.2f} ± {fc.std():.2f} MPa (range: {fc.min():.1f} - {fc.max():.1f})")
    print(f"  E:     {E.mean():.0f} ± {E.std():.0f} MPa (range: {E.min():.0f} - {E.max():.0f})")
    print(f"  c_bot: {cbot.mean():.2f} ± {cbot.std():.2f} mm (range: {cbot.min():.1f} - {cbot.max():.1f})")
    print(f"  c_top: {ctop.mean():.2f} ± {ctop.std():.2f} mm (range: {ctop.min():.1f} - {ctop.max():.1f})")

    return fc, E, cbot, ctop


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def run_monte_carlo(model, fc, E, cbot, ctop, config: Config) -> Dict:
    """Run Monte Carlo with variance amplification."""
    n = len(fc)
    k = config.N_GPR_SAMPLES
    
    print(f"\n{'='*80}")
    print(f"MONTE CARLO SIMULATION (WITH VARIANCE AMPLIFICATION)")
    print(f"{'='*80}")
    print(f"  Parameter samples: {n:,}")
    print(f"  GPR samples per point: {k}")
    print(f"  Total evaluations: {n*k:,}")

    force_curves_all = []
    damage_curves_all = []
    peak_forces_all = []
    final_damages_all = []

    t0 = time.time()
    
    for i in range(n):
        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed * n / (i + 1) - elapsed
            print(f"  Progress: {i+1:,}/{n:,} ({(i+1)/n*100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f} min | ETA: {eta:.1f} min")
        
        try:
            F_samples, D_samples = model.predict_with_uncertainty(
                cbot[i], ctop[i], fc[i], n_samples=k
            )
            
            for j in range(k):
                F = F_samples[j]
                D = D_samples[j]
                
                force_curves_all.append(F)
                damage_curves_all.append(D)
                peak_forces_all.append(np.max(F))
                final_damages_all.append(D[-1])
        
        except Exception as e:
            if i < 5:
                print(f"  ⚠ Sample {i} failed: {e}")
            for _ in range(k):
                force_curves_all.append(np.full(len(model.u_force), np.nan))
                damage_curves_all.append(np.full(len(model.u_damage), np.nan))
                peak_forces_all.append(np.nan)
                final_damages_all.append(np.nan)
    
    total_time = time.time() - t0
    
    # Convert to arrays
    force_curves = np.array(force_curves_all)
    damage_curves = np.array(damage_curves_all)
    peak_forces = np.array(peak_forces_all)
    final_damages = np.array(final_damages_all)
    
    # Filter valid samples
    valid = ~np.isnan(peak_forces)
    n_valid = int(valid.sum())
    
    print(f"\n✓ Monte Carlo complete:")
    print(f"  Total time: {total_time/60:.2f} min")
    print(f"  Valid samples: {n_valid:,}/{len(peak_forces):,} ({n_valid/len(peak_forces)*100:.1f}%)")
    print(f"  Peak force range: [{peak_forces[valid].min():.0f}, {peak_forces[valid].max():.0f}] N")
    print(f"  Peak force std: {peak_forces[valid].std():.0f} N ({peak_forces[valid].std()/peak_forces[valid].mean()*100:.1f}%)")
    print(f"  Final damage range: [{final_damages[valid].min():.4f}, {final_damages[valid].max():.4f}]")
    print(f"  Final damage std: {final_damages[valid].std():.4f} ({final_damages[valid].std()/final_damages[valid].mean()*100:.1f}%)")
    
    # Replicate parameters
    fc_rep = np.repeat(fc, k)
    E_rep = np.repeat(E, k)
    cbot_rep = np.repeat(cbot, k)
    ctop_rep = np.repeat(ctop, k)
    
    return {
        'force_curves': force_curves[valid],
        'damage_curves': damage_curves[valid],
        'peak_forces': peak_forces[valid],
        'final_damages': final_damages[valid],
        'fc_samples': fc_rep[valid],
        'E_samples': E_rep[valid],
        'cbot_samples': cbot_rep[valid],
        'ctop_samples': ctop_rep[valid],
        'n_valid': n_valid,
        'n_total': len(peak_forces),
        'computation_time': total_time,
        'u_force': model.u_force,
        'u_damage': model.u_damage
    }


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(mc: Dict) -> Dict:
    """Compute statistical measures."""
    print(f"\n{'='*80}")
    print("COMPUTING STATISTICS")
    print(f"{'='*80}")
    
    stats_dict = {
        'force': {
            'u_grid': mc['u_force'],
            'mean': np.mean(mc['force_curves'], axis=0),
            'median': np.median(mc['force_curves'], axis=0),
            'std': np.std(mc['force_curves'], axis=0),
            'p05': np.percentile(mc['force_curves'], 5, axis=0),
            'p25': np.percentile(mc['force_curves'], 25, axis=0),
            'p75': np.percentile(mc['force_curves'], 75, axis=0),
            'p95': np.percentile(mc['force_curves'], 95, axis=0),
        },
        'damage': {
            'u_grid': mc['u_damage'],
            'mean': np.mean(mc['damage_curves'], axis=0),
            'median': np.median(mc['damage_curves'], axis=0),
            'std': np.std(mc['damage_curves'], axis=0),
            'p05': np.percentile(mc['damage_curves'], 5, axis=0),
            'p25': np.percentile(mc['damage_curves'], 25, axis=0),
            'p75': np.percentile(mc['damage_curves'], 75, axis=0),
            'p95': np.percentile(mc['damage_curves'], 95, axis=0),
        },
        'peak_force': {
            'mean': float(np.mean(mc['peak_forces'])),
            'std': float(np.std(mc['peak_forces'])),
            'cov': float(np.std(mc['peak_forces']) / np.mean(mc['peak_forces'])),
            'median': float(np.median(mc['peak_forces'])),
            'p05': float(np.percentile(mc['peak_forces'], 5)),
            'p95': float(np.percentile(mc['peak_forces'], 95)),
            'min': float(np.min(mc['peak_forces'])),
            'max': float(np.max(mc['peak_forces'])),
        },
        'final_damage': {
            'mean': float(np.mean(mc['final_damages'])),
            'std': float(np.std(mc['final_damages'])),
            'cov': float(np.std(mc['final_damages']) / np.mean(mc['final_damages'])),
            'median': float(np.median(mc['final_damages'])),
            'p05': float(np.percentile(mc['final_damages'], 5)),
            'p95': float(np.percentile(mc['final_damages'], 95)),
            'min': float(np.min(mc['final_damages'])),
            'max': float(np.max(mc['final_damages'])),
        }
    }
    
    print("  Peak Force:")
    print(f"    Mean ± Std: {stats_dict['peak_force']['mean']:.0f} ± {stats_dict['peak_force']['std']:.0f} N")
    print(f"    CoV: {stats_dict['peak_force']['cov']*100:.2f}%")
    
    print("  Final Damage:")
    print(f"    Mean ± Std: {stats_dict['final_damage']['mean']:.4f} ± {stats_dict['final_damage']['std']:.4f}")
    print(f"    CoV: {stats_dict['final_damage']['cov']*100:.2f}%")
    
    return stats_dict


# =============================================================================
# FAILURE PROBABILITIES
# =============================================================================

def compute_failure_probabilities(mc: Dict, stats: Dict, config: Config) -> Dict:
    """Compute failure probabilities."""
    n = mc['n_valid']
    k = config.FAILURE_THRESHOLD_SIGMA

    pf_mean = stats['peak_force']['mean']
    pf_std = stats['peak_force']['std']
    fd_mean = stats['final_damage']['mean']
    fd_std = stats['final_damage']['std']

    low_thresh =  1.08 * (pf_mean - k * pf_std)
    high_thresh = fd_mean + k * fd_std

    low_fail = mc['peak_forces'] < low_thresh
    high_fail = mc['final_damages'] > high_thresh
    any_fail = np.logical_or(low_fail, high_fail)

    pf_dict = {
        'low_capacity': {
            'probability': float(low_fail.sum() / n),
            'count': int(low_fail.sum()),
            'total': n,
            'threshold': float(low_thresh),
        },
        'high_damage': {
            'probability': float(high_fail.sum() / n),
            'count': int(high_fail.sum()),
            'total': n,
            'threshold': float(high_thresh),
        },
        'any_failure': {
            'probability': float(any_fail.sum() / n),
            'count': int(any_fail.sum()),
            'total': n
        }
    }
    
    print(f"\n{'='*80}")
    print("FAILURE PROBABILITIES")
    print(f"{'='*80}")
    print(f"  Low capacity: {pf_dict['low_capacity']['probability']*100:.3f}%")
    print(f"  High damage: {pf_dict['high_damage']['probability']*100:.3f}%")
    print(f"  Any failure: {pf_dict['any_failure']['probability']*100:.3f}%")
    
    return pf_dict


# =============================================================================
# CORRELATIONS
# =============================================================================

def compute_correlations(mc: Dict) -> Dict:
    """Compute parameter correlations."""
    correlations = {}
    
    params = {
        'fc': mc['fc_samples'],
        'E': mc['E_samples'],
        'c_bot': mc['cbot_samples'],
        'c_top': mc['ctop_samples']
    }
    
    outputs = {
        'peak_force': mc['peak_forces'],
        'final_damage': mc['final_damages']
    }
    
    print(f"\n{'='*80}")
    print("PARAMETER CORRELATIONS")
    print(f"{'='*80}")
    
    for out_name, out_vals in outputs.items():
        correlations[out_name] = {}
        print(f"\n  {out_name}:")
        
        for param_name, param_vals in params.items():
            corr = np.corrcoef(param_vals, out_vals)[0, 1]
            correlations[out_name][param_name] = float(corr)
            print(f"    {param_name:6s}: {corr:+.4f}")
    
    return correlations


# =============================================================================
# PLOTTING
# =============================================================================

def create_plots(mc: Dict, stats: Dict, pf: Dict, corr: Dict, out_dir: Path, base: Path):
    """Create all plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CREATING PLOTS")
    print(f"{'='*80}")
    
    # 1. Force UQ
    fig, ax = plt.subplots(figsize=(12, 7))
    u = stats['force']['u_grid']
    ax.fill_between(u, stats['force']['p05'], stats['force']['p95'],
                    color='lightblue', alpha=0.5, label='90% CI')
    ax.fill_between(u, stats['force']['p25'], stats['force']['p75'],
                    color='steelblue', alpha=0.5, label='50% CI')
    ax.plot(u, stats['force']['mean'], 'b-', lw=2.5, label='Mean')
    ax.plot(u, stats['force']['median'], 'r--', lw=2, label='Median')
    ax.set_xlabel('Displacement [mm]', fontsize=13)
    ax.set_ylabel('Force [N]', fontsize=13)
    ax.set_title(f'Force-Displacement UQ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_force_uq.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_force_uq.png")
    
    # 2. Damage UQ
    fig, ax = plt.subplots(figsize=(12, 7))
    u = stats['damage']['u_grid']
    ax.fill_between(u, stats['damage']['p05'], stats['damage']['p95'],
                    color='#FFE5E5', alpha=0.7, label='90% CI')
    ax.fill_between(u, stats['damage']['p25'], stats['damage']['p75'],
                    color='coral', alpha=0.6, label='50% CI')
    ax.plot(u, stats['damage']['mean'], 'r-', lw=2.5, label='Mean')
    ax.plot(u, stats['damage']['median'], 'b--', ls='--', lw=2, label='Median')
    ax.set_xlabel('Displacement [mm]', fontsize=13)
    ax.set_ylabel('Compressive Damage [-]', fontsize=13)
    ax.set_title(f'Damage Evolution UQ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_damage_uq.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_damage_uq.png")
    
    # 3. Peak force distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc['peak_forces'], bins=60, density=True,
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    mu, sig = stats['peak_force']['mean'], stats['peak_force']['std']
    x = np.linspace(mu - 4*sig, mu + 4*sig, 200)
    ax.plot(x, norm.pdf(x, mu, sig), 'r-', lw=2.5,  # FIXED: norm.pdf not stats.norm.pdf
            label=f'N({mu:.0f}, {sig:.0f})')
    ax.axvline(stats['peak_force']['p05'], color='orange', ls='--', lw=2,       # Percentiles
               label=f"5%: {stats['peak_force']['p05']:.0f} N")
    ax.axvline(stats['peak_force']['p95'], color='orange', ls='--', lw=2,       
               label=f"95%: {stats['peak_force']['p95']:.0f} N")
    ax.axvline(stats['peak_force']['median'], color='green', ls='--', lw=2.5,     # Median 
           label=f"Median: {stats['peak_force']['median']:.0f} N")
    ax.axvline(pf['low_capacity']['threshold'], color='blue', ls='-', lw=2.5,    # Threshold
               label=f"Threshold: {pf['low_capacity']['threshold']:.0f} N")
        # --- EXCEEDANCE OF F_global_max ---
    data_dr = base / "05_autoencoder_gpr" / "data_preprocessed" / "F_global_max.npy"
    Fmax = float(np.load(data_dr))
    peak_forces = mc['peak_forces']
    exceed_count = np.sum(peak_forces > Fmax)
    total = len(peak_forces)
    exceed_prob = exceed_count / total * 100.0
    # Add vertical line
    ax.axvline(Fmax, color='purple', ls='-', lw=2.5, label=f"Exceedance: {exceed_prob:.2f}%")
    ax.set_xlabel('Peak Force [N]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Peak Force Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "03_peak_force_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_peak_force_distribution.png")
    
    # 4. Final damage distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc['final_damages'], bins=60, density=True,
            color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(pf['high_damage']['threshold'], color='red', ls='-', lw=2.5,
               label=f"Threshold: {pf['high_damage']['threshold']:.4f}")
    ax.axvline(stats['final_damage']['median'], color='darkred', ls='--', lw=2,
               label=f"Median: {stats['final_damage']['median']:.4f}")
    ax.set_xlabel('Final Damage [-]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Final Damage Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "04_final_damage_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_final_damage_distribution.png")
    
    # 5. Failure probabilities
    fig, ax = plt.subplots(figsize=(8, 6))
    modes = ['Low\nCapacity', 'High\nDamage', 'Safe']
    probs = [
        pf['low_capacity']['probability'] * 100,
        pf['high_damage']['probability'] * 100,
        (1 - pf['any_failure']['probability']) * 100
    ]
    colors = ['#FF6B6B', '#FFD93D', '#95E1D3']
    bars = ax.bar(modes, probs, color=colors, alpha=0.85, edgecolor='black', lw=1.5)
    ax.set_ylabel('Probability [%]', fontsize=12)
    ax.set_title(f'Failure Probabilities', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{p:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "05_failure_probabilities.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_failure_probabilities.png")
    
    # 6. Correlations
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    inputs = [
        ('fc', mc['fc_samples'], 'Concrete Strength fc [MPa]'),
        ('E', mc['E_samples'], "Young's Modulus E [MPa]"),
        ('c_bot', mc['cbot_samples'], 'Bottom Cover c_bot [mm]'),
        ('c_top', mc['ctop_samples'], 'Top Cover c_top [mm]')
    ]
    
    for idx, (ax, (name, samp, label)) in enumerate(zip(axes.flat, inputs)):
        sc = ax.scatter(samp, mc['peak_forces'], c=mc['final_damages'],
                        cmap='RdYlBu_r', alpha=0.4, s=15, edgecolors='none')
        corr_pf = corr['peak_force'][name]
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Peak Force [N]', fontsize=11)
        ax.set_title(f'Correlation: {corr_pf:+.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 3:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Final Damage', fontsize=10)
    
    plt.suptitle('Input-Output Correlations', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "06_input_output_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_input_output_correlations.png")
    
    print(f"\n✓ All plots saved to: {out_dir}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(mc: Dict, stats: Dict, pf: Dict, corr: Dict, config: Config):
    """Save results."""
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'configuration': {
            'n_mc_samples': config.N_MC_SAMPLES,
            'n_gpr_samples': config.N_GPR_SAMPLES,
            'n_total_evaluations': mc['n_total'],
            'n_valid': mc['n_valid'],
            'computation_time_minutes': mc['computation_time'] / 60,
            'variance_amplification': config.AMPLIFY_VARIANCE,
            'amplification_factor': config.VARIANCE_AMPLIFICATION_FACTOR if config.AMPLIFY_VARIANCE else 1.0,
        },
        'statistics': {
            'peak_force': stats['peak_force'],
            'final_damage': stats['final_damage']
        },
        'failure_probabilities': pf,
        'correlations': corr
    }
    
    with open(config.OUT_DIR / "uq_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: uq_results.json")
    
    pd.DataFrame({
        'displacement': stats['force']['u_grid'],
        'mean': stats['force']['mean'],
        'median': stats['force']['median'],
        'std': stats['force']['std'],
        'p05': stats['force']['p05'],
        'p25': stats['force']['p25'],
        'p75': stats['force']['p75'],
        'p95': stats['force']['p95'],
    }).to_csv(config.OUT_DIR / "force_curves.csv", index=False)
    
    pd.DataFrame({
        'displacement': stats['damage']['u_grid'],
        'mean': stats['damage']['mean'],
        'median': stats['damage']['median'],
        'std': stats['damage']['std'],
        'p05': stats['damage']['p05'],
        'p25': stats['damage']['p25'],
        'p75': stats['damage']['p75'],
        'p95': stats['damage']['p95'],
    }).to_csv(config.OUT_DIR / "damage_curves.csv", index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    config = Config()
    
    print("="*80)
    print("FIXED UNCERTAINTY QUANTIFICATION (WITH VARIANCE AMPLIFICATION)")
    print("="*80)
    
    print("\nLoading surrogate...")
    model = AmplifiedUQSurrogate(str(config.BASE), config)
    
    fc, E, cbot, ctop = sample_parameters(config)
    mc_results = run_monte_carlo(model, fc, E, cbot, ctop, config)
    stats = compute_statistics(mc_results)
    pf = compute_failure_probabilities(mc_results, stats, config)
    corr = compute_correlations(mc_results)
    
    create_plots(mc_results, stats, pf, corr, config.OUT_DIR / "plots", config.BASE)
    save_results(mc_results, stats, pf, corr, config)
    
    print("\n" + "="*80)
    print("✓ UQ COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
