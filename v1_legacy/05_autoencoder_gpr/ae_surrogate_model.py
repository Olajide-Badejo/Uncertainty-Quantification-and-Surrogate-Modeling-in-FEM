import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as stats_module
import warnings

warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


class Config:
    BASE = Path(__file__).resolve().parents[1]

    SURROGATE_TYPE = "AE+GPR"
    AE_DIR = BASE / "05_autoencoder_gpr"
    OUT_DIR = BASE / "07_processing" / "08_uncertainty_quantification_vector"

    N_MC_SAMPLES = 10000
    BATCH_SIZE = 256  # logical batch size at UQ level

    FC_DIST = {'type': 'normal', 'mean': 30.0, 'std': 2.5, 'bounds': (25.0, 35.0)}
    E_RELATIONSHIP = 'eurocode'
    C_BOT_DIST = {'type': 'normal', 'mean': 25.0, 'std': 2.0, 'bounds': (20.0, 30.0)}
    C_TOP_DIST = {'type': 'normal', 'mean': 215.0, 'std': 5.0, 'bounds': (200.0, 230.0)}

    FAILURE_THRESHOLD_SIGMA = 2.0
    SEED = 42


def sample_parameters(config: Config):
    np.random.seed(config.SEED)
    n = config.N_MC_SAMPLES

    fc = np.random.normal(config.FC_DIST['mean'], config.FC_DIST['std'], n)
    fc = np.clip(fc, *config.FC_DIST['bounds'])

    E = 22000 * (fc / 10.0) ** 0.3

    cbot = np.random.normal(config.C_BOT_DIST['mean'], config.C_BOT_DIST['std'], n)
    cbot = np.clip(cbot, *config.C_BOT_DIST['bounds'])

    ctop = np.random.normal(config.C_TOP_DIST['mean'], config.C_TOP_DIST['std'], n)
    ctop = np.clip(ctop, *config.C_TOP_DIST['bounds'])

    return fc, E, cbot, ctop


def load_surrogate(config: Config):
    import sys
    sys.path.insert(0, str(config.AE_DIR))
    from ae_surrogate_model import ImprovedAESurrogateModel

    model = ImprovedAESurrogateModel(str(config.BASE), use_improved=True)

    data_dir = config.AE_DIR / "data_preprocessed"
    model.global_Fmax = float(np.load(data_dir / "F_global_max.npy"))
    model.C_max_all = np.load(data_dir / "C_max.npy")
    model.u_force = np.load(data_dir / "u_force.npy")
    model.u_damage = np.load(data_dir / "u_crack.npy")

    return model


def predict_batch_scalar(model, cbot_batch, ctop_batch, fc_batch):
    """
    Batch wrapper around scalar predict().

    Currently loops internally, but if the surrogate is upgraded to accept arrays,
    this is the only function that needs to change.
    """
    n_b = len(fc_batch)
    u_force = model.u_force
    u_damage = model.u_damage

    F_norm_all = np.zeros((n_b, len(u_force)))
    D_norm_all = np.zeros((n_b, len(u_damage)))

    for i in range(n_b):
        _, F_norm, _, D_norm = model.predict(
            cbot=float(cbot_batch[i]),
            ctop=float(ctop_batch[i]),
            fcm=float(fc_batch[i])
        )
        F_norm_all[i] = F_norm
        D_norm_all[i] = D_norm

    return u_force, F_norm_all, u_damage, D_norm_all


def run_monte_carlo_vector(model, fc, E, cbot, ctop, config: Config):
    """
    Monte Carlo simulation using a batch driver.

    Still uses scalar surrogate calls internally, but structured so that
    true batching can be enabled later by modifying predict_batch_scalar().
    """
    n = len(fc)
    print(f"\nRunning Monte Carlo (vector/batch driver) with {n:,} samples...")

    # Probe first prediction
    u_f_test, F_test, u_d_test, D_test = model.predict(cbot=cbot[0], ctop=ctop[0], fcm=fc[0])
    n_force = len(F_test)
    n_damage = len(D_test)

    force_curves = np.zeros((n, n_force))
    damage_curves = np.zeros((n, n_damage))
    peak_forces = np.zeros(n)
    final_damages = np.zeros(n)

    np.random.seed(config.SEED + 1)
    C_max_indices = np.random.randint(0, len(model.C_max_all), size=n)

    t0 = time.time()
    n_fail = 0

    batch_size = config.BATCH_SIZE
    n_batches = int(np.ceil(n / batch_size))

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)

        fc_b = fc[start:end]
        E_b = E[start:end]          # currently unused, but kept for completeness
        cbot_b = cbot[start:end]
        ctop_b = ctop[start:end]
        C_idx_b = C_max_indices[start:end]

        try:
            u_f, F_norm_b, u_d, D_norm_b = predict_batch_scalar(model, cbot_b, ctop_b, fc_b)

            F_b = F_norm_b * model.global_Fmax
            D_b = D_norm_b * model.C_max_all[C_idx_b][:, None]

            force_curves[start:end] = F_b
            damage_curves[start:end] = D_b
            peak_forces[start:end] = np.max(F_b, axis=1)
            final_damages[start:end] = D_b[:, -1]
        except Exception:
            # Mark this batch as invalid
            force_curves[start:end] = np.nan
            damage_curves[start:end] = np.nan
            peak_forces[start:end] = np.nan
            final_damages[start:end] = np.nan
            n_fail += (end - start)

        if (b + 1) % max(1, n_batches // 10) == 0:
            done = end
            elapsed = (time.time() - t0) / 60
            eta = elapsed * n / done - elapsed
            print(f"  Batch {b+1}/{n_batches} | {done:,}/{n:,} ({done/n*100:.1f}%) | "
                  f"{elapsed:.1f} min | ETA: {eta:.1f} min")

    print(f"\n✓ Vector/batch Monte Carlo complete in {(time.time() - t0) / 60:.2f} min")
    if n_fail > 0:
        print(f"  Warning: {n_fail} surrogate evaluations failed and were discarded.")

    valid = ~np.isnan(peak_forces)
    n_valid = int(valid.sum())
    print(f"✓ Valid samples: {n_valid:,}/{n:,}")

    return {
        'force_curves': force_curves[valid],
        'damage_curves': damage_curves[valid],
        'peak_forces': peak_forces[valid],
        'final_damages': final_damages[valid],
        'fc_samples': fc[valid],
        'E_samples': E[valid],
        'cbot_samples': cbot[valid],
        'ctop_samples': ctop[valid],
        'n_valid': n_valid,
        'n_total': n,
        'computation_time': time.time() - t0,
        'u_force': model.u_force,
        'u_damage': model.u_damage
    }


def compute_stats(mc):
    return {
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
        },
        'final_damage': {
            'mean': float(np.mean(mc['final_damages'])),
            'std': float(np.std(mc['final_damages'])),
            'cov': float(np.std(mc['final_damages']) / np.mean(mc['final_damages'])),
            'median': float(np.median(mc['final_damages'])),
            'p05': float(np.percentile(mc['final_damages'], 5)),
            'p95': float(np.percentile(mc['final_damages'], 95)),
        }
    }


def compute_failures(mc, stats, config: Config):
    n = mc['n_valid']
    k = config.FAILURE_THRESHOLD_SIGMA

    pf_mean, pf_std = stats['peak_force']['mean'], stats['peak_force']['std']
    fd_mean, fd_std = stats['final_damage']['mean'], stats['final_damage']['std']

    low_thresh = pf_mean - k * pf_std
    high_thresh = fd_mean + k * fd_std

    low_fail = mc['peak_forces'] < low_thresh
    high_fail = mc['final_damages'] > high_thresh

    return {
        'low_capacity': {
            'probability': float(low_fail.sum() / n),
            'count': int(low_fail.sum()),
            'total': n,
            'threshold': float(low_thresh),
            'definition': f'Peak force < μ - {k}σ'
        },
        'high_damage': {
            'probability': float(high_fail.sum() / n),
            'count': int(high_fail.sum()),
            'total': n,
            'threshold': float(high_thresh),
            'definition': f'Final damage > μ + {k}σ'
        },
        'any_failure': {
            'probability': float(np.logical_or(low_fail, high_fail).sum() / n),
            'count': int(np.logical_or(low_fail, high_fail).sum()),
            'total': n
        }
    }


def create_plots(mc, stats, pf, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force UQ
    fig, ax = plt.subplots(figsize=(12, 7))
    u = stats['force']['u_grid']
    ax.fill_between(u, stats['force']['p05'], stats['force']['p95'], color='lightblue', alpha=0.5, label='90% CI')
    ax.fill_between(u, stats['force']['p25'], stats['force']['p75'], color='steelblue', alpha=0.5, label='50% CI')
    ax.plot(u, stats['force']['mean'], 'b-', lw=2.5, label='Mean')
    ax.plot(u, stats['force']['median'], 'r--', lw=2, label='Median')
    ax.set_xlabel('Displacement [mm]', fontsize=13)
    ax.set_ylabel('Force [N]', fontsize=13)
    ax.set_title(f'Force-Displacement UQ (n={mc["n_valid"]:,})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_force_uq.png", dpi=300)
    plt.close()

    # Damage UQ
    fig, ax = plt.subplots(figsize=(12, 7))
    u = stats['damage']['u_grid']
    ax.fill_between(u, stats['damage']['p05'], stats['damage']['p95'], color='#FFE5E5', alpha=0.7, label='90% CI')
    ax.fill_between(u, stats['damage']['p25'], stats['damage']['p75'], color='coral', alpha=0.6, label='50% CI')
    ax.plot(u, stats['damage']['mean'], 'r-', lw=2.5, label='Mean')
    ax.plot(u, stats['damage']['median'], 'darkred', ls='--', lw=2, label='Median')
    ax.set_xlabel('Displacement [mm]', fontsize=13)
    ax.set_ylabel('Compressive Damage [-]', fontsize=13)
    ax.set_title(f'Damage Evolution UQ (n={mc["n_valid"]:,})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_damage_uq.png", dpi=300)
    plt.close()

    # Peak force distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc['peak_forces'], bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='black')
    mu, sig = stats['peak_force']['mean'], stats['peak_force']['std']
    x = np.linspace(mu - 4 * sig, mu + 4 * sig, 200)
    ax.plot(x, stats_module.norm.pdf(x, mu, sig), 'r-', lw=2.0, label=f'N({mu:.0f}, {sig:.0f})')
    ax.axvline(stats['peak_force']['p05'], color='orange', ls='--', lw=1.5, label=f"5%: {stats['peak_force']['p05']:.0f}")
    ax.axvline(stats['peak_force']['p95'], color='orange', ls='--', lw=1.5, label=f"95%: {stats['peak_force']['p95']:.0f}")
    ax.axvline(pf['low_capacity']['threshold'], color='red', ls='-', lw=2, label=f"Threshold: {pf['low_capacity']['threshold']:.0f}")
    ax.set_xlabel('Peak Force [N]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Peak Force Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "03_peak_force_distribution.png", dpi=300)
    plt.close()

    # Final damage distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc['final_damages'], bins=50, density=True, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(pf['high_damage']['threshold'], color='red', ls='-', lw=2, label=f"Threshold: {pf['high_damage']['threshold']:.3f}")
    ax.axvline(stats['final_damage']['median'], color='darkred', ls='--', lw=2, label=f"Median: {stats['final_damage']['median']:.3f}")
    ax.set_xlabel('Final Damage [-]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Final Damage Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "04_final_damage_distribution.png", dpi=300)
    plt.close()

    # Failure probabilities
    fig, ax = plt.subplots(figsize=(8, 6))
    modes = ['Low\nCapacity', 'High\nDamage', 'Safe']
    probs = [
        pf['low_capacity']['probability'] * 100,
        pf['high_damage']['probability'] * 100,
        (1 - pf['any_failure']['probability']) * 100
    ]
    colors = ['#FF6B6B', '#FFD93D', '#95E1D3']
    bars = ax.bar(modes, probs, color=colors, alpha=0.8, edgecolor='black', lw=1.5)
    ax.set_ylabel('Probability [%]', fontsize=12)
    ax.set_title(f'Failure Probabilities (n={mc["n_valid"]:,})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{p:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "05_failure_probabilities.png", dpi=300)
    plt.close()

    # Input-output correlations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    inputs = [
        ('fc', mc['fc_samples'], 'Concrete Strength [MPa]'),
        ('E', mc['E_samples'], "Young's Modulus [MPa]"),
        ('c_bot', mc['cbot_samples'], 'Bottom Cover [mm]'),
        ('c_top', mc['ctop_samples'], 'Top Cover [mm]')
    ]
    for idx, (ax, (name, samp, label)) in enumerate(zip(axes.flat, inputs)):
        sc = ax.scatter(samp, mc['peak_forces'], c=mc['final_damages'],
                        cmap='RdYlBu_r', alpha=0.3, s=10)
        corr = np.corrcoef(samp, mc['peak_forces'])[0, 1]
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Peak Force [N]', fontsize=11)
        ax.set_title(f'Correlation: {corr:.3f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 3:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Final Damage', fontsize=10)
    plt.suptitle('Input-Output Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "06_input_output_correlations.png", dpi=300)
    plt.close()

    print("\n✓ All plots saved")


def save_results(mc, stats, pf, config: Config):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.OUT_DIR / "uq_statistics.json", 'w') as f:
        json.dump({
            'configuration': {
                'n_samples': mc['n_total'],
                'n_valid': mc['n_valid'],
                'computation_time_minutes': mc['computation_time'] / 60,
                'surrogate_type': config.SURROGATE_TYPE,
                'qois': ['peak_force', 'final_damage']
            },
            'peak_force': stats['peak_force'],
            'final_damage': stats['final_damage'],
            'failure_probabilities': pf
        }, f, indent=2)

    pd.DataFrame({
        'displacement': stats['force']['u_grid'],
        'mean': stats['force']['mean'],
        'median': stats['force']['median'],
        'std': stats['force']['std'],
        'p05': stats['force']['p05'],
        'p25': stats['force']['p25'],
        'p75': stats['force']['p75'],
        'p95': stats['force']['p95'],
    }).to_csv(config.OUT_DIR / "force_uq_curves.csv", index=False)

    pd.DataFrame({
        'displacement': stats['damage']['u_grid'],
        'mean': stats['damage']['mean'],
        'median': stats['damage']['median'],
        'std': stats['damage']['std'],
        'p05': stats['damage']['p05'],
        'p25': stats['damage']['p25'],
        'p75': stats['damage']['p75'],
        'p95': stats['damage']['p95'],
    }).to_csv(config.OUT_DIR / "damage_uq_curves.csv", index=False)


def main():
    config = Config()
    plots_dir = config.OUT_DIR / "uq_plots"

    print("=" * 80)
    print("UNCERTAINTY QUANTIFICATION - VECTOR/BATCH DRIVER (IMPROVED)")
    print("=" * 80)

    model = load_surrogate(config)
    print("✓ Surrogate loaded")

    fc, E, cbot, ctop = sample_parameters(config)

    mc = run_monte_carlo_vector