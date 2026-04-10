"""
===============================================================================
VISUALIZATION - SAMPLING QUALITY & DISTRIBUTION VERIFICATION
===============================================================================
Purpose: Create comprehensive plots to verify sampling quality and distributions

Plots generated:
    1. Individual parameter distributions (histogram + PDF + CDF)
    2. Correlation/independence verification (scatter matrices)
    3. Sampling method comparison (LHS vs Random, Sobol, etc.)
    4. Sample coverage analysis
    5. Statistical validation plots (Q-Q plots)
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, probplot
import seaborn as sns

# Set professional plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "uq_lhs_samples_training.csv"
SAVE_DIR = ROOT / "01_samplying" / "Sampling_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Distribution parameters (must match sampling script)
FCM_MEAN = 28.0
FCM_COV = 0.10
C_BOT_MEAN = 27.0
C_BOT_STD = 3.0
C_TOP_MEAN = 223.0
C_TOP_STD = 5.0

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} samples from {INPUT_CSV}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lognormal_params(mean, cov):
    """Convert lognormal mean/CoV to mu/sigma"""
    variance = (cov * mean) ** 2
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

mu_f, sigma_f = lognormal_params(FCM_MEAN, FCM_COV)

# ============================================================
# PLOT 1: COMBINED HISTOGRAM + PDF + CDF
# ============================================================

def plot_distribution_validation(variable_name, samples, x_range, 
                                  pdf_func, cdf_func, xlabel, unit, filename):
    """
    Create side-by-side histogram+PDF and CDF plots
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Histogram + PDF overlay
    axs[0].hist(samples, bins=30, density=True, color='lightgray', 
                edgecolor='black', alpha=0.7, label="Samples")
    axs[0].plot(x_range, pdf_func(x_range), 'r-', linewidth=2, label="Theoretical PDF")
    axs[0].set_title(f"{variable_name} - PDF")
    axs[0].set_xlabel(f"{xlabel} [{unit}]")
    axs[0].set_ylabel("Probability density [-]")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Right: CDF
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    
    axs[1].plot(sorted_samples, empirical_cdf, 'b-', linewidth=1.5, 
                label="Empirical CDF", alpha=0.7)
    axs[1].plot(x_range, cdf_func(x_range), 'r--', linewidth=2, 
                label="Theoretical CDF")
    axs[1].set_title(f"{variable_name} - CDF")
    axs[1].set_xlabel(f"{xlabel} [{unit}]")
    axs[1].set_ylabel("Cumulative probability [-]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filename}")


# Concrete strength (lognormal)
x_fcm = np.linspace(df['Fcm_MPa'].min() * 0.8, df['Fcm_MPa'].max() * 1.2, 400)
plot_distribution_validation(
    variable_name="Concrete Strength",
    samples=df['Fcm_MPa'].values,
    x_range=x_fcm,
    pdf_func=lambda x: lognorm.pdf(x, s=sigma_f, scale=np.exp(mu_f)),
    cdf_func=lambda x: lognorm.cdf(x, s=sigma_f, scale=np.exp(mu_f)),
    xlabel="$f_{cm}$",
    unit="MPa",
    filename="01_fcm_distribution.png"
)

# Bottom cover (normal)
x_bot = np.linspace(df['c_nom_bottom_mm'].min() * 0.9, 
                    df['c_nom_bottom_mm'].max() * 1.1, 400)
plot_distribution_validation(
    variable_name="Bottom Cover",
    samples=df['c_nom_bottom_mm'].values,
    x_range=x_bot,
    pdf_func=lambda x: norm.pdf(x, loc=C_BOT_MEAN, scale=C_BOT_STD),
    cdf_func=lambda x: norm.cdf(x, loc=C_BOT_MEAN, scale=C_BOT_STD),
    xlabel="$c_{bot}$",
    unit="mm",
    filename="02_c_bot_distribution.png"
)

# Top cover (normal)
x_top = np.linspace(df['c_nom_top_mm'].min() * 0.95, 
                    df['c_nom_top_mm'].max() * 1.05, 400)
plot_distribution_validation(
    variable_name="Top Cover",
    samples=df['c_nom_top_mm'].values,
    x_range=x_top,
    pdf_func=lambda x: norm.pdf(x, loc=C_TOP_MEAN, scale=C_TOP_STD),
    cdf_func=lambda x: norm.cdf(x, loc=C_TOP_MEAN, scale=C_TOP_STD),
    xlabel="$c_{top}$",
    unit="mm",
    filename="03_c_top_distribution.png"
)

# ============================================================
# PLOT 2: Q-Q PLOTS (DISTRIBUTION VERIFICATION)
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# fcm (lognormal)
probplot(np.log(df['Fcm_MPa']), dist="norm", plot=axs[0])
axs[0].set_title("Q-Q Plot: $\ln(f_{cm})$ vs Normal")
axs[0].grid(True, alpha=0.3)

# c_bot (normal)
probplot(df['c_nom_bottom_mm'], dist="norm", plot=axs[1])
axs[1].set_title("Q-Q Plot: $c_{bot}$ vs Normal")
axs[1].grid(True, alpha=0.3)

# c_top (normal)
probplot(df['c_nom_top_mm'], dist="norm", plot=axs[2])
axs[2].set_title("Q-Q Plot: $c_{top}$ vs Normal")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "04_qq_plots.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 04_qq_plots.png")

# ============================================================
# PLOT 3: CORRELATION MATRIX (VERIFY INDEPENDENCE)
# ============================================================

# Select relevant columns
corr_data = df[['Fcm_MPa', 'c_nom_bottom_mm', 'c_nom_top_mm', 'E_MPa']]

# Compute correlation
corr_matrix = corr_data.corr()

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            xticklabels=['$f_{cm}$', '$c_{bot}$', '$c_{top}$', '$E$'],
            yticklabels=['$f_{cm}$', '$c_{bot}$', '$c_{top}$', '$E$'])
ax.set_title("Correlation Matrix (Independence Verification)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "05_correlation_matrix.png"), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 05_correlation_matrix.png")

# ============================================================
# PLOT 4: SCATTER MATRIX (PAIRWISE INDEPENDENCE)
# ============================================================

fig = plt.figure(figsize=(10, 10))
pd.plotting.scatter_matrix(corr_data, alpha=0.6, figsize=(10, 10), 
                           diagonal='hist', edgecolors='k', s=15)
plt.suptitle("Scatter Matrix - Pairwise Independence Check", y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "06_scatter_matrix.png"), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 06_scatter_matrix.png")

# ============================================================
# PLOT 5: LHS COVERAGE (2D PROJECTIONS)
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# fcm vs c_bot
axs[0].scatter(df['Fcm_MPa'], df['c_nom_bottom_mm'], 
               edgecolor='k', alpha=0.6, s=20)
axs[0].set_xlabel("$f_{cm}$ [MPa]")
axs[0].set_ylabel("$c_{bot}$ [mm]")
axs[0].set_title("LHS Coverage: $f_{cm}$ vs $c_{bot}$")
axs[0].grid(True, alpha=0.3)

# fcm vs c_top
axs[1].scatter(df['Fcm_MPa'], df['c_nom_top_mm'], 
               edgecolor='k', alpha=0.6, s=20)
axs[1].set_xlabel("$f_{cm}$ [MPa]")
axs[1].set_ylabel("$c_{top}$ [mm]")
axs[1].set_title("LHS Coverage: $f_{cm}$ vs $c_{top}$")
axs[1].grid(True, alpha=0.3)

# c_bot vs c_top
axs[2].scatter(df['c_nom_bottom_mm'], df['c_nom_top_mm'], 
               edgecolor='k', alpha=0.6, s=20)
axs[2].set_xlabel("$c_{bot}$ [mm]")
axs[2].set_ylabel("$c_{top}$ [mm]")
axs[2].set_title("LHS Coverage: $c_{bot}$ vs $c_{top}$")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "07_lhs_coverage.png"), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 07_lhs_coverage.png")

# ============================================================
# PLOT 6: SAMPLING METHOD COMPARISON
# ============================================================

from scipy.stats import qmc

N = 400
D = 2
SEED = 42

# Generate samples with different methods
rng = np.random.default_rng(SEED)
random_samples = rng.random((N, D))

lhs_sampler = qmc.LatinHypercube(d=D, seed=SEED)
lhs_samples = lhs_sampler.random(n=N)

sobol_sampler = qmc.Sobol(d=D, scramble=True, seed=SEED)
m = int(np.floor(np.log2(N)))
sobol_samples = sobol_sampler.random_base2(m=m)

halton_sampler = qmc.Halton(d=D, scramble=True, seed=SEED)
halton_samples = halton_sampler.random(n=N)

grid_n = int(np.sqrt(N))
xg = np.linspace(0, 1, grid_n)
yg = np.linspace(0, 1, grid_n)
Xg, Yg = np.meshgrid(xg, yg)
grid_samples = np.column_stack([Xg.ravel(), Yg.ravel()])

# Plot
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

methods = [
    ("Random", random_samples),
    ("LHS (Used)", lhs_samples),
    ("Sobol", sobol_samples),
    ("Halton", halton_samples),
    ("Grid", grid_samples)
]

for ax, (title, data) in zip(axs, methods):
    ax.scatter(data[:, 0], data[:, 1], s=10, edgecolor='k', alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold' if 'LHS' in title else 'normal')
    ax.set_xlabel("$x_1$ [-]")
    ax.set_ylabel("$x_2$ [-]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.suptitle("Sampling Method Comparison (2D Unit Hypercube)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "08_sampling_methods_comparison.png"), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 08_sampling_methods_comparison.png")

# ============================================================
# PLOT 7: YOUNG'S MODULUS RELATIONSHIP
# ============================================================

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# E vs fcm (scatter + theoretical curve)
fcm_theory = np.linspace(df['Fcm_MPa'].min(), df['Fcm_MPa'].max(), 100)
E_theory = 22000 * (fcm_theory / 10) ** 0.3

axs[0].scatter(df['Fcm_MPa'], df['E_MPa'], edgecolor='k', alpha=0.6, s=20, 
               label='Samples')
axs[0].plot(fcm_theory, E_theory, 'r-', linewidth=2, label='EC2 Formula')
axs[0].set_xlabel("$f_{cm}$ [MPa]")
axs[0].set_ylabel("$E$ [MPa]")
axs[0].set_title("Young's Modulus from Concrete Strength")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Distribution of E
axs[1].hist(df['E_MPa'], bins=30, density=True, edgecolor='k', alpha=0.7)
axs[1].set_xlabel("$E$ [MPa]")
axs[1].set_ylabel("Probability density [-]")
axs[1].set_title("Distribution of Young's Modulus")
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "09_youngs_modulus.png"), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved 09_youngs_modulus.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(df[['Fcm_MPa', 'c_nom_bottom_mm', 'c_nom_top_mm', 'E_MPa']].describe())

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)
print(corr_matrix)

print("\n" + "=" * 70)
print(f"All plots saved to: {SAVE_DIR}/")
print("=" * 70)
