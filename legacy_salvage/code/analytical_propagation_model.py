import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================
# BEAM MODEL (PROPPED CANTILEVER) - ANALYTICAL CALCULATION FOR DEFLECTION (STATISTICALLY INDETERMINATE) - JUST FOR DOING SAKE
# =========================

def beam_tip_deflection(E, Fcm, P=50e3, L=1.6, I=1.958e-4):
    """
    Physics-informed surrogate forward model
    Propped cantilever (roller at midspan)

    E   : Elastic modulus (MPa)
    Fcm : Compressive strength (MPa)
    P   : Load (N)
    L   : Length (m)
    I   : Second moment of area (m^4)

    Returns deflection in mm
    """

    # Convert MPa → Pa
    E_pa = E * 1e6

    # Elastic deflection
    # δ = 7PL^3 / (96EI)
    delta_elastic = (7 * P * L**3) / (96 * E_pa * I)

    # Strength degradation factor (soft coupling)
    strength_factor = 1.0 - 0.15 * (28.0 - Fcm) / 28.0
    strength_factor = np.clip(strength_factor, 0.7, 1.1)

    delta_total = delta_elastic / strength_factor

    return delta_total * 1e3  # meters → mm


# =========================
# LOAD SAMPLES
# =========================

df = pd.read_csv("uq_lhs_samples_training.csv")

# =========================
# COMPUTE DEFLECTION
# =========================

df["tip_deflection_mm"] = beam_tip_deflection(
    df["E_concrete_MPa"].values,
    df["Fcm_MPa"].values
)

# Split cases
df_ind = df[df["case"] == "independent"]
df_corr = df[df["case"] == "correlated"]

# =========================
# PDF PLOT (KDE)
# =========================

plt.figure(figsize=(8, 5))

for data, label in zip(
    [df_ind, df_corr],
    ["Independent inputs", "Correlated inputs (ρ = 0.6)"]
):
    kde = gaussian_kde(data["tip_deflection_mm"])
    x = np.linspace(
        data["tip_deflection_mm"].min(),
        data["tip_deflection_mm"].max(),
        300
    )
    plt.plot(x, kde(x), label=label)

plt.xlabel("Tip deflection [mm]")
plt.ylabel("Probability density")
plt.title("Uncertainty propagation: effect of input dependence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("deflection_pdf.png", dpi=300)
plt.close()




# =========================
# CDF PLOT (EMPIRICAL)
# =========================

plt.figure(figsize=(8, 5))

for data, label in zip(
    [df_ind, df_corr],
    ["Independent", "Correlated"]
):
    sorted_vals = np.sort(data["tip_deflection_mm"])
    cdf = np.linspace(0, 1, len(sorted_vals), endpoint=False)
    plt.plot(sorted_vals, cdf, label=label)

plt.xlabel("Tip deflection [mm]")
plt.ylabel("Cumulative probability")
plt.title("CDF of beam deflection under uncertainty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.savefig("deflection_cdf.png", dpi=300)
plt.close()


# =========================
# SCATTER PLOTS: Analytical deflection vs E and Fcm
# =========================

# 1D Scatter: δ_tip vs E
plt.figure(figsize=(7, 5))
plt.scatter(df["E_concrete_MPa"], df["tip_deflection_mm"], 
            c='blue', alpha=0.7, edgecolor='k')
plt.xlabel("Elastic Modulus E [MPa]")
plt.ylabel("Tip Deflection [mm]")
plt.title("Tip deflection vs E")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("deflection_vs_E.png", dpi=300)
plt.close()

# 1D Scatter: δ_tip vs Fcm
plt.figure(figsize=(7, 5))
plt.scatter(df["Fcm_MPa"], df["tip_deflection_mm"], 
            c='green', alpha=0.7, edgecolor='k')
plt.xlabel("Compressive Strength Fcm [MPa]")
plt.ylabel("Tip Deflection [mm]")
plt.title("Tip deflection vs Fcm")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("deflection_vs_Fcm.png", dpi=300)
plt.close()

# Optional: 3D scatter δ_tip vs (E, Fcm)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["E_concrete_MPa"], df["Fcm_MPa"], df["tip_deflection_mm"],
           c=df["tip_deflection_mm"], cmap='viridis', s=50)
ax.set_xlabel("E [MPa]")
ax.set_ylabel("Fcm [MPa]")
ax.set_zlabel("Tip Deflection [mm]")
ax.set_title("Tip deflection vs E and Fcm")
plt.tight_layout()
plt.show()
plt.savefig("deflection_vs_Fcm.png", dpi=300)
plt.close()
