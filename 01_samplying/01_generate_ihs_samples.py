"""
===============================================================================
UNCERTAINTY QUANTIFICATION - INPUT PARAMETER SAMPLING
===============================================================================
Purpose: Generate Latin Hypercube Samples for concrete beam uncertainty analysis

Uncertain Parameters:
    - Concrete compressive strength (fcm): Lognormal distribution
    - Bottom concrete cover (c_nom_bottom): Normal distribution  
    - Top concrete cover (c_nom_top): Normal distribution

Deterministic (derived):
    - Young's modulus (E): Computed from fcm using EC2 formula

Sampling method: Latin Hypercube Sampling (LHS)
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

# ============================================================
# CONFIGURATION
# ============================================================

# Random seed for reproducibility
SEED = 42
N_SAMPLES = 400


# Output path
OUTPUT_CSV = "uq_lhs_samples_training.csv"

# ============================================================
# PARAMETER DEFINITIONS
# ============================================================

# Concrete compressive strength (MPa) - Lognormal
FCM_MEAN = 28.0
FCM_COV = 0.10

# Bottom concrete cover (mm) - Normal
C_BOT_MEAN = 27.0
C_BOT_STD = 3.0

# Top concrete cover (mm) - Normal  
C_TOP_MEAN = 223.0
C_TOP_STD = 5.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lognormal_params(mean, cov):
    """
    Convert lognormal mean and CoV to underlying normal parameters.
    
    Parameters:
        mean: Mean of lognormal distribution
        cov: Coefficient of variation
        
    Returns:
        mu, sigma: Parameters of underlying normal distribution
    """
    variance = (cov * mean) ** 2
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

# ============================================================
# MAIN SAMPLING ROUTINE
# ============================================================

def generate_samples():
    """
    Generate LHS samples for all uncertain parameters.
    
    Returns:
        DataFrame with columns: sample_id, Fcm_MPa, c_nom_bottom_mm, 
                                c_nom_top_mm, E_MPa, seed
    """
    
    # Compute lognormal parameters for fcm
    mu_f, sigma_f = lognormal_params(FCM_MEAN, FCM_COV)
    
    # Initialize LHS sampler (3 independent dimensions)
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    
    # Generate uniform samples [0,1]
    U = sampler.random(n=N_SAMPLES)
    
    # Transform to standard normal space
    Z = norm.ppf(U)
    
    # --------------------------------------------------------
    # Sample 1: Concrete strength (lognormal)
    # --------------------------------------------------------
    Fcm_samples = np.exp(mu_f + sigma_f * Z[:, 0])
    
    # --------------------------------------------------------
    # Sample 2: Bottom concrete cover (normal)
    # --------------------------------------------------------
    c_bot_samples = C_BOT_MEAN + C_BOT_STD * Z[:, 1]
    
    # --------------------------------------------------------
    # Sample 3: Top concrete cover (normal)
    # --------------------------------------------------------
    c_top_samples = C_TOP_MEAN + C_TOP_STD * Z[:, 2]
    
    # --------------------------------------------------------
    # Derive: Young's modulus from fcm (EC2 formula)
    # --------------------------------------------------------
    # E = 22000 * (fcm/10)^0.3  [MPa]
    E_samples = 22000 * (Fcm_samples / 10) ** 0.3
    
    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------
    df = pd.DataFrame({
        "sample_id": np.arange(N_SAMPLES),
        "Fcm_MPa": Fcm_samples,
        "c_nom_bottom_mm": c_bot_samples,
        "c_nom_top_mm": c_top_samples,
        "E_MPa": E_samples,
        "seed": SEED
    })
    
    return df

# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("GENERATING LHS SAMPLES FOR UQ ANALYSIS")
    print("=" * 70)
    
    # Generate samples
    df = generate_samples()
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Generated {N_SAMPLES} samples")
    print(f"✓ Saved to: {OUTPUT_CSV}")
    print(f"✓ Random seed: {SEED}")
    
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(df.describe())
    
    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)