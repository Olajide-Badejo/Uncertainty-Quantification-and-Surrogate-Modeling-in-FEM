"""
===============================================================================
DATA QUALITY CHECKS - DEDUPLICATION & VALIDATION
===============================================================================
Purpose: Verify sample quality and detect issues

Checks performed:
    1. Duplicate detection
    2. Range validation (physical constraints)
    3. Missing value detection
    4. Statistical outlier detection
    5. Sample coverage uniformity
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "uq_lhs_samples_training.csv"

# Distribution parameters
FCM_MEAN = 28.0
FCM_COV = 0.10
C_BOT_MEAN = 27.0
C_BOT_STD = 3.0
C_TOP_MEAN = 223.0
C_TOP_STD = 5.0

# Physical constraints
FCM_MIN = 15.0    # Minimum realistic concrete strength [MPa]
FCM_MAX = 50.0    # Maximum realistic concrete strength [MPa]
C_BOT_MIN = 15.0  # Minimum bottom cover [mm]
C_BOT_MAX = 40.0  # Maximum bottom cover [mm]
C_TOP_MIN = 210.0 # Minimum top cover [mm]
C_TOP_MAX = 240.0 # Maximum top cover [mm]

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
print("=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)
print(f"\nLoaded {len(df)} samples from {INPUT_CSV}")

# ============================================================
# CHECK 1: DUPLICATE DETECTION
# ============================================================

print("\n" + "-" * 70)
print("CHECK 1: DUPLICATE DETECTION")
print("-" * 70)

# Check for duplicate sample IDs
duplicate_ids = df['sample_id'].duplicated().sum()
print(f"  Duplicate sample IDs: {duplicate_ids}")

# Check for duplicate parameter combinations (round to avoid floating point issues)
df_rounded = df[['Fcm_MPa', 'c_nom_bottom_mm', 'c_nom_top_mm']].round(6)
duplicate_params = df_rounded.duplicated().sum()
print(f"  Duplicate parameter combinations: {duplicate_params}")

if duplicate_params > 0:
    print("\n  ⚠ WARNING: Found duplicate samples!")
    duplicate_rows = df[df_rounded.duplicated(keep=False)]
    print(duplicate_rows[['sample_id', 'Fcm_MPa', 'c_nom_bottom_mm', 'c_nom_top_mm']])
else:
    print("  ✓ No duplicates found")

# ============================================================
# CHECK 2: MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("CHECK 2: MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("  ✓ No missing values")
else:
    print("  ⚠ WARNING: Missing values detected!")

# ============================================================
# CHECK 3: RANGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CHECK 3: RANGE VALIDATION")
print("-" * 70)

# Concrete strength
fcm_out_of_range = ((df['Fcm_MPa'] < FCM_MIN) | (df['Fcm_MPa'] > FCM_MAX)).sum()
print(f"  Fcm out of range [{FCM_MIN}, {FCM_MAX}]: {fcm_out_of_range}")

# Bottom cover
c_bot_out_of_range = ((df['c_nom_bottom_mm'] < C_BOT_MIN) | 
                      (df['c_nom_bottom_mm'] > C_BOT_MAX)).sum()
print(f"  c_bot out of range [{C_BOT_MIN}, {C_BOT_MAX}]: {c_bot_out_of_range}")

# Top cover
c_top_out_of_range = ((df['c_nom_top_mm'] < C_TOP_MIN) | 
                      (df['c_nom_top_mm'] > C_TOP_MAX)).sum()
print(f"  c_top out of range [{C_TOP_MIN}, {C_TOP_MAX}]: {c_top_out_of_range}")

total_out_of_range = fcm_out_of_range + c_bot_out_of_range + c_top_out_of_range

if total_out_of_range == 0:
    print("\n  ✓ All samples within physical constraints")
else:
    print(f"\n  ⚠ WARNING: {total_out_of_range} samples out of range!")

# ============================================================
# CHECK 4: STATISTICAL OUTLIERS (3-SIGMA RULE)
# ============================================================

print("\n" + "-" * 70)
print("CHECK 4: STATISTICAL OUTLIERS (3-SIGMA)")
print("-" * 70)

def lognormal_params(mean, cov):
    variance = (cov * mean) ** 2
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

mu_f, sigma_f = lognormal_params(FCM_MEAN, FCM_COV)

# fcm outliers (in log-space for lognormal)
log_fcm = np.log(df['Fcm_MPa'])
fcm_outliers = (np.abs(log_fcm - mu_f) > 3 * sigma_f).sum()
print(f"  Fcm outliers (>3σ): {fcm_outliers}")

# c_bot outliers
c_bot_outliers = (np.abs(df['c_nom_bottom_mm'] - C_BOT_MEAN) > 3 * C_BOT_STD).sum()
print(f"  c_bot outliers (>3σ): {c_bot_outliers}")

# c_top outliers
c_top_outliers = (np.abs(df['c_nom_top_mm'] - C_TOP_MEAN) > 3 * C_TOP_STD).sum()
print(f"  c_top outliers (>3σ): {c_top_outliers}")

total_outliers = fcm_outliers + c_bot_outliers + c_top_outliers

if total_outliers == 0:
    print("\n  ✓ No statistical outliers detected")
else:
    print(f"\n  ⚠ Note: {total_outliers} samples beyond 3σ (expected for LHS)")

# ============================================================
# CHECK 5: YOUNG'S MODULUS CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("CHECK 5: YOUNG'S MODULUS CONSISTENCY")
print("-" * 70)

# Recompute E and check against stored values
E_computed = 22000 * (df['Fcm_MPa'] / 10) ** 0.3
E_error = np.abs(df['E_MPa'] - E_computed)

max_error = E_error.max()
mean_error = E_error.mean()

print(f"  Max error: {max_error:.6e} MPa")
print(f"  Mean error: {mean_error:.6e} MPa")

if max_error < 1e-6:
    print("  ✓ Young's modulus correctly computed")
else:
    print("  ⚠ WARNING: Young's modulus computation errors detected!")

# ============================================================
# CHECK 6: SAMPLE COVERAGE UNIFORMITY
# ============================================================

print("\n" + "-" * 70)
print("CHECK 6: SAMPLE COVERAGE UNIFORMITY")
print("-" * 70)

# Check if samples span the full parameter space
# For LHS, each dimension should be well-distributed

def check_coverage(samples, name, n_bins=10):
    """Check if samples uniformly cover the parameter range"""
    hist, bin_edges = np.histogram(samples, bins=n_bins)
    expected_per_bin = len(samples) / n_bins
    
    # Chi-square test statistic
    chi2 = np.sum((hist - expected_per_bin)**2 / expected_per_bin)
    
    # Coefficient of variation of bin counts
    cv = hist.std() / hist.mean()
    
    print(f"\n  {name}:")
    print(f"    Expected per bin: {expected_per_bin:.1f}")
    print(f"    Actual range: [{hist.min()}, {hist.max()}]")
    print(f"    CV of counts: {cv:.3f}")
    
    if cv < 0.3:
        print(f"    ✓ Good coverage uniformity")
    else:
        print(f"    ⚠ Non-uniform coverage (CV > 0.3)")

check_coverage(df['Fcm_MPa'], "Fcm")
check_coverage(df['c_nom_bottom_mm'], "c_bot")
check_coverage(df['c_nom_top_mm'], "c_top")

# ============================================================
# CHECK 7: CORRELATION VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("CHECK 7: CORRELATION VERIFICATION (INDEPENDENCE)")
print("-" * 70)

# Compute correlations
corr_fcm_cbot = np.corrcoef(df['Fcm_MPa'], df['c_nom_bottom_mm'])[0, 1]
corr_fcm_ctop = np.corrcoef(df['Fcm_MPa'], df['c_nom_top_mm'])[0, 1]
corr_cbot_ctop = np.corrcoef(df['c_nom_bottom_mm'], df['c_nom_top_mm'])[0, 1]

print(f"  corr(Fcm, c_bot):  {corr_fcm_cbot:+.4f}")
print(f"  corr(Fcm, c_top):  {corr_fcm_ctop:+.4f}")
print(f"  corr(c_bot, c_top): {corr_cbot_ctop:+.4f}")

max_corr = max(abs(corr_fcm_cbot), abs(corr_fcm_ctop), abs(corr_cbot_ctop))

if max_corr < 0.1:
    print(f"\n  ✓ Variables are independent (max |corr| = {max_corr:.4f})")
else:
    print(f"\n  ⚠ Note: Some correlation detected (max |corr| = {max_corr:.4f})")
    print("     This is expected for LHS but should be small")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CHECK SUMMARY")
print("=" * 70)

issues = []
if duplicate_params > 0:
    issues.append(f"❌ {duplicate_params} duplicate samples")
if missing.sum() > 0:
    issues.append(f"❌ {missing.sum()} missing values")
if total_out_of_range > 0:
    issues.append(f"⚠ {total_out_of_range} samples out of physical range")
if max_error > 1e-6:
    issues.append(f"❌ Young's modulus computation errors")
if max_corr > 0.15:
    issues.append(f"⚠ Correlation above 0.15")

if len(issues) == 0:
    print("\n✓ ALL CHECKS PASSED - Dataset ready for FEM simulations")
else:
    print("\nIssues detected:")
    for issue in issues:
        print(f"  {issue}")
    
print("\n" + "=" * 70)