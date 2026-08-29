"""
===============================================================================
STEP 4c.1: DATA SPLITTING
===============================================================================
Purpose: Split dataset into train/validation/test sets (70/15/15)

Ensures:
    - Reproducible split (fixed seed)
    - No data leakage between sets
    - Balanced distribution across sets

Output: train_jobs.txt, val_jobs.txt, test_jobs.txt
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "06_shape_scale_gpr"

UQ_CSV = BASE / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# Output directory
SPLIT_DIR = CLEAN / "split"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATA SPLITTING")
print("=" * 70)

df = pd.read_csv(UQ_CSV)
print(f"\n✓ Loaded {len(df)} samples from UQ dataset")

# Convert sample_id to job names
if "job_aug" not in df.columns:
    df["job_aug"] = df["sample_id_aug"].apply(lambda i: f"sample_{int(i):03d}")

jobs = df["job_aug"].to_numpy()

# ============================================================
# SHUFFLE AND SPLIT
# ============================================================

print("\nShuffling with seed =", SEED)
np.random.seed(SEED)
np.random.shuffle(jobs)

n_total = len(jobs)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)
n_test = n_total - n_train - n_val

train_jobs = jobs[:n_train]
val_jobs = jobs[n_train:n_train + n_val]
test_jobs = jobs[n_train + n_val:]

print(f"\n✓ Split sizes:")
print(f"  Training:   {len(train_jobs)} ({100*len(train_jobs)/n_total:.1f}%)")
print(f"  Validation: {len(val_jobs)} ({100*len(val_jobs)/n_total:.1f}%)")
print(f"  Test:       {len(test_jobs)} ({100*len(test_jobs)/n_total:.1f}%)")

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n" + "-" * 70)
print("VALIDATION CHECKS")
print("-" * 70)

# Check no overlap
train_set = set(train_jobs)
val_set = set(val_jobs)
test_set = set(test_jobs)

overlap_train_val = train_set & val_set
overlap_train_test = train_set & test_set
overlap_val_test = val_set & test_set

if overlap_train_val or overlap_train_test or overlap_val_test:
    print("❌ ERROR: Overlap detected between sets!")
    if overlap_train_val:
        print(f"  Train-Val overlap: {len(overlap_train_val)}")
    if overlap_train_test:
        print(f"  Train-Test overlap: {len(overlap_train_test)}")
    if overlap_val_test:
        print(f"  Val-Test overlap: {len(overlap_val_test)}")
else:
    print("✓ No overlap between sets")

# Check coverage
all_split = set(train_jobs) | set(val_jobs) | set(test_jobs)
all_original = set(jobs)

if all_split == all_original:
    print("✓ All samples accounted for")
else:
    missing = all_original - all_split
    extra = all_split - all_original
    print(f"⚠ WARNING:")
    if missing:
        print(f"  Missing: {len(missing)} samples")
    if extra:
        print(f"  Extra: {len(extra)} samples")

# Check for duplicates within each set
train_dupes = len(train_jobs) - len(train_set)
val_dupes = len(val_jobs) - len(val_set)
test_dupes = len(test_jobs) - len(test_set)

if train_dupes or val_dupes or test_dupes:
    print("❌ ERROR: Duplicates detected within sets!")
    print(f"  Train duplicates: {train_dupes}")
    print(f"  Val duplicates: {val_dupes}")
    print(f"  Test duplicates: {test_dupes}")
else:
    print("✓ No duplicates within sets")

# ============================================================
# SAVE SPLITS
# ============================================================

print("\n" + "-" * 70)
print("SAVING SPLITS")
print("-" * 70)

np.savetxt(SPLIT_DIR / "train_jobs.txt", train_jobs, fmt="%s")
np.savetxt(SPLIT_DIR / "val_jobs.txt", val_jobs, fmt="%s")
np.savetxt(SPLIT_DIR / "test_jobs.txt", test_jobs, fmt="%s")

print(f"✓ Saved to: {SPLIT_DIR}")
print(f"  - train_jobs.txt ({len(train_jobs)} samples)")
print(f"  - val_jobs.txt ({len(val_jobs)} samples)")
print(f"  - test_jobs.txt ({len(test_jobs)} samples)")

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

# Merge with UQ data to get statistics
df_train = df[df["job_aug"].isin(train_jobs)]
df_val = df[df["job_aug"].isin(val_jobs)]
df_test = df[df["job_aug"].isin(test_jobs)]

params = ["fc", "E", "c_nom_bottom_mm", "c_nom_top_mm"]

print("\nParameter distributions across splits:")
for param in params:
    print(f"\n{param}:")
    print(f"  Train - Mean: {df_train[param].mean():.3f}, Std: {df_train[param].std():.3f}")
    print(f"  Val   - Mean: {df_val[param].mean():.3f}, Std: {df_val[param].std():.3f}")
    print(f"  Test  - Mean: {df_test[param].mean():.3f}, Std: {df_test[param].std():.3f}")

print("\n" + "=" * 70)
print("SPLITTING COMPLETE")
print("=" * 70)