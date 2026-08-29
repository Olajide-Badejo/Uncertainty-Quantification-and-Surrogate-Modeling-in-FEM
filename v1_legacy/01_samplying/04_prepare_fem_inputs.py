"""
===============================================================================
PREPARE PROCESSED INPUTS FOR FEM SIMULATION
===============================================================================
Purpose: Create processed input file matching PCA job list

This script:
    1. Loads the PCA meta.json to get list of valid samples
    2. Filters the full LHS dataset to only include those samples
    3. Adds job names (sample_XXX format)
    4. Exports minimal CSV for surrogate model training

Output columns:
    - job: Sample identifier (sample_000, sample_001, ...)
    - fc: Concrete compressive strength [MPa]
    - E: Young's modulus [MPa]
    - c_nom_bottom_mm: Bottom cover thickness [mm]
    - c_nom_top_mm: Top cover thickness [mm]
===============================================================================
"""

import json
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Input paths
META_JSON = ROOT / "04_PCA" / "01_pca_reduction" / "models" / "meta.json"
LHS_CSV = ROOT / "uq_lhs_samples_training.csv"

# Output path
OUTPUT_CSV = ROOT / "01_samplying" / "processed_inputs_4.csv"

# ============================================================
# LOAD PCA JOB LIST
# ============================================================

print("=" * 70)
print("PREPARING PROCESSED INPUTS FOR FEM")
print("=" * 70)

with open(META_JSON, 'r') as f:
    meta = json.load(f)

pca_jobs = meta["jobs"]
print(f"\n✓ Loaded {len(pca_jobs)} jobs from meta.json")
print(f"  First 5 jobs: {pca_jobs[:5]}")

# ============================================================
# LOAD FULL LHS DATASET
# ============================================================

df = pd.read_csv(LHS_CSV)
print(f"✓ Loaded {len(df)} samples from LHS dataset")

# ============================================================
# CREATE JOB NAMES
# ============================================================

# Map row index to job name format
df["job"] = df.index.map(lambda i: f"sample_{i:03d}")

# ============================================================
# FILTER TO PCA JOBS ONLY
# ============================================================

df_filtered = df[df["job"].isin(pca_jobs)].copy()
print(f"✓ Filtered to {len(df_filtered)} samples matching PCA jobs")

# ============================================================
# ADD DERIVED PARAMETERS
# ============================================================

# fc is just fcm (same variable, different notation)
df_filtered["fc"] = df_filtered["Fcm_MPa"]

# Young's modulus from EC2 formula (if not already computed)
if "E_MPa" not in df_filtered.columns:
    df_filtered["E"] = 22000 * (df_filtered["fc"] / 10) ** 0.3
    print("✓ Computed Young's modulus from fc")
else:
    df_filtered["E"] = df_filtered["E_MPa"]

# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_df = df_filtered[[
    "job",
    "fc",
    "E",
    "c_nom_bottom_mm",
    "c_nom_top_mm"
]]

# ============================================================
# SAVE OUTPUT
# ============================================================

output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Processed inputs saved to:")
print(f"  {OUTPUT_CSV}")
print(f"\n✓ Contains {len(output_df)} samples")

# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)

# Check if all PCA jobs are present
missing_jobs = set(pca_jobs) - set(output_df["job"])
if len(missing_jobs) == 0:
    print("✓ All PCA jobs present in output")
else:
    print(f"⚠ WARNING: {len(missing_jobs)} PCA jobs missing!")
    print(f"  Missing: {list(missing_jobs)[:5]}...")

# Check for duplicates
duplicates = output_df["job"].duplicated().sum()
if duplicates == 0:
    print("✓ No duplicate jobs")
else:
    print(f"⚠ WARNING: {duplicates} duplicate jobs!")

# Show sample
print("\n" + "-" * 70)
print("SAMPLE OUTPUT (first 5 rows):")
print("-" * 70)
print(output_df.head())

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)
