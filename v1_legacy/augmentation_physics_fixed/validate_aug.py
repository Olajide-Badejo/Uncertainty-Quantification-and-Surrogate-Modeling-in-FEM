#!/usr/bin/env python3
"""
Validation of physics-consistent augmented FEM datasets.

Checks:
1) Input distribution drift (original vs augmented)
2) Output physical sanity (force & crack curves)
3) Structural consistency across CSV files

This script is READ-ONLY and safe.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# SETTINGS (same folder as augmentation script)
# ------------------------------------------------------------

ROOT = Path(__file__).parent

INPUT_AUG = ROOT / "processed_inputs_2_aug.csv"
LOAD_AUG = ROOT / "load_displacement_full_aug.csv"
CRACK_AUG = ROOT / "crack_evolution_full_aug.csv"

INPUT_ORIG = ROOT.parent / "01_samplying/processed_inputs_2.csv"
LOAD_ORIG  = ROOT.parent / "03_postprocess/results_4/load_displacement_full.csv"
CRACK_ORIG = ROOT.parent / "03_postprocess/results_4/crack_evolution_full.csv"


PLOT_DIR = ROOT / "validation_plots"
PLOT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def describe(name, s):
    return f"{name}: mean={s.mean():.3f}, std={s.std():.3f}"


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Reading datasets...")
    df_in_o = pd.read_csv(INPUT_ORIG)
    df_in_a = pd.read_csv(INPUT_AUG)

    df_load_o = pd.read_csv(LOAD_ORIG)
    df_load_a = pd.read_csv(LOAD_AUG)

    df_crack_o = pd.read_csv(CRACK_ORIG)
    df_crack_a = pd.read_csv(CRACK_AUG)

    # --------------------------------------------------------
    # 1) Structural consistency
    # --------------------------------------------------------

    jobs_in = set(df_in_a["job"])
    jobs_load = set(df_load_a["job"])
    jobs_crack = set(df_crack_a["job"])

    assert jobs_in == jobs_load == jobs_crack, "Job mismatch across augmented files"

    print(f"✓ Structural check passed ({len(jobs_in)} samples)")

    # --------------------------------------------------------
    # 2) Input distribution checks
    # --------------------------------------------------------

    INPUTS = ["fc", "E", "c_nom_top_mm", "c_nom_bottom_mm"]

    print("\nInput distributions:")
    for k in INPUTS:
        print(" ", describe(k + " (orig)", df_in_o[k]))
        print(" ", describe(k + " (aug )", df_in_a[k]))

        plt.figure()
        plt.hist(df_in_o[k], bins=30, alpha=0.6, label="original")
        plt.hist(df_in_a[k], bins=30, alpha=0.6, label="augmented")
        plt.legend()
        plt.title(k)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"input_{k}.png")
        plt.close()

    print("✓ Input distribution plots saved")

    # --------------------------------------------------------
    # 3) Output physics checks
    # --------------------------------------------------------

    # Force
    max_force_o = df_load_o.groupby("job")["RF2"].max()
    max_force_a = df_load_a.groupby("job")["RF2"].max()

    ratio = max_force_a.mean() / max_force_o.mean()
    print(f"\nForce peak ratio (aug/orig): {ratio:.3f}")
    assert 0.9 < ratio < 1.1, "Force peak drift too large"

    assert (df_load_a["RF2"] >= 0).all(), "❌ Negative force detected"

    # Crack
    for col in ["PEEQ_max", "DAMAGET_max", "DAMAGEC_max", "SDEG"]:
        if col in df_crack_a.columns:
            assert (df_crack_a[col] >= 0).all(), f" Negative {col}"

    print("✓ Output physics checks passed")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n✓ Augmented dataset validation SUCCESSFUL")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()