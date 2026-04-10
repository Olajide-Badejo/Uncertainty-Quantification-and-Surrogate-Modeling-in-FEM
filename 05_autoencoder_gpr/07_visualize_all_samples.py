#!/usr/bin/env python3
"""
AE + GPR Pipeline - Step 7: Visualize All Samples
==================================================
Generate comparison plots for ALL validation or test samples.
Useful for comprehensive visual inspection of model performance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ae_surrogate_model import ImprovedAESurrogateModel

plt.rcParams["font.family"] = "Times New Roman"


def main():
    base = Path(
        r"."
    )

    print("="*70)
    print("VISUALIZING ALL SAMPLES")
    print("="*70)

    # --------------------------------------------------------
    # LOAD TRAIN / VAL / TEST SPLIT
    # --------------------------------------------------------
    ae_out = base / "05_autoencoder_gpr" / "data_preprocessed"

    train_idx = np.load(ae_out / "train_indices.npy")
    val_idx = np.load(ae_out / "val_indices.npy")
    test_idx = np.load(ae_out / "test_indices.npy")

    print(f"\nDataset sizes:")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")

    # --------------------------------------------------------
    # LOAD PREPROCESSED CURVES (FEM ground truth)
    # --------------------------------------------------------
    data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"

    u_force = np.load(data_dir / "u_force.npy")
    u_damage = np.load(data_dir / "u_crack.npy")

    F_norm_all = np.load(data_dir / "F_norm_all.npy")
    C_norm_all = np.load(data_dir / "C_norm_all.npy")

    # Load normalization factors
    global_Fmax = float(np.load(data_dir / "F_global_max.npy"))
    C_max_all = np.load(data_dir / "C_max.npy")

    print(f"\nNormalization factors:")
    print(f"  Global force max: {global_Fmax:.3e} N")
    print(f"  Damage max range: [{C_max_all.min():.6f}, {C_max_all.max():.6f}]")

    # --------------------------------------------------------
    # LOAD INPUT PARAMETERS
    # --------------------------------------------------------
    uq_csv = base / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"
    df_uq = pd.read_csv(uq_csv)

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    split = "test"  # Change to "val" to visualize all validation samples

    print(f"\n" + "-"*70)
    print(f"Configuration")
    print("-"*70)
    print(f"  Split: {split}")

    # --------------------------------------------------------
    # PREPARE OUTPUT FOLDER
    # --------------------------------------------------------
    plot_dir = base / "05_autoencoder_gpr" / "output_plots" / f"all_{split}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD SURROGATE MODEL
    # --------------------------------------------------------
    print(f"\nLoading surrogate model...")
    model = ImprovedAESurrogateModel(str(base))

    # --------------------------------------------------------
    # SELECT SAMPLES
    # --------------------------------------------------------
    if split == "val":
        pool = val_idx
    elif split == "test":
        pool = test_idx
    else:
        raise ValueError("split must be 'val' or 'test'")

    total = len(pool)
    print(f"\nGenerating plots for ALL {total} {split} samples...")

    # --------------------------------------------------------
    # LOOP OVER ALL SAMPLES
    # --------------------------------------------------------
    print(f"\n" + "-"*70)
    print("Generating Plots")
    print("-"*70)

    skipped = 0
    generated = 0

    for count, idx in enumerate(pool, 1):
        if count % 10 == 0 or count == 1:
            print(f"\nProgress: {count}/{total} ({count/total*100:.1f}%)")

        # Physical inputs
        cbot = float(df_uq.iloc[idx]["c_nom_bottom_mm"])
        ctop = float(df_uq.iloc[idx]["c_nom_top_mm"])
        fcm = float(df_uq.iloc[idx]["fc"])

        # FEM truth curves (normalized)
        F_true_norm = F_norm_all[idx]
        C_true_norm = C_norm_all[idx]

        # Un-normalize
        F_true = F_true_norm * global_Fmax
        C_true = C_true_norm * C_max_all[idx]

        # Check for corrupted data
        if F_true.max() < 1e-6:
            if count % 10 == 0:
                print(f"  ⚠️  Skipping index {idx}: Near-zero force (corrupted)")
            skipped += 1
            continue

        # Surrogate prediction
        uF, F_pred_norm, uC, C_pred_norm = model.predict(
            cbot=cbot, ctop=ctop, fcm=fcm
        )

        # Un-normalize surrogate predictions
        F_pred = F_pred_norm * global_Fmax
        C_pred = C_pred_norm * C_max_all[idx]

        # --------------------------------------------------------
        # PLOT AND SAVE
        # --------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Force curve
        ax = axes[0]
        ax.plot(u_force, F_true, "k-", linewidth=2.5, label=f"FEM ({split})")
        ax.plot(uF, F_pred, "r--", linewidth=2, label="AE+GPR Surrogate")
        ax.set_xlabel("Displacement (mm)", fontsize=12)
        ax.set_ylabel("Force (N)", fontsize=12)
        ax.set_title(f"Force–Displacement Curve\n({split} sample {idx})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Compressive damage curve
        ax = axes[1]
        ax.plot(u_damage, C_true, "k-", linewidth=2.5, label=f"FEM ({split})")
        ax.plot(uC, C_pred, "r--", linewidth=2, label="AE+GPR Surrogate")
        ax.set_xlabel("Displacement (mm)", fontsize=12)
        ax.set_ylabel("Compression Damage", fontsize=12)
        ax.set_title(f"Damage Evolution Curve\n({split} sample {idx})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add parameter text box
        param_text = f"$c_{{bot}}$ = {cbot:.1f} mm, $c_{{top}}$ = {ctop:.1f} mm, $f_c$ = {fcm:.1f} MPa"
        fig.text(0.5, 0.02, param_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save figure
        out_path = plot_dir / f"{split}_sample_{idx:04d}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

        generated += 1

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Plots generated: {generated}")
    print(f"  Samples skipped: {skipped}")
    print(f"\nPlots saved to: {plot_dir}")
    print("="*70)


if __name__ == "__main__":
    main()