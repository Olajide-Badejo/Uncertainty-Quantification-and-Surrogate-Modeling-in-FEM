#!/usr/bin/env python3
"""
AE + GPR Pipeline - Step 3: Encode Curves (IMPROVED VERSION)
=============================================================
Use improved trained autoencoders to map curves to latent space.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ae_model import ImprovedCurveAutoencoder
from ae_model import MonotonicDamageAutoencoder


def load_ae(path: Path):
    """Load autoencoder, automatically detecting monotonic vs standard."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Check if this is a monotonic model
    use_monotonic = ckpt.get("use_monotonic", False)
    
    if use_monotonic:
        print(f"  Detected: MonotonicDamageAutoencoder")
        model = MonotonicDamageAutoencoder(
            n_points=ckpt["n_points"],
            latent_dim=ckpt["latent_dim"]
        )
    else:
        print(f"  Detected: Standard CurveAutoencoder")
        model = ImprovedCurveAutoencoder(
            n_points=ckpt["n_points"],
            latent_dim=ckpt["latent_dim"]
        )
    
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    print(f"  Architecture: {ckpt['n_points']} → {ckpt['latent_dim']} → {ckpt['n_points']}")
    if "best_epoch" in ckpt:
        print(f"  Best epoch: {ckpt['best_epoch']}")
        print(f"  Best val loss: {ckpt['best_val_loss']:.6f}")
    
    return model


def main():
    base = Path(
        r"."
    )

    # --------------------------------------------------------
    # PATHS (IMPROVED VERSION)
    # --------------------------------------------------------
    data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
    ae_out = base / "05_autoencoder_gpr" / "output_autoencoder_improved"  # CHANGED
    gpr_out = base / "05_autoencoder_gpr" / "output_surrogates_improved"  # CHANGED
    gpr_out.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ENCODING CURVES TO LATENT SPACE (IMPROVED MODELS)")
    print("="*60)

    # --------------------------------------------------------
    # LOAD TRAIN / VAL / TEST SPLIT
    # --------------------------------------------------------
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    test_idx = np.load(data_dir / "test_indices.npy")

    print(f"\nDataset Split:")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    print(f"  Total: {len(train_idx) + len(val_idx) + len(test_idx)}")

    # --------------------------------------------------------
    # LOAD PREPROCESSED CURVES
    # --------------------------------------------------------
    print("\nLoading preprocessed curves...")
    F_norm_all = np.load(data_dir / "F_norm_all.npy")
    C_norm_all = np.load(data_dir / "C_norm_all.npy")

    print(f"  Force curves shape: {F_norm_all.shape}")
    print(f"  Damage curves shape: {C_norm_all.shape}")

    # Split curves
    F_train = F_norm_all[train_idx]
    F_val = F_norm_all[val_idx]
    F_test = F_norm_all[test_idx]

    C_train = C_norm_all[train_idx]
    C_val = C_norm_all[val_idx]
    C_test = C_norm_all[test_idx]

    # --------------------------------------------------------
    # LOAD IMPROVED AUTOENCODERS
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Loading Improved Autoencoders")
    print("-"*60)
    
    print("\nForce Autoencoder:")
    ae_force = load_ae(ae_out / "ae_force.pt")
    
    print("\nDamage Autoencoder:")
    ae_damage = load_ae(ae_out / "ae_damage.pt")

    # --------------------------------------------------------
    # ENCODE CURVES TO LATENT SPACE
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Encoding Curves")
    print("-"*60)
    
    with torch.no_grad():
        print("\nForce curves...")
        Z_force_train = ae_force.encode(torch.from_numpy(F_train.astype(np.float32))).numpy()
        Z_force_val = ae_force.encode(torch.from_numpy(F_val.astype(np.float32))).numpy()
        Z_force_test = ae_force.encode(torch.from_numpy(F_test.astype(np.float32))).numpy()
        print(f"  Train: {Z_force_train.shape}")
        print(f"  Val:   {Z_force_val.shape}")
        print(f"  Test:  {Z_force_test.shape}")

        print("\nDamage curves...")
        Z_damage_train = ae_damage.encode(torch.from_numpy(C_train.astype(np.float32))).numpy()
        Z_damage_val = ae_damage.encode(torch.from_numpy(C_val.astype(np.float32))).numpy()
        Z_damage_test = ae_damage.encode(torch.from_numpy(C_test.astype(np.float32))).numpy()
        print(f"  Train: {Z_damage_train.shape}")
        print(f"  Val:   {Z_damage_val.shape}")
        print(f"  Test:  {Z_damage_test.shape}")

    # --------------------------------------------------------
    # LOAD INPUT PARAMETERS
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Loading Physical Parameters")
    print("-"*60)
    
    uq_csv = base / r"augmentation_physics_fixed\processed_inputs_2_aug.csv"
    df_uq = pd.read_csv(uq_csv)

    cbot = df_uq["c_nom_bottom_mm"].to_numpy(float)
    ctop = df_uq["c_nom_top_mm"].to_numpy(float)
    fcm = df_uq["fc"].to_numpy(float)

    X_all = np.column_stack([cbot, ctop, fcm])

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    print(f"\nPhysical parameters shape: {X_all.shape}")

    # --------------------------------------------------------
    # SAVE EVERYTHING FOR GPR TRAINING
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Saving Data for GPR Training")
    print("-"*60)
    
    np.save(gpr_out / "X_train.npy", X_train)
    np.save(gpr_out / "X_val.npy", X_val)
    np.save(gpr_out / "X_test.npy", X_test)
    print("\n✓ Saved input parameters")

    np.save(gpr_out / "Z_force_train.npy", Z_force_train)
    np.save(gpr_out / "Z_force_val.npy", Z_force_val)
    np.save(gpr_out / "Z_force_test.npy", Z_force_test)
    print("✓ Saved force latent vectors")

    np.save(gpr_out / "Z_damage_train.npy", Z_damage_train)
    np.save(gpr_out / "Z_damage_val.npy", Z_damage_val)
    np.save(gpr_out / "Z_damage_test.npy", Z_damage_test)
    print("✓ Saved damage latent vectors")

    print("\n" + "="*60)
    print("ENCODING COMPLETE")
    print("="*60)
    print(f"\nAll files saved to: {gpr_out}")
    print("\nReady for GPR training (Step 4)")
    print("="*60)


if __name__ == "__main__":
    main()