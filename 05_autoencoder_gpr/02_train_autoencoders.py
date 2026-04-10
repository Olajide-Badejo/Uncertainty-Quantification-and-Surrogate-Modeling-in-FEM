#!/usr/bin/env python3
"""
IMPROVED AE Training - Step 2 (Fixed Damage Prediction)
========================================================
Key improvements:
- Increased damage latent dim: 6 → 12
- Monotonic damage autoencoder
- Smoothness regularization
- Better loss function
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Import both standard and monotonic autoencoders
from ae_model import ImprovedCurveAutoencoder
from ae_model import MonotonicDamageAutoencoder, SmoothL1ReconstructionLoss

plt.rcParams["font.family"] = "Times New Roman"


def train_autoencoder(
    curves_train,
    curves_val,
    n_points,
    latent_dim,
    out_path,
    curve_name,
    use_monotonic=False,
    use_smooth_loss=False,
    n_epochs=500,
    batch_size=64,
    lr=1e-3,
    patience=50,  # Increased patience
):
    """
    Train an autoencoder with optional monotonicity and smoothness constraints.
    """
    device = torch.device("cpu")

    # Training loader
    X_train = torch.from_numpy(curves_train.astype(np.float32))
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    # Validation loader
    X_val = torch.from_numpy(curves_val.astype(np.float32))
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    # Choose model type
    if use_monotonic:
        print(f"  Using MonotonicDamageAutoencoder")
        model = MonotonicDamageAutoencoder(n_points=n_points, latent_dim=latent_dim).to(device)
    else:
        print(f"  Using standard CurveAutoencoder")
        model = ImprovedCurveAutoencoder(n_points=n_points, latent_dim=latent_dim).to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Choose loss function
    if use_smooth_loss:
        print(f"  Using SmoothL1ReconstructionLoss")
        loss_fn = SmoothL1ReconstructionLoss(alpha=0.1)
    else:
        loss_fn = nn.MSELoss(reduction="mean")

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"\n{'='*60}")
    print(f"Training {curve_name} Autoencoder")
    print(f"{'='*60}")
    print(f"Architecture: {n_points} → {latent_dim} → {n_points}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        epoch_train_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_train_loss += loss.item() * batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_loss = np.sqrt(epoch_train_loss)  # Convert to RMSE
        train_losses.append(epoch_train_loss)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = loss_fn(recon, batch)
                epoch_val_loss += loss.item() * batch.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_loss = np.sqrt(epoch_val_loss)  # Convert to RMSE
        val_losses.append(epoch_val_loss)

        # Progress reporting
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train RMSE: {epoch_train_loss:.6f} | Val RMSE: {epoch_val_loss:.6f}")

        # Early stopping based on validation loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_points": n_points,
                    "latent_dim": latent_dim,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "use_monotonic": use_monotonic,
                },
                out_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch}")
                break

    print(f"\nTraining complete!")
    print(f"Best model saved to: {out_path}")
    print(f"Best epoch: {best_epoch} with validation RMSE: {best_loss:.6f}\n")

    return train_losses, val_losses, model, best_epoch


def compute_test_loss(model, curves_test, use_smooth_loss=False):
    """Compute test RMSE on held-out test set."""
    device = torch.device("cpu")
    X_test = torch.from_numpy(curves_test.astype(np.float32))
    test_loader = DataLoader(TensorDataset(X_test), batch_size=64, shuffle=False)

    if use_smooth_loss:
        loss_fn = SmoothL1ReconstructionLoss(alpha=0.1)
    else:
        loss_fn = nn.MSELoss()
    
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            total_loss += loss.item() * batch.size(0)

    mse = total_loss / len(test_loader.dataset)
    return np.sqrt(mse)  # RMSE


def plot_training_curve(train_losses, val_losses, best_epoch, title, filename, out_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Training Loss", color="darkred", linewidth=1.5)
    plt.plot(epochs, val_losses, label="Validation Loss", color="steelblue", linewidth=1.5)
    
    # Mark best epoch
    plt.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1, 
                label=f"Best Epoch ({best_epoch})")
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("RMSE Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def plot_test_loss_bar(test_loss, title, filename, out_dir):
    """Plot test loss as a bar chart."""
    plt.figure(figsize=(6, 4))
    plt.bar(["Test Loss"], [test_loss], color="darkgreen", width=0.5)
    plt.ylabel("RMSE Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def main():
    base = Path(
        r"."
    )

    data_dir = base / "05_autoencoder_gpr" / "data_preprocessed"
    out_dir = base / "05_autoencoder_gpr" / "output_autoencoder_improved"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    u_force = np.load(data_dir / "u_force.npy")
    u_damage = np.load(data_dir / "u_crack.npy")

    F_norm_all = np.load(data_dir / "F_norm_all.npy")
    C_norm_all = np.load(data_dir / "C_norm_all.npy")

    # Load split indices
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    test_idx = np.load(data_dir / "test_indices.npy")

    print(f"Total samples: {F_norm_all.shape[0]}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Split data
    F_train, F_val, F_test = F_norm_all[train_idx], F_norm_all[val_idx], F_norm_all[test_idx]
    C_train, C_val, C_test = C_norm_all[train_idx], C_norm_all[val_idx], C_norm_all[test_idx]

    # ============================================================
    # TRAIN FORCE AUTOENCODER (Standard)
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING FORCE AUTOENCODER")
    print("="*70)
    
    train_loss_F, val_loss_F, model_F, best_epoch_F = train_autoencoder(
        curves_train=F_train,
        curves_val=F_val,
        n_points=F_train.shape[1],
        latent_dim=12,  # Keep at 12
        out_path=out_dir / "ae_force.pt",
        curve_name="Force",
        use_monotonic=False,
        use_smooth_loss=False,
    )

    test_loss_F = compute_test_loss(model_F, F_test)
    print(f"Force AE - Final Test RMSE: {test_loss_F:.6f}\n")

    # Plot force AE losses
    plot_training_curve(
        train_loss_F, val_loss_F, best_epoch_F,
        "Force Autoencoder: Training vs Validation Loss",
        "force_ae_training_curve.png",
        out_dir
    )
    plot_test_loss_bar(
        test_loss_F,
        "Force Autoencoder: Test Loss",
        "force_ae_test_loss.png",
        out_dir
    )

    # ============================================================
    # TRAIN DAMAGE AUTOENCODER (Monotonic + Increased Latent Dim)
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING DAMAGE AUTOENCODER (IMPROVED)")
    print("="*70)
    print("Improvements:")
    print("  - Latent dim increased: 6 → 12")
    print("  - Monotonic architecture")
    print("  - Smooth L1 loss")
    print("="*70)
    
    train_loss_C, val_loss_C, model_C, best_epoch_C = train_autoencoder(
        curves_train=C_train,
        curves_val=C_val,
        n_points=C_train.shape[1],
        latent_dim=12,  # INCREASED from 6 to 12
        out_path=out_dir / "ae_damage.pt",
        curve_name="Compression Damage",
        use_monotonic=True,  # USE MONOTONIC MODEL
        use_smooth_loss=True,  # USE SMOOTH LOSS
    )

    test_loss_C = compute_test_loss(model_C, C_test, use_smooth_loss=True)
    print(f"Damage AE - Final Test RMSE: {test_loss_C:.6f}\n")

    # Plot damage AE losses
    plot_training_curve(
        train_loss_C, val_loss_C, best_epoch_C,
        "Damage Autoencoder: Training vs Validation Loss (Improved)",
        "damage_ae_training_curve.png",
        out_dir
    )
    plot_test_loss_bar(
        test_loss_C,
        "Damage Autoencoder: Test Loss (Improved)",
        "damage_ae_test_loss.png",
        out_dir
    )

    # ============================================================
    # SAVE METADATA AND RESULTS
    # ============================================================
    # Save displacement grids
    np.save(out_dir / "u_force.npy", u_force)
    np.save(out_dir / "u_damage.npy", u_damage)

    # Save results summary
    import json
    results = {
        "force_ae": {
            "latent_dim": 12,
            "best_epoch": int(best_epoch_F),
            "best_val_loss": float(val_loss_F[best_epoch_F - 1]),
            "test_loss": float(test_loss_F),
            "monotonic": False,
        },
        "damage_ae": {
            "latent_dim": 12,  # Updated
            "best_epoch": int(best_epoch_C),
            "best_val_loss": float(val_loss_C[best_epoch_C - 1]),
            "test_loss": float(test_loss_C),
            "monotonic": True,  # New
            "smooth_loss": True,  # New
        }
    }

    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print("Training Summary")
    print("="*70)
    print(f"Force AE:")
    print(f"  Latent dim: {results['force_ae']['latent_dim']}")
    print(f"  Best epoch: {results['force_ae']['best_epoch']}")
    print(f"  Val RMSE: {results['force_ae']['best_val_loss']:.6f}")
    print(f"  Test RMSE: {results['force_ae']['test_loss']:.6f}")
    print(f"\nDamage AE (IMPROVED):")
    print(f"  Latent dim: {results['damage_ae']['latent_dim']}")
    print(f"  Monotonic: {results['damage_ae']['monotonic']}")
    print(f"  Smooth loss: {results['damage_ae']['smooth_loss']}")
    print(f"  Best epoch: {results['damage_ae']['best_epoch']}")
    print(f"  Val RMSE: {results['damage_ae']['best_val_loss']:.6f}")
    print(f"  Test RMSE: {results['damage_ae']['test_loss']:.6f}")
    print("="*70)
    print(f"\nAll plots and models saved to: {out_dir}")


if __name__ == "__main__":
    main()