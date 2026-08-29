#!/usr/bin/env python3
"""
AE + GPR Pipeline - Step 4: Train GPR (IMPROVED VERSION)
=========================================================
Train GPR surrogates using latent vectors from improved autoencoders.
Note: Damage latent dim is now 12 (was 6).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

plt.rcParams["font.family"] = ["Times New Roman"]


def train_gpr_models(X_train, X_val, X_test, Z_train, Z_val, Z_test, out_dir, prefix, curve_name):
    """Train independent GPR models for each latent dimension."""
    n_latent = Z_train.shape[1]

    train_losses = []
    val_losses = []
    test_losses = []
    r2_scores = []
    models = []

    print(f"\n{'='*60}")
    print(f"Training GPR Models for {curve_name}")
    print(f"{'='*60}")
    print(f"Number of latent dimensions: {n_latent}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"{'='*60}\n")

    for i in range(n_latent):
        print(f"Training GPR for latent dimension {i+1}/{n_latent}...")

        y_train = Z_train[:, i]
        y_val = Z_val[:, i]
        y_test = Z_test[:, i]

        # Define kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2)
        )

        # Train GPR
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
        )

        gpr.fit(X_train, y_train)
        models.append(gpr)

        # Compute metrics
        y_train_pred = gpr.predict(X_train)
        y_val_pred = gpr.predict(X_val)
        y_test_pred = gpr.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2 = r2_score(y_test, y_test_pred)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        test_losses.append(test_rmse)
        r2_scores.append(r2)

        print(f"  Train RMSE: {train_rmse:.6f}")
        print(f"  Val RMSE:   {val_rmse:.6f}")
        print(f"  Test RMSE:  {test_rmse:.6f}")
        print(f"  Test R²:    {r2:.4f}\n")

    # Save models
    joblib.dump(models, out_dir / f"{prefix}_latent_gpr.joblib")

    return (
        np.array(train_losses),
        np.array(val_losses),
        np.array(test_losses),
        np.array(r2_scores),
        models
    )


def plot_gpr_losses(train, val, test, title, filename, out_dir):
    """Plot train/val/test RMSE for each latent dimension."""
    dims = np.arange(1, len(train) + 1)

    plt.figure(figsize=(10, 5))
    width = 0.25

    plt.bar(dims - width, train, width, label="Train RMSE", color="darkred", alpha=0.8)
    plt.bar(dims, val, width, label="Validation RMSE", color="steelblue", alpha=0.8)
    plt.bar(dims + width, test, width, label="Test RMSE", color="darkgreen", alpha=0.8)

    plt.xlabel("Latent Dimension", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(dims)
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def plot_r2_scores(r2_scores, title, filename, out_dir):
    """Plot R² scores for each latent dimension."""
    dims = np.arange(1, len(r2_scores) + 1)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(dims, r2_scores, color="purple", alpha=0.7, width=0.6)
    
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

    plt.axhline(y=0.9, color='green', linestyle='--', linewidth=1)
    plt.axhline(y=0.95, color='darkgreen', linestyle='--', linewidth=1)
    
    plt.xlabel("Latent Dimension", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(dims)
    plt.ylim([0, 1.05])
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def main():
    base = Path(
        r"."
    )
    
    # IMPROVED VERSION - use improved directory
    gpr_out = base / "05_autoencoder_gpr" / "output_surrogates_improved"
    gpr_out.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TRAINING GPR SURROGATES (IMPROVED VERSION)")
    print("="*70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    X_train = np.load(gpr_out / "X_train.npy")
    X_val = np.load(gpr_out / "X_val.npy")
    X_test = np.load(gpr_out / "X_test.npy")

    Z_force_train = np.load(gpr_out / "Z_force_train.npy")
    Z_force_val = np.load(gpr_out / "Z_force_val.npy")
    Z_force_test = np.load(gpr_out / "Z_force_test.npy")

    Z_damage_train = np.load(gpr_out / "Z_damage_train.npy")
    Z_damage_val = np.load(gpr_out / "Z_damage_val.npy")
    Z_damage_test = np.load(gpr_out / "Z_damage_test.npy")

    print(f"\nForce latent dimensions: {Z_force_train.shape[1]}")
    print(f"Damage latent dimensions: {Z_damage_train.shape[1]}")

    # --------------------------------------------------------
    # SCALE INPUTS
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, gpr_out / "input_scaler.joblib")

    # ============================================================
    # TRAIN FORCE GPR
    # ============================================================
    train_F, val_F, test_F, r2_F, models_F = train_gpr_models(
        X_train_scaled, X_val_scaled, X_test_scaled,
        Z_force_train, Z_force_val, Z_force_test,
        gpr_out, prefix="gpr_force", curve_name="Force Curves"
    )

    plot_gpr_losses(
        train_F, val_F, test_F,
        "Force GPR: Train / Validation / Test RMSE",
        "gpr_force_losses.png",
        gpr_out
    )
    
    plot_r2_scores(
        r2_F,
        "Force GPR: R² Scores on Test Set",
        "gpr_force_r2_scores.png",
        gpr_out
    )

    # ============================================================
    # TRAIN DAMAGE GPR
    # ============================================================
    train_C, val_C, test_C, r2_C, models_C = train_gpr_models(
        X_train_scaled, X_val_scaled, X_test_scaled,
        Z_damage_train, Z_damage_val, Z_damage_test,
        gpr_out, prefix="gpr_damage", curve_name="Compression Damage Curves"
    )

    plot_gpr_losses(
        train_C, val_C, test_C,
        "Damage GPR: Train / Validation / Test RMSE",
        "gpr_damage_losses.png",
        gpr_out
    )
    
    plot_r2_scores(
        r2_C,
        "Damage GPR: R² Scores on Test Set",
        "gpr_damage_r2_scores.png",
        gpr_out
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        "force_gpr": {
            "n_latent_dims": len(train_F),
            "train_rmse_mean": float(train_F.mean()),
            "val_rmse_mean": float(val_F.mean()),
            "test_rmse_mean": float(test_F.mean()),
            "test_r2_mean": float(r2_F.mean()),
        },
        "damage_gpr": {
            "n_latent_dims": len(train_C),
            "train_rmse_mean": float(train_C.mean()),
            "val_rmse_mean": float(val_C.mean()),
            "test_rmse_mean": float(test_C.mean()),
            "test_r2_mean": float(r2_C.mean()),
        }
    }

    with open(gpr_out / "gpr_training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("GPR Training Summary (Improved)")
    print("="*70)
    print(f"\nForce GPR ({results['force_gpr']['n_latent_dims']} latent dims):")
    print(f"  Test RMSE: {results['force_gpr']['test_rmse_mean']:.6f}")
    print(f"  Test R²:   {results['force_gpr']['test_r2_mean']:.4f}")
    
    print(f"\nDamage GPR ({results['damage_gpr']['n_latent_dims']} latent dims):")
    print(f"  Test RMSE: {results['damage_gpr']['test_rmse_mean']:.6f}")
    print(f"  Test R²:   {results['damage_gpr']['test_r2_mean']:.4f}")
    
    print("="*70)
    print(f"\nAll files saved to: {gpr_out}\n")


if __name__ == "__main__":
    main()