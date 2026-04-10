#!/usr/bin/env python3
"""
Improved AE Surrogate Model Wrapper
====================================
Handles both standard and monotonic autoencoders.
"""

import numpy as np
import torch
import joblib
from pathlib import Path
from ae_model import ImprovedCurveAutoencoder
from ae_model import MonotonicDamageAutoencoder


class ImprovedAESurrogateModel:
    """
    Autoencoder + GPR surrogate for force and compression damage curves.
    Supports both standard and monotonic autoencoders.
    
    Inputs:  [c_nom_bottom, c_nom_top, f_cm]
    Outputs: (u_force, F_pred_norm), (u_damage, D_pred_norm)
    """

    def __init__(self, base_path, use_improved=True):
        base = Path(base_path)

        # Choose directory based on model version
        if use_improved:
            ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder_improved"
            gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates_improved"
        else:
            ae_dir = base / "05_autoencoder_gpr" / "output_autoencoder"
            gpr_dir = base / "05_autoencoder_gpr" / "output_surrogates"

        # --------------------------------------------------------
        # LOAD SCALER + GPR MODELS
        # --------------------------------------------------------
        self.scaler = joblib.load(gpr_dir / "input_scaler.joblib")

        self.gpr_force = joblib.load(gpr_dir / "gpr_force_latent_gpr.joblib")
        self.gpr_damage = joblib.load(gpr_dir / "gpr_damage_latent_gpr.joblib")

        # --------------------------------------------------------
        # LOAD AE DECODERS
        # --------------------------------------------------------
        self.ae_force = self._load_ae(ae_dir / "ae_force.pt")
        self.ae_damage = self._load_ae(ae_dir / "ae_damage.pt")

        # --------------------------------------------------------
        # LOAD DISPLACEMENT GRIDS
        # --------------------------------------------------------
        self.u_force = np.load(ae_dir / "u_force.npy")
        self.u_damage = np.load(ae_dir / "u_damage.npy")

    def _load_ae(self, path):
        """Load autoencoder, automatically detecting monotonic vs standard."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        # Check if this is a monotonic model
        use_monotonic = ckpt.get("use_monotonic", False)
        
        if use_monotonic:
            model = MonotonicDamageAutoencoder(
                n_points=ckpt["n_points"],
                latent_dim=ckpt["latent_dim"]
            )
        else:
            model = ImprovedCurveAutoencoder(
                n_points=ckpt["n_points"],
                latent_dim=ckpt["latent_dim"]
            )
        
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    # ------------------------------------------------------------
    # PREDICT LATENT VECTOR USING GPR
    # ------------------------------------------------------------
    def _predict_latent(self, X_scaled, gpr_list):
        z = np.zeros(len(gpr_list))
        for i, gpr in enumerate(gpr_list):
            z[i] = gpr.predict(X_scaled)[0]
        return z

    # ------------------------------------------------------------
    # DECODE CURVE FROM LATENT VECTOR
    # ------------------------------------------------------------
    def _decode(self, ae_model, z):
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            curve = ae_model.decode(z_t).numpy()[0]
        return curve

    # ------------------------------------------------------------
    # MAIN PREDICTION FUNCTION
    # ------------------------------------------------------------
    def predict(self, cbot, ctop, fcm):
        """
        Returns:
            u_force,  F_pred_norm
            u_damage, D_pred_norm
        """

        # 1) Build input vector
        X = np.array([[cbot, ctop, fcm]])

        # 2) Scale
        Xs = self.scaler.transform(X)

        # 3) Predict latent vectors
        zF = self._predict_latent(Xs, self.gpr_force)
        zD = self._predict_latent(Xs, self.gpr_damage)

        # 4) Decode curves (normalized)
        F_pred = self._decode(self.ae_force, zF)
        D_pred = self._decode(self.ae_damage, zD)

        return self.u_force, F_pred, self.u_damage, D_pred