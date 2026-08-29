#!/usr/bin/env python3
"""
SURROGATE MODEL CLASS
=====================
Unified interface for PCA+GPR surrogate modeling.
Handles loading, prediction, and uncertainty quantification.

Updated to use DAMAGEC (compression damage) instead of crack metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler


class SurrogateModel:
    """
    Unified PCA+GPR surrogate for force & damage curves.
    
    Attributes:
        scaler: Input feature scaler
        pca_force: PCA model for force curves
        pca_damage: PCA model for damage curves
        u_grid_force: Displacement grid for force
        u_grid_damage: Displacement grid for damage
        force_scale: Global normalization scale for force
        damage_scale: Global normalization scale for damage
        force_gpr_list: List of GPR models for force PCA scores
        damage_gpr_list: List of GPR models for damage PCA scores
    """

    def __init__(
        self,
        scaler: StandardScaler,
        pca_force: PCA,
        pca_damage: PCA,
        u_grid_force: np.ndarray,
        u_grid_damage: np.ndarray,
        force_scale: float,
        damage_scale: float,
        force_gpr_list: List[GaussianProcessRegressor],
        damage_gpr_list: List[GaussianProcessRegressor],
    ):
        self.scaler = scaler
        self.pca_force = pca_force
        self.pca_damage = pca_damage
        
        self.u_grid_force = u_grid_force
        self.u_grid_damage = u_grid_damage
        
        self.force_scale = float(force_scale)
        self.damage_scale = float(damage_scale)
        
        self.force_gpr_list = force_gpr_list
        self.damage_gpr_list = damage_gpr_list

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    @classmethod
    def load(
        cls,
        pca_dir: str | Path,
        surrogate_dir: str | Path,
    ) -> "SurrogateModel":
        """
        Load trained surrogate model from disk.
        
        Args:
            pca_dir: Directory containing PCA models and metadata
            surrogate_dir: Directory containing trained GPR models
            
        Returns:
            Loaded SurrogateModel instance
        """
        pca_dir = Path(pca_dir)
        surrogate_dir = Path(surrogate_dir)

        # Load scaler
        scaler = joblib.load(surrogate_dir / "input_scaler.joblib")

        # Load PCA models
        pca_force = joblib.load(pca_dir / "pca_force.joblib")
        pca_damage = joblib.load(pca_dir / "pca_damage.joblib")
        
        # Load metadata
        meta = json.loads((pca_dir / "meta.json").read_text())

        u_grid_force = np.array(meta["u_grid_force"], dtype=float)
        u_grid_damage = np.array(meta["u_grid_damage"], dtype=float)
        force_scale = float(meta.get("global_force_scale", 1.0))
        damage_scale = float(meta.get("global_damage_scale", 1.0))

        # Load GPR models
        force_gpr_list = joblib.load(surrogate_dir / "force_gpr_models.joblib")
        damage_gpr_list = joblib.load(surrogate_dir / "damage_gpr_models.joblib")

        return cls(
            scaler=scaler,
            pca_force=pca_force,
            pca_damage=pca_damage,
            u_grid_force=u_grid_force,
            u_grid_damage=u_grid_damage,
            force_scale=force_scale,
            damage_scale=damage_scale,
            force_gpr_list=force_gpr_list,
            damage_gpr_list=damage_gpr_list,
        )

    # --------------------------------------------------------
    # PREDICTION HELPERS
    # --------------------------------------------------------
    @staticmethod
    def build_X(fc: float, E: float, cbot: float, ctop: float) -> np.ndarray:
        """Build input feature array from material parameters."""
        return np.array([[fc, E, cbot, ctop]], dtype=float)

    def _predict_pca_scores(
        self,
        X_raw: np.ndarray,
        target: str = "force",
        return_std: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict PCA scores with optional uncertainty.
        
        Args:
            X_raw: Raw input features (n_samples, n_features)
            target: "force" or "damage"
            return_std: Whether to return standard deviations
            
        Returns:
            means: Predicted PCA scores (n_samples, n_components)
            stds: Standard deviations (n_samples, n_components) or None
        """
        X_scaled = self.scaler.transform(X_raw)
        
        gpr_list = self.force_gpr_list if target == "force" else self.damage_gpr_list
        
        n_components = len(gpr_list)
        n_samples = X_scaled.shape[0]
        
        means = np.zeros((n_samples, n_components), dtype=float)
        stds = np.zeros((n_samples, n_components), dtype=float) if return_std else None
        
        for i, gpr in enumerate(gpr_list):
            if return_std:
                m, s = gpr.predict(X_scaled, return_std=True)
                means[:, i] = m
                stds[:, i] = s
            else:
                means[:, i] = gpr.predict(X_scaled)
        
        return means, stds

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------
    def predict_scores(
        self,
        fc: float,
        E: float,
        cbot: float,
        ctop: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict PCA scores for given material parameters.
        
        Args:
            fc: Concrete compressive strength (MPa)
            E: Young's modulus (MPa)
            cbot: Bottom cover (mm)
            ctop: Top cover (mm)
            
        Returns:
            force_scores: Force PCA scores (1, n_force_components)
            damage_scores: Damage PCA scores (1, n_damage_components)
        """
        X_raw = self.build_X(fc, E, cbot, ctop)
        force_scores, _ = self._predict_pca_scores(X_raw, "force", return_std=False)
        damage_scores, _ = self._predict_pca_scores(X_raw, "damage", return_std=False)
        return force_scores, damage_scores

    def predict_curves(
        self,
        fc: float,
        E: float,
        cbot: float,
        ctop: float,
        return_uncertainty: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict full force and damage curves with uncertainty.
        
        Args:
            fc: Concrete compressive strength (MPa)
            E: Young's modulus (MPa)
            cbot: Bottom cover (mm)
            ctop: Top cover (mm)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            If return_uncertainty=True:
                force_mean: Predicted force curve (n_grid,)
                force_std: Force uncertainty (n_grid,)
                damage_mean: Predicted damage curve (n_grid,)
                damage_std: Damage uncertainty (n_grid,)
            If return_uncertainty=False:
                force_mean: Predicted force curve (n_grid,)
                damage_mean: Predicted damage curve (n_grid,)
        """
        X_raw = self.build_X(fc, E, cbot, ctop)

        # Predict PCA scores
        force_scores_mean, force_scores_std = self._predict_pca_scores(
            X_raw, "force", return_std=True
        )
        damage_scores_mean, damage_scores_std = self._predict_pca_scores(
            X_raw, "damage", return_std=True
        )

        # Inverse PCA transform (normalized domain)
        force_mean_norm = self.pca_force.inverse_transform(force_scores_mean)[0]
        damage_mean_norm = self.pca_damage.inverse_transform(damage_scores_mean)[0]

        # De-normalize
        force_mean = force_mean_norm * self.force_scale
        damage_mean = damage_mean_norm * self.damage_scale

        if return_uncertainty:
            # Propagate uncertainty through PCA
            force_plus_norm = self.pca_force.inverse_transform(
                force_scores_mean + force_scores_std
            )[0]
            damage_plus_norm = self.pca_damage.inverse_transform(
                damage_scores_mean + damage_scores_std
            )[0]

            force_std = (force_plus_norm - force_mean_norm) * self.force_scale
            damage_std = (damage_plus_norm - damage_mean_norm) * self.damage_scale

            return force_mean, force_std, damage_mean, damage_std
        else:
            return force_mean, damage_mean

    def get_grid_info(self) -> dict:
        """Get displacement grid information."""
        return {
            "force_grid": self.u_grid_force,
            "damage_grid": self.u_grid_damage,
            "force_scale": self.force_scale,
            "damage_scale": self.damage_scale,
        }

    @staticmethod
    def normalized_uncertainty(
        curve_mean: np.ndarray, 
        curve_std: np.ndarray
    ) -> float:
        """
        Compute normalized uncertainty metric.
        
        Args:
            curve_mean: Mean curve prediction
            curve_std: Standard deviation of prediction
            
        Returns:
            Normalized uncertainty (0 to ~1)
        """
        denom = max(np.max(np.abs(curve_mean)), 1e-9)
        return float(np.mean(np.abs(curve_std)) / denom)