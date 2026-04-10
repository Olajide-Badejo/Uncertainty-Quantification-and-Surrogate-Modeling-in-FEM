"""
===============================================================================
SHAPE-SCALE SURROGATE MODEL CLASS
===============================================================================
Purpose: Wrapper class for loading and using trained GPR surrogates

Usage:
    model = ShapeScaleSurrogate.load(base_dir)
    u_force, F_pred, u_damage, D_pred = model.predict_curves(fc, E, cbot, ctop)
===============================================================================
"""

import numpy as np
from pathlib import Path
from joblib import load


class ShapeScaleSurrogate:
    """
    Shape-scale surrogate model for force and damage prediction.
    
    Components:
        - Force shape surrogate: Predicts PCA scores for normalized force curve
        - Force scale surrogate: Predicts scale factor
        - Damage surrogate: Predicts PCA scores for damage curve
    
    Attributes:
        pca_force: PCA model for force shapes
        pca_damage: PCA model for damage curves
        u_force: Displacement grid for force curves
        u_damage: Displacement grid for damage curves
        gpr_force_shape: GPR for force shape scores
        gpr_force_scale: GPR for force scale
        gpr_damage: GPR for damage scores
        scaler: Input feature scaler
    """
    
    def __init__(self, pca_dir: Path, sur_dir: Path):
        """
        Initialize surrogate from saved models.
        
        Parameters:
            pca_dir: Directory containing PCA models
            sur_dir: Directory containing GPR models
        """
        # Load PCA models
        self.pca_force = load(pca_dir / "pca_force.joblib")
        self.pca_damage = load(pca_dir / "pca_damage.joblib")
        
        # Load displacement grids
        self.u_force = np.load(pca_dir / "u_force.npy")
        self.u_damage = np.load(pca_dir / "u_damage.npy")
        
        # Load GPR surrogates
        self.gpr_force_shape = load(sur_dir / "gpr_force_shape.joblib")
        self.gpr_force_scale = load(sur_dir / "gpr_force_scale.joblib")
        self.gpr_damage = load(sur_dir / "gpr_damage.joblib")
        
        # Load input scaler
        self.scaler = load(sur_dir / "input_scaler.joblib")
    
    @classmethod
    def load(cls, base: Path):
        """
        Load surrogate from base directory.
        
        Parameters:
            base: Base directory (e.g., .../shape_scale_pipeline_clean)
            
        Returns:
            ShapeScaleSurrogate instance
        """
        pca_dir = base / "output_pca_shapes"
        sur_dir = base / "output_surrogates"
        return cls(pca_dir=pca_dir, sur_dir=sur_dir)
    
    def _prep_X(self, fc, E, cbot, ctop):
        """
        Prepare input features for prediction.
        
        Parameters:
            fc: Concrete compressive strength [MPa]
            E: Young's modulus [MPa]
            cbot: Bottom cover thickness [mm]
            ctop: Top cover thickness [mm]
            
        Returns:
            Scaled input array (1 × 4)
        """
        X = np.array([[fc, E, cbot, ctop]], dtype=float)
        return self.scaler.transform(X)
    
    def predict_curves(self, fc, E, cbot, ctop):
        """
        Predict force-displacement and damage-displacement curves.
        
        Parameters:
            fc: Concrete compressive strength [MPa]
            E: Young's modulus [MPa]
            cbot: Bottom cover thickness [mm]
            ctop: Top cover thickness [mm]
            
        Returns:
            u_force: Displacement grid for force [mm]
            F_pred: Predicted reaction force [N]
            u_damage: Displacement grid for damage [mm]
            D_pred: Predicted tension damage [-]
        """
        # Scale inputs
        Xs = self._prep_X(fc, E, cbot, ctop)
        
        # Predict force shape scores
        scores_force = self.gpr_force_shape.predict(Xs)[0]
        
        # Predict force scale
        scale_force = self.gpr_force_scale.predict(Xs)[0]
        
        # Predict damage scores
        scores_damage = self.gpr_damage.predict(Xs)[0]
        
        # Reconstruct normalized force shape from PCA
        F_norm = self.pca_force.inverse_transform(scores_force.reshape(1, -1))[0]
        
        # Apply scale to get actual force
        F_pred = F_norm * scale_force
        
        # Reconstruct damage curve from PCA
        D_pred = self.pca_damage.inverse_transform(scores_damage.reshape(1, -1))[0]
        
        return self.u_force, F_pred, self.u_damage, D_pred
    
    def predict_batch(self, fc_array, E_array, cbot_array, ctop_array):
        """
        Predict curves for multiple input samples.
        
        Parameters:
            fc_array: Array of concrete strengths [MPa]
            E_array: Array of Young's moduli [MPa]
            cbot_array: Array of bottom covers [mm]
            ctop_array: Array of top covers [mm]
            
        Returns:
            u_force: Displacement grid for force
            F_pred_batch: Predicted forces (n_samples × n_grid)
            u_damage: Displacement grid for damage
            D_pred_batch: Predicted damages (n_samples × n_grid)
        """
        n_samples = len(fc_array)
        
        # Prepare inputs
        X = np.column_stack([fc_array, E_array, cbot_array, ctop_array])
        Xs = self.scaler.transform(X)
        
        # Predict
        scores_force = self.gpr_force_shape.predict(Xs)
        scales_force = self.gpr_force_scale.predict(Xs)
        scores_damage = self.gpr_damage.predict(Xs)
        
        # Reconstruct
        F_norm_batch = self.pca_force.inverse_transform(scores_force)
        F_pred_batch = F_norm_batch * scales_force[:, np.newaxis]
        
        D_pred_batch = self.pca_damage.inverse_transform(scores_damage)
        
        return self.u_force, F_pred_batch, self.u_damage, D_pred_batch
    
    def get_info(self):
        """
        Get information about the surrogate model.
        
        Returns:
            dict: Model information
        """
        info = {
            "n_force_modes": self.pca_force.n_components_,
            "n_damage_modes": self.pca_damage.n_components_,
            "n_force_grid": len(self.u_force),
            "n_damage_grid": len(self.u_damage),
            "force_variance_explained": self.pca_force.explained_variance_ratio_.sum(),
            "damage_variance_explained": self.pca_damage.explained_variance_ratio_.sum(),
            "u_force_range": [float(self.u_force.min()), float(self.u_force.max())],
            "u_damage_range": [float(self.u_damage.min()), float(self.u_damage.max())],
        }
        return info