#!/usr/bin/env python3
"""
STEP 5: INTERACTIVE SURROGATE EXPLORER
=======================================
Interactive GUI for exploring surrogate model predictions.
Features sliders for material parameters, real-time curve updates,
and uncertainty visualization.

Usage:
    python step5_interactive_gui.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from surrogate_model import SurrogateModel


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """GUI configuration."""
    
    BASE = Path(
        r"."
    )
    
    PCA_DIR = BASE / "04_PCA" / "01_pca_reduction" / "models"
    SURROGATE_DIR = BASE / "04_PCA" / "01_pca_reduction" / "outputs"
    OUT_DIR = BASE / "04_PCA/05_interactive_outputs"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_E_from_fcm(fcm: float) -> float:
    """Eurocode 2 relationship: E = 22000 * (fcm/10)^0.3"""
    return 22000 * (fcm / 10.0) ** 0.3


def compute_fcm_from_E(E: float) -> float:
    """Inverse relationship: fcm = 10 * (E/22000)^(1/0.3)"""
    return 10.0 * (E / 22000.0) ** (1.0 / 0.3)


# ============================================================
# GUI CLASS
# ============================================================

class SurrogateExplorerGUI:
    """Interactive GUI for surrogate model exploration."""
    
    def __init__(self, model: SurrogateModel, config: Config):
        self.model = model
        self.config = config
        
        # Get grids
        self.u_force = model.u_grid_force
        self.u_damage = model.u_grid_damage
        
        # Track which slider was last changed (for fc/E sync)
        self.last_changed = None
        
        # Initial values
        self.fcm0 = 30.0
        self.E0 = compute_E_from_fcm(self.fcm0)
        self.cbot0 = 25.0
        self.ctop0 = 215.0
        
        # Setup GUI
        self._setup_figure()
        self._create_sliders()
        self._create_buttons()
        
        # Initial update
        self.update(None)
    
    def _setup_figure(self):
        """Setup matplotlib figure and axes."""
        plt.close("all")
        self.fig, (self.axF, self.axD) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(left=0.10, bottom=0.35, right=0.95, top=0.90, wspace=0.30)
        
        # Force plot
        self.force_line, = self.axF.plot(
            self.u_force, np.zeros_like(self.u_force),
            color="blue", lw=2.5, label="Mean prediction"
        )
        self.force_band = self.axF.fill_between(
            self.u_force,
            np.zeros_like(self.u_force),
            np.zeros_like(self.u_force),
            color="blue", alpha=0.2, label="±2σ uncertainty"
        )
        
        self.axF.set_xlabel("Displacement (mm)", fontsize=12)
        self.axF.set_ylabel("Force (N)", fontsize=12)
        self.axF.set_title("Force–Displacement Response", fontsize=14, fontweight="bold")
        self.axF.grid(True, alpha=0.3)
        self.axF.legend(fontsize=10)
        
        # Damage plot
        self.damage_line, = self.axD.plot(
            self.u_damage, np.zeros_like(self.u_damage),
            color="red", lw=2.5, label="Mean prediction"
        )
        self.damage_band = self.axD.fill_between(
            self.u_damage,
            np.zeros_like(self.u_damage),
            np.zeros_like(self.u_damage),
            color="red", alpha=0.2, label="±2σ uncertainty"
        )
        
        self.axD.set_xlabel("Displacement (mm)", fontsize=12)
        self.axD.set_ylabel("Compression Damage (DAMAGEC)", fontsize=12)
        self.axD.set_title("Damage Evolution", fontsize=14, fontweight="bold")
        self.axD.grid(True, alpha=0.3)
        self.axD.legend(fontsize=10)
    
    def _create_sliders(self):
        """Create parameter sliders."""
        axcolor = "lightgoldenrodyellow"
        
        # Slider positions
        ax_fcm = plt.axes([0.10, 0.26, 0.35, 0.03], facecolor=axcolor)
        ax_E = plt.axes([0.55, 0.26, 0.35, 0.03], facecolor=axcolor)
        ax_cbot = plt.axes([0.10, 0.21, 0.35, 0.03], facecolor=axcolor)
        ax_ctop = plt.axes([0.55, 0.21, 0.35, 0.03], facecolor=axcolor)
        ax_unc = plt.axes([0.10, 0.16, 0.35, 0.03], facecolor=axcolor)
        
        # Create sliders
        self.s_fcm = Slider(ax_fcm, "fc [MPa]", 20, 40, valinit=self.fcm0, valstep=0.5)
        self.s_E = Slider(ax_E, "E [MPa]", 28000, 38000, valinit=self.E0, valstep=100)
        self.s_cbot = Slider(ax_cbot, "c_bottom [mm]", 20, 35, valinit=self.cbot0, valstep=1)
        self.s_ctop = Slider(ax_ctop, "c_top [mm]", 200, 235, valinit=self.ctop0, valstep=1)
        self.s_unc = Slider(ax_unc, "Show Uncertainty", 0, 1, valinit=1, valstep=1)
        
        # Connect sliders
        self.s_fcm.on_changed(self.on_fcm_change)
        self.s_E.on_changed(self.on_E_change)
        self.s_cbot.on_changed(self.update)
        self.s_ctop.on_changed(self.update)
        self.s_unc.on_changed(self.update)
    
    def _create_buttons(self):
        """Create control buttons."""
        # Reset button
        btn_reset_ax = plt.axes([0.10, 0.09, 0.12, 0.05])
        self.btn_reset = Button(btn_reset_ax, "Reset")
        self.btn_reset.on_clicked(self.reset)
        
        # Save CSV button
        btn_csv_ax = plt.axes([0.25, 0.09, 0.12, 0.05])
        self.btn_save_csv = Button(btn_csv_ax, "Save CSV")
        self.btn_save_csv.on_clicked(lambda event: self.save_curves(False))
        
        # Save Excel button
        btn_xlsx_ax = plt.axes([0.40, 0.09, 0.12, 0.05])
        self.btn_save_xlsx = Button(btn_xlsx_ax, "Save Excel")
        self.btn_save_xlsx.on_clicked(lambda event: self.save_curves(True))
    
    def on_fcm_change(self, val):
        """Handle fc slider change."""
        self.last_changed = "fcm"
        self.update(None)
    
    def on_E_change(self, val):
        """Handle E slider change."""
        self.last_changed = "E"
        self.update(None)
    
    def update(self, val):
        """Update plots based on current slider values."""
        fcm = self.s_fcm.val
        E = self.s_E.val
        
        # Sync fc and E based on Eurocode 2
        if self.last_changed == "fcm":
            E_new = compute_E_from_fcm(fcm)
            if abs(E_new - E) > 1e-6:
                self.s_E.eventson = False
                self.s_E.set_val(E_new)
                self.s_E.eventson = True
            E = E_new
        
        elif self.last_changed == "E":
            fcm_new = compute_fcm_from_E(E)
            if abs(fcm_new - fcm) > 1e-6:
                self.s_fcm.eventson = False
                self.s_fcm.set_val(fcm_new)
                self.s_fcm.eventson = True
            fcm = fcm_new
        
        self.last_changed = None
        
        # Predict curves
        F_mean, F_std, D_mean, D_std = self.model.predict_curves(
            fc=fcm,
            E=E,
            cbot=self.s_cbot.val,
            ctop=self.s_ctop.val,
            return_uncertainty=True,
        )
        
        # Update force plot
        self.axF.clear()
        self.axF.plot(self.u_force, F_mean, color="blue", lw=2.5, label="Mean prediction")
        
        if self.s_unc.val == 1:
            self.axF.fill_between(
                self.u_force,
                F_mean - 2*F_std,
                F_mean + 2*F_std,
                color="blue", alpha=0.2, label="±2σ uncertainty"
            )
        
        self.axF.set_title(
            f"Force–Displacement (fc={fcm:.1f} MPa, E={E:.0f} MPa)",
            fontsize=14, fontweight="bold"
        )
        self.axF.set_xlabel("Displacement (mm)", fontsize=12)
        self.axF.set_ylabel("Force (N)", fontsize=12)
        self.axF.grid(True, alpha=0.3)
        self.axF.legend(fontsize=10)
        
        # Update damage plot
        self.axD.clear()
        self.axD.plot(self.u_damage, D_mean, color="red", lw=2.5, label="Mean prediction")
        
        if self.s_unc.val == 1:
            self.axD.fill_between(
                self.u_damage,
                D_mean - 2*D_std,
                D_mean + 2*D_std,
                color="red", alpha=0.2, label="±2σ uncertainty"
            )
        
        self.axD.set_title(
            f"Damage Evolution (c_bot={self.s_cbot.val:.0f}, c_top={self.s_ctop.val:.0f} mm)",
            fontsize=14, fontweight="bold"
        )
        self.axD.set_xlabel("Displacement (mm)", fontsize=12)
        self.axD.set_ylabel("Compression Damage", fontsize=12)
        self.axD.grid(True, alpha=0.3)
        self.axD.legend(fontsize=10)
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset all sliders to initial values."""
        self.s_fcm.reset()
        self.s_E.reset()
        self.s_cbot.reset()
        self.s_ctop.reset()
        self.s_unc.reset()
    
    def save_curves(self, to_excel: bool):
        """Save current predictions to file."""
        self.config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        fcm = self.s_fcm.val
        E = self.s_E.val
        cbot = self.s_cbot.val
        ctop = self.s_ctop.val
        
        F_mean, F_std, D_mean, D_std = self.model.predict_curves(
            fc=fcm, E=E, cbot=cbot, ctop=ctop, return_uncertainty=True
        )
        
        # Create dataframe
        df = pd.DataFrame({
            "u_force": self.u_force,
            "force_mean": F_mean,
            "force_std": F_std,
            "u_damage": self.u_damage,
            "damage_mean": D_mean,
            "damage_std": D_std,
        })
        
        # Add metadata
        metadata = {
            "fc": fcm,
            "E": E,
            "c_bottom": cbot,
            "c_top": ctop,
        }
        
        if to_excel:
            filepath = self.config.OUT_DIR / "surrogate_prediction.xlsx"
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="curves", index=False)
                pd.DataFrame([metadata]).to_excel(writer, sheet_name="parameters", index=False)
            print(f"✓ Saved to: {filepath}")
        else:
            filepath = self.config.OUT_DIR / "surrogate_prediction.csv"
            df.to_csv(filepath, index=False)
            
            # Save metadata separately
            meta_path = self.config.OUT_DIR / "parameters.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            print(f"✓ Saved to: {filepath}")
            print(f"✓ Parameters: {meta_path}")
    
    def show(self):
        """Display the GUI."""
        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    """Launch interactive GUI."""
    
    config = Config()
    
    print("\n" + "="*60)
    print("INTERACTIVE SURROGATE EXPLORER")
    print("="*60)
    print("\nLoading model...")
    
    model = SurrogateModel.load(
        pca_dir=config.PCA_DIR,
        surrogate_dir=config.SURROGATE_DIR,
    )
    
    print("✓ Model loaded")
    print("\nLaunching GUI...")
    print("\nControls:")
    print("  - Adjust sliders to explore parameter space")
    print("  - fc and E are linked via Eurocode 2 relationship")
    print("  - Toggle 'Show Uncertainty' to hide/show ±2σ bands")
    print("  - 'Reset' returns to initial values")
    print("  - 'Save CSV' or 'Save Excel' exports current prediction")
    print("\n" + "="*60 + "\n")
    
    gui = SurrogateExplorerGUI(model, config)
    gui.show()


if __name__ == "__main__":
    main()