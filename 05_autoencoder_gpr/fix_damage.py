#!/usr/bin/env python3
"""
QUICK START: Fix Damage Predictions
====================================
Automated script to retrain models with improvements.
Run this to fix the oscillating damage curves.
"""

from pathlib import Path
import subprocess
import sys

def run_script(script_path, description):
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    base = Path(
        r".\Scripts"
    )
    
    print("="*70)
    print("AUTOMATED FIX FOR DAMAGE CURVE PREDICTIONS")
    print("="*70)
    print("\nThis script will:")
    print("  1. Retrain autoencoders with monotonicity constraint")
    print("  2. Re-encode curves to new latent space")
    print("  3. Retrain GPR models")
    print("  4. Run diagnostic comparison")
    print(f"\nBase directory: {base}")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # --------------------------------------------------------
    # STEP 1: RETRAIN AUTOENCODERS
    # --------------------------------------------------------
    script1 = base / "2_train_autoencoders_IMPROVED.py"
    if not script1.exists():
        print(f"\n❌ Script not found: {script1}")
        print("Please copy the improved scripts to your directory first.")
        return
    
    success1 = run_script(script1, "Step 1: Retrain Autoencoders (with improvements)")
    if not success1:
        print("\n⚠️  Autoencoder training failed. Please check errors above.")
        return
    
    # --------------------------------------------------------
    # STEP 2: RE-ENCODE CURVES
    # --------------------------------------------------------
    # Need to create modified version that uses improved directory
    print("\n" + "="*70)
    print("Step 2: Re-encoding curves")
    print("="*70)
    print("\n⚠️  You need to manually run the modified 3_encode_curves.py")
    print("Modify line:")
    print("  ae_out = base / 'autoencoder_gpr' / 'output_autoencoder_improved'")
    print("  gpr_out = base / 'autoencoder_gpr' / 'output_surrogates_improved'")
    
    response = input("\nHave you modified and run 3_encode_curves.py? (y/n): ")
    if response.lower() != 'y':
        print("\nPlease run it, then restart this script.")
        return
    
    # --------------------------------------------------------
    # STEP 3: RETRAIN GPR
    # --------------------------------------------------------
    # Similar modification needed
    print("\n" + "="*70)
    print("Step 3: Retraining GPR")
    print("="*70)
    print("\n⚠️  You need to manually run the modified 4_train_gpr.py")
    print("Modify lines to use:")
    print("  gpr_out = base / 'autoencoder_gpr' / 'output_surrogates_improved'")
    
    response = input("\nHave you modified and run 4_train_gpr.py? (y/n): ")
    if response.lower() != 'y':
        print("\nPlease run it, then restart this script.")
        return
    
    # --------------------------------------------------------
    # STEP 4: DIAGNOSTIC COMPARISON
    # --------------------------------------------------------
    script4 = base / "DIAGNOSTIC_comparison.py"
    if script4.exists():
        success4 = run_script(script4, "Step 4: Diagnostic Comparison")
    else:
        print(f"\n⚠️  Diagnostic script not found: {script4}")
    
    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("PROCESS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review diagnostic plots in:")
    print("     autoencoder_gpr/diagnostics_comparison/")
    print("  2. Check training_summary.json for metrics")
    print("  3. If satisfied, update your UQ script to use improved model")
    print("  4. Re-run UQ with fixed surrogate")
    print("\nTo use improved model in UQ:")
    print("  from ae_surrogate_model_improved import ImprovedAESurrogateModel")
    print("  model = ImprovedAESurrogateModel(base, use_improved=True)")
    

if __name__ == "__main__":
    main()