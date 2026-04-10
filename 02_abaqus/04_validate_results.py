"""
===============================================================================
STEP 2.4: VALIDATE EXTRACTED DATA
===============================================================================
Purpose: Quality checks on extracted FEM results

Validation checks:
    1. Completeness: All expected files present
    2. Data integrity: No NaN/Inf values, monotonic displacement
    3. Physical plausibility: Reaction forces positive, damage in [0,1]
    4. Outlier detection: Identify anomalous results
    5. Convergence indicators: Check for non-converged solutions

Outputs:
    - Validation report (console + text file)
    - List of problematic samples
    - Statistical summary
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_ROOT = ROOT / "02_abaqus" / "extracted_data"
RESULTS_ROOT = ROOT / "02_abaqus" / "results"

# Validation thresholds
MAX_DISPLACEMENT = 100.0  # mm (unrealistic if exceeded)
MAX_REACTION = 200000.0   # N (unrealistic if exceeded)
MIN_POINTS = 10           # Minimum data points in curve

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def check_file_completeness(sample_id):
    """
    Check if all expected output files exist for a sample.
    
    Returns:
        dict: Status of each file type
    """
    job_name = f"sample_{sample_id:03d}"
    
    status = {
        'load_displacement': (EXTRACTED_ROOT / f"{job_name}_load_displacement.csv").exists(),
        'damage': (EXTRACTED_ROOT / f"{job_name}_damage.csv").exists(),
        'odb': (RESULTS_ROOT / job_name / f"{job_name}.odb").exists(),
    }
    
    return status


def validate_load_displacement(file_path):
    """
    Validate load-displacement data.
    
    Returns:
        dict: Validation results
    """
    issues = []
    
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ['time', 'displacement', 'reaction_force']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return {'valid': False, 'issues': issues, 'stats': {}}
        
        # Check for NaN/Inf
        if df.isnull().any().any():
            issues.append("Contains NaN values")
        
        if np.isinf(df.values).any():
            issues.append("Contains Inf values")
        
        # Check number of points
        if len(df) < MIN_POINTS:
            issues.append(f"Too few points: {len(df)} < {MIN_POINTS}")
        
        # Check displacement monotonicity (should generally increase)
        disp = df['displacement'].values
        if not np.all(np.diff(disp) >= -1e-6):  # Allow small numerical errors
            non_monotonic_count = np.sum(np.diff(disp) < -1e-6)
            if non_monotonic_count > 5:  # More than 5 backward steps is suspicious
                issues.append(f"Non-monotonic displacement ({non_monotonic_count} reversals)")
        
        # Check for negative reaction forces (should be positive in tension/compression)
        if np.any(df['reaction_force'] < -1e-6):
            issues.append("Negative reaction forces detected")
        
        # Check for unrealistic magnitudes
        max_disp = df['displacement'].max()
        max_rf = df['reaction_force'].max()
        
        if max_disp > MAX_DISPLACEMENT:
            issues.append(f"Unrealistic displacement: {max_disp:.2f} mm")
        
        if max_rf > MAX_REACTION:
            issues.append(f"Unrealistic reaction force: {max_rf:.2f} N")
        
        # Check for plateau (possible non-convergence)
        if len(df) > 20:
            last_20_disp = disp[-20:]
            if np.std(last_20_disp) < 1e-8:
                issues.append("Displacement plateau detected (possible non-convergence)")
        
        # Compute statistics
        stats = {
            'n_points': len(df),
            'max_displacement': max_disp,
            'max_reaction': max_rf,
            'final_displacement': disp[-1] if len(disp) > 0 else 0,
            'final_reaction': df['reaction_force'].values[-1] if len(df) > 0 else 0
        }
        
        valid = len(issues) == 0
        
        return {'valid': valid, 'issues': issues, 'stats': stats}
        
    except Exception as e:
        return {'valid': False, 'issues': [f"Error reading file: {str(e)}"], 'stats': {}}


def validate_damage(file_path):
    """
    Validate damage evolution data.
    
    Returns:
        dict: Validation results
    """
    issues = []
    
    try:
        df = pd.read_csv(file_path)
        
        # Check for NaN/Inf
        if df.isnull().any().any():
            issues.append("Contains NaN values")
        
        if np.isinf(df.values).any():
            issues.append("Contains Inf values")
        
        # Damage variables should be in [0, 1]
        damage_cols = ['damagec_max', 'damaget_max', 'sdeg_max', 'damagec_avg', 'damaget_avg']
        
        for col in damage_cols:
            if col in df.columns:
                values = df[col].values
                
                if np.any(values < -1e-6):
                    issues.append(f"{col}: Negative values detected")
                
                if np.any(values > 1.0 + 1e-6):
                    issues.append(f"{col}: Values > 1.0 detected")
        
        # Check for monotonic damage (damage should not decrease)
        for col in damage_cols:
            if col in df.columns:
                values = df[col].values
                if not np.all(np.diff(values) >= -1e-6):
                    decreasing_count = np.sum(np.diff(values) < -1e-6)
                    if decreasing_count > 3:
                        issues.append(f"{col}: Non-monotonic ({decreasing_count} reversals)")
        
        # Compute statistics
        stats = {
            'final_damagec_max': df['damagec_max'].values[-1] if 'damagec_max' in df.columns else 0,
            'final_damaget_max': df['damaget_max'].values[-1] if 'damaget_max' in df.columns else 0,
            'final_sdeg_max': df['sdeg_max'].values[-1] if 'sdeg_max' in df.columns else 0
        }
        
        valid = len(issues) == 0
        
        return {'valid': valid, 'issues': issues, 'stats': stats}
        
    except Exception as e:
        return {'valid': False, 'issues': [f"Error reading file: {str(e)}"], 'stats': {}}


# ============================================================
# MAIN VALIDATION ROUTINE
# ============================================================

def main():
    
    print("=" * 70)
    print("VALIDATING EXTRACTED DATA")
    print("=" * 70)
    
    # Find all extracted load-displacement files
    ld_files = sorted(EXTRACTED_ROOT.glob("sample_*_load_displacement.csv"))
    
    print(f"\nFound {len(ld_files)} load-displacement files to validate\n")
    
    if len(ld_files) == 0:
        print("⚠ No extracted data found!")
        return
    
    # Validation results
    results = []
    
    for ld_file in ld_files:
        
        # Extract sample ID
        sample_id = int(ld_file.stem.split("_")[1])
        job_name = f"sample_{sample_id:03d}"
        
        print(f"Validating {job_name}...")
        
        # --------------------------------------------------------
        # Check file completeness
        # --------------------------------------------------------
        file_status = check_file_completeness(sample_id)
        
        # --------------------------------------------------------
        # Validate load-displacement
        # --------------------------------------------------------
        ld_result = validate_load_displacement(ld_file)
        
        # --------------------------------------------------------
        # Validate damage data
        # --------------------------------------------------------
        damage_file = EXTRACTED_ROOT / f"{job_name}_damage.csv"
        if damage_file.exists():
            damage_result = validate_damage(damage_file)
        else:
            damage_result = {'valid': False, 'issues': ['Damage file not found'], 'stats': {}}
        
        # --------------------------------------------------------
        # Compile results
        # --------------------------------------------------------
        all_valid = ld_result['valid'] and damage_result['valid'] and all(file_status.values())
        
        all_issues = []
        if not all(file_status.values()):
            missing = [k for k, v in file_status.items() if not v]
            all_issues.append(f"Missing files: {missing}")
        all_issues.extend(ld_result['issues'])
        all_issues.extend(damage_result['issues'])
        
        result = {
            'sample_id': sample_id,
            'job_name': job_name,
            'valid': all_valid,
            'issues': all_issues,
            'ld_stats': ld_result['stats'],
            'damage_stats': damage_result['stats']
        }
        
        results.append(result)
        
        # Print status
        if all_valid:
            print(f"  ✓ VALID")
        else:
            print(f"  ✗ ISSUES FOUND:")
            for issue in all_issues[:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(all_issues) > 3:
                print(f"    ... and {len(all_issues)-3} more")
    
    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    n_total = len(results)
    n_valid = sum(1 for r in results if r['valid'])
    n_invalid = n_total - n_valid
    
    print(f"Total samples:    {n_total}")
    print(f"Valid:            {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"Invalid:          {n_invalid} ({100*n_invalid/n_total:.1f}%)")
    
    # --------------------------------------------------------
    # List problematic samples
    # --------------------------------------------------------
    if n_invalid > 0:
        print("\n" + "-" * 70)
        print("PROBLEMATIC SAMPLES:")
        print("-" * 70)
        
        for r in results:
            if not r['valid']:
                print(f"\n{r['job_name']}:")
                for issue in r['issues']:
                    print(f"  - {issue}")
    
    # --------------------------------------------------------
    # Statistical summary of valid results
    # --------------------------------------------------------
    if n_valid > 0:
        print("\n" + "-" * 70)
        print("STATISTICS (VALID SAMPLES ONLY):")
        print("-" * 70)
        
        valid_results = [r for r in results if r['valid']]
        
        # Extract statistics
        max_disps = [r['ld_stats']['max_displacement'] for r in valid_results if 'max_displacement' in r['ld_stats']]
        max_rfs = [r['ld_stats']['max_reaction'] for r in valid_results if 'max_reaction' in r['ld_stats']]
        final_damagec = [r['damage_stats']['final_damagec_max'] for r in valid_results if 'final_damagec_max' in r['damage_stats']]
        final_damaget = [r['damage_stats']['final_damaget_max'] for r in valid_results if 'final_damaget_max' in r['damage_stats']]
        
        if max_disps:
            print(f"\nMax displacement [mm]:")
            print(f"  Mean:   {np.mean(max_disps):.3f}")
            print(f"  Std:    {np.std(max_disps):.3f}")
            print(f"  Range:  [{np.min(max_disps):.3f}, {np.max(max_disps):.3f}]")
        
        if max_rfs:
            print(f"\nMax reaction force [N]:")
            print(f"  Mean:   {np.mean(max_rfs):.1f}")
            print(f"  Std:    {np.std(max_rfs):.1f}")
            print(f"  Range:  [{np.min(max_rfs):.1f}, {np.max(max_rfs):.1f}]")
        
        if final_damagec:
            print(f"\nFinal compression damage:")
            print(f"  Mean:   {np.mean(final_damagec):.3f}")
            print(f"  Std:    {np.std(final_damagec):.3f}")
            print(f"  Range:  [{np.min(final_damagec):.3f}, {np.max(final_damagec):.3f}]")
        
        if final_damaget:
            print(f"\nFinal tension damage:")
            print(f"  Mean:   {np.mean(final_damaget):.3f}")
            print(f"  Std:    {np.std(final_damaget):.3f}")
            print(f"  Range:  [{np.min(final_damaget):.3f}, {np.max(final_damaget):.3f}]")
    
    # --------------------------------------------------------
    # Save validation report
    # --------------------------------------------------------
    report_file = EXTRACTED_ROOT / "validation_report.txt"
    with open(report_file, 'w') as f:
        f.write("VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total samples:    {n_total}\n")
        f.write(f"Valid:            {n_valid} ({100*n_valid/n_total:.1f}%)\n")
        f.write(f"Invalid:          {n_invalid} ({100*n_invalid/n_total:.1f}%)\n\n")
        
        if n_invalid > 0:
            f.write("PROBLEMATIC SAMPLES:\n")
            f.write("-" * 70 + "\n")
            for r in results:
                if not r['valid']:
                    f.write(f"\n{r['job_name']}:\n")
                    for issue in r['issues']:
                        f.write(f"  - {issue}\n")
    
    print(f"\n✓ Validation report saved to: {report_file}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
