"""
===============================================================================
STEP 2.6: GENERATE SUMMARY REPORT
===============================================================================
Purpose: Create comprehensive summary of all FEM simulations

Report includes:
    - Execution statistics (success/failure rates)
    - Data quality metrics
    - Statistical summary of results
    - List of problematic samples
    - Recommendations for next steps

Output: HTML and text reports
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "02_abaqus" / "results"
EXTRACTED_ROOT = ROOT / "02_abaqus" / "extracted_data"
LHS_CSV = ROOT / "uq_lhs_samples_training.csv"
REPORT_DIR = ROOT / "02_abaqus" / "fem_reports"

REPORT_DIR.mkdir(exist_ok=True)

# ============================================================
# DATA COLLECTION
# ============================================================

def collect_execution_stats():
    """
    Collect statistics from metadata files.
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'timeout': 0,
        'runtimes': []
    }
    
    for metadata_file in RESULTS_ROOT.glob("*/metadata.txt"):
        stats['total'] += 1
        
        content = metadata_file.read_text()
        
        if 'status: SUCCESS' in content:
            stats['success'] += 1
        elif 'status: FAILED' in content:
            stats['failed'] += 1
        elif 'status: TIMEOUT' in content:
            stats['timeout'] += 1
        
        # Extract runtime
        for line in content.split('\n'):
            if line.startswith('runtime:'):
                try:
                    runtime = float(line.split(':')[1].strip().split()[0])
                    stats['runtimes'].append(runtime)
                except:
                    pass
    
    return stats


def collect_extraction_stats():
    """
    Count extracted files.
    """
    stats = {
        'load_displacement': len(list(EXTRACTED_ROOT.glob("*_load_displacement.csv"))),
        'damage': len(list(EXTRACTED_ROOT.glob("*_damage.csv"))),
    }
    
    return stats


def load_validation_results():
    """
    Load validation report if exists.
    """
    validation_file = EXTRACTED_ROOT / "validation_report.txt"
    
    if validation_file.exists():
        return validation_file.read_text()
    else:
        return "Validation report not found. Run validation script first."


def compute_output_statistics():
    """
    Compute statistics on extracted outputs.
    """
    stats = {
        'max_loads': [],
        'max_displacements': [],
        'final_damagec': [],
        'final_damaget': []
    }
    
    # Load all load-displacement files
    for ld_file in EXTRACTED_ROOT.glob("*_load_displacement.csv"):
        try:
            df = pd.read_csv(ld_file)
            stats['max_loads'].append(df['reaction_force'].max())
            stats['max_displacements'].append(df['displacement'].max())
        except:
            pass
    
    # Load all damage files
    for damage_file in EXTRACTED_ROOT.glob("*_damage.csv"):
        try:
            df = pd.read_csv(damage_file)
            stats['final_damagec'].append(df['damagec_max'].values[-1])
            stats['final_damaget'].append(df['damaget_max'].values[-1])
        except:
            pass
    
    # Convert to arrays
    for key in stats:
        stats[key] = np.array(stats[key])
    
    return stats


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_text_report():
    """
    Generate comprehensive text report.
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("FEM SIMULATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # --------------------------------------------------------
    # Execution Statistics
    # --------------------------------------------------------
    report_lines.append("-" * 80)
    report_lines.append("1. EXECUTION STATISTICS")
    report_lines.append("-" * 80)
    
    exec_stats = collect_execution_stats()
    
    report_lines.append(f"Total jobs submitted:     {exec_stats['total']}")
    report_lines.append(f"Successful:               {exec_stats['success']} ({100*exec_stats['success']/max(exec_stats['total'],1):.1f}%)")
    report_lines.append(f"Failed:                   {exec_stats['failed']} ({100*exec_stats['failed']/max(exec_stats['total'],1):.1f}%)")
    report_lines.append(f"Timeout:                  {exec_stats['timeout']} ({100*exec_stats['timeout']/max(exec_stats['total'],1):.1f}%)")
    
    if exec_stats['runtimes']:
        runtimes = np.array(exec_stats['runtimes'])
        report_lines.append(f"\nRuntime statistics:")
        report_lines.append(f"  Total runtime:          {runtimes.sum()/60:.1f} minutes")
        report_lines.append(f"  Mean runtime per job:   {runtimes.mean():.1f} seconds")
        report_lines.append(f"  Std runtime:            {runtimes.std():.1f} seconds")
        report_lines.append(f"  Min/Max runtime:        {runtimes.min():.1f} / {runtimes.max():.1f} seconds")
    
    report_lines.append("")
    
    # --------------------------------------------------------
    # Extraction Statistics
    # --------------------------------------------------------
    report_lines.append("-" * 80)
    report_lines.append("2. DATA EXTRACTION STATISTICS")
    report_lines.append("-" * 80)
    
    extract_stats = collect_extraction_stats()
    
    report_lines.append(f"Load-displacement files:  {extract_stats['load_displacement']}")
    report_lines.append(f"Damage data files:        {extract_stats['damage']}")
    report_lines.append("")
    
    # --------------------------------------------------------
    # Output Statistics
    # --------------------------------------------------------
    report_lines.append("-" * 80)
    report_lines.append("3. OUTPUT STATISTICS")
    report_lines.append("-" * 80)
    
    output_stats = compute_output_statistics()
    
    if len(output_stats['max_loads']) > 0:
        report_lines.append("Maximum Load [N]:")
        report_lines.append(f"  Mean:   {output_stats['max_loads'].mean():.2f}")
        report_lines.append(f"  Std:    {output_stats['max_loads'].std():.2f}")
        report_lines.append(f"  Min:    {output_stats['max_loads'].min():.2f}")
        report_lines.append(f"  Max:    {output_stats['max_loads'].max():.2f}")
        report_lines.append("")
    
    if len(output_stats['max_displacements']) > 0:
        report_lines.append("Maximum Displacement [mm]:")
        report_lines.append(f"  Mean:   {output_stats['max_displacements'].mean():.4f}")
        report_lines.append(f"  Std:    {output_stats['max_displacements'].std():.4f}")
        report_lines.append(f"  Min:    {output_stats['max_displacements'].min():.4f}")
        report_lines.append(f"  Max:    {output_stats['max_displacements'].max():.4f}")
        report_lines.append("")
    
    if len(output_stats['final_damagec']) > 0:
        report_lines.append("Final Compression Damage [-]:")
        report_lines.append(f"  Mean:   {output_stats['final_damagec'].mean():.4f}")
        report_lines.append(f"  Std:    {output_stats['final_damagec'].std():.4f}")
        report_lines.append(f"  Min:    {output_stats['final_damagec'].min():.4f}")
        report_lines.append(f"  Max:    {output_stats['final_damagec'].max():.4f}")
        report_lines.append("")
    
    if len(output_stats['final_damaget']) > 0:
        report_lines.append("Final Tension Damage [-]:")
        report_lines.append(f"  Mean:   {output_stats['final_damaget'].mean():.4f}")
        report_lines.append(f"  Std:    {output_stats['final_damaget'].std():.4f}")
        report_lines.append(f"  Min:    {output_stats['final_damaget'].min():.4f}")
        report_lines.append(f"  Max:    {output_stats['final_damaget'].max():.4f}")
        report_lines.append("")
    
    # --------------------------------------------------------
    # Validation Results
    # --------------------------------------------------------
    report_lines.append("-" * 80)
    report_lines.append("4. VALIDATION RESULTS")
    report_lines.append("-" * 80)
    
    validation_text = load_validation_results()
    report_lines.append(validation_text)
    report_lines.append("")
    
    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------
    report_lines.append("-" * 80)
    report_lines.append("5. RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    success_rate = exec_stats['success'] / max(exec_stats['total'], 1)
    
    if success_rate < 0.8:
        report_lines.append("⚠ WARNING: Success rate < 80%")
        report_lines.append("  - Review failed job metadata files")
        report_lines.append("  - Check for common error patterns")
        report_lines.append("  - Consider adjusting solver parameters")
    elif success_rate < 0.95:
        report_lines.append("⚠ CAUTION: Success rate < 95%")
        report_lines.append("  - Some jobs failed, review individual cases")
    else:
        report_lines.append("✓ Excellent success rate (>95%)")
    
    report_lines.append("")
    
    if len(output_stats['max_loads']) < exec_stats['success']:
        missing = exec_stats['success'] - len(output_stats['max_loads'])
        report_lines.append(f"⚠ WARNING: {missing} successful jobs have missing extracted data")
        report_lines.append("  - Re-run extraction script")
        report_lines.append("  - Check ODB file integrity")
    
    report_lines.append("")
    report_lines.append("NEXT STEPS:")
    report_lines.append("  1. Review validation report for data quality issues")
    report_lines.append("  2. Check visualizations in fem_visualizations/")
    report_lines.append("  3. Proceed to PCA reduction (Step 3)")
    report_lines.append("  4. Train surrogate model (Step 4)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


# ============================================================
# MAIN
# ============================================================

def main():
    
    print("=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    # Generate text report
    print("\nGenerating text report...")
    report_text = generate_text_report()
    
    # Save text report
    text_report_path = REPORT_DIR / "fem_summary_report.txt"
    with open(text_report_path, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: {text_report_path}")
    
    # Print to console
    print("\n" + report_text)
    
    print("\n" + "=" * 70)
    print(f"Report saved to: {REPORT_DIR}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
