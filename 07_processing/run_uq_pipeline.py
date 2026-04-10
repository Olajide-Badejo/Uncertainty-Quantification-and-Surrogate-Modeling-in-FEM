"""
===============================================================================
MASTER EXECUTION SCRIPT: STEPS 5-8
===============================================================================
Automated execution of surrogate comparison, validation, UQ, and sensitivity.

Usage:
    python run_uq_pipeline.py [--mode MODE]

Modes:
    all          - Run all four steps sequentially (default)
    compare      - Run only Step 5 (surrogate comparison)
    validate     - Run only Step 6 (FEM validation)
    uq           - Run only Step 7 (uncertainty quantification)
    sensitivity  - Run only Step 8 (sensitivity analysis)
===============================================================================
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{message:^80}")
    print(f"{'='*80}{Colors.ENDC}\n")


def print_success(message):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def run_script(script_path: Path, step_name: str) -> bool:
    """
    Execute a Python script and report status.
    
    Args:
        script_path: Path to Python script
        step_name: Name of the step for reporting
        
    Returns:
        True if successful, False otherwise
    """
    print_header(f"EXECUTING: {step_name}")
    
    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False
    
    print_info(f"Script: {script_path.name}")
    print_info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        print_success(f"{step_name} completed successfully")
        print_info(f"Elapsed time: {elapsed/60:.2f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print_error(f"{step_name} failed with exit code {e.returncode}")
        print_info(f"Elapsed time before failure: {elapsed/60:.2f} minutes")
        return False
        
    except KeyboardInterrupt:
        print_warning(f"\n{step_name} interrupted by user")
        return False
        
    except Exception as e:
        print_error(f"{step_name} failed with exception: {e}")
        return False


def check_prerequisites(base_dir: Path) -> bool:
    """
    Check if prerequisite steps have been completed.
    
    Returns:
        True if all prerequisites met, False otherwise
    """
    print_header("CHECKING PREREQUISITES")
    
    checks = {
        "PCA+GPR surrogate": base_dir / "04_PCA" / "01_pca_reduction" / "models" / "pca_force.joblib",
        "AE+GPR surrogate": base_dir / "05_autoencoder_gpr" / "output_autoencoder_improved" / "ae_force.pt",
        "Shape-Scale surrogate": base_dir / "06_shape_scale_gpr" / "output_pca_shapes" / "pca_force.joblib",
    }
    
    all_good = True
    
    for name, path in checks.items():
        if path.exists():
            print_success(f"{name} found")
        else:
            print_error(f"{name} NOT FOUND: {path}")
            all_good = False
    
    if not all_good:
        print_warning("\nSome surrogates are missing. Please complete training first.")
        print_info("Required steps: 3 (PCA+GPR), 4a (AE+GPR), 4c (Shape-Scale)")
    
    return all_good


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Execute UQ pipeline steps 5-8",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'compare', 'validate', 'uq', 'sensitivity'],
        default='all',
        help='Execution mode (default: all)'
    )
    
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip prerequisite checks'
    )
    
    args = parser.parse_args()
    
    # Get base directory (assuming script is in Scripts_2_0)
    base_dir = Path(__file__).resolve().parents[1]
    
    # Print header
    print_header("UQ PIPELINE EXECUTION")
    print_info(f"Mode: {args.mode}")
    print_info(f"Base directory: {base_dir}")
    print_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not args.skip_check:
        if not check_prerequisites(base_dir):
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print_warning("Execution cancelled")
                return 1
    
    # Define steps
    steps = {
        'compare': {
            'script': base_dir / "07_processing" / "surrogate_comparison_05.py",
            'name': "Step 5: Surrogate Comparison & Selection"
        },
        'validate': {
            'script': base_dir / "07_processing" / "fem_validation_06.py",
            'name': "Step 6: FEM Validation"
        },
        'uq': {
            'script': base_dir / "07_processing" / "uncertainty_quantification_07.py",
            'name': "Step 7: Uncertainty Quantification"
        },
        'sensitivity': {
            'script': base_dir / "07_processing" / "sensitivity_analysis_08.py",
            'name': "Step 8: Sensitivity Analysis"
        }
    }
    
    # Determine which steps to run
    if args.mode == 'all':
        steps_to_run = ['compare', 'validate', 'uq', 'sensitivity']
    else:
        steps_to_run = [args.mode]
    
    # Execute steps
    results = {}
    overall_start = time.time()
    
    for step_key in steps_to_run:
        step = steps[step_key]
        success = run_script(step['script'], step['name'])
        results[step_key] = success
        
        if not success:
            print_warning(f"\n{step['name']} failed!")
            response = input("Continue with next step? (y/n): ")
            if response.lower() != 'y':
                print_warning("Execution stopped")
                break
    
    overall_elapsed = time.time() - overall_start
    
    # Print summary
    print_header("EXECUTION SUMMARY")
    
    for step_key, success in results.items():
        step_name = steps[step_key]['name']
        if success:
            print_success(f"{step_name}")
        else:
            print_error(f"{step_name}")
    
    print(f"\n{Colors.BOLD}Total elapsed time: {overall_elapsed/60:.2f} minutes{Colors.ENDC}")
    
    # Determine exit code
    if all(results.values()):
        print_success("\nAll steps completed successfully! 🎉")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Review comparison results in 06_surrogate_comparison/")
        print("2. Check validation plots in 07_fem_validation/")
        print("3. Analyze UQ results in 08_uncertainty_quantification/")
        print("4. Review sensitivity analysis in 09_sensitivity_analysis/")
        print("5. Read README_STEPS_5_6_7.md for detailed interpretation")
        print("="*80)
        
        return 0
    else:
        print_error("\nSome steps failed. Please check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())