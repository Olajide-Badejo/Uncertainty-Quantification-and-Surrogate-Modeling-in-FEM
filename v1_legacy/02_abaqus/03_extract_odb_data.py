"""
===============================================================================
STEP 2.3: EXTRACT DATA FROM ODB FILES
===============================================================================
Purpose: Extract load-displacement curves and damage variables from .odb files

This script MUST be run with Abaqus Python:
    abaqus python 03_extract_odb_data.py

Extracted data:
    1. Load-displacement curve (reaction force vs displacement)
    2. Damage variables (DAMAGET, DAMAGEC, SDEG) at critical locations
    3. Element-wise damage fields (optional, for visualization)

Output format:
    - CSV files for load-displacement curves
    - CSV files for damage evolution
    - NPZ files for full damage fields (optional)

IMPORTANT: This uses the Abaqus Python API (odbAccess), which is only
available when running through 'abaqus python', NOT regular Python.
===============================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
from odbAccess import openOdb
from abaqusConstants import *

# ============================================================
# ABAQUS PYTHON CHECK
# ============================================================

try:
    from odbAccess import openOdb
    from abaqusConstants import *
except ImportError:
    print("=" * 70)
    print("ERROR: Abaqus Python API not found!")
    print("=" * 70)
    print("This script must be run with:")
    print("    abaqus python 03_extract_odb_data.py")
    print("\nNOT with:")
    print("    python 03_extract_odb_data.py")
    print("=" * 70)
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

# Directories
ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "02_abaqus" / "results"
EXTRACTED_ROOT = ROOT / "02_abaqus" / "extracted_data"

# Extraction control
START_ID = 0
END_ID = 399

# Output file format
SAVE_FULL_FIELDS = False  # If True, saves entire damage field (large files)

# Node/element sets for extraction (adjust based on your model)
LOAD_NODE_SET = "LOAD_POINT"      # Node set where load is applied
SUPPORT_NODE_SET = "SUPPORT"      # Node set for reaction forces
REFERENCE_NODE_SET = "REF_NODE"   # Reference node for displacement

# Create output directory
EXTRACTED_ROOT.mkdir(exist_ok=True)

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_load_displacement(odb, job_name):
    """
    Extract load-displacement curve from ODB.
    
    Parameters:
        odb: Open ODB object
        job_name: Job identifier
        
    Returns:
        displacement: Array of displacement values
        load: Array of reaction force values
    """
    
    # Get the step (assuming single step, adjust if multiple)
    step = odb.steps.values()[0]
    
    # Initialize arrays
    times = []
    displacements = []
    reactions = []
    
    # Loop through frames
    for frame in step.frames:
        
        time_value = frame.frameValue
        
        # --------------------------------------------------------
        # Get displacement at reference point
        # --------------------------------------------------------
        try:
            # Method 1: Using node set
            ref_nodes = odb.rootAssembly.nodeSets[REFERENCE_NODE_SET]
            u_field = frame.fieldOutputs['U']
            u_subset = u_field.getSubset(region=ref_nodes)
            
            if len(u_subset.values) > 0:
                # Assuming displacement in direction 2 (Y-axis for typical beam)
                disp = u_subset.values[0].data[1]  # Index 1 = Y-direction
            else:
                disp = 0.0
                
        except KeyError:
            # Alternative: Get displacement from specific node
            disp = 0.0
            print(f"  Warning: Could not extract displacement for {job_name}")
        
        # --------------------------------------------------------
        # Get reaction force
        # --------------------------------------------------------
        try:
            support_nodes = odb.rootAssembly.nodeSets[SUPPORT_NODE_SET]
            rf_field = frame.fieldOutputs['RF']
            rf_subset = rf_field.getSubset(region=support_nodes)
            
            # Sum reaction forces in Y-direction (typical beam loading)
            total_rf = sum(val.data[1] for val in rf_subset.values)
            
        except KeyError:
            total_rf = 0.0
            print(f"  Warning: Could not extract reaction force for {job_name}")
        
        # Store values
        times.append(time_value)
        displacements.append(abs(disp))
        reactions.append(abs(total_rf))
    
    return np.array(times), np.array(displacements), np.array(reactions)


def extract_damage_variables(odb, job_name):
    """
    Extract damage evolution from ODB.
    
    Parameters:
        odb: Open ODB object
        job_name: Job identifier
        
    Returns:
        Dictionary with damage data:
            - times: Time values
            - damagec_max: Maximum compression damage
            - damaget_max: Maximum tension damage
            - sdeg_max: Maximum stiffness degradation
            - damagec_avg: Average compression damage
            - damaget_avg: Average tension damage
    """
    
    step = odb.steps.values()[0]
    
    data = {
        'times': [],
        'damagec_max': [],
        'damaget_max': [],
        'sdeg_max': [],
        'damagec_avg': [],
        'damaget_avg': []
    }
    
    for frame in step.frames:
        
        time_value = frame.frameValue
        data['times'].append(time_value)
        
        # --------------------------------------------------------
        # Compression damage (DAMAGEC)
        # --------------------------------------------------------
        try:
            damagec_field = frame.fieldOutputs['DAMAGEC']
            values = [v.data for v in damagec_field.values]
            data['damagec_max'].append(max(values) if values else 0.0)
            data['damagec_avg'].append(np.mean(values) if values else 0.0)
        except KeyError:
            data['damagec_max'].append(0.0)
            data['damagec_avg'].append(0.0)
        
        # --------------------------------------------------------
        # Tension damage (DAMAGET)
        # --------------------------------------------------------
        try:
            damaget_field = frame.fieldOutputs['DAMAGET']
            values = [v.data for v in damaget_field.values]
            data['damaget_max'].append(max(values) if values else 0.0)
            data['damaget_avg'].append(np.mean(values) if values else 0.0)
        except KeyError:
            data['damaget_max'].append(0.0)
            data['damaget_avg'].append(0.0)
        
        # --------------------------------------------------------
        # Stiffness degradation (SDEG)
        # --------------------------------------------------------
        try:
            sdeg_field = frame.fieldOutputs['SDEG']
            values = [v.data for v in sdeg_field.values]
            data['sdeg_max'].append(max(values) if values else 0.0)
        except KeyError:
            data['sdeg_max'].append(0.0)
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def save_load_displacement(times, displacements, reactions, output_path):
    """
    Save load-displacement data to CSV.
    """
    with open(output_path, 'w') as f:
        f.write("time,displacement,reaction_force\n")
        for t, d, r in zip(times, displacements, reactions):
            f.write(f"{t:.6e},{d:.6e},{r:.6e}\n")


def save_damage_data(damage_data, output_path):
    """
    Save damage evolution data to CSV.
    """
    with open(output_path, 'w') as f:
        f.write("time,damagec_max,damaget_max,sdeg_max,damagec_avg,damaget_avg\n")
        
        n_points = len(damage_data['times'])
        for i in range(n_points):
            f.write(
                f"{damage_data['times'][i]:.6e},"
                f"{damage_data['damagec_max'][i]:.6e},"
                f"{damage_data['damaget_max'][i]:.6e},"
                f"{damage_data['sdeg_max'][i]:.6e},"
                f"{damage_data['damagec_avg'][i]:.6e},"
                f"{damage_data['damaget_avg'][i]:.6e}\n"
            )


# ============================================================
# MAIN EXTRACTION LOOP
# ============================================================

def main():
    
    print("=" * 70)
    print("EXTRACTING DATA FROM ODB FILES")
    print("=" * 70)
    print(f"Results directory: {RESULTS_ROOT}")
    print(f"Output directory:  {EXTRACTED_ROOT}")
    print(f"Sample range:      {START_ID} to {END_ID}")
    print("=" * 70)
    
    # Find all result directories
    result_dirs = sorted(RESULTS_ROOT.glob("sample_*"))
    
    # Filter by ID range
    jobs_to_process = []
    for result_dir in result_dirs:
        try:
            sample_id = int(result_dir.name.split("_")[1])
            if START_ID <= sample_id <= END_ID:
                odb_file = result_dir / f"{result_dir.name}.odb"
                if odb_file.exists():
                    jobs_to_process.append((sample_id, odb_file, result_dir))
        except (ValueError, IndexError):
            continue
    
    print(f"\nFound {len(jobs_to_process)} ODB files to process\n")
    
    if len(jobs_to_process) == 0:
        print("⚠ No ODB files found in specified range!")
        return
    
    # Statistics
    successful = 0
    failed = 0
    
    # Process each ODB
    for i, (sample_id, odb_path, result_dir) in enumerate(jobs_to_process, 1):
        
        job_name = f"sample_{sample_id:03d}"
        
        print(f"[{i}/{len(jobs_to_process)}] Processing: {job_name}")
        
        try:
            # --------------------------------------------------------
            # Open ODB
            # --------------------------------------------------------
            odb = openOdb(str(odb_path), readOnly=True)
            
            # --------------------------------------------------------
            # Extract load-displacement
            # --------------------------------------------------------
            times, displacements, reactions = extract_load_displacement(odb, job_name)
            
            load_disp_file = EXTRACTED_ROOT / f"{job_name}_load_displacement.csv"
            save_load_displacement(times, displacements, reactions, load_disp_file)
            print(f"  ✓ Load-displacement saved ({len(times)} points)")
            
            # --------------------------------------------------------
            # Extract damage variables
            # --------------------------------------------------------
            damage_data = extract_damage_variables(odb, job_name)
            
            damage_file = EXTRACTED_ROOT / f"{job_name}_damage.csv"
            save_damage_data(damage_data, damage_file)
            print(f"  ✓ Damage data saved")
            
            # --------------------------------------------------------
            # Close ODB
            # --------------------------------------------------------
            odb.close()
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            
            # Log error
            error_file = EXTRACTED_ROOT / f"{job_name}_extraction_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Extraction failed for {job_name}\n")
                f.write(f"Error: {str(e)}\n")
    
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total processed:  {len(jobs_to_process)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"\nExtracted data saved to: {EXTRACTED_ROOT}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
