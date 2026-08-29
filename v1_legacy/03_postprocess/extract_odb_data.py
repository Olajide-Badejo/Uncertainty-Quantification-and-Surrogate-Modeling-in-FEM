#!/usr/bin/env python3
"""
STEP 1: ABAQUS ODB DATA EXTRACTION
===================================
Extracts force-displacement and damage evolution data from Abaqus ODB files.
Now uses DAMAGEC (compression damage) instead of PEEQ_h for crack metric.

Outputs:
    - load_displacement_full.csv: Force-displacement curves
    - damage_evolution_full.csv: Compression damage evolution curves
"""

from odbAccess import openOdb
from abaqusConstants import *
import numpy as np
import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "02_abaqus" / "results"
OUTPUT_DIR = ROOT / "03_postprocess" / "01_extracted_data"
OUTPUT_DIR.mkdir(exist_ok=True)

NODE_SET_NAME = "LOADING_POINT"

LD_FILE = OUTPUT_DIR / "load_displacement_full.csv"
DAMAGE_FILE = OUTPUT_DIR / "damage_evolution_full.csv"  # Renamed from crack_evolution

# Sample numbers to process
SAMPLE_NUMBERS = [
    0, 1, 2, 5, 6, 7, 9, 10, 13, 14, 15, 17, 19, 20, 21, 24, 26, 27, 29, 33, 34, 38, 39, 43, 44, 48, 50, 53, 54, 56,
    63, 64, 66, 68, 69, 71, 73, 74, 75, 76, 78, 81, 82, 83, 86, 88, 90, 92, 96, 97, 99, 100, 103, 104, 105, 106,
    110, 115, 116, 118, 120, 121, 123, 124, 125, 126, 127, 139, 140, 141, 144, 151, 158, 160, 164, 165, 174,
    175, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 194, 195, 198, 199, 200, 201, 202, 205, 206,
    212, 216, 215, 220, 222, 223, 224, 226, 229, 230, 231, 232, 233, 235, 238, 239, 240, 242, 251, 252, 253, 254,
    257, 258, 260, 261, 263, 265, 266, 267, 268, 269, 273, 274, 275, 276, 280, 284, 286, 288, 289, 290, 291, 294,
    295, 297, 299, 300, 302, 303, 304, 305, 308, 309, 311, 313, 316, 317, 318, 319, 320, 321, 323, 324, 325, 326,
    327, 328, 335, 337, 338, 339, 341, 342, 347, 350, 351, 352, 354, 355, 356, 358, 359, 361, 363, 367, 368, 369,
    370, 371, 373, 374, 375, 379, 382, 383, 387, 394, 398,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_last_step(odb):
    """Get the last analysis step from ODB."""
    return odb.steps[list(odb.steps.keys())[-1]]


def find_loading_node(assembly, node_set_name):
    """Find the node label and instance for the loading point."""
    target = node_set_name.upper()

    # Assembly-level node sets
    for name, node_set in assembly.nodeSets.items():
        if name.upper() == target:
            node = node_set.nodes[0][0]
            return node.instanceName, node.label

    # Instance-level node sets
    for inst_name, inst in assembly.instances.items():
        for name, node_set in inst.nodeSets.items():
            if name.upper() == target:
                node = node_set.nodes[0]
                return inst_name, node.label

    raise RuntimeError(f"Node set '{node_set_name}' not found in ODB.")


def get_history_u2_rf2(step, inst_name, label):
    """Extract U2 and RF2 history data for a specific node."""
    key = f"Node {inst_name}.{label}"

    if key not in step.historyRegions:
        raise RuntimeError(
            f"History region '{key}' not found. "
            "Check that U2 and RF2 history outputs were requested for this node."
        )

    region = step.historyRegions[key]
    u2_hist = region.historyOutputs["U2"].data
    rf2_hist = region.historyOutputs["RF2"].data

    return u2_hist, rf2_hist


def get_max_field(frame, name):
    """Get maximum value of a field output variable."""
    return max(v.data for v in frame.fieldOutputs[name].values)


def nearest_history_u2(u2_hist, t_frame):
    """Find U2 value at nearest time point to frame time."""
    t_hist, u_hist = zip(*u2_hist)
    t_hist = np.array(t_hist)
    u_hist = np.array(u_hist)

    idx = np.argmin(np.abs(t_hist - t_frame))
    return u_hist[idx]


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():
    """Extract data from all sample ODB files."""
    
    target_jobs = [RESULTS_ROOT / f"sample_{n:03d}" for n in SAMPLE_NUMBERS]
    
    print(f"Starting extraction for {len(target_jobs)} samples...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    with open(LD_FILE, "w", newline="") as f_ld, \
         open(DAMAGE_FILE, "w", newline="") as f_dmg:

        ld_writer = csv.writer(f_ld)
        dmg_writer = csv.writer(f_dmg)

        # Write headers
        ld_writer.writerow(["job", "time", "U2", "RF2"])
        dmg_writer.writerow(["job", "time", "U2", "DAMAGEC_max"])  # Using DAMAGEC instead of crack_metric

        for job_dir in target_jobs:
            if not job_dir.exists():
                print(f"Warning: Directory {job_dir.name} not found. Skipping.")
                continue

            odb_path = job_dir / f"{job_dir.name}.odb"
            if not odb_path.exists():
                print(f"Warning: ODB file {odb_path.name} not found. Skipping.")
                continue

            print(f"Processing {job_dir.name}")

            try:
                odb = openOdb(str(odb_path), readOnly=True)
                step = get_last_step(odb)
                assembly = odb.rootAssembly

                # Find loading node
                inst_name, label = find_loading_node(assembly, NODE_SET_NAME)
                u2_hist, rf2_hist = get_history_u2_rf2(step, inst_name, label)

                # Write full load-displacement history
                for (t_u, u), (t_r, rf) in zip(u2_hist, rf2_hist):
                    ld_writer.writerow([job_dir.name, t_u, -u, -rf])

                # Write damage evolution (using DAMAGEC)
                for frame in step.frames:
                    t = frame.frameValue
                    u2_frame = -nearest_history_u2(u2_hist, t)

                    dmg_writer.writerow([
                        job_dir.name,
                        t,
                        u2_frame,
                        get_max_field(frame, "DAMAGEC"),  # Compression damage
                    ])

                odb.close()
                
            except Exception as e:
                print(f"Error processing {job_dir.name}: {str(e)}")

    print("\n✓ Extraction completed successfully")
    print(f"✓ Saved: {LD_FILE}")
    print(f"✓ Saved: {DAMAGE_FILE}")


if __name__ == "__main__":
    main()
