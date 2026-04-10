"""
===============================================================================
STEP 2.1: GENERATE ABAQUS INPUT FILES
===============================================================================
Purpose: Generate .inp files from template with scaled material properties

For each sample:
    1. Load material properties from LHS sampling
    2. Scale compression/tension tables based on fcm
    3. Apply Eurocode formulas for elastic modulus and tensile strength
    4. Fill template with concrete covers and material tables
    5. Export to .inp file

Material scaling approach:
    - Compression: Linear scaling (α = fcm/fcm_base)
    - Tension: Eurocode scaling (α = (fck/fck_base)^(2/3))
    - Damage tables: NOT scaled (dimensionless, strain-based)
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Input files
TEMPLATE_FILE = ROOT / "Lean_model.inp"
CSV_FILE = ROOT / "uq_lhs_samples_training.csv"

# Output directory
OUTPUT_DIR = ROOT / "outputs_inp"

# Reference parameters
FCM_BASE = 28.0      # MPa (base concrete strength for material tables)

# ============================================================
# BASE MATERIAL TABLES (REFERENCE: fcm = 28 MPa)
# ============================================================

# Compression hardening (stress [MPa], inelastic strain [-])
COMP_HARDENING_BASE = np.array([
    [28.00, 0.0000],
    [27.59, 0.0001121],
    [27.19, 0.0002242],
    [26.78, 0.0003364],
    [26.38, 0.0004485],
    [25.97, 0.0005606],
    [25.57, 0.0006727],
    [25.16, 0.0007849],
    [24.75, 0.0008970],
    [24.35, 0.0010091],
    [23.94, 0.0011212],
    [23.54, 0.0012333],
    [23.13, 0.0013455],
    [22.72, 0.0014576],
    [22.32, 0.0015697],
    [21.91, 0.0016818],
    [21.51, 0.0017940],
    [21.10, 0.0019061],
    [20.70, 0.0020182],
    [20.29, 0.0021303],
    [19.88, 0.0022425],
    [19.48, 0.0023546],
    [19.07, 0.0024667],
    [18.67, 0.0025788],
    [18.26, 0.0026909],
    [17.86, 0.0028031],
    [17.45, 0.0029152],
    [17.04, 0.0030273],
    [16.64, 0.0031394],
    [16.23, 0.0032516],
    [15.83, 0.0033637],
    [15.42, 0.0034758],
    [15.01, 0.0035879],
    [14.61, 0.0037000],
    [14.20, 0.0038122],
    [13.80, 0.0039243],
    [13.39, 0.0040364],
    [12.99, 0.0041485],
    [12.58, 0.0042607],
    [12.17, 0.0043728],
    [11.77, 0.0044849],
    [11.36, 0.0045970],
    [10.96, 0.0047091],
    [10.55, 0.0048213],
    [10.14, 0.0049334],
    [9.74,  0.0050455],
    [9.33,  0.0051576],
    [8.93,  0.0052698],
    [8.52,  0.0053819],
    [8.12,  0.0054940],
    [7.71,  0.0056061],
    [7.30,  0.0057182],
    [6.90,  0.0058304],
    [6.49,  0.0059425],
    [6.09,  0.0060546],
    [5.68,  0.0061667],
    [5.28,  0.0062789],
    [4.87,  0.0063910],
    [4.46,  0.0065031],
    [4.06,  0.0066152],
    [3.65,  0.0067274],
    [3.25,  0.0068395],
    [2.84,  0.0069516],
    [2.43,  0.0070637],
    [2.03,  0.0071758],
    [1.62,  0.0072880],
    [1.22,  0.0074001],
    [0.81,  0.0075122],
])

# Tension stiffening (stress [MPa], cracking strain [-])
TENSION_STIFF_BASE = np.array([
    [3.80, 0.0000],
    [3.66, 1.45e-05],
    [3.52, 2.89e-05],
    [3.37, 4.34e-05],
    [3.23, 5.78e-05],
    [3.09, 7.23e-05],
    [2.95, 8.67e-05],
    [2.80, 0.000101],
    [2.66, 0.000116],
    [2.52, 0.000130],
    [2.38, 0.000145],
    [2.23, 0.000159],
    [2.09, 0.000173],
    [1.95, 0.000188],
    [1.81, 0.000202],
    [1.66, 0.000217],
    [1.52, 0.000231],
    [1.38, 0.000246],
    [1.24, 0.000260],
    [1.09, 0.000275],
    [0.95, 0.000289],
    [0.903, 0.000388],
    [0.855, 0.000487],
    [0.808, 0.000586],
    [0.760, 0.000685],
    [0.713, 0.000784],
    [0.665, 0.000883],
    [0.618, 0.000982],
    [0.570, 0.001080],
    [0.523, 0.001180],
    [0.475, 0.001280],
    [0.428, 0.001380],
    [0.380, 0.001480],
    [0.333, 0.001580],
    [0.285, 0.001680],
    [0.238, 0.001770],
    [0.190, 0.001870],
    [0.143, 0.001970],
    [0.095, 0.002070],
    [0.0475, 0.002170],
    [0.0000, 0.002270],
])

# Compression damage (damage variable [0-1], inelastic strain [-])
COMPRESSION_DAMAGE_BASE = np.array([
    [0.,     0.],
    [0.0526, 0.00167],
    [0.105,  0.00208],
    [0.158,  0.00249],
    [0.211,  0.00289],
    [0.263,  0.0033],
    [0.316,  0.00371],
    [0.368,  0.00411],
    [0.421,  0.00452],
    [0.474,  0.00493],
    [0.526,  0.00534],
    [0.579,  0.00574],
    [0.632,  0.00615],
    [0.684,  0.00656],
    [0.737,  0.00696],
    [0.789,  0.00737],
    [0.842,  0.00778],
    [0.895,  0.00819],
    [0.947,  0.00859],
])

# Tension damage (damage variable [0-1], cracking strain [-])
TENSION_DAMAGE_BASE = np.array([
    [0.,     0.],
    [0.0375, 1.45e-05],
    [0.075,  2.89e-05],
    [0.113,  4.34e-05],
    [0.15,   5.78e-05],
    [0.187,  7.23e-05],
    [0.225,  8.67e-05],
    [0.262,  0.000101],
    [0.3,    0.000116],
    [0.337,  0.00013],
    [0.375,  0.000145],
    [0.412,  0.000159],
    [0.45,   0.000173],
    [0.487,  0.000188],
    [0.525,  0.000202],
    [0.562,  0.000217],
    [0.6,    0.000231],
    [0.637,  0.000246],
    [0.675,  0.00026],
    [0.712,  0.000275],
    [0.75,   0.000289],
    [0.763,  0.000388],
    [0.775,  0.000487],
    [0.788,  0.000586],
    [0.8,    0.000685],
    [0.813,  0.000784],
    [0.825,  0.000883],
    [0.838,  0.000982],
    [0.85,   0.00108],
    [0.863,  0.00118],
    [0.875,  0.00128],
    [0.888,  0.00138],
    [0.9,    0.00148],
    [0.913,  0.00158],
    [0.925,  0.00168],
    [0.938,  0.00177],
    [0.95,   0.00187],
    [0.963,  0.00197],
    [0.975,  0.00207],
    [0.988,  0.00217],
])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def scale_stress_only(table, alpha):
    """
    Scale only the stress column (column 0) by factor alpha.
    Strain column (column 1) remains unchanged.
    """
    out = table.copy()
    out[:, 0] *= alpha
    return out


def table_to_string(table):
    """
    Convert numpy array to Abaqus-formatted string.
    Format: "stress, strain" per line with appropriate precision.
    """
    return "\n".join(f"{s:.5f}, {e:.7f}" for s, e in table)


def compute_scaling_factors(fcm, fcm_base=FCM_BASE):
    """
    Compute material scaling factors based on Eurocode relationships.
    
    Parameters:
        fcm: Mean concrete compressive strength [MPa]
        fcm_base: Base strength for reference tables [MPa]
        
    Returns:
        alpha_c: Compression scaling factor
        alpha_t: Tension scaling factor
    """
    # Compression: Linear scaling
    alpha_c = fcm / fcm_base
    
    # Tension: Eurocode fctm ∝ fck^(2/3)
    # fck = fcm - 8 MPa (Eurocode conversion)
    fck_base = fcm_base - 8.0
    fck_new = fcm - 8.0
    alpha_t = (fck_new / fck_base) ** (2/3)
    
    return alpha_c, alpha_t


def validate_template(template_path):
    """
    Verify that template file exists and contains required placeholders.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    template_text = template_path.read_text()
    
    required_placeholders = [
        "{{E_CONCRETE}}",
        "{{c_nom_bottom}}",
        "{{c_nom_top}}",
        "{{COMP_HARDENING_TABLE}}",
        "{{TENSION_STIFFENING_TABLE}}",
        "{{COMPRESSION_DAMAGE_TABLE}}",
        "{{TENSION_DAMAGE_TABLE}}"
    ]
    
    missing = [p for p in required_placeholders if p not in template_text]
    
    if missing:
        raise ValueError(f"Template missing placeholders: {missing}")
    
    return template_text

# ============================================================
# MAIN GENERATION ROUTINE
# ============================================================

def generate_inp_files():
    """
    Generate .inp files for all samples in the LHS dataset.
    """
    
    print("=" * 70)
    print("GENERATING ABAQUS INPUT FILES")
    print("=" * 70)
    
    # Load sample data
    df = pd.read_csv(CSV_FILE)
    print(f"\n✓ Loaded {len(df)} samples from {CSV_FILE}")
    
    # Validate and load template
    template_path = TEMPLATE_FILE
    template = validate_template(template_path)
    print(f"✓ Validated template: {TEMPLATE_FILE}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Statistics tracking
    generated_count = 0
    
    print("\n" + "-" * 70)
    print("GENERATING FILES...")
    print("-" * 70)
    
    # Process each sample
    for idx, row in df.iterrows():
        
        sample_id = int(row["sample_id"])
        fcm = row["Fcm_MPa"]
        c_nom_bottom = row["c_nom_bottom_mm"]
        c_nom_top = row["c_nom_top_mm"]
        
        # --------------------------------------------------------
        # 1. Compute elastic modulus (Eurocode formula)
        # --------------------------------------------------------
        E_concrete = 22000 * (fcm / 10) ** 0.3
        
        # --------------------------------------------------------
        # 2. Compute scaling factors
        # --------------------------------------------------------
        alpha_c, alpha_t = compute_scaling_factors(fcm, FCM_BASE)
        
        # --------------------------------------------------------
        # 3. Scale material tables
        # --------------------------------------------------------
        comp_scaled = scale_stress_only(COMP_HARDENING_BASE, alpha_c)
        tens_scaled = scale_stress_only(TENSION_STIFF_BASE, alpha_t)
        
        # Damage tables are NOT scaled (dimensionless, strain-based)
        comp_damage = COMPRESSION_DAMAGE_BASE.copy()
        tens_damage = TENSION_DAMAGE_BASE.copy()
        
        # --------------------------------------------------------
        # 4. Fill template with parameters
        # --------------------------------------------------------
        inp_text = template
        inp_text = inp_text.replace("{{E_CONCRETE}}", f"{E_concrete:.2f}")
        inp_text = inp_text.replace("{{c_nom_bottom}}", f"{c_nom_bottom:.3f}")
        inp_text = inp_text.replace("{{c_nom_top}}", f"{c_nom_top:.3f}")
        inp_text = inp_text.replace("{{COMP_HARDENING_TABLE}}", table_to_string(comp_scaled))
        inp_text = inp_text.replace("{{TENSION_STIFFENING_TABLE}}", table_to_string(tens_scaled))
        inp_text = inp_text.replace("{{COMPRESSION_DAMAGE_TABLE}}", table_to_string(comp_damage))
        inp_text = inp_text.replace("{{TENSION_DAMAGE_TABLE}}", table_to_string(tens_damage))
        
        # --------------------------------------------------------
        # 5. Write to file
        # --------------------------------------------------------
        out_file = OUTPUT_DIR / f"sample_{sample_id:03d}.inp"
        out_file.write_text(inp_text)
        
        generated_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    print(f"\n✓ Generated {generated_count} .inp files")
    
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total samples:     {len(df)}")
    print(f"Files generated:   {generated_count}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Template used:     {TEMPLATE_FILE}")
    print(f"Base fcm:          {FCM_BASE} MPa")
    print("\n" + "=" * 70)

# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    generate_inp_files()
