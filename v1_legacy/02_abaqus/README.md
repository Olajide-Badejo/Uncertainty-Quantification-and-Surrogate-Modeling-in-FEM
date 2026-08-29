# Step 2: ABAQUS Simulation & Data Extraction

**Author:** Elijah  
**Course:** Uncertainty FEM, RUB Third Semester  
**Date:** January 2026

---

## Overview

This directory contains the complete workflow for running ABAQUS simulations, extracting results, and validating data quality for uncertainty quantification analysis.

---

## Workflow Steps

### Step 2.1: Generate Input Files
```bash
python 01_generate_inp_files.py
```

**Purpose:** Create .inp files from template with scaled material properties

**Inputs:**
- `Lean_model.inp` - Template file with placeholders
- `uq_lhs_samples_training.csv` - LHS samples from Step 1

**Outputs:**
- `outputs_inp/sample_XXX.inp` - One .inp file per sample (400 files)

**Material Scaling:**
- **Compression:** Linear scaling α = fcm/fcm_base
- **Tension:** Eurocode scaling α = (fck/fck_base)^(2/3)
- **Damage tables:** NOT scaled (dimensionless, strain-based)
- **Elastic modulus:** E = 22000 × (fcm/10)^0.3

---

### Step 2.2: Run ABAQUS Jobs
```bash
python 02_run_abaqus_jobs.py
```

**Purpose:** Submit jobs to ABAQUS solver with monitoring

**Features:**
- Range-based execution (START_ID to END_ID)
- Automatic error capture and logging
- Timeout protection (1 hour per job)
- Results organized by job name

**Outputs:**
- `abaqus_jobs/sample_XXX/` - Temporary working directories
- `results/sample_XXX/` - Permanent results storage
  - `sample_XXX.odb` - Output database
  - `sample_XXX.dat` - Data file
  - `sample_XXX.msg` - Message file
  - `sample_XXX.sta` - Status file
  - `metadata.txt` - Job information

---

### Step 2.3: Extract Data from ODB Files
```bash
abaqus python 03_extract_odb_data.py
```

**⚠ IMPORTANT:** Must run with `abaqus python`, NOT regular `python`!

**Purpose:** Extract load-displacement curves and damage variables

**Outputs:**
- `extracted_data/sample_XXX_load_displacement.csv`
  - Columns: time, displacement, reaction_force
- `extracted_data/sample_XXX_damage.csv`
  - Columns: time, damagec_max, damaget_max, sdeg_max, damagec_avg, damaget_avg

**Extracted Data:**
1. **Load-displacement curve:** Reaction force vs displacement
2. **Damage evolution:** DAMAGEC, DAMAGET, SDEG over time
3. **Element-wise fields:** Optional full damage fields (large files)

---

### Step 2.4: Validate Results
```bash
python 04_validate_results.py
```

**Purpose:** Quality checks on extracted data

**Validation Checks:**
- ✓ File completeness (all expected files present)
- ✓ Data integrity (no NaN/Inf values)
- ✓ Physical plausibility (forces positive, damage in [0,1])
- ✓ Monotonicity (displacement/damage should not decrease)
- ✓ Outlier detection (unrealistic values)
- ✓ Convergence indicators (plateaus, reversals)

**Outputs:**
- `extracted_data/validation_report.txt` - Detailed validation report
- Console output with statistics

---

### Step 2.5: Visualize Results
```bash
python 05_visualize_results.py
```

**Purpose:** Create comprehensive plots of FEM results

**Generated Plots:**
1. `01_all_load_displacement_curves.png` - Overlay of all curves + mean ± std
2. `02_selected_load_displacement.png` - Grid of individual curves
3. `03_damage_evolution.png` - Damage evolution for selected samples
4. `04_metric_distributions.png` - Histograms of key outputs
5. `05_input_output_correlations.png` - Input parameters vs outputs
6. `06_outlier_detection.png` - Box plots with outlier identification

**Additional Output:**
- `fem_visualizations/summary_metrics.csv` - Tabular summary of all metrics

---

### Step 2.6: Generate Summary Report
```bash
python 06_generate_summary.py
```

**Purpose:** Comprehensive summary of entire simulation campaign

**Report Includes:**
- Execution statistics (success/failure rates, runtimes)
- Data extraction statistics
- Output statistics (means, std deviations, ranges)
- Validation results summary
- Recommendations for next steps

**Output:**
- `fem_reports/fem_summary_report.txt` - Full text report

---

## Directory Structure

```
Step_2_ABAQUS/
├── 01_generate_inp_files.py
├── 02_run_abaqus_jobs.py
├── 03_extract_odb_data.py      # Run with 'abaqus python'
├── 04_validate_results.py
├── 05_visualize_results.py
├── 06_generate_summary.py
├── README.md
│
├── Lean_model.inp              # Template file
├── uq_lhs_samples_training.csv # From Step 1
│
├── outputs_inp/                # Generated .inp files
│   ├── sample_000.inp
│   ├── sample_001.inp
│   └── ...
│
├── abaqus_jobs/                # Temporary working directories
│   ├── sample_000/
│   ├── sample_001/
│   └── ...
│
├── results/                    # Permanent ABAQUS outputs
│   ├── sample_000/
│   │   ├── sample_000.odb
│   │   ├── sample_000.dat
│   │   ├── sample_000.msg
│   │   ├── sample_000.sta
│   │   └── metadata.txt
│   └── ...
│
├── extracted_data/             # Extracted CSV data
│   ├── sample_000_load_displacement.csv
│   ├── sample_000_damage.csv
│   ├── validation_report.txt
│   └── ...
│
├── fem_visualizations/         # Plots
│   ├── 01_all_load_displacement_curves.png
│   ├── 02_selected_load_displacement.png
│   ├── ...
│   └── summary_metrics.csv
│
└── fem_reports/                # Summary reports
    └── fem_summary_report.txt
```

---

## Execution Guide

### Full Workflow (All Samples)

```bash
# 1. Generate .inp files
python 01_generate_inp_files.py

# 2. Run ABAQUS simulations (this takes a long time!)
python 02_run_abaqus_jobs.py

# 3. Extract data from .odb files
abaqus python 03_extract_odb_data.py

# 4. Validate extracted data
python 04_validate_results.py

# 5. Create visualizations
python 05_visualize_results.py

# 6. Generate summary report
python 06_generate_summary.py
```

### Partial Workflow (Selected Range)

Edit the following in each script:
```python
START_ID = 0    # First sample
END_ID = 50     # Last sample (inclusive)
```

Then run scripts as above.

---

## Key Configuration Parameters

### `01_generate_inp_files.py`
```python
TEMPLATE_FILE = "Lean_model.inp"
CSV_FILE = "uq_lhs_samples_training.csv"
OUTPUT_DIR = Path("outputs_inp")
FCM_BASE = 28.0  # Base concrete strength for material tables
```

### `02_run_abaqus_jobs.py`
```python
ABAQUS_CMD = r"C:\SIMULIA\Commands\abaqus.bat"
START_ID = 0
END_ID = 399
CPUS = 1  # CPUs per job
```

### `03_extract_odb_data.py`
```python
LOAD_NODE_SET = "LOAD_POINT"      # Adjust to your model
SUPPORT_NODE_SET = "SUPPORT"
REFERENCE_NODE_SET = "REF_NODE"
SAVE_FULL_FIELDS = False  # Set True for full damage fields (large!)
```

---

## Template File Requirements

The `Lean_model.inp` template must contain these placeholders:

```
{{E_CONCRETE}}                     # Young's modulus
{{c_nom_bottom}}                   # Bottom cover thickness
{{c_nom_top}}                      # Top cover thickness
{{COMP_HARDENING_TABLE}}           # Compression stress-strain
{{TENSION_STIFFENING_TABLE}}       # Tension stress-strain
{{COMPRESSION_DAMAGE_TABLE}}       # Compression damage evolution
{{TENSION_DAMAGE_TABLE}}           # Tension damage evolution
```

---

## Troubleshooting

### Issue: "Abaqus Python API not found"
**Solution:** You must run extraction script with:
```bash
abaqus python 03_extract_odb_data.py
```
NOT with:
```bash
python 03_extract_odb_data.py
```

### Issue: Jobs failing with solver errors
**Solution:** 
1. Check .msg files in results directories
2. Review validation report for patterns
3. Adjust solver parameters in template
4. Consider mesh refinement if convergence issues

### Issue: Missing node/element sets in extraction
**Solution:**
1. Open a sample .odb file in Abaqus CAE
2. Identify actual node set names
3. Update `LOAD_NODE_SET`, `SUPPORT_NODE_SET`, etc. in script

### Issue: Extraction fails for some samples
**Solution:**
1. Check if .odb files exist and are not corrupted
2. Review error logs in `extracted_data/sample_XXX_extraction_error.txt`
3. Manually inspect problematic .odb files in Abaqus CAE

---

## Performance Notes

- **Total runtime:** ~100-200 hours for 400 samples (depends on model complexity)
- **Recommended approach:** Run in batches overnight
- **Disk space:** ~2-5 GB per 100 samples (including .odb files)
- **Parallelization:** Can run multiple instances of script with different ID ranges

---

## Expected Outputs

After completing all steps, you should have:

✅ **400 .inp files** generated with unique material properties  
✅ **>95% success rate** in ABAQUS executions  
✅ **400 .odb files** with FEM results  
✅ **400 load-displacement curves** extracted to CSV  
✅ **400 damage evolution curves** extracted to CSV  
✅ **Validation report** confirming data quality  
✅ **6 visualization plots** showing result distributions and trends  
✅ **Summary report** with statistics and recommendations  

---

## Next Steps (Pipeline Continuation)

After completing Step 2:

1. **Perform PCA** on displacement/damage fields (Step 3)
2. **Train surrogate model** using extracted data (Step 4)
3. **Propagate uncertainty** through surrogate (Step 5)
4. **Compute reliability indices** (Step 6)

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Template file not found" | Wrong path to Lean_model.inp | Check TEMPLATE_FILE path |
| "Abaqus command failed" | ABAQUS_CMD incorrect | Update to your Abaqus installation |
| "Node set not found" | Model-specific node sets | Update LOAD_NODE_SET, etc. |
| "Permission denied" | Output directories locked | Close Abaqus CAE, delete lock files |
| "Timeout" | Job takes >1 hour | Increase timeout in script or simplify model |

---

## References

1. ABAQUS User Manual (2023). Dassault Systèmes.
2. EN 1992-1-1:2004. Eurocode 2: Design of concrete structures.
3. ABAQUS Scripting Reference Guide (odbAccess module).

---
