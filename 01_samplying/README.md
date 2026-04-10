# Uncertainty Quantification - Input Sampling Pipeline

---

## Overview

This directory contains the complete workflow for generating Latin Hypercube Samples (LHS) for uncertainty quantification of a reinforced concrete beam.

---

## Uncertain Parameters

| Parameter | Symbol | Distribution | Mean | Std Dev / CoV | Units |
|-----------|--------|--------------|------|---------------|-------|
| Concrete compressive strength | f_cm | Lognormal | 28.0 | CoV = 0.10 | MPa |
| Bottom concrete cover | c_bot | Normal | 27.0 | σ = 3.0 | mm |
| Top concrete cover | c_top | Normal | 223.0 | σ = 5.0 | mm |

### Derived (Deterministic) Parameters

| Parameter | Symbol | Formula | Units |
|-----------|--------|---------|-------|
| Young's modulus | E | 22000 × (f_cm/10)^0.3 | MPa |

---

# Workflow
---

### Step 1: Generate LHS Samples
```bash
python 01_generate_lhs_samples.py
```

## Sampling Summary

## Overall Method
All uncertain parameters were sampled using **Latin Hypercube Sampling (LHS)**, which provides stratified coverage of each parameter’s distribution.

---

## Parameter Distributions

### 1. Concrete Strength (fcm)
- **Distribution:** Lognormal  
- **Sampling:**  
  - LHS → uniform → standard normal  
  - Lognormal transform: `fcm = exp(mu + sigma * Z)`

### 2. Bottom Concrete Cover (c_nom_bottom)
- **Distribution:** Normal  
- **Sampling:**  
  - LHS → uniform → standard normal  
  - Normal transform: `c_bottom = mean + std * Z`

### 3. Top Concrete Cover (c_nom_top)
- **Distribution:** Normal  
- **Sampling:**  
  - LHS → uniform → standard normal  
  - Normal transform: `c_top = mean + std * Z`

---

## Derived Parameter
### Young’s Modulus (E)
Computed deterministically from sampled `fcm` using EC2:  
`E = 22000 * (fcm / 10) ** 0.3`


---
**Output:** `uq_lhs_samples_training.csv`
---

Generates 400 Latin Hypercube Samples with:
- Sample ID (000-399)
- Fcm_MPa: Concrete strength
- c_nom_bottom_mm: Bottom cover
- c_nom_top_mm: Top cover
- E_MPa: Young's modulus (derived)
- seed: Random seed used

### Step 2: Visualize & Verify Distributions
```bash
python 02_visualize_samples.py
```

**Output:** Directory `sampling_plots/` containing:
- `01_fcm_distribution.png` - Concrete strength PDF/CDF
- `02_c_bot_distribution.png` - Bottom cover PDF/CDF
- `03_c_top_distribution.png` - Top cover PDF/CDF
- `04_qq_plots.png` - Q-Q plots for distribution verification
- `05_correlation_matrix.png` - Correlation heatmap
- `06_scatter_matrix.png` - Pairwise scatter plots
- `07_lhs_coverage.png` - 2D projections of parameter space
- `08_sampling_methods_comparison.png` - LHS vs other methods
- `09_youngs_modulus.png` - E vs f_cm relationship

### Step 3: Data Quality Checks
```bash
python 03_quality_checks.py
```

Performs:
- ✓ Duplicate detection
- ✓ Missing value check
- ✓ Range validation (physical constraints)
- ✓ Statistical outlier detection (3-sigma rule)
- ✓ Young's modulus consistency check
- ✓ Sample coverage uniformity
- ✓ Correlation verification (independence)

### Step 4: Prepare FEM Inputs
```bash
python 04_prepare_fem_inputs.py
```

**Output:** `processed_inputs_final.csv`

Filters samples to match PCA job list and exports minimal format for surrogate model:
- job: Sample identifier (sample_XXX)
- fc: Concrete strength [MPa]
- E: Young's modulus [MPa]
- c_nom_bottom_mm: Bottom cover [mm]
- c_nom_top_mm: Top cover [mm]

---

## Key Design Decisions

### Why Lognormal for Concrete Strength?
- Concrete strength cannot be negative (physical constraint)
- Lognormal distribution ensures f_cm > 0
- Typical distribution used in structural reliability
- CoV = 10% represents good quality control

### Why Normal for Concrete Cover?
- Cover thickness variations follow normal distribution in practice
- Construction tolerances are well-characterized
- Bottom cover: σ = 3mm (easier to control, forms directly on formwork)
- Top cover: σ = 5mm (harder to control during pouring)

### Why Latin Hypercube Sampling?
- **Better space-filling** than Monte Carlo
- **Fewer samples needed** for same accuracy
- **Stratified sampling** ensures all regions of parameter space are explored
- **No correlation** between variables (independence assumption)

### Independence Assumption
- Variables are assumed **statistically independent**
- Concrete strength independent of cover thickness (different construction phases)
- Top and bottom covers may have weak correlation in practice, but neglected for simplicity
- Verified by correlation matrix (|ρ| < 0.1 expected)

---

## File Descriptions

### Input Files
- `uq_lhs_samples_training.csv` - Full LHS dataset (400 samples)
- `meta.json` - PCA job list (from Step 4 of pipeline)

### Output Files
- `processed_inputs_final.csv` - Filtered samples for FEM (198 samples)
- `sampling_plots/*.png` - Verification plots

### Scripts
- `01_generate_lhs_samples.py` - Main sampling script
- `02_visualize_samples.py` - Comprehensive plotting
- `03_quality_checks.py` - Data validation
- `04_prepare_fem_inputs.py` - FEM input preparation

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
```

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn
```

---

## Sampling Configuration

All parameters are defined at the top of `01_generate_lhs_samples.py`:

```python
SEED = 42              # Random seed (reproducibility)
N_SAMPLES = 400        # Number of samples

FCM_MEAN = 28.0        # Concrete strength mean [MPa]
FCM_COV = 0.10         # Coefficient of variation

C_BOT_MEAN = 27.0      # Bottom cover mean [mm]
C_BOT_STD = 3.0        # Bottom cover std dev [mm]

C_TOP_MEAN = 223.0     # Top cover mean [mm]
C_TOP_STD = 5.0        # Top cover std dev [mm]
```

---

## Expected Results

After running all scripts, we should see:

✓ **400 samples** generated with good space-filling properties  
✓ **No duplicates** or missing values  
✓ **Distributions match** theoretical PDF/CDF  
✓ **Q-Q plots** show linearity (distribution fit)  
✓ **Correlation matrix** shows independence (|ρ| < 0.1)  
✓ **198 samples** exported for FEM (matching PCA job list)  

---

## References

1. McKay, M.D., et al. (1979). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." *Technometrics*.

2. EN 1992-1-1:2004 (Eurocode 2). Design of concrete structures.

3. JCSS Probabilistic Model Code (2001). Joint Committee on Structural Safety.

---

## Next Steps (Pipeline Continuation)

Next steps in the uncertainty quantification pipeline:

1. **Run FEM simulations** with generated inputs
2. **Extract outputs** (displacement, stress, etc.)
3. **Perform PCA** on output fields
4. **Train surrogate model** using processed inputs
5. **Uncertainty propagation** analysis

---

## Contact
