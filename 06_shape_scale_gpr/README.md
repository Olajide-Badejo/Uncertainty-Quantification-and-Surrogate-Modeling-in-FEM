# Step 4c: Shape-Scale PCA + GPR Surrogate

**Complete Rewrite - Using DAMAGET (Tension Damage)**

---

## Overview

This workflow implements a **shape-scale hybrid surrogate model** using PCA for dimensionality reduction and Gaussian Process Regression (GPR) for uncertainty quantification.

### Key Features
✅ **DAMAGET (Tension Damage)** - Switched from crack_metric to proper ABAQUS damage variable  
✅ **Comprehensive loss tracking** - Train/val/test metrics with visualizations  
✅ **Learning curves** - Monitoring overfitting and data sufficiency  
✅ **Error analysis** - Distribution plots, input-output correlations  
✅ **Professional plots** - Publication-ready figures  
✅ **Clean code organization** - Numbered scripts, clear naming  

---

## Workflow Scripts

### 1. **`01_data_splitting.py`**
**Purpose:** Split dataset into train/validation/test (70/15/15)

**Features:**
- Reproducible random split (seed=42)
- Validation checks (no overlap, no duplicates, complete coverage)
- Statistical summary of parameter distributions across splits

**Output:**
- `split/train_jobs.txt`
- `split/val_jobs.txt`
- `split/test_jobs.txt`

---

### 2. **`02_pca_preparation.py`**
**Purpose:** Shape-scale decomposition and PCA (training data only)

# PCA Parameters (Shape–Scale PCA)

## 1. Number of PCA Components
- **K_FORCE = 10**  
  Number of PCA components (modes) for **force shape curves**  
  (after normalizing each curve by its max value)

- **K_DAMAGE = 7**  
  Number of PCA components for **damage curves**  
  (damage is already in [0,1], so no scaling)

Both are later adjusted to:
`min(K, n_samples, n_grid)`  
to avoid requesting more components than available data.

---

## 2. Displacement Grids
- **N_FORCE_GRID = 400**  
  Number of interpolation points for force curves

- **N_DAMAGE_GRID = 400**  
  Number of interpolation points for damage curves

These define the dimensionality of the PCA input vectors.

---

## 3. Shape–Scale Decomposition (Force Only)
- **Scale factor:**  
  `scale = max(abs(f_grid))`  
- **Shape:**  
  `shape = f_grid / scale`

PCA is applied **only to the shape**, not the scale.

---

## 4. Damage Variable
- **DAMAGE_VAR = 'DAMAGEC_max'**  
  Determines which damage curve is used for PCA.

---

## 5. Training Strategy
- PCA is **fitted only on training curves**  
- Validation and test curves are transformed using the **training PCA basis**

Prevents data leakage.

---

## 6. Saved PCA Metadata
The script stores:
- `k_force`, `k_damage`
- explained variance ratios
- cumulative variance
- grids (`u_force`, `u_damage`)
- PCA models (`pca_force.joblib`, `pca_damage.joblib`)


**Shape-Scale Strategy:**
- **Force curves:** Normalize by max value → PCA on shapes
- **Damage curves:** Direct PCA (already in [0,1])

**Features:**
- Fits PCA on TRAINING data only
- Transforms val/test using training PCA basis (no data leakage!)
- Saves explained variance ratios

**Output:**
- `output_pca_shapes/pca_force.joblib`
- `output_pca_shapes/pca_damage.joblib`
- `output_pca_shapes/pca_shapes_data_train.npz`
- `output_pca_shapes/pca_shapes_data_val.npz`
- `output_pca_shapes/pca_shapes_data_test.npz`
- `output_pca_shapes/meta.json`

---

### 3. **`03_train_surrogates.py`**
**Purpose:** Train GPR models with comprehensive monitoring

**Surrogates Trained:**
1. **Force Shape** - Multi-output GPR predicting PCA scores
2. **Force Scale** - Single-output GPR predicting scale factor
3. **Damage** - Multi-output GPR predicting damage PCA scores

**NEW Features:**
- ✅ **Train vs Val RMSE plots** - Bar charts comparing performance
- ✅ **Test RMSE plots** - Unbiased final performance
- ✅ **R² scores** - Model quality metrics across all splits
- ✅ **Learning curves** - RMSE vs training set size
- ✅ **Metrics CSV** - All statistics saved

**Output:**
- `output_surrogates/gpr_force_shape.joblib`
- `output_surrogates/gpr_force_scale.joblib`
- `output_surrogates/gpr_damage.joblib`
- `output_surrogates/input_scaler.joblib`
- `output_surrogates/training_metrics.csv`
- `output_surrogates/training_meta.json`
- `output_surrogates/training_plots/` (4 plots)

---

### 4. **`shape_scale_surrogate.py`**
**Purpose:** Wrapper class for loading and using trained models

**Usage:**
```python
from shape_scale_surrogate import ShapeScaleSurrogate

model = ShapeScaleSurrogate.load(base_dir)
u_force, F_pred, u_damage, D_pred = model.predict_curves(fc, E, cbot, ctop)
```

**Methods:**
- `predict_curves()` - Single sample prediction
- `predict_batch()` - Batch prediction
- `get_info()` - Model metadata

---

### 5. **`04_validation_evaluation.py`**
**Purpose:** Comprehensive validation analysis

**Features:**
- ✅ **Error distributions** - Histograms of RMSE, NRMSE, peak errors
- ✅ **Peak force prediction** - Scatter plot vs perfect prediction line
- ✅ **Error vs inputs** - Identify where model struggles
- ✅ **Sample comparisons** - 5 random val samples with FEM vs surrogate
- ✅ **Validation metrics CSV** - All errors per sample

**Output:**
- `output_validation/validation_metrics.csv`
- `output_validation/plots/` (7+ plots)

---

## Added:
### 1. **Data Splitting** ⭐
- Train vs Val RMSE (side-by-side bars)
- Test RMSE (final unbiased performance)
- R² scores (all splits)
- Learning curves (shows if more data would help)

### 2. **Validation Analysis** ⭐
- Error distributions (see spread and outliers)
- Peak prediction quality (most important metric for engineers)
- Error vs input parameters (identify problematic regions)
- Sample-wise comparisons (visual inspection)

### 3. **Code Organization** ⭐
 Numbered scripts, clear naming, single responsibility

---

## Execution Guide

### Full Workflow

```bash
# Step 1: Split data
python 01_data_splitting.py

# Step 2: Prepare PCA (uses DAMAGET now!)
python 02_pca_preparation.py

# Step 3: Train surrogates (creates 4 loss plots)
python 03_train_surrogates.py

# Step 4: Validate (creates 7+ plots)
python 04_validation_evaluation.py
```

---

## Expected Outputs

After running all scripts:

### Files Created
```
shape_scale_pipeline_clean/
├── split/
│   ├── train_jobs.txt
│   ├── val_jobs.txt
│   └── test_jobs.txt
│
├── output_pca_shapes/
│   ├── pca_force.joblib
│   ├── pca_damage.joblib
│   ├── u_force.npy
│   ├── u_damage.npy
│   ├── pca_shapes_data_train.npz
│   ├── pca_shapes_data_val.npz
│   ├── pca_shapes_data_test.npz
│   └── meta.json
│
├── output_surrogates/
│   ├── gpr_force_shape.joblib
│   ├── gpr_force_scale.joblib
│   ├── gpr_damage.joblib
│   ├── input_scaler.joblib
│   ├── training_metrics.csv
│   ├── training_meta.json
│   └── training_plots/
│       ├── 01_train_val_rmse.png
│       ├── 02_test_rmse.png
│       ├── 03_r2_scores.png
│       └── 04_learning_curves.png
│
└── output_validation/
    ├── validation_metrics.csv
    └── plots/
        ├── 01_error_distributions.png
        ├── 02_peak_force_prediction.png
        ├── 03_error_vs_inputs.png
        └── sample_XXX.png (5 samples)
```

### Plots Generated
**Training Plots (4):**
1. Train vs Validation RMSE (bar chart)
2. Test RMSE (bar chart)
3. R² scores across splits (grouped bars)
4. Learning curves (RMSE vs data size)

**Validation Plots (7+):**
1. Error distributions (4 histograms)
2. Peak force prediction quality (scatter)
3. Error vs input parameters (4 subplots)
4. Sample comparisons (5 samples)

---

## Key Configuration Parameters

### `02_pca_preparation.py`
```python
N_FORCE_GRID = 400    # Force curve grid size
N_DAMAGE_GRID = 400   # Damage curve grid size
K_FORCE = 10          # Force PCA modes
K_DAMAGE = 7          # Damage PCA modes
DAMAGE_VAR = 'DAMAGET'  # ⭐ CRITICAL: Tension damage
```

### `03_train_surrogates.py`
```python
KERNEL_TYPE = 'rbf'   # or 'matern'
ALPHA = 1e-4          # Noise level
NORMALIZE_Y = True    # Normalize outputs
```

---

## Completion Status

### ✅ What's Done (100%)

| Task | Status | Script |
|------|--------|--------|
| Data splitting | ✅ Complete | 01 |
| Shape-scale PCA | ✅ Complete | 02 |
| GPR training | ✅ Complete | 03 |
| Loss visualization | ✅ Complete | 03 |
| Learning curves | ✅ Complete | 03 |
| Validation eval | ✅ Complete | 04 |
| Error analysis | ✅ Complete | 04 |
| DAMAGET switch | ✅ Complete | 02, 04 |

### What This Gives:

✅ **Train/val/test performance** - Know your model's true accuracy  
✅ **Learning curves** - See if more data would help  
✅ **Error distributions** - Understand prediction quality  
✅ **Peak prediction** - Most important for structural engineering  
✅ **Input sensitivity** - Know where model is weak  
✅ **Visual validation** - See actual FEM vs surrogate curves  

---
## Performance Expectations

**Typical Results (good surrogate):**
- Force RMSE: 100-500 N (depends on scale)
- Force NRMSE: 0.01-0.05
- Peak error: <5%
- Damage RMSE: 0.01-0.05
- R² scores: >0.95

**If results are poor, then:**
1. Check PCA variance explained (should be >90%)
2. Increase K_FORCE or K_DAMAGE
3. Try different kernel (Matérn vs RBF)
4. Check for outliers in validation metrics CSV

---

## Next Steps

After Step 4c:
1. Use trained surrogate for **uncertainty propagation** in 05_processing
2. **Monte Carlo simulation** with surrogate (fast!)
3. **Sensitivity analysis** (Sobol indices)
4. **Reliability analysis** (FORM/SORM)
5. **Design optimization** using surrogate

---

## References

1. Rasmussen & Williams (2006). "Gaussian Processes for Machine Learning"
2. Jolliffe (2002). "Principal Component Analysis"
3. Sudret (2012). "Meta-models for Structural Reliability and Uncertainty Quantification"

---
