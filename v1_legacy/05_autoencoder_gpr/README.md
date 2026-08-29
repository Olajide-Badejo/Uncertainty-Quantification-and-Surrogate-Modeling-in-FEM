# Autoencoder + GPR Surrogate Model Pipeline

## Overview
This pipeline implements a deep surrogate modeling approach combining autoencoders (AE) with Gaussian Process Regression (GPR) to predict force-displacement and compression damage curves from structural parameters.

## Pipeline Architecture

```
Physical Parameters → GPR → Latent Space → Decoder → Predicted Curves
   [c_bot, c_top, fc]        [Z_force, Z_damage]      [Force, Damage]
```

## Directory Structure

```
autoencoder_gpr/
├── data_preprocessed/          # Preprocessed FEM data
│   ├── F_norm_all.npy         # Normalized force curves
│   ├── C_norm_all.npy         # Normalized damage curves
│   ├── u_force.npy            # Force displacement grid
│   ├── u_crack.npy            # Damage displacement grid
│   ├── train/val/test indices
│   └── normalization factors
│
├── output_autoencoder/         # Trained autoencoders
│   ├── ae_force.pt            # Force autoencoder (12 latent dims)
│   ├── ae_damage.pt           # Damage autoencoder (12 latent dims)
│   ├── training_summary.json  # Training results
│   └── loss plots
│
├── output_surrogates/          # Trained GPR models
│   ├── gpr_force_latent.joblib   # Force GPR models
│   ├── gpr_damage_latent.joblib  # Damage GPR models
│   ├── input_scaler.joblib       # Input standardization
│   ├── gpr_training_summary.json
│   └── loss and R² plots
│
└── output_evaluation/          # Test set evaluation
    ├── test_sample_metrics.csv
    ├── test_aggregate_metrics.json
    └── evaluation plots
```

## Execution Order

### 1. Data Preprocessing (`1_preprocess_data.py`)
**Purpose:** Load FEM results, normalize curves, create train/val/test split

**Outputs:**
- Normalized force and damage curves
- Common displacement grids
- 70/15/15 train/val/test split indices
- Normalization factors for denormalization

**Key Features:**
- Global normalization for force curves (preserves physical variation)
- Per-curve normalization for damage curves
- Compression damage (DAMAGEC) from Abaqus

### 2. Train Autoencoders (`2_train_autoencoders.py`)
**Purpose:** Train separate autoencoders for force and damage curves

**Architecture:**
- Force AE: 200 points → 12 latent dims → 200 points
- Damage AE: 200 points → 12 latent dims → 200 points

**Training:**
- Optimizer: Adam (lr=1e-3)
- Loss: MSE (reported as RMSE)
- Early stopping: 30 epochs patience
- Validation-based model selection

**Outputs:**
- Trained autoencoder models (.pt files)
- Training/validation loss curves
- Test loss evaluation
- Training summary JSON

### 3. Encode Curves (`3_encode_curves.py`)
**Purpose:** Map all curves to latent representations using trained encoders

**Process:**
1. Load trained autoencoders
2. Encode train/val/test curves → latent vectors
3. Load physical parameters from CSV
4. Save latent vectors and inputs for GPR training

**Outputs:**
- Z_force_{train,val,test}.npy - Force latent vectors
- Z_damage_{train,val,test}.npy - Damage latent vectors
- X_{train,val,test}.npy - Physical parameters

### 4. Train GPR Models (`4_train_gpr.py`)
**Purpose:** Train GPR surrogates to map physical parameters to latent space

**Setup:**
- Inputs: [c_nom_bottom, c_nom_top, fc] (standardized)
- Outputs: Latent vectors (one GPR per dimension)
- Kernel: Constant × RBF
- Hyperparameter optimization: Log-marginal likelihood

**Training:**
- Independent GPR for each latent dimension
- 10 random restarts for optimization
- Validation set for monitoring

**Outputs:**
- Trained GPR models (.joblib files)
- Input scaler
- Train/val/test RMSE plots per dimension
- R² scores on test set
- Summary comparison plots
- Training summary JSON

### 5. Comprehensive Evaluation (`5_evaluate_model.py`)
**Purpose:** Complete test set evaluation with all metrics

**Metrics Computed:**
1. **Reconstruction Metrics:**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² score
   - Peak force/damage error
   - Relative peak error

2. **Latent Space Analysis:**
   - Input-latent distance correlation
   - Smoothness metrics
   - Continuity verification

**Outputs:**
- Per-sample metrics CSV
- Aggregate metrics JSON
- Reconstruction comparison plots
- Error distribution plots
- Latent smoothness scatter plots

### 6. Visualization Scripts
**6_visualize_random_samples.py:** Plot random test samples
**7_visualize_all_samples.py:** Generate plots for all test samples

## Usage

### Complete Pipeline Execution
```python
# Step 1: Preprocess data
python 1_preprocess_data.py

# Step 2: Train autoencoders
python 2_train_autoencoders.py

# Step 3: Encode curves to latent space
python 3_encode_curves.py

# Step 4: Train GPR surrogates
python 4_train_gpr.py

# Step 5: Comprehensive evaluation
python 5_evaluate_model.py

# Optional: Visualizations
python 6_visualize_random_samples.py
python 7_visualize_all_samples.py
```

### Using the Trained Surrogate
```python
from ae_surrogate_model import AESurrogateModel

# Initialize model
model = AESurrogateModel("path/to/Scripts")

# Predict curves for new parameters
u_force, F_pred, u_damage, C_pred = model.predict(
    cbot=20.0,  # mm
    ctop=25.0,  # mm
    fcm=35.0    # MPa
)

# F_pred and C_pred are normalized [0,1]
# Denormalize using saved factors if needed
```

## Model Performance Tracking

### Key Files to Check:
1. `output_autoencoder/training_summary.json` - AE performance
2. `output_surrogates/gpr_training_summary.json` - GPR performance
3. `output_evaluation/test_aggregate_metrics.json` - Final test results

### Plots to Review:
1. **Training curves** - Monitor convergence and overfitting
2. **Test loss** - Generalization performance
3. **R² scores** - Predictive quality per latent dimension
4. **Error distributions** - Identify outliers
5. **Latent smoothness** - Verify continuity

## Data Requirements

### Input CSV Files:
- `load_displacement_full_aug.csv` - Force-displacement data
- `crack_evolution_full_aug.csv` - Damage evolution data (DAMAGEC column)
- `processed_inputs_2_aug.csv` - Physical parameters

### Required Columns:
- Force CSV: `job_aug`, `U2`, `RF2`
- Damage CSV: `job_aug`, `U2`, `DAMAGEC_max`
- Input CSV: `c_nom_bottom_mm`, `c_nom_top_mm`, `fc`

## Hyperparameters

### Autoencoders:
- Force latent dim: 12
- Damage latent dim: 12
- Learning rate: 1e-3
- Batch size: 64
- Max epochs: 500
- Early stopping patience: 50

### GPR:
- Kernel: Constant × RBF
- Optimizer restarts: 10
- Alpha (noise): 1e-6
- Normalize outputs: True

## Outputs Summary

### Quantitative Results:
- AE reconstruction RMSE on test set
- GPR prediction RMSE per latent dimension
- Overall R² scores
- Peak prediction errors
- Latent smoothness metrics

### Qualitative Outputs:
- Training/validation/test loss curves
- Sample reconstruction comparisons
- Error distribution histograms
- Latent space smoothness visualizations

## Notes

1. **Train/Val/Test Split:** 70/15/15 with fixed random seed (42)
2. **Normalization:** 
   - Force: Global max (preserves relative magnitudes)
   - Damage: Per-curve max
3. **No Data Leakage:** 
   - Scaler fit only on training data
   - Early stopping uses validation set
   - Test set used only for final evaluation
4. **Reproducibility:** Fixed random seeds throughout
