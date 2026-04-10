# Uncertainty Quantification & Surrogate Modelling Pipeline  
*A complete workflow for FEM‑based structural response prediction and probabilistic analysis*

---
## Members

- Olajide Badejo  - (Matric No. 108 024 20445 3 )
- Sulaiman Abdul-Hafiz Akanmu - (Matric No. 108 024 25111 7)
---
## Overview

This repository documents a full end‑to‑end pipeline for **uncertainty quantification (UQ)** and **surrogate modelling** of structural response curves derived from **Abaqus finite element simulations**.  
It includes sampling, simulation, postprocessing, surrogate training, validation, and large‑scale UQ.

---

## Pipeline Structure

1. [Input Definition & Sampling](#1-input-definition--sampling)  
2. [Abaqus Simulation](#2-abaqus-simulation)  
3. [Postprocessing & Data Preparation](#3-postprocessing--data-preparation)  
4. [Surrogate Modelling](#4-surrogate-modelling)  
   - PCA + GPR  
   - Autoencoder + GPR  
   - Shape–Scale PCA + GPR  
5. [Surrogate Comparison & Model Selection](#5-surrogate-comparison--model-selection)  
6. [Surrogate Validation Against FEM](#6-surrogate-validation-against-fem)  
7. [Uncertainty Propagation](#7-uncertainty-propagation-uq)  
8. [Sensitivity Analysis (Optional)](#8-sensitivity-analysis-optional)  
9. [Final Outputs & Visualization](#9-final-outputs--visualization)

---

## 1. INPUT DEFINITION & SAMPLING

### Uncertain Input Parameters
- Concrete compressive strength **fcm** (Random)  
- Concrete cover thickness (top, bottom) (Random)  
- Young’s modulus **E** (Deterministic function of fcm)

### Specification
- Probability distributions  
- Parameter bounds  
- Independence / correlation assumptions  

### Sampling Strategy
- **Latin Hypercube Sampling (LHS)** *(chosen)*  
- Monte Carlo Sampling (MCS)  
- Sobol / quasi‑random sampling  

### Dataset Generation
- Training set  
- Validation set  
- Test set  

### Job IDs
- sample_000, sample_001, ...

---

## 2. ABAQUS SIMULATION

### For Each Sampled Input
- Generate `.inp` file  
- Submit job to Abaqus  
- Monitor job status & handle failures  

### Extract FEM Outputs
- Load–displacement curves  
- Damage variables (DAMAGET, DAMAGEC, SDEG)  
- Reaction forces  

### FEM Verification
- Mesh convergence  
- Time increment stability  

### Directory Structure
- /results/job_000/  
- /results/job_001/


---

## 3. POSTPROCESSING & DATA PREPARATION

- Parse Abaqus ODB outputs  
- Interpolate curves onto a common displacement grid  

### Extract Response Quantities
- Force–displacement curves  
- Damage evolution curves  

### Normalization
- Fit scaler on **training set only**  
- Apply to validation and test sets  

### Save Processed Data
- Normalized force matrices  
- Normalized damage matrices  
- PCA‑ready shape matrices  
- Shape–scale decomposition inputs  

---

## 4. SURROGATE MODELLING

Dataset split: **70% train / 15% val / 15% test**

---

## 4a. PCA + GPR (Classical Surrogate)

### PCA on Force Curves
- Fit PCA basis (training only)  
- Select number of modes via energy threshold + validation error  
- Project train/val/test curves → PCA scores  

### Train GPR Models
- One GPR per PCA score  
- Inputs: physical parameters  
- Outputs: PCA scores  
- Kernel: RBF or Matérn  
- Hyperparameters via log‑marginal likelihood  

### Prediction & Reconstruction
- Predict PCA scores  
- Reconstruct curves using PCA basis  

### Evaluation
- RMSE on PCA scores  
- Curve‑wise reconstruction RMSE  
- Test‑set performance reported  

---

## 4b. Autoencoder + GPR (Deep Surrogate)

### Train Autoencoders
- Force AE latent dim: ~8–16  
- Damage AE latent dim: ~4–8  
- Train on training data  
- Validation used for early stopping & tuning  
- Freeze encoder/decoder after training  

### Encode Curves
- Map train/val/test curves → latent vectors  

### Train GPR Models
- One GPR per latent dimension  
- Inputs: physical parameters  
- Outputs: latent variables  

### Workflow Summary
- Curves → Encoder → Latent Z → GPR → Predicted Z → Decoder → Reconstructed curves


### Evaluation
- Reconstruction RMSE  
- Latent‑space smoothness  
- Test‑set generalization metrics  

---

## 4c. Shape–Scale PCA + GPR (Hybrid Surrogate)

### Shape–Scale Decomposition
- Decompose each curve into:
  - **Scale** (norm, peak, integral, etc.)  
  - **Shape** (normalized curve)  
- Fit PCA on shape curves  
- Project curves → shape PCA scores  

### Train Surrogates
#### Shape surrogate
- Multi‑output GPR predicting PCA shape scores  

#### Scale surrogate
- Single‑output GPR predicting scale  

### Prediction & Reconstruction
- Predict shape PCA scores + scale  
- Reconstruct shape curves  
- Reapply scale to recover full curves  

### Evaluation
- RMSE on reconstructed curves  
- Separate metrics for shape & scale  
- Test‑set accuracy reported  

---

## 5. SURROGATE COMPARISON & MODEL SELECTION

### Compare Surrogate Families
- PCA + GPR  
- AE + GPR  
- Shape–Scale PCA + GPR  

### Metrics
- Train/Val/Test RMSE  
- Reconstruction error distribution  
- Computational efficiency  
- Physical interpretability  

---

## 6. SURROGATE VALIDATION AGAINST FEM

### FEM Monte Carlo
Compare FEM vs surrogate:
- Mean curves  
- Confidence envelopes  
- Quantile curves  

Ensures surrogate fidelity before large‑scale UQ.

---

## 7. UNCERTAINTY PROPAGATION (UQ)

Using the selected surrogate, run large‑scale Monte Carlo:

- ≥ 10,000 samples  

### Compute
- Mean response curve  
- 5–95% confidence bands  
- Quantiles at critical displacements  
- Probability of failure / threshold exceedance  

---

## 8. SENSITIVITY ANALYSIS
### Sensitivity Measures
- Sobol indices  
- Gradient‑based sensitivity  
- Surrogate‑based feature importance  

Identify dominant parameters.

---

## 9. FINAL OUTPUTS & VISUALIZATION

- Probabilistic response envelopes  
- Surrogate vs FEM comparison plots  
- PCA basis & mode shapes  
- Autoencoder reconstructions  
- GPR uncertainty estimates  
- Sensitivity analysis plots  

---

## 🏁 End of Pipeline


