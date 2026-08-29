# Project Summary

## Problem
Nonlinear FEM of reinforced-concrete members is expensive for large Monte Carlo studies in reliability/UQ.

## Solution
This repository builds surrogate models on top of Abaqus-generated response curves, enabling fast uncertainty propagation while preserving key structural behavior trends.

## Key Technical Contributions
- Physics-consistent sampling and preprocessing pipeline.
- Multi-surrogate benchmarking:
  - PCA + Gaussian Process Regression (GPR)
  - Autoencoder + GPR
  - Shape-Scale PCA + GPR
- FEM-vs-surrogate validation framework.
- UQ and sensitivity analysis at scale.

## Inputs and Outputs
- Inputs: concrete strength `fc`, concrete covers (`c_nom_bottom_mm`, `c_nom_top_mm`), derived modulus `E`.
- Outputs: load-displacement and damage evolution statistics, confidence bands, exceedance probabilities, sensitivity rankings.

## Value
- Major runtime reduction for probabilistic studies.
- Clear and auditable pipeline from raw simulation to decision metrics.
- Reusable workflow for research and engineering studies involving expensive FE models.
