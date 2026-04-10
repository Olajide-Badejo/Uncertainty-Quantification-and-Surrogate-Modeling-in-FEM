# Uncertainty Quantification for RC Beam FEM using Surrogates

End-to-end pipeline for uncertainty quantification (UQ) of a reinforced-concrete beam response, built on Abaqus FEM data and reduced-order surrogate models.

## What This Project Does
- Generates probabilistic input samples (LHS).
- Runs Abaqus simulations and extracts response curves.
- Builds surrogate models:
  - PCA + GPR
  - Autoencoder + GPR
  - Shape-Scale PCA + GPR
- Compares and validates surrogates against FEM.
- Performs large-scale UQ and sensitivity analysis.
- Produces publication-ready plots and summary outputs.

## Project Layout
- `01_samplying/`: input sampling and quality checks
- `02_abaqus/`: Abaqus job orchestration and extraction
- `03_postprocess/`: post-processing utilities
- `04_PCA/`: PCA-based surrogate workflow
- `05_autoencoder_gpr/`: AE + GPR workflow
- `06_shape_scale_gpr/`: shape-scale surrogate workflow
- `07_processing/`: comparison, validation, UQ, sensitivity, final outputs
- `augmentation_physics_fixed/`: augmented data assets

## Quick Start
1. Create a clean Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. From repository root, run:

```bash
python 07_processing/run_uq_pipeline.py --mode all
```

## Abaqus Notes
- Abaqus-required scripts must be run with Abaqus Python where applicable.
- Set Abaqus command if needed:

```bash
# Windows PowerShell
$env:ABAQUS_CMD="C:\SIMULIA\Commands\abaqus.bat"
```

`02_abaqus/02_run_abaqus_jobs.py` uses `ABAQUS_CMD` from environment and defaults to `abaqus`.

## Reproducibility and Portability
- Hardcoded machine-specific absolute paths were removed.
- Core scripts now use repository-relative paths via `Path(__file__)`.
- Run scripts from the repository root for consistent behavior.

## Typical End-to-End Flow
1. `01_samplying/*`
2. `02_abaqus/*`
3. `03_postprocess/*`
4. `04_PCA/*`, `05_autoencoder_gpr/*`, `06_shape_scale_gpr/*`
5. `07_processing/run_uq_pipeline.py --mode all`

## Authors
- Olajide Badejo
- Sulaiman Abdul-Hafiz Akanmu
