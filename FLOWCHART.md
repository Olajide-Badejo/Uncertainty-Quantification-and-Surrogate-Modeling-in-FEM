+------------------------------------------------------+
|                    START                             |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 1. INPUT DEFINITION & SAMPLING                       |
| - Define fcm, cover, E(fcm)                          |
| - Set distributions, bounds                          |
| - Choose sampling (LHS/MCS/Sobol)                    |
| - Generate train/val/test sets                       |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 2. ABAQUS SIMULATION                                 |
| - Generate .inp files                                |
| - Submit jobs, monitor status                        |
| - Extract FEM outputs                                |
| - Mesh/time-step verification                        |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 3. POSTPROCESSING & DATA PREPARATION                 |
| - Parse ODB                                          |
| - Interpolate curves                                 |
| - Extract force/damage curves                        |
| - Normalize (train → val/test)                       |
| - Save processed datasets                            |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 4. SURROGATE MODELLING                               |
|                                                      |
|   +----------------------+   +----------------------+ |
|   | PCA + GPR           |   | Autoencoder + GPR    | |
|   | - PCA on curves     |   | - Train AE           | |
|   | - GPR per PCA mode  |   | - Encode latent Z    | |
|   | - Reconstruct       |   | - GPR per Z dim      | |
|   +----------------------+   +----------------------+ |
|                  \             /                      |
|                   \           /                       |
|                    \         /                        |
|                     v       v                         |
|   +-----------------------------------------------+   |
|   | Shape–Scale PCA + GPR                         |   |
|   | - Decompose shape/scale                       |   |
|   | - PCA on shape                                |   |
|   | - GPR for shape + scale                       |   |
|   +-----------------------------------------------+   |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 5. SURROGATE COMPARISON & SELECTION                 |
| - Compare RMSE, efficiency, interpretability        |
| - Select best surrogate                             |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 6. VALIDATION AGAINST FEM                           |
| - Small FEM Monte Carlo                             |
| - Compare mean, quantiles, envelopes                |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 7. UNCERTAINTY PROPAGATION (UQ)                     |
| - Large-scale Monte Carlo (>= 10,000 samples)       |
| - Compute mean, bands, quantiles, Pf                |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 8. SENSITIVITY ANALYSIS (OPTIONAL)                  |
| - Sobol indices                                     |
| - Gradient-based sensitivity                        |
| - Feature importance                                |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| 9. FINAL OUTPUTS & VISUALIZATION                    |
| - Response envelopes                                |
| - Surrogate vs FEM plots                            |
| - PCA modes, AE reconstructions                     |
| - GPR uncertainty                                   |
| - Sensitivity plots                                 |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|                        END                           |
+------------------------------------------------------+
