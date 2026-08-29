# Quarantine

This directory holds no data. It is the written record of the three things from v1 that I
deliberately refuse to use, so that a future me cannot rediscover them and assume they were
simply overlooked. Nothing listed here is read by any pipeline stage, and a test in
`tests/test_laws.py` asserts that no module under `src/ufem/` references these paths.

## 1. The 570 row augmented dataset

Lives at `v1_legacy/augmentation_physics_fixed/` and stays quarantined in place: I did not
move it here, because moving it would be a change to a frozen tree. It is inadmissible as
evidence for four independent reasons, any one of which would be enough.

Its generator script is not in the repository, so I cannot say what transformation produced
the rows. Its job IDs were renamed to `sample_000` through `sample_569`, which collides
directly with the real FEM job IDs, so a merge silently overwrites real runs with synthetic
ones. It carries no synthetic flag and no parent lineage column, so after any concatenation
the two populations are indistinguishable. And its augmented children were split across
train and test with no group awareness, which is a large part of why v1 reported train
scores near 1.0 next to test scores at or below 0.

The 2.0 pipeline trains on the valid FEM runs only. If augmentation is ever wanted, it gets
written new, with a generator in `src/ufem/`, a `synthetic` flag, a `parent_job` column, and
group aware folds.

## 2. The published v1 metrics

Every headline number in the v1 README, FINAL_REPORT, and PDF descends from a stage that
manufactured its own uncertainty: it floored the GP standard deviation at 0.01, multiplied
the variance by an amplification factor of 1.1, injected 0.5 percent multiplicative noise on
the force, and drew the damage magnitude from a random training sample independently of the
inputs. The consequence is visible in that stage's own output: predicted peak force
correlates with concrete strength at r = -0.006, where the source FEM data correlates at
0.80. The sensitivity stage used a third, different distribution set, and its first order
Sobol indices sum to 0.23 against total indices near 0.88 for three independent inputs,
which is the signature of a noise dominated surface rather than a physical one.

Separately, the surrogate metrics do not survive contact with held out data: PCA+GPR force
score test R2 0.083, AE+GPR force test R2 -0.007 in latent space against a train RMSE of
4e-7, and the shape scale model at train R2 0.99999999 with all three test heads at R2 at or
below 0. The comparison stage then re evaluated on 50 random test jobs with a different
metric and published force R2 0.655 to 0.763, which is the number the README quotes.

None of these numbers may be cited, compared against, or used as a baseline. Section 5 of
`docs/BUILD_SPEC.md` is the full autopsy.

## 3. The hard coded 198 sample literal

The v1 extraction script carried the list of valid samples as an unexplained literal in
source. That is why the censoring was never noticed: with the survivor list frozen into
code, nothing in the pipeline could observe that 202 of 400 runs produced nothing, or that
the failures cluster at low top cover and high concrete strength.

The 2.0 audit stage reclassifies all 400 design rows from the raw data alone and derives the
valid set. The literal is never read, and reintroducing a hard coded sample list is a build
error.
