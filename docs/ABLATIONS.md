# Ablations

Ground rule 12: predict before you measure. Every ablation's predicted direction is
committed before its first number is recorded, and the commit order in `git log` is the
evidence. An ablation whose prediction was written after the result is worthless as
evidence, because a prediction that cannot be wrong is not a prediction.

The rule has teeth only if a contradicted prediction is reported as contradicted. Where the
measurement disagrees with what I wrote here, the results section says so plainly and the
prediction stays on the page above it, unedited.

Each ablation is one script, one manifest, one table row, one report paragraph. None of them
ships in the production path.

## Ablation 1: registration before reduction

**Build spec 10.6.1. Prediction committed 2026-08-30, before the ablation script existed.**

### What is being compared

Two reduction pipelines over the same 198 load displacement curves on the same 201 point
displacement grid, differing in exactly one step:

- **Registered.** Landmarks, arc length reparameterization with the fixed global normalizers
  P0 = 40 kN and u0 = 20 mm, elastic SRVF registration, then PCA on the registered amplitude
  functions.
- **Unregistered.** PCA directly on the curves as they sit on the displacement grid.

Both retain components to 99 percent explained variance, both center the data the same way,
and both are measured by the same three model free reconstruction metrics. No surrogate, no
Gaussian process, no cross validation enters this comparison: the claim under test is about
the representation, so the measurement is about the representation.

### The mechanism I expect to see

The displacement at peak varies by more than a factor of two across this family (the audit
measured a CoV of 0.176 on u_peak). That is phase variation, not amplitude variation. Linear
PCA has no way to express "the same shape, shifted", so it is forced to spend components
approximating a shift by a sum of fixed shapes. The classical signature of that failure is
that the leading correction mode looks like the derivative of the mean curve, because to
first order a small shift of a function is exactly minus the derivative times the shift:
f(u - d) is about f(u) - d f'(u). PCA therefore manufactures a derivative shaped mode that
encodes phase, and it is a spurious mode in the sense that it is an artifact of the
representation rather than a real mode of structural response.

The consequence for a structural surrogate is worse than the component count. Averaging
curves whose peaks sit at different displacements smears the peak: the mean of the family is
flatter and lower at its maximum than any individual curve, and a truncated reconstruction
inherits that. A surrogate that systematically under predicts peak load is the worst
possible artifact for this application, because peak load is the quantity the reliability
analysis thresholds.

### Predicted outcome, stated so it can fail

1. **Components at 99 percent variance.** The unregistered representation needs **2 to 3
   times** as many components as the registered one. Spec 10.2 expects 3 to 6 for registered
   amplitude, so I expect unregistered to land in roughly the 8 to 18 range. Direction:
   unregistered strictly greater.
2. **Spurious derivative shaped mode.** The absolute Pearson correlation between the second
   unregistered PC loading and the numerical derivative of the mean curve,
   |corr(PC2, d mean/du)|, is **high, above 0.7**. The corresponding correlation on the
   registered side is **substantially lower**, because registration has removed the phase
   variation that the derivative mode exists to encode. Direction: unregistered clearly
   greater than registered.
3. **Peak load reconstruction bias.** Truncating the unregistered basis at the registered
   component count and reconstructing gives a mean peak load that is **biased low**, that is
   a negative mean signed error in newtons. The registered pipeline at its own retained rank
   shows a **smaller magnitude** bias. Direction: unregistered bias negative, and larger in
   magnitude than registered.

### How I could be wrong

Three ways, all worth stating in advance rather than discovering and rationalizing later.

The peak bias could come out small in absolute terms even while negative, because 201 points
over 20 mm is a fine grid and the peak region is broad in these curves rather than sharp. A
statistically clean but engineering irrelevant bias would still confirm the direction while
weakening the practical argument, and I would report it that way.

The component count ratio could fall below 2. The family is dominated by an amplitude scaling
that correlates strongly with Fcm (Pearson 0.80 on peak load), and a dominant common mode can
carry a lot of variance regardless of alignment, which would compress the ratio.

The registered side could show a nontrivial derivative correlation of its own. Registration
removes phase from the amplitude functions but does not make the mean curve's derivative
orthogonal to everything, so a moderate correlation on the registered side would not by
itself refute the mechanism. What would refute it is the registered correlation matching or
exceeding the unregistered one.

If the measurement contradicts any of the three, the honest conclusion is that registration
is not doing what spec 7.2 claims for this data, and the report says so rather than keeping
the claim and burying the number.
