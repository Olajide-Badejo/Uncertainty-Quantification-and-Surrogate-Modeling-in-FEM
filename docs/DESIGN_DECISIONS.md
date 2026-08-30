# Design decisions

Dated, append only. Every deviation from `docs/BUILD_SPEC.md` is recorded here with its
reason, so that a future me reading the code can tell a deliberate choice from a mistake.

## 2026-08-30, Phase P0

### Version policy

The spec says to pin the newest stable release of every tool, resolved at P0, never a
version copied out of the document. I did that: the resolved matrix at the bottom of this
file is written by `ufem doctor`, not typed, and it is regenerated rather than edited. Every
dependency in `pyproject.toml` is pinned with `==`, not a lower bound, because a
reproducibility claim that floats its dependencies is not a claim.

### Deviation: torch 2.13.0 from PyPI, not the cu130 index

Spec section 0.3 and 3.2 call for `pip install torch==2.13.0 --index-url
https://download.pytorch.org/whl/cu130`, so that the RTX 5070 (sm_120) is usable. I
installed `torch 2.13.0+cpu` from the default PyPI index instead. This is a deliberate
deviation and here is the argument for it.

The production path is CPU by design, not by accident. Spec section 17.2 requires the whole
chain from ingest through propagate to run on NumPy, scikit-learn, and GPyTorch on one CPU
thread with `torch.use_deterministic_algorithms(True)`, precisely so the results are
bitwise reproducible on this machine. Nothing in Track A ever touches the GPU. The only
work that wants CUDA is the neural ablations of spec section 10.6 (the autoencoder and the
deep ensemble), and those are deferred to P9. Installing a 3 GB CUDA wheel at P0 to serve
code that does not exist yet buys nothing and costs disk, install time, and a CI story that
has to explain why the lockfile names a wheel the runners cannot use.

The `pyproject.toml` pin is therefore `torch==2.13.0`, without a local version segment,
which resolves to the CPU wheel on PyPI and to whatever local build is already present in a
venv that was fed the cu130 index. That keeps the CI runners honest (they have no GPU and
must not pretend otherwise, per spec section 18) without blocking the GPU install here.

Revisit this when ablations 2 and 3 of section 10.6 are actually run. At that point the
correct move is to reinstall torch from the cu130 index into this same venv, confirm
`torch.cuda.is_available()`, rerun `ufem doctor` so the matrix below records the change,
and note in the report that the ablation claims are statistically reproducible rather than
bitwise. `ufem doctor` prints the torch build and device on every run, so the deviation
cannot go unnoticed.

### pandas 3.0.5

The spec leaves pandas as "pinned deliberately, 3.0.x is out; pin and record". I pinned
3.0.5. It installs cleanly on CPython 3.14 alongside pyarrow 25.0.1, which matters because
the ingest stage of P1 writes Parquet through pyarrow, and the copy on write semantics that
became mandatory in pandas 3 are what I want anyway: the v1 pipeline had several places
where a chained assignment either silently did nothing or silently mutated a caller's frame.

### License: MIT

Recorded here per spec section 0.3, declared in `pyproject.toml`, and the full text is in
`LICENSE` at the repository root. The inherited v1 tree was already MIT, so there is no
relicensing question over the salvaged assets.

### UI: NiceGUI, with FastAPI as the pre authorized fallback

UFEM Lab (spec section 15) is built on NiceGUI 3.16.0. It gives me a local web dashboard
with sliders bound to Python callbacks without writing a frontend, and it runs on FastAPI
underneath, so if I hit a wall (a plot component that will not update at the 50 ms budget,
or a packaging problem on Windows) I can drop to plain FastAPI plus Plotly and serve the
same artifact store without changing the pipeline at all. I am recording that fallback now
so that taking it later is not a scope change.

### Config hashing

The run identity is the SHA-256 of the canonical JSON of the fully resolved config, both
YAML files together, computed in `ufem.config.config_hash`. Canonical means sorted keys and
no insignificant whitespace, so the digest is stable across two loads and moves the moment
any parameter moves. This is what `manifest.py` stamps into every artifact.

## 2026-08-30, Phase P1

### The spec numbers held exactly, so there is nothing to relax

Build spec section 24 says to stop and record when reality disagrees with the document. On
the two counts P1 asserts, reality agreed to the digit. The strict increasing time filter
removes 165 rows across 26 jobs of the load displacement table, which is exactly what
sections 6.1 and 9.3 pin, so the assertion in `ingest.py` is a live check rather than a
number I had to soften. The damage table needs no deduplication at all: zero jobs, zero
rows, because it was written on the near uniform 199 to 200 point output grid rather than on
the solver's adaptive increments. Displacement control holds at `max |U2 - 20t| = 1.43e-06`
mm against the 1e-3 mm tolerance, and both signals carry the same 198 jobs.

### Deviation: the headline statistics are compared on two different bases, not one

This one is a real disagreement with the task as I received it, and it is the reason this
entry exists.

The golden gate was specified as: recompute the headline statistics from the pipeline's grid
output and match `audit_summary.json` to 1e-9. The first half of that works perfectly. The
`RF2_on_common_U2_grid.npy` matrix reproduces from the grid stage at a maximum absolute
deviation of **0.0** over all 198 by 201 values, which is the strongest result the gate could
have returned. But the headline peak load, displacement at peak and initial stiffness in
`statistics_valid` were never computed on that grid. Reading the committed
`data/audit_reference/audit_script.py`, the audit measured them on the solver's own adaptive
increments, thousands of points per curve, before any interpolation. The 201 point grid is a
resample of those curves, and a resample can only lower a maximum, so the two bases give
genuinely different numbers:

| Quantity | Raw increment basis | 201 point grid basis |
|---|---|---|
| Peak load mean | 38145.407 N | 38125.486 N |
| Displacement at peak mean | 11.0820 mm | 11.0753 mm |
| Initial stiffness mean | 13128.68 N/mm | 14144.58 N/mm |

The stiffness gap is the largest and the most instructive. The window is `0 < u <= 0.1
u_peak`, about 1.1 mm wide. On the raw increments that window holds thousands of points and
the through origin fit is well determined; on the grid it holds two, at 0.1 and 0.2 mm, and
the fit is dominated by the curvature the coarse spacing cannot resolve. Neither number is
wrong. They measure different things, and forcing them to agree to 1e-9 would have meant
either interpolating the audit or degrading the grid.

So the gate compares each quantity against the basis that produced it, and
`tests/test_golden_audit.py` says in its module docstring exactly which fields are compared
and against which basis. `grid.py` grows one function for this, `raw_curve_qoi`, which is the
pipeline's own implementation of the raw increment measurement, so the gate compares
pipeline code against committed values rather than comparing the audit script to itself. On
that footing every compared field matches at **0.0**, not merely within 1e-9: peak load,
displacement at peak and initial stiffness on all five moments each, damage at 10 mm on the
grid basis (10 mm is a grid node, so the bases coincide there), and the residual load at 20
mm against the audit's own gridded `RF2_at_u_max` block.

The QoI table that ships in `qoi.parquet` stays on the grid basis. It is the table the
surrogate will be trained against, and the surrogate predicts curves on that grid, so a QoI
measured somewhere else would be a QoI the model cannot be held to. What P4 must not do is
quote the grid peak next to the audit's 38.15 kN as though they were the same measurement.

### Two fields are deliberately not compared

`damage_final` is not compared because it is the banned QoI of spec section 5.6: it is the
same saturated table cap for all 198 runs and carries zero variance. It is absent from
`qoi.parquet` entirely and a test enforces that.

`damage_U2_at_half_max` is not compared because the two implementations are different
estimators and I chose the second deliberately. The audit thresholded each curve against half
of that curve's own maximum and returned the displacement of the first raw sample at or above
it, which quantizes the answer onto the solver's increments. The pipeline thresholds against
the fixed 0.947 saturation that spec section 9.5 names and interpolates linearly between the
two bracketing grid points. Same intent, better conditioned, and about 0.05 mm different in
the mean. Since it is a different estimator there is no 1e-9 claim to make about it, and
pretending otherwise would be exactly the kind of number laundering this project exists to
prevent.

### The initial stiffness extractor raises where v1 fell back

`audit_script.py` falls back to the first three points when the stiffness window holds fewer
than two, inside a construct that hides the substitution. That is the silent fallback class
of spec section 5.8, so `grid.initial_stiffness` raises instead, naming the window and the
point count. On the real campaign the fallback never fires, which is precisely why leaving it
in would have been free and wrong.

## 2026-08-30, Phase P2

### The completion model is the GPC, and the choice was measured

Build spec 9.4 names a GP classifier as the completion probability model with a regularized
logistic regression as a pre authorized fallback. The Gaussian process classifier shipped.
Here is why, and more importantly here is how the pipeline decides rather than how I did.

`audit.fit_completion_model` does not pick an estimator. It tries the configured primary,
cross validates it, and checks two floors that live in `configs/pipeline.yaml`: a minimum ROC
AUC (0.55) and a minimum spread of the predicted probabilities (0.01). The second floor is
the one that catches the real failure mode of a GPC on a nearly balanced binary outcome,
which is collapsing to a near constant prediction at the base rate that still scores a
respectable AUC. Only if a floor is breached does the logistic fallback run, and the report
and manifest record the attempt, the measurement, and the reason either way. So the estimator
that shipped is a recorded outcome, not a decision buried in a commit message.

The GPC cleared both floors on the first attempt: cross validated AUC 0.7018 against the 0.55
floor, prediction spread 0.65 against the 0.01 floor. The fallback was never taken, and
`completion_model.json` says `"fallback_taken": false` alongside the measurements that made
it so.

**Why prefer the GPC at all, given that a logistic model would have been simpler.** The
failure rate is not monotone in the top cover. It runs 76, 55, 26, 45 percent across the four
quartiles, so the highest quartile fails more than the third. A logistic regression in the
raw features cannot represent that shape at all; it would fit a monotone surface through a
non monotone pattern and report the residual as noise. The Matern kernel represents it
directly, which is visible in the left panel of the completion surface figure as a closed
high failure region rather than a half plane. Since the whole purpose of this model is to
draw a boundary around a specific corner of the input space, the ability to draw a boundary
that is not a hyperplane is the requirement, not a refinement.

**The cost of that choice, stated.** The GPC is about fifty times slower to fit than the
logistic model and dominates the audit stage's 9.5 s wall time. At n = 400 that is
irrelevant. If a Track B campaign pushes the design into the thousands, this is the first
thing that will need revisiting, and the fallback is already wired to take over.

**A fitted lengthscale that saturates, and why it is not a defect.** The fit reports
lengthscales of 1.38, 19.7 and 0.80 on the standardized (Fcm, c_bottom, c_top). The bottom
cover's is an order of magnitude larger than the others and some cross validation folds push
it to the configured upper bound, which makes scikit-learn emit a convergence warning. That
is the kernel stating that the bottom cover carries no signal, which is exactly what the chi
squared test independently says at p = 0.30. I raised the bound from 100 to 1000 to 10000 to
check whether it was a fitting artifact; the cross validated AUC stayed at 0.7018 to four
decimals every time, so it is a flat direction in the marginal likelihood rather than an
optimizer that has not converged. The bound is left at 1000, which is far above the unit
scale of standardized features, and the fitted lengthscales are recorded in the manifest
under `fitted_hyperparameters` so the saturation is a visible measurement rather than
something a reader has to infer from a warning.

### Deviation: the audit stage runs after grid, not before it

The P0 skeleton registered the stage order as ingest, audit, grid, following the order build
spec section 7.1 lists the stages in. P2 moved audit after grid.

The reclassification and the censoring statistics need only the ingest artifacts, so on that
part of the stage the original order was correct. The importance weighting study of spec 9.4
is what forces the move: it reweights the headline QoI statistics, and those are extracted by
the grid stage into `qoi.parquet`. The alternatives were to split audit into two stages that
bracket grid, or to have audit recompute the QoI schedule itself. The first doubles the
artifact directories and the manifests for one dependency; the second is a second
implementation of the QoI extractors, which is the kind of duplication that eventually
disagrees with itself. Reordering one entry in `runner.STAGES` was the smaller change and it
changes nothing about what any stage computes.

`ufem run all` therefore runs ingest, grid, audit. The reason is recorded as a comment on the
`STAGES` table itself, where someone changing the order will actually read it.

### The validity domain is an intersection, not a threshold

Build spec 9.4 defines the validity domain as the region where P(complete) is at least 0.5
*and* the design density is non negligible. It would have been easy to implement only the
first half, since that is the part the completion model provides directly, and the tests
would have passed: the censored corner is rejected on probability alone.

I implemented both, and `tests/test_validity.py` pins the distinction explicitly. One test
asserts that a query far outside the design box is rejected; another asserts that the
censored corner sits *inside* the box and is rejected by the probability, so the two
conditions are demonstrably doing different work. Without that second test, the contract
could silently degrade into a bounding box check the day someone simplified the model, and
every test would still be green.

The design density condition is implemented as the box of the executed design rather than as
a convex hull. At three dimensions with 400 space filling LHS points the two are nearly the
same region, the box is exactly reproducible from six numbers stored in the artifact, and a
hull would need scipy.spatial at query time in the UI. If a later campaign produces a design
with a genuinely non convex or clustered footprint, this is the assumption to revisit, and
`hull_expansion` is already in the config as the knob that would widen it.

### Generated report fragments are committed, the PDF is not

`report/tables/*.tex` is generated by `scripts/make_data_card.py` from the artifact store, and
it is committed anyway. That looks like a contradiction of the rule that generated files stay
out of git, so the reasoning is worth writing down.

The fragments are small text files, they are what makes `report.yml` able to build the PDF on
a runner with no artifact store and no Python stack, and committing them means a reader of the
repository can see every number the report claims without running anything. The staleness gate
in `tests/test_data_card.py` is what keeps that safe: it regenerates every one of those files
and asserts byte identity, so a committed fragment that has drifted from the pipeline fails
the suite rather than quietly misreporting. The PDF is the opposite case, a large binary that
no test can meaningfully diff, so it stays gitignored and is published as a CI artifact
instead. The `.gitignore` carries this reasoning as a comment next to the rule.

Amended 2026-08-30: the figures are now committed on the same reasoning, which this entry
originally got wrong by calling them "the opposite case" and grouping them with the PDF. They
are small vector files (556 KB for the whole set), they are what lets `report.yml` build on a
container with no Python stack, and the style module pins `SOURCE_DATE_EPOCH` so a regenerated
figure is byte identical. Ignoring them while `report.yml` claimed they were committed is what
turned the P2 merge red, and it is recorded in `docs/DEFECT_LOG.md`.

## 2026-08-30, Phase P3

### SRVF ships, and the pre authorized fallback is not needed

Build spec 10.1 pre authorizes landmark piecewise linear registration as the fallback if SRVF
"proves numerically fragile on these curves". It did not, so the shipped path is SRVF and the
fallback stays unused. The evidence, all measured rather than assumed: the stage runs in 12.99
seconds on the 198 by 201 family against a budget of minutes; every one of the 198 warps is
monotone with both endpoint errors exactly 0.0 against a 1e-8 tolerance; and a forced rerun
reproduces all five artifacts bitwise.

The exact call is `fdasrsf.fdawarp(f, t).srsf_align(parallel=False)` with every other argument
at the library default, which is `method='mean'`, `omethod='DP2'`, `center=True`, `MaxItr=20`,
`lam=0.0` and `thresh=0.01`. The single departure from the defaults is `parallel=False`, and
it is there for determinism rather than for correctness: the parallel path splits the family
across workers, so the association order of the Karcher mean would depend on scheduling and
the bitwise reproducibility of spec 17.2 would be gone. At 13 seconds the wall time it costs
is not worth arguing about.

Because the registration is bitwise reproducible, no tolerance gate and no downgrade to
"statistically reproducible" is needed anywhere in this phase. If that ever changes, the brief
and spec 17.2 both require measuring the real reproducibility and recording it here rather
than deleting the assertion; `tests/test_p3_determinism.py` carries that instruction in its
failure message.

### The warp tangent space comes from the library, not from me

Spec 10.1 says to use fdasrsf's utilities for the psi transform and the Karcher mean, and only
to hand roll if the API lacks them. It does not lack them:
`fdasrsf.utility_functions.SqrtMean(gam)` returns the Karcher mean psi, the mean warp, the psi
family and the shooting vectors, and the shooting vectors are exactly the log map image the
phase block needs. So `warp_tangent_vectors` is a thin wrapper that transposes into the
library's orientation and back, and there is no hand written sphere geometry in this
repository to go stale.

### numpy SVD rather than scikit-learn PCA, and a pinned sign convention

Spec 10.2 permits either. I took `numpy.linalg.svd(centered, full_matrices=False)` because the
centering, the truncation rule and the sign convention are then explicit and testable rather
than being library defaults that can move under an upgrade, and because a full deterministic
SVD avoids the randomized solver path entirely.

The sign convention is the part worth recording. An SVD determines its factors only up to a
simultaneous sign flip of a singular vector pair, so without a convention the loadings and the
scores can both flip between platforms or library versions while describing exactly the same
subspace. That would break the bitwise reproducibility gate, and worse, it would silently flip
the sign of any correlation a later stage reports against a score. Every component is
therefore flipped so its largest magnitude entry is positive, and a test asserts it.

### The 85 percent landmark is missing rather than clamped

One curve of the 198 never falls to 85 percent of its peak anywhere on its descending branch.
The landmark for that curve is NaN with a `u85_reached` flag set False, not the stroke end.
Clamping would have been easy and would have looked fine, and it would have put a fictitious
landmark at exactly 20 mm into the column, which is a manufactured number in the sense binding
law 1 and ground rule 8 both forbid. Downstream consumers read the flag and decide for
themselves.

Worth recording alongside it: the naive version of this measurement, testing only whether the
final value sits above 85 percent of the peak, gives 34 curves rather than 1. 33 of those dip
below the level somewhere on the branch and then recover load. The landmark scans the whole
descending branch, so it finds the real crossings, and the two counts are both in the
`register.py` docstring so the discrepancy cannot be rediscovered as a bug later.

### The knee window is 0.2 mm, and the first choice of 0.5 mm was wrong

The cracking knee estimator takes the most negative smoothed second difference over
`0.2 mm < u < 0.8 u_peak`. I first wrote the lower bound as 0.5 mm, reasoning that the first
half millimetre is a straight elastic line whose numerical curvature is interpolation noise.
That reasoning was right about the noise and wrong about this beam: these curves are linear to
four significant digits for their first six grid points and break between 0.6 and 0.8 mm, so
0.5 mm was inside the knee. The symptom was unmissable once looked at, with the curvature
minimum landing exactly on the window edge for 135 of 198 curves, and a test now asserts that
no landmark sits within 0.1 mm of the bound.

The estimator reports the knee 0.1 to 0.2 mm late, consistently, because the second difference
of a smoothed signal peaks just past a break in slope. A consistent sub grid bias is tolerable
for a landmark since it shifts every curve the same way; what would not be tolerable is an
error that wandered in sign between curves, so the test pins the consistency rather than the
offset.

### Damage curves are not registered, and the count is higher than the spec guessed

Spec 10.2 expects the damage family to be very low rank raw and floats the possibility that
two components clear 99 percent. Measured, it needs 11: the leading pair carries 85.1 percent
and 94.4 percent arrives at four components, after which a long tail of small modes runs out
to the target. The stage reports what it measured. The decision not to register them stands
on its own reasoning rather than on the rank: they are monotone saturating curves that all
terminate at the same material table cap, so there is no phase variation to separate.

The phase block needing 63 components is the number most likely to matter at P4. 63 scores
over 198 samples is not a comfortable ratio for independent GPs, and reducing the phase block
harder than 99 percent is the obvious lever if it becomes a problem. That is a P4 decision and
is deliberately not pre empted here.

## 2026-08-30, Phase P4: Gaussian process surrogate, baselines, fold honest validation

### The Karcher mean of the warps was being computed with an undocumented smoothing pass, and it cost accuracy and stability

`ufem.register.warp_tangent_vectors` calls `fdasrsf.utility_functions.SqrtMean(gam)`, and the
library defaults that call to `smooth=True`: a `UnivariateSpline(..., s=1e-4)` fit to each warp
before it is differentiated, with the result clipped at zero. This project's own inverse of
the same map, `ufem.surrogate.srsf_curve`, applies no smoothing at all (it is the exponential
map followed by exact quadrature), so the forward and inverse were not matched pairs, and the
round trip test caught it: `tests/test_surrogate.py::TestSquareRootSlopeRepresentation
::test_the_round_trip_recovers_the_family_it_came_from` crashed with an `IndexError` inside
`fdasrsf`'s own Karcher mean iteration, one past the end of its fixed 500 entry log array,
because the smoothed psi family never converged to the iteration's `1e-8` tolerance.

Measured rather than assumed: 14 seeded synthetic monotone families, round tripped through
`SqrtMean` at both settings. Under the library default, `smooth=True`, 6 of the 14 families
never converged at all (the crash above) and the other 8 gave a round trip error of 1 to 5
percent, ten to a hundred times this representation's discretization floor. Under
`smooth=False`, all 14 converged, with errors of 1.5 to 4.7e-3, consistent across seeds. The
fix is one line: `warp_tangent_vectors` now calls `SqrtMean(gam.T, smooth=False)`. This is a
defect in how the project was calling a library, not a tolerance to widen around a flaky test,
and it is recorded in `docs/DEFECT_LOG.md`.

Consequence, measured rather than assumed: refitting the register and reduce stages under the
fix leaves the amplitude block exactly where P3 measured it (5 components, the ablation's
three metrics reproduce to the digit), because the amplitude family and its warps come from
`srsf_align`, which this fix does not touch. The phase block, which is built from the tangent
vectors this fix does touch, moves slightly: still 63 components at 99 percent, but the
reconstruction error moves from a median of 9.22 percent to 9.98 percent. A worse number, not
a better one, and it ships anyway, because the alternative is a representation that crashes on
a fraction of plausible inputs and quietly loses accuracy on the rest.

### The phase and displacement blocks are capped, and the cost is measured in the manifest

The reduce stage's own 99 percent target asks for 63 phase components and, for the new
displacement block introduced at this phase, 18. Fitting that many independent Gaussian
processes on 198 samples is noise chasing: a component past the leading handful carries well
under one percent of its block's variance, and a GP fitted to it is fitting mostly prior.
`surrogate.phase_max_components` and `surrogate.displacement_max_components` cap both blocks,
in this configuration at 8 and 10, reached by a `phase_variance_target` /
`displacement_variance_target` of 0.90 each. Measured, the fitted ranks carry 77.1 percent and
83.6 percent of their blocks' variance respectively; what the cap leaves behind is not
discarded, it is carried forward as reconstruction residual variance in every predicted curve,
per build spec 10.4, and both numbers are in the surrogate stage's manifest rather than only
in this paragraph.

### The noise hyperprior is centered on what the noise means, not on the solver's resolution

Build spec 10.3 suggests centering the fitted noise's hyperprior on the solver's numerical
resolution. Tried first, centered near `1e-4`: 36 of the 45 targets converged to a lengthscale
pinned at its lower bound and a noise near `1.8e-6`, which is the interpolate the scatter
failure of build spec 5.2, reproduced. The center is now the share of a standardized target's
variance the three inputs are not expected to explain, `noise_prior_median_variance = 0.1`,
which is what a nugget actually is on a censored campaign with two covers and a strength as
the only predictors. The fit then leaves that center by a factor of 25 in both directions
across the 45 targets, so the center is a center and not a floor: ground rule 4 is satisfied
by measurement, and `TestInterpolation::test_a_noiseless_smooth_target_is_fitted_with_a_small
_noise` is the test that would catch a regression back to a floor.

### The lengthscale lower bound is 0.11, not the spec's suggested 0.05

Build spec 10.3 justifies the lower bound by the minimum site spacing of the design, and its
own suggested value of 0.05 is below that spacing on this campaign: the minimum nearest
neighbour distance over the 198 standardized design points is 0.1138, with a fifth percentile
of 0.1565. A lengthscale under the closest pair in the design describes correlation the design
cannot observe, which is exactly the mechanism of the interpolate the scatter failure above.
0.11 is the measured minimum spacing rounded down, and
`TestTheFittedArtifact::test_the_lengthscale_lower_bound_is_not_below_the_design_site_spacing`
asserts the relationship rather than the literal number, so a redesigned campaign keeps the
bound honest automatically.

### The fold harness is 10 grouped folds, not 198 leave one out, and the arithmetic is recorded

Build spec 16.3 requires the registration reference, both principal component bases, and every
standardization statistic recomputed inside every fold. For curves that is expensive: the SRVF
registration alone costs about 13 seconds on the full 198, so a leave one out fold harness
would cost 198 registrations, about 43 minutes, for the surrogate alone and well over 3 hours
once the four baselines go through the same harness. Ten grouped folds cost ten registrations,
and the measured wall time of the fold harness is about 10 minutes end to end (595 seconds of
folds against 0.5 seconds for the closed form scalar leave one out, which needs no refit). The
scalars keep the exact leave one out, because nothing about a scalar target needs the
registration refit; only the curve level comparison takes the grouped compromise, and it is
exactly the compromise build spec 16.3 pre authorizes. The leak test of the same section
applies to `ufem.validate.make_folds` regardless of which harness calls it.

### A measured side effect: `fdasrsf.fdawarp.srsf_align` silently parallelizes past 100 curves

`register.srsf_register` calls `srsf_align(parallel=False)` deliberately, for the determinism
reasoning in its own docstring. The library does not honor that argument unconditionally: it
overrides `parallel` to `True` whenever the family has more than 100 curves or more than 500
time points, regardless of what the caller passed. Every registration in this project's
production path and every non trivial validation fold exceeds 100 curves, so this override is
live throughout. It was not designed around; it is recorded here because it explains the fold
harness's wall clock (a joblib process pool spawns per registration, and a Windows `loky`
worker reimports the scientific stack, which costs real time and memory) and because
determinism survives it only because joblib's `Parallel` returns results in submission order
regardless of completion order: the forced double run of the surrogate stage (build spec 17.2)
reproduces every output byte for byte with this override active, which is the measurement that
matters, not the absence of the override.

### The quadratic chaos baseline has 10 terms, not the 15 build spec 10.5 names

Build spec 10.5 calls the baseline "the full quadratic PCE (15 terms at d = 3, OLS)" and then
gives the wrong count for its own d: the number of multi indices of total degree at most two
in $d$ dimensions is $\binom{d+2}{2}$, which is 15 at $d = 4$ and 10 at $d = 3$. The feature
contract of build spec 9.2 has three inputs, because the elastic modulus is derived rather
than independent (Section 9.1), so the baseline that is actually full quadratic in this
project's three inputs has 10 terms. This is an arithmetic slip in the spec's own parenthetical,
not a reduced baseline: `hermite_multi_indices(3, 2)` enumerates every multi index of total
degree at most two over three variables, which is what "full quadratic" means, and
`tests/test_baselines.py::TestQuadraticChaos::test_the_term_count_is_the_full_quadratic
_expansion_in_three_inputs` pins both the count and the reasoning.

<!-- BEGIN RESOLVED VERSIONS -->

### Resolved version matrix, 2026-08-30

Written by `ufem doctor` on Windows-11-10.0.26200-SP0. Regenerate it, do not edit it.

| Component | Resolved |
|---|---|
| ufem | 1.1.0.dev0 |
| python | 3.14.0 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| pyarrow | 25.0.1 |
| pydantic | 2.13.5 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.1 |
| torch | 2.13.0 |
| gpytorch | 1.15.2 |
| openturns | 1.27.post1 |
| SALib | 1.5.2 |
| fdasrsf | 2.6.10 |
| torch device | torch 2.13.0+cpu, CUDA build None, no visible GPU (CPU path) |
| config SHA-256 | `e8bd99810d8bc145347d236d0a794ac50258431590378262b5fbb142a60ace0f` |

<!-- END RESOLVED VERSIONS -->
