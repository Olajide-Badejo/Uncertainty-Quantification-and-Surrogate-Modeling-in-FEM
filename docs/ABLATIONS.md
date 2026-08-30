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

### Results, measured 2026-08-30

Everything above this line was committed before `scripts/ablation_1_registration.py` existed
and is left exactly as written. Two predictions held and one was wrong.

| Metric | Registered | Unregistered | Prediction | Verdict |
|---|---|---|---|---|
| Components at 99 percent variance | 5 | 15 | unregistered 2 to 3 times higher | **held**, ratio 3.00 |
| \|corr(PC2, d mean/du)\| | 0.117 | 0.111 | unregistered above 0.7 and clearly higher | **refuted** |
| Peak load bias at rank 5 [N] | -60.8 | -228.0 | both negative, unregistered larger | **held**, 3.75 times larger |

**Components: held, at the top of the predicted range.** The unregistered family needs 15
components to reach 99 percent of its variance where the registered family needs 5, a ratio
of exactly 3.00 against a predicted 2 to 3, and the absolute count lands inside the predicted
8 to 18 window. The registered count also sits inside the 3 to 6 that spec 10.2 expected.
This is the clearest of the three results: the phase variation really was consuming about ten
extra components, and separating it out really does buy them back.

**Peak load bias: held, in direction and in the ordering.** Both reconstructions under
predict the peak, as predicted, and the unregistered one is worse by a factor of 3.75: -228.0
N against -60.8 N on a mean peak of about 38.1 kN, so 0.60 percent against 0.16 percent. The
comparison is at matched rank, k = 5, and the registered side is carried back through its own
warps to the displacement grid before its peak is read, so neither side is flattered by being
scored in a space where the phase never had to be reproduced. The caveat I flagged in advance
is the right one to keep: 0.60 percent of peak load is a real bias in the predicted direction,
but it is small in engineering terms, and the argument for registration rests more on the
component count than on this number.

**The derivative mode: refuted, and specifically wrong about where to look.** I predicted the
unregistered PC2 would correlate with the derivative of the mean curve above 0.7 and would
clearly exceed the registered side. Measured, the two are 0.111 and 0.117: both negligible,
and the registered side is fractionally the *higher* of the two. The prediction fails on
magnitude and on direction at once.

Sweeping the correlation across the first six components afterwards, which is a post hoc
diagnostic and not evidence on the same footing as the three committed metrics, shows where
the structure actually is:

| Component | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Unregistered | 0.620 | 0.111 | 0.027 | 0.060 | 0.054 | 0.247 |
| Registered | 0.499 | 0.117 | 0.205 | 0.549 | 0.053 | 0.068 |

The derivative shaped structure sits in **PC1**, not PC2, and the ordering there does run the
way the mechanism argues, 0.620 unregistered against 0.499 registered. So the underlying
physics I described is not obviously wrong; my prediction about which component would carry
it was. On this family PC1 is not a pure amplitude mode with PC2 as the leading correction,
which is the picture I had in mind when I wrote the prediction. PC1 already mixes amplitude
with the leading phase effect, because the curves whose peaks come late are also, through
Fcm, the curves whose peaks are high, so the two effects are correlated in the design rather
than orthogonal.

I am not going to promote the PC1 number to a confirmation. It was not the committed
prediction, the gap is modest, and a metric selected after seeing six of them is not the same
evidence as a metric named in advance. The correct summary is that this metric, as specified,
failed, and that the specification was at fault rather than the mechanism.

**What this ablation supports.** Registration is worth its place in the pipeline on the
component count, which is unambiguous and large, and on the peak bias, which is directionally
right though modest. It is not supported by the spurious mode check as I defined it, and the
report says so rather than quietly substituting the PC1 number. The methodological claim of
spec 7.2 survives on two legs of three, and the third leg was mis specified by me rather than
disproved by the data.

## Ablation 2: autoencoder plus Gaussian process

**Build spec 10.6.2. Prediction committed 2026-08-31, before
`scripts/ablation_2_autoencoder.py` existed.**

### What is being compared

The predecessor project's second surrogate architecture, rebuilt honestly, against the
registered principal component pipeline that ships. Both are measured in the P4 fold harness:
the same ten grouped folds from the same seed, every representation refitted inside the fold,
and the same per curve relative L2 on the physical load displacement curve in newtons against
the displacement grid.

- **Production.** Landmarks, arc length, SRVF registration, principal component analysis on
  the registered amplitude, the phase and the displacement coordinate, damage reduced on its
  own, then one Matern 5/2 automatic relevance determination Gaussian process per retained
  score. Its numbers are read from the `validate` artifact rather than recomputed, because
  they were measured on exactly these folds.
- **Ablation.** A pair of small autoencoders on the raw gridded curves, one for force and one
  for damage, then the same Gaussian process machinery on the latent codes. The damage decoder
  is the salvaged idea of build spec 6.4 item 7: softplus increments passed through a cumulative
  sum, so a decoded damage curve is non decreasing by construction rather than by penalty. The
  terminal renormalization the salvaged file applied after the cumulative sum is deliberately
  not carried over, because dividing every decoded curve by its own last value forces all of
  them to end at 1 and destroys exactly the amplitude information the surrogate exists to
  predict.

Everything the predecessor got wrong about this architecture is corrected rather than
reproduced: group aware folds instead of augmented children on both sides of a split, a fitted
noise model on the latent processes instead of a kernel that interpolates its own scatter, and
the three input feature contract instead of strength and modulus fed in as two features. If
this ablation loses, it loses on the architecture and not on the malpractice.

The network is small on purpose and the size is stated rather than tuned: 178 training curves
in a fold cannot support a wide encoder over 201 points. The force autoencoder carries 8 latent
coordinates, which sits between the 5 amplitude components the production basis retains and the
23 score quantities it actually predicts to rebuild a force curve, so the ablation is not
handicapped on budget. The damage autoencoder carries 4, against the 11 the production damage
basis retains.

### The mechanism I expect to see

An autoencoder learns its basis from data with no prior structure, and at this sample size the
data cannot pay for that freedom. Principal component analysis on a registered family is the
same idea with the basis solved in closed form and the phase variation removed first, so it
spends its degrees of freedom on amplitude alone. The autoencoder has to discover the alignment
that registration hands the production pipeline for free, from 178 curves, through a nonlinear
map fitted by gradient descent, and a nonlinear latent space also breaks the property the
production side leans on hardest: principal component scores are uncorrelated by construction,
which is what makes independent Gaussian processes on them the right model rather than a
convenience. Autoencoder latents are not, so the independent processes on them are a modeling
error the production side does not make.

### Predicted outcome, stated so it can fail

1. **Force curve error.** The autoencoder pipeline's median out of sample relative L2 on the
   load displacement curve is **higher** than the production pipeline's on the same folds.
   Direction: ablation strictly worse.
2. **Peak load.** Out of sample R2 on the peak load read off the decoded curve is **below 0.6**,
   against 0.72 for the production scalar process. This comparison is asymmetric and I am
   stating the asymmetry in advance: production predicts the peak with a dedicated Gaussian
   process on the measured peak, while the autoencoder has no such head and the only peak it
   offers is the maximum of a decoded curve. That is the architecture's own answer, so it is
   the one measured, and the asymmetry is reported beside the number.
3. **Damage curve error.** The monotone decoder's median out of sample relative L2 on the
   damage curve is **higher** than the production damage reduction's. Direction: ablation
   strictly worse.

### How I could be wrong

Prediction 1 is the one I am least sure of, and the reason is a number already on the page.
The P4 baseline table found that at the level of the whole reconstructed curve, all three non
trivial baselines beat the production pipeline: 22.5 to 23.0 percent median relative L2 against
23.1 percent for the Gaussian process. Anything that predicts the curve directly avoids the
compounding the production reconstruction pays for, where an error in a phase or displacement
score moves the whole abscissa. An autoencoder is a direct curve model in exactly that sense,
so it could well land under 23.1 percent while still being the wrong model for every reason
above. If that happens, the honest reading is that the reconstruction path, not the reduction,
is where the production pipeline leaks, and the finding belongs in the report next to the P4
caveat rather than being explained away.

Prediction 2 could fail on the other side of the same asymmetry: a curve level model that gets
the amplitude roughly right can read a decent peak off a smooth decoded curve, and 0.6 is a
line I picked because it sits well below the production process and well above the 0.37 the
predecessor's own artifacts recorded for its latent space fit.

Prediction 3 is the safest of the three, because the damage family is nearly degenerate and 11
linear components describe it to 99 percent of its variance. There is very little for a
nonlinear encoder to find, and a nonlinear encoder with nothing to find is a nonlinear encoder
with extra parameters. If it wins here, the monotone decoder is the reason, and that would be a
result worth carrying into Track B rather than a defeat for the pipeline.

## Ablation 3: five member deep ensemble, direct curve regression

**Build spec 10.6.3. Prediction committed 2026-08-31, before
`scripts/ablation_3_deep_ensemble.py` existed.**

### What is being compared

Five seeded multilayer perceptrons, each mapping the three standardized inputs straight to the
201 point load displacement curve, against the production pipeline on the same ten grouped
folds. Every member has its own seed for both its initialization and its minibatch order, and
every member carries a Gaussian negative log likelihood head, so it predicts a variance per
station as well as a mean. The ensemble prediction is the mixture: the mean of the member
means, and the total variance is the mean of the member variances plus the variance of the
member means, which is the decomposition that makes a deep ensemble a predictive distribution
rather than five point predictions with a spread.

The comparison is against the production surrogate's own out of fold predictive distribution,
recomputed in the same folds from the same code path the shipped model uses, so both sides
carry a mean and a pointwise variance and the same four metrics can be read off each. All four
metrics are out of fold: pointwise root mean square error in newtons, Gaussian negative log
predictive density per station in nats, empirical pointwise 90 percent coverage, and the per
curve relative L2 the rest of this project reports.

### The mechanism I expect to see

Two reasons to expect the ensemble to lose, and they are different in kind. The first is the
sample size. A network that outputs 201 numbers has at least a few hundred parameters in its
last layer alone, and 178 training curves in a fold is not a training set for that, so its
error will be dominated by variance rather than by bias. The five member average suppresses
some of that, which is exactly what deep ensembles are for, but averaging five overfitted
functions gives a smoother overfitted function, not a fitted one.

The second is what the variance means. A Gaussian process posterior variance grows away from
the training design because the kernel says so, which is a statement about where the data is.
A deep ensemble's variance is the disagreement between five optimization runs, which is a
statement about the loss surface. Those coincide only by luck, and the published behaviour is
that ensembles are overconfident in the extrapolation regions where a reliability analysis
spends its time. The softening branch of this family is where the spread is widest, and it is
also where I expect the ensemble's variance to be least honest.

### Predicted outcome, stated so it can fail

1. **Pointwise RMSE.** The ensemble's out of fold pointwise RMSE in newtons is **higher** than
   the production surrogate's on the same folds. Direction: ablation worse.
2. **Negative log predictive density.** The ensemble's mean NLPD per station is **higher**
   (worse) than the production surrogate's. Direction: ablation worse.
3. **Coverage.** The ensemble's empirical pointwise 90 percent coverage is **below 0.90**, and
   **further below nominal** than the production surrogate's uncalibrated pointwise coverage on
   the same folds. Overconfidence, not a nominal miss in either direction: I am predicting the
   sign.
4. **Curve relative L2.** The ensemble's median per curve relative L2 is **higher** than the
   production pipeline's measured 23.09 percent.

### How I could be wrong

Prediction 4 is the weak one, for the reason ablation 2 already records: the P4 table has the
linear, quadratic chaos and nearest neighbour baselines all beating the production pipeline at
curve level, so a model that predicts the curve directly starts with an advantage the
reconstruction path gives away. If the ensemble also lands under 23.1 percent then three
independent direct curve models have now beaten the reconstruction, and the finding is about
the reconstruction rather than about the ensemble.

Prediction 3 could fail in the direction of over coverage rather than under. With five members
trained to convergence on 178 points, the member disagreement can be large, and a Gaussian
negative log likelihood head fitted on a small sample often learns a generous variance because
that is the cheapest way to reduce the loss on the points it cannot fit. If the coverage comes
out above 0.90 the model is not thereby well calibrated, it is differently miscalibrated, and
the NLPD is the number that will say which.

Prediction 2 is the one I would defend hardest. NLPD punishes a confident mistake quadratically
through the exponent and rewards an honest wide interval only logarithmically, so it is the
metric on which a model whose variance does not know where the data is loses most clearly.

## Phase P6 prediction: what the functional sensitivity indices should look like

**Build spec 12.3. Prediction committed 2026-08-30, before `src/ufem/sensitivity.py` existed
and before a single index had been computed.** This is not an ablation, it is a prediction
about a result, and ground rule 12 makes no distinction between the two: the commit that
carries this text is older than the commit that carries the numbers, and `git log` is the
evidence.

### What is being predicted

Build spec 12.3 asks for pointwise first order indices `S_i(u)` and total indices `T_i(u)`
along the response, computed on the registered amplitude curves, and it states the expected
physics in one sentence: covers dominating the service range stiffness, and the concrete
strength taking over near the peak and through softening. That sentence is the thing under
test here, and it is worth being precise about it before the picture exists, because a
stacked band figure is exactly the kind of output a reader will nod at whatever it shows.

### What is already measured, and therefore not a prediction

Three numbers from earlier phases constrain the answer and I am not going to claim credit for
them. The audit measured the peak load correlating with the concrete strength at Pearson 0.80,
with the top cover at 0.24, and with the bottom cover at -0.14; the initial stiffness
correlates with the top cover at 0.50 and much less with the strength. So the endpoints of the
story are known: strength governs the peak, geometry governs the elastic slope. What is not
known, and what this phase produces for the first time, is the whole curve between them, the
crossover location, and how much of the variance is interaction rather than main effect.

### The mechanism I expect to see

Early in the response the member is uncracked or barely cracked and the section is responding
elastically. Its stiffness is set by the second moment of area of the transformed section,
which is a geometry quantity: where the two reinforcement layers sit changes the lever arm
directly, while the concrete modulus enters only through the Eurocode 2 exponent of 0.3 on
strength, so a ten percent strength change buys about a three percent modulus change. The
covers, and the top cover in particular, should therefore dominate the first few tenths of the
curve. As the tension zone cracks and the section works toward its capacity the compressive
strength becomes the binding quantity, both directly through the concrete damaged plasticity
compression card, which is scaled by `fcm/28`, and indirectly through the tension card, which
is scaled by `((fcm-8)/20)^(2/3)`. Through the peak and into softening the strength should
therefore dominate. Bottom cover should be close to irrelevant throughout: it moves a small
lever arm on the layer that is in tension over most of this member's span, and the surrogate
already told us so, with an automatic relevance determination lengthscale for that input
pinned at the top of its allowed range on the peak load process.

### Predicted outcome, stated so it can fail

1. **Crossover exists and runs in the stated direction.** There is a displacement below which
   `S_ctop(u) > S_Fcm(u)` and above which `S_Fcm(u) > S_ctop(u)`, and the crossover happens
   before the peak rather than after it. Direction: top cover leads early, strength leads late.
   This is the sentence of spec 12.3 and it is the one that matters.
2. **The crossover is early.** I expect it in the first quarter of the response, below about
   3 mm of the 20 mm stroke, because the family cracks early: the knee landmark sits at a
   median well under 2 mm. Stated as a number so it can be wrong: crossover displacement
   below 5 mm.
3. **Strength dominance at and after the peak is large, not marginal.** `S_Fcm(u)` exceeds
   0.6 at the peak station and stays above 0.5 through the softening branch.
4. **Bottom cover is negligible everywhere.** `T_cbot(u) < 0.10` at every station.
5. **Interaction is a minority effect on the amplitude block.** Aggregated over the amplitude
   components, the sum of first order indices is above 0.85, that is the interaction share is
   under 0.15. The three inputs enter the material card through separate mechanisms and the
   design is close to orthogonal, so a strongly interacting response would be a surprise.

### How I could be wrong

The first way is that there may be no clean early station to read at all. Force is exactly
zero at the origin for every run because loading is displacement controlled, so the variance
there is zero and the indices are 0/0. The pointwise indices only exist where the family
actually varies, which is the same finding the P5 band domain ran into. If the crossover sits
inside that degenerate span I will not be able to see it, and the honest report of that is
that the measurement cannot resolve the prediction rather than that the prediction held.

The second is the abscissa. These indices are computed on the registered amplitude functions,
which live on an arc length parameter, not on displacement, and the mapping back to a physical
axis is through the mean displacement coordinate. Registration is what makes the indices
measure amplitude rather than phase, and that is the whole reason spec 12.3 insists on it, but
it also means the early part of the registered abscissa is not exactly the early part of any
one physical curve. If the crossover lands near a station where the mean displacement map is
steep, its location is less well determined than three significant figures would suggest and I
will say so.

The third is that the top cover may not separate from the strength early on at all. The
Eurocode 2 modulus does move with strength, and if the elastic branch turns out to be governed
by the modulus more than by the lever arm, prediction 1 fails on the early side while
predictions 3 and 4 still hold. That outcome would say the sentence in spec 12.3 is half right,
and half right is what the report would then say.

The fourth is prediction 5. Correlated inputs are not the issue, since the reparameterization
of spec 9.1 made the three genuinely independent, but the concrete damaged plasticity response
near a peak is a nonlinear function of a strength and a geometry at once, and a large
`T_i - S_i` gap would not be shocking. If the interaction share exceeds 0.15 the finding is
that the response is not additive and the first order indices alone are not a summary of it.

### The gate this prediction does not decide

None of the five is a pass or fail criterion for the phase. The phase gate is build spec 22's:
the Q2 thresholds applied per target, and the polynomial chaos and Gaussian process indices
agreeing within their uncertainties or the discrepancy diagnosed in writing. This section
exists so that the physics claim in the report is a prediction that survived rather than a
description written after looking at the picture.

### Results, measured 2026-08-30

Everything above this line was committed in `0ebc224`, before `src/ufem/sensitivity.py` existed
in `8788ced`, and is left exactly as written.

**The verdict is that this campaign cannot decide the prediction.** All 24 sparse chaos
expansions failed the corrected leave one out Q2 gate of build spec 12.1, so every index the
five predictions would be read from is withheld. What follows is therefore not a verdict on the
predictions; it is a record of which way the withheld numbers ran, kept so that whoever repeats
this on a corrected campaign can see what was expected and what a failed campaign showed.

| Prediction | Predicted | Measured (withheld) | Direction |
|---|---|---|---|
| 1. Crossover exists, top cover leads early | crossover present, before the peak | 0 crossings over 200 usable stations; Fcm leads everywhere | against |
| 2. Crossover below 5 mm | below 5 mm | no crossover to locate | not applicable |
| 3. S_Fcm above 0.6 at the peak, above 0.5 in softening | yes | Fcm at 0.931 at the first usable station and never below 0.583 | with, and then some |
| 4. Bottom cover negligible, T_cbot below 0.10 everywhere | yes | maximum 0.0012 | with |
| 5. Interaction share below 0.15 aggregated | yes | maximum 0.006 pointwise | with |

**Prediction one ran against me, on direction and on location at once.** I expected the top
cover to govern the early response through the lever arm of the transformed section, with the
strength taking over as the tension zone cracked. The withheld decomposition has the strength
holding 0.931 of the amplitude variance at the earliest station the decomposition exists at,
0.03 mm on the mean displacement coordinate, and never yielding the lead: the top cover's
closest approach is 0.167 behind, at 13.7 mm, on the softening branch rather than before the
peak. If that ordering survives on a corrected campaign, the reading is that the elastic branch
of these curves is governed by the Eurocode~2 modulus, which moves with strength, more than by
the reinforcement lever arm, which was the third way I wrote down that I could be wrong.

**Predictions four and five ran with me, and neither is surprising.** The bottom cover is
negligible everywhere by a factor of eighty against the ceiling I named, which is consistent
with the surrogate's own automatic relevance determination pinning that input's length scale at
the top of its allowed range. The response is close to additive, at least in the sparse
expansion's reading of it, and that is the reading whose sparsity the Gaussian process cross
check disputes; see the discussion of the additivity gap in the report.

**What I got wrong about the exercise, rather than about the physics.** The five predictions all
presume the indices exist as statements about the beam. None of them anticipated that the
prior question, whether any smooth metamodel can describe this campaign well enough for a
variance decomposition to mean anything, would come back no. The right prediction to have
committed alongside these would have been a Q2 one, and I did not write it. That is a lesson
about what to predict rather than a lesson about beams, and the P7 predictions will carry a
validity threshold of their own.
