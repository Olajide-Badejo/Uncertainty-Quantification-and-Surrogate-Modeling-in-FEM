# Engineering log

Dated entries, append only, never deleted. The discussion section of the report is built
from this file, so it records what actually happened including the parts that did not work.

## 2026-08-30, Phase P0: foundations

Built the skeleton the rest of the project hangs off. Nothing here computes a result; the
whole point of P0 is that by the time a result exists, the machinery to make it traceable is
already in place and already tested.

**The restructure.** The entire v1 tree moved to `v1_legacy/` in one commit, frozen at
release v1.0.0 and read only. I prepended a warning to its README rather than replacing it,
because the original text is evidence: it is the document that quotes force R2 0.763 and a
10.16 percent failure probability, and it is worth keeping visible next to the reason those
numbers cannot be used. The build specification moved to `docs/BUILD_SPEC.md`, byte
identical at 73344 bytes, and is exempt from the dash lint since it is the input document.
The pre build audit output moved from a loose `data_audit/` to `data/audit_reference/` and
is now committed at 576 KB: six files that P1 will gate against to 1e-9.

**Config.** Both YAML files are validated by Pydantic v2 models that are frozen and reject
unknown keys. Rejecting extras matters more than it looks: a silently ignored misspelled key
is how a config drifts away from what the pipeline actually ran. The geometry validator
rejects a cover mean outside the 250 mm section depth, a non positive sigma or CoV, and a
bottom cover sitting above the top cover. `feature_order` must equal the tuple pinned in
`config.py`, so reordering the design matrix is a deliberate code change rather than a
config edit; that is the direct fix for v1 feeding three stages three different feature sets
in three different orders.

The lognormal is parameterized from the declared mean and CoV rather than from log space
parameters, so the YAML says what a structural engineer means by it: `sigma_ln =
sqrt(ln(1 + CoV^2))`, `mu_ln = ln(mean) - sigma_ln^2 / 2`. The test asserts the frozen
scipy object comes back with mean exactly 28.0 and CoV 0.10 to 1e-9, which it does.

**Manifests.** About 160 lines including docstrings, no external service. Content addressed:
`cache_key` hashes the stage name, the stage's own source file, the config hash, and the
input hashes. Hashing the source file is the part I would have skipped if I were not
writing down why: a cache keyed only on data serves last week's results after an algorithm
change, and that is a bug I would rather pay a few milliseconds per run to never have.

**The runner.** `ufem run <stage>|all` and `ufem doctor`. All eleven stages are registered
and imported lazily; asking for one that does not exist yet raises `NotImplementedError`
naming the phase that will add it. There is no `input()` anywhere in the file and a test
parses the AST to prove it, because grepping for the string fails on the docstring that
explains the ban.

**What the gates caught on their first run.** Three things, which is the argument for
writing them before the code they check rather than after.

1. `check_file_sizes.py` found two tracked files over 5 MB, both inside the frozen v1 tree:
   `load_displacement_full_aug.csv` at 67.8 MB and `crack_evolution_full_aug.csv` at 14.5 MB.
   These are the quarantined 570 row augmented dataset. They came across with the git move
   because v1 tracked them. I untracked both (they stay on disk inside `v1_legacy/`) and
   added the ignore rules. This is exactly the case the 5 MB rule exists for, and it fired
   on the first commit it could have.
2. `dash_lint.py` found two en dashes in `report/figures_src/make_figures.py`, at lines 222
   and 362, both range separators in plot labels. Replaced with the word "to", which is
   what they meant.
3. `dash_lint.py` initially reported itself, because its own `EM_DASH` and `EN_DASH`
   constants held the literal characters. They are now built with `chr(0x2014)` and
   `chr(0x2013)`, so the file that bans those characters contains none of them.

**Deviation from the spec.** Torch is installed from PyPI as 2.13.0+cpu rather than from the
cu130 index. The reasoning, and the conditions under which I revisit it, are in
`docs/DESIGN_DECISIONS.md`. Short version: the production path is CPU by design per spec
17.2, the only GPU work is the neural ablations of 10.6, and those are deferred to P9.

**Measured, not estimated.** Full suite: 49 tests, 6.3 s. `ufem doctor`: under a second.
Both lint scripts and `ruff check src tests scripts`: clean, under a second each. The spec
estimates P0 at 2 to 3 sessions; it took one.

**Still open at the end of P0.** The proof of failure CI gate of spec section 18.1 requires
four deliberately broken commits producing four red runs with their URLs recorded here. That
needs a GitHub remote and a push, and this phase is under a standing instruction not to
push, so it is deferred to the first session that has one. The workflow itself is written
and its three jobs are defined. The `report.yml` LaTeX workflow is not written yet either;
it belongs with the first compiled PDF at P2.

## 2026-08-30, Phase P1: ingest and grid

The first phase that touches real data. Two stages, one golden gate, and a result I did not
expect to be as clean as it turned out to be.

**What was measured, against what the spec pinned.** Both dedup numbers held exactly. The
strict increasing time filter removes 165 rows across 26 jobs of the load displacement
table, which is what build spec sections 6.1 and 9.3 name to the digit, so the assertion in
`ingest.py` is a live check and not a number I had to soften. The damage table needs no
deduplication at all, zero jobs and zero rows, because it was written on the near uniform
199 to 200 point output grid rather than on the solver's adaptive increments; the spec talks
about 26 jobs without saying which table, and now the manifest records both separately.
Displacement control holds at `max |U2 - 20t| = 1.4305e-06` mm against a 1e-3 mm tolerance,
on both signals. 198 unique jobs in each, identical job sets, every job id present in the
400 row design. The design's largest absolute cross correlation among the three independent
inputs is 0.0434, inside the 0.05 bound.

| Measurement | Value |
|---|---|
| Load displacement rows in | 1,869,676 |
| Load displacement rows dropped | 165 over 26 jobs |
| Damage rows in | 39,569 |
| Damage rows dropped | 0 over 0 jobs |
| Jobs, both signals | 198, identical sets |
| max abs(U2 - 20t) | 1.4305e-06 mm |
| Design cross correlation, worst | 0.0434 |
| Ingest wall time | 3.08 s |
| Grid wall time | 0.21 s |

**The golden gate.** `RF2_on_common_U2_grid.npy` regenerates from the grid stage at a
maximum absolute deviation of **0.0** across all 198 by 201 values, in the job order of
`common_grid_sample_ids.csv`, which is the strongest answer the gate could return. Not
within 1e-9: identical. The same holds for the common grid abscissa and for the audit's
`RF2_at_u_max` block, all four moments.

**Where the specified gate was wrong, and what I did instead.** The headline peak load,
displacement at peak and initial stiffness in `audit_summary.json` do not reproduce from the
201 point grid, and they should not. Reading `data/audit_reference/audit_script.py`, the
audit measured them on the solver's raw adaptive increments, thousands of points per curve,
before any interpolation. A resample can only lower a maximum, so the grid peak sits about
20 N low, and the initial stiffness is off by a kilonewton per millimetre because its window
is 1.1 mm wide and holds two grid points where it held thousands of raw ones. I recorded the
discrepancy in `docs/DESIGN_DECISIONS.md` under spec section 24 rather than loosening a
tolerance until it passed, added `grid.raw_curve_qoi` so the gate compares pipeline code
against the committed values instead of comparing the audit script to itself, and gated each
quantity on the basis that produced it. Every compared field then matches at 0.0 as well. The
module docstring of `tests/test_golden_audit.py` names each field compared and each field
deliberately not, which is the part that stops this from being a tolerance I quietly widened.

**Two things I refused to carry over from the audit script.** Its `jobid_to_int` returns -1
from inside a bare except on a malformed label, which silently merges every unparseable row
onto one key; `ingest.job_to_sample_id` raises and names the offending labels. Its initial
stiffness falls back to the first three points when the window is too thin, which is the
silent fallback of spec section 5.8; `grid.initial_stiffness` raises with the window and the
point count. Neither path fires on this campaign, which is exactly why leaving them in would
have cost nothing and been wrong.

**Interpolation refuses to extrapolate.** `np.interp` holds the endpoint value outside the
data range without complaint, which would silently invent a flat tail for any curve that
stopped short of 20 mm. `interpolate_onto_grid` checks the span first and raises. All 198
curves cover the grid, so again the check never fires on this data, and again that is the
point: it fires the day a Track B rerun does not reach full displacement.

**Determinism, checked rather than assumed.** I deleted both stage directories, reran cold,
and the full suite still passes, including the tests that compare the committed
`data/processed/` copies against the SHA-256 recorded in each stage manifest. Byte identical
output across a full delete and rerun, which is the P4 bitwise reproducibility requirement
arriving three phases early for free.

**Measured, not estimated.** Ingest 3.08 s, grid 0.21 s, both from the manifests. Rerun of
either is a cache hit in well under a second. Full suite 139 tests in 8.4 s. `ruff check src
tests scripts`, `dash_lint.py` and `check_file_sizes.py` all clean, the last over 143 tracked
files. The spec estimates P1 at 1 to 2 sessions; it took one.

**What P2 inherits.** A `qoi.parquet` on the grid basis, which is the right basis for a
surrogate that predicts curves on that grid, plus the warning written into
`DESIGN_DECISIONS.md` that its peak load is not the audit's 38.15 kN and must never be quoted
as though it were. The audit stage will want `raw_curve_qoi` for its own headline table.

## 2026-08-30, CI proof of failure gate (spec 18.1) executed

Build spec 18.1 says a CI that has never been red protects nothing, so before P2 wrote any
real code I planted each of the four faults the gate exists to catch, one at a time, pushed,
and recorded the run. Each fault was reverted before the next was planted, and the fifth run
proves the tree is green again with no fault left behind.

| Run | Planted fault | Gate that caught it | Outcome |
|---|---|---|---|
| [33279017399](https://github.com/Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM/actions/runs/33279017399) | banned identifier in `src/` | `scripts/dash_lint.py`, ground rule 4 | red |
| [33279026622](https://github.com/Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM/actions/runs/33279026622) | distribution literal outside `config.py` | `scripts/dash_lint.py`, binding law 2 | red |
| [33279028700](https://github.com/Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM/actions/runs/33279028700) | broken import | pytest collection | red |
| [33279030135](https://github.com/Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM/actions/runs/33279030135) | em dash in a tracked text file | `scripts/dash_lint.py`, ground rule 3 | red |
| [33279031419](https://github.com/Olajide-Badejo/Uncertainty-Quantification-and-Surrogate-Modeling-in-FEM/actions/runs/33279031419) | none, all four reverted | every gate | green |

The four faults are the four the spec names, and each one failed the specific job it was
supposed to fail rather than failing the build for some unrelated reason: the three lint
faults stopped the `lint` job and left the test jobs untouched, and the broken import stopped
collection in both `test-fast` matrix legs. That distinction is the whole value of the
exercise. A gate that goes red for the wrong reason is a gate that will go green for the
wrong reason too.

## 2026-08-30, Phase P2: audit, censoring model, first compiled report

The phase that turns "the campaign is censored" from a caveat in a document into an object
the pipeline can be held to.

**The reclassification is a measurement, not a recall.** `audit.classify_samples` reads the
ingest artifacts and decides all 400 rows against five criteria that live in the `audit:`
block of `configs/pipeline.yaml`: full displacement span, zero non finite, monotone
displacement, full step time, and a minimum increment count. It measured 198 valid, 202
missing, 0 partial and agreed with the committed `sample_validity.csv` on all 400 rows with
zero disagreements, which is the P2 gate. The 12 quartile failure rates matched the committed
audit exactly, integer for integer, and the regenerated tests reproduce the audit's own p
values to 1e-9: c_top chi squared 2.929e-11 with point biserial -0.2432, Fcm 0.0058,
c_bottom 0.303. I gated on the integer counts rather than on the float rates, because a rate
is a ratio of two integers over the same 400 samples and there is no floating point slack to
allow for. Allowing some would have meant the gate could not tell a real change from a
rounding difference.

**The `partial` tier has no positive case in this data, so I built one.** Every present run
in the inherited campaign is complete, so the branch that classifies a partial run is never
exercised by the real data. That is precisely the branch that has to work the day a Track B
rerun stops at 12 mm. `tests/test_audit.py` drives it on synthetic frames, one per failure
mode: short displacement span, too few increments, reversed displacement, unfinished step
time. A tier that is only ever tested by data that happens not to contain it is an untested
branch wearing a passing test.

**GPC over logistic, and the guard that made it a measurement.** The Gaussian process
classifier shipped. It was not assumed: `fit_completion_model` tries the primary, measures
its cross validated AUC and its prediction spread, and takes the pre authorized logistic
fallback only if a configured floor is breached, recording the attempt either way. The GPC
cleared both floors on the first attempt (AUC 0.7018 against a 0.55 floor, spread 0.65
against 0.01) so the fallback was never taken, and the manifest says so rather than leaving
it to be inferred from the absence of a note.

**The AUC is modest and I did not tune it.** 0.7018, with a 90 percent bootstrap interval of
[0.6576, 0.7419] over 1000 resamples. That is what a three input model of a binary solver
outcome buys when half the design failed, the effect is carried mostly by one input, and the
production logs that would have explained the rest were not preserved. What matters for the
use the model is actually put to is calibration, not ranking, because the validity domain
thresholds a probability: Brier 0.2193 against 0.2500 for the base rate predictor, expected
calibration error 0.0257, and a reliability table whose predicted and empirical rates track
each other across all eight populated bins. The one visibly off bin holds two samples out of
400, and the figure draws it at two samples' worth of area, which is the honest way to show
it.

**The fold honesty is inside the pipeline, not around it.** The standardizer is a step of the
sklearn `Pipeline`, so every cross validation fold refits it on its own training half. Doing
it the easy way, scaling all 400 rows once and then cross validating, would have leaked each
test fold's location and scale into its training fold. The leak is small at this sample size,
which is exactly why it would never have been noticed.

**The lengthscale that wants to be infinite.** The GPC fit reports a lengthscale of about 20
on the standardized bottom cover against 1.4 and 0.8 for the other two, and some folds push
it to the configured bound and emit a convergence warning. This is the kernel correctly
saying the bottom cover has no effect, which is the same thing the chi squared test says at
p = 0.30. I raised the bound twice to see whether it mattered and the AUC did not move from
0.7018 in the fourth decimal, so it is a flat direction in the likelihood rather than a fit
problem. Rather than chase an unbounded lengthscale, I recorded the fitted values in the
manifest under `fitted_hyperparameters`, so the saturation is visible as data instead of
being inferred from a warning nobody reads.

**The validity domain is an intersection, deliberately.** P(complete) at or above 0.5 *and*
inside the box of the executed design. The second half is the one that is easy to leave out
and the one that matters most: a smooth kernel asked about a point far outside the design
will answer, and a high completion probability with no design points beneath it is the
manufactured confidence this whole project exists to eliminate. The domain admits 52.2
percent of the design, but it separates the outcomes properly, 71.2 percent of completed runs
against 33.7 percent of failed ones, and `tests/test_validity.py` pins that the censored
corner (low top cover, high strength) is rejected by the *probability* rather than merely by
the box, so the contract cannot silently degrade into a bounding box check.

**The weighting study bounds the bias, and the answer is small.** Inverse probability of
completion weights on the 198 survivors move the peak load mean by 0.08 percent and nothing
in the table by more than 4.68 percent, on the residual load at 20 mm, which is the quantity
most sensitive to the softening branch and therefore to exactly the configurations that
failed. Normalized weights top out at 3.20 for an effective sample size of 170 against a
nominal 198, so the reweighting is not being carried by a few survivors. That result cuts
both ways and the report says so: it licenses the unweighted descriptive statistics, and it
does not license extrapolation into the corner, where the problem is missing data rather than
a biased average.

**Every number in the report now arrives through a file.** `scripts/make_data_card.py` reads
only artifacts and manifests and writes `docs/DATA_CARD.md` plus 61 LaTeX macros and four
table fragments under `report/tables/`. `main.tex` contains no digits in its prose that were
not inputs. A staleness test regenerates all six files and asserts byte identity, so a card
that has drifted from the pipeline is a failing test rather than a document nobody rechecked.
Writing the generator caught a real defect immediately: my first macro naming scheme took the
first two underscore separated parts of each column name, and both covers begin `c_nom_`, so
the top cover's statistics were silently overwriting the bottom cover's. A duplicate name
guard in the generator now makes that collision an error at the point the name is chosen.

**The figure script no longer reaches outside the repository.** It used to read
`Scripts_2_0/03_postprocess/` and a loose `data_audit/` directory by absolute path. It now
reads the artifact store for the current config hash and fails with a named diagnostic if a
stage has not run. The style moved into `src/ufem/plotting/style.py`, the single style module
of spec section 8, and the two new figures (completion surface, calibration diagram) live in
`src/ufem/plotting/censoring.py` and take arrays rather than fitting anything, so no figure
is ever the place a number is first computed.

**One deviation, recorded.** The `audit` stage runs after `grid`, not between `ingest` and
`grid` as the P0 skeleton registered it. Its classification and censoring statistics need
only the ingest artifacts, but the importance weighting study reweights the QoI table that
`grid` extracts, and the alternatives were to split the stage in two or to have it recompute
the QoI schedule itself. Reordering was the smaller change; the reasoning is in
`docs/DESIGN_DECISIONS.md`.

**Measured, not estimated.** Ingest 3.27 s, grid 0.20 s, audit 9.50 s, all from the
manifests; the audit stage's time is almost entirely the 5 fold cross validation of the GPC
plus the 1000 resample bootstrap. Full suite 188 tests in 18.5 s, up from 139 at P1.
`latexmk -pdf -halt-on-error` builds `main.pdf` at 10 pages with zero overfull boxes; the two
it originally had were a genuinely too wide weighting table and one unbreakable line, both
fixed at the source rather than by widening the margin. `ruff`, `dash_lint.py` and
`check_file_sizes.py` all clean.

**A defect the staleness gate caught on itself.** After everything else was committed I
deleted the audit stage directory and reran cold to check determinism. The AUC and the Brier
score came back bit identical, 0.7018451845184518 and 0.21928970359220137, which is the
result I wanted. But `tests/test_data_card.py` then failed, and the diff was one line: the
provenance table's audit wall time had moved from 9.50 s to 9.56 s. I had put wall times into
a document that a test compares byte for byte. That is a gate that fires on scheduling noise
rather than on numbers, and a gate that cries wolf is a gate someone eventually mutes, at
which point it stops catching the drift it exists for. Wall times still live in every stage
manifest and in this log, where they belong; the card's provenance table now lists the output
digests instead, which is the thing that actually establishes provenance. A test now asserts
that no generated file embeds a wall time at all, so the next person cannot reintroduce it
without being told why. Full suite 194 tests after the fix, and the card survives a
`ufem run audit --force` unchanged.

**What P3 inherits.** A callable validity domain that raises rather than guessing, a
completion model whose calibration is measured rather than assumed, and a report whose
numbers cannot drift from the pipeline because they are not in the report. The registration
stage should consult `ufem.validity.in_validity_domain` before it claims anything about a
region of the input space, and the surrogate stage must carry the domain into its own
artifact.

## 2026-08-30, CI red on clean checkouts, between P2 and P3

Two failures on the P2 merge, both the same root mistake: every gate had been verified on
this machine, which has an artifact store and generated figures, while CI gets a checkout
that has neither. The gates were testing the developer's disk, not the repository.

Six `tests/test_validity.py` tests read the validity domain without depending on the fixture
that skips when the audit stage has not run, so a gitignored `experiments/results/` failed
the suite instead of skipping it. And `.gitignore` carried `report/figures/*.pdf` while
`report.yml`'s own header claimed the LaTeX build needs no artifact store because the figures
are committed, so the TeX Live container had no figures and `latexmk -halt-on-error` died on
a missing `fig_E_collinearity.pdf`.

Both fixed and both verified the way they should have been the first time, by cloning the
repository to a scratch directory and running against that: 134 passed with 0 failed against
6 before, and the report built at 10 pages. The figures are 556 KB in total, so committing
them is comfortably inside the 5 MB rule, and the style module already pins
`SOURCE_DATE_EPOCH` so a regenerated figure is byte identical. Both entries are in
`docs/DEFECT_LOG.md` with their red run URLs.

The lesson worth keeping is narrow. A gate that has only ever been run in the environment it
was written in has not been tested; it has been rehearsed.

## 2026-08-30, Phase P3: registration, reduction, ablation 1

**SRVF is fast and, on this machine, exactly deterministic.** The whole registration stage is
12.99 s on the 198 by 201 family, against a brief that budgeted minutes and authorized
halving the effort knob past 15 minutes. No knob needed touching. More usefully, a forced
rerun reproduces all five artifacts bitwise, so `fdasrsf` with `parallel=False` needs no
tolerance gate and no downgrade to statistically reproducible; `tests/test_p3_determinism.py`
asserts it rather than trusting it. The `parallel=False` is deliberate and is the one
departure from the library defaults: the parallel path splits the family across workers, and
a Karcher mean whose association order depends on scheduling would not reproduce.

**Measured wall times**, all from the manifests: ingest 3.27 s, grid 0.20 s, audit 9.69 s,
register 12.99 s, reduce 0.11 s. The full pipeline is about 26 s, against the 30 minute
budget of spec section 2.

**The gamma gate passes with room.** All 198 warps are monotone with a minimum increment of
3.471e-04, and both endpoint errors are exactly 0.0, against the 1e-8 tolerance the P3 gate
names. Not near the tolerance: at it.

**Two landmark estimators were wrong before they were right, and the data said so.** The
knee window started at 0.5 mm, which put the curvature minimum on the window edge for 135 of
198 curves. These curves are linear to four significant digits over their first few grid
points (1815.2, 1815.0, 1814.7 N per 0.1 mm step) and break between 0.6 and 0.8 mm, so a
0.5 mm bound was cutting into the knee rather than excluding noise. At 0.2 mm no curve pins
at the edge and the knee spreads over 0.5 to 0.9 mm. Separately, the 85 percent post peak
landmark: 34 curves end above 85 percent of their peak at 20 mm, but 33 of those dip below
and recover, so only 1 curve genuinely never softens that far. It is recorded as NaN with a
`u85_reached` flag rather than clamped to the stroke end, which would have put a fictitious
landmark at 20 mm into 17 percent of the family.

**Component counts, two of which contradict the spec's expectation.** Registered amplitude 5
(spec 10.2 expected 3 to 6, and PC1 alone carries 83.6 percent). Phase 63. Damage 11, where
spec 10.2 floated the possibility that 2 would do; it reaches 94.4 percent at 4 and then
carries a long tail. Reported as measured. The phase block being genuinely high dimensional
is the number most likely to matter at P4, because it is the block the warp GPs will have to
predict.

**The ablation refuted one of its own three predictions, which is the point of committing
them first.** Components at 99 percent: 5 registered against 15 unregistered, a ratio of
exactly 3.00 against a predicted 2 to 3. Held. Peak load bias at matched rank 5: -60.8 N
registered against -228.0 N unregistered, both negative and the unregistered one 3.75 times
larger. Held. And `|corr(PC2, d mean/du)|`: 0.111 unregistered against 0.117 registered,
where I had predicted above 0.7 and clearly higher on the unregistered side. Refuted on
magnitude and on direction at once.

A post hoc sweep of the first six components finds the derivative structure in PC1 (0.620
unregistered against 0.499 registered), where the ordering does run as the mechanism argues.
I did not promote that to a confirmation and the report says why: a metric chosen after
seeing six of them is not the evidence a metric named in advance would have been. The likely
explanation is that PC1 here is not a pure amplitude mode, because in this design the curves
that peak late are also, through Fcm, the curves that peak high. Registration keeps its place
on two legs of three, and the report states which leg gave way.

**A gap closed while adding the report section.** `report/tables/ablation_registration.tex`
is written by the ablation script rather than by the card generator, so the byte identity
staleness gate did not cover it: it would have been the one committed number in the report
with nothing behind it. It now has a test asserting it matches the ablation's JSON artifact,
demonstrated failing on a planted 15 to 14 edit and passing on restore.

**Gates.** 243 tests including the slow determinism rerun, up from 194 at P2. `ruff`,
`dash_lint.py` and `check_file_sizes.py` clean over 184 tracked files. `latexmk` builds
`main.pdf` at 14 pages from a cold start with zero overfull boxes and zero undefined
references. The ten pre existing figures regenerate byte identical, so the four new ones are
the only figure changes in the phase.

**What P4 inherits.** Five amplitude scores, 63 phase scores and 11 damage scores per job in
one `scores.parquet`, each block's basis stored with its mean and loadings so a fold can
refit rather than reuse it, and a registration stage fast and deterministic enough to sit
inside a cross validation loop if the fold honest requirement of binding law 3 demands it.
The phase block's dimension is the thing to look at first: 63 scores over 198 samples is not
a comfortable ratio, and it may argue for reducing the phase block harder than 99 percent.

## 2026-08-30, Phase P4: Gaussian process surrogate, baselines, fold honest validation

**One test failure led back into P3.** The inherited surrogate code failed one property test,
`TestSquareRootSlopeRepresentation::test_the_round_trip_recovers_the_family_it_came_from`, with
an `IndexError` inside `fdasrsf`'s own Karcher mean iteration rather than a tolerance miss.
Fourteen seeded synthetic families, round tripped through `warp_tangent_vectors`, showed the
library's default `smooth=True` failing to converge on 6 of them and losing 1 to 5 percent of
round trip accuracy on the rest, ten to a hundred times the representation's discretization
floor; `smooth=False` converged on all 14 at 1.5 to 4.7e-3. The fix is one keyword in
`ufem.register.warp_tangent_vectors`, and it touches the phase block computed at P3: refitting
register and reduce under the fix leaves the amplitude block and the registration ablation's
three metrics exactly where P3 measured them, and moves the phase block's own reconstruction
error from a median of 9.22 percent to 9.98 percent, a worse number that ships because the
alternative crashes. Recorded in `docs/DEFECT_LOG.md` and `docs/DESIGN_DECISIONS.md`.

**The Gaussian process fit meets its budget with room, and reproduces bitwise.** Forty five
targets (5 amplitude, 8 of 63 possible phase, 11 damage, 10 of 18 possible displacement scores,
plus 8 scalar quantities of interest and 3 landmarks), 8 restarts each, 360 individual GP fits,
54.99 seconds of fitting against the 60 second single threaded budget of build spec 10.3; the
representation refit (its own SRVF registration and three principal component bases) costs
12.57 seconds more, for 69.45 seconds of total stage wall time. Restart dispersion in marginal
log likelihood: median 0.0165 nats per point across the 45 targets, worst case 0.462, zero
failed restarts, so the multi start policy is doing real work rather than standing on a flat
surface. A forced rerun (`tests/test_p4_determinism.py`, newly written; the previous agent's
own docstring promised this file and it did not exist) reproduces every one of the nine output
artifacts byte for byte, `gp_state.npy` included, which is the build spec 17.2 gate.

**The baseline gate passes on the four headline scalars and does not on the whole curve, and
both numbers are in the table.** Leave one out out of sample R2: peak load 0.726 against the
best baseline's 0.677 (linear); displacement at peak 0.281 against 0.182; initial stiffness
0.297 against 0.234; absorbed energy 0.445 against 0.428. The surrogate beats all four
baselines (climatology, linear, quadratic chaos, 3 nearest neighbour) on all four headline
quantities build spec 10.5 decides the gate on. At curve level, over 10 grouped folds, it does
not: median relative L2 of 23.09 percent against 22.98 (linear), 22.53 (quadratic chaos) and
22.47 percent (nearest neighbour), all three beating the surrogate, though all four comfortably
beat the training mean's 25.76 percent. The reduced representation asks the process to predict
phase and displacement scores no scalar quantity of interest needs, and those errors compound
through the reconstruction in a way a direct curve average does not pay for. Reported as
measured in `report/tables/baselines_table.tex`, not tuned away and not omitted from the
README status line.

**The fold harness costs about ten minutes, mostly in a library override nobody asked for.**
Total wall time 597 seconds: 0.5 seconds for the closed form scalar leave one out (no refit
needed) and 595 for the 10 grouped folds, each of which refits the registration, all three
principal component bases and the full 45 target Gaussian process ensemble. Measured during
this run: `fdasrsf.fdawarp.srsf_align`, called with `parallel=False` for the determinism reason
its own docstring gives, silently switches to `parallel=True` whenever a family exceeds 100
curves, which every non trivial fold does; a Windows `loky` worker pool spawns and re-imports
the scientific stack per registration. Determinism was not compromised (joblib preserves
submission order in its results regardless of completion order, and the forced surrogate rerun
above proves it byte for byte), but the wall clock and the memory footprint are real. Recorded
in `docs/DESIGN_DECISIONS.md`, not treated as a stop condition, because nothing about correctness
argues for one.

**The manufactured solution converges at the expected rate and would have been vacuous
otherwise.** The cut off agent's note said the family was too easy and hit the representation
floor at n = 64, which would have made the "error decreases with n" assertion pass on any
implementation, broken or not; the committed test already carries the fix (a rising branch
times an exponential tail whose peak, location, rise and width are each a genuinely wiggly
function of the three standardized inputs, not a linear one). Measured: median relative L2 of
0.0234 at n = 64, 0.0166 at n = 128, 0.0151 at n = 198, monotone and a 35 percent reduction
from smallest to largest sample against the 20 percent the test requires, comfortably inside
the 10 percent threshold and the 300 second wall clock budget (152.6 seconds measured).

**Gates.** 315 tests, up from 243 at P3, all passing including the slow markers (the pytest
configuration here does not exclude them by default, so one `pytest tests -q` run is the full
gate). `ruff`, `dash_lint.py` and `check_file_sizes.py` clean. `latexmk` builds `main.pdf` at
16 pages, zero overfull boxes in the new material (one pre-existing underfull box in the P2
importance weighting table, unrelated to this phase). The baselines table and the peak load
predicted against actual figure are both generated from the validate stage's own artifacts,
never a second computation for the report to show.

## 2026-08-30, Phase P5: conformal calibration, scalar and functional

**Two inherited defects were fixed before the phase's own work started, and one of them was
mine to find.** `ufem.runner.is_cache_hit` recomputed a stage's cache key from the input hashes
that stage's own manifest had recorded, which compares a number against a copy of itself and
agrees no matter what happened upstream. An artifact regenerated or edited after a downstream
stage ran was served stale unless somebody passed `--force`, which is exactly how a calibration
stage ends up reading last week's surrogate. Every stage now exposes `declared_input_hashes`,
the runner hashes the files afresh at check time, and a stage without the declaration raises
rather than being trusted. `tests/test_runner.py` is red on 3 of 7 cases before the fix and
green after. The second was found by reading the P4 merge's CI: two tests in
`tests/test_surrogate.py` read the stage manifest without the fixture that skips when the
artifact store is absent, so the clean checkout job raised where the other 83 artifact tests
skipped, and `test-full (ubuntu)` had been red since the P4 merge. Both are in
`docs/DEFECT_LOG.md` with their evidence.

While that CI failure was being diagnosed it looked like it was hiding a second, real defect:
the committed surrogate manifest records `gp_fit_wall_time_s = 61.28` against the 60 second
budget of build spec 10.3, with `fit_budget_met = false`. Refitting the stage on an unloaded
machine measures 53.95 s, which agrees with the 54.99 s the P4 log recorded. So the budget was
not systematically exceeded; the stored artifact came from a run that happened to be 14 percent
slower than the same fit is now. Worth writing down because the assertion is on a wall clock,
and a wall clock assertion with 14 percent of run to run spread and a 10 percent margin will go
red again on a busy machine.

**The scalar recalibration is a null result, which is the outcome that argues for the P4 noise
hyperprior.** The out of fold variance scaling factors over the 11 scalar targets run from
0.998 to 1.199, and nine of the eleven sit within one percent of unity. Build spec 11.3 expects
a factor near 1 when the Gaussian process noise fit is honest, and this is that expectation
measured rather than assumed: the predictive variance adequacy before any scaling is +0.016 or
better on nine targets. The two exceptions are the knee landmarks, `u_knee_mm` at 1.199 (PVA
+0.362) and `P_knee_N` at 1.105 (PVA +0.199), which are the noisiest quantities in the schedule
and the two whose extraction the P3 log already recorded as delicate. Nothing was done about
them beyond reporting them: conformal is valid either way, and a target whose variance is 20
percent optimistic gets a 20 percent wider conformal quantile automatically.

**The curve variance is too small by nearly a factor of two, and the construction says why.**
The force curve's scaling factor is 1.793, from a predictive variance adequacy of +1.168. That
is not a surprise and it is not a defect: `predict_curve` propagates the amplitude score
variance linearly through the reconstruction, and phase and displacement uncertainty enter that
reconstruction nonlinearly, so they are absent from the propagated variance by design (build
spec 10.4 says so and prescribes sampling for the bands). Drawing the configured 256 posterior
realizations through the full nonlinear reconstruction on 12 curves gives a pointwise spread
1.23 times the propagated one at the median, 2.34 at the 95th percentile: same direction, same
order, from an independent route. The damage curve needs almost nothing, 0.972.

**A gate failure that was mine, not the model's.** The first complete run failed the gate on two
of four headline quantities, and it failed by over covering: 95.5 percent for displacement at
peak and 96.5 percent for initial stiffness against a nominal 90. The band was not wrong, the
measurement was. Evaluating a jackknife+ interval at a training point uses n-1 leave one out
models that were every one of them fitted on that point's response, so at that point they
interpolate rather than predict, and the effect is largest for exactly the quantities whose in
sample fit most exceeds their out of sample fit, which is what displacement at peak (LOO R2
0.281) and initial stiffness (0.297) are. The fix is a nested leave one out: remove the query
point first, then build the ensemble, the scores and the scaling factor inside what is left.
The closed form makes it affordable, because the reduced inverse comes from the inverse already
computed rather than from a second factorization, so 198 nested problems per target cost O(n^3)
in total and 2.5 seconds for all 11 targets. It is tested against explicit leave two out refits
on all 56 pairs of a toy design to 1e-9. I record this as a gate that did its job: a threshold
that only ever passes is not a gate, and the first thing this one caught was an evaluation
that flattered the model.

**Measured coverage, after that fix.** Every construction lands on 179 of 198 at the 90 percent
level and 189 of 198 at 95, which is 0.9040 and 0.9545. That is not a coincidence between
targets: the conformal rank rule returns a fixed count by construction, and the finite sample
bracket at n = 198 is [0.900, 0.905] and [0.950, 0.955] respectively. The 95 percent Wilson
interval is [0.855, 0.938] at the 90 percent level, which contains 0.90. The jackknife+ and the
10 fold CV+ cross check agree to the last digit on all four headline quantities, so the
hyperparameter reuse caveat build spec 11.1 asks to be stated costs nothing measurable at this
sample size. The simultaneous functional bands reach the same 0.9040 on both the load
displacement and the damage families.

**The damage family cannot support a band over most of its domain, and that is the saturation
finding again.** Standardizing a residual needs a variance, and 84 of the 201 damage stations
have none: the first 0.4 mm, before damage initiates, and everything past 12.2 mm, where all
198 runs sit at the same saturated value. The band is calibrated on the 117 stations where the
family actually varies, the excluded span is recorded in the artifact, and no variance was
floored to keep the rest, which would have been the fabricated uncertainty of build spec 5.1
reintroduced through the back door. The load displacement family loses only the origin, where
displacement control makes the force exactly zero for every run.

**The PIT gate criterion, and how close the uncalibrated model came to failing it.** Build spec
11.5 words its third criterion visually, no gross U shape on the softening branch, and a visual
criterion inside a machine checked gate is not a criterion. It is implemented as the fraction of
PIT values in the outer two deciles over the post peak part of every curve, which a calibrated
predictive distribution puts at 0.20 by definition. The threshold, 0.35, was written into
`calibrate.py` before the first measurement was taken and the commit order is the evidence.
Measured on the force curves: 0.348 before the variance scaling, 0.112 after. The uncalibrated
model was within a percentage point of failing, and the measured scaling is the whole of what
moved it. The after value being well below 0.20 rather than at it says the recalibrated
predictive is now conservative in the tails, which is what matching a mean square on a heavy
tailed residual field does; it is reported rather than tuned.

**The bands are wide, and the calibration is not why.** The simultaneous 90 percent multiplier
on the force curves is 5.73, and the median sup norm score is 1.57 against a worst case of 7.08.
Part of that is the price of simultaneity over 200 abscissae, and the rest is the curve level
accuracy P4 already reported, where the surrogate loses to three of its four baselines. The
figure in the report shows the worst run in the campaign deliberately. A band that looked
comfortable on three cherry picked curves would be a decoration.

**The R cross check of build spec 11.2 was replaced, and the substitution is argued rather than
assumed.** R is not installed on this machine, and adding an R toolchain for one function call
would put the check somewhere CI cannot run it, which is how a cross check quietly stops being
true. `tests/test_conformal_functional.py` replaces it with constructions whose answers are
computable by hand, where the assertion is exact equality rather than agreement to a tolerance,
plus two measurements of the guarantee itself: the coverage of the constructed band against the
exact rank probability k/(n+1) over 40000 replications, and a 500 replication simulation on
synthetic curve valued data that lands at 0.896 against the finite sample bracket [0.900, 0.925]
with a standard error of 0.0134. What the substitution gives up, an independent author's reading
of the same paper, is stated in the test module's docstring.

**Wall times**, from the manifests: the whole calibrate stage is 141.9 s, of which the scalar
jackknife+ including the nested evaluation of all 11 targets is 2.5 s, the functional bands are
1.0 s, and the 10 fold CV+ refit is 135.9 s. In other words the entire conformal apparatus this
phase exists to build costs 3.5 seconds, and 96 percent of the stage is the honest cross check
that build spec 11.1 asks for. That is the right ratio for a project whose predecessor's
uncertainty was free and fictional.

**Gates.** 358 tests, up from 315 at P4, all passing including the slow markers. The calibrate
stage reproduces all nine of its artifacts bitwise on a forced rerun
(`tests/test_p5_determinism.py`), which matters more here than at P4 because the CV+ cross check
refits 110 Gaussian processes and the modulation check draws 256 realizations per curve.
`ruff`, `dash_lint.py` and `check_file_sizes.py` clean. `latexmk` builds `main.pdf` at 20 pages,
up from 16, with no new overfull or underfull boxes and no undefined references. The two new
table fragments are generated by the calibrate stage's own table functions and regenerated
through `scripts/make_data_card.py`, so the byte identity staleness gate covers them.

**What P6 inherits.** A calibrated surrogate whose intervals mean what they say, with the gate
of build spec 11.5 passed and recorded in the manifest, which is what unblocks the propagation
of Section 13. The number to carry forward is the curve scaling factor of 1.793: the sensitivity
stage works on the score processes directly, where the scalars' honest variance applies, but
anything that reaches for a curve level predictive spread should use the recalibrated one and
say so.

## 2026-08-30, Phase P6: global sensitivity, and a gate that stopped the phase

**The headline is that nothing was published.** All 24 sparse chaos expansions failed the
corrected leave one out Q2 gate of build spec 12.1. The best of them reaches 0.688 on peak
load, the second best 0.631 on the leading amplitude score, and the remaining 22 sit between
-0.015 and 0.421. The threshold to publish a ranking is 0.80 and to publish values 0.95. So no
Sobol index value and no input ranking leaves this campaign, and what the phase publishes
instead is the measurement that decided it plus a diagnosis of why.

The Q2 values are not an artifact of my implementation and the cross check that says so was
already in the repository. The P4 fold honest harness measured out of sample R2 for the same
quantities under an entirely different model: 0.726 for peak load, 0.281 for displacement at
peak, 0.297 for initial stiffness, 0.445 for absorbed energy, against chaos Q2 of 0.688, 0.145,
0.231 and 0.421. The chaos expansion lands where the Gaussian process landed, a little below it
as a simpler model should, and comfortably above the quadratic chaos baseline of P4 (0.666,
0.071, 0.158, 0.401) as a degree 5 sparse basis should. Three independent fits agree about how
predictable this campaign is, and the answer is: not very.

**Why, measured two ways.** Neither measurement is about the expansion.

The first reuses the surrogate's own noise fit. In standardized units the fitted nugget and the
kernel outputscale account for the whole target variance, so `outputscale / (outputscale +
nugget)` is what an entirely different model family concluded the three inputs determine. For
the four headline quantities that ceiling is 0.894, 0.504, 0.566 and 0.695. Three of the four
expansions are already at or near their own ceiling. A richer basis has nothing to reach for.

The second is model free and it is the one I would put in front of somebody who does not want
to hear about metamodels. Take each of the 198 training points, find its nearest neighbour in
the standardized input space, keep the closest tenth of those pairs, which are separated by at
most 0.185 in standardized units against fitted correlation lengths of 0.66 to 10. The median
absolute difference in peak load between those nearly coincident designs is 0.393 times the
peak load standard deviation over the whole campaign. For displacement at peak it is 0.695 and
for initial stiffness 0.557. Two beams differing by a fraction of a millimetre of cover and a
fraction of a megapascal of strength produce responses differing by a large share of the entire
campaign's spread. The response surface varies on a scale finer than this design resolves. No
smooth metamodel of any family certifies against that, and the Q2 gate is reporting it
faithfully.

The likely cause is the pair of modeling defects the build spec named in section 6.2 before any
of this was measured: tension softening left unregularized by a fracture energy while the
strength varies sample to sample, and no mesh convergence study. Under that combination the
localization pattern is free to jump between neighbouring designs. I want to be careful here,
because this is an inference and not a measurement: what is measured is that the response is
rough at the design's resolution, and what is inferred is why. Track B's first two items are
exactly the corrections that would test the inference, and this phase is the strongest
quantitative argument the project has for spending that solver time.

**A diagnostic that needed a caveat of its own.** The explainable ceiling is meaningless where
the Gaussian process fit is itself degenerate, and on 10 of the 24 targets it is: those
processes drove at least one length scale onto its configured lower bound with a nugget near
zero, which is the interpolate the scatter corner of build spec 5.2 and reports a ceiling near
one for a target nothing can predict. Every one of the ten is a trailing principal component
carrying almost no variance. Their ceilings are withheld from the table rather than printed,
and the flag is computed rather than eyeballed. I nearly shipped that column without the
caveat, which would have put four rows reading "ceiling 0.99, Q2 0.00" in a report as though
they meant something.

**The two constructions do not agree, and that is the same finding again.** Build spec 12.2
makes agreement the acceptance criterion. The analytic chaos index falls inside the Gaussian
process posterior 90 percent interval on 45 of 144 index rows. Investigated, in the order I
investigated it:

1. *Not the pathwise approximation.* Raising the Fourier feature count from 1024 to 16384, which
   more than halves the measured kernel deviation, moves the posterior median first order index
   for peak load by 0.004 and for initial stiffness by 0.023, against gaps to the chaos value of
   0.11 and 0.31. Ruled out.
2. *Mostly structural.* Least angle regression selected no interaction term at all on 16 of the
   24 expansions, so the chaos total index equals the chaos first order index exactly on 53 of
   72 input slots and the median chaos interaction share is 0.000. A Gaussian process
   realization is a generic rough function whose interaction is never exactly zero: 0 of 72
   slots, median interaction share 0.312. Two model families that disagree about whether the
   response is additive disagree about total indices whatever the data says, and the counts show
   exactly that shape: first order indices agree on 32 of 72 rows, total indices on 13 of 72.
3. *Partly a boundary technicality.* Where the expansion drops an input the chaos index is
   exactly zero while the posterior interval for a negligible input starts just above zero, so
   the two differ by a thousandth and containment records a miss. A tolerance of 0.01 raises the
   count to 66 of 144 and 0.05 raises it to 85.
4. *The rest is real.* Two models neither of which reaches Q2 0.80 attribute the explained part
   of the variance differently, and there is no reason they should not. Agreement does not
   improve with Q2 (Spearman -0.098), because none of the expansions is in the regime where it
   would. The posterior mean's indices, computed only as a diagnostic because build spec 12.2
   is right that they are biased, sit between the realizations and the chaos values on initial
   stiffness (0.765 against 0.636 and 0.942), which is the smoothing bias the spec warns about
   showing up exactly where it was predicted to.

I considered whether the containment criterion is simply too strict, since the chaos index has
no interval of its own. It is strict, and that is why the gap ladder is reported beside the
count rather than instead of it. What I did not do is relax the criterion after seeing the
result.

**The functional indices, and a prediction I could not decide.** The prediction of build spec
12.3, five falsifiable statements, was committed to `docs/ABLATIONS.md` in commit `0ebc224`,
before `src/ufem/sensitivity.py` existed in commit `8788ced`. Measured: the leading and second
ranked inputs swap 0 times over the 200 usable stations. The concrete strength leads everywhere,
already at 0.931 of the variance at the first station the decomposition exists at (0.03 mm), and
the top cover comes no closer than 0.167 behind, at 13.7 mm on the softening branch. That
refutes prediction one on direction and on location at once. Predictions four and five hold: the
bottom cover's total index never exceeds 0.0012 against a predicted 0.10, and the interaction
share never exceeds 0.006 against a predicted 0.15.

And none of that decides anything, because the expansions those indices come from are the ones
the gate withheld. A prediction cannot be decided against evidence the same report declines to
publish. The report says the prediction is undecided on this campaign and states the direction
the withheld numbers ran, and `docs/ABLATIONS.md` keeps the prediction unedited. This is the
first time in this project that the honest answer to "did the prediction hold" has been "the
measurement is not admissible", and I would rather write that than quietly promote a withheld
number because it happens to be interesting.

**Deviation taken: the Sobol design is 2^13 per realization, not 2^15.** Measured on this
machine over 24 targets and 200 realizations each, 2^15 costs 11.0 minutes and 2^13 costs 2.7,
and the two produce index medians and 90 percent interval widths that agree to three decimals.
The reason is structural rather than lucky: the scrambled Sobol design is drawn once and shared
by all 200 realizations, so the Saltelli Monte Carlo error is common to them rather than spread
between them, and it shifts every realization together instead of widening the reported
interval. Quadrupling the design would move no reported digit and would put the pipeline over
the 30 minute gate of build spec section 2. Recorded with both measurements in
`docs/DESIGN_DECISIONS.md`.

**Two library traps, both now tests.** `SALib.sample.sobol` is not an attribute of
`SALib.sample` until the submodule is imported, which is SALib issue 663 and the gotcha build
spec 12.2 warns about; the test checks it in a fresh interpreter so the module cache cannot hide
it. And `ot.FunctionalChaosValidation` raises `InvalidArgumentException: Cannot perform fast
cross-validation with a polynomial chaos expansion involving model selection`, while OpenTURNS
1.27's `FunctionalChaosResult` has no `getRelativeError` at all. Both are correct refusals, so
the corrected leave one out is written out in this stage with its approximation stated, and it
is tested against explicit refits on every fold of a toy design to a relative 1e-9.

**The oracle.** This phase is the first with a genuine one. An additive plus one interaction
polynomial in the three inputs has closed form Sobol indices from the marginal moments alone,
including a third input whose first order index is exactly zero and whose total index is not,
which is the asymmetry a symmetric test function would not catch. The chaos route recovers those
indices to better than 1e-12 against a required 1e-6, and its Q2 comes out at 1.0 because the
function is exactly in the span of the candidate basis. The Gaussian process route recovers them
inside its own posterior interval. After four phases of property tests standing in for an
absent oracle, having one is a relief.

**Wall times**, from the manifests: sensitivity 193.6 s, of which the 24 chaos expansions are
0.13 s, the functional decomposition 0.49 s, and the 200 posterior realizations per target on a
40960 point Saltelli design 190.7 s. In other words the primary construction of build spec 12.1
costs a tenth of a second and 98.5 percent of the stage is the cross check that says it is not
an implementation error. Full pipeline, ingest through sensitivity: 995 s, about 16.6 minutes,
inside the 30 minute gate of build spec section 2 with 582 s of that being the P4 fold harness.

**Gates.** 421 fast tests, up from 375 at P5, plus the slow markers. `ruff`, `dash_lint.py` and
`check_file_sizes.py` clean over 218 tracked files. The sensitivity stage reproduces all seven
of its artifacts bitwise on a forced rerun. `latexmk` builds `main.pdf` at 27 pages, up from 20,
with no new overfull or underfull boxes and no undefined references. Three new generated table
fragments joined the byte identity staleness gate, and one of them, `sobol_indices.tex`, is
entirely dashes, which is what the gate firing looks like in a report.

**What P7 inherits.** Nothing it can use, which is the point. There is no published sensitivity
ranking to prioritize a limit state with, and the propagation stage should not invent one. What
it does inherit is a number worth carrying: the response is rough at the scale this design
resolves, by a median of 39 percent of the peak load standard deviation between nearest
neighbours, so any failure probability computed from a smooth surrogate of this campaign carries
that roughness as an unmodeled error and the report has to say so next to the Pf.

## 2026-08-30, Phase P7: propagation, limit states, reliability with honest bounds

**The headline number is not the failure probability, it is the gap between it and its bound.**
Over 100000 input draws through the mean surrogate, the peak load falls below its characteristic
value with probability 0.04787 (binomial standard error 0.00068), the residual capacity ratio at
20 mm falls below one half in no draw at all, and the compressive damage at 10 mm exceeds 0.93
with probability 0.29580 (standard error 0.00144). The surrogate aware bounds of build spec
13.2, counting a failure whenever the calibrated 90 percent jackknife+ interval crosses the
threshold, are 0.26537, 0.80672 and 0.95976. The bounds are five, unboundedly and three times the
point estimates. That is the result of this phase: on this campaign the reliability estimate is
dominated by what the surrogate does not know rather than by the variability of the beam, and any
report quoting only the left hand column would be describing a precision it does not have.

The residual capacity row is the clearest case. The mean surrogate never once predicts a
softening ratio below one half, so the point estimate is a bound at the resolution of the sample
and would, alone, read as a member that cannot fail this criterion. The band says 0.81. Both
numbers are true statements about different questions, and the second is the one a reader should
carry away.

**46.6 percent of the Monte Carlo mass falls outside the validity domain.** This is the largest
single qualification on everything above and it was not a surprise: 202 of the 400 designed runs
never converged, the completion model puts the domain boundary at P(complete) = 0.5, so a
population drawn from the declared input distributions lands outside that boundary about half
the time. It appears in every table, on every figure, and in the stage manifest. The stage also
reports each probability restricted to the draws inside the domain: 0.0491 for peak load against
0.0479 overall, and 0.3259 for damage against 0.2958, so the censored corner is the more
optimistic one for the damage limit state by three points of probability.

**The threshold revisit, and the config change it caused.** Build spec 13.2 asks for
characteristic value logic on the peak load limit state, and the number committed with the config
skeleton at P0 was a placeholder of 31000 N with a comment saying so. The 5th percentile of the
propagated aleatory peak load is 33253 N. The placeholder sat 6.8 percent below it, which is a
different limit state rather than a rounding, so the config was changed once to 33200 N, the
measurement rounded down to the nearest 100 N. That changed the config hash and forced a cold
rerun of the whole pipeline, 17.0 minutes, which is the mechanism working rather than a cost to
be avoided. The stage recomputes the characteristic value on every run and records the gap to the
configured number, now 0.16 percent, and a test fails if the two drift apart again. Recorded with
its reasoning in docs/DESIGN_DECISIONS.md.

**The two layers, and what the second one says.** The aleatory layer is the input distributions
through the mean surrogate. The epistemic layer is 64 draws from the calibrated predictive
distribution at each of 20000 seeded subsampled inputs, where calibrated means the fitted
predictive standard deviation times the variance scaling factor P5 measured against held out
data. The ratio of epistemic to aleatory standard deviation is 0.627 for peak load and above one
for seven of the eleven propagated quantities, reaching 1.844 for the softening ratio: for most
of the QoI schedule the surrogate's uncertainty about a single beam exceeds the variation between
beams. The layers are never added, and the separation is a property test rather than a
convention, since the epistemic layer takes the standard deviation as an argument and returns the
mean exactly when it is zero.

Two internal consistency checks came out clean and are worth recording because they were the
places a mistake would have hidden. The sampled predictive failure probabilities agree with the
closed form Gaussian integral of the same quantity to 3e-4 or better on all three limit states
(0.0769 against 0.0772, 0.1517 against 0.1514, 0.3842 against 0.3840). And the conservative
counting rule is a union of the band crossing with the point failure, guarding against a
jackknife+ interval that fails to contain the full model's mean; the union fired zero times over
the 300000 draw and limit state combinations, so the guard was never needed and the count says so
rather than the guard being assumed away.

**The floor.** No probability below 1e-4 is claimed. The limit is not the Monte Carlo sample,
which resolves 1e-5 per draw; it is 198 training runs on a response P6 measured to be rough at
the design's own resolution. The statement is in the artifact, in the table caption, and in the
report.

**The analytic cross check disagrees, and the disagreement localizes exactly.** The v1 physics
informed script was reimplemented as mechanics rather than transcribed. What survived is the
geometry: its second moment is the gross value of the 250 by 150 mm section to three figures and
its 1.6 m is the pin to load distance, so the script had the member in front of it. What did not
survive is the structural system. It used a propped cantilever deflection of 7PL^3/(96EI), which
matches no standard case (the propped cantilever with a central load gives 7PL^3/(768EI)) and in
any event the model of build spec 6.2 is a determinate span with an overhang; it applied an
invented strength factor on top of a modulus that is already a function of the strength; and it
computed a deflection under a fixed 50 kN load, so its output could not be compared with anything
the campaign measured. The clean model is a plane section rectangular stress block on the hogging
section at the roller, whose effective depth is exactly the top cover, divided by the 800 mm
lever arm.

Verdict, against a model error of 15 percent declared before the comparison ran: the central
tendency brackets and the dispersion does not. Analytic median 43.48 kN against the surrogate's
38.27 kN, a ratio of 1.136, inside the stated error. Analytic coefficient of variation 0.0252
against 0.0772, a factor of 0.326, and the analytic 5th percentile is 1.253 times the
surrogate's, outside it. The two distributions agree about where the capacity is and disagree
about how much it varies.

The elasticities say why, and this is the part worth keeping. Regressing log peak load on log
inputs over the 198 runs gives 0.735 for the strength, 1.001 for the top cover and -0.050 for the
bottom cover, at a fit R2 of 0.698. The analytic model gives 0.048, 1.088 and -0.040. Two of the
three agree closely: the geometric sensitivity is reproduced almost exactly, which says the
structural idealization is right and the top cover really is the effective depth in both. The
strength elasticity is out by a factor of fifteen. A section whose tension steel yields carries
its reinforcement force whatever the concrete strength is, so an under reinforced flexural
mechanism has almost no strength sensitivity, and 0.048 is what that mechanism implies. The
campaign's 0.735 sits near the elasticity of the finite element model's own tensile strength
scaling, ((fcm - 8)/20)^(2/3), which is 0.935 at the mean, and near the compression scaling's
1.0. The peak load of the inherited campaign is governed by the concrete constitutive response,
not by reinforcement yielding.

That is a statement about the finite element model, and build spec 6.2 predicted it before any of
this was measured: tension softening defined without fracture energy regularization while the
strength varies sample to sample, and no mesh convergence study. It is the same defect P6
measured from the other direction as roughness between neighbouring designs. Two phases, two
methods, one cause, and it is the strongest argument this project has for the Track B
corrections.

Two further checks are consistent with that reading. Evaluated at each of the 198 training
designs the analytic capacity correlates with the measured peak load at 0.406 and overpredicts it
by 15.5 percent in the mean, which is the correlation a model reproducing only the geometric part
of the variation would show. And the analytic elastic stiffness brackets the measurement without
being tuned to it: 16.54 kN/mm on the gross section and 6.25 kN/mm on the fully cracked
transformed section, against a campaign mean initial stiffness of 14.14 kN/mm, which is where a
secant stiffness measured over a range beginning above the 4.30 kN cracking load belongs.

**Wall times**, from the manifests: propagate 38.6 s, of which the eleven targets' means,
variances, quantiles and density estimates are 29.1 s, the three calibrated bands over 100000
draws 2.6 s, the 2000 curve fan 2.6 s, the input sample and the validity domain 1.7 s, and the
whole analytic model plus its cross check 0.34 s. The budget was 8 minutes and no reduction of
the epistemic layer was needed. The batching is what bought that: one cross covariance block per
chunk yields the posterior mean, the posterior variance and all 198 leave one out cross
predictions, so a 1e5 sample propagation through eleven Gaussian processes is a few matrix
products rather than 1e5 library calls. Full pipeline, ingest through propagate: 1019.9 s, 17.0
minutes, inside the 30 minute gate of build spec section 2, with 576.1 s of it still the P4 fold
harness.

**Gates.** 508 fast tests, up from 421 at P6, and 526 with the slow markers. `ruff`,
`dash_lint.py` and `check_file_sizes.py` clean over 236 tracked files. The propagate stage
reproduces all ten of its artifacts bitwise on a forced rerun. `latexmk` builds `main.pdf` at 33
pages, up from 27, with no new overfull or underfull boxes and no undefined references. Three new
generated table fragments joined the byte identity staleness gate.

One overfull box did appear and was fixed rather than tolerated: the reliability table at its
first full labels ran 96.7 pt past the text block, so each limit state now carries a short label
for the table beside the full one for the artifact and the summary. The zero count cell was the
other half of that width, and it now prints as a bound at the resolution of the sample rather
than as a parenthetical, which is both narrower and more accurate.

**One defect shipped inside this phase's own commits and is logged.** The density figure inferred
which side of each threshold was the failure region from where the threshold sat relative to the
median, which is backwards for a below type limit state by construction, so all three panels
shaded the region where the member passes. Nothing but rendering the figure and looking at it
would have caught it, and the regression test now reads the vertices of the shaded polygon, which
is the closest a test gets to looking: 2 failed of 3 with the heuristic restored, 3 passed after.
The direction is an argument now, taken from the limit state declaration that already carried it.
Five figure labels also carried LaTeX spacing escapes, which matplotlib prints literally, so a
legend read "47 backslash comma backslash percent"; escapes belong in the caption, which LaTeX
typesets.

**What P8 inherits.** A validity domain that excludes 46.6 percent of the input distribution's
mass, which the UI has to shade rather than mention, and a reliability panel whose honest default
view is the bound rather than the point estimate.

## 2026-08-30, Phase P8: UFEM Lab, the latency budget, and the README GIF

**What shipped.** `ufem lab` serves the five panels of build spec section 15 at
`http://127.0.0.1:8080`, over the artifact store and over nothing else. The UI package is
`src/ufem/ui/`: `layout.py` for presentation constants, `store.py` for the artifact store,
`predict.py` for one prediction with its band and its validity verdict, `figures.py` for the
Plotly figures, `app.py` for the panels and the server. Beside it, `scripts/capture_ui_gif.py`
records the README GIF from the running dashboard, and `scripts/make_model_card.py` generates
`docs/MODEL_CARD.md` from the same artifacts the model card panel reads.

**The latency budget, measured.** Build spec 15 sets 50 ms from a slider move to a repaint and
says the surrogate inference is sub millisecond, so the budget is serialization and plot update.
That was almost right and the almost is worth recording. The naive path, `predict_curve` plus
`predict_qoi` through GPyTorch's own prediction, costs **78.7 ms median** for one design point,
because each of the 45 fitted processes pays its own library call overhead and 45 of anything is
not sub millisecond. Assembling the query independent part of every posterior once at startup and
predicting through `ufem.propagate.predict_mean_and_variance` instead brings the same 45 targets
to **11.9 ms**, which is the same arithmetic reorganized rather than a different answer, and a
test in `tests/test_propagate.py` already asserted that equivalence to 1e-9 before this phase
needed it.

Server side, over 100 seeded slider positions inside the executed design box, after 10 warm up
positions discarded:

| Stage of one slider move | Median | p90 |
|---|---|---|
| Predict, 45 targets plus the reconstruction plus 11 jackknife+ intervals | 16.0 ms | 17.4 ms |
| Predict plus both Plotly figures plus JSON serialization | **32.8 ms** | 35.3 ms |

The serialized repaint payload is 42.1 kB for the two curve panels. The assertion in
`tests/test_ui.py` is on the 32.8 ms figure against the 50 ms budget, which leaves 17 ms of
headroom, most of it spent on figure construction rather than on the model.

Browser side, measured by the playwright test: **59 ms** from the mouse click on the strength
slider to the predicted mean curve's SVG path changing in headless chromium, with the server and
the browser on the same machine. The asserted bound is 500 ms, which is generous by an order of
magnitude on purpose: it covers the websocket round trip, Plotly's repaint and whatever the
machine was doing, and a regression that crossed it would be a real one rather than noise. The
probe reads the rendered path rather than the trace data, because Plotly 3 may hand a trace its
`y` as a typed array specification object rather than as a JavaScript array, so reading the data
proves nothing about what is on the screen.

Startup, which is where the interaction cost was moved to: 1.79 s to import, 2.29 s to load the
artifact store and rebuild the 45 GPyTorch modules from their stored parameters, 0.58 s to
assemble the posterior pieces. 4.67 s in total, paid once.

**The GIF.** 180 captured frames stored as 93, 14.66 s of playback, 960 by 600 px, **0.81 MB**,
captured in 24.9 s of wall time with a further 3 s to quantize and write. It is committed at
`docs/media/ufem_lab.gif` and embedded at the top of the README.

Three measurements about it that changed what the spec's numbers mean, all recorded in
`docs/DESIGN_DECISIONS.md`. Pillow merges a run of identical frames into one and accumulates
their delays, so the 180 captured frames become 93 stored ones with no effect on playback. The
GIF format carries a delay in hundredths of a second, so a nominal 12 fps interval of 83.3 ms is
written as 80 ms and the file plays at 12.5 fps, which is why 180 frames last 14.66 s rather than
15.00 s; the script reads the duration back out of the written file and asserts the 12 to 20
second window against what a viewer will see rather than against what was requested. And the
palette pass build spec 15.1 asks ffmpeg for is done by median cut to 128 colors over a strided
sample of 24 frames, shared across every frame, which is also what lets Pillow write later frames
as the bounding box of what changed.

**No file size exemption was needed.** The GIF was expected to force one, since build spec 15.1
fixes 12 fps and 960 px and build spec 3.3 caps tracked files at 5 MB. At 0.81 MB it does not, so
`scripts/check_file_sizes.py` is still one rule with no carve outs and `tests/test_laws.py`
asserts the GIF is tracked rather than asserting it is exempt.

**One new artifact, and the determinism check that let it in.** The reliability panel's threshold
slider needs Monte Carlo rows to recount, and the propagate stage was holding its 100000 draws in
memory and writing summaries. It now also writes `mc_subsample.parquet`, 20000 rows and 5.75 MB
inside the artifact store, carrying the three inputs, the validity flag, the mean and calibrated
sigma of all eleven targets, and the jackknife+ band ends for the three limit state targets. The
rows are the seeded epistemic subsample the stage already draws, so no new randomness enters. The
rerun took **37.3 s** against 38.6 s at P7 and reproduced all ten previous outputs bitwise,
checked against the hashes recorded before the change; `propagation.json` is the only file whose
digest moved, and it moved because it gained a `subsample` block.

That subsample is a subsample, and the panel says so. At the configured peak load threshold it
gives Pf = 0.0463 where the headline sample gives 0.0479, a gap of about one standard error of
the smaller estimate, and a test asserts the two agree within three.

**Binding law 5, made mechanical.** `dash_lint.check_ui_constants` parses every module under
`src/ufem/ui/` and rejects any numeric literal that is neither one of `{0, 1, 2, -1}` nor part of
a module level assignment in `ui/layout.py` whose name ends in a presentational suffix. Parsed
rather than grepped, because `0.9` in a band level and `0.9` in an opacity are the same four
characters. It runs in the CI lint job with no Python stack at all, and `tests/test_ui.py` imports
the same function rather than reimplementing it.

It fired four times during development, and every one was a real finding rather than a false
positive: `int(pair, 16)` in a hex to rgba converter, which became `bytes.fromhex`; three
`[:12]` hash truncations, which became a named `SHORT_HASH_CHARS`; and the port `8080`, which
moved to `ufem/runner.py` where the rest of the command line defaults live, because a port is an
environment fact and not a presentational choice. The check is planted against four ways in the
test file.

**The sensitivity panel is the honest one, and it is the one that took the longest to decide.**
Build spec 15 asks that panel for Sobol bars with uncertainty whiskers and the pointwise stacked
band. Every one of the 24 chaos expansions failed the Q2 gate at P6, so there is nothing to put on
those axes that would not be a fabrication. The panel now shows the gate outcome, the per target
Q2 table with the explainable ceiling and the design roughness beside it, a horizontal bar chart
of Q2 with the two publication thresholds drawn as rules so a reader can see every bar falling
left of both, and the Gaussian process posterior distributions as intervals labeled indicative
only in the P6 report's own words. The functional indices are not drawn at all; the panel says
where they are and that they are withheld. A dashboard is where a withheld number is most likely
to reappear as a picture, which is exactly why it does not here.

**Wall times.** Propagate rerun 37.3 s, of which the eleven targets' moments and densities are
28.9 s, the three calibrated bands over 100000 draws 2.4 s, the curve fan 2.4 s. GIF capture 24.9 s
plus 3 s to assemble. Model card generation under 1 s. The playwright test takes 8 s end to end
including launching the dashboard and the browser. Full test suite including the slow markers:
**676 s**, 11 minutes 16 seconds, which is the P4 fold harness and the manufactured pipeline test
as it has been since P4.

**Gates.** **579 tests pass** with everything selected, up from 526 at P7, of which 39 are
new in this phase: 26 on the dashboard and 13 on the model card. 560 pass with the slow
markers excluded, which is the `test-full` CI job's selection, and 433 pass under
`not slow and not fullstack`, which is the light stack `test-fast` job's selection on a runner
with no torch and no artifact store. `ruff check src tests scripts`, `scripts/dash_lint.py` with
its new third sweep, and `scripts/check_file_sizes.py` over 248 tracked files are all clean. The
staleness gate on `docs/DATA_CARD.md` and the 16 report fragments still passes untouched, and a
second staleness gate now covers `docs/MODEL_CARD.md`.

**`ci.yml` was not changed, and the reason is worth writing down rather than leaving as an
absence.** `test-fast` installs an explicit light stack list and never reads the `[dev]` extras,
so adding playwright and Pillow there cannot reach it. `test-full` does install `[dev]` and will
now acquire playwright, but not a browser, and every test that needs one is marked `slow` and
excluded from both jobs anyway; the browser test additionally probes for the chromium build and
skips with a named reason when it is absent. The lint job runs the new `check_ui_constants` sweep
with no Python stack at all, because it is `ast` and `re`. Nothing about the new dependencies
reaches a job that cannot afford them.

**Deviations from the specification, all four recorded in `docs/DESIGN_DECISIONS.md`.** Pillow
rather than ffmpeg for the GIF. A step driven capture rather than a real time one. No Sobol bars
on the sensitivity panel, because there are no publishable indices. And no size exemption in
`check_file_sizes.py`, because the file that was supposed to need one does not.

**One defect shipped inside this phase's own commits and is logged.** The model card printed
the git commit and the branch the surrogate artifact was fitted at, both read from its manifest.
Rerunning any stage rewrites that manifest with the current commit while reproducing every
output bitwise, so the P4 through P7 determinism tests, which do exactly that, made the byte
gated card stale with two lines changed and no number among them. This is the data card's wall
time defect in a second form and it was not recognized as one, because a commit feels like the
opposite of noise. The card now carries the config hash and the digest of the surrogate record,
which move only when the model does, and points at the manifest for the rest of the chain; a
regression test matches a bare 40 hex character token and fails on the old card. The rule
generalized: a byte gated document carries only quantities that change when the numbers change.

**What P9 inherits.** A dashboard that reads the artifact store and a model card generated from
it, both of which will go stale the moment an ablation changes a fitted model, and both of which
have a test that will say so. The report gained nothing this phase, deliberately: the UI is not a
report section, and the model card figures it points at were already committed at P5.

## 2026-08-31, Phase P9: ablations 2 through 5, and a report that is structurally complete

**What shipped.** Four ablation scripts, one shared yardstick, one table fragment, four results
entries in `docs/ABLATIONS.md`, and three new report sections. `src/ufem/ablation_reference.py`
refits the production pipeline in the same ten grouped folds the validation stage used and keeps
its out of fold mean and pointwise variance, so the four rivals can be scored against it on
identical held out sets. `src/ufem/ablation_table.py` holds the verdict logic and the thresholds
the five predictions committed. The report gained an ablations section, a limitations section and
an outlook, and lost its Phase P2 title.

**The prediction commits are the evidence and they are all in place.** `git log` for this branch
runs: four `docs(ablations): predict ...` commits, then the code, then one results commit per
ablation. No result commit precedes its prediction, and every prediction is left on the page
unedited above the number that contradicted it.

**The reference reproduces the shipped numbers exactly, which is the point of it.** The one
number that had to come out right before anything else could be believed: refitting the
production pipeline fold by fold gives median out of fold relative L2 errors of 0.23088834480636
on the force family and 0.07889520498854 on the damage family, against the values the `validate`
stage committed at P4 to the last digit stored. The stated tolerance was 1e-9 for summation
order; the measured deviation is 0.0.

**Results, in one place.**

| Ablation | Headline | Production | Ablation | Claims held |
|---|---|---|---|---|
| 2. Autoencoder plus GP | median curve L2 | 23.09 % | 20.46 % | 2 of 3 |
| 3. Deep ensemble, 5 members | median curve L2 | 23.09 % | 19.37 % | 2 of 4 |
| 4. B-spline coefficients | median curve L2 | 23.09 % | 20.53 % | 1 of 3 |
| 5. Sobol against thinning | leave one out RMSE at n = 64 | 1929 N | 1680 N | 2 of 3 |

Seven of thirteen committed claims held. The six that failed are more useful than the seven that
held, and they say one thing between them: every direct curve model beats the production
reconstruction at curve level, and every one of them is useless at the peak. The autoencoder
reads a peak load R2 of -0.117 off its own curve and the B-spline reconstruction under predicts
the peak by 3035 N, 7.7 percent, where the production pipeline is off by 285 N. The deep ensemble
loses on both uncertainty metrics, NLPD 12.393 against 10.575 and pointwise coverage 0.761
against 0.791, while beating the production pipeline on both error metrics.

**Wall times, all CPU, all single threaded under the production determinism policy.** The
production fold reference **492 s** (ten elastic registrations plus 340 Gaussian process fits),
paid once and cached. Ablation 2 **259.7 s**, of which 104.7 s is training the twenty
autoencoders and the rest is 120 latent processes; the timebox was 600 s of training and was not
approached, so no epochs were cut. Ablation 3 **247.5 s**, of which 245.4 s is training the fifty
ensemble members. Ablation 4 **205.9 s**, almost all of it the 160 coefficient processes.
Ablation 5 **76.9 s** for 60 process fits. Total 1282 s, 21 minutes, of which the reference is 38
percent.

**Both neural ablations reproduced bitwise across two independent runs**, which was not planned
as a determinism test and became one: the first run of ablations 2 and 3 crashed in the metric
block after training, the metric was fixed, and the reruns produced identical per fold losses and
identical pooled metrics. Build spec 17.2's GPU relaxation therefore never arose, the ablations
keep the production bitwise claim, and `docs/DESIGN_DECISIONS.md` records that rather than
claiming a downgrade nobody needed.

**One thing broke twice and it was the same thing.** A Gaussian log density needs a positive
variance, and this campaign has stations where the honest predictive variance is exactly zero:
every run is displacement controlled from zero, so the force and the damage are identically zero
at the first station, and the damage family is identically zero over an initial span in almost
every run. Where a fold's training half is constant at a station, the fitted basis has a zero
mean, zero loadings and a zero truncation residual, so the production variance there is zero by
construction rather than by overconfidence. The first fix excluded stations where the whole
observed family is constant, which is the rule the P5 calibration stage already applies, and it
was not enough: three damage stations vary across all 198 runs while being constant in some
fold's training half. The second fix intersects that rule with the stations where every model in
the comparison reports a positive variance, so both sides are scored on the same abscissae and
the excluded count is recorded beside the metric. One station of 201 is excluded on the force
family and 87 on the damage family. Nothing was floored (ground rule 4), and the failure was a
raise with a named diagnostic both times, which is why it cost ten minutes rather than a wrong
number.

**A units defect was caught by reading the artifact rather than by a test.** The shared metric
helper named its error keys `rmse_N` and `mae_N`, which is right for a load displacement curve in
newtons and wrong for a dimensionless damage curve, and the damage block of ablation 2's artifact
duly carried `"mae_N": 0.1247`. Ground rule 14 says units are typed at every module boundary, and
a key that hard codes one signal's unit for two signals is that rule broken quietly. Renamed to
`rmse` and `mae` with the unit stated in the docstring, and all four ablations rerun, which cost
nothing because the reference was cached and confirmed the reproducibility above.

**Gates.** **606 tests pass** with everything selected, in 10 minutes 21 seconds, up from 579 at
P8. Twenty four of the new ones are `tests/test_ablations.py`, one per ablation core, the shared
metric definitions, the README status gate of ground rule 10 and the fragment against its
artifacts; the other three are the data card's own staleness gate acquiring a seventeenth
fragment. 587 pass with the slow markers excluded, which is the `test-full` CI job's selection,
and 458 under `not slow and not fullstack`, which is what the light stack `test-fast` job runs on
a machine with no torch and no artifact store. `ruff check` over the whole tree, `scripts/dash_lint.py` and
`scripts/check_file_sizes.py` over 256 tracked files are clean. `latexmk -pdf -halt-on-error`
builds `report/main.pdf` at **38 pages**, up from 33, with zero LaTeX warnings: no unresolved
reference and no overfull box.

**What P10 inherits.** A report that is structurally what build spec 20 asks for and a set of
ablations whose verdicts are computed rather than asserted. The open thread is the one the
ablations made hard to ignore: four independent models now beat the shipped pipeline at curve
level while losing at the peak, which is a finding about the reconstruction path, and the
outlook says so rather than the conclusion quietly not mentioning it.

## 2026-08-31, Phase P10: final QA, the README, and the release

**What shipped.** `scripts/readme_inject.py`, which injects every numeric claim in `README.md`
from the artifact manifests into twelve named marker pairs; `scripts/make_readme_media.py`,
which exports six result images from the report's own figure functions;
`tests/test_readme_consistency.py`, the staleness gate and the ban on numbers outside the
markers; `scripts/make_release.py`, which prepares a release and refuses to publish one;
`docs/RELEASE_CHECKLIST.md`, the definition of done swept item by item; `CONTRIBUTING.md`, which
build spec 19 asked for and the documentation set was missing; and a README rewritten from the
title down.

**The README was the phase, and the interesting decision in it was what not to put in.** The old
page carried a phase status table with a state column and a paragraph of measurements per row.
It had to go, and not because it was wrong: it was correct at every commit, kept correct by
three separate tests that read its rows and compared them against the manifests. It had to go
because it is the build order, which is the one thing a reader arriving at a release does not
need, and because a document that says "P10: not started" is a document that is wrong the moment
it ships. The phase history is in this file and the README points here in one line.

**Twelve marker pairs, not one.** Phase P0 left a single `BEGIN INJECTED RESULTS` pair at the top
of the README as a placeholder, and the obvious move at P10 was to fill it. That would have put
every number in the document into one slab under the introduction, which is the shape of a
generated page rather than of a page somebody reads. A reader wants the reliability sentence
under the reliability heading. So the injector owns twelve named regions (`badges`, `scope`,
`schematic`, `results`, `coverage`, `reliability`, `evidence`, `caveats`, `laws`, `quickstart`,
`versioning`, `gates`) and splices each one where it belongs; a stale block is now a stale block
with a name.

**The mermaid schematic is injected too, because nobody diffs a picture.** Its nodes carry the
campaign split, the grid size, the component counts, the measured coverage, the ablation count,
the withheld index count, the Monte Carlo size and the headline failure probability. Those are
exactly the numbers that rot in a diagram, because a diagram is the one artifact a reviewer
looks at rather than reads.

**The wall time is the only claim in the README that was deliberately made less precise, and
`docs/DEFECT_LOG.md` is why.** The regeneration budget the README quotes is the sum of ten stage
manifests' `wall_time_s`. Every one of those is rewritten whenever its stage is rerun while
reproducing its outputs bitwise, and the P4 to P7 determinism tests rerun five of them on every
full suite run. That is precisely the quantity that made the model card stale twice, once as a
wall time and once as a git commit, and the lesson written down after the second time was that a
byte gated document carries only quantities that change when the numbers change.

Rounding to the nearest minute would not have been enough. The measured total is 1043.9 s, 17.4
minutes, which is six seconds under the boundary that flips it to eighteen, against stage times
that vary by seconds between runs. So the README states the total rounded up to the enclosing
five minutes, next to the budget it is measured against, which is read out of build spec section
23 rather than typed. The exact per stage times are above in the P7 entry, where nothing gates
them. The brief for this phase asked for the seventeen minute figure on the page; the claim is
the same claim, and the precision is what was traded for a gate that keeps working.

**The half of the rule a byte gate cannot enforce.** A staleness gate proves the injected blocks
are current and proves nothing at all about the sentence above them, and a reader cannot tell
the two apart, because both are text on a page. So the test strips the marker pairs and the
fenced code blocks and refuses any digit carrying a unit, and any decimal with two or more
places, in what is left. It found three things while the page was being written: the repository
layout tree quoted a module's line count, the "built in eleven gated phases" pointer carried a
count that nothing regenerates, and the five binding laws quoted the censoring split in law 4.
The first two were reworded. The third was fixed properly, by reading the five law titles out of
build spec section 0.2 at injection time, so a law reworded in the specification and not in the
README is now a failing test rather than a quiet divergence.

That check needs no artifact store, which makes it the half of the gate that actually runs in
CI. The byte identity half needs the store, which no CI job has by design, so it skips there
exactly like the data card and model card gates it copies. `ci.yml` now says so in a comment
rather than leaving the next reader to work it out.

**The README images are exports, not a second set.** The temptation at this point in a project is
to draw prettier figures for the front page, and the result is a README that disagrees with the
document it advertises. `scripts/make_readme_media.py` runs `report/figures_src/make_figures.py`
unchanged with the raster hook of `ufem.plotting.style.save_figure` pointed at a scratch
directory and copies six PNGs out of it, so the image here and the figure in the PDF are the same
figure from the same artifacts on the same run. The only change needed in `src/` was to read the
preview resolution from `UFEM_FIG_PNG_DPI`, defaulting to the 200 it always used. Six images at
150 dpi: **852 KB total**, largest 258 KB, against a 5 MB gate. The vector PDFs the report uses
were rewritten byte identically on the same run, which is the determinism claim of the style
module holding.

**Three tests had to be rewritten rather than deleted.** `test_calibrate.py`,
`test_propagate.py` and `test_ablations.py` each carried a gate that read a row of the removed
status table and compared it against the manifests. Deleting them with the table would have
quietly dropped three ground rule 10 checks at the exact commit that claimed to strengthen ground
rule 10. They now read the injected page instead, still asserting against the stage artifacts
rather than against the generator, so a builder that reads the wrong key fails there even though
it would pass its own byte comparison forever. The ablation gate now counts all five ablations
rather than the four that belonged to P9, because the injected sentence covers all five.

**The definition of done sweep.** All twelve statements of build spec section 23, in
`docs/RELEASE_CHECKLIST.md`, with the evidence located for each. Ten are clean passes. Two are
passes with a named exception, and both exceptions are the specification anticipating a negative
result rather than a gap: item 4, where the surrogate beats all four baselines on every headline
scalar and loses the whole force curve to three of them, which the README's results table states
in its verdict cell and its fourth caveat repeats in prose; and item 5, whose published index
clause holds vacuously because every index was withheld, and whose agreement clause is the
diagnosed discrepancy the P6 gate explicitly allows, at 45 of 144 rows with the additivity
diagnosis written in three places.

**The tag is `v1.1.0`, and section 23 says `v2.0.0`.** The repository owner's decision, recorded
in `docs/DESIGN_DECISIONS.md` with its reason: the predecessor is `v1.0.0` and this is the same
project's second published state, not a second project, and no interface anybody depends on
changed. Nothing in the definition of done depends on the number. The one place it is enforced
mechanically is `scripts/make_release.py`, which refuses to print a release command while
`pyproject.toml` still declares a `.dev` version.

**Gates.** **629 tests pass** with everything selected, up from 606 at P9; the 23 new ones are
`tests/test_readme_consistency.py`. 610 pass under `not slow`, which is the `test-full` CI job's
selection, and 481 under `not slow and not fullstack`, which is what the light stack `test-fast`
job runs on a machine with no torch and no artifact store. `ruff check src tests scripts`,
`scripts/dash_lint.py` and `scripts/check_file_sizes.py` over the tracked tree are clean.
`latexmk -pdf -halt-on-error` builds `report/main.pdf` at **38 pages**, 1.87 MB, with zero
overfull boxes and zero LaTeX warnings.

**What the release inherits.** A repository whose front page is generated, whose every published
number resolves to a manifest, and whose two honest losses, the curve level reconstruction and
the withheld sensitivity indices, are on the front page rather than in a footnote. What it does
not have is Track B: the corrected Abaqus campaign of build spec section 14 is gated on solver
access and none of it is claimed here. Everything in this repository is conditional on an
inherited campaign whose material card was frozen, and the report's outlook says what would
change if it were not.
