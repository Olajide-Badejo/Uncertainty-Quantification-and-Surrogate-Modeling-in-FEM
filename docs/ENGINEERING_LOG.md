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
