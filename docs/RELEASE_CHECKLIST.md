# Release checklist: the definition of done, item by item

Build spec section 23 lists twelve statements that must be true and machine checkable before
the overhaul is taggable. This document walks all twelve, states the verdict, and points at the
evidence rather than asserting it. Where an item is not a clean pass it says so in the same
words it would use for a pass, because a checklist that only records successes is a checklist
that was written after the decision.

Swept on 2026-08-31 at phase P10, against config hash `d35707765c5f` and the artifact store it
names. Two conventions: **PASS** means the statement is true and something checks it; **PASS
with a named exception** means the statement is true in the form the specification allows for a
negative result, and the exception is published rather than buried.

| # | Statement | Verdict |
|---|---|---|
| 1 | Full regeneration, bit for bit, under half an hour | PASS |
| 2 | All tests green, CI red proofs on record | PASS |
| 3 | Functional bands and scalar intervals cover at the nominal level | PASS |
| 4 | The surrogate beats all four baselines, or the README says which it does not | PASS with a named exception |
| 5 | Chaos and Gaussian process indices agree, every published index passed its gate | PASS with a named exception |
| 6 | Every reliability number carries an error, a bound and an out of domain fraction | PASS |
| 7 | Censoring documented, completion model shipped, the UI grays the corner | PASS |
| 8 | The report compiles in CI, no typed numbers, limitations complete | PASS |
| 9 | UFEM Lab runs, passes the latency test, and the GIF is current | PASS |
| 10 | No dash, no oversized file, no venv, no absolute path, no bare except, no banned identifier | PASS |
| 11 | Both logs populated, every defect has its regression test | PASS |
| 12 | The predecessor's numbers are nowhere repeated, and are labeled invalid | PASS |

One deviation from the specification's own wording is recorded separately at the end: the tag
is `v1.1.0`, not the `v2.0.0` that section 23 names.

---

## 1. `ufem run all` regenerates every figure, table and README number bit for bit, in under thirty minutes

- [x] **PASS.**

The measurement is P7's, and it is still the one on record: the whole pipeline, ingest through
propagate, ran in 1019.9 s, about 17 minutes, on the reference machine
(`docs/ENGINEERING_LOG.md`, phase P7, "Measured wall times"). The largest single term is the P4
fold harness at 576.1 s, which recomputes the registration and every reduction basis inside each
of its ten folds and is the price of fold honesty rather than an inefficiency. P10 re verified
the sum from the stage manifests: `scripts/readme_inject.py` reads `wall_time_s` out of each of
the ten stage manifests and totals them, which is where the README's regeneration claim comes
from.

The README states that total rounded up to the enclosing five minutes rather than to the minute,
and that is deliberate. `docs/DEFECT_LOG.md` records the same quantity making the model card
stale twice, once as a wall time and once as a git commit: a byte gated document must not carry
a number that moves without a measurement moving. The exact per stage times live in the
engineering log, which is written by hand and gated by nothing.

Bit for bit is checked separately from the wall time, by `tests/test_p3_determinism.py` through
`tests/test_p7_determinism.py`, which rerun each stage and compare the artifact digests, and by
the staleness gates on the three generated documents (`tests/test_data_card.py`,
`tests/test_model_card.py`, `tests/test_readme_consistency.py`), which regenerate and compare
byte for byte.

## 2. All tests green, including property, golden, manufactured, seeding and grep law tests, with CI red proofs on record

- [x] **PASS.**

The suite is green at this commit; the counts are in the P10 entry of `docs/ENGINEERING_LOG.md`.
Every family the item names is present and is named after what it enforces:
`tests/test_laws.py` (the grep laws), `tests/test_golden_audit.py` (the pre build audit values to
1e-9), `tests/test_manufactured.py` (the manufactured solution), `tests/test_p3_determinism.py`
through `tests/test_p7_determinism.py` (seeding and bitwise reproduction), and property tests
under Hypothesis in `tests/test_register.py`, `tests/test_grid.py` and
`tests/test_conformal_functional.py`.

The CI red proofs are the four planted faults of build spec 18.1, each pushed on its own, each
recorded with its run URL and the specific job it stopped, in the "CI proof of failure gate"
entry of `docs/ENGINEERING_LOG.md`. The fifth run in that table is the green one with all four
faults reverted.

One test in the suite has a verdict that depends on what else the machine is doing, and it is
named here rather than left to be discovered: `test_the_fit_budget_of_build_spec_10_3_was_met`
asserts on a wall clock with a 10 percent margin against a fit whose run to run spread is about
14 percent. It went red during this sweep's first full run, under concurrent load, and green at
58.03 s with nothing else running, with every artifact reproducing its digest bitwise both
times. The budget was not widened. The P5 and P10 entries of the engineering log carry both
measurements. Read that assertion as a regression alarm and recheck it on an idle machine before
acting on it.

## 3. Simultaneous ninety percent functional bands and scalar jackknife plus intervals have leave one out coverage whose Wilson interval contains 0.90

- [x] **PASS.**

Source: `experiments/results/calibrate/<config hash>/calibration.json`, the `gate` block. Six
counted criteria, the simultaneous sup norm bands on the force and damage curve families and the
jackknife plus intervals on each of the four headline scalars, all measured 0.9040 leave one out
coverage with a 95 percent Wilson interval of [0.8550, 0.9377], which contains the nominal 0.90.
The seventh criterion is the probability integral transform outer decile mass on the softening
branch, which is not a counted proportion and has no Wilson interval; it passed at 0.1122 against
a ceiling of 0.35.

The calibration figures are in the report: `fig_calibration.pdf`, `fig_coverage_sweep.pdf`,
`fig_conformal_band.pdf`, `fig_pit_heatmap.pdf` and `fig_crps_skill.pdf`, all in
Section "Calibration" of `report/main.tex`, and the coverage numbers reach the prose through
generated fragments only.

## 4. The surrogate beats all four baselines out of sample on the headline quantities, or the README says which it does not beat and the simpler model ships

- [x] **PASS with a named exception, which is the case the item exists for.**

On the four headline scalars the gate passes cleanly:
`experiments/results/validate/<config hash>/baselines.json` records `gate.passed = true` with an
empty `failing_targets`, and the surrogate beats climatology, linear, nearest neighbour and
quadratic chaos on peak load, displacement at peak, initial stiffness and absorbed energy.

The exception is the whole curve. At the median relative L2 on the force curves, three of the
four baselines reconstruct better than the registered and reduced surrogate: linear, nearest
neighbour and quadratic chaos. The README's results table carries the surrogate's median against
the best rival's and names all three in the verdict cell, and the fourth caveat under "What those
numbers do not say" states it in prose. The report says the same in its ablations and limitations
sections, and `docs/ABLATIONS.md` carries the direct curve models that make the same point.

The simpler model does not ship, and the reason is stated rather than assumed: the quantity the
reliability analysis thresholds is the peak load, which comes from its own Gaussian process and
is never read off a reconstructed curve, and on that quantity the surrogate wins. A reader who
wants the curve and nothing else is told, in the README and in the report, which model to prefer.

## 5. Chaos and Gaussian process sensitivity indices agree within stated uncertainty, and every published index passed its Q2 gate

- [x] **PASS with a named exception.**

The second clause holds outright, and holds in the strongest possible form: nothing was
published. All 24 sparse chaos expansions failed the publication gate of build spec 12.1, zero
reached the value threshold of 0.95 and zero the ranking threshold of 0.80, so
`sensitivity.json` records `gate_outcome = not_published` and no Sobol index value and no input
ranking is published anywhere in this project. The README, the model card, the report and the
dashboard all say so, and the dashboard's sensitivity panel draws no bar at all.

The first clause does not hold as agreement, and the specification's P6 gate anticipated that:
"PCE and GP indices agree within uncertainties **or the discrepancy is diagnosed in writing**".
The chaos index lies inside the Gaussian process posterior 90 percent interval on 45 of 144
assessed rows. The diagnosis is written, in the P6 entry of `docs/ENGINEERING_LOG.md`, in the
sensitivity section of the report, and in the artifact itself: the fitted chaos expansions are
almost perfectly additive, with a median interaction share of about 5.6e-17 across the target
set, while the posterior realizations carry a median interaction share of 0.31, so the two
families are disagreeing about interaction structure at a sample size where neither can resolve
it. That is itself the finding, and it is why the gate withheld everything.

## 6. Every reliability number carries a Monte Carlo standard error, a surrogate aware bound, and the out of domain mass fraction

- [x] **PASS.**

Source: `experiments/results/propagate/<config hash>/propagation.json`. Every one of the three
limit states carries `pf_point`, `pf_standard_error`, the Wilson bracket, `pf_conservative` (a
failure counted whenever the calibrated interval crosses the threshold), `pf_inside_domain` and
`n_inside_domain`; the run level `validity.out_of_domain_fraction` is 0.46648. The resolvable
floor of 1e-4 is imposed by the 198 training runs rather than by the Monte Carlo size, and a
probability below it prints as a bound rather than as a number, in the report
(`ufem.propagate.format_probability`), in the model card, and in the README.

The one non resolvable limit state, the residual capacity ratio, counted zero failures in
100000 draws and is reported as below the resolution of the sample rather than as zero.

## 7. The censoring bias is documented in the data card, the completion model ships, and the UI grays out the censored corner

- [x] **PASS.**

`docs/DATA_CARD.md` is generated from the artifact store by `scripts/make_data_card.py` and
carries the 198 / 202 / 0 extraction split, the per input bias tables with their chi squared
tests, the quartile failure rates, the completion model's cross validated performance and the
validity domain. `tests/test_data_card.py` gates it byte for byte.

The completion model ships as an artifact (`completion_model.pkl` and `completion_model.json` in
the audit stage) and is consulted through one contract, `ufem.validity.in_validity_domain`, by
every downstream stage. UFEM Lab grays a prediction whose inputs fall outside the domain and
names the corner it fell into; the behavior is tested in `tests/test_ui.py`.

## 8. The report PDF compiles in CI, contains no hand typed numbers, and its limitations section names the censored design, the frozen fracture energy, the fixed model parameters and the Pf floor

- [x] **PASS.**

The report builds under `latexmk -pdf -halt-on-error` locally and in `.github/workflows/report.yml`
on a TeX Live container with no Python and no artifact store, which is possible only because
every fragment under `report/tables/` and every figure under `report/figures/` is committed and
generated. No number is typed into the prose: values reach it through `\input` fragments and
through the macros in `report/tables/macros.tex`, and `tests/test_data_card.py` regenerates every
fragment and compares it byte for byte.

The limitations section names all four required items, each as its own paragraph: the censored
design, the frozen fracture energy of the inherited campaign, the model parameters held fixed,
and the failure probability floor. It names three more that the specification did not ask for.

## 9. UFEM Lab runs with `ufem lab`, passes the latency test, and the README GIF shows the current UI

- [x] **PASS.**

The five panels of build spec 15 serve over the artifact store. The latency budget is 50 ms from
a slider move to a repaint; the measurement is 32.8 ms median server side over 100 seeded slider
positions, and 59 ms end to end in a headless browser, both in the P8 entry of
`docs/ENGINEERING_LOG.md`. The test is in `tests/test_ui.py` and skips with a named reason when
no browser is installed, since `pip install -e .[dev]` does not bring one.

`docs/media/ufem_lab.gif` is 0.81 MB, recorded from the running dashboard by
`scripts/capture_ui_gif.py`, and is embedded at the top of the README.
`tests/test_laws.py::test_the_readme_gif_is_committed` and
`tests/test_readme_consistency.py::test_the_readme_shows_the_dashboard_gif_first` both check it.

## 10. No em or en dash anywhere; no file over 5 MB; no venv artifact; no absolute path; no bare except; no banned identifier

- [x] **PASS.**

`scripts/dash_lint.py` is clean over the tree, covering the dashes, the banned identifiers of
build spec 5.1, seeded global RNG, bare except, distribution construction outside `config.py`,
and the computed constant ban inside the dashboard package. `scripts/check_file_sizes.py`
reports every tracked file under the limit. Both run in the `lint` job of `.github/workflows/ci.yml`
and again as tests in `tests/test_laws.py`, which also checks that no venv or interpreter is
tracked and that `src/` never references a quarantined path. Ruff covers bare except a second
time through E722.

The README carries its own dash assertion in `tests/test_readme_consistency.py`, so a README only
change cannot slip one through on a branch where the lint job is skipped.

## 11. `docs/ENGINEERING_LOG.md` and `docs/DEFECT_LOG.md` are populated and every defect entry has its regression test

- [x] **PASS.**

The engineering log has one entry per phase, P0 through P10, plus two entries for CI incidents
between phases. The defect log has seven entries, each with the evidence that exposed it, the fix
commit and the named regression test, and each of those tests exists and is green: the validity
domain fixture tests, the report figure tracking check, the SRVF round trip test, the runner cache
tests, the surrogate manifest fixture tests, the sensitivity import guards, the shaded polygon
extent test, and the model card's no commit test.

## 12. The predecessor's published numbers are nowhere repeated as results; where mentioned they are labeled invalid with the section 5 evidence

- [x] **PASS.**

The predecessor's numbers appear in exactly three places, and each one labels them. The README's
versioning block says its published metrics are invalid and points at section 5 of the build
specification. `v1_legacy/README.md` says the same at the head of the frozen tree. The report's
abstract states that this document replaces all previously circulated summary numbers for this
campaign, and names the injected variance as the reason.

Two of the predecessor's thresholds are quoted inside the propagation artifact's own
justifications, both as counterexamples: the two sigma peak load threshold and the damage
threshold of 0.9591, which is above the reachable maximum of the campaign and therefore has a
failure probability of exactly zero for reasons that have nothing to do with the beam. Neither
is reported as a result of this project.

---

## The one deviation from the specification: `v1.1.0`, not `v2.0.0`

Build spec section 23 opens with "`v2.0.0` is taggable when ...", and every statement in it was
swept against this tree. The tag is nevertheless `v1.1.0`, by the repository owner's decision:
the published artifact is the continuation of one project rather than the second of two, and a
major version bump would imply an interface that consumers depend on and that changed. The
decision is recorded with its date in `docs/DESIGN_DECISIONS.md`. Nothing in the definition of
done depends on the number, and the version string reaches the badge, the README's versioning
block and `ufem doctor` from `pyproject.toml`, which declares it once.

## What the release still needs, and who does it

P10 prepared the release and did not perform it. Three things are left, in this order, and all
three belong to whoever tags:

1. **Merge `phase/p10-release` into `main`.** `scripts/make_release.py` refuses to run anywhere
   else, because build spec 21 tags from `main` only.
2. **Drop the development suffix, in a commit of its own.** `pyproject.toml` and
   `src/ufem/__init__.py` both declare `1.1.0.dev0`; the release is `1.1.0`. Rerun
   `ufem doctor` afterwards so the resolved version matrix at the bottom of
   `docs/DESIGN_DECISIONS.md` agrees, and `python scripts/readme_inject.py` so the README's
   versioning block does. `scripts/make_release.py` refuses to print a release command while the
   suffix is still there, which is the only place this is enforced mechanically.
3. **Run `python scripts/make_release.py` and then the command it prints.** The script builds the
   PDF, verifies the branch, the tree, the three generated documents and both lints, and prints
   the `gh release create` line with `report/main.pdf` attached. It does not run `gh`, and it has
   no override flag: tagging is the one step in this project that rerunning a stage cannot undo.

## What is deliberately not claimed at this release

Track B is not started. The corrected Abaqus campaign of build spec section 14, with fracture
energy coupled to strength, the mesh convergence study, the rerun of the 202 failures and the
model parameters promoted to inputs, is gated on solver access and none of it is in this
release. Everything in this repository is conditional on an inherited campaign whose material
card was frozen, and the report's outlook section says what would change if it were not.
