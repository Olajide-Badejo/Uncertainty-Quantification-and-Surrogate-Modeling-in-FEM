# Defect log

Ground rule 7: every bug fixed gets a regression test that fails before the fix and passes
after, with both runs shown in the phase report. This table is the index of those. Dated
entries, append only, never deleted.

A defect belongs here when it was a genuine bug in 2.0 code: wrong output, a silent
fallback, a broken invariant. Findings about the inherited v1 tree are not defects of this
project and live in section 5 of `docs/BUILD_SPEC.md` instead.

| Date | Defect | Evidence | Fix commit | Regression test |
|---|---|---|---|---|
| 2026-08-30 | Six `TestTheContract` tests read the validity domain without depending on the fixture that skips when the audit stage has not run, so a gitignored artifact store failed the suite instead of skipping it | CI run 33281932279 on the P2 merge, red on ubuntu: `6 failed, 134 passed, 52 skipped`, every failure `ValidityDomainUnavailable` | `15cec71` | `tests/test_validity.py::TestTheContract`, run against a bare clone with no `experiments/results/` |
| 2026-08-30 | `.gitignore` ignored `report/figures/*.pdf` while `report.yml` claimed the LaTeX build needs no artifact store because the figures are committed, so the TeX Live container had no figures | CI run 33281932306, red: ``LaTeX Warning: File `fig_E_collinearity.pdf' not found``, `Fatal error occurred, no output PDF file produced`, exit 12 | `5fab5ef` | `latexmk -pdf -halt-on-error` on a bare clone, 10 pages; `scripts/check_file_sizes.py` covers the 5 MB rule on the newly tracked figures |
| 2026-08-30 | `ufem.register.warp_tangent_vectors` called `fdasrsf.utility_functions.SqrtMean` with the library's undocumented `smooth=True` default, which silently smoothed the warps before differentiating them and was not matched by the project's own unsmoothed inverse map | `pytest tests/test_surrogate.py::TestSquareRootSlopeRepresentation::test_the_round_trip_recovers_the_family_it_came_from`, red: `IndexError: index 501 is out of bounds for axis 0 with size 501` inside `fdasrsf`'s own Karcher mean iteration; 14 seeded synthetic families measured 6 of 14 non convergent and the rest at 1 to 5 percent round trip error under `smooth=True`, all 14 convergent at 1.5 to 4.7e-3 under `smooth=False` | `1ab136c` | the same test, green; `tests/test_p4_determinism.py` and `tests/test_register.py`/`tests/test_reduce.py` (44 passing, unaffected) as the no regression check on the amplitude block and the registration ablation |
| 2026-08-30 | `ufem.runner.is_cache_hit` recomputed a stage's cache key from the input hashes that stage's own manifest recorded, instead of hashing the upstream artifacts as they are on disk, so an upstream artifact that changed after the stage ran was served stale without `--force` | `pytest tests/test_runner.py -q`, red before the fix: `3 failed, 4 passed`, with `test_a_changed_upstream_artifact_is_not_a_cache_hit` asserting `False` where the runner returned `True` after one byte was appended to the upstream `load_displacement.parquet`; green after: `7 passed` | `a4ce37b` | `tests/test_runner.py::test_a_changed_upstream_artifact_is_not_a_cache_hit`, with `::test_a_deleted_upstream_artifact_is_not_a_cache_hit` and `::test_every_implemented_stage_declares_its_inputs` alongside it |
| 2026-08-30 | Two `TestTheFittedArtifact` tests read the surrogate stage's manifest while taking `repo_root` instead of the fixture that skips when the artifact store is absent, so the clean checkout CI job raised `FileNotFoundError` where every other test in the class skipped | CI run 33289864780 on the P4 merge, red on ubuntu in `test-full (ubuntu, editable install)`: `2 failed, 221 passed, 83 skipped`, both failures `FileNotFoundError: no manifest at .../experiments/results/surrogate/6999093f.../manifest.json` | `7a77f54` | the same two tests, now on the `surrogate_manifest` fixture: they skip with no artifact store and run locally, where `pytest tests/test_surrogate.py -q` is `22 passed` |
| 2026-08-30 | `tests/test_p6_determinism.py` had no `importorskip` guard for the sensitivity stack, so the light stack `test-fast` CI jobs (no SALib, no OpenTURNS installed by design) hit `ModuleNotFoundError` inside `ufem.sensitivity.saltelli_design` instead of skipping; third instance of the gate-verified-only-on-a-full-machine class | CI run 33299426619 on the P6 merge, red on both `test-fast (ubuntu-latest)` and `test-fast (windows-latest)`: `ModuleNotFoundError: No module named 'SALib'` at `src/ufem/sensitivity.py:813` in `test_the_saltelli_design_depends_only_on_the_spawned_seed` | see this branch | the same module, now guarded by `pytest.importorskip("SALib")` and `pytest.importorskip("openturns")` at import time, verified by running the suite with those modules absent from a scratch venv |
| 2026-08-30 | `ufem.plotting.propagation.qoi_densities` inferred which side of a limit state threshold was the failure region from where the threshold fell relative to the sample median, which is backwards for a below type limit state, so all three panels of `fig_qoi_densities.pdf` shaded the safe side | the rendered figure: the peak load panel shaded everything above its 33.2 kN threshold and the damage panel everything below its 0.93 threshold, in both cases the region where the member passes | `551c659` | `tests/test_propagate.py::TestTheFailureRegionIsShadedOnTheFailingSide`, which reads the shaded polygon's extent rather than the code path: 2 failed with the heuristic restored, 3 passed after |
| 2026-08-30 | `scripts/make_model_card.py` printed the git commit and branch the surrogate artifact was fitted at, both read from its manifest. Rerunning any stage rewrites that manifest with the current commit while reproducing every output bitwise, so the P4 to P7 determinism tests made the byte gated model card stale without a single measurement changing: the wall time defect of the data card in a second form | `pytest tests -q` (the slow markers included) reran register, surrogate, calibrate, sensitivity and propagate in place; `python scripts/make_model_card.py` then rewrote `docs/MODEL_CARD.md` with `2ceec0bd2dbf` replaced by `6802c816e9f6` and `phase/p7-propagation` by `phase/p8-ufem-lab`, two lines changed and no number among them | see this branch | `tests/test_model_card.py::test_the_card_embeds_no_git_commit`, which matches a bare 40 hex character token and fails on the old card; the card now carries the config hash and the surrogate record's own digest, which move only when the model does, and `::test_the_provenance_hashes_are_the_ones_that_move_only_when_the_model_does` pins them to the manifest |

The shading defect is the first in this project that no test could have caught without
rendering the figure and looking at it, and it shipped inside a commit before the look
happened. The regression test now reads the vertices of the shaded polygon, which is the
closest a test gets to looking. The lesson recorded rather than the fix: a figure whose
meaning depends on which side of a line is filled needs its own assertion, and the direction
should come from the declaration that already carries it instead of being inferred from the
data.

The cache defect was found while reading `runner.py` at the start of P5 and fixed before the
phase's own work began, because a cache that cannot notice an upstream change is exactly the
kind of thing that makes a calibration stage read last week's surrogate. The comparison is now
between the hashes the manifest recorded and the same files hashed again at check time, and
every stage exposes a `declared_input_hashes` for it; a stage that did not would raise rather
than be trusted. The cost is one hash of each declared input per cache check, milliseconds on
this campaign's artifacts.

The first two entries are the same underlying mistake in two places: a gate was verified on a
developer machine that happened to have the artifact store, so neither the test suite nor the
report build was ever checked against the clean checkout CI actually gets. Both were
reproduced locally by cloning the repository to a scratch directory, which has no
`experiments/results/`, and both fixes were verified there before being committed.

Phase P0 built the scaffolding and the gates; the gates caught three issues in inherited and
scaffolding files on their first run (two oversized tracked CSVs, two en dashes in a figure
script, and the dash linter matching its own constants), but those were caught before any
commit shipped them, so they are recorded in `docs/ENGINEERING_LOG.md` rather than here.

The model card defect is the wall time lesson arriving a second time and being recognized one
gate later than it should have been. The data card had already banned an embedded wall time for
exactly this reason and its test says so in as many words; the commit is the same category of
quantity, one that moves without a measurement moving, and it went into the model card anyway
because provenance felt like the opposite of noise. It was caught by running the full suite and
then regenerating, which is the sequence that reproduces it, rather than by reading the code.
The rule generalized and now tested: a byte gated document carries only quantities that change
when the numbers change.
