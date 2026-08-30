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

Both entries are the same underlying mistake in two places: a gate was verified on a
developer machine that happened to have the artifact store, so neither the test suite nor the
report build was ever checked against the clean checkout CI actually gets. Both were
reproduced locally by cloning the repository to a scratch directory, which has no
`experiments/results/`, and both fixes were verified there before being committed.

Phase P0 built the scaffolding and the gates; the gates caught three issues in inherited and
scaffolding files on their first run (two oversized tracked CSVs, two en dashes in a figure
script, and the dash linter matching its own constants), but those were caught before any
commit shipped them, so they are recorded in `docs/ENGINEERING_LOG.md` rather than here.
