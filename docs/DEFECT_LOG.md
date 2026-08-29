# Defect log

Ground rule 7: every bug fixed gets a regression test that fails before the fix and passes
after, with both runs shown in the phase report. This table is the index of those. Dated
entries, append only, never deleted.

A defect belongs here when it was a genuine bug in 2.0 code: wrong output, a silent
fallback, a broken invariant. Findings about the inherited v1 tree are not defects of this
project and live in section 5 of `docs/BUILD_SPEC.md` instead.

| Date | Defect | Evidence | Fix commit | Regression test |
|---|---|---|---|---|
| | | | | |

No entries yet. Phase P0 built the scaffolding and the gates; the gates caught three issues
in inherited and scaffolding files on their first run (two oversized tracked CSVs, two en
dashes in a figure script, and the dash linter matching its own constants), but those were
caught before any commit shipped them, so they are recorded in
`docs/ENGINEERING_LOG.md` rather than here.
