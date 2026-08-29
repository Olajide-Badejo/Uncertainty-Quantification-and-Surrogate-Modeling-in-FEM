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
