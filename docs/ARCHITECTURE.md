# Architecture

## The pipeline

```
probabilistic_model.yaml + pipeline.yaml        (one config, Pydantic validated, hashed)
  -> ingest      raw CSVs -> deduplicated, typed Parquet curve store
  -> grid        RF2 and DAMAGEC on the common 201 point displacement grid
  -> audit       400 way validity classification, censoring stats, completion GP classifier
  -> register    landmarks -> arc length -> SRVF elastic registration -> amplitude + warp
  -> reduce      fPCA on registered amplitude and on warp tangent space; scalar QoI table
  -> surrogate   Matern 5/2 ARD GP per retained score and per scalar QoI (CPU, seconds)
  -> validate    fold honest LOO metrics, baselines, ablations; the one harness
  -> calibrate   closed form LOO jackknife+ conformal (sigma normalized); functional sup norm
                 bands; coverage, PIT, CRPS, NLPD, PVA; the build spec 11.5 gate
  -> sensitivity sparse LARS PCE (corrected LOO Q2 gate) + GP posterior Sobol + functional indices
  -> propagate   1e5+ MC through calibrated surrogate; limit states; Pf with bounds; analytic cross check
  -> report      figures + tables + LaTeX -> latexmk -> PDF
  -> ui          UFEM Lab dashboard reading the same artifact store
```

This is the execution order `ufem.runner.STAGES` declares, and it differs from build spec 7.1
in two places, both recorded with their reasoning in `docs/DESIGN_DECISIONS.md`: `grid` runs
before `audit`, because the audit's importance weighting study reweights the QoI table `grid`
extracts, and `validate` runs before `calibrate`, because the conformal calibration is built on
out of sample residuals and `validate` is the stage that produces them honestly.

At P0 the runner registers all eleven stages and none of them are implemented. Asking for
one that has not been built yet gets a `NotImplementedError` naming the phase that will add
it, rather than a silent no op. That distinction is the whole point: v1 shipped four scripts
that had never run and could not have, and nothing in the pipeline noticed.

## The artifact contract

Every stage is a pure function from a pair of hashes to a directory. The inputs are the
SHA-256 of the fully resolved configuration (both YAML files, canonical JSON, sorted keys)
and the SHA-256 of each artifact the stage consumes. The output is
`experiments/results/<stage>/<config hash>/`, containing whatever files the stage produces
plus exactly one `manifest.json`.

Stages never import each other's internals. If `reduce` needs what `register` produced, it
reads the files, and it learns which files by their hashes in the manifest. This is what
makes the pipeline restartable in the middle and what makes a stale artifact impossible to
serve accidentally: there is no in memory state that can disagree with what is on disk.

The manifest records the config hash, every input hash, every output file with its name,
size, and hash, the root `SeedSequence` entropy, the resolved version of every package in
the core stack, the git commit with a dirty flag, the hostname, the wall time, and an ISO
timestamp. That list is chosen so that any number in the report can be walked backwards:
from the figure to the table to the artifact to the manifest to the commit and the exact
package versions that produced it. Binding law 5 says no number appears anywhere unless
that walk terminates, and the manifest is what makes the walk possible.

The cache key is a separate hash over four things: the stage name, the SHA-256 of the
stage's own source file, the config hash, and the input hashes. The runner computes it,
compares it to the key stored in the existing manifest, and skips the stage on a match,
printing `[cache hit] <stage>`. Hashing the source file is deliberate. A cache keyed only
on data would happily serve last week's results after I changed the algorithm, which is a
subtle enough failure that I would rather pay to rehash a few kilobytes of Python on every
run than ever debug it.

Cache validity is checked, not assumed. Before declaring a hit the runner reruns
`verify_manifest`, which rehashes every declared output and compares it to the recorded
digest. A file edited or truncated behind the pipeline's back is a miss, not a hit.
`--force` reruns regardless.

## Readers, and why they are not stages

Five things read the artifact store without writing into it: `scripts/make_data_card.py`,
`scripts/make_model_card.py`, `scripts/readme_inject.py`, `report/figures_src/make_figures.py`
(and `scripts/make_readme_media.py`, which is that script with its raster hook enabled), and
UFEM Lab (`ufem lab`, `src/ufem/ui/`). None of them is a stage, because none of them produces
an artifact another stage consumes, and giving them cache keys would have meant a document
could be served stale on a hit.

They are held honest a different way. The three document generators write files that a
staleness test regenerates and compares byte for byte, so a document that has drifted from the
pipeline is a failing test. The dashboard writes nothing, so it is held to binding law 5 directly:
`dash_lint.check_ui_constants` parses every module under `src/ufem/ui/` and rejects any numeric
literal that is neither structurally trivial nor a presentation constant declared in
`ui/layout.py`. Anything else a panel displays has to have been read from an artifact.

One consequence shaped the propagate stage. The reliability panel's threshold slider recomputes
a failure probability, and recomputing it in the dashboard would have published a number no
manifest covers, so the stage persists the Monte Carlo rows the recount needs
(`mc_subsample.parquet`) and the panel calls the stage's own `recompute_limit_state` on them.
A reader that needs to compute is a stage that did not write enough.

## Failure behavior

The runner is a pure batch CLI. There is no `input()` anywhere in it, no interactive
prompt, and no path on which a failure results in a zero exit code. A stage that raises
prints `[failed] <stage>: <type>: <message>` to stderr and the process exits nonzero,
stopping the remaining stages. Ground rule 8 extends that downward: a missing input file, an
absent manifest, or an output whose hash no longer matches raises with a named diagnostic
rather than degrading to a default. v1 published an APPROVED verdict computed on its own
training data because a missing split file silently fell back to every job it could find.
Nothing in this tree is allowed to do that.
