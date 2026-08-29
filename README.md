# UFEM 2.0: Calibrated Surrogate Modeling and UQ of a Softening RC Beam

This project builds a reproducible uncertainty quantification pipeline for a reinforced
concrete beam with material softening. From a completed campaign of Abaqus concrete damaged
plasticity simulations it constructs a functional surrogate of the whole load displacement
and damage evolution response, calibrates the predictive uncertainty against held out data
rather than asserting it, quantifies the global sensitivity of the response to the
independent input random variables three separate ways that have to agree, estimates failure
probabilities with stated error bars, and exposes the result through a local dashboard, a
compiled LaTeX report, and this repository, where every published number is regenerable from
a committed manifest whose hashes resolve to real files and a real commit.

The governing rule, and the reason the project was rebuilt rather than patched: no number
leaves here unless the pipeline can regenerate it from raw inputs, and no uncertainty is
reported unless it was propagated rather than manufactured.

<!-- BEGIN INJECTED RESULTS -->

Results are injected here from the artifact manifests at Phase P10. Nothing is claimed yet.

<!-- END INJECTED RESULTS -->

## Status

| Phase | What it delivers | State |
|---|---|---|
| P0 | Scaffold, config, manifests, runner, lint, CI | Complete |
| P1 | Ingest and common grid | Complete at this commit |
| P2 | Audit, censoring model, first compiled report | Not started |
| P3 | Registration and reduction | Not started |
| P4 | Gaussian process surrogate, baselines, validation | Not started |
| P5 | Conformal calibration, scalar and functional | Not started |
| P6 | Sensitivity | Not started |
| P7 | Propagation and reliability | Not started |
| P8 | UFEM Lab dashboard | Not started |
| P9 | Ablations and complete report | Not started |
| P10 | Final QA, README injection, release | Not started |

The full phase definitions and their gates are in `docs/BUILD_SPEC.md` section 22.

## Quick start

Requires CPython 3.14 on Windows or Linux.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\ufem doctor
.venv\Scripts\ufem run all
```

`ufem doctor` prints the resolved version matrix, the torch build and device, and the
SHA-256 of the configuration, then records that matrix in `docs/DESIGN_DECISIONS.md`. Run it
first; if it does not agree with what you expect, nothing downstream is worth reading.

`ufem run all` walks the stages in order. A stage whose cache key is unchanged prints
`[cache hit]` and is skipped; `--force` reruns it. Stages that a later phase will implement
raise with the phase named, so at this commit `run ingest` and `run grid` do real work and
`run all` then reports that `audit` arrives in P2 and exits nonzero. That is the intended
behavior, not a failure of the install.

Ingest and grid read the two raw campaign CSVs from `legacy_salvage/data/`. Those are 123 MB
and are deliberately not tracked by git; the ingest manifest records the SHA-256 of each at
its pinned location. If they are absent, the stage raises naming the file rather than
producing an empty result. The gridded outputs are committed under `data/processed/`, so the
repository still reads without them.

## Repository layout

```
UFEM_2.0/
  pyproject.toml            single source of pinned dependencies, dev extras, ufem entry point
  configs/
    probabilistic_model.yaml  THE distributions. Nothing else in the repo declares one.
    pipeline.yaml             grids, thresholds, kernel settings, MC size, limit states, paths
  src/ufem/
    config.py               Pydantic models, the feature contract, config hashing
    manifest.py             content addressed artifact store
    runner.py               ufem run <stage>|all and ufem doctor, pure batch
    ingest.py               raw CSVs to deduplicated, typed Parquet
    grid.py                 both signals on the common displacement grid, plus the scalar QoIs
  data/
    audit_reference/        committed golden values from the pre build audit, P1 gates on these
    processed/              the small pipeline outputs, committed for self containment
    quarantine/             what is deliberately not used, and why
  legacy_salvage/           read only inputs carried over from v1, never edited in place
  v1_legacy/                the frozen v1 pipeline, release v1.0.0, read only
  experiments/results/      artifact store, <stage>/<config hash>/manifest.json
  report/                   main.tex plus figures and tables written only by the pipeline
  tests/                    contract, property, golden, manufactured, integration
  scripts/                  dash_lint.py, check_file_sizes.py
  docs/                     BUILD_SPEC, ARCHITECTURE, DESIGN_DECISIONS, ENGINEERING_LOG, DEFECT_LOG
  .github/workflows/        ci.yml
```

## Versioning

- **v1.0.0** is the frozen v1 pipeline, preserved in `v1_legacy/`. Its published metrics are
  invalid, for the reasons set out in section 5 of `docs/BUILD_SPEC.md` and summarized in
  `v1_legacy/README.md`. It is kept as a record of the simulation campaign and as the
  forensic archive for the corrected campaign of Track B, not as a source of results.
- **v1.1.0** is the 2.0 overhaul, built from scratch on the same inherited data. The
  in progress version is `1.1.0.dev0`, declared in `pyproject.toml` and reported by
  `ufem doctor`.

## Contributing gates

A change lands only when the dash and banned identifier lint passes, no tracked file exceeds
5 MB, `ruff check src tests scripts` is clean, and the test suite passes. All four run in
CI on every push to `main` and to any `phase/**` branch, and on every pull request.

## License

MIT. See `LICENSE`.
