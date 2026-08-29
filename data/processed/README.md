# data/processed

These four Parquet files are pipeline outputs, committed so the repository is self
contained. A fresh clone can read the gridded campaign and the scalar quantities of interest
without the raw CSVs, which are large and deliberately untracked.

| File | What it holds |
|---|---|
| `rf2_grid.parquet` | Reaction force in N for every job, interpolated onto the common displacement grid |
| `damage_grid.parquet` | The compression damage scalar for every job on the same grid |
| `qoi.parquet` | The scalar quantities of interest, one row per job, joined to its design inputs |
| `design.parquet` | The Latin hypercube design, typed and validated |

Nothing here is edited by hand. Regenerate all four with

```powershell
.venv\Scripts\ufem run ingest
.venv\Scripts\ufem run grid
.venv\Scripts\python scripts\export_processed.py
```

which copies them out of the artifact store under `experiments/results/`. Each is a byte
copy of a stage output whose SHA-256 is recorded in that stage's `manifest.json`, so a
committed file that has drifted from the artifact it claims to be is detectable rather than
merely unlikely.

The full resolution per signal tables stay in the artifact store only. The deduplicated load
displacement Parquet is about 19 MB on its own, which is over the 5 MB rule of build spec
section 3.3, and it is regenerable from the raw CSVs in seconds.

The two raw CSVs are not in git either. They are referenced by SHA-256 in the ingest stage
manifest at their pinned location under `legacy_salvage/data/`, which is where the salvage of
build spec section 6.4 staged them. The digests recorded in the manifest are what ties every
number downstream back to the exact bytes the campaign produced.

Units, stated once and checked by contract tests: force in N, displacement in mm, strength in
MPa, absorbed energy in N mm, initial stiffness in N/mm. Damage and the softening ratio are
dimensionless.

Terminal damage is deliberately absent from `qoi.parquet`. It saturates at the concrete
damaged plasticity table cap for every completed run, so it carries no variance and is
useless as a surrogate target; build spec section 5.6 bans it and a test enforces the ban.
