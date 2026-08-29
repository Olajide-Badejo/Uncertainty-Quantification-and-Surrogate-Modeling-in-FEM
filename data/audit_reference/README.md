# Audit reference: the committed golden values

These six files are the machine readable output of the pre build audit I ran on
2026-08-28 over the inherited 400 sample Abaqus campaign. They are committed, small, and
frozen, and they exist for exactly one purpose: Phase P1 regenerates the same quantities
from the raw CSVs through the new pipeline and gates against these values to 1e-9. If the
new ingest and grid stages disagree with what is here, P1 does not pass.

| File | What it holds |
|---|---|
| `audit_script.py` | The script that produced everything else here, kept so the audit is reproducible rather than asserted. |
| `audit_summary.json` | Headline numbers: the validity split, per QoI statistics, the censoring tests, the correlation structure. |
| `sample_validity.csv` | Per sample classification across all 400 design rows, with the metrics each classification was made from. |
| `common_grid_sample_ids.csv` | The job IDs whose curves survived interpolation onto the common displacement grid, in order. |
| `common_U2_grid.npy` | The common displacement grid itself. |
| `RF2_on_common_U2_grid.npy` | Reaction force interpolated onto that grid, one row per surviving job. |

Two things these files are not. They are not a pipeline input: no stage in `src/ufem/`
reads this directory, and P1 ingests from `legacy_salvage/data/` like every other run.
And they are not a source of truth about the physics, only about what the v1 campaign
actually produced on disk.

The validity list here is derived from the raw data. The hard coded 198 sample literal
that v1 carried in its extraction script is recorded in `data/quarantine/README.md` as a
defect, and is never read by anything.
