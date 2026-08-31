# Contributing

Build spec section 19 asks this file for one thing: the gates a change must pass. They are
listed below in the order they will stop you, with what each one is protecting.

## The gates

| Gate | Command | What it protects |
|---|---|---|
| Dash and banned identifier lint | `python scripts/dash_lint.py` | Ground rules 3, 4, 8 and 13, binding laws 2 and 5 |
| File size | `python scripts/check_file_sizes.py` | Build spec 3.3: the repository is not a data store and not a venv |
| Style | `ruff check src tests scripts` | Line length, imports, and bare except a second time |
| Tests | `pytest tests -m "not slow"` | Everything else |
| Full suite | `pytest tests` | The determinism checks, which rerun stages and compare digests |

The first four run in `.github/workflows/ci.yml` on every push to `main` and to any `phase/**`
branch, and on every pull request. The report builds separately in
`.github/workflows/report.yml`, on a TeX Live container with no Python and no artifact store,
which works only because every table fragment and every figure is committed.

## The rules that are easy to break by accident

**No em dash and no en dash, anywhere.** Not in code, not in prose, not in a commit message,
not in a mermaid label. Use a comma, a colon, parentheses, or the word "to". The linter reports
the file and line.

**No number is typed into a published document.** `README.md`, `report/main.tex`, the data
card, the model card and the dashboard all get their numbers from artifacts. If you need to
state a measurement, add it to the generator that owns the document:

| Document | Generator | Staleness gate |
|---|---|---|
| `README.md` | `scripts/readme_inject.py` | `tests/test_readme_consistency.py` |
| `docs/DATA_CARD.md` and `report/tables/*.tex` | `scripts/make_data_card.py` | `tests/test_data_card.py` |
| `docs/MODEL_CARD.md` | `scripts/make_model_card.py` | `tests/test_model_card.py` |
| `report/figures/*.pdf` | `report/figures_src/make_figures.py` | committed and diffable, byte reproducible |
| `docs/media/*.png` | `scripts/make_readme_media.py` | the same figure functions as the report |

Each staleness gate regenerates its document and compares byte for byte, so an edit by hand is
a failing test rather than a document that quietly disagrees with the pipeline. Regenerate and
commit the result instead.

**A generated document must not carry a quantity that moves without a measurement moving.** No
wall times, no dates, no git commits. `docs/DEFECT_LOG.md` records this lesson costing the
project twice. A gate that fires on noise gets muted, and a muted gate catches nothing.

**Every bug gets a regression test.** Ground rule 7: it fails before the fix and passes after,
both runs shown, and the entry goes in `docs/DEFECT_LOG.md` with the evidence, the fix commit
and the test name.

**Every deviation from `docs/BUILD_SPEC.md` gets written down.** Dated, in
`docs/DESIGN_DECISIONS.md`, with the reason. A future reader has to be able to tell a
deliberate choice from a mistake, and silence reads as a mistake.

## Working shape

Branch from `main` as `phase/<name>` or `fix/<name>`, use Conventional Commits, and keep the
tree clean. Releases are tagged from `main` only, and `python scripts/make_release.py` is what
checks that everything a release attaches is current. It prints the publish command; it does
not run it.

## Running the pipeline

`ufem doctor` first, always. Then `ufem run all`, which prints `[cache hit]` for any stage whose
inputs are unchanged and reruns one under `--force`. The raw campaign CSVs are not tracked; if
they are absent the ingest stage raises naming the file rather than producing an empty result,
and the gridded outputs under `data/processed/` are committed so the repository still reads.

Issues are welcome, and defect reports that mirror an entry in `docs/DEFECT_LOG.md`, evidence
first, are the most useful shape.
