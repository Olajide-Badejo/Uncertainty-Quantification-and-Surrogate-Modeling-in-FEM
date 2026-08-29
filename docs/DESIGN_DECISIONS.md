# Design decisions

Dated, append only. Every deviation from `docs/BUILD_SPEC.md` is recorded here with its
reason, so that a future me reading the code can tell a deliberate choice from a mistake.

## 2026-08-30, Phase P0

### Version policy

The spec says to pin the newest stable release of every tool, resolved at P0, never a
version copied out of the document. I did that: the resolved matrix at the bottom of this
file is written by `ufem doctor`, not typed, and it is regenerated rather than edited. Every
dependency in `pyproject.toml` is pinned with `==`, not a lower bound, because a
reproducibility claim that floats its dependencies is not a claim.

### Deviation: torch 2.13.0 from PyPI, not the cu130 index

Spec section 0.3 and 3.2 call for `pip install torch==2.13.0 --index-url
https://download.pytorch.org/whl/cu130`, so that the RTX 5070 (sm_120) is usable. I
installed `torch 2.13.0+cpu` from the default PyPI index instead. This is a deliberate
deviation and here is the argument for it.

The production path is CPU by design, not by accident. Spec section 17.2 requires the whole
chain from ingest through propagate to run on NumPy, scikit-learn, and GPyTorch on one CPU
thread with `torch.use_deterministic_algorithms(True)`, precisely so the results are
bitwise reproducible on this machine. Nothing in Track A ever touches the GPU. The only
work that wants CUDA is the neural ablations of spec section 10.6 (the autoencoder and the
deep ensemble), and those are deferred to P9. Installing a 3 GB CUDA wheel at P0 to serve
code that does not exist yet buys nothing and costs disk, install time, and a CI story that
has to explain why the lockfile names a wheel the runners cannot use.

The `pyproject.toml` pin is therefore `torch==2.13.0`, without a local version segment,
which resolves to the CPU wheel on PyPI and to whatever local build is already present in a
venv that was fed the cu130 index. That keeps the CI runners honest (they have no GPU and
must not pretend otherwise, per spec section 18) without blocking the GPU install here.

Revisit this when ablations 2 and 3 of section 10.6 are actually run. At that point the
correct move is to reinstall torch from the cu130 index into this same venv, confirm
`torch.cuda.is_available()`, rerun `ufem doctor` so the matrix below records the change,
and note in the report that the ablation claims are statistically reproducible rather than
bitwise. `ufem doctor` prints the torch build and device on every run, so the deviation
cannot go unnoticed.

### pandas 3.0.5

The spec leaves pandas as "pinned deliberately, 3.0.x is out; pin and record". I pinned
3.0.5. It installs cleanly on CPython 3.14 alongside pyarrow 25.0.1, which matters because
the ingest stage of P1 writes Parquet through pyarrow, and the copy on write semantics that
became mandatory in pandas 3 are what I want anyway: the v1 pipeline had several places
where a chained assignment either silently did nothing or silently mutated a caller's frame.

### License: MIT

Recorded here per spec section 0.3, declared in `pyproject.toml`, and the full text is in
`LICENSE` at the repository root. The inherited v1 tree was already MIT, so there is no
relicensing question over the salvaged assets.

### UI: NiceGUI, with FastAPI as the pre authorized fallback

UFEM Lab (spec section 15) is built on NiceGUI 3.16.0. It gives me a local web dashboard
with sliders bound to Python callbacks without writing a frontend, and it runs on FastAPI
underneath, so if I hit a wall (a plot component that will not update at the 50 ms budget,
or a packaging problem on Windows) I can drop to plain FastAPI plus Plotly and serve the
same artifact store without changing the pipeline at all. I am recording that fallback now
so that taking it later is not a scope change.

### Config hashing

The run identity is the SHA-256 of the canonical JSON of the fully resolved config, both
YAML files together, computed in `ufem.config.config_hash`. Canonical means sorted keys and
no insignificant whitespace, so the digest is stable across two loads and moves the moment
any parameter moves. This is what `manifest.py` stamps into every artifact.

<!-- BEGIN RESOLVED VERSIONS -->

### Resolved version matrix, 2026-08-30

Written by `ufem doctor` on Windows-11-10.0.26200-SP0. Regenerate it, do not edit it.

| Component | Resolved |
|---|---|
| ufem | 1.1.0.dev0 |
| python | 3.14.0 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| pyarrow | 25.0.1 |
| pydantic | 2.13.5 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.1 |
| torch | 2.13.0 |
| gpytorch | 1.15.2 |
| openturns | 1.27.post1 |
| SALib | 1.5.2 |
| fdasrsf | 2.6.10 |
| torch device | torch 2.13.0+cpu, CUDA build None, no visible GPU (CPU path) |
| config SHA-256 | `e8bd99810d8bc145347d236d0a794ac50258431590378262b5fbb142a60ace0f` |

<!-- END RESOLVED VERSIONS -->
