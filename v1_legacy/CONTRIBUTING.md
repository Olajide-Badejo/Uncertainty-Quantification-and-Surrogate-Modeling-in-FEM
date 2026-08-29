# Contributing

## Environment

1. Create and activate a Python environment.
2. Install runtime dependencies:
   - `pip install -r requirements.txt`
3. Install development dependencies:
   - `pip install -r requirements-dev.txt`

## Quality Gates

Before opening a pull request, run:

1. `python tests/test_syntax.py`
2. `pytest tests/ -v --tb=short`
3. `flake8 . --select=E9,F63,F7,F82 --max-line-length=120`

## Code Style

- Prefer `pathlib.Path` for file-system paths.
- Avoid bare `except:` blocks; use specific exception classes.
- Keep scripts repository-relative and platform-independent.
