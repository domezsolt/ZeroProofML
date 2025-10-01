# Contributing to ZeroProof

Thanks for your interest in contributing! This guide helps you set up a
development environment and describes our conventions.

## Quick Start

1) Fork and clone the repo, then set up a virtualenv:

```bash
git clone https://github.com/zeroproof/zeroproof.git
cd zeroproof
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
```

2) Install in editable mode with dev extras:

```bash
pip install -e .[dev]
```

3) Run tests and linters locally:

```bash
pytest -q
ruff check zeroproof tests examples
black --check zeroproof tests examples
isort --check-only zeroproof tests examples
mypy zeroproof
```

4) Enable pre-commit hooks (recommended):

```bash
pre-commit install
```

Now, changes will be auto-formatted and linted on commit.

## Development Tips

- Keep optional backends optional. Code under `zeroproof/bridge/{torch,jax}`
  must not be imported at top-level in `zeroproof/__init__.py`.
- Maintain TR semantics. Operations should be total and deterministic; avoid
  epsilon hacks. Prefer policy flags for behavior choices.
- Deterministic reductions. Honor `TRPolicy.deterministic_reduction` by using
  pairwise trees/compensated sums in reductions.
- Type hints. Public API should be typed. Use the Protocols in
  `zeroproof/protocols.py` for interfaces.
- Tests. Add unit tests for new functionality and property tests where
  applicable. Mark slow/e2e appropriately.

## Pull Requests

- Write clear PR titles and descriptions. Link issues where applicable.
- Include tests and docs for new features.
- Keep diffs focused; avoid unrelated refactors.
- Ensure CI is green.

## Reporting Issues

Please include a minimal reproducer, environment info (Python version, OS),
and expected vs actual behavior. See the issue templates for guidance.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.

