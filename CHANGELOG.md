# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-10-01

First external-ready release candidate (repo hardening).

Added
- PEP 621 packaging via `pyproject.toml` with extras: `torch`, `jax`, `all`, `dev`.
- `py.typed` marker to ship type information for the public API.
- GitHub Actions release workflow to build wheels/sdist and publish on tag.
- Dependabot configuration for dependency updates.
- Initial mypy configuration scoped to the public API.

Changed
- CI matrix aligned to Python 3.9–3.12; lint job installs `black`, `ruff`, `isort`, `mypy`.
- README installation section clarifies install-from-source and extras.

Fixed
- Minor import in evaluator utilities to avoid `NameError` when generating default evaluation grid.
- Hybrid trainer now aggregates per‑sample losses via a balanced pairwise
  reduction to bound graph depth and avoid recursion issues during backprop.

Notes
- Until PyPI publication, use `pip install -e .[dev]` for development and `pip install -e .[all]` for full features.
- Torch/JAX remain optional dependencies; `import zeroproof` should work without them.
