# ZeroProofML — TODO for v0.1.0 (repo hardening & first external release)

> **Goal:** Ship a clean **v0.1.0** with modern packaging, green CI across platforms, a typed public API, runnable examples/CLIs, and first‑class evaluation tooling that matches the project's principles.

---

## P0 — Packaging & Release

- [ ] **Adopt PEP 621 packaging**: add `pyproject.toml` with `[project]` metadata, Trove classifiers, and extras:
  - `zeroproof[torch]`, `zeroproof[jax]`, `zeroproof[all]` (aggregate).
  - Include `package-data` for demo configs/plot styles and add `py.typed`.
- [ ] **Versioning & changelog**:
  - Add `__version__` in `zeroproof/__init__.py`.
  - Create `CHANGELOG.md` and follow SemVer (`v0.1.0`).
- [ ] **Release workflow**:
  - GitHub Actions: build sdist+wheel on tag and publish to PyPI (Trusted Publisher/OIDC or API token).
  - Add `release.yml` that also creates GitHub Release notes from `CHANGELOG.md`.
- [ ] **README install section**:
  - Until PyPI is live, include *install‑from‑source* instructions; switch to `pip install zeroproof[...]` once published.
  - Clarify backend extras and minimal requirements (Python 3.9+).
- [ ] **License file**: Ensure `LICENSE` file exists at repo root with full MIT license text.
- [ ] **Python version compatibility testing**: Explicitly test upper bound (3.13 if available) and document any known incompatibilities.
- [ ] **Dependency pinning strategy**: Document whether you pin exact versions or use ranges; consider `requirements-dev.txt` separate from runtime deps.
- [ ] **Wheel compatibility tags**: Ensure wheels are built as `py3-none-any` (pure Python) or specify platform requirements if you have compiled extensions.

---

## P0 — CI / Quality Gates

- [ ] **Matrix CI:** Ubuntu, macOS, Windows × Python 3.9–3.12 × (minimal deps / +torch / +jax).
- [ ] **Linters & formatters:** `ruff`, `black`, `isort` with **pre‑commit**; fail build on lint errors.
- [ ] **Type checking:** `mypy` (or Pyright) on the **public API** and examples.
- [ ] **Tests:** unit + property tests; mark slow/e2e to keep PR CI under ~10 min.
- [ ] **Determinism & safety job:**
  - Seed‑fixed mini‑training loop; assert **no NaNs**, **stable tag counts**, and **identical losses** across two runs.
  - Deterministic pairwise reductions when the policy flag is on.
- [ ] **Coverage:** upload to a service; add badges (build, coverage) to README.
- [ ] **Security scanning**: Add Dependabot or similar for dependency vulnerability alerts.
- [ ] **License header validation**: Automated check that all source files have proper MIT headers.
- [ ] **Import time test**: Ensure `import zeroproof` works without requiring optional dependencies (torch/jax).
- [ ] **Documentation build test**: If you add docs later, ensure they build without errors in CI.

---

## P0 — Public API Typing & Contracts

- [ ] Add **type hints** across the public surface (e.g., `TRTag`, `TRScalar`, `TRNode`, layer `forward` returning `(value, tag)`).
- [ ] Mark package as typed (`py.typed`), and add minimal `Protocol`s for evaluator/policy interfaces.
- [ ] **Docstring contracts** for tags and policies:
  - REAL/±INF/Φ semantics.
  - Reduction modes (STRICT vs DROP_NULL).
  - Hybrid schedule & hysteresis knobs (τ_on/τ_off, δ).
- [ ] Export a stable set in `zeroproof/__init__.py` and document **backward‑compat** expectations for v0.1.x.
- [ ] **Deprecation policy**: Document how you'll handle breaking changes (important even for 0.x versions).
- [ ] **Error messages audit**: Ensure error messages are helpful and guide users toward solutions.
- [ ] **Input validation**: Add explicit validation for common user errors (wrong shapes, invalid tag values, etc.).

---

## P0 — Docs & README

- [ ] **Getting Started**: quick arithmetic + layer forward snippet (NumPy‑only path first), link to examples.
- [ ] **Concept links**: point to Foundations (semantics), Autodiff Modes (Mask‑REAL/Saturating/Hybrid), Layers, Training Policies, Sampling, Evaluation.
- [ ] **Backend status table**: Torch/JAX marked **supported/experimental**, with extras and minimal versions.
- [ ] **FAQ**: what happens at 1/0, 0/0, log(≤0), how tags propagate, when to pick Hybrid.
- [ ] **Badges section**: Build status, PyPI version, Python versions, license, code coverage.
- [ ] **Quick motivation**: 1-2 sentences explaining *why* ZeroProofML exists (what problem it solves).
- [ ] **30-second pitch**: Put the core value prop in the first 3 lines of README.
- [ ] **Comparison table**: Brief comparison to alternatives (standard autodiff, symbolic approaches) highlighting when to use ZeroProofML.
- [ ] **System requirements**: Disk space, RAM recommendations for examples.
- [ ] **Troubleshooting section**: Common installation/runtime issues and solutions.
- [ ] **Animated GIF/demo in README**: Visual showing tag propagation or a training curve.
- [ ] **Documentation hosting**: Decide where full docs will live (ReadTheDocs, GitHub Pages, etc.).

---

## P0 — Golden‑path Examples & CLIs

- [ ] **1D pole tutorial** (CLI + notebook): train on `1/(x−1)`, log PLE & sign consistency; save JSON metrics and plots.
  - CLI: `python examples/one_d_pole.py --out runs/1d`.
- [ ] **RR‑arm IK quick run**: dataset script with |det(J)| buckets; TR vs ε‑rational vs MLP baselines; JSON outputs + plots.
  - CLI: `python examples/robotics/rr_ik_quick.py --out runs/rr_ik_quick`.
- [ ] **Hybrid schedule demo**: show MR↔SAT switching stats (q_min, near‑pole ratio, saturation ratio).

---

## P0 — First‑class Evaluation Module

- [ ] Ship `IntegratedEvaluator` & `PoleEvaluator` as **stable imports** with docstrings/examples.
  - `evaluator.evaluate_model(model, x_values)` → dataclass; optional plots + JSON.
  - CLI: `python -m zeroproof.eval --model-path ... --xgrid ... --out ...`.
- [ ] Logs include thresholds, bucket edges, seed, policy flags (τ, δ), and device info.

---

## P1 — Performance & Benchmarks

- [ ] **Micro‑benchmarks**: REAL vs near‑pole throughput; record `avg_step_ms`, `optim_time_ms` in training summaries.
- [ ] **Expose knobs**: guard bands `tau_Q_on/off`, `tau_P_on/off`, saturation bound, hybrid δ; persist in run summaries.
- [ ] **Ablations**: switching‑threshold sensitivity; MR vs SAT vs Hybrid; stability vs speed plots.
- [ ] **Benchmark baselines**: Document what you're comparing against and provide baseline implementations.

---

## P1 — Training Policies & Sampling

- [ ] **Adaptive λ (coverage control)** as a pluggable object with `get_statistics()`; plot coverage vs epoch.
- [ ] **Near‑pole sampling utilities** (importance/active); diagnostics for `q_min`, |Q| stats; save sampler state.
- [ ] **Unified trainer**: expose coverage %, flip‑rate, saturating ratio, curvature bound; export JSON/CSV.

---

## P1 — Layers & Backends

- [ ] Harden docs for `TRRational`, `HybridTRRational`, `TRNorm`, multi‑output/shared‑Q, and multi‑input rational.
- [ ] Provide examples for **multi‑output** (shared Q) and **multi‑input** (light TR‑MLP → rational heads).
- [ ] Clarify backend extras and ensure conditional imports keep NumPy‑only installs working.
- [ ] **Platform-specific issues**: Document any known issues on ARM Macs, Windows with certain backends, etc.

---

## P1 — Reproducibility & Paper Parity

- [ ] **`scripts/run_paper_suite.sh`** to regenerate *key* figures/tables (seeded, CPU‑friendly configs).
- [ ] **Environment capture**: commit SHA, seeds, bucket edges, policy flags in each result file.
- [ ] **Verifier**: assert metric ranges to catch regressions (e.g., PLE decay, B0/B1 MSE bands).

---

## P1 — Developer Experience

- [ ] **Development setup documentation**: Step-by-step guide for contributors to set up dev environment.
- [ ] **Debug logging**: Structured logging with configurable levels (especially for tag propagation debugging).
- [ ] **Minimal reproducer template**: Template for bug reports that includes version info, minimal code, expected vs actual behavior.
- [ ] **Performance profiling helpers**: Built-in utilities to profile tag overhead vs standard operations.
- [ ] **CONTRIBUTING.md**: Add contribution guidelines, development workflow, and code standards.

---

## P1 — Examples & Tutorials

- [ ] **Failure cases gallery**: Document known limitations and show what happens (helps set expectations).
- [ ] **Migration guide**: If users are coming from standard PyTorch/JAX, show equivalent code translations.
- [ ] **Jupyter notebook CI**: Ensure notebooks execute cleanly (consider `nbval` or similar).

---

## P1 — Testing Infrastructure

- [ ] **Regression test suite**: Save outputs from key examples as golden files; detect unexpected changes.
- [ ] **Numerical stability tests**: Test behavior near machine epsilon, very large/small values.
- [ ] **Backend parity tests**: Same model should give same results on NumPy/Torch/JAX (within tolerance).

---

## P1 — Community & Governance

- [ ] **Issue triage process**: Document who responds to issues and expected response time.
- [ ] **CODE_OF_CONDUCT.md**: Add code of conduct for community interactions.
- [ ] **Issue/PR templates**: Provide templates for bug reports, feature requests, and pull requests.
- [ ] **GitHub Discussions**: Enable for Q&A and community discussions.
- [ ] **CITATION.cff**: Add citation file for academic use.
- [ ] **DOI**: Link to DOI when ready.
- [ ] **Backward compatibility promise**: Clarify what changes might break between minor versions (even for 0.x).

---

## P2 — TR‑Softmax & Policies (optional for v0.1)

- [ ] Bundle a **TR‑safe softmax surrogate** with unit tests and a tiny classifier example.
- [ ] Deterministic reduction trees in normalization paths when policy enabled.

---

## P2 — Observability & Debugging

- [ ] **Visualization tools**: Helper to visualize tag propagation through a computation graph.
- [ ] **Tag statistics collector**: Runtime tool to gather stats on tag distribution during training.
- [ ] **Integration with tensorboard/wandb**: Log tag-specific metrics alongside standard training metrics.

---

## P2 — Ecosystem Integration

- [ ] **Type stub files**: Consider shipping `.pyi` stubs for better IDE support.
- [ ] **Hugging Face Hub integration**: If relevant, allow loading/sharing trained models.
- [ ] **Serialization format**: Document how to save/load TR models consistently across versions.

---

## P2 — Documentation & Non-Goals

- [ ] **Explicit non-goals**: Document what ZeroProofML won't support to help set boundaries.
- [ ] **Architecture decision records**: Document key design decisions for future reference.

---

## Milestones

- **v0.1.0 — First external release** (P0)  
  Packaging, CI matrix + determinism checks, typed public API, README/Getting Started, golden‑path examples/CLIs, evaluator v1, basic documentation.
 
- **v0.2.0 — Training & sampling suite** (P1)  
  Adaptive coverage, sampling utilities, benchmarks & ablations, richer layer examples, developer experience improvements, testing infrastructure.
 
- **v0.3.0 — Policies & softmax** (P2)  
  TR‑softmax surrogate, deterministic reductions everywhere; extended policy docs, observability tools, ecosystem integrations.

---

## Definition of Done

- A newcomer can: `pip install zeroproof[torch]` (or install from source), run **two demos**, and reproduce RR‑IK quick results with JSON metrics + plots on CPU in <10 minutes.
- CI is green on all OS/Python combos; determinism tests pass; coverage ≥ 80%.
- Public API is **typed** and **documented** with explicit tag/contract behaviors; evaluator & CLI are discoverable.
- README has clear motivation, installation instructions, troubleshooting guide, and visual examples.
- Security scanning and dependency management are automated.
- Contributing guidelines are clear and welcoming to first-time contributors.
