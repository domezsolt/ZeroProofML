# ZeroProof Documentation Index

Welcome to the ZeroProof documentation. This guide combines conceptual foundations and technical details. Use the table of contents below to explore topic-by-topic. Items marked (new) are curated summaries that tie together existing docs and the codebase.

## Table of Contents

- Start Here
  - [Getting Started](topics/00_getting_started.md) (new)

- Concepts
  - [Topic 1: Overview & Principles](topics/01_overview.md) (new)
  - [Topic 2: Mathematical Foundations](topics/02_foundations.md) (new)
  - [Topic 3: Autodiff Modes](topics/03_autodiff_modes.md) (new)
  - [Autodiff: Mask-REAL](autodiff_mask_real.md)
  - [Saturating Gradients](saturating_grad_guide.md)
  - [Wheel Mode (Optional)](wheel_mode_guide.md)
  - [Float64 Enforcement & Precision](float64_enforcement.md)
  - [Quick Reference](quick_reference.md)

- Mathematics
  - [Foundations & Operations (Spec Clarifications)](../complete_v2.md)
  - [Adaptive Loss: Summary](adaptive_loss_summary.md)
  - [Optimization: Summary](optimization_summary.md)
  - [Wheel Mode: Summary](wheel_mode_summary.md)
  - [Saturating Grad: Summary](saturating_grad_summary.md)

- Architecture
  - [Topic 4: Layers & Variants](topics/04_layers.md) (new)
  - [Layers Overview](layers.md)
  - Bridges and Interop
    - [Bridge Summary](bridge_summary.md)
    - [Bridge Extended](bridge_extended.md)
  - [Verification Report](verification_report.md)
  - [Implementation Summary](implementation_summary.md)
  - [Implementation Complete](implementation_complete.md)
  - [ZeroProof Complete Overview](zeroproof_complete.md)

- Guides
  - [Adaptive Loss Guide](adaptive_loss_guide.md)
  - [L1 Projection Guide](l1_projection_guide.md)
  - [Optimization Guide](optimization_guide.md)
  - [Wheel Mode Guide](wheel_mode_guide.md)
  - [How‑To Checklists](topics/08_howto_checklists.md) (new)

- Training
  - [Topic 5: Training Policies](topics/05_training_policies.md) (new)
  - [Adaptive Loss Summary](adaptive_loss_summary.md)
  - [Optimization Summary](optimization_summary.md)

- Examples
  - Browse runnable scripts in `examples/`
  - Highlights: `examples/basic_usage.py`, `examples/complete_demo.py`, `examples/optimization_demo.py`

- Testing & Validation
  - Property and unit tests: `tests/`
  - Benchmarks: `benchmarks/`
  - Diagnostics: `test_diagnostics/`

- Sampling
  - [Topic 6: Sampling & Curriculum](topics/06_sampling_curriculum.md) (new)

- Evaluation
  - [Topic 7: Evaluation & Metrics](topics/07_evaluation_metrics.md) (new)

- Additional Reading
  - Conceptual whitepaper: `../concept_250908.md`
  - Key innovations: `../KEY_INNOVATIONS_250908.md`
  - Critical fixes & test notes: `../CRITICAL_FIXES_SUMMARY.md`, `../TEST_FIXES_SUMMARY.md`

## How to Use This Docs Set

- Start with Concepts → Topic 1 for a high-level understanding.
- Use Mathematics and Spec Clarifications for precise semantics and tag tables.
- Visit Architecture for code structure and implementation notes.
- Follow Guides when applying features in your projects.
# ZeroProof Documentation Index
- Check Examples for end-to-end usage patterns.

If you notice gaps or want a topic prioritized, open an issue or ping the team.
