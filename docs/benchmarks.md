# Benchmarks & Overhead

This page summarizes the built‑in benchmarking tools and how to use them.

## Mini Benchmark Suite

Run a portable set of microbenchmarks and save JSON + a text summary:

```bash
python -m zeroproof.bench --suite all --out benchmark_results
```

By default, `all` runs:

- `arithmetic`: core TR ops
- `autodiff`: simple gradients, chain rule
- `layers`: TR‑Rational and TR‑Norm forward passes
- `parallel`: sequential vs threaded map speedup
- `memory`: memory estimate for a large graph
- `overhead`: TR core ops vs IEEE float ops
- `reductions`: sequential vs pairwise reduction
- `tag_overhead`: composite expression with undefined cases (denominator=0)
- `caching`: fibonacci with/without memoization

Optional backend microbenches (run only if installed):

- `torch`: simple vector op with PyTorch
- `jax`: simple vector op with JAX

You can customize the micro‑bench load with:

- `--iterations N` (per sample; default 1000)
- `--samples S` (repeats; default 5)

A compact text summary is saved to `benchmark_results/summary.txt`.

## Overhead (Hybrid vs Mask‑REAL)

Compare per‑batch timing and hybrid activation stats:

```bash
python -m zeroproof.overhead_cli --out runs/overhead.json
```

Outputs include avg_step_ms for baseline/hybrid, `slowdown_x`, and hybrid mode stats.

## Compare Two JSONs (Regression Check)

Detect performance regressions with a slowdown threshold:

```bash
python -m zeroproof.bench_compare \
  --baseline benchmark_results/bench_old.json \
  --candidate benchmark_results/bench_new.json \
  --max-slowdown 1.20
```

Return code is non‑zero when slowdown exceeds the threshold.

## CI Baseline

If you place a baseline JSON at `benchmarks/baseline.json`, the CI job compares
new results against it (non‑blocking). Update the baseline locally with:

```bash
python scripts/update_benchmark_baseline.py --src benchmark_results
```

## Tips

- Keep `--iterations` and `--samples` modest in CI to reduce noise and runtime (e.g., 300 and 3).
- Use pairwise (tree) reductions when aggregating many terms to bound graph depth and improve determinism.
- Consider plotting JSONs offline for deeper analysis of distributions and slowdowns.
