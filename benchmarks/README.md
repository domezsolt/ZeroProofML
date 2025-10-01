# ZeroProof Benchmarks

This folder holds benchmark utilities and (optionally) a baseline JSON file
for CI comparison.

- Mini suite (portable):
  - `python -m zeroproof.bench --suite all --out benchmark_results`
  - Default `all` runs `arithmetic`, `autodiff`, `layers`, `parallel`, `memory`,
    and a TR vs IEEE `overhead` comparison.

- Full suite (longer):
  - `python benchmarks/run_benchmarks.py --suite all --output benchmark_results`

- Compare two runs (regression check):
  - `python -m zeroproof.bench_compare --baseline A.json --candidate B.json --max-slowdown 1.20`

- CI baseline (optional):
  - If you want CI to compare against a baseline, place a file at
    `benchmarks/baseline.json`. Keep it small (mini suite) and generated on a
    representative runner. The CI job is non‑blocking and will skip comparison
    if the baseline is absent.

