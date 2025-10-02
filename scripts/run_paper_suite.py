#!/usr/bin/env python3
"""
Run the ZeroProofML paper experiment suite end-to-end.

This orchestrates:
- 2R dataset generation with |det(J)| binning and ensured near-pole coverage
- Baselines + TR runs across seeds (quick or full) via experiments/robotics/run_all.py
- Aggregation across seeds into CSV + LaTeX + B0–B2 bars
- Optional closed-loop rollout near singularities
- Optional 3R dataset + TR training/evaluation for multi-singularity evidence

Examples (quick, CPU-friendly):
  python scripts/run_paper_suite.py --quick --seeds 3 \
    --out-root results/robotics/paper_suite --rollout

Full(er) 2R run (longer):
  python scripts/run_paper_suite.py --seeds 5 \
    --zp-epochs 100 --mlp-epochs 50 --rat-epochs 50 \
    --out-root results/robotics/paper_suite_full

Add 3R evidence (TR-only quick):
  python scripts/run_paper_suite.py --quick --seeds 3 --include-3r \
    --out-root results/robotics/paper_suite --rrr-epochs 60
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    # Ensure repo root is on PYTHONPATH for example scripts importing 'zeroproof'
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    subprocess.run(cmd, check=True, env=env)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gen_rr_dataset(path: str, n: int, ensure_buckets: bool, seed: int) -> None:
    ensure_dir(os.path.dirname(path))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "examples/robotics/rr_ik_dataset.py"),
        "--n_samples",
        str(n),
        "--singular_ratio",
        "0.35",
        "--displacement_scale",
        "0.06",
        "--stratify_by_detj",
        "--train_ratio",
        "0.8",
        "--force_exact_singularities",
        "--ensure_buckets_nonzero" if ensure_buckets else "--no-ensure_buckets_nonzero",
        "--bucket-edges",
        "0",
        "1e-5",
        "1e-4",
        "1e-3",
        "1e-2",
        "inf",
        "--seed",
        str(seed),
        "--output",
        path,
        "--format",
        "json",
    ]
    try:
        run(cmd)
    except Exception:
        # Fallback: reuse existing dataset if available
        fallback = os.path.join(REPO_ROOT, "data", "rr_ik_dataset.json")
        if os.path.exists(fallback):
            print(f"[Fallback] Copying existing dataset {fallback} -> {path}")
            import shutil

            shutil.copyfile(fallback, path)
        else:
            raise


def _load_json(p: str) -> dict:
    import json

    with open(p, "r") as fh:
        return json.load(fh)


def _save_json(p: str, obj: dict) -> None:
    import json

    ensure_dir(os.path.dirname(p))
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2)


def make_holdout_b0(in_path: str, out_path: str) -> str:
    """Create a variant of the 2R dataset with B0 held out from training.

    Reads the input JSON, re-partitions samples so that all B0 samples go to test.
    Keeps other buckets roughly at the original train_ratio.
    """
    import math

    data = _load_json(in_path)
    samples = data.get("samples", [])
    md = data.get("metadata", {})
    # Edges
    raw_edges = md.get("bucket_edges")
    if not raw_edges:
        from zeroproof.utils.config import DEFAULT_BUCKET_EDGES

        raw_edges = DEFAULT_BUCKET_EDGES
    edges = []
    for e in raw_edges:
        try:
            edges.append(float(e))
        except Exception:
            edges.append(float("inf"))

    # Compute |detJ|≈|sin(theta2)| and assign bins
    def detj(s):
        try:
            return abs(math.sin(float(s["theta2"])))
        except Exception:
            return 0.0

    def bin_idx(d):
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (d >= lo if i == 0 else d > lo) and d <= hi:
                return i
        return len(edges) - 2

    bins = [bin_idx(detj(s)) for s in samples]
    # Group by bin
    by_bin: list[list[int]] = [[] for _ in range(len(edges) - 1)]
    for i, b in enumerate(bins):
        by_bin[b].append(i)
    # Build train/test: all of B0 go to test; others split by original train_ratio (default 0.8)
    tr_ratio = float(md.get("train_ratio", 0.8))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for b, idxs in enumerate(by_bin):
        if b == 0:
            test_idx.extend(idxs)
        else:
            k = int(round(tr_ratio * len(idxs)))
            train_idx.extend(idxs[:k])
            test_idx.extend(idxs[k:])
    # Reorder samples as train then test
    ordered = [samples[i] for i in train_idx] + [samples[i] for i in test_idx]
    data["samples"] = ordered
    data.setdefault("metadata", {})
    data["metadata"]["train_ratio"] = tr_ratio
    data["metadata"]["bucket_edges"] = edges
    data["metadata"]["train_bucket_counts"] = [
        sum(1 for i in train_idx if bins[i] == b) for b in range(len(edges) - 1)
    ]
    data["metadata"]["test_bucket_counts"] = [
        sum(1 for i in test_idx if bins[i] == b) for b in range(len(edges) - 1)
    ]
    data["metadata"]["stratified_by_detj"] = True
    _save_json(out_path, data)
    print(f"Created held-out-B0 dataset at {out_path}")
    return out_path


def run_rr_baselines(
    dataset: str,
    out_root: str,
    seed: int,
    quick: bool,
    mlp_epochs: int,
    rat_epochs: int,
    zp_epochs: int,
    models: List[str],
) -> str:
    outdir = os.path.join(out_root, f"quick_s{seed}" if quick else f"seed_{seed}")
    ensure_dir(outdir)
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "experiments/robotics/run_all.py"),
        "--dataset",
        dataset,
        "--out",
        outdir,
        "--seed",
        str(seed),
        "--models",
        *models,
    ]
    if quick:
        cmd += [
            "--profile",
            "quick",
            "--mlp_epochs",
            str(mlp_epochs),
            "--rat_epochs",
            str(rat_epochs),
            "--zp_epochs",
            str(zp_epochs),
        ]
    else:
        cmd += [
            "--mlp_epochs",
            str(mlp_epochs),
            "--rat_epochs",
            str(rat_epochs),
            "--zp_epochs",
            str(zp_epochs),
        ]
    run(cmd)
    return outdir


def aggregate_seeds(glob_pat: str, out_csv: str, out_tex: str | None) -> None:
    ensure_dir(os.path.dirname(out_csv))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/aggregate_quick_seeds.py"),
        "--glob",
        glob_pat,
        "--out",
        out_csv,
    ]
    if out_tex:
        cmd += ["--latex", out_tex]
    run(cmd)


def plot_b012_bars(csv_path: str, out_png: str, title: str = "B0–B2 MSE by method") -> None:
    ensure_dir(os.path.dirname(out_png))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/plot_b012_bars.py"),
        "--csv",
        csv_path,
        "--out",
        out_png,
        "--title",
        title,
        "--yscale",
        "log",
    ]
    run(cmd)


def rollout_rr(dataset: str, out_json: str, max_train: int = 2000) -> None:
    ensure_dir(os.path.dirname(out_json))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/rollout_near_singularity.py"),
        "--dataset",
        dataset,
        "--out",
        out_json,
        "--max_train",
        str(max_train),
    ]
    run(cmd)


def plot_tr_schematic(outdir: str) -> None:
    # Render TR pipeline schematic (used by paper.tex)
    ensure_dir(outdir)
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/plot_tr_schematic.py"),
        "--outdir",
        outdir,
    ]
    run(cmd)


def gen_rrr_dataset(path: str, n: int, seed: int) -> None:
    ensure_dir(os.path.dirname(path))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "examples/robotics/rrr_ik_dataset.py"),
        "--output",
        path,
        "--n_samples",
        str(n),
        "--train_ratio",
        "0.8",
        "--singular_ratio",
        "0.35",
        "--displacement_scale",
        "0.06",
        "--singularity_threshold",
        "1e-3",
        "--damping_factor",
        "0.01",
        "--force_exact_singularities",
        "--stratify_by_manip",
        "--bucket_edges",
        "0,1e-5,1e-4,1e-3,1e-2,inf",
        "--ensure_buckets_nonzero",
        "--seed",
        str(seed),
    ]
    run(cmd)


def train_rrr_tr(dataset: str, out_dir: str, epochs: int = 60) -> None:
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "examples/robotics/rrr_ik_train.py"),
        "--dataset",
        dataset,
        "--output_dir",
        out_dir,
        "--epochs",
        str(epochs),
    ]
    run(cmd)


def gen_ik6r_dataset(path: str, n: int, seed: int) -> None:
    ensure_dir(os.path.dirname(path))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "examples/robotics/ik6r_dataset.py"),
        "--output",
        path,
        "--n_samples",
        str(n),
        "--ensure_buckets_nonzero",
        "--seed",
        str(seed),
    ]
    run(cmd)


def train_ik6r(
    dataset: str,
    out_dir: str,
    epochs: int = 40,
    lr: float = 0.01,
    batch_size: int = 256,
    seed: int | None = None,
) -> None:
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "examples/robotics/ik6r_train.py"),
        "--dataset",
        dataset,
        "--output_dir",
        out_dir,
        "--epochs",
        str(epochs),
        "--learning_rate",
        str(lr),
        "--batch_size",
        str(batch_size),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    run(cmd)


def aggregate_ik6r(glob_pat: str, out_csv: str, out_tex: str | None) -> None:
    ensure_dir(os.path.dirname(out_csv))
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/aggregate_ik6r_seeds.py"),
        "--glob",
        glob_pat,
        "--out",
        out_csv,
    ]
    if out_tex:
        cmd += ["--latex", out_tex]
    run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ZeroProofML paper experiment suite (2R baselines + optional 3R evidence)"
    )
    ap.add_argument(
        "--out-root", default="results/robotics/paper_suite", help="Root directory for all outputs"
    )
    ap.add_argument("--quick", action="store_true", help="Quick profile (small data, few epochs)")
    ap.add_argument("--seeds", type=int, default=3, help="Number of seeds for 2R runs")
    ap.add_argument("--start-seed", type=int, default=1, help="Starting seed value")
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Use existing 2R dataset JSON instead of generating one",
    )
    ap.add_argument(
        "--holdout-b0",
        action="store_true",
        help="Create a held-out-B0 variant for 2R and use it for training/testing",
    )
    ap.add_argument("--n2r", type=int, default=12000, help="2R dataset size (total samples)")
    ap.add_argument("--n3r", type=int, default=16000, help="3R dataset size (total samples)")
    ap.add_argument("--mlp-epochs", type=int, default=2, help="MLP epochs (2R)")
    ap.add_argument("--rat-epochs", type=int, default=5, help="Rational+ε epochs (2R)")
    ap.add_argument("--zp-epochs", type=int, default=5, help="ZeroProof/TR epochs (2R)")
    ap.add_argument("--rrr-epochs", type=int, default=60, help="TR epochs for 3R evidence")
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "mlp",
            "rational_eps",
            "smooth",
            "learnable_eps",
            "eps_ens",
            "tr_basic",
            "tr_full",
            "dls",
        ],
        help="Subset of models to run (2R)",
    )
    ap.add_argument(
        "--rollout",
        action="store_true",
        help="Also run closed-loop rollout near singularities (2R)",
    )
    ap.add_argument(
        "--include-3r",
        action="store_true",
        help="Also run 3R dataset + TR training for multi-singularity evidence",
    )
    ap.add_argument(
        "--include-6r",
        action="store_true",
        help="Also run 6R synthetic dataset + TR training for multi-singularity evidence",
    )
    ap.add_argument("--ik6r-epochs", type=int, default=40, help="TR epochs for 6R evidence")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    # 1) 2R dataset (stratified, ensured near-pole bins)
    # 1) 2R dataset (existing or generated)
    if args.dataset and os.path.exists(args.dataset):
        rr_ds = os.path.abspath(args.dataset)
        print(f"Using existing 2R dataset: {rr_ds}")
    else:
        rr_ds = os.path.join(out_root, "rr_ik_dataset.json")
        # Use start-seed for dataset reproducibility
        gen_rr_dataset(
            rr_ds,
            n=args.n2r if not args.quick else max(6000, args.n2r // 2),
            ensure_buckets=True,
            seed=args.start_seed,
        )
    # Optional held-out B0 variant
    if args.holdout_b0:
        rr_ds_holdout = os.path.join(out_root, "rr_ik_dataset_holdout_b0.json")
        rr_ds = make_holdout_b0(rr_ds, rr_ds_holdout)

    # 2) 2R baselines/TR across seeds
    run_dirs: List[str] = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        rd = run_rr_baselines(
            dataset=rr_ds,
            out_root=out_root,
            seed=seed,
            quick=True if args.quick else False,
            mlp_epochs=args.mlp_epochs,
            rat_epochs=args.rat_epochs,
            zp_epochs=args.zp_epochs,
            models=args.models,
        )
        run_dirs.append(rd)

    # 3) Aggregate across seeds (overall + near-pole buckets)
    glob_pat = os.path.join(
        out_root,
        "quick_s*/comprehensive_comparison.json"
        if args.quick
        else "seed_*/comprehensive_comparison.json",
    )
    agg_csv = os.path.join(out_root, "aggregated", "seed_summary.csv")
    agg_tex = os.path.join(out_root, "aggregated", "seed_summary.tex")
    aggregate_seeds(glob_pat, agg_csv, agg_tex)

    # 4) Plot B0–B2 bars (log scale)
    bars_png = os.path.join(out_root, "figures", "b012_bars.png")
    # Convert comprehensive results into aggregated buckets CSV (if available)
    # Prefer using scripts/aggregate_buckets.py when buckets.json files exist; otherwise use seed_summary
    try:
        # Attempt aggregate_buckets over any pre-existing buckets.json
        buckets_csv = os.path.join(out_root, "aggregated", "buckets_table.csv")
        # Scan runs under repo 'runs' if user created them; else skip
        any_buckets = glob.glob(os.path.join("runs", "**", "buckets.json"), recursive=True)
        if any_buckets:
            run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "scripts/aggregate_buckets.py"),
                    "--scan",
                    "runs",
                    "--out",
                    buckets_csv,
                ]
            )
            plot_b012_bars(buckets_csv, bars_png)
        else:
            # Fallback: draw bars from seed_summary CSV (overall buckets may be available in comprehensive JSONs only)
            print(
                "Note: No buckets.json found under runs/. Skipping buckets_to_latex path; using seed summary only."
            )
    except Exception as e:
        print(f"Warning: aggregate_buckets failed: {e}")

    # 5) Optional closed-loop rollout
    if args.rollout:
        rollout_json = os.path.join(out_root, "rollout_summary.json")
        try:
            rollout_rr(rr_ds, rollout_json, max_train=2000 if args.quick else 8000)
        except Exception as e:
            print(f"Warning: rollout failed: {e}")

    # 6) Optional 3R evidence (TR-only)
    if args.include_3r:
        rrr_ds = os.path.join(out_root, "ik3r_dataset.json")
        rrr_out = os.path.join(out_root, "e3r")
        try:
            gen_rrr_dataset(
                rrr_ds,
                n=args.n3r if not args.quick else max(8000, args.n3r // 2),
                seed=args.start_seed,
            )
            train_rrr_tr(
                rrr_ds,
                rrr_out,
                epochs=args.rrr_epochs if not args.quick else min(60, args.rrr_epochs),
            )
        except Exception as e:
            print(f"Warning: 3R run failed: {e}")

    # 6b) Optional 6R evidence (TR-only)
    if args.include_6r:
        ik6r_ds = os.path.join(out_root, "ik6r_dataset.json")
        try:
            gen_ik6r_dataset(
                ik6r_ds,
                n=args.n3r if not args.quick else max(12000, args.n3r // 2),
                seed=args.start_seed,
            )
            # Train across seeds
            for i in range(args.seeds):
                s = args.start_seed + i
                train_ik6r(
                    ik6r_ds,
                    os.path.join(out_root, f"ik6r_s{s}"),
                    epochs=args.ik6r_epochs if not args.quick else min(args.ik6r_epochs, 40),
                    seed=s,
                )
            aggregate_ik6r(
                os.path.join(out_root, "ik6r_s*/ik6r_results.json"),
                os.path.join(out_root, "aggregated", "ik6r_summary.csv"),
                os.path.join(out_root, "aggregated", "ik6r_summary.tex"),
            )
        except Exception as e:
            print(f"Warning: 6R run failed: {e}")

    # 7) TR schematic figure for the paper
    try:
        plot_tr_schematic(os.path.join(out_root, "figures"))
    except Exception as e:
        print(f"Warning: TR schematic plotting failed: {e}")

    print("\nSuite complete. Key artifacts:")
    print(f"  2R dataset: {rr_ds}")
    print(f"  Seed comprehensive JSONs: {glob_pat}")
    print(f"  Seed aggregate CSV: {agg_csv}")
    if os.path.exists(agg_tex):
        print(f"  Seed aggregate LaTeX: {agg_tex}")
    if os.path.exists(bars_png):
        print(f"  B0–B2 bars: {bars_png}")
    if args.rollout:
        print(f'  Closed-loop rollout (2R): {os.path.join(out_root, "rollout_summary.json")}')
    if args.include_3r:
        print(f'  3R dataset: {os.path.join(out_root, "ik3r_dataset.json")}')
        print(f'  3R TR results: {os.path.join(out_root, "e3r", "e3r_results.json")}')


if __name__ == "__main__":
    main()
