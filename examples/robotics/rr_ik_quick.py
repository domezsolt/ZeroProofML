"""
RR-arm IK quick runner

Generates a small RR IK dataset (with |det(J)| buckets) and runs a quick
training/evaluation for selected models, saving compact JSON results.

Usage:
  python examples/robotics/rr_ik_quick.py --out runs/rr_ik_quick
"""

from __future__ import annotations

import argparse
import os
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RR IK quick run (dataset + training)")
    ap.add_argument("--out", type=str, default="runs/rr_ik_quick", help="Output directory")
    ap.add_argument("--n", type=int, default=4000, help="Total samples")
    ap.add_argument("--singular-ratio", type=float, default=0.35, help="Near-pole fraction")
    ap.add_argument("--seed", type=int, default=123, help="Global seed")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["tr_rat", "rat_eps", "mlp"],
        help="Models to run: tr_rat, rat_eps, mlp",
    )
    ap.add_argument("--profile", choices=["quick", "full"], default="quick")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) Generate dataset with buckets and metadata
    try:
        from .rr_ik_dataset import RobotConfig, RRDatasetGenerator
    except Exception:
        from rr_ik_dataset import RRDatasetGenerator, RobotConfig
    from zeroproof.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    config = RobotConfig(L1=1.0, L2=1.0)
    gen = RRDatasetGenerator(config)
    _ = gen.generate_dataset(
        n_samples=int(args.n),
        singular_ratio=float(args.singular_ratio),
        displacement_scale=0.1,
        singularity_threshold=1e-3,
        damping_factor=0.01,
        force_exact_singularities=True,
        min_detj_regular=1e-6,
    )
    dataset_file = os.path.join(args.out, "rr_ik_dataset.json")
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    gen.save_dataset(dataset_file)
    print(f"Dataset saved to {dataset_file}")

    # 2) Run quick training for each requested model
    try:
        from .rr_ik_train import TrainingConfig, run_experiment
    except Exception:
        from rr_ik_train import TrainingConfig, run_experiment
    results_files: List[str] = []
    for model in args.models:
        out_dir = os.path.join(args.out, model)
        cfg = TrainingConfig(model_type=model)
        # Quick profile defaults
        if args.profile == "quick":
            cfg.epochs = 20
            cfg.batch_size = 1024
            cfg.evaluate_ple = False
            cfg.enable_anti_illusion = False
        else:
            cfg.epochs = 100
            cfg.batch_size = 2048
            cfg.evaluate_ple = True

        _ = run_experiment(
            dataset_file,
            cfg,
            out_dir,
            seed=args.seed,
            limit_train=2000 if args.profile == "quick" else None,
            limit_test=500 if args.profile == "quick" else None,
            log_policy_console=False,
        )
        results_files.append(os.path.join(out_dir, f"results_{model}.json"))

    print("Quick run completed. Result files:")
    for rf in results_files:
        print(" -", rf)


if __name__ == "__main__":
    main()
