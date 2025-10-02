"""
Generate a compact RR-IK dataset tailored for Hybrid overhead checks.

Creates a small dataset with a balanced mix of near-pole and far-pole
configurations and saves it to data/rr_ik_overhead.json by default.

Run:
  python examples/make_overhead_dataset.py --n 2000 --singular_ratio 0.5 --output data/rr_ik_overhead.json
"""

from __future__ import annotations

import argparse
from typing import Tuple

from examples.robotics.rr_ik_dataset import RobotConfig, RRDatasetGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="Number of samples")
    p.add_argument("--singular_ratio", type=float, default=0.5, help="Fraction near singularities")
    p.add_argument("--displacement_scale", type=float, default=0.1, help="Δx/Δy scale")
    p.add_argument("--damping_factor", type=float, default=0.01, help="DLS damping λ")
    p.add_argument(
        "--threshold", type=float, default=1e-4, help="Singularity threshold for sampling"
    )
    p.add_argument(
        "--min_detj_regular", type=float, default=1e-3, help="Minimum |detJ| for regular samples"
    )
    p.add_argument("--output", type=str, default="data/rr_ik_overhead.json", help="Output JSON")
    args = p.parse_args()

    gen = RRDatasetGenerator(RobotConfig())
    cfgs = gen.sample_configurations(
        n_samples=args.n,
        singular_ratio=args.singular_ratio,
        singularity_threshold=args.threshold,
        force_exact_singularities=True,
        min_detj_regular=args.min_detj_regular,
    )
    samples = gen.generate_ik_samples(
        cfgs,
        displacement_scale=args.displacement_scale,
        damping_factor=args.damping_factor,
        vectorized=True,
    )
    gen.samples = samples
    gen.save_dataset(args.output, format="json")


if __name__ == "__main__":
    main()
