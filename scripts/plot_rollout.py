#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot rollout metrics bars from JSON")
    ap.add_argument("--summary", default="results/robotics/rollout_summary.json")
    ap.add_argument("--out", default="results/robotics/figures/rollout_bars.png")
    args = ap.parse_args()

    with open(args.summary, "r") as fh:
        data = json.load(fh)

    methods = [
        m for m in ["MLP", "Rational+ε", "ZeroProofML-Basic", "ZeroProofML-Full"] if m in data
    ]
    mean_err = [data[m]["mean_tracking_error"] for m in methods]
    max_step = [data[m]["max_joint_step"] for m in methods]

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=150)
    axs[0].bar(methods, mean_err)
    axs[0].set_ylabel("Mean tracking error")
    axs[0].set_title("Task-space error")
    axs[0].tick_params(axis="x", rotation=15)

    axs[1].bar(methods, max_step)
    axs[1].set_ylabel("Max |Δθ| per step")
    axs[1].set_title("Joint step (saturation proxy)")
    axs[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    plt.close(fig)
    print(f"Saved rollout bars to {args.out}")


if __name__ == "__main__":
    main()
