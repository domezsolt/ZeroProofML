#!/usr/bin/env python3
"""
Compact plotting utility for ZeroProofML training metrics.

Reads a results JSON (as produced by examples/robotics/rr_ik_train.py) and
plots key hybrid/policy/safeguard metrics over epochs:
  - flip_rate
  - saturating_ratio
  - tau_q_on / tau_q_off
  - curvature_bound (log scale)

Usage:
  python scripts/plot_training_curves.py --results runs/ik_experiment/results_tr_rat.json \
      --outdir runs/ik_experiment
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25


def _load_history(results_path: str) -> List[Dict[str, Any]]:
    with open(results_path, "r") as f:
        data = json.load(f)
    hist = []
    try:
        ts = data.get("training_summary", {})
        hist = ts.get("final_metrics", []) or []
        # Some result variants store history under a different key; be permissive
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    return hist


def _extract_series(hist: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for m in hist:
        v = m.get(key, None)
        if isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(float("nan"))
    return out


def plot_training_curves(results_path: str, outdir: str | None = None) -> str:
    hist = _load_history(results_path)
    if not hist:
        raise RuntimeError(f"No training history found in {results_path}")

    epochs = list(range(1, len(hist) + 1))

    flip_rate = _extract_series(hist, "flip_rate")
    sat_ratio = _extract_series(hist, "saturating_ratio")
    tau_on = _extract_series(hist, "tau_q_on")
    tau_off = _extract_series(hist, "tau_q_off")
    curv = _extract_series(hist, "curvature_bound")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]

    # Flip rate
    ax1.plot(epochs, flip_rate, color="#1f77b4", lw=1.8)
    ax1.set_title("Flip Rate (policy hysteresis)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("flip_rate")
    ax1.set_ylim(bottom=0.0)

    # Saturating ratio
    ax2.plot(epochs, sat_ratio, color="#ff7f0e", lw=1.8)
    ax2.set_title("Saturating Ratio")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ratio")
    ax2.set_ylim(bottom=0.0)

    # Tau thresholds
    ax3.plot(epochs, tau_on, label="tau_q_on", color="#2ca02c", lw=1.8)
    ax3.plot(epochs, tau_off, label="tau_q_off", color="#d62728", lw=1.8)
    ax3.set_title("Policy Thresholds (|Q| guard bands)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("tau")
    ax3.legend(frameon=False)

    # Curvature bound (log)
    # Avoid issues with nonpositive values
    safe_curv = [c if (isinstance(c, float) and c > 0) else float("nan") for c in curv]
    ax4.semilogy(epochs, safe_curv, color="#9467bd", lw=1.8)
    ax4.set_title("Curvature Bound (contract)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("bound (log)")

    fig.tight_layout()

    base = os.path.splitext(os.path.basename(results_path))[0]
    outdir = outdir or os.path.dirname(results_path) or "."
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"training_curves_{base}.png")
    out_pdf = os.path.join(outdir, f"training_curves_{base}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    try:
        fig.savefig(out_pdf, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ZeroProofML training curves")
    ap.add_argument("--results", required=True, help="Path to results_*.json")
    ap.add_argument("--outdir", default=None, help="Directory for output figures")
    args = ap.parse_args()
    out = plot_training_curves(args.results, args.outdir)
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()

