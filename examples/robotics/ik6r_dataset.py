"""
Synthetic 6R (6-DOF) inverse kinematics dataset generator.

Generates differential IK samples (twist -> joint increments) for a generic
6R serial manipulator using simple Denavit–Hartenberg kinematics.

We bucket by the smallest singular value d1 = σ_min(J) to analyze near-pole
performance; we also track d2 = σ_2nd_min(J) to flag rank≥2 deficiency.

Saved JSON schema mirrors other generators in this repo:
  {
    'config': { dh_params, damping, etc. },
    'samples': [
        {
          'q': [q1..q6],
          'twist': [vx,vy,vz, wx,wy,wz],
          'dq_target': [dq1..dq6],
          'd1': float, 'd2': float,
          'd1_v': float, 'd1_w': float,
          'bin_idx': int,
          'is_multi': bool
        }, ...
    ],
    'metadata': { bucket_edges, train/test counts, seed, etc. }
  }
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from zeroproof.utils.config import DEFAULT_BUCKET_EDGES
from zeroproof.utils.env import collect_env_info
from zeroproof.utils.seeding import set_global_seed
from zeroproof.utils.serialization import to_builtin


@dataclass
class DH6R:
    """Simple DH parameter set for a generic 6R arm."""

    a: Tuple[float, float, float, float, float, float] = (0.35, 0.30, 0.20, 0.0, 0.0, 0.0)
    d: Tuple[float, float, float, float, float, float] = (0.10, 0.00, 0.00, 0.08, 0.08, 0.06)
    alpha: Tuple[float, float, float, float, float, float] = (
        math.pi / 2,
        0.0,
        0.0,
        math.pi / 2,
        -math.pi / 2,
        0.0,
    )


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def fk_and_jacobian(
    dh: DH6R, q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Compute end-effector pose (T) and geometric Jacobian J (6x6).

    Returns (T, J, p_end, origins, z_axes), where p_end is 3D position.
    """
    Ts: List[np.ndarray] = []
    T = np.eye(4, dtype=float)
    origins: List[np.ndarray] = [T[:3, 3].copy()]
    z_axes: List[np.ndarray] = [T[:3, 2].copy()]  # z of base
    for i in range(6):
        Ti = _dh_transform(dh.a[i], dh.alpha[i], dh.d[i], float(q[i]))
        T = T @ Ti
        Ts.append(T.copy())
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())
    p_end = origins[-1]
    # Geometric Jacobian in base frame
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        zi = z_axes[i]
        pi = origins[i]
        J[:3, i] = np.cross(zi, p_end - pi)  # linear
        J[3:, i] = zi  # angular
    return Ts[-1], J, p_end, origins, z_axes


def dls_dq(J: np.ndarray, dx: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    JJt = J @ J.T
    lamI = (lam**2) * np.eye(JJt.shape[0], dtype=float)
    try:
        inv = np.linalg.inv(JJt + lamI)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(JJt + lamI)
    dq = J.T @ inv @ dx
    return dq


def sv_metrics(J: np.ndarray) -> Tuple[float, float, float, float]:
    s = np.linalg.svd(J, compute_uv=False)
    s = np.array(sorted([float(x) for x in s], reverse=True))
    if s.size < 6:
        # pad
        s = np.pad(s, (0, 6 - s.size), constant_values=0.0)
    d1 = float(s[-1])
    d2 = float(s[-2])
    return float(np.min(s)), float(np.max(s)), d1, d2


def split_jacobian(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return J[:3, :], J[3:, :]


def compute_bins(vals: List[float], edges: List[float]) -> List[int]:
    out: List[int] = []
    for v in vals:
        b = 0
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (v >= lo if i == 0 else v > lo) and v <= hi:
                b = i
                break
        out.append(b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 6R synthetic IK dataset with d1 binning")
    ap.add_argument("--output", type=str, default="data/ik6r_dataset.json")
    ap.add_argument("--n_samples", type=int, default=24000)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument(
        "--singular_ratio",
        type=float,
        default=0.35,
        help="Target fraction of near-singularity samples (d1 small)",
    )
    ap.add_argument(
        "--d1_threshold", type=float, default=1e-3, help="Threshold for near-singularity (by d1)"
    )
    ap.add_argument(
        "--d2_threshold",
        type=float,
        default=5e-3,
        help="Multi-singularity threshold (2nd smallest)",
    )
    ap.add_argument(
        "--twist_pos_scale",
        type=float,
        default=0.03,
        help="Std dev for translational component (m)",
    )
    ap.add_argument(
        "--twist_rot_scale", type=float, default=0.03, help="Std dev for rotational component (rad)"
    )
    ap.add_argument("--damping", type=float, default=0.01, help="DLS damping λ")
    ap.add_argument(
        "--bucket_edges",
        type=str,
        default=None,
        help="Comma-separated edges for d1 binning (last can be inf)",
    )
    ap.add_argument("--ensure_buckets_nonzero", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    set_global_seed(args.seed)
    edges: List[float]
    if args.bucket_edges:
        edges = []
        for p in [p.strip() for p in args.bucket_edges.split(",") if p.strip()]:
            edges.append(float("inf") if p.lower() in ("inf", "+inf") else float(p))
    else:
        edges = DEFAULT_BUCKET_EDGES

    dh = DH6R()
    n_sing = int(round(args.singular_ratio * args.n_samples))
    n_reg = args.n_samples - n_sing

    samples: List[Dict[str, Any]] = []

    # Helper to draw a twist
    def draw_twist() -> np.ndarray:
        vx, vy, vz = np.random.normal(0.0, args.twist_pos_scale, size=3)
        wx, wy, wz = np.random.normal(0.0, args.twist_rot_scale, size=3)
        return np.array([vx, vy, vz, wx, wy, wz], dtype=float)

    # Draw near-singular by rejection on d1
    def draw_q_near() -> Tuple[np.ndarray, float, float, float, float, float, float]:
        attempts = 0
        while True:
            attempts += 1
            q = np.random.uniform(-math.pi, math.pi, size=(6,)).astype(float)
            _, J, _, _, _ = fk_and_jacobian(dh, q)
            smin, smax, d1, d2 = sv_metrics(J)
            if d1 <= args.d1_threshold or attempts > 500:
                Jv, Jw = split_jacobian(J)
                d1_v = float(np.linalg.svd(Jv, compute_uv=False).min()) if Jv.size else float("nan")
                d1_w = float(np.linalg.svd(Jw, compute_uv=False).min()) if Jw.size else float("nan")
                return q, d1, d2, smin, smax, d1_v, d1_w

    # Draw regular far from singularities
    def draw_q_reg() -> Tuple[np.ndarray, float, float, float, float, float, float]:
        attempts = 0
        min_reg = max(args.d1_threshold * 3.0, 1e-4)
        while True:
            attempts += 1
            q = np.random.uniform(-math.pi, math.pi, size=(6,)).astype(float)
            _, J, _, _, _ = fk_and_jacobian(dh, q)
            smin, smax, d1, d2 = sv_metrics(J)
            if d1 >= min_reg or attempts > 300:
                Jv, Jw = split_jacobian(J)
                d1_v = float(np.linalg.svd(Jv, compute_uv=False).min()) if Jv.size else float("nan")
                d1_w = float(np.linalg.svd(Jw, compute_uv=False).min()) if Jw.size else float("nan")
                return q, d1, d2, smin, smax, d1_v, d1_w

    # Generate
    for _ in range(n_sing):
        q, d1, d2, smin, smax, d1_v, d1_w = draw_q_near()
        twist = draw_twist()
        _, J, _, _, _ = fk_and_jacobian(dh, q)
        dq = dls_dq(J, twist, lam=args.damping)
        samples.append(
            {
                "q": [float(x) for x in q.tolist()],
                "twist": [float(x) for x in twist.tolist()],
                "dq_target": [float(x) for x in dq.tolist()],
                "d1": float(d1),
                "d2": float(d2),
                "d1_v": float(d1_v),
                "d1_w": float(d1_w),
                "is_multi": bool(d2 < args.d2_threshold),
            }
        )
    for _ in range(n_reg):
        q, d1, d2, smin, smax, d1_v, d1_w = draw_q_reg()
        twist = draw_twist()
        _, J, _, _, _ = fk_and_jacobian(dh, q)
        dq = dls_dq(J, twist, lam=args.damping)
        samples.append(
            {
                "q": [float(x) for x in q.tolist()],
                "twist": [float(x) for x in twist.tolist()],
                "dq_target": [float(x) for x in dq.tolist()],
                "d1": float(d1),
                "d2": float(d2),
                "d1_v": float(d1_v),
                "d1_w": float(d1_w),
                "is_multi": bool(d2 < args.d2_threshold),
            }
        )

    # Stratify by d1 buckets into train/test
    d1_vals = [float(s["d1"]) for s in samples]
    bin_idx = compute_bins(d1_vals, edges)
    for s, b in zip(samples, bin_idx):
        s["bin_idx"] = int(b)
    # Bucket lists
    buckets: List[List[int]] = [[] for _ in range(len(edges) - 1)]
    for i, b in enumerate(bin_idx):
        buckets[b].append(i)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for blist in buckets:
        k = int(round(args.train_ratio * len(blist)))
        train_idx.extend(blist[:k])
        test_idx.extend(blist[k:])

    # Optionally ensure non-empty first 4 buckets in both splits (augment by duplication if needed)
    if args.ensure_buckets_nonzero:

        def ensure_nonzero(sub_idx: List[int]) -> List[int]:
            counts = [0] * (len(edges) - 1)
            for i in sub_idx:
                counts[bin_idx[i]] += 1
            out = sub_idx[:]
            for b in range(min(4, len(counts))):
                if counts[b] == 0 and buckets[b]:
                    out.append(buckets[b][0])
                    counts[b] = 1
            return out

        train_idx = ensure_nonzero(train_idx)
        test_idx = ensure_nonzero(test_idx)

    # Reorder samples as train then test
    ordered = [samples[i] for i in train_idx] + [samples[i] for i in test_idx]

    # Counts
    def bucket_counts(idxs: List[int]) -> List[int]:
        c = [0] * (len(edges) - 1)
        for i in idxs:
            c[bin_idx[i]] += 1
        return [int(x) for x in c]

    data = {
        "config": to_builtin(
            {
                "dh": asdict(dh),
                "damping": float(args.damping),
                "singular_ratio": float(args.singular_ratio),
                "d1_threshold": float(args.d1_threshold),
                "d2_threshold": float(args.d2_threshold),
                "twist_pos_scale": float(args.twist_pos_scale),
                "twist_rot_scale": float(args.twist_rot_scale),
            }
        ),
        "samples": to_builtin(ordered),
        "metadata": to_builtin(
            {
                "n_samples": len(samples),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_ratio": float(args.train_ratio),
                "bucket_edges": edges,
                "train_bucket_counts": bucket_counts(train_idx),
                "test_bucket_counts": bucket_counts(test_idx),
                "stratified_by_d1": True,
                "seed": args.seed,
                "env": collect_env_info(),
            }
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved 6R dataset to {args.output}")
    print(f"Train {len(train_idx)} / Test {len(test_idx)} | First 4 bucket counts (train/test):")
    print(
        "  ",
        data["metadata"]["train_bucket_counts"][:4],
        "/",
        data["metadata"]["test_bucket_counts"][:4],
    )


if __name__ == "__main__":
    main()
