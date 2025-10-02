#!/usr/bin/env python3
"""
Trajectory rollout near singularities (control-style stress test).

Trains quick models (MLP, Rational+ε, TR-Basic, TR-Full) and rolls out
short differential IK sequences that skim near-singular regions.

Reports per-method:
 - mean tracking error in task space (||achieved Δx,Δy - commanded||)
 - max joint step (proxy for actuator saturation)
 - failure rate: % steps with non-REAL outputs (tags/NaNs/Infs)

Outputs JSON summary and optional figure placeholder.
"""

import argparse
import json
import math
import os

# Path safety: ensure repo root
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.baselines.mlp_baseline import MLPBaseline, MLPConfig, MLPTrainer
from examples.baselines.rational_eps_baseline import (
    RationalEpsConfig,
    RationalEpsModel,
    RationalEpsTrainer,
    grid_search_epsilon,
)
from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import TRMultiInputRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer


def fwd_kin(theta1: float, theta2: float, L1: float = 1.0, L2: float = 1.0) -> Tuple[float, float]:
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y


def prepare_data(
    ds: Dict[str, Any], max_train: int = 2000
) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    samples = ds["samples"]
    inputs = [[s["theta1"], s["theta2"], s["dx"], s["dy"]] for s in samples]
    targets = [[s["dtheta1"], s["dtheta2"]] for s in samples]
    n_train_full = int(0.8 * len(inputs))
    train_idx = list(range(n_train_full))
    test_idx = list(range(n_train_full, len(inputs)))
    train_idx = train_idx[: min(max_train, len(train_idx))]
    train = ([inputs[i] for i in train_idx], [targets[i] for i in train_idx])
    test = ([inputs[i] for i in test_idx], [targets[i] for i in test_idx])
    return train, test


def train_models_quick(train, val) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    # MLP quick
    mlp_cfg = MLPConfig(epochs=1, hidden_dims=[32, 16], learning_rate=1e-2, batch_size=32)
    mlp = MLPBaseline(mlp_cfg)
    mlp_tr = MLPTrainer(mlp, Optimizer(mlp.parameters(), learning_rate=mlp_cfg.learning_rate))
    mlp_tr.train(*train, *val, verbose=True)
    models["MLP"] = mlp

    # Rational+ε quick
    rat_cfg = RationalEpsConfig(
        epochs=3, epsilon_values=[1e-4, 1e-3], learning_rate=1e-2, batch_size=32
    )
    gs = grid_search_epsilon(train, val, rat_cfg, output_dir="results/robotics/rollout_rational")
    best_eps = gs.get("best_epsilon", 1e-4)
    rat = RationalEpsModel(rat_cfg, best_eps)
    rat_tr = RationalEpsTrainer(
        rat, Optimizer(rat.parameters(), learning_rate=rat_cfg.learning_rate)
    )
    rat_tr.train(*train, *val, verbose=False)
    models["Rational+ε"] = rat

    # TR Basic
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    tr_basic = TRMultiInputRational(
        input_dim=4, n_outputs=2, d_p=3, d_q=2, hidden_dims=[8], shared_Q=True
    )
    # tiny loop
    for epoch in range(3):
        for inp, tgt in zip(*train):
            tr_inputs = [TRNode.constant(real(x)) for x in inp]
            outs = tr_basic.forward(tr_inputs)
            loss = TRNode.constant(real(0.0))
            valid = 0
            for j, (y, tag) in enumerate(outs):
                if tag == TRTag.REAL:
                    diff = y - TRNode.constant(real(tgt[j]))
                    loss = loss + diff * diff
                    valid += 1
            if valid == 0:
                continue
            loss.backward()
            Optimizer(tr_basic.parameters(), learning_rate=1e-2).step(tr_basic)
    models["ZeroProofML-Basic"] = tr_basic

    # TR Full (Hybrid)
    tr_full = TRMultiInputRational(
        input_dim=4, n_outputs=2, d_p=3, d_q=2, hidden_dims=[8], shared_Q=True
    )
    tr_trainer = HybridTRTrainer(
        model=tr_full,
        optimizer=Optimizer(tr_full.parameters(), learning_rate=1e-2),
        config=HybridTrainingConfig(
            learning_rate=1e-2,
            max_epochs=3,
            use_hybrid_gradient=True,
            use_tag_loss=True,
            lambda_tag=0.05,
            use_pole_head=True,
            lambda_pole=0.1,
            enable_anti_illusion=True,
            lambda_residual=0.02,
        ),
    )
    # emulate simple epochs
    for epoch in range(3):
        for inp, tgt in zip(*train):
            tr_inputs = [TRNode.constant(real(x)) for x in inp]
            outs = tr_full.forward(tr_inputs)
            loss = TRNode.constant(real(0.0))
            valid = 0
            for j, (y, tag) in enumerate(outs):
                if tag == TRTag.REAL:
                    diff = y - TRNode.constant(real(tgt[j]))
                    loss = loss + diff * diff
                    valid += 1
            if valid == 0:
                continue
            loss.backward()
            if hasattr(tr_trainer, "step_all"):
                tr_trainer.step_all()
            else:
                tr_trainer.optimizer.step(tr_full)
    models["ZeroProofML-Full"] = tr_full

    return models


def predict_step(
    method: str, model, theta1: float, theta2: float, dx: float, dy: float
) -> Tuple[float, float, bool]:
    non_real = False
    # Build TRNodes
    if method == "Rational+ε":
        # follow baseline convention: use first input only
        x = TRNode.constant(real(theta1))
        outs = model.forward_with_eps_regularization(x)
        vals = []
        for out in outs:
            if out.tag != TRTag.REAL or math.isnan(out.value.value) or math.isinf(out.value.value):
                non_real = True
                vals.append(0.0)
            else:
                vals.append(float(out.value.value))
        d1, d2 = (vals + [0.0, 0.0])[:2]
        return d1, d2, non_real
    elif method.startswith("ZeroProofML"):
        tr_inputs = [
            TRNode.constant(real(theta1)),
            TRNode.constant(real(theta2)),
            TRNode.constant(real(dx)),
            TRNode.constant(real(dy)),
        ]
        outs = model.forward(tr_inputs)
        vals = []
        for y, tag in outs:
            if tag != TRTag.REAL:
                non_real = True
                vals.append(0.0)
            else:
                vals.append(float(y.value.value))
        d1, d2 = (vals + [0.0, 0.0])[:2]
        return d1, d2, non_real
    else:  # MLP
        tr_inputs = [
            TRNode.constant(real(theta1)),
            TRNode.constant(real(theta2)),
            TRNode.constant(real(dx)),
            TRNode.constant(real(dy)),
        ]
        outs = model.forward(tr_inputs)
        vals = []
        for out in outs:
            if out.tag != TRTag.REAL or math.isnan(out.value.value) or math.isinf(out.value.value):
                non_real = True
                vals.append(0.0)
            else:
                vals.append(float(out.value.value))
        d1, d2 = (vals + [0.0, 0.0])[:2]
        return d1, d2, non_real


def rollout(
    models: Dict[str, Any],
    n_traj: int = 4,
    n_steps: int = 50,
    start_theta2: float = 0.03,
    step_size: float = 0.01,
    phi_list: List[float] = (0.0, 90.0),
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    # Build desired displacement vectors per time step
    phis = [math.radians(p) for p in phi_list]
    dvecs = [(step_size * math.cos(p), step_size * math.sin(p)) for p in phis]

    # Generate trajectories
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    for name, model in models.items():
        total_err = []
        max_joint = 0.0
        non_real_steps = 0
        total_steps = 0
        for sign in (+1, -1):
            for dvx, dvy in dvecs:
                theta1 = np.random.uniform(-math.pi, math.pi)
                theta2 = sign * start_theta2
                x0, y0 = fwd_kin(theta1, theta2)
                for t in range(n_steps):
                    d1, d2, bad = predict_step(name, model, theta1, theta2, dvx, dvy)
                    theta1 = wrap(theta1 + d1)
                    theta2 = wrap(theta2 + d2)
                    x1, y1 = fwd_kin(theta1, theta2)
                    ach_dx, ach_dy = (x1 - x0), (y1 - y0)
                    err = math.hypot(ach_dx - dvx, ach_dy - dvy)
                    total_err.append(err)
                    max_joint = max(max_joint, abs(d1), abs(d2))
                    if bad:
                        non_real_steps += 1
                    total_steps += 1
                    x0, y0 = x1, y1
        results[name] = {
            "mean_tracking_error": float(np.mean(total_err)) if total_err else float("inf"),
            "max_joint_step": float(max_joint),
            "failure_rate": (100.0 * non_real_steps / total_steps) if total_steps else 0.0,
            "total_steps": total_steps,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description="Trajectory rollout near singularities")
    ap.add_argument("--dataset", default="data/rr_ik_dataset.json")
    ap.add_argument("--out", default="results/robotics/rollout_summary.json")
    ap.add_argument("--max_train", type=int, default=2000)
    ap.add_argument("--n_traj", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--start_theta2", type=float, default=0.03)
    ap.add_argument("--step_size", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.dataset, "r") as fh:
        ds = json.load(fh)
    train, val = prepare_data(ds, max_train=args.max_train)
    models = train_models_quick(train, val)
    summary = rollout(
        models,
        n_traj=args.n_traj,
        n_steps=args.n_steps,
        start_theta2=args.start_theta2,
        step_size=args.step_size,
        phi_list=[0.0, 90.0],
    )
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Rollout summary:")
    for k, v in summary.items():
        print(
            f"{k:18s}  mean_err={v['mean_tracking_error']:.4f}  max_dtheta={v['max_joint_step']:.4f}  failure%={v['failure_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
