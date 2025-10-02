"""
Smooth-regularized rational baseline.

Implements P / sqrt(Q^2 + α^2) to avoid exact zeros with a smooth substitute.
Grid-searches α ∈ {1e-1,1e-2,1e-3} by default.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.autodiff.tr_ops_grad import tr_sqrt
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import Optimizer


@dataclass
class SmoothConfig:
    input_dim: int = 4
    output_dim: int = 2
    degree_p: int = 3
    degree_q: int = 2
    alpha_values: List[float] = None
    learning_rate: float = 0.01
    epochs: int = 60
    batch_size: int = 32
    l2_regularization: float = 1e-3

    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [1e-1, 1e-2, 1e-3]


class SmoothRationalModel:
    def __init__(self, cfg: SmoothConfig, alpha: float):
        self.config = cfg
        self.alpha = float(alpha)
        self.basis = MonomialBasis()
        self.rationals: List[TRRational] = []
        for _ in range(cfg.output_dim):
            self.rationals.append(TRRational(d_p=cfg.degree_p, d_q=cfg.degree_q, basis=self.basis))

    def parameters(self) -> List[TRNode]:
        params: List[TRNode] = []
        for r in self.rationals:
            params.extend(r.parameters())
        return params

    def _forward_one(self, rational: TRRational, x: TRNode) -> TRNode:
        max_degree = max(len(rational.theta), len(rational.phi) + 1)
        psi = self.basis(x, max_degree)
        P = TRNode.constant(real(0.0))
        for k, theta_k in enumerate(rational.theta):
            if k < len(psi):
                P = P + theta_k * psi[k]
        Q = TRNode.constant(real(1.0))
        for k, phi_k in enumerate(rational.phi):
            idx = k + 1
            if idx < len(psi):
                Q = Q + phi_k * psi[idx]
        # denom = sqrt(Q^2 + α^2)
        Q2 = Q * Q
        a2 = TRNode.constant(real(self.alpha * self.alpha))
        denom = tr_sqrt(Q2 + a2)
        return P / denom

    def forward(self, inputs: List[TRNode]) -> List[TRNode]:
        x = inputs[0] if inputs else TRNode.constant(real(0.0))
        return [self._forward_one(r, x) for r in self.rationals]

    def regularization_loss(self) -> TRNode:
        l2 = TRNode.constant(real(0.0))
        for r in self.rationals:
            for p in r.parameters():
                l2 = l2 + p * p
        return TRNode.constant(real(self.config.l2_regularization)) * l2


class SmoothTrainer:
    def __init__(self, model: SmoothRationalModel, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.history: List[Dict[str, Any]] = []
        self.training_time = 0.0

    def _step(self, inp: List[float], tgt: List[float]) -> float:
        tr_in = [TRNode.constant(real(x)) for x in inp]
        outs = self.model.forward(tr_in)
        loss = TRNode.constant(real(0.0))
        valid = 0
        for y, t in zip(outs, tgt):
            if (
                y.tag == TRTag.REAL
                and (not math.isnan(y.value.value))
                and (not math.isinf(y.value.value))
            ):
                diff = y - TRNode.constant(real(float(t)))
                loss = loss + diff * diff
                valid += 1
        if valid == 0:
            return 0.0
        loss = loss + self.model.regularization_loss()
        loss.backward()
        self.optimizer.step(self.model)
        return float(loss.value.value) if loss.tag == TRTag.REAL else 0.0

    def train(
        self,
        train_inputs: List[List[float]],
        train_targets: List[List[float]],
        val_inputs=None,
        val_targets=None,
    ) -> Dict[str, Any]:
        GradientModeConfig.set_mode(GradientMode.MASK_REAL)
        t0 = time.time()
        bs = self.model.config.batch_size
        for ep in range(self.model.config.epochs):
            losses = []
            for i in range(0, len(train_inputs), bs):
                batch_in = train_inputs[i : i + bs]
                batch_tg = train_targets[i : i + bs]
                for inp, tgt in zip(batch_in, batch_tg):
                    losses.append(self._step(inp, tgt))
            if losses:
                self.history.append({"epoch": ep, "loss": float(np.mean(losses))})
                if ep % max(1, self.model.config.epochs // 10) == 0:
                    print(
                        f"[Smooth α={self.model.alpha}] Epoch {ep}: loss={self.history[-1]['loss']:.6f}"
                    )
        self.training_time = time.time() - t0
        return {
            "history": self.history,
            "training_time": self.training_time,
            "config": asdict(self.model.config),
        }

    def evaluate(self, inputs: List[List[float]], targets: List[List[float]]) -> Dict[str, Any]:
        per_sample_mse: List[float] = []
        predictions: List[List[float]] = []
        for inp, tgt in zip(inputs, targets):
            tr_in = [TRNode.constant(real(x)) for x in inp]
            outs = self.model.forward(tr_in)
            pred = []
            mse = 0.0
            valid = 0
            for y, t in zip(outs, tgt):
                if (
                    y.tag == TRTag.REAL
                    and (not math.isnan(y.value.value))
                    and (not math.isinf(y.value.value))
                ):
                    val = float(y.value.value)
                    pred.append(val)
                    mse += (val - float(t)) ** 2
                    valid += 1
            if valid > 0:
                per_sample_mse.append(mse / valid)
                predictions.append(pred)
        return {
            "mse": float(np.mean(per_sample_mse)) if per_sample_mse else float("inf"),
            "per_sample_mse": per_sample_mse,
            "predictions": predictions,
        }


def grid_search_alpha(
    train_data: Tuple[List, List], val_data: Tuple[List, List], cfg: SmoothConfig, out_dir: str
) -> Dict[str, Any]:
    best_alpha = None
    best_mse = float("inf")
    results: List[Dict[str, Any]] = []
    for a in cfg.alpha_values:
        print(f"[Smooth] Trying α={a}")
        model = SmoothRationalModel(cfg, a)
        tr = SmoothTrainer(model, Optimizer(model.parameters(), learning_rate=cfg.learning_rate))
        tr.train(*train_data)
        met = tr.evaluate(*val_data)
        mse = float(met["mse"])
        results.append({"alpha": a, "mse": mse})
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smooth_grid.json"), "w") as fh:
        json.dump(
            {"results": results, "best_alpha": best_alpha, "best_mse": best_mse}, fh, indent=2
        )
    return {"best_alpha": best_alpha, "best_mse": best_mse, "results": results}


def run_smooth_baseline(
    train_data: Tuple[List, List],
    test_data: Tuple[List, List],
    cfg: Optional[SmoothConfig] = None,
    output_dir: str = "results",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = SmoothConfig()
    cfg.input_dim = len(train_data[0][0])
    cfg.output_dim = len(train_data[1][0])
    gs = grid_search_alpha(train_data, test_data, cfg, os.path.join(output_dir, "smooth_grid"))
    alpha = gs.get("best_alpha", cfg.alpha_values[-1])
    model = SmoothRationalModel(cfg, alpha)
    tr = SmoothTrainer(model, Optimizer(model.parameters(), learning_rate=cfg.learning_rate))
    trn = tr.train(*train_data)
    met = tr.evaluate(*test_data)
    res = {
        "model_type": "SmoothRational",
        "alpha": float(alpha),
        "config": asdict(cfg),
        "training_results": trn,
        "test_metrics": met,
        "n_parameters": len(model.parameters()),
        "training_time": trn["training_time"],
        "seed": seed,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"smooth_alpha_{alpha}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res
