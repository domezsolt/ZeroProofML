"""
Epsilon-ensemble baseline: average predictions from M rational+ε models.

Each member is trained independently with a different ε; predictions are averaged.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeroproof.training import Optimizer

from .rational_eps_baseline import RationalEpsConfig, RationalEpsModel, RationalEpsTrainer


@dataclass
class EnsembleConfig:
    member_eps: List[float] = None
    learning_rate: float = 0.01
    epochs: int = 40
    batch_size: int = 32
    degree_p: int = 3
    degree_q: int = 2

    def __post_init__(self):
        if self.member_eps is None:
            self.member_eps = [1e-4, 1e-3, 1e-2]


def run_eps_ensemble_baseline(
    train_data: Tuple[List, List],
    test_data: Tuple[List, List],
    cfg: Optional[EnsembleConfig] = None,
    output_dir: str = "results",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = EnsembleConfig()
    train_inputs, train_targets = train_data
    test_inputs, test_targets = test_data
    input_dim = len(train_inputs[0])
    output_dim = len(train_targets[0])
    preds_members: List[List[List[float]]] = []
    total_time = 0.0
    for eps in cfg.member_eps:
        recfg = RationalEpsConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            degree_p=cfg.degree_p,
            degree_q=cfg.degree_q,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
        )
        model = RationalEpsModel(recfg, eps)
        trainer = RationalEpsTrainer(
            model, Optimizer(model.parameters(), learning_rate=recfg.learning_rate)
        )
        trn = trainer.train(train_inputs, train_targets, verbose=False)
        total_time += float(trn.get("training_time", 0.0))
        tm = trainer._evaluate_simple(test_inputs, test_targets)
        preds_members.append(tm.get("predictions", []))
    # Average predictions
    predictions: List[List[float]] = []
    per_sample_mse: List[float] = []
    for i in range(len(test_inputs)):
        # Collect member preds for this sample
        cols = [pm[i] for pm in preds_members if i < len(pm)]
        if not cols:
            continue
        # Average per-output
        avg = [float(np.mean([c[j] for c in cols if j < len(c)])) for j in range(len(cols[0]))]
        predictions.append(avg)
        tgt = test_targets[i]
        per_sample_mse.append(float(np.mean([(a - t) ** 2 for a, t in zip(avg, tgt)])))
    results = {
        "model_type": "EpsEnsemble",
        "members": cfg.member_eps,
        "config": asdict(cfg),
        "test_metrics": {
            "mse": float(np.mean(per_sample_mse)) if per_sample_mse else float("inf"),
            "per_sample_mse": per_sample_mse,
            "predictions": predictions,
        },
        "training_time_total": total_time,
        "seed": seed,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eps_ensemble.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results
