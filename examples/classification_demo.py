"""
Tiny TR classification demo using tr_softmax and TR autodiff.

Generates a simple 2D dataset (two Gaussian blobs), trains a linear
classifier with a TR-safe softmax and cross-entropy, and prints loss/accuracy.

Run:
  python examples/classification_demo.py --epochs 50 --n 200 --lr 0.2
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real, tr_add, tr_div, tr_log, tr_mul, tr_sum
from zeroproof.layers import tr_softmax
from zeroproof.training.trainer import Optimizer


def make_blobs(n: int, seed: int = 0) -> Tuple[List[Tuple[float, float]], List[int]]:
    random.seed(seed)
    xs: List[Tuple[float, float]] = []
    ys: List[int] = []
    for _ in range(n // 2):
        xs.append((-1.0 + random.gauss(0, 0.4), -1.0 + random.gauss(0, 0.4)))
        ys.append(0)
    for _ in range(n - n // 2):
        xs.append((1.0 + random.gauss(0, 0.4), 1.0 + random.gauss(0, 0.4)))
        ys.append(1)
    return xs, ys


class TinyClassifier:
    def __init__(self):
        # Parameters for 2-class linear model: logits = W x + b
        self.W = [
            [TRNode.parameter(real(0.0), name="w00"), TRNode.parameter(real(0.0), name="w01")],
            [TRNode.parameter(real(0.0), name="w10"), TRNode.parameter(real(0.0), name="w11")],
        ]
        self.b = [TRNode.parameter(real(0.0), name="b0"), TRNode.parameter(real(0.0), name="b1")]

    def parameters(self) -> List[TRNode]:
        ps: List[TRNode] = []
        for row in self.W:
            ps.extend(row)
        ps.extend(self.b)
        return ps

    def logits(self, x0: TRNode, x1: TRNode) -> List[TRNode]:
        # y_i = w_i0*x0 + w_i1*x1 + b_i
        out: List[TRNode] = []
        for i in range(2):
            s = tr_add(tr_add(tr_mul(self.W[i][0], x0), tr_mul(self.W[i][1], x1)), self.b[i])
            out.append(s)
        return out

    def forward(self, x0: TRNode, x1: TRNode) -> List[TRNode]:
        return tr_softmax(self.logits(x0, x1))


def nll(probs: List[TRNode], y: int) -> TRNode:
    # Negative log likelihood for true class y
    p = probs[y]
    # -log p, TR-safe (Padé softmax ensures p>0 in REAL regions)
    return TRNode.constant(real(0.0)) - tr_log(p)


def accuracy(model: TinyClassifier, X: List[Tuple[float, float]], y: List[int]) -> float:
    correct = 0
    for (x0, x1), yi in zip(X, y):
        p = model.forward(TRNode.constant(real(x0)), TRNode.constant(real(x1)))
        vals = [float(pi.value.value) if pi.tag == TRTag.REAL else float("nan") for pi in p]
        pred = 0 if vals[0] >= vals[1] else 1
        if pred == yi:
            correct += 1
    return correct / max(1, len(X))


def train(epochs: int, n: int, lr: float, seed: int) -> None:
    X, y = make_blobs(n, seed=seed)
    model = TinyClassifier()
    opt = Optimizer(model.parameters(), learning_rate=lr)

    for epoch in range(epochs):
        # Build full-batch loss
        losses: List[TRNode] = []
        for (x0f, x1f), yi in zip(X, y):
            x0 = TRNode.constant(real(x0f))
            x1 = TRNode.constant(real(x1f))
            probs = model.forward(x0, x1)
            losses.append(nll(probs, yi))

        # Mean loss: sum / n
        total = tr_sum([l.value for l in losses])
        loss = TRNode.constant(tr_div(total, real(float(len(losses)))))

        # Backprop and step
        opt.zero_grad()
        loss.backward()
        opt.step(model)

        # Report
        loss_val = float(loss.value.value) if loss.value.tag == TRTag.REAL else float("nan")
        acc = accuracy(model, X, y)
        print(f"Epoch {epoch+1:3d} | loss={loss_val:.4f} | acc={acc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.epochs, args.n, args.lr, args.seed)


if __name__ == "__main__":
    main()
