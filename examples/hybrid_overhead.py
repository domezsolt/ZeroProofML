"""
One-shot Hybrid vs Mask‑REAL overhead report.

Builds a small TRRational, a tiny data loader, and prints the overhead
envelope using zeroproof.utils.overhead.overhead_report.

Run:
  python examples/hybrid_overhead.py
"""

from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.utils.overhead import overhead_report


def _trscalar_list(vals):
    return [real(float(v)) for v in vals]


def main():
    model = TRRational(d_p=2, d_q=2, basis=MonomialBasis())
    # Make a modestly nontrivial Q
    model.phi[0]._value = real(0.3)
    model.phi[1]._value = real(-0.1)

    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=0.01),
        config=HybridTrainingConfig(
            max_epochs=1, batch_size=4, verbose=False, use_hybrid_gradient=True
        ),
    )

    # Tiny data loader with a couple of near-pole and far-pole points
    inputs = _trscalar_list([-0.9, -0.2, 0.1, 0.8])
    targets = _trscalar_list([0.0, 0.0, 0.0, 0.0])
    data_loader = [(inputs, targets)]

    overhead_report(trainer, data_loader)


if __name__ == "__main__":
    main()
