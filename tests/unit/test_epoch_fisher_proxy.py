"""Epoch-level Fisher/GN proxy logging tests.

Ensures HybridTRTrainer emits epoch-level Fisher proxies computed from
gradient statistics.
"""

from zeroproof.core import real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer


def _trscalar_list(vals):
    return [real(float(v)) for v in vals]


def test_epoch_fisher_proxies_present():
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    model.phi[0]._value = real(0.2)

    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=0.01),
        config=HybridTrainingConfig(max_epochs=1, batch_size=3, verbose=False),
    )

    inputs = _trscalar_list([-0.2, 0.0, 0.4])
    targets = _trscalar_list([0.0, 0.0, 0.0])

    metrics = trainer.train_epoch([(inputs, targets)])

    for key in ("grad_norm_epoch", "fisher_trace", "fisher_diag_mean"):
        assert key in metrics, f"Missing {key} in metrics"
        assert metrics[key] >= 0.0
