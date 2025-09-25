"""Curvature safeguard logging tests.

Verifies that HybridTRTrainer logs curvature and gradient proxies,
and that they are finite in a simple 1D setting.
"""

from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.training import HybridTRTrainer, HybridTrainingConfig, Optimizer
from zeroproof.core import real


def _trscalar_list(vals):
    return [real(float(v)) for v in vals]


def test_curvature_and_grad_proxies_present_and_finite():
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    # Make Q well away from zero for |x|<=1: Q(x) = 1 + 0.5 x
    model.phi[0]._value = real(0.5)

    opt = Optimizer(model.parameters(), learning_rate=0.01)
    cfg = HybridTrainingConfig(max_epochs=1, batch_size=3, verbose=False)
    trainer = HybridTRTrainer(model=model, optimizer=opt, config=cfg)

    # One epoch, one batch
    inputs = _trscalar_list([-0.5, 0.0, 0.7])
    targets = _trscalar_list([0.0, 0.0, 0.0])
    metrics = trainer.train_epoch([(inputs, targets)])

    # Curvature and grad proxies should be present
    assert 'curvature_proxy' in metrics
    assert 'gn_proxy' in metrics
    assert 'grad_max' in metrics

    # Finite values
    assert metrics['curvature_proxy'] >= 0.0
    assert metrics['gn_proxy'] >= 0.0
    assert metrics['grad_max'] >= 0.0

