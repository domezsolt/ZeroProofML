from zeroproof.layers import TRRational, MonomialBasis
from zeroproof.training import HybridTRTrainer, HybridTrainingConfig, Optimizer
from zeroproof.core import real
from zeroproof.autodiff import TRNode


def _trscalar_list(vals):
    return [real(float(v)) for v in vals]


def test_hybrid_trainer_bench_fields_present():
    # Simple single-input TRRational so HybridTRTrainer uses _train_batch path
    model = TRRational(d_p=1, d_q=1, basis=MonomialBasis())
    opt = Optimizer(model.parameters(), learning_rate=0.01)
    cfg = HybridTrainingConfig(max_epochs=1, batch_size=2, verbose=False)
    trainer = HybridTRTrainer(model=model, optimizer=opt, config=cfg)

    # Tiny data loader with one batch of two samples
    inputs = _trscalar_list([0.1, -0.2])
    targets = _trscalar_list([0.0, 0.0])
    data_loader = [(inputs, targets)]

    metrics = trainer.train_epoch(data_loader)

    # Bench fields should be present and non-negative
    for key in ("avg_step_ms", "data_time_ms", "optim_time_ms", "batches"):
        assert key in metrics
    assert metrics["batches"] == 1.0
    assert metrics["avg_step_ms"] >= 0.0
    assert metrics["data_time_ms"] >= 0.0
    assert metrics["optim_time_ms"] >= 0.0

    # Trainer should retain a bench history entry
    assert hasattr(trainer, 'bench_history')
    assert len(trainer.bench_history) >= 1
    rec = trainer.bench_history[-1]
    for key in ("epoch", "avg_step_ms", "data_time_ms", "optim_time_ms", "batches"):
        assert key in rec

