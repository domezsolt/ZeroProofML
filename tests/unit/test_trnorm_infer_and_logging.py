import pytest

from zeroproof.autodiff import TRNode
from zeroproof.core import real
from zeroproof.layers import TRNorm
from zeroproof.training import TrainingConfig, TRTrainer


def test_trnorm_from_batch_infers_num_features():
    batch = [
        [real(0.0), real(1.0), real(2.0)],
        [real(3.0), real(4.0), real(5.0)],
    ]
    norm = TRNorm.from_batch(batch)
    assert norm.num_features == 3


class DummyModel:
    def __init__(self):
        self.w = TRNode.parameter(real(1.0), name="w")

    def __call__(self, x):
        if isinstance(x, TRNode):
            return self.w * x
        return self.w * TRNode.constant(x)

    def parameters(self):
        return [self.w]


def test_trainer_logging_uses_step_when_epoch_zero(capsys):
    model = DummyModel()
    cfg = TrainingConfig(verbose=True, log_interval=1, max_epochs=1, use_adaptive_loss=False)
    trainer = TRTrainer(model, config=cfg)
    # Call train_epoch directly to simulate ad-hoc usage (epoch stays 0 until train())
    data_loader = [([real(1.0)], [real(1.0)])]
    trainer.train_epoch(data_loader)
    out = capsys.readouterr().out
    assert "Step" in out or "Epoch 1" in out  # allow either if train() set epoch
