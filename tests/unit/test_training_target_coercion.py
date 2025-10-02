import numpy as np
import pytest

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import TRRational
from zeroproof.training import TrainingConfig, TRTrainer


def make_model():
    # y = (1 + x) / (1 + 0.5 x)
    layer = TRRational(d_p=1, d_q=1)
    layer.theta[0]._value = real(1.0)
    layer.theta[1]._value = real(1.0)
    layer.phi[0]._value = real(0.5)
    return layer


def test_trainer_accepts_python_float_targets():
    model = make_model()
    trainer = TRTrainer(model, config=TrainingConfig(use_adaptive_loss=False))
    inputs = [real(0.0), real(1.0)]
    targets = [0.0, 1.0]  # python floats
    metrics = trainer.train_step(inputs, targets)
    assert "loss" in metrics


def test_trainer_accepts_numpy_targets():
    model = make_model()
    trainer = TRTrainer(model, config=TrainingConfig(use_adaptive_loss=False))
    inputs = [real(0.0), real(1.0)]
    # numpy array of floats
    targets = np.asarray([0.0, 1.0], dtype=float)
    metrics = trainer.train_step(inputs, list(targets))
    assert "loss" in metrics


def test_adaptive_loss_accepts_mixed_target_types():
    model = make_model()
    cfg = TrainingConfig(use_adaptive_loss=True)
    trainer = TRTrainer(model, config=cfg)
    inputs = [real(0.0), real(1.0), real(2.0)]
    # mix TRScalar, python float, numpy scalar
    targets = [real(0.0), 1.0, np.float64(2.0)]
    metrics = trainer.train_step(inputs, targets)
    assert "loss" in metrics
