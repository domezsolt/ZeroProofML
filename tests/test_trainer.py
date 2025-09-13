from zeroproof.layers import MonomialBasis
from zeroproof.layers.multi_input_rational import TRMultiInputRational
from zeroproof.training import HybridTRTrainer, HybridTrainingConfig, Optimizer


def test_persistent_optimizers():
    model = TRMultiInputRational(input_dim=4, n_outputs=2, d_p=2, d_q=1,
                                 basis=MonomialBasis(), hidden_dims=[4], shared_Q=True, enable_pole_head=False)
    trainer = HybridTRTrainer(model=model,
                              optimizer=Optimizer(model.parameters(), learning_rate=0.01),
                              config=HybridTrainingConfig())
    # Persistent optimizers should be initialized for heads and frontend
    assert trainer.head_optimizers is not None
    assert len(trainer.head_optimizers) == len(model.heads)
    # Frontend optimizer should be present when frontend params exist
    assert hasattr(model, 'layers') and model.layers
    assert trainer.frontend_optimizer is not None
