from zeroproof.training import create_adaptive_loss


def test_adaptive_loss_configurable_args():
    policy = create_adaptive_loss(
        target_coverage=0.9,
        learning_rate=0.05,
        initial_lambda=2.0,
        base_loss="mse",
        momentum=0.8,
        warmup_steps=5,
        update_frequency=3,
        exponential_decay=0.99,
        lambda_min=0.1,
        lambda_max=10.0,
    )
    stats = policy.get_statistics()
    assert "lambda_rej" in stats
