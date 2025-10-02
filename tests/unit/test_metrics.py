from examples.baselines.compare_all import run_zeroproof_baseline
from zeroproof.metrics.pole_2d import compute_pole_metrics_2d


def test_bucketed_mse_schema(tmp_path):
    # Tiny synthetic dataset
    train_inputs = [[0.1, -0.2, 0.01, -0.02], [0.2, 0.3, -0.01, 0.02], [0.0, 0.1, 0.0, 0.0]]
    train_targets = [[0.0, 0.0], [0.01, -0.01], [0.0, 0.0]]
    test_inputs = [[0.4, 0.0, 0.0, 0.0], [0.5, 3.14159, 0.0, 0.0], [0.2, 0.2, 0.0, 0.0]]
    test_targets = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    # |det(J)| approx for RR with L1=L2=1 is |sin(theta2)|
    import math

    test_detj = [abs(math.sin(x[1])) for x in test_inputs]
    edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]
    res = run_zeroproof_baseline(
        (train_inputs, train_targets),
        (test_inputs, test_targets),
        enable_enhancements=False,
        output_dir=str(tmp_path),
        test_detJ=test_detj,
        bucket_edges=edges,
        epochs=1,
    )

    assert "near_pole_bucket_mse" in res
    nb = res["near_pole_bucket_mse"]
    assert set(nb.keys()) == {"edges", "bucket_mse", "bucket_counts"}
    assert len(nb["edges"]) == len(edges)
    # Buckets is edges-1 entries
    assert len(nb["bucket_mse"]) == (len(edges) - 1)
    assert len(nb["bucket_counts"]) == (len(edges) - 1)


def test_pole_metrics_2d():
    # Simple inputs around theta2=0 and pi
    import math

    inputs = [[0.0, -0.05, 0.0, 0.0], [0.0, 0.05, 0.0, 0.0], [0.0, math.pi - 0.05, 0.0, 0.0]]
    preds = [[0.0, 0.1], [0.0, -0.1], [0.0, 0.05]]  # arbitrary small deltas
    metrics = compute_pole_metrics_2d(inputs, preds)
    for k in ("ple", "sign_consistency", "slope_error", "residual_consistency"):
        assert k in metrics
        assert isinstance(metrics[k], float)
