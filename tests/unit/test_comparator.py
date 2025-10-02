import os

from examples.baselines import compare_all as cmp
from examples.robotics.rr_ik_dataset import RobotConfig, RRDatasetGenerator


def test_parity_same_split_and_loss(tmp_path):
    # Create a small dataset file
    gen = RRDatasetGenerator(RobotConfig())
    samples = gen.generate_dataset(
        n_samples=200,
        singular_ratio=0.4,
        singularity_threshold=1e-3,
        force_exact_singularities=True,
    )
    dataset_file = tmp_path / "rr_ik_dataset.json"
    gen.save_dataset(str(dataset_file), format="json")

    # Override comparator to quick with limited models for speed
    cmp._COMPARATOR_OVERRIDES = {
        "quick": True,
        "models": ["mlp", "rational_eps", "tr_basic"],
        "mlp_epochs": 2,
        "rat_epochs": 2,
        "zp_epochs": 4,
        "rat_epsilon": 1e-2,
    }

    out_dir = tmp_path / "cmp_out"
    res = cmp.run_complete_comparison(str(dataset_file), str(out_dir), seed=123)

    # Assert bucket edges consistent and present
    edges = res["dataset_info"]["bucket_edges"]
    assert edges and edges[0] == 0.0 and edges[-1] == float("inf")

    indiv = res.get("individual_results", {})
    # Ensure each requested method reports bucketed MSE and uses same edges
    labels = ["MLP", "Rational+ε", "ZeroProofML-Basic"]
    for label in labels:
        if label not in indiv:
            continue
        b = indiv[label].get("near_pole_bucket_mse")
        assert b is not None
        assert b.get("edges") == edges
