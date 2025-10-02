import json
import os

import numpy as np

from examples.robotics.rr_ik_dataset import RobotConfig, RRDatasetGenerator


def test_json_npz_roundtrip(tmp_path):
    # Generate a small dataset
    gen = RRDatasetGenerator(RobotConfig())
    samples = gen.generate_dataset(n_samples=50, singular_ratio=0.3)
    assert len(samples) == 50

    # Save JSON
    json_file = tmp_path / "rr_ik_dataset.json"
    gen.save_dataset(str(json_file), format="json")

    # Load JSON
    gen2 = RRDatasetGenerator.load_dataset(str(json_file))
    assert hasattr(gen2, "samples") and len(gen2.samples) == 50
    # Metadata present
    md = getattr(gen2, "metadata", {})
    assert isinstance(md.get("n_samples", 50), (int, float))

    # Save NPZ
    npz_file = tmp_path / "rr_ik_dataset.npz"
    gen.save_dataset(str(npz_file), format="npz")

    # Load NPZ and verify arrays
    data = np.load(str(npz_file))
    for key in [
        "dx",
        "dy",
        "theta1",
        "theta2",
        "dtheta1",
        "dtheta2",
        "det_J",
        "cond_J",
        "manipulability",
        "distance_to_singularity",
        "is_singular",
    ]:
        assert key in data
        assert len(data[key]) == 50
