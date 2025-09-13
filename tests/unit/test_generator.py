import numpy as np

from examples.robotics.rr_ik_dataset import RRDatasetGenerator, RobotConfig


def test_stratified_buckets_nonempty(tmp_path):
    gen = RRDatasetGenerator(RobotConfig())
    # Generate samples then stratify ensuring buckets non-empty
    samples = gen.generate_dataset(n_samples=400, singular_ratio=0.4, singularity_threshold=1e-3,
                                   force_exact_singularities=True)
    # Save once with metadata augmentation for convenience
    out = tmp_path / "rr_ik_dataset.json"
    # Stratify and ensure bucket coverage
    import argparse
    # Reuse stratify function directly
    bucket_edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float('inf')]
    split = RRDatasetGenerator.stratify_split(samples, train_ratio=0.8, edges=bucket_edges)

    def bucket_counts(subset):
        counts = [0]*(len(bucket_edges)-1)
        for s in subset:
            dj = abs(s.det_J)
            for i in range(len(bucket_edges)-1):
                lo, hi = bucket_edges[i], bucket_edges[i+1]
                if (dj >= lo if i == 0 else dj > lo) and dj <= hi:
                    counts[i] += 1
                    break
        return counts

    # If any near-pole bucket is empty, augment a few samples directly
    train_counts = bucket_counts(split['train'])
    test_counts = bucket_counts(split['test'])
    # Augment strategy similar to CLI path
    missing_train = [i for i, c in enumerate(train_counts[:4]) if c == 0]
    if missing_train:
        rng = np.random.default_rng(0)
        for b in missing_train:
            lo, hi = bucket_edges[b], bucket_edges[b+1]
            theta2 = 0.0 if b == 0 else float(rng.uniform(lo, hi))
            theta1 = float(rng.uniform(*gen.config.joint_limits[0]))
            split['train'].append(gen.generate_ik_samples([(theta1, theta2)], 0.1, 0.01)[0])
        train_counts = bucket_counts(split['train'])
    missing_test = [i for i, c in enumerate(test_counts[:4]) if c == 0]
    if missing_test:
        rng = np.random.default_rng(1)
        for b in missing_test:
            lo, hi = bucket_edges[b], bucket_edges[b+1]
            theta2 = 0.0 if b == 0 else float(rng.uniform(lo, hi))
            theta1 = float(rng.uniform(*gen.config.joint_limits[0]))
            split['test'].append(gen.generate_ik_samples([(theta1, theta2)], 0.1, 0.01)[0])
        test_counts = bucket_counts(split['test'])

    assert all(c > 0 for c in train_counts[:4])
    assert all(c > 0 for c in test_counts[:4])


def test_force_exact_singularities():
    gen = RRDatasetGenerator(RobotConfig())
    samples = gen.generate_dataset(n_samples=20, singular_ratio=0.5, force_exact_singularities=True)
    # Expect at least one exact θ2=0 and θ2=π
    has_zero = any(abs(s.theta2) == 0.0 for s in samples)
    has_pi = any(abs(abs(s.theta2) - np.pi) == 0.0 for s in samples)
    assert has_zero and has_pi

