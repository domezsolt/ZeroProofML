"""
RRR-arm inverse kinematics dataset generator (planar 3R).

Generates datasets for differential IK near rank-deficient Jacobians using a
planar 3-link RRR robot. We bucket by manipulability σ1·σ2 = sqrt(det(J J^T))
to mirror the 2R |det J| protocol.
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from zeroproof.utils.config import DEFAULT_BUCKET_EDGES
from zeroproof.utils.env import collect_env_info
from zeroproof.utils.seeding import set_global_seed
from zeroproof.utils.serialization import to_builtin


@dataclass
class Robot3RConfig:
    """Configuration for planar 3R robot."""

    L1: float = 1.0
    L2: float = 1.0
    L3: float = 1.0
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    )  # (θ1, θ2, θ3)


@dataclass
class IK3RSample:
    """Single IK sample with kinematics data for 3R."""

    # Input: desired end-effector displacement
    dx: float
    dy: float

    # Current joint configuration
    theta1: float
    theta2: float
    theta3: float

    # Target joint displacement (DLS)
    dtheta1: float
    dtheta2: float
    dtheta3: float

    # Jacobian/manipulability proxies
    det_J: float  # For 3R we store σ1·σ2 for compatibility
    cond_J: float
    is_singular: bool

    # End-effector position
    x_ee: float
    y_ee: float

    # Additional metadata
    manipulability: float
    distance_to_singularity: float


class RRRKinematics:
    """Planar 3R kinematics and Jacobian utilities."""

    def __init__(self, config: Robot3RConfig):
        self.config = config
        self.L1, self.L2, self.L3 = config.L1, config.L2, config.L3

    def forward_kinematics(
        self, theta1: float, theta2: float, theta3: float
    ) -> Tuple[float, float]:
        t12 = theta1 + theta2
        t123 = t12 + theta3
        x = self.L1 * math.cos(theta1) + self.L2 * math.cos(t12) + self.L3 * math.cos(t123)
        y = self.L1 * math.sin(theta1) + self.L2 * math.sin(t12) + self.L3 * math.sin(t123)
        return x, y

    def jacobian(self, theta1: float, theta2: float, theta3: float) -> np.ndarray:
        t1 = theta1
        t12 = theta1 + theta2
        t123 = t12 + theta3
        s1, c1 = math.sin(t1), math.cos(t1)
        s12, c12 = math.sin(t12), math.cos(t12)
        s123, c123 = math.sin(t123), math.cos(t123)
        # 2x3 Jacobian
        J = np.array(
            [
                [
                    -self.L1 * s1 - self.L2 * s12 - self.L3 * s123,
                    -self.L2 * s12 - self.L3 * s123,
                    -self.L3 * s123,
                ],
                [
                    self.L1 * c1 + self.L2 * c12 + self.L3 * c123,
                    self.L2 * c12 + self.L3 * c123,
                    self.L3 * c123,
                ],
            ],
            dtype=float,
        )
        return J

    def manipulability_index(self, theta1: float, theta2: float, theta3: float) -> float:
        J = self.jacobian(theta1, theta2, theta3)
        JJt = J @ J.T
        try:
            val = float(np.linalg.det(JJt))
        except Exception:
            return 0.0
        if val < 0 and abs(val) < 1e-12:
            val = 0.0
        return float(math.sqrt(val))

    def jacobian_condition_number(self, theta1: float, theta2: float, theta3: float) -> float:
        J = self.jacobian(theta1, theta2, theta3)
        try:
            return float(np.linalg.cond(J))
        except Exception:
            return float("inf")

    def distance_to_singularity(self, theta2: float, theta3: float) -> float:
        # Distance to nearest of theta2∈{0,π} or theta3∈{0,π}
        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        d2 = min(abs(wrap_pi(theta2)), abs(wrap_pi(theta2 - math.pi)))
        d3 = min(abs(wrap_pi(theta3)), abs(wrap_pi(theta3 - math.pi)))
        return float(min(d2, d3))

    def is_singular(
        self, theta1: float, theta2: float, theta3: float, threshold: float = 1e-3
    ) -> bool:
        return bool(self.manipulability_index(theta1, theta2, theta3) < threshold)

    def dls_ik(
        self,
        theta1: float,
        theta2: float,
        theta3: float,
        dx: float,
        dy: float,
        damping: float = 0.01,
    ) -> Tuple[float, float, float]:
        """3R damped least squares IK: Δθ = J^T (JJ^T + λ²I)^(-1) Δx."""
        J = self.jacobian(theta1, theta2, theta3)
        e = np.array([dx, dy], dtype=float)
        JJt = J @ J.T
        lam2I = (damping**2) * np.eye(2)
        try:
            inv_term = np.linalg.inv(JJt + lam2I)
            dtheta = J.T @ inv_term @ e
            return float(dtheta[0]), float(dtheta[1]), float(dtheta[2])
        except Exception:
            return 0.0, 0.0, 0.0


class RRRDatasetGenerator:
    """Generate IK datasets for the planar 3R robot near singularities."""

    def __init__(self, config: Robot3RConfig):
        self.config = config
        self.robot = RRRKinematics(config)
        self.samples: List[IK3RSample] = []

    def sample_configurations(
        self,
        n_samples: int,
        singular_ratio: float = 0.3,
        singularity_threshold: float = 1e-3,
        force_exact_singularities: bool = False,
        min_manip_regular: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """Sample joint configurations; near singularities and regular."""
        cfgs: List[Tuple[float, float, float]] = []
        n_sing = int(n_samples * singular_ratio)
        n_reg = n_samples - n_sing

        # Exact singulars: theta2 ∈ {0,π}, theta3 ∈ {0,π}
        exact_inserted = 0
        if force_exact_singularities and n_sing >= 4:
            for t2, t3 in [(0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)]:
                t1 = float(np.random.uniform(*self.config.joint_limits[0]))
                cfgs.append((t1, t2, t3))
                exact_inserted += 1

        # Near singulars (Gaussian around lines for both θ2 and θ3)
        for _ in range(max(0, n_sing - exact_inserted)):
            t1 = float(np.random.uniform(*self.config.joint_limits[0]))
            # choose which lines to sample near
            t2_base = 0.0 if np.random.random() < 0.5 else math.pi
            t3_base = 0.0 if np.random.random() < 0.5 else math.pi
            t2 = float(np.random.normal(t2_base, singularity_threshold))
            t3 = float(np.random.normal(t3_base, singularity_threshold))
            # clamp to limits
            t2 = float(np.clip(t2, *self.config.joint_limits[1]))
            t3 = float(np.clip(t3, *self.config.joint_limits[2]))
            cfgs.append((t1, t2, t3))

        # Regular configurations: away from singulars/manip threshold
        for _ in range(n_reg):
            attempts = 0
            while True:
                attempts += 1
                t1 = float(np.random.uniform(*self.config.joint_limits[0]))
                t2 = float(np.random.uniform(*self.config.joint_limits[1]))
                t3 = float(np.random.uniform(*self.config.joint_limits[2]))
                if self.robot.is_singular(t1, t2, t3, singularity_threshold * 2):
                    if attempts < 200:
                        continue
                if min_manip_regular is not None:
                    if self.robot.manipulability_index(t1, t2, t3) < float(min_manip_regular):
                        if attempts < 200:
                            continue
                break
            cfgs.append((t1, t2, t3))

        return cfgs

    def generate_ik_samples(
        self,
        configurations: List[Tuple[float, float, float]],
        displacement_scale: float = 0.1,
        damping_factor: float = 0.01,
    ) -> List[IK3RSample]:
        """Generate IK samples using DLS at given configurations."""
        L1, L2, L3 = self.config.L1, self.config.L2, self.config.L3
        samples: List[IK3RSample] = []
        for t1, t2, t3 in configurations:
            x_ee, y_ee = self.robot.forward_kinematics(t1, t2, t3)
            dx = float(np.random.normal(0.0, displacement_scale))
            dy = float(np.random.normal(0.0, displacement_scale))
            d1, d2, d3 = self.robot.dls_ik(t1, t2, t3, dx, dy, damping=damping_factor)
            # Manipulability proxy and condition number
            manip = self.robot.manipulability_index(t1, t2, t3)
            cond = self.robot.jacobian_condition_number(t1, t2, t3)
            dist_sing = self.robot.distance_to_singularity(t2, t3)
            is_sing = bool(manip < 1e-3)
            samples.append(
                IK3RSample(
                    dx=dx,
                    dy=dy,
                    theta1=float(t1),
                    theta2=float(t2),
                    theta3=float(t3),
                    dtheta1=float(d1),
                    dtheta2=float(d2),
                    dtheta3=float(d3),
                    det_J=float(manip),  # compatibility: use manip as det_J proxy
                    cond_J=float(cond),
                    is_singular=bool(is_sing),
                    x_ee=float(x_ee),
                    y_ee=float(y_ee),
                    manipulability=float(manip),
                    distance_to_singularity=float(dist_sing),
                )
            )
        return samples

    def generate_dataset(
        self,
        n_samples: int = 2000,
        singular_ratio: float = 0.3,
        displacement_scale: float = 0.1,
        singularity_threshold: float = 1e-3,
        damping_factor: float = 0.01,
        force_exact_singularities: bool = True,
        min_manip_regular: Optional[float] = None,
    ) -> List[IK3RSample]:
        print(f"Generating {n_samples} IK samples (3R)...")
        cfgs = self.sample_configurations(
            n_samples,
            singular_ratio=singular_ratio,
            singularity_threshold=singularity_threshold,
            force_exact_singularities=force_exact_singularities,
            min_manip_regular=min_manip_regular,
        )
        self.samples = self.generate_ik_samples(cfgs, displacement_scale, damping_factor)
        n_sing = sum(1 for s in self.samples if bool(s.is_singular))
        print(
            f"Generated samples: {len(self.samples)}  singular: {n_sing} ({n_sing/len(self.samples):.1%})"
        )
        return self.samples

    @staticmethod
    def stratify_split(
        samples: List[IK3RSample], train_ratio: float = 0.8, edges: List[float] = None
    ) -> Dict[str, List[IK3RSample]]:
        """Stratify train/test by manipulability (det_J proxy)."""
        edges = edges or DEFAULT_BUCKET_EDGES
        buckets: List[List[IK3RSample]] = [[] for _ in range(len(edges) - 1)]
        for s in samples:
            dj = abs(float(s.det_J))
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if (dj >= lo if i == 0 else dj > lo) and dj <= hi:
                    buckets[i].append(s)
                    break
        train: List[IK3RSample] = []
        test: List[IK3RSample] = []
        for b in buckets:
            n = len(b)
            k = int(round(train_ratio * n))
            train.extend(b[:k])
            test.extend(b[k:])
        return {"train": train, "test": test}

    def save_dataset(self, filename: str, format: str = "json") -> None:
        if not self.samples:
            raise ValueError("No samples to save. Generate dataset first.")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if format == "json":
            data = {
                "config": to_builtin(asdict(self.config)),
                "samples": [to_builtin(asdict(s)) for s in self.samples],
                "metadata": to_builtin(
                    {
                        "n_samples": len(self.samples),
                        "n_singular": sum(1 for s in self.samples if bool(s.is_singular)),
                        "generator": "RRRDatasetGenerator",
                        "schema_version": "1.0",
                        "env": collect_env_info(),
                    }
                ),
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "npz":
            arrays = {}
            keys = [
                "dx",
                "dy",
                "theta1",
                "theta2",
                "theta3",
                "dtheta1",
                "dtheta2",
                "dtheta3",
                "det_J",
                "cond_J",
                "x_ee",
                "y_ee",
                "manipulability",
                "distance_to_singularity",
            ]
            for key in keys:
                arrays[key] = np.array([getattr(s, key) for s in self.samples])
            arrays["is_singular"] = np.array([bool(s.is_singular) for s in self.samples])
            np.savez(filename, **arrays)
        print(f"Dataset saved to {filename}")


def _parse_edges(s: Optional[str]) -> List[float]:
    if not s:
        return DEFAULT_BUCKET_EDGES
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = []
    for p in parts:
        if p.lower() == "inf":
            vals.append(float("inf"))
        else:
            vals.append(float(p))
    return vals


def main():
    ap = argparse.ArgumentParser(description="Generate 3R IK dataset with manipulability bucketing")
    ap.add_argument("--output", type=str, default="data/ik3r_dataset.json")
    ap.add_argument("--n_samples", type=int, default=16000)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--singular_ratio", type=float, default=0.3)
    ap.add_argument("--displacement_scale", type=float, default=0.1)
    ap.add_argument("--singularity_threshold", type=float, default=1e-3)
    ap.add_argument("--damping_factor", type=float, default=0.01)
    ap.add_argument("--force_exact_singularities", action="store_true")
    ap.add_argument("--min_manip_regular", type=float, default=None)
    ap.add_argument("--stratify_by_manip", action="store_true")
    ap.add_argument(
        "--bucket_edges", type=str, default=None, help="Comma-separated, last can be inf"
    )
    ap.add_argument("--ensure_buckets_nonzero", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--format", type=str, choices=["json", "npz"], default="json")
    args = ap.parse_args()

    set_global_seed(args.seed)
    generator = RRRDatasetGenerator(Robot3RConfig())
    samples = generator.generate_dataset(
        n_samples=args.n_samples,
        singular_ratio=args.singular_ratio,
        displacement_scale=args.displacement_scale,
        singularity_threshold=args.singularity_threshold,
        damping_factor=args.damping_factor,
        force_exact_singularities=args.force_exact_singularities,
        min_manip_regular=args.min_manip_regular,
    )

    bucket_edges = _parse_edges(args.bucket_edges)
    split_info = None
    if args.stratify_by_manip:
        split = RRRDatasetGenerator.stratify_split(
            samples, train_ratio=args.train_ratio, edges=bucket_edges
        )
        split_info = {
            "train_bucket_counts": {},
            "test_bucket_counts": {},
        }

        def _bucket_counts(subset: List[IK3RSample]):
            counts = [0] * (len(bucket_edges) - 1)
            for s in subset:
                dj = abs(float(s.det_J))
                for i in range(len(bucket_edges) - 1):
                    lo, hi = bucket_edges[i], bucket_edges[i + 1]
                    if (dj >= lo if i == 0 else dj > lo) and dj <= hi:
                        counts[i] += 1
                        break
            return counts

        split_info["train_bucket_counts"] = _bucket_counts(split["train"])
        split_info["test_bucket_counts"] = _bucket_counts(split["test"])

        # Optional ensure non-zero for first 4 buckets
        if args.ensure_buckets_nonzero:

            def augment(subset: List[IK3RSample], counts: List[int]) -> List[IK3RSample]:
                missing = [i for i, c in enumerate(counts[:4]) if c == 0]
                if not missing:
                    return subset
                augmented = subset[:]
                rng = np.random.default_rng()
                attempts = 0
                while missing and attempts < 400:
                    attempts += 1
                    b = missing[0]
                    lo, hi = bucket_edges[b], bucket_edges[b + 1]
                    # choose theta2/theta3 around corresponding implied magnitude by inverse of sin small-angle ~ value
                    t1 = float(rng.uniform(*generator.config.joint_limits[0]))
                    # sample around exact lines for low buckets
                    t2 = 0.0 if rng.random() < 0.5 else float(math.pi)
                    t3 = 0.0 if rng.random() < 0.5 else float(math.pi)
                    new_s = generator.generate_ik_samples(
                        [(t1, t2, t3)], args.displacement_scale, args.damping_factor
                    )[0]
                    augmented.append(new_s)
                    dj = abs(float(new_s.det_J))
                    for i in range(len(bucket_edges) - 1):
                        lo2, hi2 = bucket_edges[i], bucket_edges[i + 1]
                        if (dj >= lo2 if i == 0 else dj > lo2) and dj <= hi2:
                            counts[i] += 1
                            break
                    missing = [i for i, c in enumerate(counts[:4]) if c == 0]
                return augmented

            split["train"] = augment(split["train"], split_info["train_bucket_counts"])
            split["test"] = augment(split["test"], split_info["test_bucket_counts"])
            split_info["train_bucket_counts"] = [int(c) for c in split_info["train_bucket_counts"]]
            split_info["test_bucket_counts"] = [int(c) for c in split_info["test_bucket_counts"]]
            generator.samples = split["train"] + split["test"]

        # Overwrite samples order (train then test)
        generator.samples = split["train"] + split["test"]

    # Save dataset
    generator.save_dataset(args.output, args.format)
    if args.output.endswith(".json") and split_info is not None:
        try:
            with open(args.output, "r") as f:
                data = json.load(f)
            data["metadata"]["bucket_edges"] = bucket_edges
            data["metadata"]["stratified_by_detj"] = True  # compatibility key
            data["metadata"]["train_ratio"] = args.train_ratio
            data["metadata"]["train_bucket_counts"] = split_info["train_bucket_counts"]
            data["metadata"]["test_bucket_counts"] = split_info["test_bucket_counts"]
            data["metadata"]["seed"] = args.seed
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # Print summary
    print("\nDataset Summary (3R):")
    print(f"Total samples: {len(generator.samples)}")
    print(f"Singular samples: {sum(1 for s in generator.samples if s.is_singular)}")
    print(f"Average manip (σ1·σ2): {np.mean([abs(s.det_J) for s in generator.samples]):.6f}")
    print(f"Min manip: {np.min([abs(s.det_J) for s in generator.samples]):.6f}")
    finite_conds = [s.cond_J for s in generator.samples if np.isfinite(s.cond_J)]
    if finite_conds:
        print(f"Max condition number: {np.max(finite_conds):.2f}")
    if split_info is not None:
        print("Bucket edges:", bucket_edges)
        print("Train bucket counts:", split_info["train_bucket_counts"])
        print("Test bucket counts:", split_info["test_bucket_counts"])


if __name__ == "__main__":
    main()
