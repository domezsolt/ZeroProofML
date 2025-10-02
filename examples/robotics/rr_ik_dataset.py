"""
RR-arm inverse kinematics dataset generator.

This module generates datasets for training inverse kinematics near
singular Jacobians using a planar 2-link RR (Revolute-Revolute) robot.

The robot has two revolute joints with link lengths L1 and L2.
Singularities occur when the links are fully extended or retracted.
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.utils.config import DEFAULT_BUCKET_EDGES
from zeroproof.utils.env import collect_env_info
from zeroproof.utils.metrics import PoleLocation
from zeroproof.utils.seeding import set_global_seed
from zeroproof.utils.serialization import to_builtin


@dataclass
class RobotConfig:
    """Configuration for RR robot."""

    L1: float = 1.0  # Length of first link
    L2: float = 1.0  # Length of second link
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    )  # Joint angle limits for (θ1, θ2)


@dataclass
class IKSample:
    """Single IK sample with kinematics data."""

    # Input: desired end-effector displacement
    dx: float
    dy: float

    # Current joint configuration
    theta1: float
    theta2: float

    # Target joint displacement (from DLS or analytical solution)
    dtheta1: float
    dtheta2: float

    # Jacobian properties
    det_J: float
    cond_J: float
    is_singular: bool

    # End-effector position
    x_ee: float
    y_ee: float

    # Additional metadata
    manipulability: float
    distance_to_singularity: float


class RRKinematics:
    """
    Planar 2-link RR robot kinematics.

    Forward kinematics: (θ1, θ2) → (x, y)
    Jacobian: J = ∂(x,y)/∂(θ1,θ2)
    Inverse kinematics: Δ(x,y) → Δ(θ1,θ2)
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.L1 = config.L1
        self.L2 = config.L2

    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """
        Compute end-effector position from joint angles.

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            (x, y) end-effector position
        """
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        return x, y

    def jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """
        Compute Jacobian matrix J = ∂(x,y)/∂(θ1,θ2).

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            2x2 Jacobian matrix
        """
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c12, s12 = np.cos(theta1 + theta2), np.sin(theta1 + theta2)

        J = np.array(
            [
                [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
                [self.L1 * c1 + self.L2 * c12, self.L2 * c12],
            ]
        )

        return J

    def jacobian_determinant(self, theta1: float, theta2: float) -> float:
        """
        Compute Jacobian determinant (manipulability measure).

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            det(J) - zero indicates singularity
        """
        # For RR robot: det(J) = L1 * L2 * sin(θ2)
        return self.L1 * self.L2 * np.sin(theta2)

    def jacobian_condition_number(self, theta1: float, theta2: float) -> float:
        """
        Compute Jacobian condition number.

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            Condition number of Jacobian
        """
        J = self.jacobian(theta1, theta2)
        try:
            return np.linalg.cond(J)
        except:
            return float("inf")

    def manipulability_index(self, theta1: float, theta2: float) -> float:
        """
        Compute manipulability index (Yoshikawa).

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            Manipulability index
        """
        J = self.jacobian(theta1, theta2)
        val = np.linalg.det(J @ J.T)
        # Numerical guard: due to round-off, val can be tiny negative; clamp to 0
        if val < 0 and abs(val) < 1e-12:
            val = 0.0
        return float(np.sqrt(val))

    def distance_to_singularity(self, theta1: float, theta2: float) -> float:
        """
        Compute distance to nearest singularity.

        For RR robot, singularities occur when θ2 = 0 or θ2 = π.

        Args:
            theta1: First joint angle
            theta2: Second joint angle

        Returns:
            Distance to nearest singular configuration
        """
        # Distance to θ2 = 0
        dist_to_zero = abs(theta2)

        # Distance to θ2 = π
        dist_to_pi = min(abs(theta2 - np.pi), abs(theta2 + np.pi))

        return min(dist_to_zero, dist_to_pi)

    def is_singular(self, theta1: float, theta2: float, threshold: float = 1e-3) -> bool:
        """
        Check if configuration is singular.

        Args:
            theta1: First joint angle
            theta2: Second joint angle
            threshold: Singularity threshold for |det(J)|

        Returns:
            True if configuration is singular
        """
        det_J = abs(self.jacobian_determinant(theta1, theta2))
        # Ensure native Python bool (avoid numpy.bool_)
        return bool(det_J < threshold)

    def damped_least_squares_ik(
        self, theta1: float, theta2: float, dx: float, dy: float, damping: float = 0.01
    ) -> Tuple[float, float]:
        """
        Compute inverse kinematics using Damped Least Squares (DLS).

        Δθ = J^T (JJ^T + λ²I)^(-1) Δx

        Args:
            theta1: Current first joint angle
            theta2: Current second joint angle
            dx: Desired x displacement
            dy: Desired y displacement
            damping: Damping factor λ

        Returns:
            (dtheta1, dtheta2) joint displacements
        """
        J = self.jacobian(theta1, theta2)
        dx_vec = np.array([dx, dy])

        # DLS formula
        JJT = J @ J.T
        damping_matrix = damping**2 * np.eye(2)

        try:
            inv_term = np.linalg.inv(JJT + damping_matrix)
            dtheta = J.T @ inv_term @ dx_vec
            return float(dtheta[0]), float(dtheta[1])
        except:
            return 0.0, 0.0

    def analytical_ik(self, x: float, y: float) -> List[Tuple[float, float]]:
        """
        Analytical inverse kinematics for RR robot.

        Args:
            x: Target x position
            y: Target y position

        Returns:
            List of (theta1, theta2) solutions
        """
        solutions = []

        # Distance from origin to target
        r = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > self.L1 + self.L2 or r < abs(self.L1 - self.L2):
            return solutions

        # Law of cosines for theta2
        cos_theta2 = (r**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)

        if abs(cos_theta2) <= 1:
            # Two solutions for theta2
            theta2_1 = np.arccos(cos_theta2)
            theta2_2 = -theta2_1

            for theta2 in [theta2_1, theta2_2]:
                # Corresponding theta1
                k1 = self.L1 + self.L2 * np.cos(theta2)
                k2 = self.L2 * np.sin(theta2)

                theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

                solutions.append((theta1, theta2))

        return solutions


class RRDatasetGenerator:
    """
    Generate IK datasets for RR robot near singularities.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.robot = RRKinematics(config)
        self.samples = []

    def sample_configurations(
        self,
        n_samples: int,
        singular_ratio: float = 0.3,
        singularity_threshold: float = 1e-3,
        force_exact_singularities: bool = False,
        min_detj_regular: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Sample joint configurations with controlled singularity ratio.

        Args:
            n_samples: Total number of samples
            singular_ratio: Fraction of samples near singularities
            singularity_threshold: Threshold for singularity detection

        Returns:
            List of (theta1, theta2) configurations
        """
        configurations = []

        # Number of singular and regular samples
        n_singular = int(n_samples * singular_ratio)
        n_regular = n_samples - n_singular

        # Sample near singularities (θ2 ≈ 0 or θ2 ≈ π)
        exact_inserted = 0
        if force_exact_singularities and n_singular >= 2:
            # Ensure at least one exact config on each singular line
            theta1_a = np.random.uniform(*self.config.joint_limits[0])
            theta1_b = np.random.uniform(*self.config.joint_limits[0])
            configurations.append((theta1_a, 0.0))
            configurations.append((theta1_b, np.pi))
            exact_inserted = 2
        for _ in range(max(0, n_singular - exact_inserted)):
            theta1 = np.random.uniform(*self.config.joint_limits[0])
            # Choose between θ2 ≈ 0 or θ2 ≈ π
            if np.random.random() < 0.5:
                # Near θ2 = 0
                theta2 = np.random.normal(0, singularity_threshold)
            else:
                # Near θ2 = π
                theta2 = np.random.normal(np.pi, singularity_threshold)
            # Clamp to joint limits
            theta2 = np.clip(theta2, *self.config.joint_limits[1])
            configurations.append((theta1, theta2))

        # Sample regular configurations
        for _ in range(n_regular):
            attempts = 0
            while True:
                attempts += 1
                theta1 = np.random.uniform(*self.config.joint_limits[0])
                theta2 = np.random.uniform(*self.config.joint_limits[1])
                # Reject if too close to singularity
                if self.robot.is_singular(theta1, theta2, singularity_threshold * 2):
                    if attempts < 100:
                        continue
                # Enforce minimal |detJ| for regular samples if requested
                if min_detj_regular is not None:
                    if abs(self.robot.jacobian_determinant(theta1, theta2)) < float(
                        min_detj_regular
                    ):
                        if attempts < 100:
                            continue
                break
            configurations.append((theta1, theta2))

        return configurations

    def generate_ik_samples(
        self,
        configurations: List[Tuple[float, float]],
        displacement_scale: float = 0.1,
        damping_factor: float = 0.01,
        vectorized: bool = True,
    ) -> List[IKSample]:
        """
        Generate IK samples from joint configurations.

        Args:
            configurations: List of (theta1, theta2) configurations
            displacement_scale: Scale for random end-effector displacements
            damping_factor: DLS damping factor

        Returns:
            List of IK samples
        """
        # Vectorized fast path
        if vectorized and configurations:
            L1, L2 = self.config.L1, self.config.L2
            thetas = np.array(configurations, dtype=float)
            theta1 = thetas[:, 0]
            theta2 = thetas[:, 1]

            # Forward kinematics (current pose)
            c1, s1 = np.cos(theta1), np.sin(theta1)
            c12, s12 = np.cos(theta1 + theta2), np.sin(theta1 + theta2)
            x_ee = L1 * c1 + L2 * c12
            y_ee = L1 * s1 + L2 * s12

            # Random target displacements
            rng = np.random.default_rng()
            dx = rng.normal(0.0, displacement_scale, size=len(theta1))
            dy = rng.normal(0.0, displacement_scale, size=len(theta2))

            # Jacobian components
            j11 = -L1 * s1 - L2 * s12
            j12 = -L2 * s12
            j21 = L1 * c1 + L2 * c12
            j22 = L2 * c12

            # JJT components and DLS inverse blocks
            a = j11 * j11 + j12 * j12
            b = j11 * j21 + j12 * j22
            d = j21 * j21 + j22 * j22
            lam2 = damping_factor**2
            a_l = a + lam2
            d_l = d + lam2
            detA = a_l * d_l - b * b
            eps = 1e-12
            detA_safe = np.where(np.abs(detA) < eps, np.sign(detA) * eps + (detA == 0) * eps, detA)
            inv00 = d_l / detA_safe
            inv01 = -b / detA_safe
            inv10 = -b / detA_safe
            inv11 = a_l / detA_safe

            # u = inv * e, e = [dx, dy]
            ux = inv00 * dx + inv01 * dy
            uy = inv10 * dx + inv11 * dy

            # dtheta = J^T * u
            dtheta1 = j11 * ux + j21 * uy
            dtheta2 = j12 * ux + j22 * uy

            # Jacobian-derived properties
            det_J = L1 * L2 * np.sin(theta2)
            # Manipulability sqrt(det(J J^T)) = sqrt(a*d - b^2)
            man_sq = a * d - b * b
            man_sq = np.where(man_sq < 0.0, np.where(np.abs(man_sq) < 1e-12, 0.0, man_sq), man_sq)
            manipulability = np.sqrt(man_sq)
            # Distance to pole lines θ2=0 or π
            dist_to_zero = np.abs(theta2)
            dist_to_pi = np.minimum(np.abs(theta2 - np.pi), np.abs(theta2 + np.pi))
            dist_to_sing = np.minimum(dist_to_zero, dist_to_pi)
            # is_singular heuristic (use default threshold)
            is_singular = np.abs(det_J) < 1e-3

            # Condition number per-sample (fallback to loop)
            cond_list = []
            for j11i, j12i, j21i, j22i in zip(j11, j12, j21, j22):
                Ji = np.array([[j11i, j12i], [j21i, j22i]], dtype=float)
                try:
                    cond_list.append(float(np.linalg.cond(Ji)))
                except Exception:
                    cond_list.append(float("inf"))

            samples: List[IKSample] = []
            for i in range(len(theta1)):
                samples.append(
                    IKSample(
                        dx=float(dx[i]),
                        dy=float(dy[i]),
                        theta1=float(theta1[i]),
                        theta2=float(theta2[i]),
                        dtheta1=float(dtheta1[i]),
                        dtheta2=float(dtheta2[i]),
                        det_J=float(det_J[i]),
                        cond_J=float(cond_list[i]),
                        is_singular=bool(is_singular[i]),
                        x_ee=float(x_ee[i]),
                        y_ee=float(y_ee[i]),
                        manipulability=float(manipulability[i]),
                        distance_to_singularity=float(dist_to_sing[i]),
                    )
                )
            return samples

        # Fallback scalar path
        samples: List[IKSample] = []
        for theta1, theta2 in configurations:
            x_ee, y_ee = self.robot.forward_kinematics(theta1, theta2)
            dx = np.random.normal(0, displacement_scale)
            dy = np.random.normal(0, displacement_scale)
            dtheta1, dtheta2 = self.robot.damped_least_squares_ik(
                theta1, theta2, dx, dy, damping_factor
            )
            det_J = self.robot.jacobian_determinant(theta1, theta2)
            cond_J = self.robot.jacobian_condition_number(theta1, theta2)
            manipulability = self.robot.manipulability_index(theta1, theta2)
            dist_to_sing = self.robot.distance_to_singularity(theta1, theta2)
            is_singular = self.robot.is_singular(theta1, theta2)
            samples.append(
                IKSample(
                    dx=dx,
                    dy=dy,
                    theta1=theta1,
                    theta2=theta2,
                    dtheta1=dtheta1,
                    dtheta2=dtheta2,
                    det_J=det_J,
                    cond_J=cond_J,
                    is_singular=is_singular,
                    x_ee=x_ee,
                    y_ee=y_ee,
                    manipulability=manipulability,
                    distance_to_singularity=dist_to_sing,
                )
            )
        return samples

    def generate_dataset(
        self,
        n_samples: int = 1000,
        singular_ratio: float = 0.3,
        displacement_scale: float = 0.1,
        singularity_threshold: float = 1e-3,
        damping_factor: float = 0.01,
        force_exact_singularities: bool = False,
        min_detj_regular: Optional[float] = None,
    ) -> List[IKSample]:
        """
        Generate complete IK dataset.

        Args:
            n_samples: Total number of samples
            singular_ratio: Fraction of samples near singularities
            displacement_scale: Scale for end-effector displacements
            singularity_threshold: Threshold for singularity detection
            damping_factor: DLS damping factor

        Returns:
            List of IK samples
        """
        print(f"Generating {n_samples} IK samples...")
        print(f"Singular ratio: {singular_ratio:.1%}")
        print(f"Singularity threshold: {singularity_threshold}")

        # Sample configurations
        configurations = self.sample_configurations(
            n_samples,
            singular_ratio,
            singularity_threshold,
            force_exact_singularities=force_exact_singularities,
            min_detj_regular=min_detj_regular,
        )

        # Generate IK samples
        samples = self.generate_ik_samples(configurations, displacement_scale, damping_factor)

        self.samples = samples

        # Print statistics
        n_singular = sum(1 for s in samples if s.is_singular)
        print(f"Generated samples: {len(samples)}")
        print(f"Singular samples: {n_singular} ({n_singular/len(samples):.1%})")

        return samples

    @staticmethod
    def stratify_split(
        samples: List[IKSample], train_ratio: float = 0.8, edges: List[float] = None
    ) -> Dict[str, List[IKSample]]:
        """Stratify train/test by |det(J)| buckets.

        Args:
            samples: Full sample list
            train_ratio: Fraction assigned to train within each bucket
            edges: Bucket edges [0, 1e-5, 1e-4, 1e-3, 1e-2, inf]
        Returns:
            Dict with 'train' and 'test' lists
        """
        edges = edges or DEFAULT_BUCKET_EDGES
        buckets: List[List[IKSample]] = [[] for _ in range(len(edges) - 1)]
        for s in samples:
            dj = abs(s.det_J)
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if (dj >= lo if i == 0 else dj > lo) and dj <= hi:
                    buckets[i].append(s)
                    break
        train: List[IKSample] = []
        test: List[IKSample] = []
        for b in buckets:
            n = len(b)
            k = int(round(train_ratio * n))
            train.extend(b[:k])
            test.extend(b[k:])
        return {"train": train, "test": test}

    def get_pole_locations(self) -> List[PoleLocation]:
        """
        Get theoretical pole locations for evaluation.

        For RR robot, poles occur at θ2 = 0 and θ2 = π.
        In the joint space, these are lines.

        Returns:
            List of pole locations (simplified to key points)
        """
        # Representative pole locations
        poles = []

        # Sample θ1 values for pole lines
        theta1_samples = np.linspace(*self.config.joint_limits[0], 5)

        for theta1 in theta1_samples:
            # Pole at θ2 = 0
            poles.append(PoleLocation(x=theta1, y=0.0, pole_type="line"))

            # Pole at θ2 = π
            poles.append(PoleLocation(x=theta1, y=np.pi, pole_type="line"))

        return poles

    def save_dataset(self, filename: str, format: str = "json") -> None:
        """
        Save dataset to file.

        Args:
            filename: Output filename
            format: File format ("json" or "npz")
        """
        if not self.samples:
            raise ValueError("No samples to save. Generate dataset first.")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if format == "json":
            # Convert to JSON-serializable format (centralized helper)
            data = {
                "config": to_builtin(asdict(self.config)),
                "samples": [to_builtin(asdict(sample)) for sample in self.samples],
                "metadata": to_builtin(
                    {
                        "n_samples": len(self.samples),
                        "n_singular": sum(1 for s in self.samples if bool(s.is_singular)),
                        "generator": "RRDatasetGenerator",
                        "schema_version": "1.0",
                        "env": collect_env_info(),
                    }
                ),
            }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "npz":
            # Convert to numpy arrays
            arrays = {}
            for key in [
                "dx",
                "dy",
                "theta1",
                "theta2",
                "dtheta1",
                "dtheta2",
                "det_J",
                "cond_J",
                "x_ee",
                "y_ee",
                "manipulability",
                "distance_to_singularity",
            ]:
                arrays[key] = np.array([getattr(s, key) for s in self.samples])

            arrays["is_singular"] = np.array([s.is_singular for s in self.samples])

            np.savez(filename, **arrays)

        print(f"Dataset saved to {filename}")

    @classmethod
    def load_dataset(cls, filename: str) -> "RRDatasetGenerator":
        """
        Load dataset from file.

        Args:
            filename: Input filename

        Returns:
            Loaded dataset generator
        """
        if filename.endswith(".json"):
            with open(filename, "r") as f:
                data = json.load(f)

            config = RobotConfig(**data["config"])
            generator = cls(config)

            # Reconstruct samples
            samples = []
            for sample_data in data["samples"]:
                sample = IKSample(**sample_data)
                samples.append(sample)

            generator.samples = samples
            # Attach metadata if present
            try:
                generator.metadata = data.get("metadata", {})
            except Exception:
                generator.metadata = {}
            return generator

        else:
            raise ValueError(f"Unsupported file format: {filename}")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate RR robot IK dataset")

    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument(
        "--singular_ratio", type=float, default=0.3, help="Fraction of samples near singularities"
    )
    parser.add_argument(
        "--displacement_scale", type=float, default=0.1, help="Scale for end-effector displacements"
    )
    parser.add_argument(
        "--singularity_threshold",
        type=float,
        default=1e-3,
        help="Threshold for singularity detection",
    )
    parser.add_argument("--damping_factor", type=float, default=0.01, help="DLS damping factor")
    parser.add_argument("--seed", type=int, default=None, help="Global seed for reproducibility")
    parser.add_argument(
        "--stratify_by_detj", action="store_true", help="Stratify train/test by |det(J)| buckets"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train split ratio within each bucket"
    )
    parser.add_argument(
        "--force_exact_singularities",
        action="store_true",
        help="Include exact singular configurations (θ2=0, π) among singular samples",
    )
    parser.add_argument(
        "--min_detj",
        type=float,
        default=None,
        help="Minimum |det(J)| for regular samples (reject until above)",
    )
    parser.add_argument(
        "--singular_ratio_split",
        type=str,
        default=None,
        help="Train:Test singular ratio when stratifying (e.g., 0.35:0.45)",
    )
    parser.add_argument(
        "--ensure_buckets_nonzero",
        action="store_true",
        help="Ensure B0–B3 buckets have non-zero counts in both splits by augmenting near-pole samples",
    )
    parser.add_argument(
        "--output", type=str, default="data/rr_ik_dataset.json", help="Output filename"
    )
    parser.add_argument(
        "--format", type=str, choices=["json", "npz"], default="json", help="Output format"
    )
    parser.add_argument("--L1", type=float, default=1.0, help="Length of first link")
    parser.add_argument("--L2", type=float, default=1.0, help="Length of second link")
    parser.add_argument(
        "--bucket-edges",
        nargs="+",
        default=None,
        help="Custom |det(J)| bucket edges, e.g., 0 1e-5 1e-4 1e-3 1e-2 inf",
    )

    args = parser.parse_args()

    # Seeding
    set_global_seed(args.seed)

    # Create robot configuration
    config = RobotConfig(L1=args.L1, L2=args.L2)

    # Generate dataset
    generator = RRDatasetGenerator(config)
    samples: List[IKSample]
    custom_split_used = False
    if args.stratify_by_detj and args.singular_ratio_split:
        # Parse split-specific singular ratios
        try:
            sr_train_str, sr_test_str = args.singular_ratio_split.split(":")
            sr_train = float(sr_train_str)
            sr_test = float(sr_test_str)
            sr_train = max(0.0, min(1.0, sr_train))
            sr_test = max(0.0, min(1.0, sr_test))
        except Exception:
            raise SystemExit("Invalid --singular_ratio_split format. Expected e.g. 0.35:0.45")

        # Compute split sizes
        n_train = int(round(args.train_ratio * args.n_samples))
        n_test = args.n_samples - n_train

        # Sample configs and generate per split
        train_cfg = generator.sample_configurations(
            n_train,
            singular_ratio=sr_train,
            singularity_threshold=args.singularity_threshold,
            force_exact_singularities=args.force_exact_singularities,
            min_detj_regular=args.min_detj,
        )
        test_cfg = generator.sample_configurations(
            n_test,
            singular_ratio=sr_test,
            singularity_threshold=args.singularity_threshold,
            force_exact_singularities=args.force_exact_singularities,
            min_detj_regular=args.min_detj,
        )
        train_samples = generator.generate_ik_samples(
            train_cfg, args.displacement_scale, args.damping_factor
        )
        test_samples = generator.generate_ik_samples(
            test_cfg, args.displacement_scale, args.damping_factor
        )
        samples = train_samples + test_samples
        generator.samples = samples
        custom_split_used = True
    else:
        samples = generator.generate_dataset(
            n_samples=args.n_samples,
            singular_ratio=args.singular_ratio,
            displacement_scale=args.displacement_scale,
            singularity_threshold=args.singularity_threshold,
            damping_factor=args.damping_factor,
            force_exact_singularities=args.force_exact_singularities,
            min_detj_regular=args.min_detj,
        )

    # If requested, stratify into train/test by |det(J)|
    # Bucket edges: override from CLI if provided
    def _parse_edges(vals):
        out = []
        for v in vals:
            s = str(v)
            if s.lower() in ("inf", "+inf"):
                out.append(float("inf"))
            else:
                out.append(float(s))
        return out

    bucket_edges = (
        DEFAULT_BUCKET_EDGES if not args.bucket_edges else _parse_edges(args.bucket_edges)
    )
    split_info = None
    if args.stratify_by_detj:
        if custom_split_used:
            n_train = int(round(args.train_ratio * args.n_samples))
            split = {"train": generator.samples[:n_train], "test": generator.samples[n_train:]}
        else:
            split = RRDatasetGenerator.stratify_split(
                samples, train_ratio=args.train_ratio, edges=bucket_edges
            )
        split_info = {
            "train_bucket_counts": {},
            "test_bucket_counts": {},
        }

        # Compute counts
        def _bucket_counts(subset: List[IKSample]):
            counts = [0] * (len(bucket_edges) - 1)
            for s in subset:
                dj = abs(s.det_J)
                for i in range(len(bucket_edges) - 1):
                    lo, hi = bucket_edges[i], bucket_edges[i + 1]
                    if (dj >= lo if i == 0 else dj > lo) and dj <= hi:
                        counts[i] += 1
                        break
            return counts

        split_info["train_bucket_counts"] = _bucket_counts(split["train"])
        split_info["test_bucket_counts"] = _bucket_counts(split["test"])

        # Optionally augment to ensure B0–B3 non-zero counts in both splits
        if args.ensure_buckets_nonzero:

            def augment(subset: List[IKSample], counts: List[int]) -> List[IKSample]:
                missing = [i for i, c in enumerate(counts[:4]) if c == 0]
                if not missing:
                    return subset
                augmented = subset[:]
                attempts = 0
                max_attempts = 200
                rng = np.random.default_rng()
                while missing and attempts < max_attempts:
                    attempts += 1
                    b = missing[0]
                    lo, hi = bucket_edges[b], bucket_edges[b + 1]
                    if b == 0:
                        theta2 = 0.0 if rng.random() < 0.5 else float(np.pi)
                    else:
                        target = float(rng.uniform(lo, hi))
                        theta2 = target if rng.random() < 0.5 else float(np.pi - target)
                    theta1 = float(rng.uniform(*generator.config.joint_limits[0]))
                    new_s = generator.generate_ik_samples(
                        [(theta1, theta2)], args.displacement_scale, args.damping_factor
                    )[0]
                    augmented.append(new_s)
                    # Update counts
                    dj = abs(new_s.det_J)
                    for i in range(len(bucket_edges) - 1):
                        lo2, hi2 = bucket_edges[i], bucket_edges[i + 1]
                        if (dj >= lo2 if i == 0 else dj > lo2) and dj <= hi2:
                            counts[i] += 1
                            break
                    missing = [i for i, c in enumerate(counts[:4]) if c == 0]
                return augmented

            split["train"] = augment(split["train"], split_info["train_bucket_counts"])
            split["test"] = augment(split["test"], split_info["test_bucket_counts"])
            # Recompute counts
            split_info["train_bucket_counts"] = _bucket_counts(split["train"])
            split_info["test_bucket_counts"] = _bucket_counts(split["test"])
            # Update samples ordering
            generator.samples = split["train"] + split["test"]
        # Overwrite generator.samples to contain combined in saved order (train then test)
        generator.samples = split["train"] + split["test"]

    # Save dataset (with added metadata if stratified)
    generator.save_dataset(args.output, args.format)
    if args.output.endswith(".json") and split_info is not None:
        # Append split metadata to JSON file
        try:
            with open(args.output, "r") as f:
                data = json.load(f)
            data["metadata"]["bucket_edges"] = bucket_edges
            data["metadata"]["stratified_by_detj"] = True
            data["metadata"]["train_ratio"] = args.train_ratio
            data["metadata"]["train_bucket_counts"] = split_info["train_bucket_counts"]
            data["metadata"]["test_bucket_counts"] = split_info["test_bucket_counts"]
            if args.singular_ratio_split:
                data["metadata"]["singular_ratio_split"] = args.singular_ratio_split
            data["metadata"]["ensured_buckets_nonzero"] = bool(args.ensure_buckets_nonzero)
            data["metadata"]["seed"] = args.seed
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(samples)}")
    print(f"Singular samples: {sum(1 for s in samples if s.is_singular)}")
    print(f"Average |det(J)|: {np.mean([abs(s.det_J) for s in samples]):.6f}")
    print(f"Min |det(J)|: {np.min([abs(s.det_J) for s in samples]):.6f}")
    print(
        f"Max condition number: {np.max([s.cond_J for s in samples if not np.isinf(s.cond_J)]):.2f}"
    )
    if split_info is not None:
        print("\nBucket edges:", bucket_edges)
        print("Train bucket counts:", split_info["train_bucket_counts"])
        print("Test bucket counts: ", split_info["test_bucket_counts"])
        # Warn if any near-pole buckets are empty
        for name, counts in (
            ("Train", split_info["train_bucket_counts"]),
            ("Test", split_info["test_bucket_counts"]),
        ):
            if any(c == 0 for c in counts[:4]):
                print(
                    f"Warning: {name} split has empty near-pole buckets B0–B3. Consider --ensure_buckets_nonzero or adjust sampling."
                )


if __name__ == "__main__":
    main()
