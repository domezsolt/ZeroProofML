"""
Enhanced dataset generation utilities for ZeroProof.

This module provides improved dataset generation that ensures actual singularities
are included in the training data, addressing the critical issue of 100% coverage.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import TRScalar, ninf, phi, pinf, real


@dataclass
class SingularityInfo:
    """Information about a singularity in the dataset."""

    location: float
    type: str  # 'pole', 'removable', 'essential'
    strength: float  # How close samples should get


class SingularDatasetGenerator:
    """
    Generate datasets with guaranteed singularities.

    This generator ensures that:
    1. Actual singular points (Q(x) = 0) are included
    2. Near-singular points are properly distributed
    3. Ground truth pole locations are tracked
    """

    def __init__(self, domain: Tuple[float, float] = (-1.0, 1.0), seed: Optional[int] = None):
        """
        Initialize dataset generator.

        Args:
            domain: Input domain (min, max)
            seed: Random seed for reproducibility
        """
        self.domain = domain
        if seed is not None:
            np.random.seed(seed)
        self.singularities: List[SingularityInfo] = []

    def add_pole(self, location: float, strength: float = 0.01) -> None:
        """
        Add a pole singularity to the dataset.

        Args:
            location: x-coordinate of the pole
            strength: Minimum distance for near-pole samples
        """
        self.singularities.append(SingularityInfo(location, "pole", strength))

    def generate_rational_function_data(
        self,
        n_samples: int,
        singularity_ratio: float = 0.3,
        force_exact_singularities: bool = True,
        noise_level: float = 0.01,
    ) -> Tuple[List[TRScalar], List[TRScalar], Dict[str, Any]]:
        """
        Generate data for a rational function with known singularities.

        Args:
            n_samples: Total number of samples
            singularity_ratio: Fraction of samples that should be near/at singularities
            force_exact_singularities: If True, include samples exactly at singularities
            noise_level: Noise to add to targets

        Returns:
            Tuple of (inputs, targets, metadata)
        """
        if not self.singularities:
            # Add default singularities if none specified
            self.add_pole(0.5, 0.01)
            self.add_pole(-0.3, 0.01)

        x_vals = []
        y_vals = []
        metadata = {
            "singularities": self.singularities,
            "exact_singular_indices": [],
            "near_singular_indices": [],
            "regular_indices": [],
            "tag_distribution": {"REAL": 0, "PINF": 0, "NINF": 0, "PHI": 0},
        }

        n_singular = int(n_samples * singularity_ratio)
        n_regular = n_samples - n_singular

        # Generate singular samples (at and near singularities)
        # Ensure a reasonable fraction are exact singularities
        n_exact = 0
        if force_exact_singularities:
            n_exact = max(len(self.singularities), int(0.2 * n_singular))
        for i in range(n_singular):
            if force_exact_singularities and i < n_exact:
                # Place samples exactly at singularities in round-robin
                sing = self.singularities[i % len(self.singularities)]
                x = sing.location
                metadata["exact_singular_indices"].append(i)
            else:
                # Place sample near a random singularity
                sing = np.random.choice(self.singularities)

                # Use exponential distribution for distance from singularity
                # This ensures we get very close samples
                distance = np.random.exponential(sing.strength)
                distance = min(distance, 0.1)  # Cap maximum distance

                # Random side
                side = np.random.choice([-1, 1])
                x = sing.location + side * distance

                if distance < sing.strength:
                    metadata["near_singular_indices"].append(i)

            x_vals.append(real(x))

            # Compute target based on rational function
            y = self._compute_rational_target(x, noise_level)
            y_vals.append(y)

            # Track tag distribution
            if y.tag.name in metadata["tag_distribution"]:
                metadata["tag_distribution"][y.tag.name] += 1

        # Generate regular samples (away from singularities)
        for i in range(n_regular):
            x = self._sample_regular_point()
            x_vals.append(real(x))

            y = self._compute_rational_target(x, noise_level)
            y_vals.append(y)

            metadata["regular_indices"].append(n_singular + i)
            if y.tag.name in metadata["tag_distribution"]:
                metadata["tag_distribution"][y.tag.name] += 1

        # Shuffle to avoid ordering bias
        indices = np.random.permutation(n_samples)
        x_vals = [x_vals[i] for i in indices]
        y_vals = [y_vals[i] for i in indices]

        # Update indices in metadata after shuffling
        inverse_indices = {old: new for new, old in enumerate(indices)}
        metadata["exact_singular_indices"] = [
            inverse_indices[i] for i in metadata["exact_singular_indices"]
        ]
        metadata["near_singular_indices"] = [
            inverse_indices[i] for i in metadata["near_singular_indices"]
        ]
        metadata["regular_indices"] = [inverse_indices[i] for i in metadata["regular_indices"]]

        return x_vals, y_vals, metadata

    def _compute_rational_target(self, x: float, noise_level: float) -> TRScalar:
        """
        Compute target value for rational function with singularities.

        This implements: y = sum(1/(x - pole_i)) + polynomial_background
        """
        # Check if exactly at a singularity
        for sing in self.singularities:
            if abs(x - sing.location) < 1e-10:
                # Exactly at pole
                if sing.type == "pole":
                    # Determine sign based on approach direction
                    return pinf() if x > sing.location else ninf()
                elif sing.type == "removable":
                    return phi()

        # Not at singularity, compute value
        y = 0.0

        # Add pole contributions
        for sing in self.singularities:
            if sing.type == "pole":
                denominator = x - sing.location
                if abs(denominator) < 1e-8:
                    # Very close to pole, return large value
                    contribution = 1e8 / denominator
                else:
                    contribution = 1.0 / denominator
                y += contribution

        # Add smooth background
        y += 0.5 * x + 0.1 * x**2

        # Add noise
        if noise_level > 0:
            y += np.random.normal(0, noise_level)

        # Clip extreme values (but not at singularities). Preserve sign
        if abs(y) > 1e6:
            y = float(np.sign(y) * 1e6)

        return real(y)

    def _sample_regular_point(self) -> float:
        """Sample a point away from all singularities."""
        min_distance = 0.1
        max_attempts = 100

        for _ in range(max_attempts):
            x = np.random.uniform(self.domain[0], self.domain[1])

            # Check distance to all singularities
            too_close = False
            for sing in self.singularities:
                if abs(x - sing.location) < min_distance:
                    too_close = True
                    break

            if not too_close:
                return x

        # If we couldn't find a point far from singularities,
        # just return a random point
        return np.random.uniform(self.domain[0], self.domain[1])

    def generate_importance_weighted_samples(
        self, n_samples: int, weight_fn: Optional[callable] = None
    ) -> Tuple[List[TRScalar], List[TRScalar], List[float]]:
        """
        Generate samples with importance weighting near singularities.

        Args:
            n_samples: Number of samples
            weight_fn: Function to compute importance weight (default: 1/|Q(x)|²)

        Returns:
            Tuple of (inputs, targets, weights)
        """
        if weight_fn is None:

            def weight_fn(x):
                # Default: weight inversely proportional to distance from nearest pole
                min_dist = min(abs(x - s.location) for s in self.singularities)
                return 1.0 / (min_dist**2 + 0.01)

        x_vals = []
        y_vals = []
        weights = []

        for _ in range(n_samples):
            # Sample with probability proportional to weight
            x = self._importance_sample()
            x_vals.append(real(x))

            y = self._compute_rational_target(x, noise_level=0.01)
            y_vals.append(y)

            w = weight_fn(x)
            weights.append(w)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight * n_samples for w in weights]

        return x_vals, y_vals, weights

    def _importance_sample(self) -> float:
        """Sample a point with higher probability near singularities."""
        # Mix of uniform and concentrated sampling
        if np.random.random() < 0.5:
            # Sample near a singularity
            sing = np.random.choice(self.singularities)
            distance = np.random.exponential(0.02)
            side = np.random.choice([-1, 1])
            x = sing.location + side * distance
            x = np.clip(x, self.domain[0], self.domain[1])
        else:
            # Uniform sample
            x = np.random.uniform(self.domain[0], self.domain[1])

        return x


def generate_robotics_singular_configurations(
    n_samples: int, L1: float = 1.0, L2: float = 1.0, include_exact_singularities: bool = True
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Generate robot configurations including actual singularities.

    For a 2-link RR robot, singularities occur when:
    - θ2 = 0 (fully extended)
    - θ2 = π (fully retracted)

    Args:
        n_samples: Number of configurations
        L1, L2: Link lengths
        include_exact_singularities: If True, include exact singular configurations

    Returns:
        Tuple of (configurations, metadata)
    """
    configs = []
    metadata = {
        "singular_configs": [],
        "near_singular_configs": [],
        "regular_configs": [],
        "det_J_values": [],
    }

    n_singular = n_samples // 3
    n_near = n_samples // 3
    n_regular = n_samples - n_singular - n_near

    # Exact singular configurations
    if include_exact_singularities:
        for i in range(min(n_singular, 10)):  # At least 10 exact singular points
            theta1 = np.random.uniform(-np.pi, np.pi)

            if i % 2 == 0:
                theta2 = 0.0  # Exactly singular
            else:
                theta2 = np.pi  # Exactly singular

            configs.append((theta1, theta2))
            metadata["singular_configs"].append(len(configs) - 1)

            # Jacobian determinant is exactly 0
            metadata["det_J_values"].append(0.0)

    # Near-singular configurations
    for _ in range(n_near):
        theta1 = np.random.uniform(-np.pi, np.pi)

        # Very close to singularity
        if np.random.random() < 0.5:
            theta2 = np.random.normal(0, 0.01)  # Near θ2 = 0
        else:
            theta2 = np.random.normal(np.pi, 0.01)  # Near θ2 = π

        configs.append((theta1, theta2))
        metadata["near_singular_configs"].append(len(configs) - 1)

        # Compute actual det(J)
        det_J = -L1 * L2 * np.sin(theta2)
        metadata["det_J_values"].append(det_J)

    # Regular configurations
    for _ in range(n_regular):
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(0.3, np.pi - 0.3)  # Away from singularities

        configs.append((theta1, theta2))
        metadata["regular_configs"].append(len(configs) - 1)

        det_J = -L1 * L2 * np.sin(theta2)
        metadata["det_J_values"].append(det_J)

    return configs, metadata
