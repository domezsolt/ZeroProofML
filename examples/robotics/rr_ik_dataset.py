"""
RR-arm inverse kinematics dataset generator.

This module generates datasets for training inverse kinematics near
singular Jacobians using a planar 2-link RR (Revolute-Revolute) robot.

The robot has two revolute joints with link lengths L1 and L2.
Singularities occur when the links are fully extended or retracted.
"""

import numpy as np
import math
import argparse
import json
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict

from zeroproof.core import real, TRTag
from zeroproof.autodiff import TRNode
from zeroproof.utils.metrics import PoleLocation


@dataclass
class RobotConfig:
    """Configuration for RR robot."""
    L1: float = 1.0  # Length of first link
    L2: float = 1.0  # Length of second link
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi), (-np.pi, np.pi)
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
        
        J = np.array([
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
            [self.L1 * c1 + self.L2 * c12, self.L2 * c12]
        ])
        
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
            return float('inf')
    
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
        return np.sqrt(np.linalg.det(J @ J.T))
    
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
        return det_J < threshold
    
    def damped_least_squares_ik(self, 
                               theta1: float, 
                               theta2: float,
                               dx: float, 
                               dy: float,
                               damping: float = 0.01) -> Tuple[float, float]:
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
    
    def sample_configurations(self, 
                            n_samples: int,
                            singular_ratio: float = 0.3,
                            singularity_threshold: float = 1e-3) -> List[Tuple[float, float]]:
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
        for _ in range(n_singular):
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
            theta1 = np.random.uniform(*self.config.joint_limits[0])
            theta2 = np.random.uniform(*self.config.joint_limits[1])
            
            # Reject if too close to singularity
            if self.robot.is_singular(theta1, theta2, singularity_threshold * 2):
                # Resample
                theta2 = np.random.uniform(0.1, np.pi - 0.1)
            
            configurations.append((theta1, theta2))
        
        return configurations
    
    def generate_ik_samples(self,
                           configurations: List[Tuple[float, float]],
                           displacement_scale: float = 0.1,
                           damping_factor: float = 0.01) -> List[IKSample]:
        """
        Generate IK samples from joint configurations.
        
        Args:
            configurations: List of (theta1, theta2) configurations
            displacement_scale: Scale for random end-effector displacements
            damping_factor: DLS damping factor
            
        Returns:
            List of IK samples
        """
        samples = []
        
        for theta1, theta2 in configurations:
            # Current end-effector position
            x_ee, y_ee = self.robot.forward_kinematics(theta1, theta2)
            
            # Random desired displacement
            dx = np.random.normal(0, displacement_scale)
            dy = np.random.normal(0, displacement_scale)
            
            # Compute IK solution using DLS
            dtheta1, dtheta2 = self.robot.damped_least_squares_ik(
                theta1, theta2, dx, dy, damping_factor
            )
            
            # Compute Jacobian properties
            det_J = self.robot.jacobian_determinant(theta1, theta2)
            cond_J = self.robot.jacobian_condition_number(theta1, theta2)
            manipulability = self.robot.manipulability_index(theta1, theta2)
            dist_to_sing = self.robot.distance_to_singularity(theta1, theta2)
            is_singular = self.robot.is_singular(theta1, theta2)
            
            sample = IKSample(
                dx=dx, dy=dy,
                theta1=theta1, theta2=theta2,
                dtheta1=dtheta1, dtheta2=dtheta2,
                det_J=det_J, cond_J=cond_J,
                is_singular=is_singular,
                x_ee=x_ee, y_ee=y_ee,
                manipulability=manipulability,
                distance_to_singularity=dist_to_sing
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self,
                        n_samples: int = 1000,
                        singular_ratio: float = 0.3,
                        displacement_scale: float = 0.1,
                        singularity_threshold: float = 1e-3,
                        damping_factor: float = 0.01) -> List[IKSample]:
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
            n_samples, singular_ratio, singularity_threshold
        )
        
        # Generate IK samples
        samples = self.generate_ik_samples(
            configurations, displacement_scale, damping_factor
        )
        
        self.samples = samples
        
        # Print statistics
        n_singular = sum(1 for s in samples if s.is_singular)
        print(f"Generated samples: {len(samples)}")
        print(f"Singular samples: {n_singular} ({n_singular/len(samples):.1%})")
        
        return samples
    
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
            # Convert to JSON-serializable format
            data = {
                'config': asdict(self.config),
                'samples': [asdict(sample) for sample in self.samples],
                'metadata': {
                    'n_samples': len(self.samples),
                    'n_singular': sum(1 for s in self.samples if s.is_singular),
                    'generator': 'RRDatasetGenerator'
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "npz":
            # Convert to numpy arrays
            arrays = {}
            for key in ['dx', 'dy', 'theta1', 'theta2', 'dtheta1', 'dtheta2',
                       'det_J', 'cond_J', 'x_ee', 'y_ee', 'manipulability',
                       'distance_to_singularity']:
                arrays[key] = np.array([getattr(s, key) for s in self.samples])
            
            arrays['is_singular'] = np.array([s.is_singular for s in self.samples])
            
            np.savez(filename, **arrays)
        
        print(f"Dataset saved to {filename}")
    
    @classmethod
    def load_dataset(cls, filename: str) -> 'RRDatasetGenerator':
        """
        Load dataset from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded dataset generator
        """
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            config = RobotConfig(**data['config'])
            generator = cls(config)
            
            # Reconstruct samples
            samples = []
            for sample_data in data['samples']:
                sample = IKSample(**sample_data)
                samples.append(sample)
            
            generator.samples = samples
            return generator
        
        else:
            raise ValueError(f"Unsupported file format: {filename}")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate RR robot IK dataset")
    
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--singular_ratio', type=float, default=0.3,
                       help='Fraction of samples near singularities')
    parser.add_argument('--displacement_scale', type=float, default=0.1,
                       help='Scale for end-effector displacements')
    parser.add_argument('--singularity_threshold', type=float, default=1e-3,
                       help='Threshold for singularity detection')
    parser.add_argument('--damping_factor', type=float, default=0.01,
                       help='DLS damping factor')
    parser.add_argument('--output', type=str, default='data/rr_ik_dataset.json',
                       help='Output filename')
    parser.add_argument('--format', type=str, choices=['json', 'npz'], default='json',
                       help='Output format')
    parser.add_argument('--L1', type=float, default=1.0,
                       help='Length of first link')
    parser.add_argument('--L2', type=float, default=1.0,
                       help='Length of second link')
    
    args = parser.parse_args()
    
    # Create robot configuration
    config = RobotConfig(L1=args.L1, L2=args.L2)
    
    # Generate dataset
    generator = RRDatasetGenerator(config)
    samples = generator.generate_dataset(
        n_samples=args.n_samples,
        singular_ratio=args.singular_ratio,
        displacement_scale=args.displacement_scale,
        singularity_threshold=args.singularity_threshold,
        damping_factor=args.damping_factor
    )
    
    # Save dataset
    generator.save_dataset(args.output, args.format)
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(samples)}")
    print(f"Singular samples: {sum(1 for s in samples if s.is_singular)}")
    print(f"Average |det(J)|: {np.mean([abs(s.det_J) for s in samples]):.6f}")
    print(f"Min |det(J)|: {np.min([abs(s.det_J) for s in samples]):.6f}")
    print(f"Max condition number: {np.max([s.cond_J for s in samples if not np.isinf(s.cond_J)]):.2f}")


if __name__ == "__main__":
    main()
