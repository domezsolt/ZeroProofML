"""
Integration tests for ZeroProofML.

These tests validate end-to-end functionality with actual singularities.
"""

from .test_pole_reconstruction import (
    GroundTruthRational,
    PoleReconstructionConfig,
    TestMultiDimensionalPoles,
    TestPoleReconstruction,
)
from .test_robotics_ik_singularities import (
    IKNeuralNetwork,
    RobotConfig,
    TestRoboticsIKSingularities,
    TestSingularityMetrics,
    TwoLinkRobot,
)
from .test_synthetic_rational_regression import (
    RegressionTestConfig,
    SyntheticRationalDataset,
    TestConvergenceMetrics,
    TestDatasetQuality,
    TestSyntheticRationalRegression,
)

__all__ = [
    # Synthetic regression
    "TestSyntheticRationalRegression",
    "TestDatasetQuality",
    "TestConvergenceMetrics",
    "RegressionTestConfig",
    "SyntheticRationalDataset",
    # Pole reconstruction
    "TestPoleReconstruction",
    "TestMultiDimensionalPoles",
    "PoleReconstructionConfig",
    "GroundTruthRational",
    # Robotics IK
    "TestRoboticsIKSingularities",
    "TestSingularityMetrics",
    "RobotConfig",
    "TwoLinkRobot",
    "IKNeuralNetwork",
]
