"""
Integration tests for ZeroProofML.

These tests validate end-to-end functionality with actual singularities.
"""

from .test_synthetic_rational_regression import (
    TestSyntheticRationalRegression,
    TestDatasetQuality,
    TestConvergenceMetrics,
    RegressionTestConfig,
    SyntheticRationalDataset,
)

from .test_pole_reconstruction import (
    TestPoleReconstruction,
    TestMultiDimensionalPoles,
    PoleReconstructionConfig,
    GroundTruthRational,
)

from .test_robotics_ik_singularities import (
    TestRoboticsIKSingularities,
    TestSingularityMetrics,
    RobotConfig,
    TwoLinkRobot,
    IKNeuralNetwork,
)

__all__ = [
    # Synthetic regression
    'TestSyntheticRationalRegression',
    'TestDatasetQuality',
    'TestConvergenceMetrics',
    'RegressionTestConfig',
    'SyntheticRationalDataset',
    
    # Pole reconstruction
    'TestPoleReconstruction',
    'TestMultiDimensionalPoles',
    'PoleReconstructionConfig',
    'GroundTruthRational',
    
    # Robotics IK
    'TestRoboticsIKSingularities',
    'TestSingularityMetrics',
    'RobotConfig',
    'TwoLinkRobot',
    'IKNeuralNetwork',
]
