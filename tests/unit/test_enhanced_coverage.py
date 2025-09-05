"""
Unit tests for enhanced coverage control functionality.

Tests the adaptive lambda controller, near-pole sampler, and
coverage enforcement policy.
"""

import pytest
import numpy as np
from typing import List

from zeroproof.core import TRTag
from zeroproof.training.enhanced_coverage import (
    EnhancedCoverageConfig,
    AdaptiveLambdaController,
    NearPoleSampler,
    CoverageEnforcementPolicy,
    CoverageStrategy
)


class TestAdaptiveLambdaController:
    """Test adaptive lambda controller."""
    
    def test_initialization(self):
        """Test controller initialization."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            lambda_init=2.0,
            lambda_min=0.1,
            lambda_max=10.0
        )
        
        controller = AdaptiveLambdaController(config)
        
        assert controller.lambda_value == 2.0
        assert controller.epoch == 0
        assert len(controller.lambda_history) == 1
    
    def test_warmup_period(self):
        """Test that lambda doesn't change during warmup."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            lambda_init=1.0,
            warmup_epochs=5
        )
        
        controller = AdaptiveLambdaController(config)
        initial_lambda = controller.lambda_value
        
        # Update during warmup
        for i in range(5):
            controller.update(0.5)  # Low coverage
            assert controller.lambda_value == initial_lambda
        
        # After warmup, should update
        controller.update(0.5)
        assert controller.lambda_value != initial_lambda
    
    def test_lagrange_update(self):
        """Test Lagrange multiplier update strategy."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            lambda_init=1.0,
            strategy=CoverageStrategy.LAGRANGE,
            learning_rate=0.1,
            momentum=0.0,
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        
        # Low coverage -> decrease lambda
        controller.update(0.7)  # Below target
        assert controller.lambda_value < 1.0
        
        # High coverage -> increase lambda
        controller.lambda_value = 1.0
        controller.update(0.95)  # Above target
        assert controller.lambda_value > 1.0
    
    def test_pid_update(self):
        """Test PID controller update strategy."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            lambda_init=1.0,
            strategy=CoverageStrategy.PID,
            pid_gains=(1.0, 0.1, 0.01),
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        
        # Track lambda changes
        lambdas = [controller.lambda_value]
        
        # Simulate coverage oscillation
        coverages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.87, 0.85]
        for cov in coverages:
            controller.update(cov)
            lambdas.append(controller.lambda_value)
        
        # Should converge toward stable value
        late_variance = np.var(lambdas[-3:])
        early_variance = np.var(lambdas[:3])
        assert late_variance < early_variance
    
    def test_adaptive_rate_update(self):
        """Test adaptive learning rate strategy."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            min_coverage=0.70,
            lambda_init=5.0,
            strategy=CoverageStrategy.ADAPTIVE_RATE,
            learning_rate=0.1,
            momentum=0.0,
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        
        # Emergency coverage (very low)
        controller.update(0.5)
        emergency_change = abs(controller.lambda_value - 5.0)
        
        # Reset and test normal coverage
        controller.lambda_value = 5.0
        controller.velocity = 0.0
        controller.update(0.83)  # Near target
        normal_change = abs(controller.lambda_value - 5.0)
        
        # Emergency should have larger change
        assert emergency_change > normal_change * 2
    
    def test_lambda_clamping(self):
        """Test that lambda stays within bounds."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            lambda_init=1.0,
            lambda_min=0.5,
            lambda_max=2.0,
            learning_rate=10.0,  # Large to test clamping
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        
        # Try to push lambda below min
        controller.update(0.3)  # Very low coverage
        assert controller.lambda_value >= 0.5
        
        # Try to push lambda above max
        controller.lambda_value = 2.0
        controller.update(1.0)  # Perfect coverage
        assert controller.lambda_value <= 2.0
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            min_coverage=0.70,
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        
        # Generate some coverage history
        coverages = [0.6, 0.7, 0.8, 0.85, 0.9]
        for cov in coverages:
            controller.update(cov)
        
        stats = controller.get_statistics()
        
        assert stats['adjustments'] == 5
        assert stats['violations'] == 1  # Only 0.6 < 0.70
        assert abs(stats['avg_coverage'] - np.mean(coverages)) < 1e-6
        assert stats['min_coverage'] == 0.6
        assert stats['max_coverage'] == 0.9


class TestNearPoleSampler:
    """Test near-pole sampler."""
    
    def test_weight_computation(self):
        """Test sample weight computation."""
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=3.0,
            adaptive=False
        )
        
        # Q values with some near pole
        Q_values = [1.0, 0.5, 0.05, 0.2, 0.01, 0.8]
        
        weights = sampler.compute_sample_weights(Q_values)
        
        # Near-pole samples (indices 2, 4) should have higher weight
        assert weights[2] > weights[0]
        assert weights[4] > weights[0]
        assert weights[2] / weights[0] == pytest.approx(3.0, rel=0.1)
        
        # Weights should sum to 1
        assert abs(weights.sum() - 1.0) < 1e-6
    
    def test_adaptive_oversampling(self):
        """Test adaptive adjustment of oversample ratio."""
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=2.0,
            adaptive=True
        )
        
        Q_values = [0.05, 0.5, 1.0]
        
        # Low coverage -> increase oversampling
        weights1 = sampler.compute_sample_weights(Q_values, current_coverage=0.6)
        ratio1 = sampler.oversample_ratio
        
        # Good coverage -> normal oversampling
        sampler.oversample_ratio = 2.0  # Reset
        weights2 = sampler.compute_sample_weights(Q_values, current_coverage=0.85)
        ratio2 = sampler.oversample_ratio
        
        assert ratio1 > ratio2
        assert weights1[0] > weights2[0]  # More weight on near-pole when coverage low
    
    def test_sample_batch(self):
        """Test batch sampling with weights."""
        sampler = NearPoleSampler(pole_threshold=0.1)
        
        # Create dummy data
        data = [(i, i*2) for i in range(10)]
        Q_values = [0.05 if i < 2 else 0.5 for i in range(10)]
        
        # Sample batch
        batch = sampler.sample_batch(data, batch_size=100, Q_values=Q_values)
        
        # Count how many times near-pole samples appear
        near_pole_count = sum(1 for item in batch if item[0] < 2)
        far_pole_count = sum(1 for item in batch if item[0] >= 2)
        
        # Near-pole should be oversampled
        near_pole_ratio = near_pole_count / 100
        expected_ratio = 0.2 * sampler.oversample_ratio / (
            0.2 * sampler.oversample_ratio + 0.8
        )
        
        # Check within reasonable bounds (sampling is stochastic)
        assert abs(near_pole_ratio - expected_ratio) < 0.2


class TestCoverageEnforcementPolicy:
    """Test coverage enforcement policy."""
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            min_coverage=0.70
        )
        
        policy = CoverageEnforcementPolicy(config)
        
        assert policy.lambda_controller is not None
        assert policy.config.target_coverage == 0.85
    
    def test_enforcement_actions(self):
        """Test enforcement action generation."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            min_coverage=0.70,
            warmup_epochs=0,
            oversample_near_pole=True
        )
        
        policy = CoverageEnforcementPolicy(config)
        
        # Normal coverage
        actions = policy.enforce(0.82, epoch=1)
        assert actions['lambda_updated'] == True
        assert actions['intervention_triggered'] == False
        
        # Critical coverage
        actions = policy.enforce(0.65, epoch=2)
        assert actions['intervention_triggered'] == True
        assert len(policy.interventions) == 1
    
    def test_intervention_triggering(self):
        """Test that interventions are triggered correctly."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            min_coverage=0.70,
            lambda_init=2.0,
            warmup_epochs=0
        )
        
        policy = CoverageEnforcementPolicy(config)
        initial_lambda = policy.lambda_controller.lambda_value
        
        # Trigger intervention with very low coverage
        actions = policy.enforce(0.5, epoch=1)
        
        assert actions['intervention_triggered'] == True
        # Lambda should be reduced (emergency)
        assert policy.lambda_controller.lambda_value < initial_lambda
        assert len(policy.interventions) == 1
        assert policy.interventions[0]['coverage'] == 0.5
    
    def test_coverage_restoration_tracking(self):
        """Test tracking of coverage restoration."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            warmup_epochs=0
        )
        
        policy = CoverageEnforcementPolicy(config)
        
        # Low coverage initially
        policy.enforce(0.7, epoch=1)
        assert policy.coverage_restored_epoch is None
        
        # Coverage restored
        policy.enforce(0.86, epoch=5)
        assert policy.coverage_restored_epoch == 5
        
        # Stays recorded
        policy.enforce(0.84, epoch=6)
        assert policy.coverage_restored_epoch == 5
    
    def test_statistics_collection(self):
        """Test comprehensive statistics collection."""
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            oversample_near_pole=True,
            warmup_epochs=0
        )
        
        policy = CoverageEnforcementPolicy(config)
        
        # Generate some history
        coverages = [0.65, 0.7, 0.75, 0.85, 0.86]
        for i, cov in enumerate(coverages):
            policy.enforce(cov, epoch=i+1)
        
        stats = policy.get_statistics()
        
        assert 'lambda_stats' in stats
        assert stats['interventions'] == 1  # Only first was below min
        assert stats['coverage_restored_epoch'] == 4  # When hit 0.85


def test_integration_with_coverage_strategies():
    """Test different coverage strategies work together."""
    strategies = [
        CoverageStrategy.LAGRANGE,
        CoverageStrategy.PID,
        CoverageStrategy.ADAPTIVE_RATE,
        CoverageStrategy.DUAL_PHASE
    ]
    
    for strategy in strategies:
        config = EnhancedCoverageConfig(
            target_coverage=0.85,
            strategy=strategy,
            warmup_epochs=0
        )
        
        controller = AdaptiveLambdaController(config)
        initial_lambda = controller.lambda_value
        
        # All should respond to low coverage
        controller.update(0.6)
        assert controller.lambda_value != initial_lambda


def test_coverage_enforcement_with_near_pole_sampling():
    """Test that near-pole sampling integrates with enforcement."""
    config = EnhancedCoverageConfig(
        target_coverage=0.85,
        oversample_near_pole=True,
        pole_threshold=0.1,
        warmup_epochs=0
    )
    
    policy = CoverageEnforcementPolicy(config)
    
    # Provide Q values for sampling
    Q_values = [0.05, 0.2, 0.5, 1.0, 0.02, 0.8]
    
    actions = policy.enforce(0.75, epoch=1, Q_values=Q_values)
    
    assert actions['sampling_adjusted'] == True
    assert 'sample_weights' in actions
    
    # Check weights favor near-pole samples
    weights = actions['sample_weights']
    assert weights[0] > weights[2]  # 0.05 < 0.1 threshold
    assert weights[4] > weights[3]  # 0.02 < 0.1 threshold
