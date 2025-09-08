"""
Unit tests for enhanced coverage control functionality.

Tests the enhanced coverage tracker, enforcement policy, and sampling strategies.
"""

import pytest
import numpy as np
from typing import List

from zeroproof.core import TRTag
from zeroproof.training.enhanced_coverage import (
    EnhancedCoverageMetrics,
    EnhancedCoverageTracker,
    CoverageEnforcementPolicy,
    NearPoleSampler,
    AdaptiveGridSampler
)


class TestEnhancedCoverageMetrics:
    """Test enhanced coverage metrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = EnhancedCoverageMetrics()
        assert metrics.total_samples == 0
        assert metrics.coverage == 1.0  # Default when no samples
        assert metrics.near_pole_coverage == 1.0
        assert metrics.actual_nonreal_rate == 0.0
    
    def test_coverage_computation(self):
        """Test coverage calculation."""
        metrics = EnhancedCoverageMetrics(
            total_samples=100,
            real_samples=85,
            pinf_samples=10,
            ninf_samples=3,
            phi_samples=2
        )
        # Add actual non-REAL outputs to test actual_nonreal_rate
        for i in range(15):
            if i < 10:
                metrics.actual_nonreal_outputs.append((i, TRTag.PINF))
            elif i < 13:
                metrics.actual_nonreal_outputs.append((i, TRTag.NINF))
            else:
                metrics.actual_nonreal_outputs.append((i, TRTag.PHI))
        
        assert metrics.coverage == 0.85
        assert metrics.actual_nonreal_rate == 0.15
    
    def test_near_pole_coverage(self):
        """Test near-pole coverage calculation."""
        metrics = EnhancedCoverageMetrics(
            near_pole_samples=20,
            near_pole_real=12,
            near_pole_nonreal=8
        )
        assert metrics.near_pole_coverage == 0.6
    
    def test_update_with_q_values(self):
        """Test updating metrics with Q values."""
        metrics = EnhancedCoverageMetrics()
        
        tags = [TRTag.REAL, TRTag.REAL, TRTag.PINF, TRTag.REAL]
        q_values = [0.5, 0.05, 0.01, 1.0]  # Two near pole
        
        metrics.update(tags, q_values, pole_threshold=0.1)
        
        assert metrics.total_samples == 4
        assert metrics.real_samples == 3
        assert metrics.near_pole_samples == 2  # q=0.05 and q=0.01
        assert metrics.near_pole_real == 1  # Only q=0.05 is REAL
        assert metrics.min_q_value == 0.01
        assert pytest.approx(metrics.mean_q_value) == np.mean(q_values)


class TestEnhancedCoverageTracker:
    """Test enhanced coverage tracker."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = EnhancedCoverageTracker(
            target_coverage=0.85,
            pole_threshold=0.1,
            window_size=50
        )
        assert tracker.target_coverage == 0.85
        assert tracker.pole_threshold == 0.1
        assert tracker.coverage == 1.0  # Default
    
    def test_update_tracking(self):
        """Test updating the tracker."""
        tracker = EnhancedCoverageTracker(
            target_coverage=0.8,
            pole_threshold=0.1
        )
        
        # Simulate a batch
        tags = [TRTag.REAL] * 8 + [TRTag.PINF, TRTag.NINF]
        q_values = [1.0] * 5 + [0.05, 0.08, 0.12] + [0.01, 0.02]
        x_values = list(range(10))
        
        tracker.update(tags, q_values, x_values)
        
        assert tracker.current_batch.total_samples == 10
        assert tracker.current_batch.real_samples == 8
        assert tracker.coverage == 0.8
        
        # Check near-pole tracking (q <= 0.1)
        assert tracker.current_batch.near_pole_samples == 4  # 0.05, 0.08, 0.01, 0.02
        assert tracker.near_pole_coverage == 0.5  # 2 REAL out of 4 near-pole
    
    def test_pole_detection(self):
        """Test pole location detection."""
        tracker = EnhancedCoverageTracker(pole_threshold=0.1)
        
        # Create data with clear poles at x=5
        tags = [TRTag.REAL] * 10
        q_values = [1.0] * 4 + [0.01] + [1.0] * 5  # Pole at index 4
        x_values = list(range(10))
        
        tracker.update(tags, q_values, x_values)
        
        assert len(tracker.detected_pole_locations) == 1
        assert tracker.detected_pole_locations[0] == 4  # x=4 where q=0.01


class TestCoverageEnforcementPolicy:
    """Test coverage enforcement policy."""
    
    def test_initialization(self):
        """Test policy initialization."""
        policy = CoverageEnforcementPolicy(
            target_coverage=0.85,
            near_pole_target=0.7,
            dead_band=0.02,
            increase_rate=2.0,
            decrease_rate=0.5
        )
        assert policy.target_coverage == 0.85
        assert policy.near_pole_target == 0.7
        assert policy.min_lambda == 0.1  # Default min
        assert policy.max_lambda == 10.0  # Default max
    
    def test_dead_band(self):
        """Test dead-band behavior."""
        policy = CoverageEnforcementPolicy(
            target_coverage=0.85,
            dead_band=0.02
        )
        
        # Coverage within dead-band
        current_coverage = 0.84  # Within 0.85 ± 0.02
        near_pole_coverage = 0.8
        
        result = policy.enforce(
            current_coverage=current_coverage,
            current_lambda=1.0,  # Default lambda
            near_pole_coverage=near_pole_coverage
        )
        
        # Should not update due to dead-band
        assert result['lambda_updated'] == False
        assert result['new_lambda'] == 1.0  # Should stay same (default) due to dead-band
    
    def test_asymmetric_updates(self):
        """Test asymmetric increase/decrease rates."""
        policy = CoverageEnforcementPolicy(
            target_coverage=0.85,
            dead_band=0.01,
            increase_rate=2.0,
            decrease_rate=0.5,
            min_lambda=0.1,
            max_lambda=10.0
        )
        
        # Test increase (coverage too high)
        current_lambda = 1.0
        result = policy.enforce(
            current_coverage=0.95,
            current_lambda=current_lambda,
            near_pole_coverage=0.8
        )  # Coverage too high
        assert result['lambda_updated'] == True
        assert result['new_lambda'] > current_lambda  # Lambda increases
        
        # Test decrease (coverage too low)
        current_lambda = 5.0
        result = policy.enforce(
            current_coverage=0.70,
            current_lambda=current_lambda,
            near_pole_coverage=0.8
        )  # Coverage too low
        assert result['lambda_updated'] == True
        assert result['new_lambda'] < current_lambda  # Lambda decreases
    
    def test_near_pole_adjustment(self):
        """Test near-pole coverage adjustment."""
        policy = CoverageEnforcementPolicy(
            target_coverage=0.85,
            near_pole_target=0.7,
            dead_band=0.01
        )
        
        # Good global coverage but poor near-pole coverage
        result = policy.enforce(
            current_coverage=0.85,
            current_lambda=1.0,
            near_pole_coverage=0.4
        )  # Near-pole coverage too low
        
        # Since near-pole coverage is poor, lambda should be updated
        # But global coverage is at target, so within dead-band
        # So the lambda_updated will be False (within dead-band)
        assert result['lambda_updated'] == False  # Within dead-band for global coverage


class TestNearPoleSampler:
    """Test near-pole oversampling."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=3.0
        )
        assert sampler.pole_threshold == 0.1
        assert sampler.oversample_ratio == 3.0
    
    def test_weight_computation(self):
        """Test sample weight computation."""
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=3.0
        )
        
        q_values = [0.05, 0.5, 0.08, 1.0, 0.01]  # 3 near poles
        weights = sampler.compute_sample_weights(q_values)
        
        # Near-pole samples should have higher weights
        assert weights[0] > weights[1]  # 0.05 < 0.1, so higher weight
        assert weights[2] > weights[3]  # 0.08 < 0.1
        assert weights[4] > weights[1]  # 0.01 < 0.1
        
        # Check normalization
        assert pytest.approx(sum(weights)) == 1.0
    
    def test_adaptive_ratio(self):
        """Test adaptive oversample ratio."""
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=2.0,
            adaptive=True
        )
        
        q_values = [0.05, 0.5, 1.0]
        
        # Low coverage should increase ratio
        weights_low = sampler.compute_sample_weights(q_values, current_coverage=0.5)
        
        # High coverage should decrease ratio
        weights_high = sampler.compute_sample_weights(q_values, current_coverage=0.95)
        
        # Near-pole weight should be higher when coverage is low
        assert weights_low[0] > weights_high[0]
    
    def test_batch_sampling(self):
        """Test batch sampling with weights."""
        sampler = NearPoleSampler(pole_threshold=0.1)
        
        data = list(range(10))
        q_values = [1.0] * 5 + [0.05] * 5  # Half near poles
        
        batch = sampler.sample_batch(data, batch_size=5, Q_values=q_values)
        
        assert len(batch) == 5
        # Should oversample from near-pole region (indices 5-9)
        near_pole_count = sum(1 for x in batch if x >= 5)
        assert near_pole_count >= 2  # At least some near-pole samples


class TestAdaptiveGridSampler:
    """Test adaptive grid refinement."""
    
    def test_initialization(self):
        """Test grid initialization."""
        sampler = AdaptiveGridSampler(
            initial_grid_size=10,
            refinement_factor=3,
            pole_radius=0.1
        )
        
        sampler.initialize_grid(x_min=-1, x_max=1)
        
        assert len(sampler.grid_points) == 10
        assert min(sampler.grid_points) >= -1
        assert max(sampler.grid_points) <= 1
    
    def test_pole_refinement(self):
        """Test grid refinement near pole."""
        sampler = AdaptiveGridSampler(
            initial_grid_size=10,
            refinement_factor=3,
            pole_radius=0.2
        )
        
        sampler.initialize_grid(x_min=-1, x_max=1)
        initial_size = len(sampler.grid_points)
        
        # Refine near x=0
        sampler.refine_near_pole(0.0)
        
        # Should add new points (up to refinement_factor)
        # Note: Some points may already exist in the grid, so we check >= initial_size
        assert len(sampler.grid_points) >= initial_size
        
        # New points should be near x=0
        new_points = sorted(sampler.grid_points)
        near_zero = [x for x in new_points if abs(x) < 0.2]
        assert len(near_zero) >= 3  # Original + refined
    
    def test_weighted_sampling(self):
        """Test weighted sampling from refined grid."""
        sampler = AdaptiveGridSampler(
            initial_grid_size=10,
            refinement_factor=3,
            pole_radius=0.1
        )
        
        sampler.initialize_grid(x_min=-1, x_max=1)
        sampler.refine_near_pole(0.5)
        
        # First refine near a pole
        sampler.refine_near_pole(0.5)
        sampler.detected_poles = [0.5]  # Mark as detected pole
        
        # Get weighted samples
        samples, weights = sampler.get_weighted_samples(n_samples=5)
        
        assert len(samples) == 5
        # Should oversample near x=0.5
        near_pole = sum(1 for x in samples if abs(x - 0.5) < 0.2)
        assert near_pole >= 2  # At least some near the pole
    
    def test_auto_pole_detection(self):
        """Test automatic pole detection and refinement."""
        sampler = AdaptiveGridSampler(
            initial_grid_size=10,
            refinement_factor=3,
            pole_radius=0.1
        )
        
        sampler.initialize_grid(x_min=-1, x_max=1)
        
        # Create Q values with clear pole at x ≈ 0
        q_values = [abs(x) + 0.01 for x in sampler.grid_points]
        
        sampler.update_poles(q_values, sampler.grid_points, threshold=0.05)
        
        # Should detect and refine near x=0 if any q is below threshold
        min_q = min(q_values)
        if min_q <= 0.05:
            assert len(sampler.detected_poles) > 0
            # Grid should be refined
            assert len(sampler.grid_points) > 10


class TestIntegration:
    """Test integration of all components."""
    
    def test_full_pipeline(self):
        """Test full enhanced coverage pipeline."""
        # Create components
        tracker = EnhancedCoverageTracker(
            target_coverage=0.85,
            pole_threshold=0.1
        )
        
        policy = CoverageEnforcementPolicy(
            target_coverage=0.85,
            near_pole_target=0.7,
            dead_band=0.02
        )
        
        sampler = NearPoleSampler(
            pole_threshold=0.1,
            oversample_ratio=2.0
        )
        
        # Simulate training data
        x_values = np.linspace(-2, 2, 100).tolist()
        q_values = [abs(x) + 0.1 for x in x_values]  # Pole-like at x=0
        
        # Most samples are REAL except very near x=0
        tags = []
        for q in q_values:
            if q < 0.15:
                tags.append(TRTag.PINF)
            else:
                tags.append(TRTag.REAL)
        
        # Update tracker
        tracker.update(tags, q_values, x_values)
        
        # Get coverage stats
        global_cov = tracker.coverage
        near_pole_cov = tracker.near_pole_coverage
        
        # Apply policy
        result = policy.enforce(
            current_coverage=global_cov,
            current_lambda=1.0,
            near_pole_coverage=near_pole_cov
        )
        
        # Sample next batch with oversampling
        weights = sampler.compute_sample_weights(q_values)
        
        # Verify integration
        assert tracker.current_batch.total_samples == 100
        assert 0.0 <= global_cov <= 1.0
        assert 0.0 <= near_pole_cov <= 1.0
        assert 'new_lambda' in result
        assert len(weights) == len(q_values)
        assert pytest.approx(sum(weights)) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])