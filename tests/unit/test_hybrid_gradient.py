"""
Unit tests for hybrid gradient schedule functionality.

Tests the transition from Mask-REAL to Saturating modes,
pole detection, and gradient behavior near singularities.
"""

import math
from typing import List, Tuple

import pytest

from zeroproof.autodiff import TRNode, backward_pass
from zeroproof.autodiff.grad_mode import GradientMode, GradientModeConfig
from zeroproof.autodiff.hybrid_gradient import (
    HybridGradientContext,
    HybridGradientSchedule,
    ScheduleType,
    create_default_schedule,
)
from zeroproof.core import TRScalar, TRTag, ninf, phi, pinf, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.hybrid_rational import HybridTRRational


class TestHybridGradientSchedule:
    """Test the hybrid gradient schedule configuration."""

    def test_schedule_initialization(self):
        """Test schedule creation with default parameters."""
        schedule = HybridGradientSchedule(
            warmup_epochs=10, transition_epochs=20, delta_init=1e-2, delta_final=1e-6, enable=True
        )

        assert schedule.warmup_epochs == 10
        assert schedule.transition_epochs == 20
        assert schedule.delta_init == 1e-2
        assert schedule.delta_final == 1e-6
        assert schedule.enable == True

    def test_warmup_phase(self):
        """Test that warmup phase returns None for delta."""
        schedule = HybridGradientSchedule(warmup_epochs=5, transition_epochs=10, enable=True)

        # During warmup
        assert schedule.get_delta(0) is None
        assert schedule.get_delta(2) is None
        assert schedule.get_delta(4) is None
        assert schedule.is_warmup(4) == True

        # After warmup
        assert schedule.get_delta(5) is not None
        assert schedule.is_warmup(5) == False

    def test_linear_decay(self):
        """Test linear delta decay during transition."""
        schedule = HybridGradientSchedule(
            warmup_epochs=0,
            transition_epochs=10,
            delta_init=1.0,
            delta_final=0.0,
            schedule_type=ScheduleType.LINEAR,
            enable=True,
        )

        # Check linear interpolation
        assert abs(schedule.get_delta(0) - 1.0) < 1e-6
        assert abs(schedule.get_delta(5) - 0.5) < 1e-6
        assert abs(schedule.get_delta(10) - 0.0) < 1e-6

    def test_exponential_decay(self):
        """Test exponential delta decay."""
        schedule = HybridGradientSchedule(
            warmup_epochs=0,
            transition_epochs=10,
            delta_init=1e-2,
            delta_final=1e-6,
            schedule_type=ScheduleType.EXPONENTIAL,
            enable=True,
        )

        # Start value
        assert abs(schedule.get_delta(0) - 1e-2) < 1e-9

        # End value
        assert abs(schedule.get_delta(10) - 1e-6) < 1e-9

        # Middle should be geometric mean
        mid_expected = math.sqrt(1e-2 * 1e-6)
        assert abs(schedule.get_delta(5) - mid_expected) < 1e-7

    def test_cosine_decay(self):
        """Test cosine annealing decay."""
        schedule = HybridGradientSchedule(
            warmup_epochs=0,
            transition_epochs=10,
            delta_init=1.0,
            delta_final=0.0,
            schedule_type=ScheduleType.COSINE,
            enable=True,
        )

        # Check cosine shape
        assert abs(schedule.get_delta(0) - 1.0) < 1e-6
        assert abs(schedule.get_delta(10) - 0.0) < 1e-6

        # Middle point should be 0.5
        mid_val = schedule.get_delta(5)
        assert abs(mid_val - 0.5) < 1e-6

    def test_disabled_schedule(self):
        """Test that disabled schedule always returns None."""
        schedule = HybridGradientSchedule(enable=False)

        assert schedule.get_delta(0) is None
        assert schedule.get_delta(100) is None
        assert schedule.is_warmup(0) == False
        assert schedule.is_transitioning(10) == False


class TestHybridGradientContext:
    """Test the global hybrid gradient context."""

    def setup_method(self):
        """Reset context before each test."""
        HybridGradientContext.reset()
        GradientModeConfig.reset()

    def test_context_initialization(self):
        """Test context setup and configuration."""
        schedule = HybridGradientSchedule(
            warmup_epochs=5, transition_epochs=10, delta_init=1e-2, delta_final=1e-6, enable=True
        )

        HybridGradientContext.set_schedule(schedule)
        assert HybridGradientContext.get_schedule() == schedule

    def test_epoch_update(self):
        """Test epoch updates affect threshold."""
        schedule = HybridGradientSchedule(
            warmup_epochs=5, transition_epochs=10, delta_init=1e-2, delta_final=1e-6, enable=True
        )

        HybridGradientContext.set_schedule(schedule)

        # Warmup phase
        HybridGradientContext.update_epoch(2)
        assert HybridGradientContext._local_threshold is None

        # Transition phase
        HybridGradientContext.update_epoch(5)
        assert HybridGradientContext._local_threshold == 1e-2

        # Later in transition
        HybridGradientContext.update_epoch(10)
        threshold = HybridGradientContext._local_threshold
        assert threshold is not None
        assert threshold < 1e-2
        assert threshold > 1e-6

    def test_saturating_decision(self):
        """Test the decision logic for using saturating gradients."""
        schedule = HybridGradientSchedule(
            warmup_epochs=0, transition_epochs=10, delta_init=1e-2, delta_final=1e-6, enable=True
        )

        HybridGradientContext.set_schedule(schedule)
        HybridGradientContext.update_epoch(0)

        # Near pole (below threshold)
        assert HybridGradientContext.should_use_saturating(1e-3) == True
        assert HybridGradientContext.should_use_saturating(1e-4) == True

        # Far from pole (above threshold)
        assert HybridGradientContext.should_use_saturating(1e-1) == False
        assert HybridGradientContext.should_use_saturating(1.0) == False

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        schedule = HybridGradientSchedule(warmup_epochs=0, delta_init=1e-2, enable=True)

        HybridGradientContext.set_schedule(schedule)
        HybridGradientContext.update_epoch(0)
        HybridGradientContext.reset_statistics()

        # Make some decisions
        HybridGradientContext.should_use_saturating(1e-3)  # Near pole
        HybridGradientContext.should_use_saturating(1e-1)  # Far from pole
        HybridGradientContext.should_use_saturating(1e-4)  # Near pole

        stats = HybridGradientContext.get_statistics()
        assert stats["total_gradient_calls"] == 3
        assert stats["saturating_activations"] == 2
        assert stats["mask_real_activations"] == 1
        assert abs(stats["saturating_ratio"] - 2 / 3) < 1e-6


class TestHybridRationalLayer:
    """Test the hybrid rational layer implementation."""

    def setup_method(self):
        """Reset context before each test."""
        HybridGradientContext.reset()
        GradientModeConfig.reset()

    def test_layer_initialization(self):
        """Test hybrid rational layer creation."""
        schedule = create_default_schedule(warmup_epochs=5)

        layer = HybridTRRational(
            d_p=2, d_q=2, basis=MonomialBasis(), hybrid_schedule=schedule, track_Q_values=True
        )

        assert layer.hybrid_schedule == schedule
        assert layer.track_Q_values == True

    def test_forward_with_tracking(self):
        """Test forward pass with Q-value tracking."""
        schedule = HybridGradientSchedule(warmup_epochs=0, delta_init=1e-1, enable=True)

        layer = HybridTRRational(
            d_p=1, d_q=1, basis=MonomialBasis(), hybrid_schedule=schedule, track_Q_values=True
        )

        # Set parameters to create a pole at x=2
        layer.theta[0]._value = real(1.0)
        layer.theta[1]._value = real(0.0)
        layer.phi[0]._value = real(-2.0)  # Q = 1 - 2x, pole at x=0.5

        HybridGradientContext.update_epoch(0)

        # Forward pass at different distances from pole
        x1 = TRNode.constant(real(0.0))  # Q = 1.0
        x2 = TRNode.constant(real(0.4))  # Q = 0.2
        x3 = TRNode.constant(real(0.49))  # Q = 0.02 (near pole)

        y1, tag1 = layer.forward(x1)
        assert tag1 == TRTag.REAL

        y2, tag2 = layer.forward(x2)
        assert tag2 == TRTag.REAL

        y3, tag3 = layer.forward(x3)
        assert tag3 == TRTag.REAL

    def test_gradient_mode_switching(self):
        """Test that gradients switch modes based on schedule."""
        schedule = HybridGradientSchedule(warmup_epochs=0, delta_init=5e-2, enable=True)

        layer = HybridTRRational(d_p=1, d_q=1, basis=MonomialBasis(), hybrid_schedule=schedule)

        # Set up near-pole configuration
        layer.theta[0]._value = real(1.0)
        layer.phi[0]._value = real(-10.0)  # Q = 1 - 10x

        HybridGradientContext.update_epoch(0)
        GradientModeConfig.set_mode(GradientMode.HYBRID)
        GradientModeConfig.set_local_threshold(5e-2)

        # Test point very close to pole (x = 0.095, Q = 0.05)
        x_near = TRNode.parameter(real(0.095))
        y_near, _ = layer.forward(x_near)

        # Compute gradients
        y_near.backward()

        # Check that gradient was computed (not zeroed by Mask-REAL)
        # Near pole with hybrid mode should use saturating
        assert x_near.gradient is not None

        # Test point far from pole (x = 0.0, Q = 1.0)
        x_far = TRNode.parameter(real(0.0))
        x_far.zero_grad()
        y_far, _ = layer.forward(x_far)
        y_far.backward()

        # Far from pole should use standard gradients
        assert x_far.gradient is not None


class TestHybridIntegration:
    """Integration tests for hybrid gradient system."""

    def setup_method(self):
        """Reset context before each test."""
        HybridGradientContext.reset()
        GradientModeConfig.reset()

    def test_full_training_progression(self):
        """Test progression through warmup, transition, and convergence."""
        schedule = HybridGradientSchedule(
            warmup_epochs=2, transition_epochs=3, delta_init=1e-1, delta_final=1e-3, enable=True
        )

        HybridGradientContext.set_schedule(schedule)

        # Track mode descriptions
        descriptions = []

        for epoch in range(6):
            HybridGradientContext.update_epoch(epoch)
            desc = schedule.get_mode_description(epoch)
            descriptions.append(desc)

        # Check progression
        assert "warmup" in descriptions[0].lower()
        assert "warmup" in descriptions[1].lower()
        assert "transitioning" in descriptions[2].lower()
        assert "transitioning" in descriptions[3].lower()
        assert "transitioning" in descriptions[4].lower()
        assert "converged" in descriptions[5].lower()

    def test_gradient_behavior_near_pole(self):
        """Test that gradients behave correctly near poles."""
        from zeroproof.autodiff import tr_div

        # Create a simple division that creates a pole
        x = TRNode.parameter(real(0.01), name="x")  # Very small denominator
        y = TRNode.parameter(real(1.0), name="y")

        # Set up hybrid mode with threshold
        GradientModeConfig.set_mode(GradientMode.HYBRID)
        GradientModeConfig.set_local_threshold(0.02)

        schedule = HybridGradientSchedule(warmup_epochs=0, delta_init=0.02, enable=True)
        HybridGradientContext.set_schedule(schedule)
        HybridGradientContext.update_epoch(0)

        # Compute y/x (pole as x->0)
        z = tr_div(y, x)

        # Backward pass
        z.backward()

        # With hybrid mode and x < threshold, should use saturating
        # This means gradients should be bounded, not infinite
        assert x.gradient is not None
        assert x.gradient.tag == TRTag.REAL

        # Gradient should be bounded by saturation
        # For saturating reciprocal, the gradient magnitude should be limited
        grad_magnitude = abs(x.gradient.value) if x.gradient.tag == TRTag.REAL else float("inf")
        assert grad_magnitude < 1000  # Should be saturated, not infinite


def test_default_schedule_creation():
    """Test creation of default schedules."""
    # Conservative schedule
    conservative = create_default_schedule(aggressive=False)
    assert conservative.delta_init == 1e-2
    assert conservative.delta_final == 1e-6
    assert conservative.saturating_bound == 1.0

    # Aggressive schedule
    aggressive = create_default_schedule(aggressive=True)
    assert aggressive.delta_init == 1e-1
    assert aggressive.delta_final == 1e-8
    assert aggressive.saturating_bound == 0.1


def test_schedule_statistics_reset():
    """Test that statistics can be properly reset."""
    HybridGradientContext.reset()

    schedule = HybridGradientSchedule(enable=True)
    HybridGradientContext.set_schedule(schedule)
    HybridGradientContext.update_epoch(0)

    # Generate some statistics
    HybridGradientContext.should_use_saturating(1e-3)
    HybridGradientContext.should_use_saturating(1e-1)

    stats1 = HybridGradientContext.get_statistics()
    assert stats1["total_gradient_calls"] == 2

    # Reset statistics
    HybridGradientContext.reset_statistics()

    stats2 = HybridGradientContext.get_statistics()
    assert stats2["total_gradient_calls"] == 0
