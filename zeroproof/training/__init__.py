"""
Training utilities for ZeroProof.

This module provides tools for training with transreal arithmetic,
including adaptive loss policies and coverage tracking.
"""

from .coverage import CoverageTracker, CoverageMetrics
from .adaptive_loss import AdaptiveLambda, AdaptiveLossPolicy, AdaptiveLossConfig, create_adaptive_loss
from .trainer import TRTrainer, TrainingConfig, Optimizer

__all__ = [
    # Coverage tracking
    "CoverageTracker",
    "CoverageMetrics",
    
    # Adaptive loss
    "AdaptiveLambda",
    "AdaptiveLossPolicy",
    "AdaptiveLossConfig",
    "create_adaptive_loss",
    
    # Training
    "TRTrainer",
    "TrainingConfig",
    "Optimizer",
]
