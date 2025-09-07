"""
Training utilities for ZeroProof.

This module provides tools for training with transreal arithmetic,
including adaptive loss policies and coverage tracking.
"""

from .coverage import CoverageTracker, CoverageMetrics
from .adaptive_loss import AdaptiveLambda, AdaptiveLossPolicy, AdaptiveLossConfig, create_adaptive_loss
from .trainer import TRTrainer, TrainingConfig, Optimizer
from .hybrid_trainer import HybridTRTrainer, HybridTrainingConfig
from .enhanced_coverage import EnhancedCoverageConfig

__all__ = [
    # Coverage tracking
    "CoverageTracker",
    "CoverageMetrics",
    "EnhancedCoverageConfig",
    
    # Adaptive loss
    "AdaptiveLambda",
    "AdaptiveLossPolicy",
    "AdaptiveLossConfig",
    "create_adaptive_loss",
    
    # Training
    "TRTrainer",
    "TrainingConfig",
    "Optimizer",
    "HybridTRTrainer",
    "HybridTrainingConfig",
]
