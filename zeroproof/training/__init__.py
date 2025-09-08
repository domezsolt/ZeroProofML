"""
Training utilities for ZeroProof.

This module provides tools for training with transreal arithmetic,
including adaptive loss policies and coverage tracking.
"""

from .coverage import CoverageTracker, CoverageMetrics
from .adaptive_loss import AdaptiveLambda, AdaptiveLossPolicy, AdaptiveLossConfig, create_adaptive_loss
from .trainer import TRTrainer, TrainingConfig, Optimizer
from .hybrid_trainer import HybridTRTrainer, HybridTrainingConfig
from .enhanced_coverage import (
    EnhancedCoverageMetrics,
    EnhancedCoverageTracker,
    CoverageEnforcementPolicy,
    NearPoleSampler,
    AdaptiveGridSampler,
)

from .advanced_control import (
    ControlStrategy,
    PIController,
    PIControllerConfig,
    CurriculumScheduler,
    CurriculumConfig,
    HybridController,
    create_advanced_controller,
)

from .control_ablation import (
    AblationConfig,
    AblationRunner,
    run_control_ablation,
)

from .pole_supervision import (
    SupervisionType,
    TeacherConfig,
    RoboticsTeacher,
    ProxyTeacher,
    SyntheticPoleDataset,
    PoleHeadPretrainer,
    HybridTeacher,
    create_pole_teacher,
)

from .sampling_diagnostics import (
    SamplingStrategy,
    ImportanceSampler,
    ImportanceSamplerConfig,
    ActiveSampler,
    ActiveSamplerConfig,
    DiagnosticMonitor,
    DiagnosticConfig,
    IntegratedSampler,
    create_integrated_sampler,
)

__all__ = [
    # Coverage tracking
    "CoverageTracker",
    "CoverageMetrics",
    "EnhancedCoverageMetrics",
    "EnhancedCoverageTracker",
    "CoverageEnforcementPolicy",
    "NearPoleSampler",
    "AdaptiveGridSampler",
    
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
    
    # Advanced control
    "ControlStrategy",
    "PIController",
    "PIControllerConfig",
    "CurriculumScheduler",
    "CurriculumConfig",
    "HybridController",
    "create_advanced_controller",
    
    # Control ablation
    "AblationConfig",
    "AblationRunner",
    "run_control_ablation",
    
    # Pole supervision
    "SupervisionType",
    "TeacherConfig",
    "RoboticsTeacher",
    "ProxyTeacher",
    "SyntheticPoleDataset",
    "PoleHeadPretrainer",
    "HybridTeacher",
    "create_pole_teacher",
    
    # Sampling and diagnostics
    "SamplingStrategy",
    "ImportanceSampler",
    "ImportanceSamplerConfig",
    "ActiveSampler",
    "ActiveSamplerConfig",
    "DiagnosticMonitor",
    "DiagnosticConfig",
    "IntegratedSampler",
    "create_integrated_sampler",
]
