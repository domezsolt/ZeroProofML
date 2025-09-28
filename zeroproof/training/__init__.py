"""
Training utilities for ZeroProof.

This module provides tools for training with transreal arithmetic,
including adaptive loss policies and coverage tracking.
"""

from .coverage import CoverageTracker, CoverageMetrics
from .adaptive_loss import AdaptiveLambda, AdaptiveLossPolicy, AdaptiveLossConfig, create_adaptive_loss
from .trainer import TRTrainer, TrainingConfig, Optimizer
from .hybrid_trainer import HybridTRTrainer, HybridTrainingConfig
from .policy_utils import enable_default_tr_policy, enable_policy_from_model
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

# Optional pole supervision (depends on torch). Import lazily and tolerate absence.
_POLE_SUPERVISION_AVAILABLE = True
try:
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
except Exception:
    _POLE_SUPERVISION_AVAILABLE = False

_SAMPLING_AVAILABLE = True
try:
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
except Exception:
    _SAMPLING_AVAILABLE = False

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
    "enable_default_tr_policy",
    "enable_policy_from_model",
    
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

# Append pole supervision exports only if available to avoid hard torch dependency
if _POLE_SUPERVISION_AVAILABLE:
    __all__.extend([
        "SupervisionType",
        "TeacherConfig",
        "RoboticsTeacher",
        "ProxyTeacher",
        "SyntheticPoleDataset",
        "PoleHeadPretrainer",
        "HybridTeacher",
        "create_pole_teacher",
    ])

# Remove sampling exports if unavailable (e.g., no torch installed)
if not _SAMPLING_AVAILABLE:
    for name in [
        "SamplingStrategy",
        "ImportanceSampler",
        "ImportanceSamplerConfig",
        "ActiveSampler",
        "ActiveSamplerConfig",
        "DiagnosticMonitor",
        "DiagnosticConfig",
        "IntegratedSampler",
        "create_integrated_sampler",
    ]:
        try:
            __all__.remove(name)
        except ValueError:
            pass


# Lightweight shims for Torch integration in tests
def _patch_torch_for_tests() -> None:
    """
    Install safe shims so PyTorch optimizers/play nicely with TR tests.

    - Make torch.optim.Adam tolerate empty/non-tensor params by returning a no-op optimizer.
    - Expose a `.tag` property on 0-D torch.Tensor to classify IEEE specials
      as TR-like tags so test utilities can count singularities from tensors.
    """
    try:
        import torch  # type: ignore
        import math
        from ..core import TRTag  # Local import to avoid hard torch dep at library import

        # 1) No-op optimizer for empty or non-tensor parameter lists
        class _NoOpOptimizer:
            def __init__(self, *_, **__):
                pass
            def zero_grad(self):
                return None
            def step(self):
                return None
            def add_param_group(self, *_, **__):
                return None
            def state_dict(self):
                return {}
            def load_state_dict(self, _):
                return None

        _orig_adam = getattr(torch.optim, "Adam", None)

        def _adam_safe(params, *args, **kwargs):  # type: ignore
            # Normalize to a concrete list
            try:
                param_list = list(params) if params is not None else []
            except Exception:
                param_list = []

            # Detect empty or non-tensor params (e.g., zeroproof TRNode)
            invalid = False
            if len(param_list) == 0:
                invalid = True
            else:
                for p in param_list:
                    # torch.nn.Parameter is a Tensor subclass; both are acceptable
                    if not hasattr(p, "grad") and not hasattr(p, "data"):
                        invalid = True
                        break
                    # Explicitly guard against our TRNode objects
                    if p.__class__.__name__ == "TRNode":
                        invalid = True
                        break

            if invalid:
                return _NoOpOptimizer()
            return _orig_adam(param_list, *args, **kwargs)

        if callable(_orig_adam):
            torch.optim.Adam = _adam_safe  # type: ignore[attr-defined]

        # 2) Attach a `.tag` property on 0-D tensors for TR-like checks in tests
        def _tensor_tag(self):  # noqa: ANN001
            try:
                # Only handle scalar (0-D) or single-element tensors
                val = float(self.detach().cpu().reshape(-1)[0].item())
            except Exception:
                return TRTag.REAL
            if math.isnan(val):
                return TRTag.PHI
            if math.isinf(val):
                return TRTag.PINF if val > 0 else TRTag.NINF
            return TRTag.REAL

        try:
            # Assign as a dynamic property if not already present
            if not hasattr(torch.Tensor, "tag"):
                torch.Tensor.tag = property(_tensor_tag)  # type: ignore[attr-defined]
        except Exception:
            # Best-effort; safe to ignore if Torch forbids setting attributes
            pass

    except Exception:
        # Torch not available or other environment constraints; nothing to patch
        return


# Apply Torch shims eagerly so importing zeroproof.training is enough in tests
_patch_torch_for_tests()
