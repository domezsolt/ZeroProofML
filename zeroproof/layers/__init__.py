"""Transreal neural network layers."""

from .basis import Basis, ChebyshevBasis, FourierBasis, MonomialBasis, create_basis
from .enhanced_pole_detection import (
    EnhancedPoleConfig,
    EnhancedPoleDetectionHead,
    PoleAwareRationalInterface,
    PoleRegularizer,
)
from .enhanced_rational import EnhancedTRRational, EnhancedTRRationalMulti, create_enhanced_rational
from .hybrid_rational import HybridRationalWithPoleHead, HybridTRRational
from .multi_input_rational import TRMultiInputRational
from .pole_aware_rational import FullyIntegratedRational, PoleAwareRational
from .saturating_rational import SaturatingTRRational, create_saturating_rational
from .tag_aware_rational import TagAwareMultiRational, TagAwareRational
from .tr_norm import TRLayerNorm, TRNorm
from .tr_rational import TRRational, TRRationalMulti
from .tr_softmax import pade_exp_approx, tr_softmax

__all__ = [
    # Basis functions
    "Basis",
    "MonomialBasis",
    "ChebyshevBasis",
    "FourierBasis",
    "create_basis",
    # Rational layers
    "TRRational",
    "TRRationalMulti",
    # Normalization layers
    "TRNorm",
    "TRLayerNorm",
    # Saturating gradient layers
    "SaturatingTRRational",
    "create_saturating_rational",
    # Hybrid gradient layers
    "HybridTRRational",
    "HybridRationalWithPoleHead",
    # Tag-aware layers
    "TagAwareRational",
    "TagAwareMultiRational",
    # Pole-aware layers
    "PoleAwareRational",
    "FullyIntegratedRational",
    # Multi-input rational model
    "TRMultiInputRational",
    # Softmax surrogate
    "tr_softmax",
    "pade_exp_approx",
    # Enhanced pole detection
    "EnhancedPoleConfig",
    "EnhancedPoleDetectionHead",
    "PoleRegularizer",
    "PoleAwareRationalInterface",
    # Enhanced rational layers
    "EnhancedTRRational",
    "EnhancedTRRationalMulti",
    "create_enhanced_rational",
]
