"""Transreal neural network layers."""

from .basis import (
    Basis,
    MonomialBasis,
    ChebyshevBasis,
    FourierBasis,
    create_basis,
)
from .tr_rational import (
    TRRational,
    TRRationalMulti,
)
from .tr_norm import (
    TRNorm,
    TRLayerNorm,
)
from .saturating_rational import (
    SaturatingTRRational,
    create_saturating_rational,
)
from .hybrid_rational import (
    HybridTRRational,
    HybridRationalWithPoleHead,
)
from .tag_aware_rational import (
    TagAwareRational,
    TagAwareMultiRational,
)
from .pole_aware_rational import (
    PoleAwareRational,
    FullyIntegratedRational,
)
from .multi_input_rational import (
    TRMultiInputRational,
)
from .enhanced_pole_detection import (
    EnhancedPoleConfig,
    EnhancedPoleDetectionHead,
    PoleRegularizer,
    PoleAwareRationalInterface,
)
from .enhanced_rational import (
    EnhancedTRRational,
    EnhancedTRRationalMulti,
    create_enhanced_rational,
)

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
