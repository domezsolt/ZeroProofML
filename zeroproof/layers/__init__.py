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
]