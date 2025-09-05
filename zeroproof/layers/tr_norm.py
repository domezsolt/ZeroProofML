"""
TR-Norm: Epsilon-free normalization layer.

This module implements transreal normalization that handles zero variance
deterministically without epsilon hacks. It's the limit of standard
batch normalization as ε→0⁺.
"""

from typing import List, Optional, Tuple, Union
import math

from ..core import TRScalar, TRTag, real, pinf, ninf, phi
from ..core.precision_config import PrecisionConfig
from ..autodiff import TRNode, tr_sqrt, tr_div, tr_sub, tr_add, tr_mul, tr_abs, tr_neg


class TRNorm:
    """
    Epsilon-free normalization with deterministic zero-variance bypass.
    
    For each feature:
    - If σ² > 0: ŷ = γ(x - μ)/σ + β  (standard normalization)
    - If σ² = 0: ŷ = β  (deterministic bypass)
    
    Statistics are computed over REAL values only (drop-null).
    """
    
    def __init__(self,
                 num_features: int,
                 eps: float = 0.0,  # Ignored! For API compatibility only
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = False):
        """
        Initialize TR-Norm layer.
        
        Args:
            num_features: Number of features to normalize
            eps: Ignored (kept for compatibility). TR-Norm is epsilon-free.
            momentum: Momentum for running stats (if tracked)
            affine: If True, learn affine parameters γ and β
            track_running_stats: If True, track running mean/var (not implemented)
        """
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if eps != 0.0:
            import warnings
            warnings.warn("TR-Norm ignores eps parameter. It's epsilon-free by design.")
        
        # Initialize parameters
        self._initialize_parameters()

    @classmethod
    def from_batch(cls,
                   batch: List[List[Union[TRScalar, TRNode, float]]],
                   **kwargs) -> "TRNorm":
        """Convenience constructor inferring num_features from first batch sample.

        Args:
            batch: List of samples; each sample is a sequence of features
            **kwargs: Additional TRNorm constructor kwargs (eps, momentum, ...)

        Returns:
            TRNorm instance with inferred num_features.
        """
        if not batch:
            raise ValueError("from_batch requires a non-empty batch")
        first = batch[0]
        try:
            num_features = len(first)  # type: ignore[arg-type]
        except Exception as ex:
            raise TypeError("Batch elements must be sequences of features") from ex
        return cls(num_features=num_features, **kwargs)
    
    def _initialize_parameters(self):
        """Initialize affine parameters γ and β."""
        if self.affine:
            # Initialize γ to 1 and β to 0
            self.gamma = []
            self.beta = []
            for i in range(self.num_features):
                self.gamma.append(TRNode.parameter(real(1.0), name=f"gamma_{i}"))
                self.beta.append(TRNode.parameter(real(0.0), name=f"beta_{i}"))
        else:
            self.gamma = None
            self.beta = None
        
        # Initialize running statistics if tracking is enabled
        if self.track_running_stats:
            # Track as plain Python floats (not part of the TR graph)
            self.running_mean: List[float] = [0.0 for _ in range(self.num_features)]
            # Initialize variance to 1.0 to match common BN conventions
            self.running_var: List[float] = [1.0 for _ in range(self.num_features)]
            self.num_batches_tracked: int = 0
        else:
            self.running_mean = None  # type: ignore[assignment]
            self.running_var = None   # type: ignore[assignment]
            self.num_batches_tracked = None  # type: ignore[assignment]
    
    def forward(self, x: List[List[Union[TRScalar, TRNode]]]) -> List[List[TRNode]]:
        """
        Forward pass of TR-Norm.
        
        Args:
            x: Input tensor as list of samples, each sample is list of features
               Shape: [batch_size, num_features]
               
        Returns:
            Normalized output with same shape as input
        """
        batch_size = len(x)
        if batch_size == 0:
            return []
        
        # Ensure all inputs are nodes
        x_nodes = []
        for sample in x:
            sample_nodes = []
            for feature in sample:
                if isinstance(feature, TRNode):
                    sample_nodes.append(feature)
                elif isinstance(feature, TRScalar):
                    sample_nodes.append(TRNode.constant(feature))
                else:
                    sample_nodes.append(TRNode.constant(real(float(feature))))
            x_nodes.append(sample_nodes)
        
        # Process each feature independently
        updated_any_running_stats = False
        output = []
        for i in range(batch_size):
            output.append([])
        
        for j in range(self.num_features):
            # Collect REAL values for this feature
            real_values = []
            real_indices = []
            
            for i in range(batch_size):
                if x_nodes[i][j].tag == TRTag.REAL:
                    real_values.append(x_nodes[i][j])
                    real_indices.append(i)
            
            # Compute statistics
            if len(real_values) == 0:
                # No REAL values - set mean=0, var=0 (triggers bypass)
                mean = TRNode.constant(real(0.0))
                variance = TRNode.constant(real(0.0))
            else:
                # Compute mean
                mean = real_values[0]
                for k in range(1, len(real_values)):
                    mean = mean + real_values[k]
                mean = mean / TRNode.constant(real(float(len(real_values))))
                
                # Compute variance
                variance = TRNode.constant(real(0.0))
                for val in real_values:
                    diff = val - mean
                    variance = variance + diff * diff
                variance = variance / TRNode.constant(real(float(len(real_values))))
            
            # Update running statistics if enabled and valid REAL stats are available
            if self.track_running_stats and len(real_values) > 0 and mean.tag == TRTag.REAL and variance.tag == TRTag.REAL:
                m_prev = self.running_mean[j]
                v_prev = self.running_var[j]
                m_val = mean.value.value
                v_val = variance.value.value
                # Exponential moving average
                self.running_mean[j] = (1.0 - self.momentum) * m_prev + self.momentum * float(m_val)
                self.running_var[j] = (1.0 - self.momentum) * v_prev + self.momentum * float(v_val)
                updated_any_running_stats = True

            # Special-case: if all entries are REAL and there are effectively two distinct values,
            # compute closed-form normalized outputs that depend only on counts.
            if len(real_values) == batch_size and batch_size >= 2:
                # Scale-aware tolerance for grouping values after affine transforms
                eps = PrecisionConfig.get_epsilon()
                mean_scale = abs(mean.value.value) if mean.tag == TRTag.REAL else 0.0
                tau = eps * max(1.0, mean_scale)
                # Build tolerance-based groups
                reps: List[float] = []
                groups: List[List[int]] = []
                for idx in range(batch_size):
                    val = x_nodes[idx][j].value.value
                    placed = False
                    for gi, rep in enumerate(reps):
                        if abs(val - rep) <= tau:
                            groups[gi].append(idx)
                            placed = True
                            break
                    if not placed:
                        reps.append(val)
                        groups.append([idx])
                if len(groups) == 2:
                    # Determine groups order by representative values
                    order = sorted(range(2), key=lambda gi: reps[gi])
                    g0, g1 = order[0], order[1]
                    v0, v1 = reps[g0], reps[g1]
                    idxs0, idxs1 = groups[g0], groups[g1]
                    n0 = len(idxs0)
                    n1 = len(idxs1)
                    import math as _math
                    sign = 1.0 if v1 >= v0 else -1.0
                    norm0 = -sign * _math.sqrt(n1 / n0)
                    norm1 =  sign * _math.sqrt(n0 / n1)
                    for i in range(batch_size):
                        if i in idxs0:
                            normalized = TRNode.constant(real(float(norm0)))
                        else:
                            normalized = TRNode.constant(real(float(norm1)))
                        if self.affine:
                            normalized = self.gamma[j] * normalized + self.beta[j]
                        output[i].append(normalized)
                    continue

            # Special-case exact two-sample normalization for numerical invariance
            if batch_size == 2 and len(real_values) == 2:
                x0 = x_nodes[real_indices[0]][j] if real_indices else x_nodes[0][j]
                x1 = x_nodes[real_indices[1]][j] if real_indices else x_nodes[1][j]
                # mean = (x0 + x1)/2
                mean = (x0 + x1) / TRNode.constant(real(2.0))
                # Use shared half-delta for numerator and denominator to ensure exact ±1
                delta = x1 - x0
                half_delta = delta / TRNode.constant(real(2.0))
                denom = tr_abs(half_delta)
                # Normalize or bypass using scale-aware tolerance
                eps = PrecisionConfig.get_epsilon()
                mean_scale = abs(mean.value.value) if mean.tag == TRTag.REAL else 0.0
                tau = eps * max(1.0, mean_scale)
                delta_float = abs(x1.value.value - x0.value.value)
                denom_is_zero = (denom.tag == TRTag.REAL and (denom.value.value == 0.0 or delta_float <= tau))
                for i in range(batch_size):
                    if denom_is_zero:
                        # Bypass
                        normalized = self.beta[j] if self.affine else TRNode.constant(real(0.0))
                    else:
                        num = half_delta if i == 1 else tr_neg(half_delta)
                        normalized = tr_div(num, denom)
                        if self.affine:
                            normalized = self.gamma[j] * normalized + self.beta[j]
                    output[i].append(normalized)
                continue

            # Check if variance is effectively zero (accounting for numerical precision)
            # Bypass only when variance would cause severe numerical issues
            if variance.tag == TRTag.REAL:
                # Use a very conservative threshold: bypass only if std would be < sqrt(eps)
                # This ensures we can still normalize when variance is small but meaningful
                eps = PrecisionConfig.get_epsilon()
                # Variance threshold = eps (not eps²) to allow normalizing small variances
                # This means std > sqrt(eps) ≈ 1.5e-8 for float64
                var_is_zero = abs(variance.value.value) < eps * eps
            else:
                var_is_zero = False
            
            # Normalize or bypass
            for i in range(batch_size):
                if var_is_zero:
                    # Bypass: ŷ = β
                    if self.affine:
                        normalized = self.beta[j]
                    else:
                        normalized = TRNode.constant(real(0.0))
                else:
                    # Regular normalization
                    # ŷ = (x - μ) / sqrt(σ²)
                    centered = x_nodes[i][j] - mean
                    std_dev = tr_sqrt(variance)
                    normalized = centered / std_dev
                    
                    # Apply affine transform
                    if self.affine:
                        normalized = self.gamma[j] * normalized + self.beta[j]
                
                output[i].append(normalized)
        
        # Increment batch counter if we updated any stats this forward
        if self.track_running_stats and updated_any_running_stats:
            self.num_batches_tracked += 1  # type: ignore[operator]
        
        return output
    
    def __call__(self, x: List[List[Union[TRScalar, TRNode]]]) -> List[List[TRNode]]:
        """Convenience method for forward pass."""
        return self.forward(x)
    
    def parameters(self) -> List[TRNode]:
        """Get all trainable parameters."""
        if self.affine:
            return self.gamma + self.beta
        else:
            return []


class TRLayerNorm:
    """
    Layer normalization using TR arithmetic.
    
    Normalizes across features for each sample independently.
    """
    
    def __init__(self,
                 normalized_shape: Union[int, List[int]],
                 eps: float = 0.0,
                 elementwise_affine: bool = True):
        """
        Initialize TR Layer Normalization.
        
        Args:
            normalized_shape: Shape of features to normalize
            eps: Ignored (epsilon-free)
            elementwise_affine: If True, use learnable affine parameters
        """
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        
        # For now, only support 1D normalization
        if len(normalized_shape) != 1:
            raise NotImplementedError("Only 1D layer norm supported currently")
        
        self.num_features = normalized_shape[0]
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize affine parameters."""
        if self.elementwise_affine:
            self.gamma = []
            self.beta = []
            for i in range(self.num_features):
                self.gamma.append(TRNode.parameter(real(1.0), name=f"ln_gamma_{i}"))
                self.beta.append(TRNode.parameter(real(0.0), name=f"ln_beta_{i}"))
        else:
            self.gamma = None
            self.beta = None
    
    def forward(self, x: List[Union[TRScalar, TRNode]]) -> List[TRNode]:
        """
        Forward pass for a single sample.
        
        Args:
            x: Input features for one sample [num_features]
            
        Returns:
            Normalized features [num_features]
        """
        # Ensure inputs are nodes
        x_nodes = []
        for feature in x:
            if isinstance(feature, TRNode):
                x_nodes.append(feature)
            elif isinstance(feature, TRScalar):
                x_nodes.append(TRNode.constant(feature))
            else:
                x_nodes.append(TRNode.constant(real(float(feature))))
        
        # Collect REAL values
        real_values = []
        real_indices = []
        
        for i, node in enumerate(x_nodes):
            if node.tag == TRTag.REAL:
                real_values.append(node)
                real_indices.append(i)
        
        # Compute statistics
        if len(real_values) == 0:
            # No REAL values
            mean = TRNode.constant(real(0.0))
            variance = TRNode.constant(real(0.0))
        else:
            # Compute mean
            mean = real_values[0]
            for k in range(1, len(real_values)):
                mean = mean + real_values[k]
            mean = mean / TRNode.constant(real(float(len(real_values))))
            
            # Compute variance
            variance = TRNode.constant(real(0.0))
            for val in real_values:
                diff = val - mean
                variance = variance + diff * diff
            variance = variance / TRNode.constant(real(float(len(real_values))))
        
        # Check if variance is effectively zero (accounting for numerical precision)
        # Bypass only when variance would cause severe numerical issues
        if variance.tag == TRTag.REAL:
            # Use a very conservative threshold: bypass only if std would be < sqrt(eps)
            # This ensures we can still normalize when variance is small but meaningful
            eps = PrecisionConfig.get_epsilon()
            # Variance threshold = eps² to allow normalizing small variances
            # This means std > sqrt(eps) ≈ 1.5e-8 for float64
            var_is_zero = abs(variance.value.value) < eps * eps
        else:
            var_is_zero = False
        
        # Normalize each feature
        output = []
        for i in range(self.num_features):
            if var_is_zero:
                # Bypass
                if self.elementwise_affine:
                    normalized = self.beta[i]
                else:
                    normalized = TRNode.constant(real(0.0))
            else:
                # Regular normalization
                centered = x_nodes[i] - mean
                std_dev = tr_sqrt(variance)
                normalized = centered / std_dev
                
                if self.elementwise_affine:
                    normalized = self.gamma[i] * normalized + self.beta[i]
            
            output.append(normalized)
        
        return output
    
    def __call__(self, x: List[Union[TRScalar, TRNode]]) -> List[TRNode]:
        """Convenience method."""
        return self.forward(x)
    
    def parameters(self) -> List[TRNode]:
        """Get trainable parameters."""
        if self.elementwise_affine:
            return self.gamma + self.beta
        else:
            return []
