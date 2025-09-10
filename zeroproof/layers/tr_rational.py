"""
TR-Rational layer implementation.

A rational layer computes y = P_θ(x) / Q_φ(x) where P and Q are polynomials.
The layer is total under transreal arithmetic and uses the Mask-REAL rule
for stable gradients near singularities.
"""

from typing import Optional, Union, Tuple, List, Any
import math

from ..core import TRScalar, TRTag, real, pinf, ninf, phi
from ..autodiff import TRNode, gradient_tape
from .basis import Basis, MonomialBasis


class TRRational:
    """
    Transreal rational layer: y = P_θ(x) / Q_φ(x).
    
    Features:
    - Total operations (never throws exceptions)
    - Stable AD via Mask-REAL rule
    - Identifiable parameterization (leading-1 in Q)
    - Optional regularization on denominator coefficients
    - Support for adaptive loss policies
    """
    
    def __init__(self,
                 d_p: int,
                 d_q: int,
                 basis: Optional[Basis] = None,
                 shared_Q: bool = False,
                 lambda_rej: float = 0.0,
                 alpha_phi: float = 1e-3,
                 l1_projection: Optional[float] = None,
                 adaptive_loss_policy=None,
                 projection_index: Optional[int] = None):
        """
        Initialize TR-Rational layer.
        
        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q (must be ≥ 1)
            basis: Basis functions to use (default: MonomialBasis)
            shared_Q: If True, share Q across multiple outputs (not implemented)
            lambda_rej: Penalty for non-REAL outputs in loss (ignored if adaptive_loss_policy provided)
            alpha_phi: L2 regularization coefficient for φ (denominator)
            l1_projection: Optional L1 bound for φ to ensure stability
            adaptive_loss_policy: Optional adaptive loss policy for automatic lambda adjustment
        """
        if d_q < 1:
            raise ValueError("Denominator degree must be at least 1")
        
        self.d_p = d_p
        self.d_q = d_q
        self.basis = basis or MonomialBasis()
        self.shared_Q = shared_Q
        self.lambda_rej = lambda_rej
        self.alpha_phi = alpha_phi
        self.l1_projection = l1_projection
        self.adaptive_loss_policy = adaptive_loss_policy
        # If set, allows selecting a component from vector inputs
        self.projection_index = projection_index
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize θ and φ parameters."""
        # Numerator coefficients: θ_0, θ_1, ..., θ_{d_p}
        self.theta = []
        for i in range(self.d_p + 1):
            # Initialize near zero with small random values
            val = 0.1 * (2 * math.sqrt(3) * (i % 2 - 0.5)) / math.sqrt(self.d_p + 1)
            self.theta.append(TRNode.parameter(real(val), name=f"theta_{i}"))
        
        # Denominator coefficients: φ_1, ..., φ_{d_q}
        # Note: φ_0 is fixed at 1 for identifiability
        self.phi = []
        for i in range(1, self.d_q + 1):
            # Initialize small to start near Q(x) ≈ 1
            val = 0.01 * (2 * math.sqrt(3) * (i % 2 - 0.5)) / math.sqrt(self.d_q)
            self.phi.append(TRNode.parameter(real(val), name=f"phi_{i}"))
    
    def forward(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag]:
        """
        Forward pass computing y = P(x) / Q(x).
        
        Args:
            x: Input value (scalar)
            
        Returns:
            Tuple of (output_node, output_tag)
        """
        # Handle vector-like input with optional projection
        if not isinstance(x, (TRScalar, TRNode)):
            # Detect list/tuple/numpy array
            is_sequence_like = False
            try:
                # numpy scalars raise TypeError on len(); sequences return >=1
                _ = len(x)  # type: ignore
                is_sequence_like = True
            except Exception:
                is_sequence_like = False
            if is_sequence_like:
                if self.projection_index is not None:
                    try:
                        x = x[self.projection_index]  # type: ignore[index]
                    except Exception as ex:
                        raise TypeError(
                            f"Failed to apply projection_index={self.projection_index} to input"
                        ) from ex
                else:
                    # Fallback: if it's a 1-element vector, use the first element
                    try:
                        if len(x) == 1:  # type: ignore[arg-type]
                            x = x[0]  # type: ignore[index]
                        else:
                            raise TypeError(
                                "TRRational.forward expects a scalar input. "
                                "Use forward_batch for lists/ndarrays, or set projection_index to select a component."
                            )
                    except Exception:
                        raise TypeError(
                            "TRRational.forward expects a scalar input. "
                            "Use forward_batch for lists/ndarrays, or set projection_index to select a component."
                        )

        # Ensure x is a node
        if isinstance(x, TRScalar):
            x = TRNode.constant(x)
        elif not isinstance(x, TRNode):
            x = TRNode.constant(real(float(x)))
        
        # Evaluate basis functions
        psi = self.basis(x, max(self.d_p, self.d_q))
        
        # Compute P(x) = Σ θ_k ψ_k(x)
        P = self.theta[0] * psi[0]
        for k in range(1, self.d_p + 1):
            if k < len(psi):
                P = P + self.theta[k] * psi[k]
        
        # Compute Q(x) = 1 + Σ φ_k ψ_k(x)
        Q = TRNode.constant(real(1.0))  # Leading 1
        for k in range(1, self.d_q + 1):
            if k < len(psi):
                Q = Q + self.phi[k-1] * psi[k]
        # Track last |Q| for diagnostics and pole interfaces
        try:
            self._last_Q_abs = abs(Q.value.value) if Q.tag == TRTag.REAL else 0.0
        except Exception:
            self._last_Q_abs = None
        
        # Apply L1 projection if specified
        if self.l1_projection is not None:
            self._project_phi_l1()
        
        # Compute y = P / Q with TR semantics
        y = P / Q

        # Attach contextual metadata for downstream training utilities (no ε, exact pole tooling)
        try:
            if hasattr(y, '_grad_info') and y._grad_info is not None:
                # Remember input x for this prediction
                x_val = None
                if isinstance(x, TRNode) and x.value.tag == TRTag.REAL:
                    x_val = float(x.value.value)
                elif isinstance(x, TRScalar) and x.tag == TRTag.REAL:
                    x_val = float(x.value)
                y._grad_info.extra_data['input_x'] = x_val
                # Provide references needed for exact Q=0 enforcement (projection)
                y._grad_info.extra_data['tr_rational_phi'] = self.phi
                y._grad_info.extra_data['tr_rational_basis'] = self.basis
                y._grad_info.extra_data['tr_rational_dq'] = self.d_q
        except Exception:
            pass

        # Strict TR semantics (no ε): singular tags arise only from exact division rules
        
        return y, y.tag
    
    def __call__(self, x: Union[TRScalar, TRNode, Any]) -> Any:
        """
        Convenience call: accept scalar or batch-like inputs.

        - For scalar inputs, returns a TRNode.
        - For list/ndarray/torch.Tensor inputs, returns a List[TRNode].
        """
        # Detect batch-like inputs and delegate to forward_batch
        if not isinstance(x, (TRScalar, TRNode)):
            try:
                _ = len(x)  # type: ignore[arg-type]
                return self.forward_batch(x)
            except Exception:
                pass
        y, _ = self.forward(x)
        return y

    def forward_with_tag(self, x: Union[TRScalar, TRNode]) -> Tuple[TRNode, TRTag]:
        """Explicit helper returning (y, tag); alias to forward for clarity."""
        return self.forward(x)

    def forward_batch(self, xs: Any) -> List[TRNode]:
        """
        Batched forward pass over list/ndarray inputs.

        Args:
            xs: Iterable of inputs (list/tuple/ndarray). Each element may be a scalar
                or a vector; when vector, projection_index must be set to select a component.

        Returns:
            List of output nodes corresponding to each input element.
        """
        # Quick validation that xs is iterable (lists, tuples, numpy arrays)
        try:
            iterator = iter(xs)
        except Exception as ex:
            raise TypeError("forward_batch expects a list/tuple/ndarray of inputs") from ex

        outputs: List[TRNode] = []
        for x in iterator:
            # If element is vector-like and projection_index is provided, apply it
            is_sequence_like = False
            if not isinstance(x, (TRScalar, TRNode)):
                try:
                    _ = len(x)  # type: ignore
                    is_sequence_like = True
                except Exception:
                    is_sequence_like = False
            if is_sequence_like:
                if self.projection_index is not None:
                    x = x[self.projection_index]  # type: ignore[index]
                else:
                    # Fallback to first element for 1-element vectors
                    try:
                        if len(x) == 1:  # type: ignore[arg-type]
                            x = x[0]  # type: ignore[index]
                        else:
                            raise TypeError(
                                "Elements of xs are vector-like; set projection_index to select a component."
                            )
                    except Exception:
                        raise TypeError(
                            "Elements of xs are vector-like; set projection_index to select a component."
                        )

            y, _ = self.forward(x)
            outputs.append(y)

        return outputs
    
    def _project_phi_l1(self):
        """
        Project φ coefficients to L1 ball if needed.
        
        This ensures ||φ||₁ ≤ B where B is the l1_projection bound.
        When ||φ||₁ > B, we scale all coefficients uniformly to satisfy
        the constraint, which helps maintain Q(x) away from zero.
        """
        if self.l1_projection is None or self.l1_projection <= 0:
            return
        
        # Compute L1 norm of φ
        l1_norm = 0.0
        for phi_k in self.phi:
            if phi_k.value.tag == TRTag.REAL:
                l1_norm += abs(phi_k.value.value)
        
        # Project if needed
        if l1_norm > self.l1_projection:
            # Scale all coefficients to project onto L1 ball
            scale = self.l1_projection / l1_norm
            
            for phi_k in self.phi:
                if phi_k.value.tag == TRTag.REAL:
                    # Update the parameter value directly
                    scaled_value = phi_k.value.value * scale
                    phi_k._value = real(scaled_value)
                    
                    # Also scale the gradient if it exists (for consistency)
                    if phi_k.gradient is not None and phi_k.gradient.tag == TRTag.REAL:
                        scaled_grad_value = phi_k.gradient.value * scale
                        phi_k._gradient = TRNode.constant(scaled_grad_value)
    
    def regularization_loss(self) -> TRNode:
        """
        Compute L2 regularization loss on denominator coefficients.
        
        Returns:
            Regularization loss α/2 * ||φ||²
        """
        reg = TRNode.constant(real(0.0))
        
        for phi_k in self.phi:
            reg = reg + phi_k * phi_k
        
        alpha_half = TRNode.constant(real(self.alpha_phi / 2.0))
        return alpha_half * reg
    
    def compute_q_min(self, x_batch: List[Union[TRScalar, TRNode]]) -> float:
        """
        Compute minimum |Q(x)| over a batch.
        
        Args:
            x_batch: List of input values
            
        Returns:
            min |Q(x_i)| over the batch
        """
        q_min = float('inf')
        
        for x in x_batch:
            # Evaluate Q(x)
            if isinstance(x, TRScalar):
                x_node = TRNode.constant(x)
            else:
                x_node = x
            
            psi = self.basis(x_node, self.d_q)
            Q = TRNode.constant(real(1.0))
            for k in range(1, self.d_q + 1):
                if k < len(psi):
                    Q = Q + self.phi[k-1] * psi[k]
            
            # Check if Q is REAL and update minimum
            if Q.tag == TRTag.REAL:
                q_abs = abs(Q.value.value)
                q_min = min(q_min, q_abs)
        
        return q_min
    
    def parameters(self) -> List[TRNode]:
        """Get all trainable parameters."""
        return self.theta + self.phi
    
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return len(self.theta) + len(self.phi)

    # Convenience utilities used by integration tests
    def get_q_values(self, xs: Any) -> List[float]:
        """
        Compute |Q(x)| for a batch of inputs.

        Args:
            xs: Iterable of scalar inputs (list/tuple/ndarray/torch.Tensor)

        Returns:
            List of absolute Q values as Python floats
        """
        # Convert potential torch tensor to a Python list
        try:
            if hasattr(xs, 'tolist'):
                xs_list = xs.tolist()
            else:
                xs_list = list(xs)
        except TypeError:
            xs_list = [xs]

        q_abs_values: List[float] = []
        for x in xs_list:
            # If element is vector-like and projection_index is provided, apply it
            is_sequence_like = False
            if not isinstance(x, (TRScalar, TRNode)):
                try:
                    _ = len(x)  # type: ignore
                    is_sequence_like = True
                except Exception:
                    is_sequence_like = False
            if is_sequence_like:
                if self.projection_index is not None:
                    x = x[self.projection_index]  # type: ignore[index]
                else:
                    # Fallback: take first element
                    x = x[0]  # type: ignore[index]

            # Ensure TRNode
            if isinstance(x, TRScalar):
                x_node = TRNode.constant(x)
            elif isinstance(x, TRNode):
                x_node = x
            else:
                try:
                    x_node = TRNode.constant(real(float(x)))
                except Exception:
                    # If conversion fails, skip
                    q_abs_values.append(float('inf'))
                    continue

            # Evaluate basis up to denominator degree
            psi = self.basis(x_node, self.d_q)
            Q = TRNode.constant(real(1.0))
            for k in range(1, self.d_q + 1):
                if k < len(psi) and k <= len(self.phi):
                    Q = Q + self.phi[k-1] * psi[k]

            if Q.tag == TRTag.REAL:
                q_abs_values.append(abs(Q.value.value))
            else:
                # Non-REAL Q treated as 0 distance (at pole)
                q_abs_values.append(0.0)

        return q_abs_values


class TRRationalMulti:
    """
    Multi-output TR-Rational layer.
    
    Can share denominator Q across outputs for parameter efficiency.
    """
    
    def __init__(self,
                 d_p: int,
                 d_q: int,
                 n_outputs: int,
                 basis: Optional[Basis] = None,
                 shared_Q: bool = True,
                 lambda_rej: float = 0.0,
                 alpha_phi: float = 1e-3):
        """
        Initialize multi-output rational layer.
        
        Args:
            d_p: Degree of numerator polynomials
            d_q: Degree of denominator polynomial(s)
            n_outputs: Number of outputs
            basis: Basis functions to use
            shared_Q: If True, share denominator across outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominators
        """
        self.n_outputs = n_outputs
        self.shared_Q = shared_Q
        
        if shared_Q:
            # One shared denominator, multiple numerators
            self.layers = []
            shared_layer = TRRational(d_p, d_q, basis, True, lambda_rej, alpha_phi)
            
            # Create layers sharing the denominator parameters
            for i in range(n_outputs):
                if i == 0:
                    self.layers.append(shared_layer)
                else:
                    # Create new layer but share phi parameters
                    layer = TRRational(d_p, d_q, basis, True, lambda_rej, alpha_phi)
                    layer.phi = shared_layer.phi  # Share denominator
                    self.layers.append(layer)
        else:
            # Independent rational functions
            self.layers = [
                TRRational(d_p, d_q, basis, False, lambda_rej, alpha_phi)
                for _ in range(n_outputs)
            ]
    
    def forward(self, x: Union[TRScalar, TRNode]) -> List[Tuple[TRNode, TRTag]]:
        """
        Forward pass for all outputs.
        
        Args:
            x: Input value
            
        Returns:
            List of (output_node, output_tag) tuples
        """
        return [layer.forward(x) for layer in self.layers]
    
    def __call__(self, x: Union[TRScalar, TRNode]) -> List[TRNode]:
        """Convenience method returning just output nodes."""
        return [y for y, _ in self.forward(x)]
    
    def regularization_loss(self) -> TRNode:
        """Compute total regularization loss."""
        if self.shared_Q:
            # Only regularize once for shared denominator
            return self.layers[0].regularization_loss()
        else:
            # Sum regularization across all layers
            total_reg = TRNode.constant(real(0.0))
            for layer in self.layers:
                total_reg = total_reg + layer.regularization_loss()
            return total_reg
    
    def parameters(self) -> List[TRNode]:
        """Get all unique trainable parameters."""
        if self.shared_Q:
            # Collect unique parameters (avoiding duplicates)
            params = []
            params.extend(self.layers[0].phi)  # Shared denominator
            for layer in self.layers:
                params.extend(layer.theta)  # Individual numerators
            return params
        else:
            # All parameters are independent
            params = []
            for layer in self.layers:
                params.extend(layer.parameters())
            return params
