"""
TR-Rational layer implementation.

A rational layer computes y = P_θ(x) / Q_φ(x) where P and Q are polynomials.
The layer is total under transreal arithmetic and uses the Mask-REAL rule
for stable gradients near singularities.
"""

from typing import Optional, Union, Tuple, List
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
                 adaptive_loss_policy=None):
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
        
        # Apply L1 projection if specified
        if self.l1_projection is not None:
            self._project_phi_l1()
        
        # Compute y = P / Q with TR semantics
        y = P / Q
        
        return y, y.tag
    
    def __call__(self, x: Union[TRScalar, TRNode]) -> TRNode:
        """Convenience method returning just the output node."""
        y, _ = self.forward(x)
        return y
    
    def _project_phi_l1(self):
        """Project φ coefficients to L1 ball if needed."""
        if self.l1_projection is None:
            return
        
        # Compute L1 norm of φ
        l1_norm = 0.0
        for phi_k in self.phi:
            l1_norm += abs(phi_k.value.value)
        
        # Project if needed
        if l1_norm > self.l1_projection:
            scale = self.l1_projection / l1_norm
            for phi_k in self.phi:
                # This is a simplified projection - in practice we'd update the actual values
                # For now, just track that projection would be needed
                pass
    
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
