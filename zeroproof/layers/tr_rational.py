"""
TR-Rational layer implementation.

A rational layer computes y = P_θ(x) / Q_φ(x) where P and Q are polynomials.
The layer is total under transreal arithmetic and uses the Mask-REAL rule
for stable gradients near singularities.
"""

import math
from typing import Any, List, Optional, Tuple, Union

from ..autodiff import TRNode, gradient_tape
from ..core import TRScalar, TRTag, ninf, phi, pinf, real
from ..policy import TRPolicyConfig, classify_tag_with_policy
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

    def __init__(
        self,
        d_p: Optional[int] = None,
        d_q: Optional[int] = None,
        basis: Optional[Basis] = None,
        shared_Q: bool = False,
        lambda_rej: float = 0.0,
        alpha_phi: float = 1e-3,
        l1_projection: Optional[float] = None,
        adaptive_loss_policy=None,
        projection_index: Optional[int] = None,
        # Backward-compatibility aliases
        degree_p: Optional[int] = None,
        degree_q: Optional[int] = None,
        lambda_reg: Optional[float] = None,
        # Coprime surrogate regularizer
        enable_coprime_regularizer: bool = False,
        lambda_coprime: float = 0.0,
        # SoA/bit-mask fast path toggles (value-only evaluation)
        enable_soa_value_only: bool = True,
    ):
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
            degree_p: Alias for d_p (for backward compatibility)
            degree_q: Alias for d_q (for backward compatibility)
            lambda_reg: Alias for alpha_phi (for backward compatibility)
        """
        # Resolve backward-compatible aliases
        if d_p is None and degree_p is not None:
            d_p = degree_p
        if d_q is None and degree_q is not None:
            d_q = degree_q
        if d_p is None or d_q is None:
            raise TypeError("TRRational requires d_p and d_q (or degree_p and degree_q)")

        if d_q < 1:
            raise ValueError("Denominator degree must be at least 1")

        self.d_p = int(d_p)
        self.d_q = int(d_q)
        self.basis = basis or MonomialBasis()
        self.shared_Q = shared_Q
        self.lambda_rej = lambda_rej
        # Prefer explicit alpha_phi; else accept lambda_reg alias if provided
        self.alpha_phi = float(alpha_phi)
        if lambda_reg is not None:
            self.alpha_phi = float(lambda_reg)
        self.l1_projection = l1_projection
        self.adaptive_loss_policy = adaptive_loss_policy
        # If set, allows selecting a component from vector inputs
        self.projection_index = projection_index
        # Coprime surrogate config
        self.enable_coprime_regularizer = enable_coprime_regularizer
        self.lambda_coprime = float(lambda_coprime or 0.0)
        self.enable_soa_value_only = bool(enable_soa_value_only)
        # Diagnostics: last bit-mask for tags collected in fast path
        self._last_tag_mask: int = 0

        # Initialize parameters
        self._initialize_parameters()

    # -----------------------------
    # SoA value-only fast path (no autodiff graph)
    # -----------------------------
    @staticmethod
    def _neumaier_sum(vals: List[float]) -> float:
        s = 0.0
        c = 0.0
        for x in vals:
            t = s + x
            if abs(s) >= abs(x):
                c += (s - t) + x
            else:
                c += (x - t) + s
            s = t
        return s + c

    def value_only(self, x: Union[TRScalar, float, int]) -> TRScalar:
        """
        Fast value-only evaluation using SoA layout and a 1-bit tag mask.

        Returns TRScalar y = P(x)/Q(x). No computation graph is built.
        Only used for inference/micro-benchmarks; gradients won't flow.
        """
        if not self.enable_soa_value_only:
            # Fallback to regular forward and extract value
            if not isinstance(x, TRScalar):
                x = real(float(x))
            y_node, _ = self.forward(x)
            return y_node.value

        # Assemble SoA for parameters and basis values
        # Tags bit-pack: 1 bit per element, 1 indicates non-REAL
        tag_mask = 0

        # Coefficients arrays
        theta_vals: List[float] = []
        for i, th in enumerate(self.theta):
            if th.value.tag != TRTag.REAL:
                tag_mask |= 1 << i
            theta_vals.append(float(th.value.value) if th.value.tag == TRTag.REAL else 0.0)

        phi_vals: List[float] = []
        base_phi_bit = len(theta_vals)
        for j, ph in enumerate(self.phi):
            if ph.value.tag != TRTag.REAL:
                tag_mask |= 1 << (base_phi_bit + j)
            phi_vals.append(float(ph.value.value) if ph.value.tag == TRTag.REAL else 0.0)

        # Basis values as floats
        if isinstance(x, TRScalar):
            x_scalar = x
        else:
            x_scalar = real(float(x))
        psi = self.basis(x_scalar, max(self.d_p, self.d_q))
        psi_vals: List[float] = []
        base_psi_bit = base_phi_bit + len(phi_vals)
        for k, psi_k in enumerate(psi):
            tag = psi_k.tag if hasattr(psi_k, "tag") else TRTag.REAL
            if tag != TRTag.REAL:
                tag_mask |= 1 << (base_psi_bit + k)
            v = float(psi_k.value) if hasattr(psi_k, "value") else float(psi_k)
            psi_vals.append(v)

        # If any non-REAL tag encountered, fallback to regular slow path
        if tag_mask != 0:
            self._last_tag_mask = tag_mask
            if not isinstance(x, TRScalar):
                x = real(float(x))
            y_node, _ = self.forward(x)
            return y_node.value

        # Compute P and Q via compensated sums
        # P = sum_{k=0..d_p} theta_k * psi_k
        prods_p: List[float] = []
        for k in range(self.d_p + 1):
            if k < len(psi_vals):
                prods_p.append(theta_vals[k] * psi_vals[k])
        P = self._neumaier_sum(prods_p)

        # Q = 1 + sum_{k=1..d_q} phi_{k-1} * psi_k
        prods_q: List[float] = [1.0]
        for k in range(1, self.d_q + 1):
            if k < len(psi_vals) and (k - 1) < len(phi_vals):
                prods_q.append(phi_vals[k - 1] * psi_vals[k])
        Q = self._neumaier_sum(prods_q)

        # Form y = P / Q with TR semantics for division
        from ..core import tr_div

        y = tr_div(real(P), real(Q))
        # Save mask
        self._last_tag_mask = tag_mask
        return y

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

        Contracts and semantics:
        - Values follow TR algebra exactly; the returned `TRNode` `y` is computed
          under total operations (no exceptions).
        - The returned `TRTag` reflects either the TR tag of `y` or, when a
          `TRPolicy` is active, the policy-based classification using guard bands
          and hysteresis (|Q| vs τ_Q_on/off; |P| vs τ_P_on/off) to decide between
          REAL/±INF/PHI deterministically near poles.

        Args:
            x: Input scalar (TRScalar or TRNode). Vector-like inputs should be
               passed via `forward_batch` or with `projection_index` set to pick
               a component.

        Returns:
            (y, tag) where y is a TRNode and tag is TRTag.
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

        # Compute P(x) = Σ θ_k ψ_k(x) with optional deterministic pairwise reduction
        def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
            if not nodes:
                return TRNode.constant(real(0.0))
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = _pairwise_sum(nodes[:mid])
            right = _pairwise_sum(nodes[mid:])
            return left + right

        P_terms: List[TRNode] = []
        for k in range(0, self.d_p + 1):
            if k < len(psi):
                P_terms.append(self.theta[k] * psi[k])
        # Default to sequential sum; switch to pairwise if deterministic reductions requested
        use_pairwise = False
        try:
            from ..policy import TRPolicyConfig

            pol = TRPolicyConfig.get_policy()
            use_pairwise = bool(pol and pol.deterministic_reduction)
        except Exception:
            use_pairwise = False
        if use_pairwise:
            P = _pairwise_sum(P_terms)
        else:
            P = P_terms[0]
            for term in P_terms[1:]:
                P = P + term

        # Compute Q(x) = 1 + Σ φ_k ψ_k(x) with optional deterministic pairwise reduction
        Q_terms: List[TRNode] = [TRNode.constant(real(1.0))]
        for k in range(1, self.d_q + 1):
            if k < len(psi):
                Q_terms.append(self.phi[k - 1] * psi[k])
        if use_pairwise:
            Q = _pairwise_sum(Q_terms)
        else:
            Q = Q_terms[0]
            for term in Q_terms[1:]:
                Q = Q + term
        # Track last |Q| for diagnostics and pole interfaces
        try:
            self._last_Q_abs = abs(Q.value.value) if Q.tag == TRTag.REAL else 0.0
        except Exception:
            self._last_Q_abs = None

        # Update hybrid controller Q-tracking for quantile stats
        try:
            if Q.tag == TRTag.REAL:
                from ..autodiff.hybrid_gradient import HybridGradientContext

                HybridGradientContext.update_q_value(abs(Q.value.value))
        except Exception:
            pass

        # Apply L1 projection if specified
        if self.l1_projection is not None:
            self._project_phi_l1()

        # Compute y = P / Q with TR semantics
        y = P / Q

        # Attach contextual metadata for downstream training utilities (no ε, exact pole tooling)
        try:
            if hasattr(y, "_grad_info") and y._grad_info is not None:
                # Remember input x for this prediction
                x_val = None
                if isinstance(x, TRNode) and x.value.tag == TRTag.REAL:
                    x_val = float(x.value.value)
                elif isinstance(x, TRScalar) and x.tag == TRTag.REAL:
                    x_val = float(x.value)
                y._grad_info.extra_data["input_x"] = x_val
                # Provide references needed for exact Q=0 enforcement (projection)
                y._grad_info.extra_data["tr_rational_phi"] = self.phi
                y._grad_info.extra_data["tr_rational_basis"] = self.basis
                y._grad_info.extra_data["tr_rational_dq"] = self.d_q
        except Exception:
            pass

        # Determine output tag
        tag = y.tag
        try:
            policy = TRPolicyConfig.get_policy()
            if policy is not None:
                prev = getattr(self, "_last_policy_tag", None)
                # Use policy-based classification when available
                tag = classify_tag_with_policy(
                    policy, P.value, Q.value, y.tag, prev_policy_tag=prev
                )
                self._last_policy_tag = tag
        except Exception:
            # Fallback to TR tag if policy classification fails
            tag = y.tag

        # Strict TR semantics for values; tags may use policy classification
        return y, tag

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
            Regularization loss α/2 * ||φ||² + (optional) λ_coprime * surrogate(P, Q)
        """

        # Deterministic reduction for L2(φ): sum_k φ_k^2
        def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
            if not nodes:
                return TRNode.constant(real(0.0))
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = _pairwise_sum(nodes[:mid])
            right = _pairwise_sum(nodes[mid:])
            return left + right

        phi_sq_terms: List[TRNode] = [phi_k * phi_k for phi_k in self.phi]
        use_pairwise = False
        try:
            from ..policy import TRPolicyConfig

            pol = TRPolicyConfig.get_policy()
            use_pairwise = bool(pol and pol.deterministic_reduction)
        except Exception:
            use_pairwise = False
        reg = (
            _pairwise_sum(phi_sq_terms)
            if use_pairwise and phi_sq_terms
            else (phi_sq_terms[0] if phi_sq_terms else TRNode.constant(real(0.0)))
        )
        if not use_pairwise and phi_sq_terms:
            reg = phi_sq_terms[0]
            for t in phi_sq_terms[1:]:
                reg = reg + t

        alpha_half = TRNode.constant(real(self.alpha_phi / 2.0))
        total = alpha_half * reg

        # Optional coprime surrogate regularizer: penalize simultaneous small |P| and |Q|
        if self.enable_coprime_regularizer and self.lambda_coprime > 0.0:
            # Sample a few points on the domain (monomial basis default is [-1,1])
            sample_points = [-0.9, -0.3, 0.0, 0.3, 0.9]
            # Build surrogate: sum_x [ 1/(1+|P|) * 1/(1+|Q|) ]
            from ..autodiff import tr_abs, tr_add, tr_div, tr_mul

            surrogate = TRNode.constant(real(0.0))
            max_deg = max(self.d_p, self.d_q)
            for xv in sample_points:
                x_node = TRNode.constant(real(float(xv)))
                psi = self.basis(x_node, max_deg)
                # P(x)
                P = self.theta[0] * psi[0]
                for k in range(1, self.d_p + 1):
                    if k < len(psi):
                        P = P + self.theta[k] * psi[k]
                # Q(x) = 1 + Σ φ_k ψ_k(x)
                Q = TRNode.constant(real(1.0))
                for k in range(1, self.d_q + 1):
                    if k < len(psi):
                        Q = Q + self.phi[k - 1] * psi[k]
                one = TRNode.constant(real(1.0))
                invP = tr_div(one, tr_add(one, tr_abs(P)))
                invQ = tr_div(one, tr_add(one, tr_abs(Q)))
                term = tr_mul(invP, invQ)
                surrogate = surrogate + term
            lam = TRNode.constant(real(self.lambda_coprime))
            total = total + lam * surrogate

        return total

    def estimate_local_scales(self, basis_bound: Optional[float] = None) -> tuple[float, float]:
        """
        Estimate local sensitivity scales (q,p) to guide TRPolicy thresholds.

        Heuristic proxy based on coefficient L1 norms and a basis bound B:
            local_scale_q ≈ 1 + B · ||φ||₁
            local_scale_p ≈ 1 + B · ||θ||₁
        """
        try:
            B = (
                float(basis_bound)
                if basis_bound is not None
                else float(getattr(self.basis, "bound", 1.0))
            )
        except Exception:
            B = 1.0

        def _l1(nodes):
            s = 0.0
            for n in nodes:
                try:
                    if n.value.tag == TRTag.REAL:
                        s += abs(float(n.value.value))
                except Exception:
                    pass
            return s

        phi_l1 = _l1(self.phi)
        theta_l1 = _l1(self.theta)
        local_q = 1.0 + B * phi_l1
        local_p = 1.0 + B * theta_l1
        return local_q, local_p

    def compute_q_min(self, x_batch: List[Union[TRScalar, TRNode]]) -> float:
        """
        Compute minimum |Q(x)| over a batch.

        Args:
            x_batch: List of input values

        Returns:
            min |Q(x_i)| over the batch
        """
        q_min = float("inf")

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
                    Q = Q + self.phi[k - 1] * psi[k]

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

    # ---- Layer contracts (second-order bounds) ----
    def get_layer_contract(self) -> dict:
        """
        Publish a conservative second-order contract for this layer.

        Returns:
            Dict with keys {"B_k","H_k","G_max","H_max","depth_hint"}
        """
        try:
            from ..optim_utils_second_order import estimate_contract_for_tr_rational

            c = estimate_contract_for_tr_rational(self)
            return {
                "B_k": float(c.B_k),
                "H_k": float(c.H_k),
                "G_max": float(c.G_max),
                "H_max": float(c.H_max),
                "depth_hint": int(c.depth_hint),
            }
        except Exception:
            return {
                "B_k": 1.0,
                "H_k": 1.0,
                "G_max": 1.0,
                "H_max": 1.0,
                "depth_hint": 4,
            }

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
            if hasattr(xs, "tolist"):
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
                    q_abs_values.append(float("inf"))
                    continue

            # Evaluate basis up to denominator degree
            psi = self.basis(x_node, self.d_q)
            Q = TRNode.constant(real(1.0))
            for k in range(1, self.d_q + 1):
                if k < len(psi) and k <= len(self.phi):
                    Q = Q + self.phi[k - 1] * psi[k]

            if Q.tag == TRTag.REAL:
                q_abs_values.append(abs(Q.value.value))
            else:
                # Non-REAL Q treated as 0 distance (at pole)
                q_abs_values.append(0.0)

        return q_abs_values

    # Distance estimator d(x) ≈ |Q(x)| / ||∂Q/∂x||_*
    def estimate_distance(self, x: Any) -> float:
        """
        Estimate distance to the pole set using |Q|/|Q'| for univariate inputs.

        Falls back to |Q| if basis derivative is unavailable.

        Args:
            x: scalar or TRNode/TRScalar

        Returns:
            Nonnegative float distance proxy (0 at/inside pole; finite elsewhere).
        """
        # Prepare node
        if isinstance(x, TRScalar):
            x_node = TRNode.constant(x)
        elif isinstance(x, TRNode):
            x_node = x
        else:
            try:
                x_node = TRNode.constant(real(float(x)))
            except Exception:
                return float("inf")

        # Evaluate Q(x)
        psi = self.basis(x_node, self.d_q)
        Q = TRNode.constant(real(1.0))
        for k in range(1, self.d_q + 1):
            if k < len(psi) and k <= len(self.phi):
                Q = Q + self.phi[k - 1] * psi[k]

        # If Q is non-REAL, we are at a pole; distance 0
        if Q.tag != TRTag.REAL:
            return 0.0

        # Try basis derivative if available
        dQ_abs = None
        try:
            deriv = getattr(self.basis, "derivative", None)
            if callable(deriv):
                dpsi = deriv(x_node, self.d_q)
                dQ = TRNode.constant(real(0.0))
                for k in range(1, self.d_q + 1):
                    if k < len(dpsi) and k <= len(self.phi):
                        dQ = dQ + self.phi[k - 1] * dpsi[k]
                if dQ.tag == TRTag.REAL:
                    dQ_abs = abs(dQ.value.value)
        except Exception:
            dQ_abs = None

        q_abs = abs(Q.value.value)
        # Fallback when derivative unavailable or non-REAL
        if dQ_abs is None or not isinstance(dQ_abs, (int, float)):
            return float(q_abs)

        tiny = 1e-12
        denom = max(dQ_abs, tiny)
        return float(q_abs) / denom

    def estimate_distance_batch(self, xs: Any) -> List[float]:
        """Vectorized wrapper of estimate_distance for iterable inputs."""
        try:
            iterator = iter(xs)
        except Exception:
            return [self.estimate_distance(xs)]
        distances: List[float] = []
        for x in iterator:
            # Apply projection if needed
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
                    # fallback: take first element when available
                    try:
                        x = x[0]  # type: ignore[index]
                    except Exception:
                        pass
            distances.append(self.estimate_distance(x))
        return distances

    # Policy sensitivity scales (heuristic)
    def get_policy_local_scales(self) -> Tuple[float, float]:
        """
        Provide heuristic local sensitivity scales for TRPolicy thresholds.

        scale_q ≈ 1 + B * ||φ||₁; scale_p ≈ 1 + B * ||θ||₁, where B is a basis bound.
        """
        try:
            B = float(getattr(self.basis, "bound", 1.0))
        except Exception:
            B = 1.0
        l1_phi = 0.0
        for phi_k in self.phi:
            try:
                if phi_k.value.tag == TRTag.REAL:
                    l1_phi += abs(float(phi_k.value.value))
            except Exception:
                pass
        l1_theta = 0.0
        for th_k in self.theta:
            try:
                if th_k.value.tag == TRTag.REAL:
                    l1_theta += abs(float(th_k.value.value))
            except Exception:
                pass
        scale_q = 1.0 + B * l1_phi
        scale_p = 1.0 + B * l1_theta
        return (scale_q, scale_p)


class TRRationalMulti:
    """
    Multi-output TR-Rational layer.

    Can share denominator Q across outputs for parameter efficiency.
    """

    def __init__(
        self,
        d_p: int,
        d_q: int,
        n_outputs: int,
        basis: Optional[Basis] = None,
        shared_Q: bool = True,
        lambda_rej: float = 0.0,
        alpha_phi: float = 1e-3,
        # Coprime surrogate flags forwarded to TRRational heads
        enable_coprime_regularizer: bool = False,
        lambda_coprime: float = 0.0,
    ):
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
            shared_layer = TRRational(
                d_p,
                d_q,
                basis,
                True,
                lambda_rej,
                alpha_phi,
                enable_coprime_regularizer=enable_coprime_regularizer,
                lambda_coprime=lambda_coprime,
            )

            # Create layers sharing the denominator parameters
            for i in range(n_outputs):
                if i == 0:
                    self.layers.append(shared_layer)
                else:
                    # Create new layer but share phi parameters
                    layer = TRRational(
                        d_p,
                        d_q,
                        basis,
                        True,
                        lambda_rej,
                        alpha_phi,
                        enable_coprime_regularizer=enable_coprime_regularizer,
                        lambda_coprime=lambda_coprime,
                    )
                    layer.phi = shared_layer.phi  # Share denominator
                    self.layers.append(layer)
        else:
            # Independent rational functions
            self.layers = [
                TRRational(
                    d_p,
                    d_q,
                    basis,
                    False,
                    lambda_rej,
                    alpha_phi,
                    enable_coprime_regularizer=enable_coprime_regularizer,
                    lambda_coprime=lambda_coprime,
                )
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
