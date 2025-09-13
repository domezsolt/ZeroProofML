"""
Multi-input TR model with a small TR-MLP front end feeding TR-Rational heads.

This module provides a simple way to consume vector inputs (e.g., 4D)
and produce multi-output predictions (e.g., 2D) using:

- A lightweight TR-MLP front end that maps R^D -> R^K scalar features
- One TR-Rational head per output that consumes one scalar feature

By default, the denominator Q is shared across outputs for parameter efficiency.
"""

from typing import List, Tuple, Optional, Union, Any

from ..core import TRTag, real
from ..autodiff import TRNode
from .tr_rational import TRRational, TRRationalMulti
from .basis import Basis, MonomialBasis


class _TRDense:
    """A tiny dense layer using TR parameters and activations."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        # Weights and biases
        self.W: List[List[TRNode]] = []
        self.b: List[TRNode] = []

        # He/Xavier-style scaling
        import math
        if activation == "relu":
            std = math.sqrt(2.0 / max(1, in_dim))
        else:
            std = math.sqrt(1.0 / max(1, in_dim))

        # Initialize parameters
        for i in range(out_dim):
            row = []
            for j in range(in_dim):
                # Simple Gaussian init scaled by std
                # Deterministic small init to avoid RNG dependency
                val = std * ((i + 1) * (j + 1)) / (in_dim * out_dim)
                row.append(TRNode.parameter(real(val), name=f"W_{i}_{j}"))
            self.W.append(row)
            self.b.append(TRNode.parameter(real(0.0), name=f"b_{i}"))

    def forward(self, x: List[TRNode]) -> List[TRNode]:
        outputs: List[TRNode] = []
        for i in range(self.out_dim):
            a = self.b[i]
            for j in range(self.in_dim):
                a = a + self.W[i][j] * x[j]
            if self.activation == "relu":
                if a.tag == TRTag.REAL and a.value.value > 0:
                    outputs.append(a)
                else:
                    outputs.append(TRNode.constant(real(0.0)))
            elif self.activation == "linear":
                outputs.append(a)
            else:
                # Default to linear for unsupported activations
                outputs.append(a)
        return outputs

    def parameters(self) -> List[TRNode]:
        params: List[TRNode] = []
        for row in self.W:
            params.extend(row)
        params.extend(self.b)
        return params


class TRMultiInputRational:
    """
    Multi-input, multi-output TR model:
    - Front-end TR-MLP maps vector input to K scalar features
    - TR-Rational heads consume those scalar features to produce outputs

    Example: input_dim=4, n_outputs=2, features_per_output=1 -> K=2
    """

    def __init__(self,
                 input_dim: int,
                 n_outputs: int,
                 d_p: int,
                 d_q: int,
                 basis: Optional[Basis] = None,
                 hidden_dims: Optional[List[int]] = None,
                 features_per_output: int = 1,
                 shared_Q: bool = True,
                 enable_pole_head: bool = False):
        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.d_p = d_p
        self.d_q = d_q
        self.basis = basis or MonomialBasis()
        self.hidden_dims = hidden_dims or [8]
        self.features_per_output = max(1, features_per_output)
        self.shared_Q = shared_Q
        self.enable_pole_head = enable_pole_head

        # Build a simple MLP that ends with K = n_outputs * features_per_output scalars
        self._build_frontend()

        # Rational heads: one per output; each consumes 1 scalar feature by default
        # Use independent rationals for clarity; could also share Q via TRRationalMulti
        self.heads: List[TRRational] = []
        if self.shared_Q and self.features_per_output == 1:
            # Share Q across outputs via TRRationalMulti, but wrap for unified API
            self.multi = TRRationalMulti(d_p=self.d_p, d_q=self.d_q, n_outputs=self.n_outputs,
                                         basis=self.basis, shared_Q=True)
            # Create thin wrappers referencing multi.layers for parameter sharing
            self.heads = self.multi.layers  # type: ignore[attr-defined]
        else:
            # Independent heads
            for _ in range(self.n_outputs):
                self.heads.append(TRRational(d_p=self.d_p, d_q=self.d_q, basis=self.basis))

        # Optional simple pole head on concatenated features (linear layer + sigmoid approx handled in trainer)
        if self.enable_pole_head:
            from ..autodiff import TRNode
            from ..core import real
            k = self.n_outputs * self.features_per_output
            self.pole_W: List[TRNode] = []
            for i in range(k):
                # small init
                self.pole_W.append(TRNode.parameter(real(0.0), name=f"pole_W_{i}"))
            self.pole_b = TRNode.parameter(real(0.0), name="pole_b")

    def _build_frontend(self) -> None:
        dims = [self.input_dim] + self.hidden_dims
        self.layers: List[_TRDense] = []
        # Hidden layers (ReLU)
        for i in range(len(dims) - 1):
            self.layers.append(_TRDense(dims[i], dims[i+1], activation="relu"))
        # Output layer produces K scalar features (linear activation)
        k = self.n_outputs * self.features_per_output
        last_in = dims[-1] if dims else self.input_dim
        self.layers.append(_TRDense(last_in, k, activation="linear"))

    def _ensure_trnodes(self, x: Union[List[TRNode], List[float], Tuple[float, ...], Any]) -> List[TRNode]:
        # Accept list/tuple of floats or TRNodes
        if isinstance(x, (list, tuple)):
            tr = []
            for xi in x:
                if isinstance(xi, TRNode):
                    tr.append(xi)
                else:
                    tr.append(TRNode.constant(real(float(xi))))
            if len(tr) != self.input_dim:
                raise ValueError(f"Expected input_dim={self.input_dim}, got {len(tr)}")
            return tr
        raise TypeError("TRMultiInputRational expects a list/tuple input")

    def _frontend_forward(self, x_vec: List[TRNode]) -> List[TRNode]:
        h = x_vec
        for layer in self.layers:
            h = layer.forward(h)
        return h  # length = n_outputs * features_per_output

    def forward(self, x: Union[List[TRNode], List[float], Tuple[float, ...]]) -> List[Tuple[TRNode, TRTag]]:
        x_vec = self._ensure_trnodes(x)
        feats = self._frontend_forward(x_vec)

        # Consume features by heads
        outputs: List[Tuple[TRNode, TRTag]] = []
        if self.features_per_output == 1:
            # One scalar per output
            for i, head in enumerate(self.heads):
                z = feats[i]
                y, tag = head.forward(z)
                outputs.append((y, tag))
        else:
            # If multiple features per output, sum or average features into a scalar
            # Here we simply average per-output feature group
            for i, head in enumerate(self.heads):
                start = i * self.features_per_output
                end = start + self.features_per_output
                # Average features
                acc = feats[start]
                for j in range(start + 1, end):
                    acc = acc + feats[j]
                acc = acc / TRNode.constant(real(float(self.features_per_output)))
                y, tag = head.forward(acc)
                outputs.append((y, tag))
        return outputs

    def forward_pole_head(self, x: Union[List[TRNode], List[float], Tuple[float, ...]]) -> Optional[TRNode]:
        if not getattr(self, 'enable_pole_head', False):
            return None
        x_vec = self._ensure_trnodes(x)
        feats = self._frontend_forward(x_vec)
        # Aggregate features per-output groups (average) if features_per_output > 1
        if self.features_per_output > 1:
            agg = []
            for i in range(self.n_outputs):
                start = i * self.features_per_output
                end = start + self.features_per_output
                acc = feats[start]
                for j in range(start + 1, end):
                    acc = acc + feats[j]
                acc = acc / TRNode.constant(real(float(self.features_per_output)))
                agg.append(acc)
            feats_use = agg
        else:
            feats_use = feats
        # Linear score
        score = self.pole_b
        for i, f in enumerate(feats_use):
            if i < len(self.pole_W):
                score = score + self.pole_W[i] * f
        return score

    def forward_fully_integrated(self, x: Union[List[TRNode], List[float], Tuple[float, ...]]) -> dict:
        """
        Forward pass returning a structured result for interface parity.

        Returns a dictionary with:
        - 'outputs': List[TRNode] (length = n_outputs)
        - 'tags':    List[TRTag]  (length = n_outputs)
        - 'Q_abs_list': Optional[List[float]] if per-head Q magnitudes are available
        - 'pole_score': Optional[TRNode] if pole head is enabled
        """
        x_vec = self._ensure_trnodes(x)
        feats = self._frontend_forward(x_vec)

        outputs: List[TRNode] = []
        tags: List[TRTag] = []
        q_abs_list: List[float] = []

        if self.features_per_output == 1:
            for i, head in enumerate(self.heads):
                z = feats[i]
                y, tag = head.forward(z)
                outputs.append(y)
                tags.append(tag)
                q_abs = getattr(head, '_last_Q_abs', None)
                if isinstance(q_abs, (int, float)):
                    q_abs_list.append(float(q_abs))
        else:
            for i, head in enumerate(self.heads):
                start = i * self.features_per_output
                end = start + self.features_per_output
                acc = feats[start]
                for j in range(start + 1, end):
                    acc = acc + feats[j]
                acc = acc / TRNode.constant(real(float(self.features_per_output)))
                y, tag = head.forward(acc)
                outputs.append(y)
                tags.append(tag)
                q_abs = getattr(head, '_last_Q_abs', None)
                if isinstance(q_abs, (int, float)):
                    q_abs_list.append(float(q_abs))

        result = {
            'outputs': outputs,
            'tags': tags,
        }
        if q_abs_list:
            result['Q_abs_list'] = q_abs_list

        if getattr(self, 'enable_pole_head', False):
            # Provide a raw linear score for pole proximity
            result['pole_score'] = self.forward_pole_head(x_vec)

        return result

    def __call__(self, x: Union[List[TRNode], List[float], Tuple[float, ...]]) -> List[TRNode]:
        return [y for y, _ in self.forward(x)]

    def parameters(self) -> List[TRNode]:
        params: List[TRNode] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        for head in self.heads:
            params.extend(head.parameters())
        if getattr(self, 'enable_pole_head', False):
            params.extend(self.pole_W)
            params.append(self.pole_b)
        return params

    def pole_parameters(self) -> List[TRNode]:
        params: List[TRNode] = []
        if getattr(self, 'enable_pole_head', False):
            params.extend(self.pole_W)
            params.append(self.pole_b)
        return params

    # Compatibility helpers
    def forward_with_tag(self, x: Union[List[TRNode], List[float], Tuple[float, ...]]) -> List[Tuple[TRNode, TRTag]]:
        return self.forward(x)

    def regularization_loss(self) -> TRNode:
        # Sum regularization of heads (front-end has no explicit regularizer here)
        if hasattr(self, 'multi'):
            return self.multi.regularization_loss()  # type: ignore[attr-defined]
        # If independent heads, sum their regs
        reg = TRNode.constant(real(0.0))
        for head in self.heads:
            reg = reg + head.regularization_loss()
        return reg
