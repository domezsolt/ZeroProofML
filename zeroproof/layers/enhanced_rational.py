"""
Enhanced TRRational layer with improved pole detection.

This module provides an enhanced rational layer that integrates the improved
pole detection head with better initialization and training strategies.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode
from ..autodiff.hybrid_gradient import HybridGradientSchedule
from ..autodiff.tr_ops_grad import tr_add, tr_div
from .tr_rational import TRRational, TRRationalMulti
from .basis import Basis, MonomialBasis, ChebyshevBasis
from .enhanced_pole_detection import (
    EnhancedPoleConfig,
    EnhancedPoleDetectionHead,
    PoleRegularizer,
    PoleAwareRationalInterface
)


class EnhancedTRRational(TRRational):
    """
    Enhanced TRRational layer with improved pole detection.
    
    Features:
    - Integrated enhanced pole detection head
    - Pole regularization for controlling singularity locations
    - Better initialization strategies
    - Improved loss integration
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
                 projection_index: Optional[int] = None,
                 # Pole detection parameters
                 enable_pole_detection: bool = True,
                 pole_config: Optional[EnhancedPoleConfig] = None,
                 target_poles: Optional[List[float]] = None,
                 enable_pole_regularization: bool = False):
        """
        Initialize enhanced TRRational layer.
        
        Args:
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q
            basis: Basis functions to use
            shared_Q: If True, share Q across multiple outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominator
            l1_projection: Optional L1 bound for φ
            adaptive_loss_policy: Optional adaptive loss policy
            projection_index: Index for input projection
            enable_pole_detection: Whether to enable pole detection
            pole_config: Configuration for pole detection
            target_poles: Target pole locations for regularization
            enable_pole_regularization: Whether to use pole regularization
        """
        # Initialize base layer
        super().__init__(
            d_p=d_p,
            d_q=d_q,
            basis=basis,
            shared_Q=shared_Q,
            lambda_rej=lambda_rej,
            alpha_phi=alpha_phi,
            l1_projection=l1_projection,
            adaptive_loss_policy=adaptive_loss_policy,
            projection_index=projection_index
        )
        
        # Pole detection setup
        self.enable_pole_detection = enable_pole_detection
        self.pole_config = pole_config or EnhancedPoleConfig()
        self.target_poles = target_poles or []
        self.enable_pole_regularization = enable_pole_regularization
        
        # Create pole detection interface if enabled
        self.pole_interface = None
        if self.enable_pole_detection:
            self.pole_interface = PoleAwareRationalInterface(
                rational_layer=self,
                pole_config=self.pole_config,
                enable_regularization=self.enable_pole_regularization,
                target_poles=self.target_poles
            )
        
        # Tracking
        self.pole_detection_metrics = []
        self._last_pole_probability = None
    
    def forward_with_pole_detection(self, 
                                   x: Union[TRScalar, TRNode]) -> Dict[str, Any]:
        """
        Forward pass with integrated pole detection.
        
        Args:
            x: Input value
            
        Returns:
            Dictionary with output, tag, and pole information
        """
        if self.pole_interface:
            result = self.pole_interface.forward_with_pole_detection(x)
            self._last_pole_probability = result.get('pole_probability', None)
            return result
        else:
            # Standard forward without pole detection
            y, tag = self.forward(x)
            return {
                'output': y,
                'tag': tag,
                'pole_score': None,
                'pole_probability': None,
                'Q_abs': getattr(self, '_last_Q_abs', None)
            }
    
    def compute_pole_loss(self,
                         predictions: List[TRNode],
                         Q_values: Optional[List[float]] = None,
                         teacher_labels: Optional[List[float]] = None,
                         x_samples: Optional[List[float]] = None) -> TRNode:
        """
        Compute pole-related losses.
        
        Args:
            predictions: Pole detection scores
            Q_values: Actual |Q(x)| values
            teacher_labels: Optional teacher signals
            x_samples: Input samples for regularization
            
        Returns:
            Combined pole loss
        """
        if self.pole_interface:
            return self.pole_interface.compute_pole_loss(
                predictions, Q_values, teacher_labels, x_samples
            )
        else:
            return TRNode.constant(real(0.0))
    
    def parameters(self) -> List[TRNode]:
        """Get all parameters including pole detection head."""
        if self.pole_interface:
            return self.pole_interface.get_parameters()
        else:
            return super().parameters()
    
    def evaluate_pole_detection(self, 
                               test_data: List[Tuple[float, bool]]) -> Dict[str, float]:
        """
        Evaluate pole detection accuracy.
        
        Args:
            test_data: List of (x_value, is_pole) tuples
            
        Returns:
            Dictionary of metrics
        """
        if self.pole_interface:
            metrics = self.pole_interface.evaluate_accuracy(test_data)
            self.pole_detection_metrics.append(metrics)
            return metrics
        else:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def get_pole_predictions(self, x_values: List[float]) -> List[float]:
        """
        Get pole predictions for a list of x values.
        
        Args:
            x_values: Input values
            
        Returns:
            List of pole probabilities
        """
        if not self.pole_interface:
            return [0.0] * len(x_values)
        
        predictions = []
        for x_val in x_values:
            x_node = TRNode.constant(real(x_val))
            prob = self.pole_interface.pole_head.predict_pole_probability(x_node)
            predictions.append(prob)
        
        return predictions


class EnhancedTRRationalMulti(TRRationalMulti):
    """
    Enhanced multi-output rational layer with shared pole detection.
    
    This layer shares pole detection across multiple outputs, which is
    efficient when the outputs share similar singularity structures.
    """
    
    def __init__(self,
                 n_outputs: int,
                 d_p: int,
                 d_q: int,
                 basis: Optional[Basis] = None,
                 shared_Q: bool = True,
                 lambda_rej: float = 0.0,
                 alpha_phi: float = 1e-3,
                 l1_projection: Optional[float] = None,
                 adaptive_loss_policy=None,
                 # Pole detection parameters
                 enable_pole_detection: bool = True,
                 pole_config: Optional[EnhancedPoleConfig] = None,
                 shared_pole_head: bool = True,
                 target_poles: Optional[List[float]] = None,
                 enable_pole_regularization: bool = False):
        """
        Initialize enhanced multi-output rational layer.
        
        Args:
            n_outputs: Number of outputs
            d_p: Degree of numerator polynomial P
            d_q: Degree of denominator polynomial Q
            basis: Basis functions to use
            shared_Q: If True, share Q across outputs
            lambda_rej: Penalty for non-REAL outputs
            alpha_phi: L2 regularization for denominator
            l1_projection: Optional L1 bound for φ
            adaptive_loss_policy: Optional adaptive loss policy
            enable_pole_detection: Whether to enable pole detection
            pole_config: Configuration for pole detection
            shared_pole_head: Whether to share pole head across outputs
            target_poles: Target pole locations for regularization
            enable_pole_regularization: Whether to use pole regularization
        """
        # Initialize base layer
        super().__init__(
            n_outputs=n_outputs,
            d_p=d_p,
            d_q=d_q,
            basis=basis,
            shared_Q=shared_Q,
            lambda_rej=lambda_rej,
            alpha_phi=alpha_phi,
            l1_projection=l1_projection,
            adaptive_loss_policy=adaptive_loss_policy
        )
        
        # Pole detection setup
        self.enable_pole_detection = enable_pole_detection
        self.pole_config = pole_config or EnhancedPoleConfig()
        self.shared_pole_head = shared_pole_head
        self.target_poles = target_poles or []
        self.enable_pole_regularization = enable_pole_regularization
        
        # Create pole detection heads
        self.pole_interfaces = []
        if self.enable_pole_detection:
            if self.shared_pole_head:
                # Single shared pole head
                shared_interface = PoleAwareRationalInterface(
                    rational_layer=self,
                    pole_config=self.pole_config,
                    enable_regularization=self.enable_pole_regularization,
                    target_poles=self.target_poles
                )
                self.pole_interfaces = [shared_interface] * n_outputs
            else:
                # Separate pole head for each output
                for i in range(n_outputs):
                    interface = PoleAwareRationalInterface(
                        rational_layer=self,
                        pole_config=self.pole_config,
                        enable_regularization=self.enable_pole_regularization,
                        target_poles=self.target_poles
                    )
                    self.pole_interfaces.append(interface)
    
    def forward_with_pole_detection(self, 
                                   x: Union[TRScalar, TRNode]) -> List[Dict[str, Any]]:
        """
        Forward pass with pole detection for all outputs.
        
        Args:
            x: Input value
            
        Returns:
            List of dictionaries with output information
        """
        results = []
        
        # Get standard outputs
        outputs, tags = self.forward(x)
        
        # Add pole detection for each output
        for i, (y, tag) in enumerate(zip(outputs, tags)):
            if self.enable_pole_detection and i < len(self.pole_interfaces):
                # Get pole prediction for this output
                interface = self.pole_interfaces[i]
                pole_score = interface.pole_head.forward(x)
                pole_prob = interface.pole_head.predict_pole_probability(x)
                
                result = {
                    'output': y,
                    'tag': tag,
                    'pole_score': pole_score,
                    'pole_probability': pole_prob,
                    'Q_abs': getattr(self, f'_last_Q_abs_{i}', None)
                }
            else:
                result = {
                    'output': y,
                    'tag': tag,
                    'pole_score': None,
                    'pole_probability': None,
                    'Q_abs': None
                }
            
            results.append(result)
        
        return results
    
    def compute_pole_loss(self,
                         predictions_list: List[List[TRNode]],
                         Q_values_list: Optional[List[List[float]]] = None,
                         teacher_labels_list: Optional[List[List[float]]] = None,
                         x_samples: Optional[List[float]] = None) -> TRNode:
        """
        Compute pole loss for all outputs.
        
        Args:
            predictions_list: Pole scores for each output
            Q_values_list: Q values for each output
            teacher_labels_list: Teacher labels for each output
            x_samples: Input samples
            
        Returns:
            Combined pole loss
        """
        total_loss = TRNode.constant(real(0.0))
        
        if not self.enable_pole_detection:
            return total_loss
        
        # Compute loss for each output
        for i in range(self.n_outputs):
            if i < len(self.pole_interfaces):
                interface = self.pole_interfaces[i]
                
                predictions = predictions_list[i] if i < len(predictions_list) else []
                Q_values = Q_values_list[i] if Q_values_list and i < len(Q_values_list) else None
                teacher_labels = teacher_labels_list[i] if teacher_labels_list and i < len(teacher_labels_list) else None
                
                output_loss = interface.compute_pole_loss(
                    predictions, Q_values, teacher_labels, x_samples
                )
                
                total_loss = tr_add(total_loss, output_loss)
        
        # Average over outputs
        if self.n_outputs > 0:
            n_outputs_node = TRNode.constant(real(float(self.n_outputs)))
            total_loss = tr_div(total_loss, n_outputs_node)
        
        return total_loss
    
    def evaluate_pole_detection(self,
                               test_data: List[Tuple[float, bool]]) -> List[Dict[str, float]]:
        """
        Evaluate pole detection for all outputs.
        
        Args:
            test_data: Test data
            
        Returns:
            List of metrics for each output
        """
        metrics_list = []
        
        if self.shared_pole_head and self.pole_interfaces:
            # Single shared evaluation
            metrics = self.pole_interfaces[0].evaluate_accuracy(test_data)
            metrics_list = [metrics] * self.n_outputs
        else:
            # Separate evaluation for each output
            for interface in self.pole_interfaces:
                metrics = interface.evaluate_accuracy(test_data)
                metrics_list.append(metrics)
        
        return metrics_list


def create_enhanced_rational(d_p: int,
                            d_q: int,
                            basis_type: str = "polynomial",
                            enable_pole_detection: bool = True,
                            target_poles: Optional[List[float]] = None,
                            **kwargs) -> EnhancedTRRational:
    """
    Factory function to create enhanced rational layer.
    
    Args:
        d_p: Degree of numerator
        d_q: Degree of denominator
        basis_type: Type of basis ("polynomial", "chebyshev")
        enable_pole_detection: Whether to enable pole detection
        target_poles: Optional target pole locations
        **kwargs: Additional arguments for the layer
        
    Returns:
        Enhanced rational layer
    """
    # Create basis
    if basis_type == "chebyshev":
        basis = ChebyshevBasis()
    else:
        basis = MonomialBasis()
    
    # Create enhanced pole config with good defaults
    pole_config = EnhancedPoleConfig(
        hidden_dims=[32, 16, 8],
        init_strategy="xavier_uniform",
        teacher_weight=0.7,
        pole_loss_weight=0.2,  # Increased weight
        false_positive_penalty=2.0,
        false_negative_penalty=1.5
    )
    
    # Create layer
    layer = EnhancedTRRational(
        d_p=d_p,
        d_q=d_q,
        basis=basis,
        enable_pole_detection=enable_pole_detection,
        pole_config=pole_config,
        target_poles=target_poles,
        enable_pole_regularization=(target_poles is not None),
        **kwargs
    )
    
    return layer
