"""
Enhanced trainer with hybrid gradient schedule support.

This module extends the basic trainer to support sophisticated training
strategies including hybrid gradient schedules, tag-loss, and pole learning.
"""

from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass
import math
import time

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode, backward_pass, tr_mul
from ..autodiff.grad_mode import GradientMode, GradientModeConfig
from ..autodiff.hybrid_gradient import (
    HybridGradientContext, 
    HybridGradientSchedule,
    create_default_schedule
)
from ..utils.bridge import to_trnode_constant
from .trainer import TRTrainer, TrainingConfig, Optimizer
from .adaptive_loss import AdaptiveLossPolicy
from .coverage import CoverageTracker
# Enhanced coverage components imported when needed
from .pole_detection import (
    PoleDetectionConfig,
    compute_pole_loss,
    DomainSpecificPoleDetector
)
from ..utils.metrics import (
    AntiIllusionMetrics,
    PoleLocation,
    ResidualConsistencyLoss
)
from ..utils.logging import (
    StructuredLogger
)


class CoverageStrategy:
    """Strategy for coverage enforcement."""
    LAGRANGE = "lagrange"  # Lagrange multiplier adjustment
    PENALTY = "penalty"  # Direct penalty adjustment
    ADAPTIVE = "adaptive"  # Adaptive strategy based on metrics


@dataclass
class HybridTrainingConfig(TrainingConfig):
    """Extended configuration for hybrid training."""
    
    # Hybrid gradient schedule
    use_hybrid_gradient: bool = False
    hybrid_warmup_epochs: int = 20
    hybrid_transition_epochs: int = 30
    hybrid_delta_init: float = 1e-2
    hybrid_delta_final: float = 1e-6
    hybrid_aggressive: bool = False
    
    # New: Enhanced hybrid gradient parameters
    hybrid_force_pole_exploration: bool = True
    hybrid_pole_exploration_radius: float = 0.05  # δ-neighborhood radius
    hybrid_pole_exploration_epochs: int = 5  # Epochs to explore each pole
    hybrid_pole_detection_threshold: float = 0.1  # |Q| threshold for pole
    hybrid_adaptive_delta: bool = True  # Adapt delta based on q_min
    hybrid_min_delta: float = 1e-8  # Minimum delta value
    hybrid_schedule_type: str = "EXPONENTIAL"  # LINEAR, EXPONENTIAL, COSINE
    
    # Tag loss (for non-REAL outputs)
    use_tag_loss: bool = False
    lambda_tag: float = 0.05
    tag_loss_adaptive: bool = True  # Increase weight when coverage is too high
    tag_loss_max_weight: float = 0.2  # Maximum tag loss weight
    
    # Pole detection head
    use_pole_head: bool = False
    lambda_pole: float = 0.1
    pole_head_degree: int = 3
    pole_config: Optional[PoleDetectionConfig] = None
    use_teacher_signals: bool = False
    pole_proximity_threshold: float = 0.1
    
    # Enhanced metrics
    track_pole_metrics: bool = False
    compute_ple: bool = False  # Pole Localization Error
    
    # Anti-illusion metrics
    enable_anti_illusion: bool = False
    lambda_residual: float = 0.01
    ground_truth_poles: Optional[List[Tuple[float, Optional[float]]]] = None
    ple_x_range: Tuple[float, float] = (-2.0, 2.0)
    residual_near_pole_threshold: float = 0.2
    
    # Coverage enforcement
    enforce_coverage: bool = False
    min_coverage: float = 0.7
    max_lambda_for_coverage: float = 10.0
    coverage_strategy: CoverageStrategy = CoverageStrategy.LAGRANGE
    coverage_window_size: int = 50
    oversample_near_pole: bool = True  # Changed default to True (critical)
    pole_sampling_threshold: float = 0.1
    
    # Enhanced coverage tracking
    use_enhanced_coverage: bool = True
    track_near_pole_coverage: bool = True
    near_pole_target_coverage: float = 0.7
    coverage_dead_band: float = 0.02
    asymmetric_increase_rate: float = 2.0
    asymmetric_decrease_rate: float = 0.5
    
    # Adaptive grid sampling
    use_adaptive_grid: bool = True
    initial_grid_size: int = 100
    grid_refinement_factor: int = 5
    pole_refinement_radius: float = 0.1
    
    # Logging and tracking
    enable_structured_logging: bool = True
    log_interval: int = 1  # Log every N epochs

    # TR policy activation (policy-driven hybrid hysteresis)
    use_tr_policy: bool = False
    policy_ulp_scale: float = 4.0
    policy_deterministic_reduction: bool = True
    policy_g_on: Optional[float] = None
    policy_g_off: Optional[float] = None
    save_plots: bool = True
    run_dir: Optional[str] = None
    
    # Second-order safeguard: shrink LR when SAT dominates
    sat_shrink_on_ratio: bool = True
    sat_ratio_threshold: float = 0.3
    sat_shrink_factor: float = 0.5
    
    # Logging: print curvature/Fisher proxies each epoch
    log_curvature_fisher: bool = True

    # Contract-safe LR clamp (from layer contract bounds)
    use_contract_safe_lr: bool = False
    contract_c: float = 1.0
    loss_smoothness_beta: float = 1.0


class HybridTRTrainer(TRTrainer):
    """
    Enhanced trainer with hybrid gradient schedule support.
    
    This trainer supports:
    - Hybrid gradient schedules for near-pole learning
    - Tag-loss for non-REAL outputs
    - Pole detection heads
    - Advanced metrics for pole learning verification
    """
    
    def _init_samplers(self) -> None:
        """Initialize sampling components if configured."""
        # Initialize near-pole sampler
        if self.hybrid_config.oversample_near_pole:
            from .enhanced_coverage import NearPoleSampler
            self.near_pole_sampler = NearPoleSampler(
                pole_threshold=self.hybrid_config.pole_sampling_threshold,
                oversample_ratio=2.0,  # Default ratio
                adaptive=True
            )
        else:
            self.near_pole_sampler = None
        
        # Initialize adaptive grid sampler
        if self.hybrid_config.use_adaptive_grid:
            from .enhanced_coverage import AdaptiveGridSampler
            self.adaptive_grid_sampler = AdaptiveGridSampler(
                initial_grid_size=self.hybrid_config.initial_grid_size,
                refinement_factor=self.hybrid_config.grid_refinement_factor,
                pole_radius=self.hybrid_config.pole_refinement_radius
            )
        else:
            self.adaptive_grid_sampler = None
    
    def __init__(self, 
                 model: Any,
                 optimizer: Optional[Optimizer] = None,
                 config: Optional[HybridTrainingConfig] = None):
        """
        Initialize hybrid trainer.
        
        Args:
            model: Model to train (should support hybrid features)
            optimizer: Optimizer instance
            config: Hybrid training configuration
        """
        # Support legacy/init-with-config usage: HybridTRTrainer(config)
        if isinstance(model, HybridTrainingConfig) and optimizer is None and config is None:
            config = model
            # Minimal placeholder model with no trainable parameters
            class _NoModel:
                def parameters(self):
                    return []
            model = _NoModel()
        # Use hybrid config or create default
        config = config or HybridTrainingConfig()
        super().__init__(model, optimizer, config)
        
        # Cast config to HybridTrainingConfig for type checking
        self.hybrid_config: HybridTrainingConfig = config  # type: ignore
        
        # Initialize sampling components
        self._init_samplers()

        # Ensure optional policies/trackers are defined
        self.coverage_policy = None
        self.pole_detector = None
        self.anti_illusion_metrics = None
        self.residual_loss = None

        # Optionally enable a default TRPolicy for tagging and hybrid hysteresis
        if self.hybrid_config.use_tr_policy:
            try:
                from .policy_utils import enable_policy_from_model, enable_default_tr_policy
                # Prefer model-aware thresholds when possible
                if hasattr(self.model, 'estimate_local_scales') and callable(getattr(self.model, 'estimate_local_scales')):
                    enable_policy_from_model(
                        self.model,
                        ulp_scale=self.hybrid_config.policy_ulp_scale,
                        deterministic_reduction=self.hybrid_config.policy_deterministic_reduction,
                        g_on=self.hybrid_config.policy_g_on,
                        g_off=self.hybrid_config.policy_g_off,
                    )
                else:
                    enable_default_tr_policy(
                        ulp_scale=self.hybrid_config.policy_ulp_scale,
                        deterministic_reduction=self.hybrid_config.policy_deterministic_reduction,
                        g_on=self.hybrid_config.policy_g_on,
                        g_off=self.hybrid_config.policy_g_off,
                    )
            except Exception:
                # Non-fatal: training can proceed without a policy
                pass
        
        # Initialize hybrid gradient schedule if enabled
        self.hybrid_schedule = None
        if self.hybrid_config.use_hybrid_gradient:
            self.hybrid_schedule = self._create_hybrid_schedule()
            
            # Register with model if it supports hybrid
            if hasattr(model, 'hybrid_schedule'):
                model.hybrid_schedule = self.hybrid_schedule
        
        # Initialize tracking
        self.pole_metrics_history = []
        self.tag_statistics = []
        self.gradient_mode_history = []
        self.bench_history = []  # per-epoch timing summaries
        self._epoch_records = []  # per-epoch plotting records

        # Persistent per-head and frontend optimizers if model exposes heads/layers
        self.head_optimizers = None
        self.frontend_optimizer = None
        try:
            if hasattr(self.model, 'heads') and isinstance(self.model.heads, list) and self.model.heads:
                from typing import List as _List
                self.head_optimizers = [
                    Optimizer(h.parameters(), learning_rate=self.optimizer.learning_rate)
                    for h in self.model.heads
                ]
                if hasattr(self.model, 'layers') and isinstance(self.model.layers, list):
                    frontend_params = []  # type: _List[TRNode]
                    for lyr in self.model.layers:
                        if hasattr(lyr, 'parameters'):
                            frontend_params.extend(lyr.parameters())
                    if frontend_params:
                        self.frontend_optimizer = Optimizer(frontend_params, learning_rate=self.optimizer.learning_rate)
        except Exception:
            self.head_optimizers = None
            self.frontend_optimizer = None

    def zero_grad_all(self) -> None:
        if self.head_optimizers is not None:
            for opt in self.head_optimizers:
                opt.zero_grad()
            if self.frontend_optimizer is not None:
                self.frontend_optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

    def step_all(self) -> None:
        if self.head_optimizers is not None and hasattr(self.model, 'heads'):
            # Step heads
            for opt, head in zip(self.head_optimizers, self.model.heads):
                opt.step(head)
            # Step frontend
            if self.frontend_optimizer is not None:
                self.frontend_optimizer.step(self.model)
        else:
            self.optimizer.step(self.model)
        
        # Initialize domain-specific pole detector if using teacher signals
        self.pole_detector = None
        if self.hybrid_config.use_teacher_signals:
            self.pole_detector = DomainSpecificPoleDetector()
        
        # Initialize anti-illusion metrics
        self.anti_illusion_metrics = None
        self.residual_loss = None
        self.ground_truth_poles = []
        
        if self.hybrid_config.enable_anti_illusion:
            self.anti_illusion_metrics = AntiIllusionMetrics()
            self.residual_loss = ResidualConsistencyLoss(
                weight=self.hybrid_config.lambda_residual
            )
            
            # Convert ground truth poles format
            if self.hybrid_config.ground_truth_poles:
                for pole_data in self.hybrid_config.ground_truth_poles:
                    if len(pole_data) >= 2 and pole_data[1] is not None:
                        # 2D pole
                        self.ground_truth_poles.append(
                            PoleLocation(x=pole_data[0], y=pole_data[1])
                        )
                    else:
                        # 1D pole
                        self.ground_truth_poles.append(
                            PoleLocation(x=pole_data[0])
                        )
        
        # Initialize coverage enforcement if enabled
        self.coverage_policy = None
        if self.hybrid_config.enforce_coverage:
            from .enhanced_coverage import CoverageEnforcementPolicy
            self.coverage_policy = CoverageEnforcementPolicy(
                target_coverage=self.config.target_coverage,
                near_pole_target=self.hybrid_config.near_pole_target_coverage,
                dead_band=self.hybrid_config.coverage_dead_band,
                increase_rate=self.hybrid_config.asymmetric_increase_rate,
                decrease_rate=self.hybrid_config.asymmetric_decrease_rate,
                min_lambda=self.config.adaptive_lambda_min,
                max_lambda=self.hybrid_config.max_lambda_for_coverage
            )

    # Lightweight loss helper to support tests that call trainer.compute_loss(preds, targets)
    def compute_loss(self, 
                     predictions: List[TRNode], 
                     targets: Any) -> Any:
        """
        Compute batch loss for a list of TRNode predictions and targets.
        Returns an adapter with .backward() and .item() to fit simple training loops.
        """
        # Normalize targets to Python floats/TRNodes
        from ..utils.bridge import to_trnode_constant
        if hasattr(targets, 'tolist'):
            target_list = targets.tolist()
        else:
            target_list = list(targets)
        target_nodes = [to_trnode_constant(t) for t in target_list]

        # Reuse internal policy if available, else simple MSE
        if self.loss_policy:
            loss_node = self.loss_policy.compute_batch_loss(predictions, target_nodes)
        else:
            from ..core import tr_sum, tr_div
            losses = []
            for pred, target in zip(predictions, target_nodes):
                diff = pred - target
                losses.append((TRNode.constant(real(0.5)) * diff * diff).value)
            total = tr_sum(losses)
            loss_node = TRNode.constant(tr_div(total, real(float(len(losses)))))

        # Infer model parameters from the prediction graph if optimizer has none
        try:
            if isinstance(predictions, list):
                inferred = self._infer_parameters_from_predictions(predictions)
                if inferred:
                    self.optimizer.parameters = inferred
        except Exception:
            # Non-fatal: keep existing optimizer parameter list
            pass

        class _LossAdapter:
            def __init__(self, node: TRNode):
                self._node = node
            def backward(self):
                self._node.backward()
            def item(self):
                return float(self._node.value.value) if self._node.value.tag == TRTag.REAL else float('nan')

        # Register exact-pole enforcement requests (no ε): use metadata embedded in predictions
        try:
            # Ensure the optimizer can store requests
            if not hasattr(self.optimizer, '_enforce_requests'):
                self.optimizer._enforce_requests = []  # type: ignore[attr-defined]
            # Record at most d_q requests per batch to avoid over-constraint
            requests = []
            for pred, tgt in zip(predictions, target_nodes):
                tgt_tag = tgt.value.tag if hasattr(tgt, 'value') else None
                if tgt_tag is not None and tgt_tag != TRTag.REAL:
                    gi = getattr(pred, '_grad_info', None)
                    if gi is not None:
                        x_val = gi.extra_data.get('input_x', None)
                        phi_refs = gi.extra_data.get('tr_rational_phi', None)
                        basis_ref = gi.extra_data.get('tr_rational_basis', None)
                        d_q = gi.extra_data.get('tr_rational_dq', None)
                        if x_val is not None and phi_refs and basis_ref and d_q:
                            requests.append({
                                'x': float(x_val),
                                'phi': phi_refs,
                                'basis': basis_ref,
                                'd_q': int(d_q),
                            })
            # Limit number of constraints per step
            if requests:
                # Keep at most the first d_q unique x values
                seen = set()
                limited = []
                for r in requests:
                    xv = r['x']
                    if xv in seen:
                        continue
                    limited.append(r)
                    seen.add(xv)
                    if len(limited) >= min(len(requests[0]['phi']), requests[0]['d_q']):
                        break
                self.optimizer._enforce_requests.extend(limited)  # type: ignore[attr-defined]
        except Exception:
            pass

        return _LossAdapter(loss_node)

    def _infer_parameters_from_predictions(self, predictions: List[TRNode]) -> List[TRNode]:
        """
        Traverse the TR graph(s) to collect leaf parameter nodes.

        A parameter is identified as a TRNode with requires_grad=True and
        no grad_info (i.e., created via TRNode.parameter()).
        """
        params: List[TRNode] = []
        seen = set()

        def visit(node: TRNode) -> None:
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)

            # Parameter leaf
            if getattr(node, "requires_grad", False) and getattr(node, "_grad_info", None) is None:
                params.append(node)
                return

            gi = getattr(node, "_grad_info", None)
            if gi and getattr(gi, "inputs", None):
                for ref in gi.inputs:
                    inp = ref()
                    if inp is not None:
                        visit(inp)

        for p in predictions:
            if p is not None:
                visit(p)

        # Deduplicate while preserving order
        seen_ids = set()
        unique_params: List[TRNode] = []
        for p in params:
            pid = id(p)
            if pid not in seen_ids:
                unique_params.append(p)
                seen_ids.add(pid)

        return unique_params
    
    def _create_hybrid_schedule(self) -> HybridGradientSchedule:
        """Create hybrid gradient schedule from config."""
        from ..autodiff import ScheduleType
        
        # Map string to enum
        schedule_type_map = {
            "LINEAR": ScheduleType.LINEAR,
            "EXPONENTIAL": ScheduleType.EXPONENTIAL,
            "COSINE": ScheduleType.COSINE
        }
        schedule_type = schedule_type_map.get(
            self.hybrid_config.hybrid_schedule_type.upper(),
            ScheduleType.EXPONENTIAL
        )
        
        return HybridGradientSchedule(
            warmup_epochs=self.hybrid_config.hybrid_warmup_epochs,
            transition_epochs=self.hybrid_config.hybrid_transition_epochs,
            delta_init=self.hybrid_config.hybrid_delta_init,
            delta_final=self.hybrid_config.hybrid_delta_final,
            schedule_type=schedule_type,
            enable=True,
            saturating_bound=0.1 if self.hybrid_config.hybrid_aggressive else 1.0,
            force_pole_exploration=self.hybrid_config.hybrid_force_pole_exploration,
            pole_exploration_radius=self.hybrid_config.hybrid_pole_exploration_radius,
            pole_exploration_epochs=self.hybrid_config.hybrid_pole_exploration_epochs,
            pole_detection_threshold=self.hybrid_config.hybrid_pole_detection_threshold,
            adaptive_delta=self.hybrid_config.hybrid_adaptive_delta,
            min_delta=self.hybrid_config.hybrid_min_delta
        )
    
    def train_epoch(self,
                   data_loader: List[Tuple[List[TRScalar], List[TRScalar]]]
                   ) -> Dict[str, float]:
        """
        Train one epoch with hybrid gradient support.
        
        Args:
            data_loader: List of (inputs, targets) batches
            
        Returns:
            Dictionary of epoch metrics
        """
        # Update hybrid schedule if enabled
        if self.hybrid_schedule:
            HybridGradientContext.update_epoch(self.epoch)
            
            # Update model if it tracks epochs
            if hasattr(self.model, 'update_epoch'):
                self.model.update_epoch(self.epoch)
            
            # Set gradient mode
            delta = self.hybrid_schedule.get_delta(self.epoch)
            if delta is None:
                GradientModeConfig.set_mode(GradientMode.MASK_REAL)
            else:
                GradientModeConfig.set_mode(GradientMode.HYBRID)
                GradientModeConfig.set_local_threshold(delta)
            
            # Set exploration regions for this epoch
            exploration_regions = self.hybrid_schedule.get_exploration_regions(self.epoch)
            if exploration_regions:
                HybridGradientContext.set_exploration_regions(exploration_regions)
        
        metrics = {
            "loss": [],
            "coverage": [],
            "lambda_rej": [],
            "tag_loss": [],
            "pole_loss": [],
            "near_pole_ratio": [],
            # Curvature/second-order proxies
            "curvature_proxy": [],
            "gn_proxy": [],
            "grad_max": [],
            # Second-order safeguard envelope (from per-batch when available)
            "curvature_bound": [],
            "B_k": [],
            "H_k": [],
            "G_max": [],
            "H_max": [],
            # Policy thresholds (copied from hybrid stats per-batch when present)
            "tau_q_on": [],
            "tau_q_off": [],
        }
        
        # Initialize coverage tracker (use enhanced if configured)
        if self.hybrid_config.use_enhanced_coverage:
            from .enhanced_coverage import EnhancedCoverageTracker
            coverage_tracker = EnhancedCoverageTracker(
                target_coverage=self.config.target_coverage,
                pole_threshold=self.hybrid_config.pole_sampling_threshold,
                window_size=self.hybrid_config.coverage_window_size,
                track_pole_distances=self.hybrid_config.track_near_pole_coverage
            )
        else:
            coverage_tracker = CoverageTracker()
        
        # Bench accumulators
        total_step_ms = 0.0
        total_optim_ms = 0.0
        total_data_ms = 0.0  # For this trainer path, data prep time is minimal; we approximate as step - optim
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            t_step0 = time.time()
            t_opt0 = time.time()
            batch_metrics = self._train_batch(inputs, targets, coverage_tracker)
            t_opt1 = time.time()
            # Measure timings
            optim_ms = (t_opt1 - t_opt0) * 1000.0
            step_ms = (time.time() - t_step0) * 1000.0
            total_optim_ms += optim_ms
            total_step_ms += step_ms
            # Approximate data/other time as the remainder (non-negative)
            total_data_ms += max(0.0, step_ms - optim_ms)
            n_batches += 1
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key in metrics and value is not None:
                    metrics[key].append(value)
            
            # Detect poles after batch if hybrid schedule is active
            if self.hybrid_schedule and self.hybrid_schedule.force_pole_exploration:
                # Get detected near-pole samples from this batch
                near_pole_indices = HybridGradientContext.detect_poles(
                    self.hybrid_schedule.pole_detection_threshold
                )
                
                # If we have near-pole samples, extract their x values
                if near_pole_indices and hasattr(self, '_last_batch_inputs'):
                    pole_x_values = []
                    for idx in near_pole_indices:
                        if idx < len(self._last_batch_inputs):
                            x_val = self._last_batch_inputs[idx]
                            if x_val.tag == TRTag.REAL:
                                pole_x_values.append(x_val.value)
                    
                    # Update schedule with detected poles
                    if pole_x_values:
                        self.hybrid_schedule.update_detected_poles(pole_x_values, self.epoch)
            
            # Log if needed
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                self._log_batch(batch_idx, len(data_loader), batch_metrics)
            
            self.global_step += 1
        
        # Compute epoch averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)

        # Bench averages (per batch)
        if n_batches > 0:
            avg_step_ms = total_step_ms / n_batches
            avg_optim_ms = total_optim_ms / n_batches
            avg_data_ms = total_data_ms / n_batches
            avg_metrics["avg_step_ms"] = avg_step_ms
            avg_metrics["optim_time_ms"] = avg_optim_ms
            avg_metrics["data_time_ms"] = avg_data_ms
            avg_metrics["batches"] = float(n_batches)
            # Persist bench record
            self.bench_history.append({
                'epoch': self.epoch,
                'avg_step_ms': avg_step_ms,
                'data_time_ms': avg_data_ms,
                'optim_time_ms': avg_optim_ms,
                'batches': n_batches,
            })
        
        # Apply coverage enforcement if enabled
        if self.coverage_policy and 'coverage' in avg_metrics:
            # Collect Q values if available
            Q_values = None
            if hasattr(self.model, 'q_min_history') and self.model.q_min_history:
                Q_values = self.model.q_min_history
            # Current lambda and near-pole coverage
            current_lambda = None
            try:
                if self.loss_policy and hasattr(self.loss_policy, 'adaptive_lambda'):
                    current_lambda = float(self.loss_policy.adaptive_lambda.get_penalty())
            except Exception:
                current_lambda = None
            near_cov = None
            try:
                if hasattr(coverage_tracker, 'near_pole_coverage'):
                    near_cov = coverage_tracker.near_pole_coverage
            except Exception:
                near_cov = None
            # Enforce coverage policy with correct arguments
            enforcement_actions = self.coverage_policy.enforce(
                avg_metrics['coverage'],
                current_lambda if current_lambda is not None else 0.0,
                near_cov,
                Q_values
            )
            
            # Update lambda if changed
            if enforcement_actions['lambda_updated']:
                new_lambda = enforcement_actions['new_lambda']
                if self.loss_policy:
                    self.loss_policy.adaptive_lambda.lambda_rej = new_lambda
                avg_metrics['lambda_rej'] = new_lambda
                avg_metrics['coverage_enforced'] = True
                
                # Log intervention if triggered
                if enforcement_actions['intervention_triggered']:
                    print(f"[Coverage Intervention] Epoch {self.epoch}: "
                          f"Coverage {avg_metrics['coverage']:.3f} < {self.hybrid_config.min_coverage:.3f}, "
                          f"Lambda reduced to {new_lambda:.3f}")
        
        # Evaluate anti-illusion metrics if enabled
        if (self.hybrid_config.enable_anti_illusion and 
            self.anti_illusion_metrics and 
            self.ground_truth_poles and 
            self.epoch % 5 == 0):  # Evaluate every 5 epochs
            
            try:
                illusion_metrics = self.anti_illusion_metrics.evaluate_model(
                    self.model,
                    self.ground_truth_poles,
                    x_range=self.hybrid_config.ple_x_range
                )
                
                # Add to average metrics
                for key, value in illusion_metrics.items():
                    if not math.isnan(value) and not math.isinf(value):
                        avg_metrics[f'ai_{key}'] = value
                
                # Log key metrics
                if self.epoch % 10 == 0:
                    print(f"  Anti-illusion: PLE={illusion_metrics.get('ple', float('inf')):.4f}, "
                          f"Sign={illusion_metrics.get('sign_consistency', 0):.3f}, "
                          f"Score={illusion_metrics.get('anti_illusion_score', float('inf')):.4f}")
            
            except Exception as e:
                print(f"  Warning: Anti-illusion evaluation failed: {e}")
        
        # Add hybrid-specific metrics
        if self.hybrid_schedule:
            hybrid_stats = HybridGradientContext.get_statistics()
            avg_metrics['saturating_ratio'] = hybrid_stats.get('saturating_ratio', 0.0)
            avg_metrics['gradient_mode'] = self.hybrid_schedule.get_mode_description(self.epoch)
            # Also carry epoch-level q/g quantiles if available
            for k in ('q_min_epoch', 'q_p10', 'q_p50', 'q_p90', 'g_p10', 'g_p50', 'g_p90'):
                if k in hybrid_stats and hybrid_stats.get(k) is not None:
                    avg_metrics[k] = hybrid_stats.get(k)
            # Copy policy thresholds if available
            if 'tau_q_on' in hybrid_stats or 'tau_q_off' in hybrid_stats:
                avg_metrics['tau_q_on'] = hybrid_stats.get('tau_q_on')
                avg_metrics['tau_q_off'] = hybrid_stats.get('tau_q_off')
            # Also export P-thresholds from the active TRPolicy, if any
            try:
                from ..policy import TRPolicyConfig
                pol = TRPolicyConfig.get_policy()
                if pol is not None:
                    # Use lower-case keys for consistency with tau_q_* in metrics
                    avg_metrics['tau_p_on'] = float(getattr(pol, 'tau_P_on', 0.0))
                    avg_metrics['tau_p_off'] = float(getattr(pol, 'tau_P_off', 0.0))
            except Exception:
                pass
            # Copy flip statistics to monitor finite/low-density switching
            if 'policy_flip_count' in hybrid_stats:
                avg_metrics['policy_flip_count'] = hybrid_stats.get('policy_flip_count')
            if 'flip_rate' in hybrid_stats:
                avg_metrics['flip_rate'] = hybrid_stats.get('flip_rate')
            # Mask bandwidth: mask_real_activations / total_gradient_calls
            try:
                m = hybrid_stats.get('mask_real_activations', None)
                t = hybrid_stats.get('total_gradient_calls', None)
                if isinstance(m, (int, float)) and isinstance(t, (int, float)) and t > 0:
                    avg_metrics['mask_bandwidth'] = float(m) / float(t)
            except Exception:
                pass

            # Track mode for history
            self.gradient_mode_history.append({
                'epoch': self.epoch,
                'mode': avg_metrics['gradient_mode'],
                'delta': self.hybrid_schedule.get_delta(self.epoch)
            })

        # Optional identifiability diagnostic (Sylvester s_min)
        try:
            from ..metrics import compute_sylvester_smin
            if hasattr(self.model, 'theta') and hasattr(self.model, 'phi'):
                smin = compute_sylvester_smin(self.model)
                if smin == smin:  # not NaN
                    avg_metrics['sylvester_smin'] = float(smin)
        except Exception:
            pass
        
        # Epoch-level GN/Fisher proxies derived from averaged grad stats
        try:
            import math as _math
            gn_avg = float(avg_metrics.get('gn_proxy', float('nan')))
            if gn_avg == gn_avg and gn_avg >= 0.0:
                # grad_norm_epoch is L2 norm across parameters (averaged batch proxy)
                avg_metrics['grad_norm_epoch'] = _math.sqrt(gn_avg)
                # Fisher trace proxy ≈ E[||g||^2]
                avg_metrics['fisher_trace'] = gn_avg
                # Mean diagonal Fisher proxy (per-parameter average)
                n_params = 0
                if hasattr(self.model, 'parameters'):
                    try:
                        n_params = len(list(self.model.parameters()))
                    except Exception:
                        n_params = 0
                if n_params > 0:
                    avg_metrics['fisher_diag_mean'] = gn_avg / float(n_params)
        except Exception:
            pass
        
        # Second-order safeguard: shrink LR when SAT dominates
        try:
            if (self.hybrid_config.sat_shrink_on_ratio and
                'saturating_ratio' in avg_metrics and
                isinstance(avg_metrics['saturating_ratio'], (int, float)) and
                avg_metrics['saturating_ratio'] >= self.hybrid_config.sat_ratio_threshold):
                factor = max(0.1, min(1.0, float(self.hybrid_config.sat_shrink_factor)))
                # Apply to all optimizers
                self.optimizer.learning_rate *= factor
                if self.frontend_optimizer is not None:
                    self.frontend_optimizer.learning_rate *= factor
                if self.head_optimizers is not None:
                    for opt in self.head_optimizers:
                        opt.learning_rate *= factor
                avg_metrics['lr_shrunk'] = factor
        except Exception:
            pass
        
        return avg_metrics

    def train(self,
              train_data: List[Tuple[List[TRNode], List[TRNode]]],
              val_data: Optional[List[Tuple[List[TRNode], List[TRNode]]]] = None) -> Dict[str, List[float]]:
        """
        Override to capture extended per-epoch metrics for plotting.
        """
        if self.config.verbose:
            print(f"Starting training for {self.config.max_epochs} epochs")
            if self.config.use_adaptive_loss:
                print(f"Using adaptive loss with target coverage: {self.config.target_coverage}")

        start_time = time.time()
        # Reset epoch records
        self._epoch_records = []

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch + 1

            # Train one epoch
            train_metrics = self.train_epoch(train_data)

            # Base history
            self.training_history.setdefault("loss", []).append(train_metrics.get("loss", float('nan')))
            if "coverage" in train_metrics:
                self.training_history.setdefault("coverage", []).append(train_metrics["coverage"])
            if "lambda_rej" in train_metrics:
                self.training_history.setdefault("lambda_rej", []).append(train_metrics["lambda_rej"])

            # Capture extended per-epoch metrics
            keep_keys = {
                'loss','coverage','lambda_rej','q_min','q_p10','q_p50','q_p90',
                'd_p10','d_p50','d_p90','g_p10','g_p50','g_p90','saturating_ratio',
                'flip_rate','sylvester_smin'
            }
            record: Dict[str, Any] = {'epoch': self.epoch}
            for k, v in train_metrics.items():
                if k in keep_keys and isinstance(v, (int,float)):
                    record[k] = v
            self._epoch_records.append(record)

            # Validation
            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                val_loss = val_metrics["loss"]
            else:
                val_loss = train_metrics.get("loss", float('inf'))

            # Logging
            if self.config.verbose:
                self._log_epoch(epoch, train_metrics, val_metrics if val_data else None)

            # Early stopping
            if self.config.early_stopping:
                if val_loss < self.best_loss - self.config.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        if self.config.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

        training_time = time.time() - start_time
        if self.config.verbose:
            print(f"Training completed in {training_time:.2f} seconds")

        return self.training_history

    def get_epoch_records(self) -> List[Dict[str, Any]]:
        """Return per-epoch metric records for plotting."""
        return list(self._epoch_records)

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                   val_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log epoch metrics with bench summary."""
        msg = f"Epoch {epoch + 1}/{self.config.max_epochs}"
        msg += f" - Train Loss: {train_metrics['loss']:.4f}"
        if "coverage" in train_metrics:
            msg += f" Coverage: {train_metrics['coverage']:.3f}"
        if "lambda_rej" in train_metrics:
            msg += f" λ_rej: {train_metrics['lambda_rej']:.3f}"
        if val_metrics:
            msg += f" - Val Loss: {val_metrics['loss']:.4f}"
            if "coverage" in val_metrics:
                msg += f" Coverage: {val_metrics['coverage']:.3f}"
        print(msg)
        # Bench line
        if all(k in train_metrics for k in ("avg_step_ms", "data_time_ms", "optim_time_ms", "batches")):
            print(
                f"Bench: avg_step_ms={train_metrics['avg_step_ms']:.1f}, "
                f"data_time_ms={train_metrics['data_time_ms']:.1f}, "
                f"optim_time_ms={train_metrics['optim_time_ms']:.1f}, "
                f"batches={int(train_metrics['batches'])}"
            )
        # Hybrid/controller summary (if available)
        extra = []
        if 'saturating_ratio' in train_metrics:
            extra.append(f"sat={train_metrics['saturating_ratio']:.3f}")
        if 'flip_rate' in train_metrics:
            extra.append(f"flip={train_metrics['flip_rate']:.3f}")
        if 'q_p10' in train_metrics and 'q_p90' in train_metrics:
            extra.append(f"q_p10={train_metrics['q_p10']:.2e} q_p90={train_metrics['q_p90']:.2e}")
        if 'g_p90' in train_metrics:
            extra.append(f"g_p90={train_metrics['g_p90']:.2e}")
        if 'sylvester_smin' in train_metrics:
            extra.append(f"smin={train_metrics['sylvester_smin']:.2e}")
        if extra:
            print("Hybrid:", " ".join(extra))
        # Curvature/Fisher summary (optional)
        if getattr(self.hybrid_config, 'log_curvature_fisher', True):
            curv = train_metrics.get('curvature_proxy', None)
            gn = train_metrics.get('grad_norm_epoch', None)
            ft = train_metrics.get('fisher_trace', None)
            cf_parts = []
            if isinstance(curv, (int, float)):
                cf_parts.append(f"L_hat={curv:.3e}")
            if isinstance(gn, (int, float)):
                cf_parts.append(f"||g||={gn:.3e}")
            if isinstance(ft, (int, float)):
                cf_parts.append(f"F_tr={ft:.3e}")
            if cf_parts:
                print("Curv:", " ".join(cf_parts))
    
    def _train_batch(self,
                    inputs: List[TRScalar],
                    targets: List[TRScalar],
                    coverage_tracker: CoverageTracker) -> Dict[str, float]:
        
        # Store inputs for pole detection
        self._last_batch_inputs = inputs
        """
        Train on a single batch with hybrid features.
        
        Args:
            inputs: Batch inputs
            targets: Batch targets
            coverage_tracker: Coverage tracking instance
            
        Returns:
            Batch metrics
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = []
        pole_scores = []
        tags = []
        all_tag_logits = []
        Q_values = []
        
        for x in inputs:
            # Check if model supports full integration
            if hasattr(self.model, 'forward_fully_integrated'):
                # Fully integrated model with all features
                result = self.model.forward_fully_integrated(x)
                predictions.append(result['output'])
                tags.append(result['tag'])
                if 'tag_logits' in result:
                    all_tag_logits.append(result['tag_logits'])
                if 'pole_score' in result:
                    pole_scores.append(result['pole_score'])
                if 'Q_abs' in result:
                    Q_values.append(result['Q_abs'])
            # Check if model supports tag prediction
            elif self.hybrid_config.use_tag_loss and hasattr(self.model, 'forward_with_tag_pred'):
                y, tag, tag_logits = self.model.forward_with_tag_pred(x)
                predictions.append(y)
                tags.append(tag)
                if tag_logits:
                    all_tag_logits.append(tag_logits)
            # Check if model supports pole head
            elif self.hybrid_config.use_pole_head and hasattr(self.model, 'forward_with_pole_score'):
                y, tag, pole_score = self.model.forward_with_pole_score(x)
                predictions.append(y)
                pole_scores.append(pole_score)
                tags.append(tag)
                # Try to get Q value if available
                if hasattr(self.model, 'get_Q_value'):
                    q_val = self.model.get_Q_value()
                    if q_val is not None:
                        Q_values.append(q_val)
            else:
                y = self.model(x)
                predictions.append(y)
                tags.append(y.tag)
        
        # Track coverage with Q / distance values if enhanced
        if self.hybrid_config.use_enhanced_coverage:
            # Extract Q values and x values if available
            q_values_for_tracking = None
            x_values_for_tracking = None
            d_values_for_tracking = None
            
            if Q_values:
                q_values_for_tracking = [abs(q) for q in Q_values]
            
            if hasattr(self, '_last_batch_inputs'):
                x_values_for_tracking = []
                for x in self._last_batch_inputs:
                    if x.tag == TRTag.REAL:
                        x_values_for_tracking.append(x.value)
                    else:
                        x_values_for_tracking.append(None)

            # If model exposes distance estimator, compute distances for quotas
            try:
                if hasattr(self.model, 'estimate_distance_batch'):
                    d_values_for_tracking = self.model.estimate_distance_batch(self._last_batch_inputs)
            except Exception:
                d_values_for_tracking = None
            
            # Update with Q/x/d values when tracker supports it; fallback otherwise
            try:
                coverage_tracker.update(tags, q_values_for_tracking, x_values_for_tracking, d_values=d_values_for_tracking)  # type: ignore[arg-type]
            except TypeError:
                try:
                    # Try legacy enhanced signature without d_values
                    coverage_tracker.update(tags, q_values_for_tracking, x_values_for_tracking)  # type: ignore[arg-type]
                except TypeError:
                    # Basic CoverageTracker signature
                    coverage_tracker.update(tags)
        else:
            coverage_tracker.update(tags)
        
        # Store coverage for tag loss adaptive weighting
        self._last_coverage = coverage_tracker.batch_coverage
        
        # Compute main loss
        if self.loss_policy:
            # Pass tag logits to loss policy for integrated tag loss
            batch_loss = self.loss_policy.compute_batch_loss(
                predictions, targets, 
                tag_logits=all_tag_logits if self.hybrid_config.use_tag_loss else None
            )
        else:
            # Simple MSE loss
            from ..core import tr_sum, tr_div
            losses = []
            for pred, target in zip(predictions, targets):
                if pred.tag == TRTag.REAL:
                    diff = pred - to_trnode_constant(target)
                    loss = TRNode.constant(real(0.5)) * diff * diff
                else:
                    # Use minimum rejection penalty if configured
                    min_lambda = getattr(self.config, 'lambda_rej_min', 0.1)
                    lambda_val = max(self.config.initial_lambda, min_lambda)
                    loss = TRNode.constant(real(lambda_val))
                losses.append(loss)
            
            total = tr_sum([loss_node.value for loss_node in losses])
            batch_loss = TRNode.constant(tr_div(total, real(float(len(losses)))))
        
        # Add tag loss if enabled and not using adaptive loss policy
        tag_loss_value = 0.0
        if self.hybrid_config.use_tag_loss and not self.loss_policy:
            tag_loss = self._compute_tag_loss(predictions, all_tag_logits)
            if tag_loss is not None:
                batch_loss = batch_loss + tag_loss  # Already weighted in compute_tag_loss
        
        # Compute pole detection loss if enabled
        if self.hybrid_config.use_pole_head and pole_scores:
            pole_loss = self._compute_pole_loss(
                predictions, pole_scores, Q_values, inputs
            )
            if pole_loss is not None:
                weighted_pole_loss = tr_mul(
                    TRNode.constant(real(self.hybrid_config.lambda_pole)),
                    pole_loss
                )
                batch_loss = batch_loss + weighted_pole_loss
                pole_loss_value = pole_loss.value.value if pole_loss.value.tag == TRTag.REAL else 0.0
        else:
            pole_loss_value = 0.0
        
        # Compute residual consistency loss if enabled
        residual_loss_value = 0.0
        if self.hybrid_config.enable_anti_illusion and self.residual_loss:
            # Extract input values for residual computation
            input_vals = []
            for x in inputs:
                if x.value.tag == TRTag.REAL:
                    input_vals.append(x.value.value)
            
            if input_vals:
                residual_loss = self.residual_loss.compute_loss(
                    self.model, input_vals, 
                    self.hybrid_config.residual_near_pole_threshold
                )
                batch_loss = batch_loss + residual_loss
                residual_loss_value = (
                    residual_loss.value.value if residual_loss.tag == TRTag.REAL else 0.0
                )
        
        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            reg_loss = self.model.regularization_loss()
            batch_loss = batch_loss + reg_loss
        
        # Backward pass
        # Zero grads on all persistent optimizers
        self.zero_grad_all()
        batch_loss.backward()
        
        # Optional safe LR clamp across optimizers
        try:
            if getattr(self.config, 'use_safe_lr', False):
                # Compute safe LR using batch stats
                q_min = None
                if hasattr(self.model, 'compute_q_min'):
                    q_min = self.model.compute_q_min(inputs)
                Bpsi = getattr(getattr(self.model, 'basis', None), 'bound', None)
                if q_min is not None and Bpsi is not None:
                    y_vals = [p.value.value for p in predictions if p.tag == TRTag.REAL]
                    y_max = max([abs(v) for v in y_vals], default=0.0)
                    alpha = getattr(self.model, 'alpha_phi', 0.0) or 0.0
                    L_hat = (Bpsi * Bpsi) / (max(q_min, 1e-12) ** 2) * (1.0 + y_max * y_max) + alpha
                    eta_safe = 1.0 / max(L_hat, 1e-12)
                    # Clamp all optimizers
                    self.optimizer.learning_rate = min(self.optimizer.learning_rate, eta_safe)
                    if self.frontend_optimizer is not None:
                        self.frontend_optimizer.learning_rate = min(self.frontend_optimizer.learning_rate, eta_safe)
                    if self.head_optimizers is not None:
                        for opt in self.head_optimizers:
                            opt.learning_rate = min(opt.learning_rate, eta_safe)
        except Exception:
            pass

        # Curvature proxy and gradient-norm proxies (logging only)
        try:
            # Curvature proxy L_hat ≈ (B_psi^2 / q_min^2) * (1 + y_max^2) + alpha
            q_min = None
            if hasattr(self.model, 'compute_q_min'):
                try:
                    q_min = float(self.model.compute_q_min(inputs))
                except Exception:
                    q_min = None
            Bpsi = getattr(getattr(self.model, 'basis', None), 'bound', None)
            if isinstance(Bpsi, (int, float)) and Bpsi is not None:
                y_vals = [abs(float(p.value.value)) for p in predictions if p.tag == TRTag.REAL]
                y_max = max(y_vals) if y_vals else 0.0
                alpha = float(getattr(self.model, 'alpha_phi', 0.0) or 0.0)
                denom = max(abs(q_min) if q_min is not None else 0.0, 1e-12)
                L_hat = (Bpsi * Bpsi) / (denom * denom) * (1.0 + y_max * y_max) + alpha
                curvature_proxy_val = float(L_hat)
            else:
                curvature_proxy_val = float('nan')
        except Exception:
            curvature_proxy_val = float('nan')

        # Gradient norm proxies after backward
        gn_sq = 0.0
        gmax = 0.0
        try:
            params = []
            if hasattr(self.model, 'parameters'):
                params = list(self.model.parameters())
            for p in params:
                g = getattr(p, 'gradient', None)
                if g is not None and g.tag == TRTag.REAL:
                    gv = float(g.value)
                    gn_sq += gv * gv
                    gmax = max(gmax, abs(gv))
        except Exception:
            pass

        # Optimizer step (supports per-head + frontend)
        self.step_all()
        
        # Collect metrics
        metrics = {
            'loss': batch_loss.value.value if batch_loss.value.tag == TRTag.REAL else float('inf'),
            'coverage': coverage_tracker.batch_coverage,
            'tag_loss': tag_loss_value,
            'pole_loss': pole_loss_value,
            'residual_loss': residual_loss_value,
            # Logging-only curvature proxies
            'curvature_proxy': curvature_proxy_val,
            'gn_proxy': gn_sq,
            'grad_max': gmax,
        }

        # Optional: second-order curvature bound (safeguard envelope)
        try:
            if hasattr(self.model, 'get_q_values'):
                from ..optim_utils_second_order import curvature_bound_for_batch
                cb = curvature_bound_for_batch(self.model, inputs)
                metrics['curvature_bound'] = cb.get('curvature_bound')
                metrics['B_k'] = cb.get('B_k')
                metrics['H_k'] = cb.get('H_k')
                metrics['G_max'] = cb.get('G_max')
                metrics['H_max'] = cb.get('H_max')
        except Exception:
            pass

        # Contract-safe LR clamp using layer contract (theory: eta <= c / (beta * Π max{B_k,G_max}))
        try:
            if self.hybrid_config.use_contract_safe_lr and hasattr(self.model, 'get_layer_contract'):
                contract = self.model.get_layer_contract()  # type: ignore[attr-defined]
                B_k = float(contract.get('B_k', 1.0))
                G_max = float(contract.get('G_max', 1.0))
                depth = int(contract.get('depth_hint', 4))
                beta = float(self.hybrid_config.loss_smoothness_beta)
                c_const = float(self.hybrid_config.contract_c)
                prod = max(B_k, G_max) ** max(1, depth)
                eta_contract = c_const / max(1e-12, beta * prod)
                # Clamp all optimizers
                self.optimizer.learning_rate = min(self.optimizer.learning_rate, eta_contract)
                if self.frontend_optimizer is not None:
                    self.frontend_optimizer.learning_rate = min(self.frontend_optimizer.learning_rate, eta_contract)
                if self.head_optimizers is not None:
                    for opt in self.head_optimizers:
                        opt.learning_rate = min(opt.learning_rate, eta_contract)
                metrics['eta_contract'] = eta_contract
        except Exception:
            pass
        
        # Add hybrid metrics and update policy-driven hysteresis at batch end
        if self.hybrid_schedule:
            hybrid_stats = HybridGradientContext.get_statistics()
            metrics['near_pole_ratio'] = hybrid_stats.get('near_pole_ratio', 0.0)
            # Copy quantiles if present
            for k in ('q_p10', 'q_p50', 'q_p90', 'q_min_batch', 'q_mean_batch', 'q_median_batch'):
                if k in hybrid_stats:
                    metrics[k] = hybrid_stats.get(k)
            # Sensitivity quantiles if present (g = 1/|Q|)
            for k in ('g_p10', 'g_p50', 'g_p90'):
                if k in hybrid_stats:
                    metrics[k] = hybrid_stats.get(k)
            # Copy policy mode and thresholds if available
            if 'policy_mode' in hybrid_stats:
                metrics['policy_mode'] = hybrid_stats.get('policy_mode')
            if 'tau_q_on' in hybrid_stats or 'tau_q_off' in hybrid_stats:
                metrics['tau_q_on'] = hybrid_stats.get('tau_q_on')
                metrics['tau_q_off'] = hybrid_stats.get('tau_q_off')
            # Copy saturating counters for clarity
            if 'saturating_activations' in hybrid_stats:
                metrics['saturating_activations'] = hybrid_stats.get('saturating_activations')
            if 'total_gradient_calls' in hybrid_stats:
                metrics['total_gradient_calls'] = hybrid_stats.get('total_gradient_calls')
            if 'saturating_ratio' in hybrid_stats:
                metrics['saturating_ratio'] = hybrid_stats.get('saturating_ratio')
            if 'policy_flip_count' in hybrid_stats:
                metrics['policy_flip_count'] = hybrid_stats.get('policy_flip_count')
            if 'flip_rate' in hybrid_stats:
                metrics['flip_rate'] = hybrid_stats.get('flip_rate')
            # Update hybrid mode using policy hysteresis and reset batch stats
            HybridGradientContext.end_batch_policy_update()

        # If model exposes Q and/or distance estimators, add per-batch q/d quantiles when not using hybrid stats
        try:
            if 'q_p10' not in metrics and hasattr(self.model, 'get_q_values'):
                from ..metrics import compute_q_stats
                q_stats = compute_q_stats(self.model, inputs)
                metrics.update(q_stats)
        except Exception:
            pass
        try:
            if hasattr(self.model, 'estimate_distance_batch'):
                from ..metrics import compute_distance_stats
                d_stats = compute_distance_stats(self.model, inputs)
                metrics.update(d_stats)
        except Exception:
            pass
        
        # Add adaptive lambda if using policy
        if self.loss_policy:
            metrics['lambda_rej'] = self.loss_policy.adaptive_lambda.get_penalty()
        
        return metrics

    def _train_batch_multi(self,
                           batch_inputs: List[List[TRNode]],
                           batch_targets: List[List[float]]) -> Dict[str, float]:
        """Mini-batch training for multi-input, multi-output models.

        Aggregates loss across samples and outputs, performs a single
        backward pass and a single optimizer step using persistent
        per-head/front-end optimizers when available.

        Args:
            batch_inputs: List of input vectors (each a list of TRNodes)
            batch_targets: List of target vectors (floats)

        Returns:
            Dict with 'loss' and 'optim_ms' timing.
        """
        t0 = time.time()
        self.zero_grad_all()

        # Collect per-sample losses to avoid deep linear graphs that can
        # blow recursion limits during backprop. We'll reduce them via a
        # balanced pairwise tree.
        sample_losses: List[TRNode] = []
        valid_samples = 0

        for tr_inp, tr_tgt in zip(batch_inputs, batch_targets):
            try:
                # Forward: model returns List[(TRNode, TRTag)]
                outs = self.model.forward(tr_inp)
            except TypeError:
                # Fallback: treat as single-input model
                y, tag = self.model.forward(tr_inp[0])
                outs = [(y, tag)]

            # Per-sample averaged loss over REAL outputs
            sample_loss = TRNode.constant(real(0.0))
            valid = 0
            for j, (out, tag) in enumerate(outs):
                target_val = tr_tgt[j] if j < len(tr_tgt) else 0.0
                if tag == TRTag.REAL:
                    diff = out - TRNode.constant(real(float(target_val)))
                    sample_loss = sample_loss + diff * diff
                    valid += 1
            if valid == 0:
                continue
            sample_loss = sample_loss / TRNode.constant(real(float(valid)))
            sample_losses.append(sample_loss)
            valid_samples += 1

        if valid_samples == 0:
            return {'loss': float('inf'), 'optim_ms': 0.0}

        # Balanced pairwise sum to reduce graph depth
        def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
            if not nodes:
                return TRNode.constant(real(0.0))
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = _pairwise_sum(nodes[:mid])
            right = _pairwise_sum(nodes[mid:])
            return left + right

        total_loss = _pairwise_sum(sample_losses) / TRNode.constant(real(float(valid_samples)))

        # Add regularization if available
        if hasattr(self.model, 'regularization_loss'):
            reg = self.model.regularization_loss()
            total_loss = total_loss + reg

        # Backward and step
        total_loss.backward()
        self.step_all()

        t1 = time.time()
        loss_val = total_loss.value.value if total_loss.value.tag == TRTag.REAL else float('inf')
        return {'loss': loss_val, 'optim_ms': (t1 - t0) * 1000.0}
    
    def _compute_tag_loss(self, 
                         predictions: List[TRNode],
                         tag_logits: List[List[TRNode]]) -> Optional[TRNode]:
        """
        Compute auxiliary loss for tag classification.
        
        This encourages the model to correctly predict the type
        of singularity (PINF vs NINF vs PHI).
        
        Args:
            predictions: Model predictions (for true tags)
            tag_logits: Predicted tag logits from tag head
            
        Returns:
            Tag classification loss or None
        """
        if not tag_logits:
            # Fallback to simple penalty if no tag head
            tags = [pred.tag for pred in predictions]
            non_real_count = sum(1 for tag in tags if tag != TRTag.REAL)
            if non_real_count > 0:
                penalty = float(non_real_count) / len(tags)
                return TRNode.constant(real(penalty))
            return None
        
        # Use proper tag loss computation with adaptive weighting
        from ..training.tag_loss import compute_tag_loss
        
        # Get current coverage for adaptive weighting
        coverage = None
        if hasattr(self, '_last_coverage'):
            coverage = self._last_coverage
        
        # Use adaptive weight if configured
        adaptive = getattr(self.hybrid_config, 'tag_loss_adaptive', True)
        
        return compute_tag_loss(predictions, tag_logits, 
                               weight=self.hybrid_config.lambda_tag,
                               adaptive_weight=adaptive,
                               coverage=coverage)
    
    def _compute_pole_loss(self,
                          predictions: List[TRNode],
                          pole_scores: List[TRNode],
                          Q_values: List[float],
                          inputs: List[TRNode]) -> Optional[TRNode]:
        """
        Compute pole detection loss.
        
        Args:
            predictions: Model outputs
            pole_scores: Predicted pole scores
            Q_values: Absolute Q values for self-supervision
            inputs: Input values for teacher signals
            
        Returns:
            Pole detection loss or None
        """
        if not pole_scores:
            return None
        
        # Get teacher labels if available
        teacher_labels = None
        if self.pole_detector and self.hybrid_config.use_teacher_signals:
            # Extract scalar values from inputs
            input_vals = []
            for x in inputs:
                if x.value.tag == TRTag.REAL:
                    input_vals.append(x.value.value)
                else:
                    input_vals.append(0.0)  # Default for non-REAL
            
            teacher_labels = self.pole_detector.generate_labels(input_vals)
        
        # Compute pole loss using the proper function
        pole_loss = compute_pole_loss(
            predictions,
            pole_scores,
            Q_values if Q_values else None,
            teacher_labels,
            self.hybrid_config.pole_config
        )
        
        return pole_loss
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary including hybrid metrics.
        
        Returns:
            Dictionary with training statistics
        """
        summary = {
            'epochs_trained': self.epoch,
            'global_steps': self.global_step,
            'final_metrics': self.history,
            'bench_history': self.bench_history
        }
        # Attach active policy thresholds (if any) for ease of inspection
        try:
            from ..policy import TRPolicyConfig
            pol = TRPolicyConfig.get_policy()
            if pol is not None:
                summary['policy_thresholds'] = {
                    'tau_q_on': float(getattr(pol, 'tau_Q_on', 0.0)),
                    'tau_q_off': float(getattr(pol, 'tau_Q_off', 0.0)),
                    'tau_p_on': float(getattr(pol, 'tau_P_on', 0.0)),
                    'tau_p_off': float(getattr(pol, 'tau_P_off', 0.0)),
                }
        except Exception:
            pass
        # Publish layer contract if available (B_k, H_k, G_max, H_max)
        try:
            if hasattr(self.model, 'get_layer_contract'):
                summary['layer_contract'] = self.model.get_layer_contract()
        except Exception:
            pass
        
        # Add hybrid gradient history
        if self.gradient_mode_history:
            summary['gradient_modes'] = self.gradient_mode_history
            
            # Analyze transition
            warmup_end = next((i for i, m in enumerate(self.gradient_mode_history) 
                             if 'transitioning' in m['mode']), -1)
            transition_end = next((i for i, m in enumerate(self.gradient_mode_history)
                                 if 'converged' in m['mode']), -1)
            
            summary['warmup_epochs'] = warmup_end if warmup_end >= 0 else self.epoch
            summary['transition_complete'] = transition_end if transition_end >= 0 else None
        
        # Add model-specific metrics
        if hasattr(self.model, 'get_hybrid_statistics'):
            summary['model_statistics'] = self.model.get_hybrid_statistics()
        
        # Add coverage enforcement statistics
        if self.coverage_policy:
            summary['coverage_enforcement'] = self.coverage_policy.get_statistics()
        
        # Add anti-illusion metrics history
        if self.anti_illusion_metrics:
            summary['anti_illusion_trends'] = self.anti_illusion_metrics.get_trends()
            if self.anti_illusion_metrics.evaluation_history:
                latest = self.anti_illusion_metrics.evaluation_history[-1]
                summary['latest_anti_illusion'] = latest
        
        return summary
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint including hybrid state."""
        import pickle
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'history': self.history,
            'gradient_mode_history': self.gradient_mode_history,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'hybrid_schedule': self.hybrid_schedule
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint including hybrid state."""
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.gradient_mode_history = checkpoint.get('gradient_mode_history', [])
        
        if 'hybrid_schedule' in checkpoint:
            self.hybrid_schedule = checkpoint['hybrid_schedule']
            if hasattr(self.model, 'hybrid_schedule'):
                self.model.hybrid_schedule = self.hybrid_schedule
        
        # Restore model and optimizer states
        self._set_model_state(checkpoint['model_state'])
        self._set_optimizer_state(checkpoint['optimizer_state'])
        
        # Update hybrid context
        if self.hybrid_schedule:
            HybridGradientContext.set_schedule(self.hybrid_schedule)
            HybridGradientContext.update_epoch(self.epoch)
    
    def _get_model_state(self) -> Dict:
        """Get model state for checkpointing."""
        state = {}
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                state[f'param_{i}'] = param.value
        return state
    
    def _set_model_state(self, state: Dict) -> None:
        """Set model state from checkpoint."""
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                if f'param_{i}' in state:
                    param._value = state[f'param_{i}']
    
    def _get_optimizer_state(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'learning_rate': self.optimizer.learning_rate,
            'step_count': self.optimizer.step_count
        }
    
    def _set_optimizer_state(self, state: Dict) -> None:
        """Set optimizer state from checkpoint."""
        self.optimizer.learning_rate = state['learning_rate']
        self.optimizer.step_count = state['step_count']
