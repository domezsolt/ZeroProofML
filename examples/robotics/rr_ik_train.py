"""
RR-arm inverse kinematics training with ZeroProofML.

This script trains different models on the RR robot IK dataset,
comparing standard approaches with TR-enhanced methods.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import (
    FullyIntegratedRational,
    MonomialBasis,
    TRMultiInputRational,
    TRRational,
)
from zeroproof.metrics.pole_2d import compute_ple_to_lines, compute_pole_metrics_2d
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.training.pole_detection import binary_cross_entropy
from zeroproof.utils.metrics import AntiIllusionMetrics, PoleLocation
from zeroproof.utils.seeding import set_global_seed

# Support running both as a module (python -m examples.robotics.rr_ik_train)
# and as a script (python examples/robotics/rr_ik_train.py)
try:
    from .rr_ik_dataset import IKSample, RobotConfig, RRDatasetGenerator
except ImportError:  # Fallback when not executed as a package
    from rr_ik_dataset import IKSample, RobotConfig, RRDatasetGenerator


@dataclass
class TrainingConfig:
    """Configuration for IK training."""

    model_type: str = "tr_rat"  # "mlp", "rat_eps", "tr_rat"

    # Model architecture
    hidden_dim: int = 32
    n_layers: int = 2
    degree_p: int = 3
    degree_q: int = 2

    # Training parameters
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32

    # ZeroProof enhancements
    use_hybrid_schedule: bool = True
    use_tag_loss: bool = True
    use_pole_head: bool = True
    use_residual_loss: bool = True
    enable_anti_illusion: bool = True

    # Loss weights
    lambda_tag: float = 0.05
    lambda_pole: float = 0.1
    lambda_residual: float = 0.02

    # Teacher supervision for pole head
    supervise_pole_head: bool = False
    teacher_pole_threshold: float = 0.1

    # Coverage control
    enforce_coverage: bool = True
    min_coverage: float = 0.7

    # Evaluation
    evaluate_ple: bool = True
    track_convergence: bool = True
    # Logging cadence (epochs)
    log_every: int = 1
    # Performance toggles passed into HybridTrainingConfig
    no_structured_logging: bool = False
    no_plots: bool = False

    # Baseline-specific knobs
    epsilon: Optional[float] = None  # For Rational+ε (P/(Q+ε)) baseline
    clip_grad: Optional[float] = None  # Global-norm gradient clipping for baselines

    # Policy / hybrid knobs
    use_tr_policy: bool = True
    policy_ulp_scale: float = 4.0
    policy_deterministic_reduction: bool = True
    hybrid_aggressive: bool = False
    hybrid_delta_init: float = 1e-2
    hybrid_delta_final: float = 1e-6
    # Contract-safe LR clamp
    use_contract_safe_lr: bool = False
    contract_c: float = 1.0
    loss_smoothness_beta: float = 1.0
    # Coprime surrogate regularizer
    enable_coprime_regularizer: bool = False
    lambda_coprime: float = 0.0


class MLPBaseline:
    """Standard MLP baseline for comparison."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, n_layers: int = 2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Simple MLP implementation using TRNodes for consistency
        self.layers = []

        # Input layer
        prev_dim = input_dim
        for i in range(n_layers):
            layer_weights = []
            layer_biases = []

            curr_dim = hidden_dim if i < n_layers - 1 else output_dim

            # Initialize weights
            scale = np.sqrt(2.0 / prev_dim)
            for j in range(curr_dim):
                weights = []
                for k in range(prev_dim):
                    w = np.random.randn() * scale
                    weights.append(TRNode.parameter(real(w), name=f"W{i}_{j}_{k}"))
                layer_weights.append(weights)

                b = TRNode.parameter(real(0.0), name=f"b{i}_{j}")
                layer_biases.append(b)

            self.layers.append((layer_weights, layer_biases))
            prev_dim = curr_dim

    def forward(self, x: List[TRNode]) -> List[TRNode]:
        """Forward pass through MLP."""
        current = x

        for i, (weights, biases) in enumerate(self.layers):
            next_layer = []

            for j, (weight_row, bias) in enumerate(zip(weights, biases)):
                # Linear combination
                activation = bias
                for w, inp in zip(weight_row, current):
                    activation = activation + w * inp

                # ReLU activation (except last layer)
                if i < len(self.layers) - 1:
                    if activation.tag == TRTag.REAL and activation.value.value > 0:
                        next_layer.append(activation)
                    else:
                        next_layer.append(TRNode.constant(real(0.0)))
                else:
                    next_layer.append(activation)

            current = next_layer

        return current

    def parameters(self) -> List[TRNode]:
        """Get all parameters."""
        params = []
        for weights, biases in self.layers:
            for weight_row in weights:
                params.extend(weight_row)
            params.extend(biases)
        return params


class RationalEpsBaseline:
    """Rational function with epsilon regularization baseline."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree_p: int = 3,
        degree_q: int = 2,
        eps: float = 1e-4,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps

        basis = MonomialBasis()

        # Create multiple rational functions for multi-output
        self.rationals = []
        for _ in range(output_dim):
            rational = TRRational(degree_p, degree_q, basis)
            self.rationals.append(rational)

    def _forward_one_with_eps(self, rational: TRRational, x_node: TRNode) -> TRNode:
        """Compute y = P(x)/(Q(x)+ε) using θ, φ and the model basis."""
        # Build basis up to max degree
        max_degree = max(len(rational.theta), len(rational.phi) + 1)
        psi = rational.basis(x_node, max_degree)

        # P(x) = Σ_k θ_k ψ_k(x)
        P = TRNode.constant(real(0.0))
        for k, theta_k in enumerate(rational.theta):
            if k < len(psi):
                P = P + theta_k * psi[k]

        # Q(x) = 1 + Σ_k φ_k ψ_{k}(x); add ε shift
        Q = TRNode.constant(real(1.0 + float(self.eps)))
        for k, phi_k in enumerate(rational.phi):
            idx = k + 1
            if idx < len(psi):
                Q = Q + phi_k * psi[idx]

        return P / Q

    def forward(self, x: List[TRNode]) -> List[TRNode]:
        """Forward pass with explicit epsilon regularization (P/(Q+ε))."""
        x_input = x[0] if x else TRNode.constant(real(0.0))
        outputs = []
        for rational in self.rationals:
            outputs.append(self._forward_one_with_eps(rational, x_input))
        return outputs

    def parameters(self) -> List[TRNode]:
        """Get all parameters."""
        params = []
        for rational in self.rationals:
            params.extend(rational.parameters())
        return params


class IKTrainer:
    """Trainer for inverse kinematics problems."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.training_history = []
        self.bench_history = []  # per-epoch bench metrics
        self.pole_optimizer = None
        self.ple_history = []

    def create_model(self, input_dim: int, output_dim: int):
        """Create model based on configuration."""
        if self.config.model_type == "mlp":
            self.model = MLPBaseline(
                input_dim, output_dim, self.config.hidden_dim, self.config.n_layers
            )

        elif self.config.model_type == "rat_eps":
            eps = (
                self.config.epsilon
                if isinstance(self.config.epsilon, (int, float)) and self.config.epsilon is not None
                else 1e-4
            )
            self.model = RationalEpsBaseline(
                input_dim, output_dim, self.config.degree_p, self.config.degree_q, eps=float(eps)
            )

        elif self.config.model_type == "tr_rat":
            basis = MonomialBasis()
            # If problem is multi-output (2D joint deltas), use multi-input TR model
            if output_dim > 1:
                # Small TR-MLP front end that consumes full input vector
                hidden = (
                    max(4, int(self.config.hidden_dim)) if hasattr(self.config, "hidden_dim") else 8
                )
                self.model = TRMultiInputRational(
                    input_dim=input_dim,
                    n_outputs=output_dim,
                    d_p=self.config.degree_p,
                    d_q=self.config.degree_q,
                    basis=basis,
                    hidden_dims=[hidden],
                    shared_Q=True,
                    enable_pole_head=self.config.use_pole_head,
                    enable_coprime_regularizer=self.config.enable_coprime_regularizer,
                    lambda_coprime=self.config.lambda_coprime,
                )
            else:
                # Single output rational
                if (
                    self.config.use_tag_loss
                    or self.config.use_pole_head
                    or self.config.enable_anti_illusion
                ):
                    self.model = FullyIntegratedRational(
                        d_p=self.config.degree_p,
                        d_q=self.config.degree_q,
                        basis=basis,
                        enable_tag_head=self.config.use_tag_loss,
                        enable_pole_head=self.config.use_pole_head,
                        track_Q_values=True,
                    )
                else:
                    self.model = TRRational(
                        d_p=self.config.degree_p,
                        d_q=self.config.degree_q,
                        basis=basis,
                        enable_coprime_regularizer=self.config.enable_coprime_regularizer,
                        lambda_coprime=self.config.lambda_coprime,
                    )

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def setup_trainer(self):
        """Setup the trainer with ZeroProof enhancements."""
        if self.config.model_type != "tr_rat":
            # Use simple optimizer for baselines
            self.optimizer = Optimizer(
                self.model.parameters(), learning_rate=self.config.learning_rate
            )
            return

        # ZeroProof enhanced training
        hybrid_config = HybridTrainingConfig(
            learning_rate=self.config.learning_rate,
            max_epochs=self.config.epochs,  # map local 'epochs' to base TrainingConfig field
            # Hybrid gradient schedule
            use_hybrid_gradient=self.config.use_hybrid_schedule,
            hybrid_warmup_epochs=max(1, self.config.epochs // 5),
            hybrid_transition_epochs=max(1, self.config.epochs // 3),
            # Policy-driven hybrid hysteresis (enable by default)
            use_tr_policy=self.config.use_tr_policy,
            policy_ulp_scale=self.config.policy_ulp_scale,
            policy_deterministic_reduction=self.config.policy_deterministic_reduction,
            # Tag loss
            use_tag_loss=self.config.use_tag_loss,
            lambda_tag=self.config.lambda_tag,
            # Pole detection
            use_pole_head=self.config.use_pole_head,
            lambda_pole=self.config.lambda_pole,
            # Anti-illusion metrics
            enable_anti_illusion=self.config.enable_anti_illusion,
            lambda_residual=self.config.lambda_residual,
            # Coverage control
            enforce_coverage=self.config.enforce_coverage,
            min_coverage=self.config.min_coverage,
            log_interval=self.config.log_every,
            enable_structured_logging=not getattr(self.config, "no_structured_logging", False),
            save_plots=not getattr(self.config, "no_plots", False),
            # Optional: schedule aggressiveness and deltas
            hybrid_aggressive=self.config.hybrid_aggressive,
            hybrid_delta_init=self.config.hybrid_delta_init,
            hybrid_delta_final=self.config.hybrid_delta_final,
            # Contract-safe LR clamp
            use_contract_safe_lr=self.config.use_contract_safe_lr,
            contract_c=self.config.contract_c,
            loss_smoothness_beta=self.config.loss_smoothness_beta,
        )

        # Handle multi-output case
        if isinstance(self.model, list):
            # Train each output separately for simplicity
            self.trainers = []
            for model in self.model:
                trainer = HybridTRTrainer(
                    model=model,
                    optimizer=Optimizer(
                        model.parameters(), learning_rate=self.config.learning_rate
                    ),
                    config=hybrid_config,
                )
                self.trainers.append(trainer)
        else:
            self.trainer = HybridTRTrainer(
                model=self.model,
                optimizer=Optimizer(
                    self.model.parameters(), learning_rate=self.config.learning_rate
                ),
                config=hybrid_config,
            )

        # Initialize a dedicated optimizer for the simple pole head, if requested
        if (
            self.config.model_type == "tr_rat"
            and self.config.use_pole_head
            and self.config.supervise_pole_head
            and hasattr(self.model, "pole_parameters")
            and callable(getattr(self.model, "pole_parameters"))
        ):
            pole_params = self.model.pole_parameters()
            if pole_params:
                self.pole_optimizer = Optimizer(
                    pole_params, learning_rate=self.config.learning_rate
                )

    def prepare_data(self, samples: List[IKSample]) -> Tuple[List, List]:
        """Prepare training data from IK samples."""
        inputs = []
        targets = []

        for sample in samples:
            # Input: current configuration + desired displacement
            # This is a common formulation for differential IK
            input_vec = [sample.theta1, sample.theta2, sample.dx, sample.dy]

            # Target: joint displacement
            target_vec = [sample.dtheta1, sample.dtheta2]

            inputs.append(input_vec)
            targets.append(target_vec)

        return inputs, targets

    def train(self, train_samples: List[IKSample], val_samples: Optional[List[IKSample]] = None):
        """Train the model on IK data."""
        print(f"Training {self.config.model_type} model...")
        print(f"Training samples: {len(train_samples)}")
        if val_samples:
            print(f"Validation samples: {len(val_samples)}")

        # Prepare data
        train_inputs, train_targets = self.prepare_data(train_samples)
        self.val_inputs = None
        self.val_targets = None
        if val_samples is not None:
            self.val_inputs, self.val_targets = self.prepare_data(val_samples)

        # Set gradient mode
        GradientModeConfig.set_mode(GradientMode.MASK_REAL)

        start_time = time.time()

        # ZeroProof path supports both single-output (self.trainer) and
        # multi-output (self.trainers) configurations
        has_zeroproof_trainer = bool(getattr(self, "trainer", None)) or bool(
            getattr(self, "trainers", None)
        )
        if self.config.model_type == "tr_rat" and has_zeroproof_trainer:
            self._train_with_zeroproof(train_inputs, train_targets)
        else:
            # Use simple training loop
            self._train_baseline(train_inputs, train_targets)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Validation if provided
        if val_samples:
            val_metrics = self.evaluate(val_samples)
            print(f"Validation MSE: {val_metrics['mse']:.6f}")

    def _train_with_zeroproof(self, inputs: List, targets: List):
        """Train using ZeroProof enhanced trainer."""
        for epoch in range(self.config.epochs):
            # Engage hybrid schedule (if configured) to set gradient mode and exploration
            try:
                if hasattr(self, "trainer") and getattr(self.trainer, "hybrid_schedule", None):
                    from zeroproof.autodiff.hybrid_gradient import HybridGradientContext

                    # Update epoch in context
                    HybridGradientContext.update_epoch(epoch)
                    # Decide mode based on current delta
                    delta = self.trainer.hybrid_schedule.get_delta(epoch)
                    if delta is None:
                        GradientModeConfig.set_mode(GradientMode.MASK_REAL)
                    else:
                        GradientModeConfig.set_mode(GradientMode.HYBRID)
                        # Set local threshold for near-pole decisions if available
                        try:
                            GradientModeConfig.set_local_threshold(delta)
                        except Exception:
                            pass
                    # Set exploration regions for this epoch
                    regions = self.trainer.hybrid_schedule.get_exploration_regions(epoch)
                    if regions:
                        HybridGradientContext.set_exploration_regions(regions)
            except Exception:
                # Fall back silently if hybrid components are unavailable
                pass
            epoch_metrics = []
            total_data_ms = 0.0
            total_optim_ms = 0.0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(inputs), self.config.batch_size):
                batch_inputs = inputs[i : i + self.config.batch_size]
                batch_targets = targets[i : i + self.config.batch_size]

                # Convert to TRNodes
                t_data0 = time.time()
                tr_inputs = []
                float_targets = []

                for inp, tgt in zip(batch_inputs, batch_targets):
                    # Multi-input case - create list of TRNodes
                    tr_inp = [TRNode.constant(real(x)) for x in inp]
                    # Keep targets as floats for mini-batch path
                    tr_tgt = [float(y) for y in tgt]

                    tr_inputs.append(tr_inp)
                    float_targets.append(tr_tgt)

                data_ms = (time.time() - t_data0) * 1000.0
                total_data_ms += data_ms
                n_batches += 1

                # Train batch (mini-batch path for composite model)
                if isinstance(self.model, list):
                    # Fallback to existing per-head path for list models
                    for tr_inp, tr_tgt in zip(tr_inputs, float_targets):
                        total_loss = 0.0
                        for j, (model, trainer) in enumerate(zip(self.model, self.trainers)):
                            x = tr_inp[0] if len(tr_inp) > 0 else TRNode.constant(real(0.0))
                            y_target = tr_tgt[j] if j < len(tr_tgt) else 0.0
                            y_pred, tag = model.forward(x)
                            if tag == TRTag.REAL:
                                loss = (y_pred - TRNode.constant(real(y_target))) ** 2
                                loss.backward()
                                trainer.optimizer.step(model)
                                total_loss += loss.value.value
                        epoch_metrics.append(total_loss / len(self.model))
                else:
                    # Optional pole-head supervision using analytic teacher from theta2.
                    # Uses |sin(theta2)| threshold as an analytic label for proximity to theta2∈{0,π}.
                    if (
                        self.config.use_pole_head
                        and self.config.supervise_pole_head
                        and getattr(self.model, "enable_pole_head", False)
                        and getattr(self, "pole_optimizer", None) is not None
                    ):
                        batch_pole_losses = []
                        for tr_inp in tr_inputs:
                            theta2 = (
                                float(tr_inp[1].value.value) if hasattr(tr_inp[1], "value") else 0.0
                            )
                            label = (
                                1.0
                                if abs(math.sin(theta2)) < float(self.config.teacher_pole_threshold)
                                else 0.0
                            )
                            score = self.model.forward_pole_head(tr_inp)
                            if score is not None:
                                loss_node = binary_cross_entropy(score, label)
                                batch_pole_losses.append(loss_node)
                        if batch_pole_losses:
                            total = batch_pole_losses[0]
                            for ln in batch_pole_losses[1:]:
                                total = total + ln
                            pole_loss = total / TRNode.constant(real(float(len(batch_pole_losses))))
                            self.pole_optimizer.zero_grad()
                            pole_loss.backward()
                            self.pole_optimizer.step(self.model)
                            if not hasattr(self, "pole_head_loss_history"):
                                self.pole_head_loss_history = []
                            self.pole_head_loss_history.append(
                                pole_loss.value.value if pole_loss.value.tag == TRTag.REAL else 0.0
                            )

                    result = self.trainer._train_batch_multi(tr_inputs, float_targets)
                    total_optim_ms += result.get("optim_ms", 0.0)
                    epoch_metrics.append(result.get("loss", float("inf")))

            # Log progress
            if epoch_metrics:
                avg_loss = sum(epoch_metrics) / len(epoch_metrics)
                avg_data = total_data_ms / max(1, n_batches)
                avg_optim = total_optim_ms / max(1, n_batches)
                avg_step = avg_data + avg_optim
                self.training_history.append(avg_loss)
                self.bench_history.append(
                    {
                        "epoch": epoch,
                        "avg_step_ms": avg_step,
                        "data_ms": avg_data,
                        "optim_ms": avg_optim,
                        "batches": n_batches,
                    }
                )
                # Optional: evaluate PLE on validation set each epoch
                if self.config.evaluate_ple and getattr(self, "val_inputs", None) is not None:
                    try:
                        ple = self._compute_ple_on_data(self.val_inputs)
                        self.ple_history.append(ple)
                    except Exception:
                        pass
                # Logging cadence
                if getattr(self, "log_policy_console", False):
                    try:
                        from zeroproof.autodiff.hybrid_gradient import HybridGradientContext

                        h = HybridGradientContext.get_statistics()
                        policy_mode = h.get("policy_mode", "MR")
                        q_p10 = h.get("q_p10")
                        q_p50 = h.get("q_p50")
                        q_p90 = h.get("q_p90")
                        tau_on = h.get("tau_q_on")
                        tau_off = h.get("tau_q_off")
                        sat = h.get("saturating_activations", 0)
                        total = h.get("total_gradient_calls", 0)
                        npr = h.get("near_pole_ratio", 0.0)
                        print(
                            f"Epoch {epoch}/{self.config.epochs - 1}: Loss={avg_loss:.6f} | "
                            f"policy={policy_mode} q_p10={q_p10} q_p50={q_p50} q_p90={q_p90} "
                            f"tau_on={tau_on} tau_off={tau_off} | sat={sat}/{total} npr={npr:.3f} | "
                            f"Bench: avg_step_ms={avg_step:.1f}, data_ms={avg_data:.1f}, optim_ms={avg_optim:.1f}, batches={n_batches}"
                        )
                    except Exception:
                        print(
                            f"Epoch {epoch}/{self.config.epochs - 1}: Loss = {avg_loss:.6f}  "
                            f"Bench: avg_step_ms={avg_step:.1f}, data_ms={avg_data:.1f}, optim_ms={avg_optim:.1f}, batches={n_batches}"
                        )
                else:
                    print(
                        f"Epoch {epoch}/{self.config.epochs - 1}: Loss = {avg_loss:.6f}  "
                        f"Bench: avg_step_ms={avg_step:.1f}, data_ms={avg_data:.1f}, optim_ms={avg_optim:.1f}, batches={n_batches}"
                    )

    def _compute_ple_on_data(self, inputs: List[List[float]]) -> float:
        """Compute PLE against θ2∈{0,π} lines for provided inputs using current model."""
        preds: List[List[float]] = []
        for inp in inputs:
            tr_inp = [TRNode.constant(real(x)) for x in inp]
            try:
                outs = self.model.forward(tr_inp)  # Expect List[(TRNode, TRTag)]
                pred_vec = [out.value.value if tag == TRTag.REAL else 0.0 for (out, tag) in outs]
            except TypeError:
                y, tag = self.model.forward(tr_inp[0])
                pred_vec = [y.value.value if tag == TRTag.REAL else 0.0]
            preds.append(pred_vec)
        return compute_ple_to_lines(inputs, preds)

    def _train_baseline(self, inputs: List, targets: List):
        """Train baseline models."""
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(inputs), self.config.batch_size):
                batch_inputs = inputs[i : i + self.config.batch_size]
                batch_targets = targets[i : i + self.config.batch_size]

                batch_loss = 0.0

                for inp, tgt in zip(batch_inputs, batch_targets):
                    # Convert to TRNodes
                    tr_inp = [TRNode.constant(real(x)) for x in inp]

                    # Forward pass
                    outputs = self.model.forward(tr_inp)

                    # Compute loss
                    loss = TRNode.constant(real(0.0))
                    for j, (output, target) in enumerate(zip(outputs, tgt)):
                        if output.tag == TRTag.REAL:
                            diff = output - TRNode.constant(real(target))
                            loss = loss + diff * diff

                    # Backward pass
                    loss.backward()

                    batch_loss += loss.value.value if loss.tag == TRTag.REAL else 0.0

                # Optimizer step with optional gradient clipping
                self._apply_gradient_clipping()
                self.optimizer.step(self.model)

                epoch_loss += batch_loss
                n_batches += 1

                # Update policy hysteresis after each batch to reflect quantiles
                try:
                    from zeroproof.autodiff.hybrid_gradient import HybridGradientContext

                    HybridGradientContext.end_batch_policy_update()
                except Exception:
                    pass

            # Log progress (per-epoch summary)
            if n_batches > 0 and (epoch % max(1, self.config.log_every) == 0):
                avg_loss = epoch_loss / n_batches
                # Hybrid policy metrics
                try:
                    from zeroproof.autodiff.hybrid_gradient import HybridGradientContext

                    h = HybridGradientContext.get_statistics()
                    policy_mode = h.get("policy_mode", "MR")
                    q_p10 = h.get("q_p10")
                    q_p50 = h.get("q_p50")
                    q_p90 = h.get("q_p90")
                    tau_on = h.get("tau_q_on")
                    tau_off = h.get("tau_q_off")
                    sat = h.get("saturating_activations", 0)
                    total = h.get("total_gradient_calls", 0)
                    npr = h.get("near_pole_ratio", 0.0)
                    print(
                        f"Epoch {epoch}: Loss={avg_loss:.6f} | policy_mode={policy_mode} "
                        f"q_p10={q_p10} q_p50={q_p50} q_p90={q_p90} "
                        f"tau_on={tau_on} tau_off={tau_off} | sat={sat}/{total} npr={npr:.3f}"
                    )
                    # Store a compact per-epoch record for plotting
                    try:
                        if hasattr(self, "trainer") and hasattr(self.trainer, "_epoch_records"):
                            self.trainer._epoch_records.append(
                                {
                                    "epoch": epoch + 1,
                                    "loss": float(avg_loss),
                                    "q_p10": q_p10,
                                    "q_p50": q_p50,
                                    "q_p90": q_p90,
                                    "saturating_ratio": h.get("saturating_ratio", 0.0),
                                    "flip_rate": h.get("flip_rate", 0.0),
                                }
                            )
                    except Exception:
                        pass
                except Exception:
                    print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                self.training_history.append(avg_loss)

    def _apply_gradient_clipping(self) -> None:
        """Apply global-norm gradient clipping if configured (baselines)."""
        c = getattr(self.config, "clip_grad", None)
        if c is None or not isinstance(c, (int, float)) or c <= 0:
            return
        if not hasattr(self.model, "parameters"):
            return
        params = list(self.model.parameters())
        if not params:
            return
        import math as _math

        total_sq = 0.0
        for p in params:
            try:
                g = p.gradient
                if g is None or g.tag != TRTag.REAL:
                    continue
                val = float(getattr(g, "value", 0.0))
                if _math.isfinite(val):
                    total_sq += val * val
            except Exception:
                continue
        norm = total_sq**0.5
        if norm <= 0.0 or norm <= float(c):
            return
        scale = float(c) / (norm + 1e-12)
        for p in params:
            try:
                g = p.gradient
                if g is None or g.tag != TRTag.REAL:
                    continue
                new_val = float(g.value) * scale
                # Replace stored gradient with scaled value
                p._gradient = real(new_val)
            except Exception:
                continue

    def evaluate(self, test_samples: List[IKSample]) -> Dict[str, float]:
        """Evaluate model on test data."""
        test_inputs, test_targets = self.prepare_data(test_samples)

        total_mse = 0.0
        total_samples = 0
        predictions = []

        for inp, tgt in zip(test_inputs, test_targets):
            tr_inp = [TRNode.constant(real(x)) for x in inp]

            def _to_float_list(obj) -> List[float]:
                # Accept list of (TRNode, TRTag) or list of TRNode
                floats: List[float] = []
                if isinstance(obj, list) and obj:
                    first = obj[0]
                    # Case 1: list of tuples
                    if isinstance(first, tuple) and len(first) == 2:
                        for out, tag in obj:
                            try:
                                floats.append(float(out.value.value) if tag == TRTag.REAL else 0.0)
                            except Exception:
                                floats.append(0.0)
                    else:
                        # Assume list of TRNode
                        for out in obj:
                            try:
                                tag = getattr(out, "tag", None)
                                if tag == TRTag.REAL:
                                    floats.append(float(out.value.value))
                                else:
                                    floats.append(0.0)
                            except Exception:
                                floats.append(0.0)
                return floats

            if isinstance(self.model, list):
                # Rare path: list of single-output rationals
                outs = []
                for model in self.model:
                    try:
                        y, tag = model.forward(tr_inp[0])
                        outs.append((y, tag))
                    except Exception:
                        y = model.forward(tr_inp[0])
                        outs.append((y, getattr(y, "tag", TRTag.REAL)))
                outputs = _to_float_list(outs)
            else:
                try:
                    model_outputs = self.model.forward(tr_inp)
                except TypeError:
                    # Some models expect scalar input
                    model_outputs = self.model.forward(tr_inp[0])
                # Normalize to list
                if not isinstance(model_outputs, list):
                    model_outputs = [model_outputs]
                outputs = _to_float_list(model_outputs)

            # Compute MSE
            mse = sum((pred - true) ** 2 for pred, true in zip(outputs, tgt))
            total_mse += mse
            total_samples += 1

            predictions.append(outputs)

        avg_mse = total_mse / total_samples if total_samples > 0 else float("inf")

        return {"mse": avg_mse, "predictions": predictions, "n_samples": total_samples}

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        summary = {
            "model_type": self.config.model_type,
            "config": self.config.__dict__,
            "training_history": self.training_history,
            "n_parameters": len(self.model.parameters())
            if hasattr(self.model, "parameters")
            else 0,
            "bench_history": self.bench_history,
        }
        if hasattr(self, "pole_head_loss_history"):
            summary["pole_head_loss_history"] = self.pole_head_loss_history
        if self.ple_history:
            summary["ple_history"] = self.ple_history
            summary["final_ple"] = self.ple_history[-1]

        if hasattr(self, "trainer") and self.trainer:
            trainer_summary = self.trainer.get_training_summary()
            summary["zeroproof_metrics"] = trainer_summary

        return summary


def run_experiment(
    dataset_file: str,
    config: TrainingConfig,
    output_dir: str,
    seed: Optional[int] = None,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    log_policy_console: bool = False,
):
    """Run complete training experiment."""
    print(f"=== Running IK Training Experiment ===")
    print(f"Model: {config.model_type}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")

    # Load dataset
    generator = RRDatasetGenerator.load_dataset(dataset_file)
    samples = generator.samples
    metadata = getattr(generator, "metadata", {})

    print(f"Loaded {len(samples)} samples")

    # Train/test split
    if metadata.get("stratified_by_detj") and "train_bucket_counts" in metadata:
        n_train = int(sum(metadata.get("train_bucket_counts", [])))
    else:
        n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    # Optional subsampling for quick runs
    if isinstance(limit_train, int) and limit_train > 0:
        train_samples = train_samples[: min(limit_train, len(train_samples))]
    if isinstance(limit_test, int) and limit_test > 0:
        test_samples = test_samples[: min(limit_test, len(test_samples))]

    # Create trainer
    trainer = IKTrainer(config)

    # Determine input/output dimensions
    # Input: [theta1, theta2, dx, dy] -> 4D
    # Output: [dtheta1, dtheta2] -> 2D
    trainer.create_model(input_dim=4, output_dim=2)
    trainer.setup_trainer()
    # Enable console policy logs if requested
    try:
        setattr(trainer, "log_policy_console", bool(log_policy_console))
    except Exception:
        pass

    # Train model
    trainer.train(train_samples, test_samples)

    # Evaluate
    test_metrics = trainer.evaluate(test_samples)
    # Compute pole metrics on test set for convenience (profile-gated)
    pole_metrics = {}
    if getattr(config, "evaluate_ple", True):
        try:
            test_inputs, _ = trainer.prepare_data(test_samples)
            pole_metrics = compute_pole_metrics_2d(test_inputs, test_metrics.get("predictions", []))
        except Exception:
            pole_metrics = {}
    print(f"Final test MSE: {test_metrics['mse']:.6f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    from zeroproof.utils.env import collect_env_info

    results = {
        "config": config.__dict__,
        "dataset_info": {
            "file": dataset_file,
            "n_samples": len(samples),
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
        "training_summary": trainer.get_training_summary(),
        "test_metrics": test_metrics,
        "pole_metrics": pole_metrics,
        "seed": seed,
        "loss_name": "mse_mean",
        "env": collect_env_info(),
    }

    # Attach policy summary (thresholds and flags) if active
    try:
        from zeroproof.policy import TRPolicyConfig

        pol = TRPolicyConfig.get_policy()
        if pol is not None:
            results["policy"] = {
                "tau_Q_on": float(pol.tau_Q_on),
                "tau_Q_off": float(pol.tau_Q_off),
                "tau_P_on": float(pol.tau_P_on),
                "tau_P_off": float(pol.tau_P_off),
                "rounding_mode": getattr(pol, "rounding_mode", None),
                "keep_signed_zero": bool(getattr(pol, "keep_signed_zero", True)),
                "deterministic_reduction": bool(getattr(pol, "deterministic_reduction", True)),
                "g_on": float(pol.g_on) if pol.g_on is not None else None,
                "g_off": float(pol.g_off) if pol.g_off is not None else None,
            }
    except Exception:
        pass

    # Extract final curvature bound/thresholds from training summary if available
    try:
        hist = results.get("training_summary", {}).get("final_metrics", [])
        if isinstance(hist, list) and hist:
            last = hist[-1]
            for k in (
                "curvature_bound",
                "B_k",
                "H_k",
                "G_max",
                "H_max",
                "tau_q_on",
                "tau_q_off",
                "flip_rate",
                "saturating_ratio",
                "q_min_epoch",
                "mask_bandwidth",
            ):
                if k in last and last[k] is not None:
                    results[k] = last[k]
    except Exception:
        pass

    # Attach layer contract at top level for convenience
    try:
        lc = results.get("training_summary", {}).get("layer_contract", None)
        if lc is not None:
            results["layer_contract"] = lc
    except Exception:
        pass

    # Guardrail: warn if hybrid did not switch (near_pole_ratio too low)
    try:
        tz = results.get("training_summary", {}).get("zeroproof_metrics", {})
        hist = tz.get("final_metrics", [])
        last = hist[-1] if isinstance(hist, list) and hist else {}
        npr = last.get("near_pole_ratio", None)
        if isinstance(npr, (int, float)) and npr < 0.05:
            print(
                "[Guardrail] near_pole_ratio is below 0.05. Consider increasing --singular_ratio "
                "in dataset generation or lowering --min_detj to ensure sufficient near-pole coverage."
            )
    except Exception:
        pass

    results_file = os.path.join(output_dir, f"results_{config.model_type}.json")
    from zeroproof.utils.serialization import to_builtin

    with open(results_file, "w") as f:
        json.dump(to_builtin(results), f, indent=2)

    print(f"Results saved to {results_file}")

    # Save training plots (including Q/D/smin curves) if available
    try:
        if hasattr(trainer, "trainer") and trainer.trainer:
            epoch_records = []
            try:
                epoch_records = trainer.trainer.get_epoch_records()
            except Exception:
                epoch_records = []
            if epoch_records:
                from zeroproof.utils.plotting import save_all_plots

                save_all_plots(output_dir, epoch_records, model=trainer.model)
    except Exception as e:
        print(f"Warning: failed to save training plots: {e}")

    return results


def main():
    """Command-line interface for IK training."""
    parser = argparse.ArgumentParser(description="Train IK models on RR robot data")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rat_eps", "tr_rat"],
        default="tr_rat",
        help="Model type",
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/ik_experiment", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=None, help="Global seed for reproducibility")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["quick", "full"],
        default=None,
        help="Profile to apply suggested defaults for speed or thoroughness",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log cadence in epochs (currently per-epoch summaries)",
    )
    parser.add_argument(
        "--no_structured_logging",
        action="store_true",
        help="Disable structured logging inside Hybrid trainer for performance",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation inside Hybrid trainer for performance",
    )
    parser.add_argument(
        "--emit_overhead",
        action="store_true",
        help="Emit Hybrid vs Mask-REAL overhead JSON after training",
    )

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for MLP")
    parser.add_argument(
        "--degree_p", type=int, default=3, help="Numerator degree for rational models"
    )
    parser.add_argument(
        "--degree_q", type=int, default=2, help="Denominator degree for rational models"
    )

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon for Rational+ε (P/(Q+ε)); applies to --model rat_eps",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="Global-norm gradient clipping (e.g., 1.0) for baselines",
    )

    # ZeroProof features
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid gradient schedule")
    parser.add_argument("--no_tag_loss", action="store_true", help="Disable tag loss")
    parser.add_argument("--no_pole_head", action="store_true", help="Disable pole head")
    parser.add_argument(
        "--no_residual_loss", action="store_true", help="Disable residual consistency loss"
    )
    parser.add_argument(
        "--no_anti_illusion", action="store_true", help="Disable anti-illusion metrics"
    )
    parser.add_argument("--no_coverage", action="store_true", help="Disable coverage enforcement")
    parser.add_argument(
        "--supervise-pole-head",
        action="store_true",
        help="Use analytic det(J) teacher to supervise pole head",
    )
    parser.add_argument(
        "--teacher_pole_threshold",
        type=float,
        default=0.1,
        help="Threshold on |sin(theta2)| for near-pole labels (default: 0.1)",
    )

    # Loss weights
    parser.add_argument("--lambda_tag", type=float, default=0.05, help="Tag loss weight")
    parser.add_argument("--lambda_pole", type=float, default=0.1, help="Pole loss weight")
    parser.add_argument("--lambda_residual", type=float, default=0.02, help="Residual loss weight")

    # Quick run and logging controls
    parser.add_argument(
        "--quick_demo",
        action="store_true",
        help="Use very few epochs and a small data subset for a quick run",
    )
    parser.add_argument(
        "--limit_train",
        type=int,
        default=None,
        help="Limit number of training samples (for quick runs)",
    )
    parser.add_argument(
        "--limit_test",
        type=int,
        default=None,
        help="Limit number of test/val samples (for quick runs)",
    )
    parser.add_argument(
        "--log_policy_console",
        action="store_true",
        help="Print policy/hybrid metrics in console at each epoch",
    )
    # Policy/Hybrid tuning
    parser.add_argument(
        "--no_tr_policy", action="store_true", help="Disable TR policy (use raw TR tags only)"
    )
    parser.add_argument(
        "--policy_ulp_scale",
        type=float,
        default=4.0,
        help="ULP scale factor for policy guard bands (e.g., 1e12)",
    )
    parser.add_argument(
        "--no_policy_det_reduction",
        action="store_true",
        help="Disable deterministic pairwise reductions for P/Q",
    )
    parser.add_argument(
        "--hybrid_aggressive",
        action="store_true",
        help="Use aggressive hybrid schedule (smaller bound)",
    )
    parser.add_argument(
        "--hybrid_delta_init", type=float, default=1e-2, help="Initial delta for hybrid schedule"
    )
    parser.add_argument(
        "--hybrid_delta_final", type=float, default=1e-6, help="Final delta for hybrid schedule"
    )
    # Contract-safe LR clamp
    parser.add_argument(
        "--use_contract_safe_lr",
        action="store_true",
        help="Enable contract-safe LR clamp from layer contract",
    )
    parser.add_argument(
        "--contract_c",
        type=float,
        default=1.0,
        help="Contract constant c used in LR clamp (eta <= c / (beta * prod))",
    )
    parser.add_argument(
        "--loss_smoothness_beta",
        type=float,
        default=1.0,
        help="Assumed loss smoothness beta for LR clamp",
    )
    # Coprime surrogate regularizer
    parser.add_argument(
        "--enable_coprime_reg",
        action="store_true",
        help="Enable coprime surrogate regularizer on rational heads",
    )
    parser.add_argument(
        "--lambda_coprime", type=float, default=0.0, help="Weight for coprime surrogate regularizer"
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        degree_p=args.degree_p,
        degree_q=args.degree_q,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        no_structured_logging=args.no_structured_logging,
        no_plots=args.no_plots,
        use_hybrid_schedule=not args.no_hybrid,
        use_tag_loss=not args.no_tag_loss,
        use_pole_head=not args.no_pole_head,
        use_residual_loss=not args.no_residual_loss,
        enable_anti_illusion=not args.no_anti_illusion,
        enforce_coverage=not args.no_coverage,
        supervise_pole_head=args.supervise_pole_head,
        teacher_pole_threshold=args.teacher_pole_threshold,
        lambda_tag=args.lambda_tag,
        lambda_pole=args.lambda_pole,
        lambda_residual=args.lambda_residual,
        epsilon=args.epsilon,
        clip_grad=args.clip_grad,
        # Policy/Hybrid knobs
        use_tr_policy=not args.no_tr_policy,
        policy_ulp_scale=args.policy_ulp_scale,
        policy_deterministic_reduction=not args.no_policy_det_reduction,
        hybrid_aggressive=args.hybrid_aggressive,
        hybrid_delta_init=args.hybrid_delta_init,
        hybrid_delta_final=args.hybrid_delta_final,
        # Contract-safe LR
        use_contract_safe_lr=args.use_contract_safe_lr,
        contract_c=args.contract_c,
        loss_smoothness_beta=args.loss_smoothness_beta,
        # Coprime surrogate knobs
        enable_coprime_regularizer=args.enable_coprime_reg,
        lambda_coprime=args.lambda_coprime,
    )

    # Apply profile defaults (Section 7)
    if args.profile == "quick":
        # Epochs
        if args.epochs == parser.get_default("epochs"):
            config.epochs = 20
        # Batch size
        if args.batch_size == parser.get_default("batch_size"):
            config.batch_size = 1024
        # Logging cadence
        config.log_every = 200
        # Metrics toggles
        config.evaluate_ple = False
        config.enable_anti_illusion = False
    elif args.profile == "full":
        # Epochs
        if args.epochs == parser.get_default("epochs"):
            config.epochs = 100
        # Batch size
        if args.batch_size == parser.get_default("batch_size"):
            config.batch_size = 2048
        # Logging cadence
        config.log_every = 50
        # Ensure full features enabled unless explicitly disabled
        config.use_hybrid_schedule = True and config.use_hybrid_schedule
        config.use_tag_loss = True and config.use_tag_loss
        config.use_pole_head = True and config.use_pole_head
        config.enable_anti_illusion = True and config.enable_anti_illusion
        config.evaluate_ple = True

    # Seed
    set_global_seed(args.seed)

    # Run experiment
    # Optional quick demo overrides
    if hasattr(args, "quick_demo") and args.quick_demo:
        if args.epochs == parser.get_default("epochs"):
            config.epochs = 3
        if args.batch_size == parser.get_default("batch_size"):
            config.batch_size = 128
        # If dataset is large, cap to a small subset
        args.limit_train = args.limit_train or 2000
        args.limit_test = args.limit_test or 500

    results = run_experiment(
        args.dataset,
        config,
        args.output_dir,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        log_policy_console=getattr(args, "log_policy_console", False),
    )

    # Optional: overhead report
    if getattr(args, "emit_overhead", False):
        try:
            import json
            import os

            from zeroproof.core import real as tr_real
            from zeroproof.utils.overhead import overhead_report

            # Build a tiny data loader from a small subset of the dataset
            trainer_obj = trainer.trainer if hasattr(trainer, "trainer") else None
            if trainer_obj is not None:
                # Use first 32 training samples (or fewer)
                subset = train_samples[: min(32, len(train_samples))]
                inputs, targets = trainer.prepare_data(subset)
                # Convert to TRScalars for scalar path or keep vectors for multi-input
                data_loader = []
                if inputs and isinstance(inputs[0], list):
                    # Multi-input vector case
                    # Convert to TRNodes/TRScalars at _train_batch_multi path; we can pass floats and rely on trainer internals
                    data_loader = [(inputs, targets)]
                else:
                    # Scalar case
                    tr_inputs = [tr_real(float(v)) for v in inputs]
                    tr_targets = [tr_real(float(v)) for v in targets]
                    data_loader = [(tr_inputs, tr_targets)]
                rep = overhead_report(trainer_obj, data_loader)
                overhead_file = os.path.join(args.output_dir, "overhead_report.json")
                with open(overhead_file, "w") as f:
                    json.dump(rep, f, indent=2)
                print(f"Overhead report saved to {overhead_file}")
        except Exception as e:
            print(f"Warning: failed to emit overhead report: {e}")

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
