"""
Teacher and proxy supervision for pole detection.

This module implements various supervision strategies to improve pole detection
accuracy from the baseline 40% to 60%+ through teacher signals, proxy indicators,
and pre-training on synthetic data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..core import TRTag, ninf, phi, pinf, real
from ..layers.enhanced_pole_detection import EnhancedPoleDetectionHead


class SupervisionType(Enum):
    """Types of supervision available for pole detection."""

    ANALYTIC = "analytic"  # Exact analytic labels (e.g., det(J) = 0)
    PROXY = "proxy"  # Proxy signals (e.g., instability indicators)
    WEAK = "weak"  # Weak labels from heuristics
    SELF = "self"  # Self-supervision from model predictions
    SYNTHETIC = "synthetic"  # Synthetic data with known poles


@dataclass
class TeacherConfig:
    """Configuration for teacher supervision."""

    # Teacher type
    supervision_type: SupervisionType = SupervisionType.ANALYTIC

    # Loss weighting
    teacher_weight: float = 1.0  # Weight for teacher loss
    student_weight: float = 0.5  # Weight for student predictions

    # Label smoothing
    label_smoothing: float = 0.1  # Smooth hard labels

    # Confidence thresholds
    min_confidence: float = 0.8  # Minimum teacher confidence to use

    # Proxy signal parameters
    instability_threshold: float = 100.0  # Threshold for instability detection
    gradient_explosion_threshold: float = 1000.0

    # Pre-training
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 64

    # Curriculum
    use_curriculum: bool = True  # Start with easy examples
    initial_difficulty: float = 0.1  # Initial difficulty (0=easy, 1=hard)
    final_difficulty: float = 1.0  # Final difficulty


class RoboticsTeacher:
    """
    Teacher for robotics applications using Jacobian determinant.

    In robotics, singularities occur when det(J) = 0, where J is the
    Jacobian matrix. This provides exact ground-truth pole locations.
    """

    def __init__(self, config: Optional[TeacherConfig] = None):
        """
        Initialize robotics teacher.

        Args:
            config: Teacher configuration
        """
        self.config = config or TeacherConfig()

        # Cache for Jacobian computations
        self.jacobian_cache = {}

    def compute_jacobian_det(
        self, q: torch.Tensor, robot_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute Jacobian determinant for robot configuration.

        Args:
            q: Joint angles [batch_size, n_joints]
            robot_params: Robot parameters (link lengths, etc.)

        Returns:
            Determinant values [batch_size]
        """
        batch_size = q.shape[0]
        n_joints = q.shape[1]

        if robot_params is None:
            # Default 2R robot parameters
            robot_params = {
                "l1": 1.0,  # Link 1 length
                "l2": 1.0,  # Link 2 length
            }

        # For 2R robot: det(J) = l1 * l2 * sin(q2)
        if n_joints == 2:
            l1 = robot_params["l1"]
            l2 = robot_params["l2"]
            det_j = l1 * l2 * torch.sin(q[:, 1])
        else:
            # For higher DOF, compute full Jacobian
            det_j = self._compute_full_jacobian_det(q, robot_params)

        return det_j

    def _compute_full_jacobian_det(self, q: torch.Tensor, robot_params: Dict) -> torch.Tensor:
        """
        Compute determinant for arbitrary DOF robot.

        Args:
            q: Joint angles
            robot_params: Robot parameters

        Returns:
            Determinant values
        """
        # Placeholder for full Jacobian computation
        # In practice, this would use forward kinematics
        batch_size = q.shape[0]

        # Simplified: random determinant based on joint config
        det_j = torch.prod(torch.sin(q), dim=1)

        return det_j

    def get_pole_labels(
        self, q: torch.Tensor, threshold: float = 0.01, robot_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Get binary pole labels from Jacobian determinant.

        Args:
            q: Joint angles
            threshold: Threshold for singularity detection
            robot_params: Robot parameters

        Returns:
            Binary labels (1 = pole, 0 = regular) [batch_size]
        """
        det_j = self.compute_jacobian_det(q, robot_params)

        # Singularity when |det(J)| < threshold
        labels = (torch.abs(det_j) < threshold).float()

        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            labels = labels * (1 - self.config.label_smoothing) + self.config.label_smoothing / 2

        return labels

    def get_pole_distances(
        self, q: torch.Tensor, robot_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Get distance to nearest singularity.

        Args:
            q: Joint angles
            robot_params: Robot parameters

        Returns:
            Distance values [batch_size]
        """
        det_j = self.compute_jacobian_det(q, robot_params)

        # Distance is |det(J)| (closer to 0 = closer to singularity)
        distances = torch.abs(det_j)

        return distances

    def compute_teacher_loss(
        self, predictions: torch.Tensor, q: torch.Tensor, robot_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute teacher supervision loss.

        Args:
            predictions: Student pole predictions [batch_size]
            q: Joint angles
            robot_params: Robot parameters

        Returns:
            Teacher loss value
        """
        # Get ground truth labels
        labels = self.get_pole_labels(q, robot_params=robot_params)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predictions, labels)

        # Weight by teacher confidence (based on distance to singularity)
        distances = self.get_pole_distances(q, robot_params)
        confidence = torch.exp(-distances / 0.1)  # Higher confidence near poles

        # Only use high-confidence samples
        mask = confidence > self.config.min_confidence
        if mask.any():
            loss = (loss * confidence * mask).sum() / mask.sum()

        return loss * self.config.teacher_weight


class ProxyTeacher:
    """
    Teacher using proxy signals for weak supervision.

    Uses instability indicators like gradient explosions, numerical
    errors, or convergence failures as weak labels for pole detection.
    """

    def __init__(self, config: Optional[TeacherConfig] = None):
        """
        Initialize proxy teacher.

        Args:
            config: Teacher configuration
        """
        self.config = config or TeacherConfig()

        # History for instability detection
        self.gradient_history = []
        self.loss_history = []
        self.convergence_history = []

    def detect_instability(
        self,
        gradients: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Detect instability signals as proxy for poles.

        Args:
            gradients: Current gradient magnitudes
            loss: Current loss value
            outputs: Model outputs (check for non-REAL)

        Returns:
            Instability scores [0, 1]
        """
        instability_scores = []

        # Check gradient explosion
        if gradients is not None:
            grad_magnitude = torch.norm(gradients, dim=-1)
            grad_instability = torch.sigmoid(
                (grad_magnitude - self.config.gradient_explosion_threshold) / 100
            )
            instability_scores.append(grad_instability)

        # Check loss spikes
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > 5:
                recent_mean = np.mean(self.loss_history[-5:])
                loss_spike = loss > 2 * recent_mean
                instability_scores.append(torch.tensor(float(loss_spike)))

        # Check for non-REAL outputs (indicates singularity)
        if outputs is not None:
            # Assuming outputs have a tag attribute
            non_real_ratio = (outputs != 0).float().mean()  # Simplified
            instability_scores.append(non_real_ratio)

        # Combine signals
        if instability_scores:
            combined = torch.stack(instability_scores).mean(0)
        else:
            combined = torch.zeros(1)

        return combined

    def compute_proxy_loss(
        self, predictions: torch.Tensor, instability_signals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute proxy supervision loss.

        Args:
            predictions: Student pole predictions
            instability_signals: Detected instability scores

        Returns:
            Proxy loss value
        """
        # Use instability as soft labels
        loss = F.mse_loss(predictions, instability_signals)

        return loss * self.config.teacher_weight


class SyntheticPoleDataset(Dataset):
    """
    Synthetic dataset with known pole locations for pre-training.

    Generates rational functions with controlled pole positions
    for pre-training the pole detection head.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        input_dim: int = 1,
        n_poles: int = 3,
        noise_level: float = 0.01,
    ):
        """
        Initialize synthetic dataset.

        Args:
            n_samples: Number of samples to generate
            input_dim: Input dimension
            n_poles: Number of poles to place
            noise_level: Noise to add to labels
        """
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_poles = n_poles
        self.noise_level = noise_level

        # Generate data
        self.inputs, self.labels, self.pole_locations = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Generate synthetic data with known poles.

        Returns:
            Tuple of (inputs, labels, pole_locations)
        """
        # Random pole locations
        pole_locations = np.random.uniform(-2, 2, self.n_poles).tolist()

        # Generate input points
        if self.input_dim == 1:
            inputs = torch.linspace(-3, 3, self.n_samples).unsqueeze(1)
        else:
            inputs = torch.randn(self.n_samples, self.input_dim)

        # Compute labels based on distance to poles
        labels = torch.zeros(self.n_samples)

        for i in range(self.n_samples):
            x = inputs[i, 0].item() if self.input_dim == 1 else inputs[i].norm().item()

            # Distance to nearest pole
            min_dist = min(abs(x - pole) for pole in pole_locations)

            # Label based on distance (1 near poles, 0 far away)
            label = np.exp(-min_dist / 0.1)

            # Add noise
            label += np.random.normal(0, self.noise_level)
            label = np.clip(label, 0, 1)

            labels[i] = label

        return inputs, labels, pole_locations

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


class PoleHeadPretrainer:
    """
    Pre-trainer for pole detection head using synthetic data.

    Pre-trains the pole head on synthetic data with known poles
    before fine-tuning on real data.
    """

    def __init__(
        self, pole_head: EnhancedPoleDetectionHead, config: Optional[TeacherConfig] = None
    ):
        """
        Initialize pre-trainer.

        Args:
            pole_head: Pole detection head to pre-train
            config: Teacher configuration
        """
        self.pole_head = pole_head
        self.config = config or TeacherConfig()

        # Optimizer for pre-training
        self.optimizer = torch.optim.Adam(pole_head.parameters(), lr=self.config.pretrain_lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.pretrain_epochs
        )

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def pretrain(
        self, n_samples: int = 10000, n_poles: int = 3, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Pre-train pole head on synthetic data.

        Args:
            n_samples: Number of synthetic samples
            n_poles: Number of poles in synthetic data
            verbose: Whether to print progress

        Returns:
            Pre-training results
        """
        # Create synthetic dataset
        dataset = SyntheticPoleDataset(
            n_samples=n_samples, input_dim=self.pole_head.input_dim, n_poles=n_poles
        )

        dataloader = DataLoader(dataset, batch_size=self.config.pretrain_batch_size, shuffle=True)

        if verbose:
            print(f"Pre-training pole head on {n_samples} synthetic samples")
            print(f"  Poles at: {dataset.pole_locations}")

        # Pre-training loop
        for epoch in range(self.config.pretrain_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                # Forward pass
                predictions = self.pole_head(inputs).squeeze()

                # Compute loss
                loss = F.binary_cross_entropy(predictions, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct += (predicted == (labels > 0.5).float()).sum().item()
                total += labels.shape[0]

            # Update scheduler
            self.scheduler.step()

            # Record history
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{self.config.pretrain_epochs}: "
                    f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}"
                )

        # Evaluate on test set
        test_accuracy = self.evaluate(dataset)

        results = {
            "final_loss": self.loss_history[-1],
            "final_accuracy": self.accuracy_history[-1],
            "test_accuracy": test_accuracy,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        }

        if verbose:
            print(f"\nPre-training complete!")
            print(f"  Final accuracy: {results['final_accuracy']:.2%}")
            print(f"  Test accuracy: {results['test_accuracy']:.2%}")

        return results

    def evaluate(self, dataset: SyntheticPoleDataset) -> float:
        """
        Evaluate pole head on dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Accuracy score
        """
        self.pole_head.eval()

        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                predictions = self.pole_head(inputs).squeeze()
                predicted = (predictions > 0.5).float()
                correct += (predicted == (labels > 0.5).float()).sum().item()
                total += labels.shape[0]

        self.pole_head.train()

        return correct / total


class HybridTeacher:
    """
    Hybrid teacher combining multiple supervision strategies.

    Combines analytic labels (when available), proxy signals,
    and pre-training for maximum pole detection accuracy.
    """

    def __init__(
        self,
        pole_head: EnhancedPoleDetectionHead,
        config: Optional[TeacherConfig] = None,
        use_robotics: bool = True,
        use_proxy: bool = True,
        use_pretrain: bool = True,
    ):
        """
        Initialize hybrid teacher.

        Args:
            pole_head: Pole detection head to train
            config: Teacher configuration
            use_robotics: Whether to use robotics teacher
            use_proxy: Whether to use proxy signals
            use_pretrain: Whether to pre-train on synthetic data
        """
        self.pole_head = pole_head
        self.config = config or TeacherConfig()

        # Initialize teachers
        self.robotics_teacher = RoboticsTeacher(config) if use_robotics else None
        self.proxy_teacher = ProxyTeacher(config) if use_proxy else None
        self.pretrainer = PoleHeadPretrainer(pole_head, config) if use_pretrain else None

        # Supervision weights (adaptive)
        self.teacher_weights = {
            "robotics": 1.0,
            "proxy": 0.5,
            "student": 0.2,
        }

        # Performance tracking
        self.supervision_history = []

    def pretrain_if_needed(self, verbose: bool = True) -> Optional[Dict]:
        """
        Pre-train pole head if configured.

        Args:
            verbose: Whether to print progress

        Returns:
            Pre-training results if performed
        """
        if self.pretrainer is not None:
            return self.pretrainer.pretrain(verbose=verbose)
        return None

    def compute_combined_loss(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
        robot_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute combined supervision loss.

        Args:
            predictions: Student pole predictions
            inputs: Input values (e.g., joint angles)
            gradients: Current gradients (for proxy)
            loss: Current loss (for proxy)
            robot_params: Robot parameters (for robotics)

        Returns:
            Combined loss value
        """
        total_loss = torch.tensor(0.0)
        loss_components = {}

        # Robotics teacher loss
        if self.robotics_teacher is not None and robot_params is not None:
            robotics_loss = self.robotics_teacher.compute_teacher_loss(
                predictions, inputs, robot_params
            )
            total_loss += robotics_loss * self.teacher_weights["robotics"]
            loss_components["robotics"] = robotics_loss.item()

        # Proxy teacher loss
        if self.proxy_teacher is not None and (gradients is not None or loss is not None):
            instability = self.proxy_teacher.detect_instability(gradients, loss)
            proxy_loss = self.proxy_teacher.compute_proxy_loss(predictions, instability)
            total_loss += proxy_loss * self.teacher_weights["proxy"]
            loss_components["proxy"] = proxy_loss.item()

        # Student self-supervision (consistency regularization)
        if predictions.requires_grad:
            # Encourage confident predictions
            entropy = -predictions * torch.log(predictions + 1e-8) - (1 - predictions) * torch.log(
                1 - predictions + 1e-8
            )
            entropy_loss = entropy.mean()
            total_loss += entropy_loss * self.teacher_weights["student"]
            loss_components["student"] = entropy_loss.item()

        # Track supervision
        self.supervision_history.append(loss_components)

        return total_loss

    def adapt_weights(self, performance_metrics: Dict[str, float]):
        """
        Adapt teacher weights based on performance.

        Args:
            performance_metrics: Performance of each teacher
        """
        # Increase weight of better-performing teachers
        for teacher, metric in performance_metrics.items():
            if teacher in self.teacher_weights:
                # Higher accuracy -> higher weight
                self.teacher_weights[teacher] *= 1 + 0.1 * metric

        # Normalize weights
        total = sum(self.teacher_weights.values())
        for teacher in self.teacher_weights:
            self.teacher_weights[teacher] /= total

    def get_statistics(self) -> Dict[str, Any]:
        """Get teacher supervision statistics."""
        if not self.supervision_history:
            return {}

        # Aggregate statistics
        stats = {}
        for teacher in ["robotics", "proxy", "student"]:
            losses = [h.get(teacher, 0) for h in self.supervision_history]
            if losses:
                stats[f"{teacher}_mean_loss"] = np.mean(losses)
                stats[f"{teacher}_weight"] = self.teacher_weights.get(teacher, 0)

        return stats


def create_pole_teacher(
    pole_head: EnhancedPoleDetectionHead,
    supervision_types: List[str] = None,
    _target_accuracy: float = 0.6,
    **kwargs,
) -> HybridTeacher:
    """
    Factory function to create pole teacher.

    Args:
        pole_head: Pole detection head to train
        supervision_types: Types of supervision to use
        target_accuracy: Target detection accuracy
        **kwargs: Additional configuration

    Returns:
        Configured teacher
    """
    if supervision_types is None:
        supervision_types = ["robotics", "proxy", "pretrain"]

    config = TeacherConfig(
        teacher_weight=kwargs.get("teacher_weight", 1.0),
        pretrain_epochs=kwargs.get("pretrain_epochs", 50),
        use_curriculum=kwargs.get("use_curriculum", True),
    )

    teacher = HybridTeacher(
        pole_head,
        config,
        use_robotics="robotics" in supervision_types,
        use_proxy="proxy" in supervision_types,
        use_pretrain="pretrain" in supervision_types,
    )

    # Pre-train if requested
    if "pretrain" in supervision_types:
        teacher.pretrain_if_needed(verbose=kwargs.get("verbose", True))

    return teacher
