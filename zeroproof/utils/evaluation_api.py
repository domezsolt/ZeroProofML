"""
Integrated evaluation API for pole learning metrics.

This module provides a unified interface for evaluating models with
pole-related metrics and integrating them into training logs.
"""

from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode
from .pole_metrics import PoleEvaluator, PoleMetrics
from .pole_visualization import PoleVisualizer


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Pole detection
    pole_threshold: float = 0.1
    near_pole_distance: float = 0.1
    mid_range_distance: float = 0.5
    
    # Metrics to compute
    compute_ple: bool = True
    compute_sign_consistency: bool = True
    compute_asymptotic: bool = True
    compute_residual: bool = True
    compute_coverage_breakdown: bool = True
    
    # Visualization
    enable_visualization: bool = True
    plot_frequency: int = 10  # Plot every N evaluations
    save_plots: bool = True
    plot_dir: str = "plots"
    
    # Logging
    log_to_file: bool = True
    log_file: str = "evaluation_log.json"
    verbose: bool = True


class IntegratedEvaluator:
    """
    Unified evaluator integrating all pole metrics.
    
    This class provides a complete evaluation pipeline with
    metrics computation, visualization, and logging.
    """
    
    def __init__(self,
                 config: Optional[EvaluationConfig] = None,
                 true_poles: Optional[List[float]] = None):
        """
        Initialize integrated evaluator.
        
        Args:
            config: Evaluation configuration
            true_poles: Ground truth pole locations
        """
        self.config = config or EvaluationConfig()
        self.true_poles = true_poles or []
        
        # Create components
        self.pole_evaluator = PoleEvaluator(
            true_poles=self.true_poles,
            near_threshold=self.config.near_pole_distance,
            mid_threshold=self.config.mid_range_distance
        )
        
        if self.config.enable_visualization:
            self.visualizer = PoleVisualizer()
        
        # Tracking
        self.evaluation_count = 0
        self.metrics_history = []
        self.time_history = []
        
        # Create plot directory if needed
        if self.config.save_plots:
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self,
                       model: Any,
                       x_values: List[float],
                       return_intermediates: bool = False) -> Union[PoleMetrics, Tuple[PoleMetrics, Dict]]:
        """
        Evaluate a model on given inputs.
        
        Args:
            model: Model with forward() method
            x_values: Input values for evaluation
            return_intermediates: Whether to return intermediate values
            
        Returns:
            PoleMetrics or tuple of (PoleMetrics, intermediates)
        """
        start_time = time.time()
        
        # Collect model outputs
        y_values = []
        Q_values = []
        P_values = []
        tags = []
        
        for x_val in x_values:
            x = TRNode.constant(real(x_val))
            
            # Get model output
            if hasattr(model, 'forward_with_pole_detection'):
                result = model.forward_with_pole_detection(x)
                y = result['output']
                tag = result['tag']
                Q_abs = result.get('Q_abs', None)
            else:
                y, tag = model.forward(x)
                Q_abs = getattr(model, '_last_Q_abs', None)
            
            y_values.append(y)
            tags.append(tag)
            
            if Q_abs is not None:
                Q_values.append(Q_abs)
            
            # Try to get P value if available
            if hasattr(model, '_last_P_value'):
                P_values.append(model._last_P_value)
        
        # Detect poles from model if available
        predicted_poles = None
        if hasattr(model, 'get_pole_predictions'):
            pole_probs = model.get_pole_predictions(x_values)
            predicted_poles = [x for x, p in zip(x_values, pole_probs) if p > 0.5]
        
        # Compute metrics
        metrics = self.pole_evaluator.evaluate(
            x_values=x_values,
            y_values=y_values,
            Q_values=Q_values if Q_values else [1.0] * len(x_values),
            P_values=P_values if P_values else None,
            predicted_poles=predicted_poles
        )
        
        # Update tracking
        self.evaluation_count += 1
        elapsed_time = time.time() - start_time
        self.time_history.append(elapsed_time)
        
        # Convert to dictionary for history
        metrics_dict = self._metrics_to_dict(metrics)
        metrics_dict['evaluation_time'] = elapsed_time
        metrics_dict['evaluation_count'] = self.evaluation_count
        self.metrics_history.append(metrics_dict)
        
        # Visualization if enabled
        if self.config.enable_visualization and self.evaluation_count % self.config.plot_frequency == 0:
            self._create_visualizations(x_values, y_values, Q_values, tags, predicted_poles)
        
        # Logging
        if self.config.log_to_file:
            self._log_metrics(metrics_dict)
        
        if self.config.verbose:
            self._print_summary(metrics)
        
        if return_intermediates:
            intermediates = {
                'y_values': y_values,
                'Q_values': Q_values,
                'P_values': P_values,
                'tags': tags,
                'predicted_poles': predicted_poles
            }
            return metrics, intermediates
        
        return metrics
    
    def evaluate_checkpoint(self,
                           checkpoint_path: str,
                           x_values: List[float],
                           model_class: type,
                           **model_kwargs) -> PoleMetrics:
        """
        Evaluate a saved model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            x_values: Input values for evaluation
            model_class: Class of the model
            **model_kwargs: Arguments for model initialization
            
        Returns:
            PoleMetrics
        """
        # Load model
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        model = model_class(**model_kwargs)
        
        # Load state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Evaluate
        return self.evaluate_model(model, x_values)
    
    def _metrics_to_dict(self, metrics: PoleMetrics) -> Dict[str, Any]:
        """Convert PoleMetrics to dictionary."""
        # Use dataclass asdict but handle non-serializable types
        d = asdict(metrics)
        
        # Convert any numpy types to Python types
        for key, value in d.items():
            if hasattr(value, 'item'):  # numpy scalar
                d[key] = value.item()
            elif isinstance(value, (list, tuple)) and value and hasattr(value[0], 'item'):
                d[key] = [v.item() if hasattr(v, 'item') else v for v in value]
        
        return d
    
    def _create_visualizations(self,
                             x_values: List[float],
                             y_values: List[Any],
                             Q_values: List[float],
                             tags: List[TRTag],
                             predicted_poles: Optional[List[float]]) -> None:
        """Create and save visualizations."""
        if not self.config.enable_visualization:
            return
        
        base_path = Path(self.config.plot_dir) / f"eval_{self.evaluation_count:04d}"
        
        # Q comparison plot
        if Q_values:
            save_path = f"{base_path}_Q.png" if self.config.save_plots else None
            self.visualizer.plot_Q_comparison(
                x_values, Q_values,
                true_poles=self.true_poles,
                predicted_poles=predicted_poles,
                title=f"Q(x) at Evaluation {self.evaluation_count}",
                save_path=save_path
            )
        
        # Pole locations
        if self.true_poles or predicted_poles:
            save_path = f"{base_path}_poles.png" if self.config.save_plots else None
            self.visualizer.plot_pole_locations(
                x_range=(min(x_values), max(x_values)),
                true_poles=self.true_poles,
                predicted_poles=predicted_poles or [],
                title=f"Pole Locations at Evaluation {self.evaluation_count}",
                save_path=save_path
            )
        
        # Coverage breakdown
        save_path = f"{base_path}_coverage.png" if self.config.save_plots else None
        self.visualizer.plot_coverage_breakdown(
            x_values, tags, self.true_poles,
            self.config.near_pole_distance,
            self.config.mid_range_distance,
            title=f"Coverage Breakdown at Evaluation {self.evaluation_count}",
            save_path=save_path
        )
    
    def _log_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Log metrics to file."""
        if not self.config.log_to_file:
            return
        
        # Append to log file
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation': self.evaluation_count,
            'metrics': metrics_dict
        }
        
        # Read existing log if it exists
        log_path = Path(self.config.log_file)
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Append new entry
        log_data.append(log_entry)
        
        # Write back
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _print_summary(self, metrics: PoleMetrics) -> None:
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"Evaluation {self.evaluation_count} Summary")
        print(f"{'='*60}")
        
        print(f"\nPole Localization:")
        print(f"  PLE: {metrics.ple:.4f}")
        print(f"  Predicted poles: {metrics.predicted_pole_count}")
        print(f"  True poles: {metrics.actual_pole_count}")
        print(f"  Precision: {metrics.true_positive_poles / (metrics.true_positive_poles + metrics.false_positive_poles + 1e-10):.2%}")
        print(f"  Recall: {metrics.true_positive_poles / (metrics.true_positive_poles + metrics.false_negative_poles + 1e-10):.2%}")
        
        print(f"\nBehavior Metrics:")
        print(f"  Sign consistency: {metrics.sign_consistency:.2%}")
        print(f"  Asymptotic correlation: {metrics.slope_correlation:.3f}")
        print(f"  Residual error: {metrics.residual_error:.4f}")
        
        print(f"\nCoverage:")
        print(f"  Near poles: {metrics.coverage_near:.2%}")
        print(f"  Mid-range: {metrics.coverage_mid:.2%}")
        print(f"  Far: {metrics.coverage_far:.2%}")
        
        print(f"{'='*60}\n")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics over all evaluations.
        
        Returns:
            Dictionary of aggregated statistics
        """
        if not self.metrics_history:
            return {}
        
        # Latest metrics
        latest = self.metrics_history[-1]
        
        # Compute improvements
        if len(self.metrics_history) > 1:
            first = self.metrics_history[0]
            improvements = {
                'ple_improvement': first['ple'] - latest['ple'],
                'coverage_near_improvement': latest['coverage_near'] - first['coverage_near'],
                'sign_consistency_improvement': latest['sign_consistency'] - first['sign_consistency'],
            }
        else:
            improvements = {}
        
        # Best values
        best_ple = min(m['ple'] for m in self.metrics_history)
        best_coverage = max(m['coverage_near'] for m in self.metrics_history)
        
        return {
            'total_evaluations': self.evaluation_count,
            'latest_metrics': latest,
            'improvements': improvements,
            'best_ple': best_ple,
            'best_near_coverage': best_coverage,
            'average_eval_time': sum(self.time_history) / len(self.time_history)
        }
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """
        Plot training progress over all evaluations.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.config.enable_visualization and self.metrics_history:
            self.visualizer.create_summary_figure(
                self.metrics_history,
                title="Training Progress",
                save_path=save_path
            )
    
    def export_metrics(self, output_path: str) -> None:
        """
        Export all metrics to file.
        
        Args:
            output_path: Path for output file (JSON or CSV)
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            # Export as CSV
            import pandas as pd
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(output_path, index=False)
        else:
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump({
                    'configuration': asdict(self.config),
                    'true_poles': self.true_poles,
                    'metrics_history': self.metrics_history,
                    'summary': self.get_summary_statistics()
                }, f, indent=2)
        
        print(f"Metrics exported to {output_path}")


def create_evaluator(true_poles: List[float],
                    enable_viz: bool = True,
                    save_plots: bool = True,
                    plot_dir: str = "evaluation_plots") -> IntegratedEvaluator:
    """
    Factory function to create an evaluator with sensible defaults.
    
    Args:
        true_poles: Ground truth pole locations
        enable_viz: Whether to enable visualization
        save_plots: Whether to save plots
        plot_dir: Directory for plots
        
    Returns:
        Configured IntegratedEvaluator
    """
    config = EvaluationConfig(
        enable_visualization=enable_viz,
        save_plots=save_plots,
        plot_dir=plot_dir,
        plot_frequency=10,
        log_to_file=True,
        verbose=True
    )
    
    return IntegratedEvaluator(config, true_poles)


class TrainingLogger:
    """
    Logger for integrating metrics into training loops.
    
    Provides hooks for automatic evaluation during training.
    """
    
    def __init__(self,
                 evaluator: IntegratedEvaluator,
                 eval_frequency: int = 10,
                 eval_samples: Optional[List[float]] = None):
        """
        Initialize training logger.
        
        Args:
            evaluator: Evaluator instance
            eval_frequency: Evaluate every N epochs
            eval_samples: Fixed samples for evaluation
        """
        self.evaluator = evaluator
        self.eval_frequency = eval_frequency
        self.eval_samples = eval_samples
        
        if self.eval_samples is None:
            # Default evaluation grid
            self.eval_samples = list(np.linspace(-2, 2, 100))
        
        self.epoch = 0
        self.training_history = []
    
    def on_epoch_end(self,
                    model: Any,
                    epoch: int,
                    train_loss: float,
                    val_loss: Optional[float] = None) -> None:
        """
        Hook for end of epoch.
        
        Args:
            model: Model being trained
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Optional validation loss
        """
        self.epoch = epoch
        
        # Record training metrics
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Evaluate if needed
        if epoch % self.eval_frequency == 0:
            print(f"\nEvaluating at epoch {epoch}...")
            metrics = self.evaluator.evaluate_model(model, self.eval_samples)
            
            # Add to training history
            self.training_history[-1]['pole_metrics'] = self.evaluator.metrics_history[-1]
    
    def on_training_end(self, model: Any) -> None:
        """
        Hook for end of training.
        
        Args:
            model: Final trained model
        """
        print("\nFinal evaluation...")
        metrics = self.evaluator.evaluate_model(model, self.eval_samples)
        
        # Plot progress
        self.evaluator.plot_training_progress(
            save_path=f"{self.evaluator.config.plot_dir}/training_progress.png"
        )
        
        # Export results
        self.evaluator.export_metrics(f"{self.evaluator.config.plot_dir}/final_metrics.json")
        
        # Print summary
        summary = self.evaluator.get_summary_statistics()
        print(f"\nTraining Summary:")
        print(f"  Total epochs: {self.epoch}")
        print(f"  PLE improvement: {summary['improvements'].get('ple_improvement', 0):.4f}")
        print(f"  Best PLE: {summary['best_ple']:.4f}")
        print(f"  Best near-pole coverage: {summary['best_near_coverage']:.2%}")
