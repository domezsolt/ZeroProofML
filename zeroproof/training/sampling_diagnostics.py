"""
Advanced sampling strategies and diagnostic tools for pole-aware training.

This module implements importance sampling, active sampling near poles,
and comprehensive diagnostic monitoring for training near singularities.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path

from ..core import TRTag, real, pinf, ninf, phi


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    UNIFORM = "uniform"
    IMPORTANCE = "importance"  # Weight by 1/|Q(x)|²
    ACTIVE = "active"  # Adaptive grid refinement
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class ImportanceSamplerConfig:
    """Configuration for importance sampling."""
    # Weighting function
    weight_power: float = 2.0  # Use 1/|Q(x)|^power
    
    # Clipping for numerical stability
    min_q_abs: float = 1e-6  # Minimum |Q(x)| to prevent explosion
    max_weight: float = 100.0  # Maximum weight per sample
    
    # Temperature for softmax weighting
    temperature: float = 1.0
    
    # Resampling
    resample_ratio: float = 0.5  # Fraction to resample each epoch
    resample_frequency: int = 5  # Resample every N epochs
    
    # Batch construction
    importance_batch_ratio: float = 0.7  # Fraction of batch from importance sampling


class ImportanceSampler:
    """
    Importance sampler with weight proportional to 1/|Q(x)|².
    
    Focuses sampling on regions near poles where Q(x) ≈ 0.
    """
    
    def __init__(self, config: Optional[ImportanceSamplerConfig] = None):
        """
        Initialize importance sampler.
        
        Args:
            config: Sampler configuration
        """
        self.config = config or ImportanceSamplerConfig()
        
        # Cache for Q values
        self.q_cache = {}
        self.weight_cache = {}
        
        # Statistics
        self.sample_counts = defaultdict(int)
        self.total_samples = 0
        
    def compute_weights(self, 
                       x_values: torch.Tensor,
                       q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights from Q values.
        
        Args:
            x_values: Input values [batch_size, ...]
            q_values: Q(x) values [batch_size]
            
        Returns:
            Importance weights [batch_size]
        """
        # Compute |Q(x)|
        q_abs = torch.abs(q_values)
        
        # Clip for stability
        q_abs = torch.clamp(q_abs, min=self.config.min_q_abs)
        
        # Compute weights: 1/|Q(x)|^power
        weights = 1.0 / torch.pow(q_abs, self.config.weight_power)
        
        # Clip maximum weight
        weights = torch.clamp(weights, max=self.config.max_weight)
        
        # Apply temperature
        if self.config.temperature != 1.0:
            weights = torch.pow(weights, 1.0 / self.config.temperature)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Cache
        for i, x in enumerate(x_values):
            key = self._get_cache_key(x)
            self.q_cache[key] = q_values[i].item()
            self.weight_cache[key] = weights[i].item()
        
        return weights
    
    def sample_batch(self, 
                    x_pool: torch.Tensor,
                    q_pool: torch.Tensor,
                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample batch using importance weights.
        
        Args:
            x_pool: Pool of input values
            q_pool: Corresponding Q values
            batch_size: Desired batch size
            
        Returns:
            Tuple of (sampled inputs, sample indices)
        """
        # Compute weights
        weights = self.compute_weights(x_pool, q_pool)
        
        # Sample indices
        indices = torch.multinomial(weights, batch_size, replacement=True)
        
        # Get samples
        batch_x = x_pool[indices]
        
        # Update statistics
        for idx in indices:
            self.sample_counts[idx.item()] += 1
        self.total_samples += batch_size
        
        return batch_x, indices
    
    def hybrid_sample(self, 
                     x_pool: torch.Tensor,
                     q_pool: torch.Tensor,
                     batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hybrid sampling combining uniform and importance sampling.
        
        Args:
            x_pool: Pool of input values
            q_pool: Corresponding Q values
            batch_size: Desired batch size
            
        Returns:
            Tuple of (sampled inputs, sample indices)
        """
        # Split batch
        n_importance = int(batch_size * self.config.importance_batch_ratio)
        n_uniform = batch_size - n_importance
        
        # Importance sampling
        if n_importance > 0:
            weights = self.compute_weights(x_pool, q_pool)
            importance_indices = torch.multinomial(weights, n_importance, replacement=True)
        else:
            importance_indices = torch.tensor([], dtype=torch.long)
        
        # Uniform sampling
        if n_uniform > 0:
            uniform_indices = torch.randint(0, len(x_pool), (n_uniform,))
        else:
            uniform_indices = torch.tensor([], dtype=torch.long)
        
        # Combine
        indices = torch.cat([importance_indices, uniform_indices])
        indices = indices[torch.randperm(len(indices))]  # Shuffle
        
        batch_x = x_pool[indices]
        
        return batch_x, indices
    
    def _get_cache_key(self, x: torch.Tensor) -> str:
        """Get cache key for input value."""
        if x.dim() == 0:
            return f"{x.item():.6f}"
        else:
            return f"{x.flatten()[:3].tolist()}"  # Use first 3 values
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        if self.total_samples == 0:
            return {}
        
        counts = list(self.sample_counts.values())
        
        return {
            'total_samples': self.total_samples,
            'unique_samples': len(self.sample_counts),
            'max_sample_count': max(counts) if counts else 0,
            'min_sample_count': min(counts) if counts else 0,
            'sample_diversity': len(self.sample_counts) / max(1, self.total_samples),
            'mean_q_abs': np.mean(list(self.q_cache.values())) if self.q_cache else 0,
            'min_q_abs': min(self.q_cache.values()) if self.q_cache else 0,
        }


@dataclass
class ActiveSamplerConfig:
    """Configuration for active sampling with grid refinement."""
    # Initial grid
    initial_grid_size: int = 100
    grid_bounds: Tuple[float, float] = (-3.0, 3.0)
    
    # Refinement
    refinement_threshold: float = 0.1  # |Q(x)| threshold for refinement
    max_refinement_level: int = 5  # Maximum refinement depth
    refinement_ratio: int = 3  # Points to add between existing points
    
    # Adaptive
    adapt_frequency: int = 10  # Adapt grid every N epochs
    min_points_per_region: int = 5  # Minimum points in refined region
    
    # Memory limit
    max_grid_points: int = 10000  # Maximum total grid points


class ActiveSampler:
    """
    Active sampler with adaptive grid refinement near poles.
    
    Dynamically refines sampling grid in regions where |Q(x)| is small.
    """
    
    def __init__(self, 
                 config: Optional[ActiveSamplerConfig] = None,
                 input_dim: int = 1):
        """
        Initialize active sampler.
        
        Args:
            config: Sampler configuration
            input_dim: Input dimension
        """
        self.config = config or ActiveSamplerConfig()
        self.input_dim = input_dim
        
        # Initialize grid
        self.grid_points = self._initialize_grid()
        self.refinement_levels = {self._point_key(p): 0 for p in self.grid_points}
        
        # Refinement history
        self.refinement_history = []
        self.q_history = {}
        
    def _initialize_grid(self) -> List[torch.Tensor]:
        """Initialize uniform grid."""
        if self.input_dim == 1:
            points = torch.linspace(
                self.config.grid_bounds[0],
                self.config.grid_bounds[1],
                self.config.initial_grid_size
            ).unsqueeze(1)
        else:
            # Multi-dimensional grid (simplified - random for now)
            points = torch.rand(self.config.initial_grid_size, self.input_dim)
            points = points * (self.config.grid_bounds[1] - self.config.grid_bounds[0])
            points = points + self.config.grid_bounds[0]
        
        return points.tolist()
    
    def refine_grid(self, q_values: Dict[str, float]):
        """
        Refine grid based on Q values.
        
        Args:
            q_values: Dictionary mapping point keys to |Q(x)| values
        """
        # Update Q history
        self.q_history.update(q_values)
        
        # Find points needing refinement
        refinement_points = []
        
        for point in self.grid_points:
            key = self._point_key(point)
            
            if key not in q_values:
                continue
            
            q_abs = abs(q_values[key])
            level = self.refinement_levels.get(key, 0)
            
            # Check if needs refinement
            if (q_abs < self.config.refinement_threshold and 
                level < self.config.max_refinement_level and
                len(self.grid_points) < self.config.max_grid_points):
                refinement_points.append((point, level))
        
        # Refine around selected points
        new_points = []
        for point, level in refinement_points:
            neighbors = self._generate_neighbors(point, level)
            new_points.extend(neighbors)
            
            # Update refinement level
            for neighbor in neighbors:
                key = self._point_key(neighbor)
                self.refinement_levels[key] = level + 1
        
        # Add new points to grid
        self.grid_points.extend(new_points)
        
        # Record refinement
        self.refinement_history.append({
            'n_refined': len(refinement_points),
            'n_added': len(new_points),
            'total_points': len(self.grid_points),
        })
    
    def _generate_neighbors(self, 
                           point: torch.Tensor,
                           level: int) -> List[torch.Tensor]:
        """
        Generate neighbor points for refinement.
        
        Args:
            point: Center point
            level: Current refinement level
            
        Returns:
            List of neighbor points
        """
        neighbors = []
        
        # Distance based on refinement level
        distance = (self.config.grid_bounds[1] - self.config.grid_bounds[0]) / \
                  (self.config.initial_grid_size * (2 ** (level + 1)))
        
        if self.input_dim == 1:
            # 1D: Add points on both sides
            for i in range(1, self.config.refinement_ratio + 1):
                offset = i * distance / (self.config.refinement_ratio + 1)
                neighbors.append(point + offset)
                neighbors.append(point - offset)
        else:
            # Multi-dimensional: Add random points in neighborhood
            for _ in range(self.config.refinement_ratio * 2):
                offset = torch.randn(self.input_dim) * distance
                neighbors.append(point + offset)
        
        return neighbors
    
    def _point_key(self, point: torch.Tensor) -> str:
        """Get string key for point."""
        if isinstance(point, list):
            point = torch.tensor(point)
        if point.dim() == 0:
            return f"{point.item():.6f}"
        else:
            return f"{point.flatten().tolist()}"
    
    def get_current_grid(self) -> torch.Tensor:
        """Get current grid points as tensor."""
        return torch.stack([torch.tensor(p) if isinstance(p, list) else p 
                           for p in self.grid_points])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get grid refinement statistics."""
        if self.q_history:
            q_values = list(self.q_history.values())
            near_pole = sum(1 for q in q_values if abs(q) < self.config.refinement_threshold)
        else:
            q_values = []
            near_pole = 0
        
        return {
            'total_points': len(self.grid_points),
            'max_refinement_level': max(self.refinement_levels.values()) if self.refinement_levels else 0,
            'refinement_events': len(self.refinement_history),
            'near_pole_points': near_pole,
            'mean_q_abs': np.mean(q_values) if q_values else 0,
        }


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic monitoring."""
    # Export settings
    export_frequency: int = 10  # Export every N epochs
    export_path: str = "diagnostics"
    
    # Monitoring
    track_gradients: bool = True
    gradient_percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 90, 95, 99])
    
    # History limits
    max_history_length: int = 1000
    
    # Pole distance thresholds
    near_pole_threshold: float = 0.1  # |Q(x)| < threshold is "near pole"
    far_pole_threshold: float = 1.0  # |Q(x)| > threshold is "far from pole"
    
    # Tag tracking
    track_tag_distribution: bool = True
    
    # Stability monitoring
    monitor_q_min: bool = True
    q_min_warning_threshold: float = 1e-4


class DiagnosticMonitor:
    """
    Comprehensive diagnostic monitoring for training near poles.
    
    Tracks P(x), Q(x), q_min, gradients, and other critical metrics.
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize diagnostic monitor.
        
        Args:
            config: Diagnostic configuration
        """
        self.config = config or DiagnosticConfig()
        
        # Create export directory
        self.export_dir = Path(self.config.export_path)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # History with stable keys
        self.history = {
            'epoch': [],
            'lambda_rej': [],
            'coverage_train': [],
            'coverage_eval': [],
            'coverage_eval_tau': [],
            'q_min': [],
            'q_mean': [],
            'q_std': [],
            'loss_train': [],
            'loss_eval': [],
        }
        
        # Tag distribution tracking
        self.tag_history = {
            'n_real': [],
            'n_pinf': [],
            'n_ninf': [],
            'n_phi': [],
            'real_ratio': [],
        }
        
        # Gradient tracking
        self.gradient_history = {
            'near_pole_mean': [],
            'near_pole_max': [],
            'far_pole_mean': [],
            'far_pole_max': [],
            'gradient_ratio': [],  # near/far ratio
        }
        
        # P(x), Q(x) snapshots
        self.pq_snapshots = []
        
        # Batch-wise metrics
        self.batch_metrics = defaultdict(list)
        
    def update(self, 
              epoch: int,
              metrics: Dict[str, Any],
              p_values: Optional[torch.Tensor] = None,
              q_values: Optional[torch.Tensor] = None,
              gradients: Optional[Dict[str, torch.Tensor]] = None,
              tags: Optional[List[TRTag]] = None):
        """
        Update diagnostic history.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics to log
            p_values: P(x) values for current batch
            q_values: Q(x) values for current batch
            gradients: Gradient dictionary
            tags: Output tags for current batch
        """
        # Update epoch
        self.history['epoch'].append(epoch)
        
        # Update standard metrics with stable keys
        for key in ['lambda_rej', 'coverage_train', 'coverage_eval', 
                   'coverage_eval_tau', 'loss_train', 'loss_eval']:
            if key in metrics:
                self.history[key].append(metrics[key])
            else:
                # Maintain continuity with None or last value
                if self.history[key]:
                    self.history[key].append(self.history[key][-1])
                else:
                    self.history[key].append(None)
        
        # Update Q statistics
        if q_values is not None:
            q_abs = torch.abs(q_values)
            self.history['q_min'].append(q_abs.min().item())
            self.history['q_mean'].append(q_abs.mean().item())
            self.history['q_std'].append(q_abs.std().item())
            
            # Check q_min warning
            if self.config.monitor_q_min:
                q_min = q_abs.min().item()
                if q_min < self.config.q_min_warning_threshold:
                    print(f"WARNING: q_min = {q_min:.2e} below threshold {self.config.q_min_warning_threshold:.2e}")
        
        # Update tag distribution
        if tags and self.config.track_tag_distribution:
            self._update_tag_distribution(tags)
        
        # Update gradient statistics
        if gradients and self.config.track_gradients:
            self._update_gradient_statistics(gradients, q_values)
        
        # Store P(x), Q(x) snapshot
        if p_values is not None and q_values is not None:
            if epoch % self.config.export_frequency == 0:
                self.pq_snapshots.append({
                    'epoch': epoch,
                    'p_values': p_values.detach().cpu().numpy(),
                    'q_values': q_values.detach().cpu().numpy(),
                })
        
        # Limit history length
        self._trim_history()
        
        # Export if needed
        if epoch % self.config.export_frequency == 0:
            self.export_diagnostics(epoch)
    
    def update_batch(self, 
                    batch_idx: int,
                    q_min: float,
                    tags: Optional[List[TRTag]] = None,
                    loss: Optional[float] = None):
        """
        Update batch-wise metrics.
        
        Args:
            batch_idx: Batch index
            q_min: Minimum |Q(x)| in batch
            tags: Output tags
            loss: Batch loss
        """
        self.batch_metrics['batch_idx'].append(batch_idx)
        self.batch_metrics['q_min'].append(q_min)
        
        if loss is not None:
            self.batch_metrics['loss'].append(loss)
        
        if tags:
            n_real = sum(1 for t in tags if t == TRTag.REAL)
            n_pinf = sum(1 for t in tags if t == TRTag.PINF)
            n_ninf = sum(1 for t in tags if t == TRTag.NINF)
            n_phi = sum(1 for t in tags if t == TRTag.PHI)
            
            self.batch_metrics['n_real'].append(n_real)
            self.batch_metrics['n_non_real'].append(n_pinf + n_ninf + n_phi)
    
    def _update_tag_distribution(self, tags: List[TRTag]):
        """Update tag distribution statistics."""
        n_total = len(tags)
        n_real = sum(1 for t in tags if t == TRTag.REAL)
        n_pinf = sum(1 for t in tags if t == TRTag.PINF)
        n_ninf = sum(1 for t in tags if t == TRTag.NINF)
        n_phi = sum(1 for t in tags if t == TRTag.PHI)
        
        self.tag_history['n_real'].append(n_real)
        self.tag_history['n_pinf'].append(n_pinf)
        self.tag_history['n_ninf'].append(n_ninf)
        self.tag_history['n_phi'].append(n_phi)
        self.tag_history['real_ratio'].append(n_real / max(1, n_total))
        
        # Log actual non-REAL outputs
        n_non_real = n_pinf + n_ninf + n_phi
        if n_non_real > 0:
            print(f"  Non-REAL outputs: PINF={n_pinf}, NINF={n_ninf}, PHI={n_phi} "
                  f"({n_non_real/n_total:.1%} of batch)")
    
    def _update_gradient_statistics(self, 
                                   gradients: Dict[str, torch.Tensor],
                                   q_values: Optional[torch.Tensor]):
        """Update gradient statistics near/far from poles."""
        if q_values is None:
            return
        
        # Classify samples by distance to pole
        q_abs = torch.abs(q_values)
        near_pole_mask = q_abs < self.config.near_pole_threshold
        far_pole_mask = q_abs > self.config.far_pole_threshold
        
        # Compute gradient magnitudes
        grad_norms = []
        for name, grad in gradients.items():
            if grad is not None:
                grad_norms.append(torch.norm(grad, dim=-1))
        
        if not grad_norms:
            return
        
        grad_norms = torch.stack(grad_norms).mean(0)  # Average across parameters
        
        # Compute statistics
        if near_pole_mask.any():
            near_grads = grad_norms[near_pole_mask]
            self.gradient_history['near_pole_mean'].append(near_grads.mean().item())
            self.gradient_history['near_pole_max'].append(near_grads.max().item())
        else:
            self.gradient_history['near_pole_mean'].append(0)
            self.gradient_history['near_pole_max'].append(0)
        
        if far_pole_mask.any():
            far_grads = grad_norms[far_pole_mask]
            self.gradient_history['far_pole_mean'].append(far_grads.mean().item())
            self.gradient_history['far_pole_max'].append(far_grads.max().item())
        else:
            self.gradient_history['far_pole_mean'].append(0)
            self.gradient_history['far_pole_max'].append(0)
        
        # Compute ratio
        if (self.gradient_history['far_pole_mean'][-1] > 0 and 
            self.gradient_history['near_pole_mean'][-1] > 0):
            ratio = (self.gradient_history['near_pole_mean'][-1] / 
                    self.gradient_history['far_pole_mean'][-1])
            self.gradient_history['gradient_ratio'].append(ratio)
        else:
            self.gradient_history['gradient_ratio'].append(1.0)
    
    def _trim_history(self):
        """Trim history to maximum length."""
        max_len = self.config.max_history_length
        
        for key in self.history:
            if len(self.history[key]) > max_len:
                self.history[key] = self.history[key][-max_len:]
        
        for key in self.tag_history:
            if len(self.tag_history[key]) > max_len:
                self.tag_history[key] = self.tag_history[key][-max_len:]
        
        for key in self.gradient_history:
            if len(self.gradient_history[key]) > max_len:
                self.gradient_history[key] = self.gradient_history[key][-max_len:]
    
    def export_diagnostics(self, epoch: int):
        """
        Export diagnostics to files.
        
        Args:
            epoch: Current epoch
        """
        # Export history as JSON
        history_path = self.export_dir / f"history_epoch_{epoch}.json"
        with open(history_path, 'w') as f:
            json.dump({
                'main': self.history,
                'tags': self.tag_history,
                'gradients': self.gradient_history,
            }, f, indent=2, default=str)
        
        # Export P(x), Q(x) snapshots
        if self.pq_snapshots:
            pq_path = self.export_dir / f"pq_snapshots_epoch_{epoch}.pkl"
            with open(pq_path, 'wb') as f:
                pickle.dump(self.pq_snapshots[-5:], f)  # Last 5 snapshots
        
        # Export batch metrics
        if self.batch_metrics:
            batch_path = self.export_dir / f"batch_metrics_epoch_{epoch}.json"
            with open(batch_path, 'w') as f:
                json.dump(dict(self.batch_metrics), f, indent=2, default=str)
        
        print(f"  Diagnostics exported to {self.export_dir}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        summary = {}
        
        # Q statistics
        if self.history['q_min']:
            summary['q_min_global'] = min(self.history['q_min'])
            summary['q_min_recent'] = self.history['q_min'][-1]
            summary['q_mean_recent'] = self.history['q_mean'][-1] if self.history['q_mean'] else 0
        
        # Coverage
        if self.history['coverage_train']:
            summary['coverage_current'] = self.history['coverage_train'][-1]
            summary['coverage_mean'] = np.mean(self.history['coverage_train'][-10:])
        
        # Tags
        if self.tag_history['real_ratio']:
            summary['real_ratio_current'] = self.tag_history['real_ratio'][-1]
            summary['non_real_total'] = sum([
                sum(self.tag_history['n_pinf']),
                sum(self.tag_history['n_ninf']),
                sum(self.tag_history['n_phi'])
            ])
        
        # Gradients
        if self.gradient_history['gradient_ratio']:
            summary['gradient_ratio_mean'] = np.mean(self.gradient_history['gradient_ratio'][-10:])
            summary['gradient_ratio_max'] = max(self.gradient_history['gradient_ratio'])
        
        return summary


class IntegratedSampler:
    """
    Integrated sampler combining importance and active sampling.
    
    Provides unified interface for advanced sampling strategies.
    """
    
    def __init__(self,
                 strategy: SamplingStrategy = SamplingStrategy.HYBRID,
                 importance_config: Optional[ImportanceSamplerConfig] = None,
                 active_config: Optional[ActiveSamplerConfig] = None,
                 diagnostic_config: Optional[DiagnosticConfig] = None):
        """
        Initialize integrated sampler.
        
        Args:
            strategy: Sampling strategy to use
            importance_config: Importance sampler configuration
            active_config: Active sampler configuration
            diagnostic_config: Diagnostic configuration
        """
        self.strategy = strategy
        
        # Initialize samplers
        self.importance_sampler = ImportanceSampler(importance_config)
        self.active_sampler = ActiveSampler(active_config)
        self.diagnostic_monitor = DiagnosticMonitor(diagnostic_config)
        
        # Statistics
        self.sampling_history = []
        
    def sample_batch(self,
                    x_pool: torch.Tensor,
                    q_pool: torch.Tensor,
                    batch_size: int,
                    epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample batch using configured strategy.
        
        Args:
            x_pool: Pool of input values
            q_pool: Corresponding Q values
            batch_size: Desired batch size
            epoch: Current epoch
            
        Returns:
            Tuple of (batch samples, sampling info)
        """
        info = {'strategy': self.strategy.value}
        
        if self.strategy == SamplingStrategy.UNIFORM:
            # Uniform sampling
            indices = torch.randint(0, len(x_pool), (batch_size,))
            batch_x = x_pool[indices]
            
        elif self.strategy == SamplingStrategy.IMPORTANCE:
            # Pure importance sampling
            batch_x, indices = self.importance_sampler.sample_batch(
                x_pool, q_pool, batch_size
            )
            info['importance_stats'] = self.importance_sampler.get_statistics()
            
        elif self.strategy == SamplingStrategy.ACTIVE:
            # Active sampling with grid refinement
            if epoch % self.active_sampler.config.adapt_frequency == 0:
                # Refine grid
                q_dict = {self.active_sampler._point_key(x_pool[i]): q_pool[i].item() 
                         for i in range(len(x_pool))}
                self.active_sampler.refine_grid(q_dict)
            
            # Sample from refined grid
            grid = self.active_sampler.get_current_grid()
            indices = torch.randint(0, len(grid), (batch_size,))
            batch_x = grid[indices]
            info['active_stats'] = self.active_sampler.get_statistics()
            
        else:  # HYBRID
            # Combine importance and active sampling
            batch_x, indices = self.importance_sampler.hybrid_sample(
                x_pool, q_pool, batch_size
            )
            
            # Periodically refine grid
            if epoch % self.active_sampler.config.adapt_frequency == 0:
                q_dict = {self.active_sampler._point_key(x_pool[i]): q_pool[i].item() 
                         for i in range(min(100, len(x_pool)))}
                self.active_sampler.refine_grid(q_dict)
            
            info['importance_stats'] = self.importance_sampler.get_statistics()
            info['active_stats'] = self.active_sampler.get_statistics()
        
        # Record sampling
        self.sampling_history.append({
            'epoch': epoch,
            'batch_size': batch_size,
            'strategy': self.strategy.value,
            **info
        })
        
        return batch_x, info
    
    def update_diagnostics(self, **kwargs):
        """Update diagnostic monitor."""
        self.diagnostic_monitor.update(**kwargs)
    
    def update_batch_diagnostics(self, **kwargs):
        """Update batch-wise diagnostics."""
        self.diagnostic_monitor.update_batch(**kwargs)
    
    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        return self.diagnostic_monitor.get_summary()
    
    def export_all(self, epoch: int):
        """Export all diagnostics and sampling history."""
        self.diagnostic_monitor.export_diagnostics(epoch)
        
        # Export sampling history
        if self.sampling_history:
            path = self.diagnostic_monitor.export_dir / f"sampling_history_epoch_{epoch}.json"
            with open(path, 'w') as f:
                json.dump(self.sampling_history[-100:], f, indent=2, default=str)


def create_integrated_sampler(strategy: str = "hybrid",
                             weight_power: float = 2.0,
                             export_path: str = "diagnostics",
                             **kwargs) -> IntegratedSampler:
    """
    Factory function to create integrated sampler.
    
    Args:
        strategy: Sampling strategy name
        weight_power: Power for importance weighting (1/|Q|^power)
        export_path: Path for diagnostic exports
        **kwargs: Additional configuration
        
    Returns:
        Configured integrated sampler
    """
    # Parse strategy
    strategy_enum = SamplingStrategy(strategy)
    
    # Configure importance sampler
    importance_config = ImportanceSamplerConfig(
        weight_power=weight_power,
        temperature=kwargs.get('temperature', 1.0),
        importance_batch_ratio=kwargs.get('importance_ratio', 0.7),
    )
    
    # Configure active sampler
    active_config = ActiveSamplerConfig(
        initial_grid_size=kwargs.get('grid_size', 100),
        refinement_threshold=kwargs.get('refinement_threshold', 0.1),
        max_refinement_level=kwargs.get('max_refinement', 5),
    )
    
    # Configure diagnostics
    diagnostic_config = DiagnosticConfig(
        export_path=export_path,
        export_frequency=kwargs.get('export_frequency', 10),
        track_gradients=kwargs.get('track_gradients', True),
        track_tag_distribution=kwargs.get('track_tags', True),
    )
    
    return IntegratedSampler(
        strategy=strategy_enum,
        importance_config=importance_config,
        active_config=active_config,
        diagnostic_config=diagnostic_config,
    )
