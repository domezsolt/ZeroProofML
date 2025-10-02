"""
Enhanced logging utilities for ZeroProofML training.

Provides structured logging with tag counts, coverage metrics,
gradient modes, and anti-illusion metrics.
"""

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..core import TRTag

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Single log entry with timestamp and metrics."""

    timestamp: float
    epoch: int
    step: int
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
        }


@dataclass
class TrainingSession:
    """Complete training session information."""

    session_id: str
    start_time: float
    config: Dict[str, Any]
    model_info: Dict[str, Any]
    logs: List[LogEntry]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "config": self.config,
            "model_info": self.model_info,
            "logs": [log.to_dict() for log in self.logs],
        }


class StructuredLogger:
    """
    Structured logger for ZeroProofML training.

    Captures tag counts, coverage, gradient modes, lambda values,
    and anti-illusion metrics in a structured format.
    """

    def __init__(
        self, run_dir: str, session_id: Optional[str] = None, auto_save_interval: int = 10
    ):
        """
        Initialize structured logger.

        Args:
            run_dir: Directory to save logs
            session_id: Unique session identifier
            auto_save_interval: Save logs every N entries
        """
        self.run_dir = run_dir
        self.session_id = session_id or f"session_{int(time.time())}"
        self.auto_save_interval = auto_save_interval

        # Create run directory
        os.makedirs(run_dir, exist_ok=True)

        # Initialize session
        self.session = TrainingSession(
            session_id=self.session_id, start_time=time.time(), config={}, model_info={}, logs=[]
        )

        # Tracking
        self.current_epoch = 0
        self.current_step = 0
        self.last_save_time = time.time()

        logger.info("Structured logger initialized: %s", self.session_id)
        logger.info("Log directory: %s", run_dir)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set training configuration."""
        self.session.config = config

    def set_model_info(self, model_info: Dict[str, Any]) -> None:
        """Set model information."""
        self.session.model_info = model_info

    def log_metrics(
        self, metrics: Dict[str, Any], epoch: Optional[int] = None, step: Optional[int] = None
    ) -> None:
        """
        Log metrics for current step.

        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch (uses internal counter if None)
            step: Current step (uses internal counter if None)
        """
        if epoch is not None:
            self.current_epoch = epoch
        if step is not None:
            self.current_step = step

        # Create log entry
        entry = LogEntry(
            timestamp=time.time(),
            epoch=self.current_epoch,
            step=self.current_step,
            metrics=metrics.copy(),
        )

        self.session.logs.append(entry)

        # Auto-save periodically
        if len(self.session.logs) % self.auto_save_interval == 0:
            self.save()

        self.current_step += 1

    def log_tag_distribution(self, tags: List[TRTag]) -> Dict[str, float]:
        """
        Log tag distribution and return counts.

        Args:
            tags: List of output tags

        Returns:
            Tag count dictionary
        """
        tag_counts = {"REAL": 0, "PINF": 0, "NINF": 0, "PHI": 0}

        for tag in tags:
            if tag == TRTag.REAL:
                tag_counts["REAL"] += 1
            elif tag == TRTag.PINF:
                tag_counts["PINF"] += 1
            elif tag == TRTag.NINF:
                tag_counts["NINF"] += 1
            elif tag == TRTag.PHI:
                tag_counts["PHI"] += 1

        # Convert to ratios
        total = len(tags)
        tag_ratios = {}
        for tag_name, count in tag_counts.items():
            tag_ratios[f"{tag_name}_ratio"] = count / total if total > 0 else 0.0
            tag_ratios[f"{tag_name}_count"] = count

        tag_ratios["total_samples"] = total
        # Provide both derived coverage and legacy key for tests
        derived_coverage = tag_counts["REAL"] / total if total > 0 else 0.0
        tag_ratios["coverage_from_tags"] = derived_coverage
        tag_ratios["coverage"] = derived_coverage

        return tag_ratios

    def log_gradient_info(
        self,
        gradient_mode: str,
        delta: Optional[float] = None,
        saturating_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Log gradient computation information.

        Args:
            gradient_mode: Current gradient mode
            delta: Current delta threshold (for hybrid mode)
            saturating_ratio: Ratio of saturating gradient computations

        Returns:
            Gradient info dictionary
        """
        grad_info = {
            "gradient_mode": gradient_mode,
            "delta": delta,
            "saturating_ratio": saturating_ratio,
        }

        return grad_info

    def log_coverage_metrics(
        self,
        coverage: float,
        lambda_rej: float,
        coverage_enforced: bool = False,
        target_coverage: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Log coverage control metrics.

        Args:
            coverage: Current REAL coverage ratio
            lambda_rej: Current rejection penalty
            coverage_enforced: Whether enforcement was triggered
            target_coverage: Target coverage ratio

        Returns:
            Coverage metrics dictionary
        """
        coverage_metrics = {
            "coverage": coverage,
            "lambda_rej": lambda_rej,
            "coverage_enforced": coverage_enforced,
        }

        if target_coverage is not None:
            coverage_metrics["target_coverage"] = target_coverage
            coverage_metrics["coverage_gap"] = target_coverage - coverage

        return coverage_metrics

    def log_anti_illusion_metrics(self, ai_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Log anti-illusion metrics.

        Args:
            ai_metrics: Anti-illusion metrics dictionary

        Returns:
            Processed AI metrics
        """
        # Filter out NaN/inf values for logging
        clean_metrics = {}
        for key, value in ai_metrics.items():
            if isinstance(value, (int, float)):
                if not (np.isnan(value) or np.isinf(value)):
                    clean_metrics[f"ai_{key}"] = value

        return clean_metrics

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of entire training session."""
        if not self.session.logs:
            return {"error": "No logs available"}

        # Compute summary statistics
        all_metrics = {}
        for log in self.session.logs:
            for key, value in log.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

        summary = {
            "session_info": {
                "session_id": self.session.session_id,
                "start_time": self.session.start_time,
                "duration": time.time() - self.session.start_time,
                "total_logs": len(self.session.logs),
                "final_epoch": self.session.logs[-1].epoch,
                "final_step": self.session.logs[-1].step,
            },
            "config": self.session.config,
            "model_info": self.session.model_info,
        }

        # Add metric summaries
        metric_summary = {}
        for key, values in all_metrics.items():
            if values:
                metric_summary[key] = {
                    "final": values[-1],
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "improving"
                    if values[-1] < values[0]
                    else "declining"
                    if values[-1] > values[0]
                    else "stable",
                }

        summary["metrics"] = metric_summary

        return summary

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save session to JSON file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = os.path.join(self.run_dir, f"{self.session_id}_logs.json")

        # Save session
        with open(filename, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)

        self.last_save_time = time.time()
        return filename

    def save_csv(self, filename: Optional[str] = None) -> str:
        """
        Save logs to CSV format.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = os.path.join(self.run_dir, f"{self.session_id}_metrics.csv")

        if not self.session.logs:
            return filename

        # Collect all metric keys
        all_keys = set()
        for log in self.session.logs:
            all_keys.update(log.metrics.keys())

        # Sort keys for consistent ordering
        metric_keys = sorted(all_keys)

        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["timestamp", "epoch", "step"] + metric_keys
            writer.writerow(header)

            # Data rows
            for log in self.session.logs:
                row = [log.timestamp, log.epoch, log.step]
                for key in metric_keys:
                    value = log.metrics.get(key, "")
                    row.append(value)
                writer.writerow(row)

        return filename

    def export_summary(self, filename: Optional[str] = None) -> str:
        """
        Export training summary.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = os.path.join(self.run_dir, f"{self.session_id}_summary.json")

        summary = self.get_training_summary()

        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)

        return filename


class MetricsAggregator:
    """
    Aggregates metrics across multiple training runs.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.aggregated_data = {}

    def collect_runs(self, pattern: str = "*_logs.json") -> List[str]:
        """
        Collect all log files matching pattern.

        Args:
            pattern: File pattern to match

        Returns:
            List of log file paths
        """
        import glob

        search_pattern = os.path.join(self.base_dir, "**", pattern)
        log_files = glob.glob(search_pattern, recursive=True)

        logger.info("Found %d log files", len(log_files))
        return log_files

    def aggregate_metrics(self, log_files: List[str]) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple runs.

        Args:
            log_files: List of log file paths

        Returns:
            Aggregated metrics
        """
        all_sessions = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    session_data = json.load(f)
                all_sessions.append(session_data)
            except Exception as e:
                logger.exception("Failed to load %s: %s", log_file, e)
                continue

        if not all_sessions:
            return {}

        # Group by configuration
        config_groups = {}
        for session in all_sessions:
            config_key = json.dumps(session.get("config", {}), sort_keys=True)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(session)

        # Aggregate each group
        aggregated = {}
        for config_key, sessions in config_groups.items():
            group_name = f"config_{len(aggregated)}"

            # Extract final metrics from each session
            final_metrics = []
            for session in sessions:
                if session.get("logs"):
                    final_log = session["logs"][-1]
                    final_metrics.append(final_log["metrics"])

            if final_metrics:
                # Compute statistics
                aggregated[group_name] = {
                    "n_runs": len(sessions),
                    "config": json.loads(config_key),
                    "metrics_stats": self._compute_metric_stats(final_metrics),
                }

        self.aggregated_data = aggregated
        return aggregated

    def _compute_metric_stats(self, metrics_list: List[Dict]) -> Dict[str, Dict]:
        """Compute statistics for each metric across runs."""
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        stats = {}
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                value = metrics.get(key)
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    values.append(value)

            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "n_runs": len(values),
                }

        return stats

    def save_aggregated(self, filename: Optional[str] = None) -> str:
        """Save aggregated metrics."""
        if filename is None:
            filename = os.path.join(self.base_dir, "aggregated_metrics.json")

        with open(filename, "w") as f:
            json.dump(self.aggregated_data, f, indent=2)

        return filename


class ExperimentTracker:
    """
    High-level experiment tracking with automatic organization.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        self.current_experiment = None
        self.experiment_history = []

    def start_experiment(
        self, name: str, config: Dict[str, Any], model_info: Dict[str, Any]
    ) -> StructuredLogger:
        """
        Start a new experiment.

        Args:
            name: Experiment name
            config: Training configuration
            model_info: Model information

        Returns:
            Configured logger for this experiment
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.base_dir, f"{timestamp}_{name}")

        # Create structured logger for this experiment
        slogger = StructuredLogger(exp_dir)
        slogger.set_config(config)
        slogger.set_model_info(model_info)

        self.current_experiment = {
            "name": name,
            "directory": exp_dir,
            "logger": slogger,
            "start_time": time.time(),
        }

        logger.info("Started experiment: %s", name)
        logger.info("Directory: %s", exp_dir)

        return slogger

    def finish_experiment(self) -> Optional[str]:
        """
        Finish current experiment and save final results.

        Returns:
            Path to saved summary or None
        """
        if not self.current_experiment:
            return None

        slogger = self.current_experiment["logger"]

        # Save final logs
        log_file = slogger.save()
        csv_file = slogger.save_csv()
        summary_file = slogger.export_summary()

        # Add to history
        self.experiment_history.append(
            {
                "name": self.current_experiment["name"],
                "directory": self.current_experiment["directory"],
                "duration": time.time() - self.current_experiment["start_time"],
                "log_file": log_file,
                "csv_file": csv_file,
                "summary_file": summary_file,
            }
        )

        exp_name = self.current_experiment["name"]
        self.current_experiment = None

        logger.info("Finished experiment: %s", exp_name)
        logger.info("Files saved:")
        logger.info("  - Logs: %s", log_file)
        logger.info("  - CSV: %s", csv_file)
        logger.info("  - Summary: %s", summary_file)

        return summary_file

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all completed experiments."""
        return self.experiment_history.copy()

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            "total_experiments": len(self.experiment_history),
            "base_directory": self.base_dir,
            "experiments": self.experiment_history,
        }


# Global experiment tracker instance
_global_tracker = None


def get_experiment_tracker(base_dir: str = "runs") -> ExperimentTracker:
    """Get global experiment tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker(base_dir)
    return _global_tracker


def log_training_step(
    logger: StructuredLogger,
    epoch: int,
    step: int,
    loss: float,
    tags: List[TRTag],
    coverage: float,
    lambda_rej: float,
    gradient_mode: str,
    delta: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience function to log a complete training step.

    Args:
        logger: Structured logger instance
        epoch: Current epoch
        step: Current step
        loss: Training loss
        tags: Output tags for tag distribution
        coverage: REAL coverage ratio
        lambda_rej: Current rejection penalty
        gradient_mode: Current gradient mode
        delta: Current delta threshold
        additional_metrics: Additional metrics to log
    """
    # Collect all metrics
    metrics = {
        "loss": loss,
        "coverage": coverage,
        "lambda_rej": lambda_rej,
        "gradient_mode": gradient_mode,
    }

    if delta is not None:
        metrics["delta"] = delta

    # Add tag distribution without overwriting existing keys
    tag_metrics = logger.log_tag_distribution(tags)
    for k, v in tag_metrics.items():
        if k not in metrics:
            metrics[k] = v

    # Add any additional metrics
    if additional_metrics:
        metrics.update(additional_metrics)

    # Log everything
    logger.log_metrics(metrics, epoch, step)


# Import numpy for statistics
import numpy as np
