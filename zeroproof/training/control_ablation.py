"""
Ablation studies for control strategies.

This module provides tools for comparing different control strategies
for λ_rej adjustment and curriculum learning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .advanced_control import (
    ControlStrategy,
    CurriculumConfig,
    CurriculumScheduler,
    HybridController,
    PIController,
    PIControllerConfig,
)


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    # Strategies to compare
    strategies: List[ControlStrategy] = None

    # Simulation parameters
    n_epochs: int = 100
    n_samples_per_epoch: int = 100

    # Coverage simulation
    base_coverage: float = 0.5  # Starting coverage
    coverage_noise: float = 0.05  # Noise in coverage
    coverage_drift: float = 0.002  # Natural drift per epoch

    # Response characteristics
    lambda_sensitivity: float = 0.05  # How coverage responds to λ
    response_delay: int = 2  # Epochs delay in response

    # Target
    target_coverage: float = 0.85

    # Visualization
    plot_results: bool = True
    save_plots: bool = False
    plot_dir: str = "ablation_plots"

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = [
                ControlStrategy.PROPORTIONAL,
                ControlStrategy.PI,
                ControlStrategy.DEAD_BAND,
            ]


class ControlSimulator:
    """
    Simulator for testing control strategies.

    Simulates coverage dynamics to compare different control approaches
    without requiring actual training.
    """

    def __init__(self, config: AblationConfig):
        """
        Initialize simulator.

        Args:
            config: Ablation configuration
        """
        self.config = config

        # State
        self.coverage = config.base_coverage
        self.lambda_history = []
        self.response_buffer = []

    def simulate_coverage_response(self, lambda_rej: float) -> float:
        """
        Simulate how coverage responds to λ_rej.

        Args:
            lambda_rej: Current rejection penalty

        Returns:
            New coverage value
        """
        # Add to response buffer (delayed response)
        self.response_buffer.append(lambda_rej)

        if len(self.response_buffer) > self.config.response_delay:
            effective_lambda = self.response_buffer.pop(0)
        else:
            effective_lambda = self.config.base_coverage

        # Coverage increases with λ (more rejection = higher coverage)
        lambda_effect = (effective_lambda - 1.0) * self.config.lambda_sensitivity

        # Add natural drift
        drift = self.config.coverage_drift

        # Add noise
        noise = np.random.normal(0, self.config.coverage_noise)

        # Update coverage with saturation
        self.coverage += lambda_effect + drift + noise
        self.coverage = np.clip(self.coverage, 0.0, 1.0)

        return self.coverage

    def reset(self):
        """Reset simulator state."""
        self.coverage = self.config.base_coverage
        self.lambda_history.clear()
        self.response_buffer.clear()


class AblationRunner:
    """
    Runs ablation studies comparing control strategies.
    """

    def __init__(self, config: Optional[AblationConfig] = None):
        """
        Initialize ablation runner.

        Args:
            config: Ablation configuration
        """
        self.config = config or AblationConfig()
        self.results = {}

        # Create plot directory if needed
        if self.config.save_plots:
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)

    def create_controller(self, strategy: ControlStrategy) -> PIController:
        """
        Create controller for given strategy.

        Args:
            strategy: Control strategy

        Returns:
            Configured controller
        """
        if strategy == ControlStrategy.PROPORTIONAL:
            config = PIControllerConfig(
                kp=1.0,
                ki=0.0,
                kd=0.0,
                dead_band=0.0,
                target_coverage=self.config.target_coverage,
            )

        elif strategy == ControlStrategy.PI:
            config = PIControllerConfig(
                kp=1.0,
                ki=0.1,
                kd=0.0,
                dead_band=0.0,
                target_coverage=self.config.target_coverage,
            )

        elif strategy == ControlStrategy.PID:
            config = PIControllerConfig(
                kp=1.0,
                ki=0.1,
                kd=0.05,
                dead_band=0.0,
                target_coverage=self.config.target_coverage,
            )

        elif strategy == ControlStrategy.DEAD_BAND:
            config = PIControllerConfig(
                kp=1.0,
                ki=0.1,
                kd=0.0,
                dead_band=0.02,
                target_coverage=self.config.target_coverage,
            )

        elif strategy == ControlStrategy.ADAPTIVE:
            config = PIControllerConfig(
                kp=1.0,
                ki=0.1,
                kd=0.0,
                dead_band=0.01,
                adaptive_gains=True,
                gain_adaptation_rate=0.01,
                target_coverage=self.config.target_coverage,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return PIController(config)

    def run_single_strategy(self, strategy: ControlStrategy) -> Dict[str, Any]:
        """
        Run ablation for single strategy.

        Args:
            strategy: Control strategy to test

        Returns:
            Results dictionary
        """
        # Create controller and simulator
        controller = self.create_controller(strategy)
        simulator = ControlSimulator(self.config)

        # History
        coverage_history = []
        lambda_history = []
        error_history = []

        # Run simulation
        for epoch in range(self.config.n_epochs):
            # Get current coverage
            coverage = simulator.coverage
            coverage_history.append(coverage)

            # Compute control output
            lambda_rej, info = controller.compute(coverage)
            lambda_history.append(lambda_rej)
            error_history.append(coverage - self.config.target_coverage)

            # Simulate response
            simulator.simulate_coverage_response(lambda_rej)

        # Compute metrics
        errors = np.array(error_history)
        lambdas = np.array(lambda_history)
        coverages = np.array(coverage_history)

        # Settling time (time to reach ±5% of target)
        settling_threshold = 0.05
        settled = np.abs(errors) < settling_threshold
        if settled.any():
            settling_time = np.argmax(settled)
        else:
            settling_time = self.config.n_epochs

        # Overshoot
        if (coverages > self.config.target_coverage).any():
            overshoot = np.max(coverages) - self.config.target_coverage
        else:
            overshoot = 0.0

        # Oscillations (count zero crossings in error)
        zero_crossings = np.sum(np.diff(np.sign(errors)) != 0)

        # Steady-state error (last 20% of simulation)
        steady_start = int(0.8 * len(errors))
        steady_state_error = np.mean(np.abs(errors[steady_start:]))

        # Control effort (total λ adjustment)
        control_effort = np.sum(np.abs(np.diff(lambdas)))

        return {
            "strategy": strategy.value,
            "coverage_history": coverage_history,
            "lambda_history": lambda_history,
            "error_history": error_history,
            "metrics": {
                "settling_time": settling_time,
                "overshoot": overshoot,
                "oscillations": zero_crossings,
                "steady_state_error": steady_state_error,
                "control_effort": control_effort,
                "final_coverage": coverages[-1],
                "mean_error": np.mean(np.abs(errors)),
                "std_error": np.std(errors),
            },
            "controller_stats": controller.get_statistics(),
        }

    def run_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation for all configured strategies.

        Returns:
            Results for all strategies
        """
        print(f"Running ablation study with {len(self.config.strategies)} strategies...")

        for strategy in self.config.strategies:
            print(f"  Testing {strategy.value}...")
            self.results[strategy.value] = self.run_single_strategy(strategy)

        return self.results

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of all strategies.

        Args:
            save_path: Optional path to save plot
        """
        if not self.results:
            print("No results to plot. Run ablation first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        epochs = range(self.config.n_epochs)

        # Coverage evolution
        ax = axes[0, 0]
        for name, result in self.results.items():
            ax.plot(epochs, result["coverage_history"], label=name, linewidth=2)
        ax.axhline(y=self.config.target_coverage, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Coverage")
        ax.set_title("Coverage Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # λ_rej evolution
        ax = axes[0, 1]
        for name, result in self.results.items():
            ax.plot(epochs, result["lambda_history"], label=name, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("λ_rej")
        ax.set_title("Control Output (λ_rej)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error evolution
        ax = axes[0, 2]
        for name, result in self.results.items():
            ax.plot(epochs, result["error_history"], label=name, linewidth=2)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title("Coverage Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Metrics comparison
        ax = axes[1, 0]
        metrics_to_compare = ["settling_time", "overshoot", "steady_state_error"]
        x_pos = np.arange(len(self.results))
        width = 0.25

        for i, metric in enumerate(metrics_to_compare):
            values = [r["metrics"][metric] for r in self.results.values()]
            ax.bar(x_pos + i * width, values, width, label=metric)

        ax.set_xlabel("Strategy")
        ax.set_ylabel("Value")
        ax.set_title("Performance Metrics")
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(list(self.results.keys()))
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Control effort
        ax = axes[1, 1]
        efforts = [r["metrics"]["control_effort"] for r in self.results.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(efforts)))
        ax.bar(list(self.results.keys()), efforts, color=colors)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Control Effort")
        ax.set_title("Total Control Effort")
        ax.grid(True, alpha=0.3, axis="y")

        # Final comparison table
        ax = axes[1, 2]
        ax.axis("tight")
        ax.axis("off")

        # Create comparison table
        table_data = []
        headers = ["Strategy", "Final Cov", "Mean Err", "Oscillations"]

        for name, result in self.results.items():
            m = result["metrics"]
            row = [
                name,
                f"{m['final_coverage']:.3f}",
                f"{m['mean_error']:.4f}",
                f"{m['oscillations']}",
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title("Summary Comparison")

        plt.suptitle("Control Strategy Ablation Study", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")

        if self.config.plot_results:
            plt.show()

    def get_best_strategy(self) -> Tuple[str, Dict[str, Any]]:
        """
        Determine best strategy based on metrics.

        Returns:
            Tuple of (strategy name, metrics)
        """
        if not self.results:
            return None, {}

        # Compute combined score (lower is better)
        scores = {}
        for name, result in self.results.items():
            m = result["metrics"]

            # Weighted combination of metrics
            score = (
                0.3 * m["settling_time"] / self.config.n_epochs
                + 0.2 * m["overshoot"]  # Normalized settling time
                + 0.3 * m["steady_state_error"]  # Overshoot penalty
                + 0.1 * m["oscillations"] / self.config.n_epochs  # Steady-state accuracy
                + 0.1 * m["control_effort"] / 100  # Normalized oscillations  # Normalized effort
            )
            scores[name] = score

        # Find best (lowest score)
        best_name = min(scores, key=scores.get)

        return best_name, self.results[best_name]["metrics"]

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate text report of ablation results.

        Args:
            save_path: Optional path to save report

        Returns:
            Report text
        """
        if not self.results:
            return "No results available. Run ablation first."

        report = []
        report.append("=" * 60)
        report.append("Control Strategy Ablation Report")
        report.append("=" * 60)
        report.append(f"\nConfiguration:")
        report.append(f"  Target coverage: {self.config.target_coverage}")
        report.append(f"  Simulation epochs: {self.config.n_epochs}")
        report.append(f"  Strategies tested: {', '.join(s.value for s in self.config.strategies)}")

        report.append(f"\n{'='*60}")
        report.append("Results by Strategy:")
        report.append("=" * 60)

        for name, result in self.results.items():
            m = result["metrics"]
            report.append(f"\n{name.upper()}:")
            report.append(f"  Settling time: {m['settling_time']} epochs")
            report.append(f"  Overshoot: {m['overshoot']:.4f}")
            report.append(f"  Oscillations: {m['oscillations']}")
            report.append(f"  Steady-state error: {m['steady_state_error']:.4f}")
            report.append(f"  Control effort: {m['control_effort']:.2f}")
            report.append(f"  Final coverage: {m['final_coverage']:.4f}")
            report.append(f"  Mean absolute error: {m['mean_error']:.4f}")

        # Best strategy
        best_name, best_metrics = self.get_best_strategy()
        report.append(f"\n{'='*60}")
        report.append(f"Best Strategy: {best_name}")
        report.append("=" * 60)
        report.append(f"  Reasoning: Lowest combined score based on:")
        report.append(f"    - Fast settling (30% weight)")
        report.append(f"    - Low overshoot (20% weight)")
        report.append(f"    - Low steady-state error (30% weight)")
        report.append(f"    - Few oscillations (10% weight)")
        report.append(f"    - Low control effort (10% weight)")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)

        return report_text


def run_control_ablation(
    strategies: Optional[List[str]] = None,
    target_coverage: float = 0.85,
    n_epochs: int = 100,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Run control strategy ablation study.

    Args:
        strategies: List of strategy names to test
        target_coverage: Target REAL coverage
        n_epochs: Number of epochs to simulate
        plot: Whether to plot results

    Returns:
        Ablation results
    """
    # Convert strategy names to enum
    if strategies:
        strategy_enums = [ControlStrategy(s) for s in strategies]
    else:
        strategy_enums = [
            ControlStrategy.PROPORTIONAL,
            ControlStrategy.PI,
            ControlStrategy.PID,
            ControlStrategy.DEAD_BAND,
            ControlStrategy.ADAPTIVE,
        ]

    # Configure ablation
    config = AblationConfig(
        strategies=strategy_enums,
        n_epochs=n_epochs,
        target_coverage=target_coverage,
        plot_results=plot,
    )

    # Run ablation
    runner = AblationRunner(config)
    results = runner.run_all_strategies()

    # Generate report
    report = runner.generate_report()
    print(report)

    # Plot if requested
    if plot:
        runner.plot_comparison()

    # Get best strategy
    best_name, best_metrics = runner.get_best_strategy()
    print(f"\n✓ Recommended strategy: {best_name}")

    return {
        "results": results,
        "best_strategy": best_name,
        "best_metrics": best_metrics,
        "report": report,
    }
