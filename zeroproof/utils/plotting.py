"""
Plotting utilities for ZeroProofML visualization.

Provides functions to create training curves, pole heatmaps,
sign-flip visualizations, and residual histograms.
"""

import importlib.resources as importlib_resources
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import TRTag, real

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

logger = logging.getLogger(__name__)


class TrainingCurvePlotter:
    """Creates training curve visualizations."""

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize plotter.

        Args:
            style: Matplotlib style to use
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting functionality")

        try:
            plt.style.use(style)
        except Exception:
            try:
                plt.style.use("default")
            except Exception:
                pass  # Use whatever default is available

        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def plot_training_curves(
        self,
        training_history: List[Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Training Curves",
        save_path: Optional[str] = None,
    ):
        """
        Plot training curves for specified metrics.

        Args:
            training_history: List of training step metrics
            metrics: List of metric names to plot
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return None

        if metrics is None:
            metrics = ["loss", "coverage", "lambda_rej"]

        # Extract data
        epochs = [entry.get("epoch", i) for i, entry in enumerate(training_history)]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), squeeze=False)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = []
            valid_epochs = []

            for epoch, entry in zip(epochs, training_history):
                if metric in entry and isinstance(entry[metric], (int, float)):
                    if not (np.isnan(entry[metric]) or np.isinf(entry[metric])):
                        values.append(entry[metric])
                        valid_epochs.append(epoch)

            if values:
                axes[i].plot(
                    valid_epochs,
                    values,
                    color=self.colors[i % len(self.colors)],
                    linewidth=2,
                    label=metric,
                )
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

                # Add trend line for loss
                if "loss" in metric.lower() and len(values) > 5:
                    z = np.polyfit(valid_epochs, values, 1)
                    trend_line = np.poly1d(z)
                    axes[i].plot(
                        valid_epochs,
                        trend_line(valid_epochs),
                        "--",
                        alpha=0.5,
                        color="gray",
                        label="Trend",
                    )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Training curves saved to %s", save_path)

        return fig


def use_zeroproof_style(theme: str = "light") -> None:
    """Activate bundled ZeroProof Matplotlib style.

    Args:
        theme: "light" or "dark"
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for plotting functionality")

    fname = "zeroproof-light.mplstyle" if theme.lower() != "dark" else "zeroproof-dark.mplstyle"
    try:
        with importlib_resources.path("zeroproof.assets.mplstyles", fname) as p:
            plt.style.use(str(p))
    except Exception as e:
        logger.warning("Failed to load ZeroProof style %s: %s", theme, e)

    def plot_qd_smin(
        self,
        training_history: List[Dict[str, Any]],
        title: str = "Q/D Quantiles and smin",
        save_path: Optional[str] = None,
    ):
        """
        Plot q and d quantiles and Sylvester smin over epochs if present.

        Expects training_history as a list of epoch dicts.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return None

        metrics_groups = [
            (["q_p10", "q_p50", "q_p90"], "Q quantiles"),
            (["d_p10", "d_p50", "d_p90"], "D quantiles"),
            (["sylvester_smin"], "Sylvester smin"),
        ]
        epochs = [entry.get("epoch", i) for i, entry in enumerate(training_history)]

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), squeeze=False)
        axes = axes.flatten()

        for ax_idx, (metrics, label) in enumerate(metrics_groups):
            plotted = False
            for i, metric in enumerate(metrics):
                values = []
                valid_epochs = []
                for epoch, entry in zip(epochs, training_history):
                    if metric in entry and isinstance(entry[metric], (int, float)):
                        if not (np.isnan(entry[metric]) or np.isinf(entry[metric])):
                            values.append(entry[metric])
                            valid_epochs.append(epoch)
                if values:
                    axes[ax_idx].plot(
                        valid_epochs,
                        values,
                        color=self.colors[i % len(self.colors)],
                        linewidth=2,
                        label=metric,
                    )
                    plotted = True
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel(label)
            axes[ax_idx].grid(True, alpha=0.3)
            if plotted:
                axes[ax_idx].legend()

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Q/D/smin curves saved to %s", save_path)
        return fig

    def plot_curvature_fisher(
        self,
        training_history: List[Dict[str, Any]],
        title: str = "Curvature/Fisher Proxies",
        save_path: Optional[str] = None,
    ):
        """
        Plot curvature_proxy, grad_norm_epoch, and fisher_trace over epochs.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return None

        metrics = ["curvature_proxy", "grad_norm_epoch", "fisher_trace"]
        labels = ["Curvature Proxy (L_hat)", "Grad Norm (epoch)", "Fisher Trace"]
        epochs = [entry.get("epoch", i) for i, entry in enumerate(training_history)]

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), squeeze=False)
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, labels)):
            values = []
            valid_epochs = []
            for epoch, entry in zip(epochs, training_history):
                if metric in entry and isinstance(entry[metric], (int, float)):
                    v = entry[metric]
                    if not (np.isnan(v) or np.isinf(v)):
                        values.append(v)
                        valid_epochs.append(epoch)
            if values:
                axes[idx].plot(
                    valid_epochs,
                    values,
                    color=self.colors[idx % len(self.colors)],
                    linewidth=2,
                    label=metric,
                )
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(label)
            axes[idx].grid(True, alpha=0.3)
            if values:
                axes[idx].legend()

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Curvature/Fisher curves saved to %s", save_path)
        return fig

    def plot_tag_distribution(
        self,
        training_history: List[Dict[str, Any]],
        title: str = "Tag Distribution Over Time",
        save_path: Optional[str] = None,
    ):
        """
        Plot tag distribution evolution during training.

        Args:
            training_history: Training history with tag counts
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Extract tag ratios
        epochs = []
        real_ratios = []
        pinf_ratios = []
        ninf_ratios = []
        phi_ratios = []

        for i, entry in enumerate(training_history):
            if "REAL_ratio" in entry:
                epochs.append(entry.get("epoch", i))
                real_ratios.append(entry.get("REAL_ratio", 0))
                pinf_ratios.append(entry.get("PINF_ratio", 0))
                ninf_ratios.append(entry.get("NINF_ratio", 0))
                phi_ratios.append(entry.get("PHI_ratio", 0))

        if not epochs:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No tag data available", ha="center", va="center")
            ax.set_title(title)
            return fig

        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.fill_between(epochs, 0, real_ratios, alpha=0.7, color="green", label="REAL")

        bottom = np.array(real_ratios)
        ax.fill_between(
            epochs, bottom, bottom + np.array(pinf_ratios), alpha=0.7, color="red", label="PINF"
        )

        bottom += np.array(pinf_ratios)
        ax.fill_between(
            epochs, bottom, bottom + np.array(ninf_ratios), alpha=0.7, color="blue", label="NINF"
        )

        bottom += np.array(ninf_ratios)
        ax.fill_between(
            epochs, bottom, bottom + np.array(phi_ratios), alpha=0.7, color="orange", label="PHI"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Tag Ratio")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Tag distribution plot saved to %s", save_path)

        return fig


class PoleVisualizationPlotter:
    """Creates pole-specific visualizations."""

    def plot_pole_heatmap(
        self,
        model,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Optional[Tuple[float, float]] = None,
        resolution: int = 100,
        title: str = "Pole Detection Heatmap",
        save_path: Optional[str] = None,
    ):
        """
        Create heatmap of pole detection probabilities.

        Args:
            model: Model with pole detection capability
            x_range: X-axis range
            y_range: Y-axis range (uses x_range if None)
            resolution: Grid resolution
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if y_range is None:
            y_range = x_range

        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate pole probabilities
        Z = np.zeros_like(X)

        for i in range(resolution):
            for j in range(resolution):
                try:
                    if hasattr(model, "pole_head") and model.pole_head:
                        from ..autodiff import TRNode

                        x_val = TRNode.constant(real(X[i, j]))
                        prob = model.pole_head.predict_pole_probability(x_val)
                        Z[i, j] = prob
                    else:
                        # Use |Q| approximation if available
                        if hasattr(model, "get_Q_value"):
                            x_val = TRNode.constant(real(X[i, j]))
                            _ = model.forward(x_val)
                            q_val = model.get_Q_value()
                            if q_val is not None:
                                Z[i, j] = 1.0 / (1.0 + q_val)  # Sigmoid-like
                except Exception:
                    Z[i, j] = 0.0

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            Z,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Pole Probability", rotation=270, labelpad=20)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        # Add contour lines
        contours = ax.contour(X, Y, Z, levels=[0.1, 0.5, 0.9], colors="white", alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Pole heatmap saved to %s", save_path)

        return fig

    def plot_sign_consistency_path(
        self,
        model,
        path_func: callable,
        t_range: Tuple[float, float],
        pole_t: float,
        n_points: int = 100,
        title: str = "Sign Consistency Along Path",
        save_path: Optional[str] = None,
    ):
        """
        Plot sign behavior along a path crossing a pole.

        Args:
            model: Model to evaluate
            path_func: Function t -> x giving path
            t_range: Parameter range
            pole_t: Parameter value where pole is crossed
            n_points: Number of points to evaluate
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        t_vals = np.linspace(t_range[0], t_range[1], n_points)

        # Evaluate model along path
        y_values = []
        tags = []
        signs = []

        for t in t_vals:
            try:
                x_val = path_func(t)
                from ..autodiff import TRNode

                x = TRNode.constant(real(x_val))

                if hasattr(model, "forward_fully_integrated"):
                    result = model.forward_fully_integrated(x)
                    y = result["output"]
                    tag = result["tag"]
                else:
                    y, tag = model.forward(x)

                tags.append(tag)

                if tag == TRTag.REAL:
                    y_val = y.value.value
                    y_values.append(y_val)
                    signs.append(1 if y_val > 0 else -1 if y_val < 0 else 0)
                elif tag == TRTag.PINF:
                    y_values.append(float("inf"))
                    signs.append(1)
                elif tag == TRTag.NINF:
                    y_values.append(float("-inf"))
                    signs.append(-1)
                else:
                    y_values.append(float("nan"))
                    signs.append(0)

            except Exception:
                y_values.append(float("nan"))
                tags.append(TRTag.PHI)
                signs.append(0)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Top plot: function values
        finite_mask = np.isfinite(y_values)
        if np.any(finite_mask):
            ax1.plot(
                t_vals[finite_mask],
                np.array(y_values)[finite_mask],
                "b-",
                linewidth=2,
                label="y(t)",
            )

        # Mark infinities
        inf_mask = np.array([tag in [TRTag.PINF, TRTag.NINF] for tag in tags])
        if np.any(inf_mask):
            inf_signs = np.array(signs)[inf_mask]
            inf_t = t_vals[inf_mask]

            # Plot infinity markers
            ax1.scatter(
                inf_t[inf_signs > 0],
                [ax1.get_ylim()[1]] * np.sum(inf_signs > 0),
                color="red",
                marker="^",
                s=100,
                label="+∞",
                zorder=5,
            )
            ax1.scatter(
                inf_t[inf_signs < 0],
                [ax1.get_ylim()[0]] * np.sum(inf_signs < 0),
                color="blue",
                marker="v",
                s=100,
                label="-∞",
                zorder=5,
            )

        # Mark pole location
        ax1.axvline(x=pole_t, color="gray", linestyle="--", alpha=0.7, label="Pole")

        ax1.set_xlabel("Parameter t")
        ax1.set_ylabel("y(x(t))")
        ax1.set_title("Function Values Along Path")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: sign behavior
        ax2.plot(t_vals, signs, "o-", linewidth=2, markersize=4, color="purple")
        ax2.axvline(x=pole_t, color="gray", linestyle="--", alpha=0.7, label="Pole")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        ax2.set_xlabel("Parameter t")
        ax2.set_ylabel("Sign(y)")
        ax2.set_title("Sign Consistency")
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(["-", "0", "+"])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Sign consistency plot saved to %s", save_path)

        return fig

    def plot_anti_illusion_metrics(
        self,
        ai_history: List[Dict[str, float]],
        title: str = "Anti-Illusion Metrics",
        save_path: Optional[str] = None,
    ):
        """
        Plot anti-illusion metrics over training.

        Args:
            ai_history: List of anti-illusion metric dictionaries
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not ai_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No anti-illusion data available", ha="center", va="center")
            ax.set_title(title)
            return fig

        # Extract metrics
        metrics_to_plot = [
            "ple",
            "sign_consistency",
            "asymptotic_slope_error",
            "residual_consistency",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        epochs = list(range(len(ai_history)))

        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break

            values = []
            valid_epochs = []

            for epoch, entry in zip(epochs, ai_history):
                if metric in entry:
                    value = entry[metric]
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        values.append(value)
                        valid_epochs.append(epoch)

            if values:
                axes[i].plot(
                    valid_epochs, values, "o-", linewidth=2, markersize=4, color=self.colors[i]
                )

                # Add trend line
                if len(values) > 3:
                    z = np.polyfit(valid_epochs, values, 1)
                    trend_line = np.poly1d(z)
                    axes[i].plot(
                        valid_epochs, trend_line(valid_epochs), "--", alpha=0.7, color="gray"
                    )

                axes[i].set_xlabel("Evaluation Step")
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].set_title(f'{metric.replace("_", " ").title()} Evolution')
                axes[i].grid(True, alpha=0.3)

                # Add target line for some metrics
                if metric == "sign_consistency":
                    axes[i].axhline(y=1.0, color="green", linestyle=":", alpha=0.5, label="Perfect")
                elif metric == "asymptotic_slope_error":
                    axes[i].axhline(y=0.0, color="green", linestyle=":", alpha=0.5, label="Perfect")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Anti-illusion metrics plot saved to %s", save_path)

        return fig


class ResidualAnalysisPlotter:
    """Creates residual analysis visualizations."""

    def plot_residual_histogram(
        self,
        residuals: List[float],
        near_pole_residuals: Optional[List[float]] = None,
        title: str = "Residual Distribution",
        save_path: Optional[str] = None,
    ):
        """
        Plot histogram of residual values.

        Args:
            residuals: All residual values
            near_pole_residuals: Residuals specifically near poles
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter finite values
        finite_residuals = [r for r in residuals if np.isfinite(r)]

        if finite_residuals:
            # Main histogram
            ax.hist(
                finite_residuals,
                bins=50,
                alpha=0.7,
                color="skyblue",
                label=f"All Residuals (n={len(finite_residuals)})",
                density=True,
            )

            # Near-pole residuals overlay
            if near_pole_residuals:
                finite_near_pole = [r for r in near_pole_residuals if np.isfinite(r)]
                if finite_near_pole:
                    ax.hist(
                        finite_near_pole,
                        bins=30,
                        alpha=0.7,
                        color="orange",
                        label=f"Near Poles (n={len(finite_near_pole)})",
                        density=True,
                    )

            # Statistics
            mean_residual = np.mean(finite_residuals)
            std_residual = np.std(finite_residuals)

            ax.axvline(
                x=mean_residual,
                color="red",
                linestyle="--",
                alpha=0.8,
                label=f"Mean: {mean_residual:.6f}",
            )
            ax.axvline(x=0, color="green", linestyle=":", alpha=0.8, label="Perfect (R=0)")

            ax.set_xlabel("Residual Value: R(x) = Q(x)·y(x) - P(x)")
            ax.set_ylabel("Density")
            ax.set_title(f"{title}\nMean: {mean_residual:.6f}, Std: {std_residual:.6f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            ax.text(0.5, 0.5, "No finite residual data available", ha="center", va="center")
            ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Residual histogram saved to %s", save_path)

        return fig

    def plot_residual_vs_q(
        self,
        residuals: List[float],
        q_values: List[float],
        title: str = "Residuals vs |Q| Values",
        save_path: Optional[str] = None,
    ):
        """
        Plot residuals vs |Q| values to show pole behavior.

        Args:
            residuals: Residual values
            q_values: Corresponding |Q| values
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Filter valid data
        valid_data = [
            (r, q)
            for r, q in zip(residuals, q_values)
            if np.isfinite(r) and np.isfinite(q) and q > 0
        ]

        if not valid_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No valid residual/Q data available", ha="center", va="center")
            ax.set_title(title)
            return fig

        residuals_clean, q_values_clean = zip(*valid_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot with log scale
        ax.loglog(q_values_clean, np.abs(residuals_clean), "o", alpha=0.6, markersize=3)

        # Add ideal line (residual should be small for good rational approximation)
        q_range = [min(q_values_clean), max(q_values_clean)]
        ideal_residual = [1e-6, 1e-6]  # Constant small residual
        ax.loglog(q_range, ideal_residual, "g--", alpha=0.8, label="Ideal (R ≈ 0)")

        ax.set_xlabel("|Q(x)|")
        ax.set_ylabel("|R(x)| = |Q(x)·y(x) - P(x)|")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Residual vs Q plot saved to %s", save_path)

        return fig


class ComparisonPlotter:
    """Creates comparison plots between different methods."""

    def plot_method_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Method Comparison",
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison between different methods.

        Args:
            results: Dictionary of method_name -> results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["test_mse", "training_time", "success_rate"]

        # Extract data
        methods = list(results.keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            values = []
            labels = []
            colors = []

            for j, method in enumerate(methods):
                if metric in results[method]:
                    value = results[method][metric]
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        values.append(value)
                        labels.append(method)
                        colors.append(self._get_method_color(method))

            if values:
                bars = axes[i].bar(range(len(values)), values, color=colors, alpha=0.7)
                axes[i].set_xticks(range(len(values)))
                axes[i].set_xticklabels(labels, rotation=45, ha="right")
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].grid(True, alpha=0.3, axis="y")

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Method comparison plot saved to %s", save_path)

        return fig

    def _get_method_color(self, method_name: str) -> str:
        """Get consistent color for method."""
        color_map = {
            "MLP": "#1f77b4",
            "Rational+ε": "#ff7f0e",
            "DLS": "#2ca02c",
            "ZeroProofML-Basic": "#d62728",
            "ZeroProofML-Full": "#9467bd",
            "TR-Rational": "#8c564b",
        }
        return color_map.get(method_name, "#7f7f7f")


def create_paper_ready_figures(results_dir: str, output_dir: str = "paper_figures") -> List[str]:
    """
    Create paper-ready figures from results.

    Args:
        results_dir: Directory containing experimental results
        output_dir: Directory to save paper figures

    Returns:
        List of created figure paths
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping paper figures")
        return []

    logger.info("Creating Paper-Ready Figures")

    os.makedirs(output_dir, exist_ok=True)
    figure_paths = []

    # Set publication style
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    try:
        # 1. Method comparison figure
        comparison_file = os.path.join(results_dir, "comparison_table.csv")
        if os.path.exists(comparison_file):
            # Load comparison data - use csv if pandas not available
            try:
                import pandas as pd

                df = pd.read_csv(comparison_file)
                use_pandas = True
            except ImportError:
                import csv

                # Manual CSV reading
                with open(comparison_file, "r") as f:
                    reader = csv.DictReader(f)
                    df = list(reader)
                use_pandas = False

            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Test MSE comparison
            if use_pandas:
                mse_data = df[df["Test_MSE"] != "N/A"]["Test_MSE"].astype(float)
                methods = df[df["Test_MSE"] != "N/A"]["Method"]
            else:
                # Manual processing
                mse_data = []
                methods = []
                for row in df:
                    if row["Test_MSE"] != "N/A":
                        try:
                            mse_data.append(float(row["Test_MSE"]))
                            methods.append(row["Method"])
                        except ValueError:
                            continue

            if len(mse_data) > 0:
                axes[0].bar(range(len(mse_data)), mse_data, alpha=0.7)
                axes[0].set_xticks(range(len(mse_data)))
                axes[0].set_xticklabels(methods, rotation=45, ha="right")
                axes[0].set_ylabel("Test MSE")
                axes[0].set_title("Test Error Comparison")
                axes[0].set_yscale("log")
                axes[0].grid(True, alpha=0.3, axis="y")

            # Training time comparison
            if use_pandas:
                time_data = df[df["Training_Time"] > 0]["Training_Time"]
                time_methods = df[df["Training_Time"] > 0]["Method"]
            else:
                time_data = []
                time_methods = []
                for row in df:
                    try:
                        time_val = float(row["Training_Time"])
                        if time_val > 0:
                            time_data.append(time_val)
                            time_methods.append(row["Method"])
                    except (ValueError, KeyError):
                        continue

            if len(time_data) > 0:
                axes[1].bar(range(len(time_data)), time_data, alpha=0.7, color="orange")
                axes[1].set_xticks(range(len(time_data)))
                axes[1].set_xticklabels(time_methods, rotation=45, ha="right")
                axes[1].set_ylabel("Training Time (s)")
                axes[1].set_title("Training Time Comparison")
                axes[1].grid(True, alpha=0.3, axis="y")

            # Parameter count comparison
            if use_pandas:
                param_data = df[df["Parameters"] > 0]["Parameters"]
                param_methods = df[df["Parameters"] > 0]["Method"]
            else:
                param_data = []
                param_methods = []
                for row in df:
                    try:
                        param_val = int(row["Parameters"])
                        if param_val > 0:
                            param_data.append(param_val)
                            param_methods.append(row["Method"])
                    except (ValueError, KeyError):
                        continue

            if len(param_data) > 0:
                axes[2].bar(range(len(param_data)), param_data, alpha=0.7, color="green")
                axes[2].set_xticks(range(len(param_data)))
                axes[2].set_xticklabels(param_methods, rotation=45, ha="right")
                axes[2].set_ylabel("Number of Parameters")
                axes[2].set_title("Model Complexity Comparison")
                axes[2].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            fig_path = os.path.join(output_dir, "method_comparison.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            figure_paths.append(fig_path)

            # Also save as SVG for publications
            svg_path = os.path.join(output_dir, "method_comparison.svg")
            plt.savefig(svg_path, bbox_inches="tight")
            figure_paths.append(svg_path)

            plt.close()

    except Exception as e:
        logger.exception("Failed to create method comparison figure: %s", e)

    logger.info("Created %d paper-ready figures", len(figure_paths))
    for path in figure_paths:
        logger.info("  - %s", path)

    return figure_paths


def save_all_plots(
    run_dir: str,
    training_history: List[Dict[str, Any]],
    model=None,
    ai_metrics: Optional[List[Dict[str, float]]] = None,
    residuals: Optional[List[float]] = None,
    q_values: Optional[List[float]] = None,
) -> List[str]:
    """
    Save all available plots for a training run.

    Args:
        run_dir: Run directory
        training_history: Training metrics history
        model: Trained model (for pole visualization)
        ai_metrics: Anti-illusion metrics history
        residuals: Residual values
        q_values: Q values

    Returns:
        List of saved plot paths
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping all plots")
        return []

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    saved_plots = []

    # Training curves
    try:
        plotter = TrainingCurvePlotter()

        # Main training curves
        fig1 = plotter.plot_training_curves(
            training_history, metrics=["loss", "coverage", "lambda_rej"], title="Training Progress"
        )
        path1 = os.path.join(plots_dir, "training_curves.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        saved_plots.append(path1)
        plt.close(fig1)

        # Tag distribution
        fig2 = plotter.plot_tag_distribution(training_history)
        path2 = os.path.join(plots_dir, "tag_distribution.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        saved_plots.append(path2)
        plt.close(fig2)

        # Q/D/smin
        try:
            fig_qd = plotter.plot_qd_smin(training_history)
            path_qd = os.path.join(plots_dir, "qd_smin_curves.png")
            fig_qd.savefig(path_qd, dpi=150, bbox_inches="tight")
            saved_plots.append(path_qd)
            plt.close(fig_qd)
        except Exception as e:
            logger.exception("Failed to create Q/D/smin plot: %s", e)

        # Curvature / Fisher proxies
        try:
            fig_cf = plotter.plot_curvature_fisher(training_history)
            path_cf = os.path.join(plots_dir, "curvature_fisher_curves.png")
            fig_cf.savefig(path_cf, dpi=150, bbox_inches="tight")
            saved_plots.append(path_cf)
            plt.close(fig_cf)
        except Exception as e:
            logger.exception("Failed to create curvature/fisher plot: %s", e)

    except Exception as e:
        logger.exception("Failed to create training plots: %s", e)

    # Pole visualization
    if model and hasattr(model, "pole_head"):
        try:
            pole_plotter = PoleVisualizationPlotter()
            fig3 = pole_plotter.plot_pole_heatmap(model, title="Learned Pole Locations")
            path3 = os.path.join(plots_dir, "pole_heatmap.png")
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            saved_plots.append(path3)
            plt.close(fig3)

        except Exception as e:
            logger.exception("Failed to create pole heatmap: %s", e)

    # Anti-illusion metrics
    if ai_metrics:
        try:
            plotter = TrainingCurvePlotter()
            fig4 = plotter.plot_anti_illusion_metrics(ai_metrics)
            path4 = os.path.join(plots_dir, "anti_illusion_metrics.png")
            fig4.savefig(path4, dpi=150, bbox_inches="tight")
            saved_plots.append(path4)
            plt.close(fig4)

        except Exception as e:
            logger.exception("Failed to create anti-illusion plots: %s", e)

    # Residual analysis
    if residuals:
        try:
            residual_plotter = ResidualAnalysisPlotter()

            # Residual histogram
            fig5 = residual_plotter.plot_residual_histogram(residuals)
            path5 = os.path.join(plots_dir, "residual_histogram.png")
            fig5.savefig(path5, dpi=150, bbox_inches="tight")
            saved_plots.append(path5)
            plt.close(fig5)

            # Residual vs Q plot
            if q_values:
                fig6 = residual_plotter.plot_residual_vs_q(residuals, q_values)
                path6 = os.path.join(plots_dir, "residual_vs_q.png")
                fig6.savefig(path6, dpi=150, bbox_inches="tight")
                saved_plots.append(path6)
                plt.close(fig6)

        except Exception as e:
            logger.exception("Failed to create residual plots: %s", e)

    logger.info("Saved %d plots to %s", len(saved_plots), plots_dir)

    return saved_plots
