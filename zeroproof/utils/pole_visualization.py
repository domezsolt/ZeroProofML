"""
Visualization tools for pole learning and gradient flow.

This module provides plotting utilities for visualizing learned Q(x),
pole locations, and gradient behavior near singularities.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..autodiff import TRNode
from ..core import TRScalar, TRTag, real

logger = logging.getLogger(__name__)


class PoleVisualizer:
    """
    Visualization tools for pole-related analysis.

    Provides various plotting functions for understanding pole learning.
    """

    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = {
            "true_pole": "red",
            "predicted_pole": "blue",
            "Q_curve": "green",
            "y_curve": "black",
            "gradient": "purple",
            "near_pole": "lightcoral",
            "mid_range": "lightyellow",
            "far_range": "lightgreen",
        }

    def plot_Q_comparison(
        self,
        x_values: List[float],
        Q_learned: List[float],
        Q_true: Optional[List[float]] = None,
        true_poles: Optional[List[float]] = None,
        predicted_poles: Optional[List[float]] = None,
        title: str = "Q(x) Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot learned Q(x) vs ground truth.

        Args:
            x_values: Input values
            Q_learned: Learned Q values
            Q_true: Optional true Q values
            true_poles: True pole locations
            predicted_poles: Predicted pole locations
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot Q curves
        ax.plot(
            x_values, Q_learned, color=self.colors["Q_curve"], linewidth=2, label="Learned Q(x)"
        )

        if Q_true is not None:
            ax.plot(x_values, Q_true, color="gray", linewidth=2, linestyle="--", label="True Q(x)")

        # Mark poles
        if true_poles:
            for pole in true_poles:
                ax.axvline(
                    x=pole,
                    color=self.colors["true_pole"],
                    linestyle="--",
                    alpha=0.5,
                    label="True pole" if pole == true_poles[0] else None,
                )

        if predicted_poles:
            for pole in predicted_poles:
                ax.scatter(
                    [pole],
                    [0],
                    color=self.colors["predicted_pole"],
                    s=100,
                    marker="^",
                    label="Predicted pole" if pole == predicted_poles[0] else None,
                )

        # Add threshold line
        ax.axhline(y=0.1, color="orange", linestyle=":", alpha=0.5, label="Pole threshold")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("|Q(x)|", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis to log scale for better visibility
        if all(q > 0 for q in Q_learned):
            ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()

    def plot_pole_locations(
        self,
        x_range: Tuple[float, float],
        true_poles: List[float],
        predicted_poles: List[float],
        pole_probabilities: Optional[List[Tuple[float, float]]] = None,
        title: str = "Pole Locations",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize pole locations in input space.

        Args:
            x_range: Range of x values (min, max)
            true_poles: True pole locations
            predicted_poles: Predicted pole locations
            pole_probabilities: Optional (x, probability) pairs
            title: Plot title
            save_path: Optional path to save
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], 4))

        # Plot x-axis
        ax.axhline(y=0, color="black", linewidth=1)
        ax.set_xlim(x_range)
        ax.set_ylim(-0.5, 1.5)

        # Plot true poles
        for pole in true_poles:
            ax.scatter(
                [pole],
                [0],
                color=self.colors["true_pole"],
                s=200,
                marker="|",
                linewidth=3,
                label="True pole" if pole == true_poles[0] else None,
            )

        # Plot predicted poles
        for pole in predicted_poles:
            ax.scatter(
                [pole],
                [0],
                color=self.colors["predicted_pole"],
                s=150,
                marker="^",
                label="Predicted pole" if pole == predicted_poles[0] else None,
            )

        # Plot pole probabilities if provided
        if pole_probabilities:
            x_probs = [x for x, _ in pole_probabilities]
            probs = [p for _, p in pole_probabilities]

            # Create color map for probabilities
            ax2 = ax.twinx()
            ax2.bar(
                x_probs,
                probs,
                width=(x_range[1] - x_range[0]) / len(x_probs),
                alpha=0.3,
                color="purple",
                label="Pole probability",
            )
            ax2.set_ylabel("Pole Probability", fontsize=12)
            ax2.set_ylim(0, 1)

        ax.set_xlabel("x", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper left")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()

    def plot_gradient_flow(
        self,
        x_values: List[float],
        gradients: List[float],
        Q_values: List[float],
        true_poles: Optional[List[float]] = None,
        title: str = "Gradient Flow Near Singularities",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Show gradient flow near singularities.

        Args:
            x_values: Input values
            gradients: Gradient magnitudes
            Q_values: |Q(x)| values
            true_poles: True pole locations
            title: Plot title
            save_path: Optional path to save
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Top: Gradient magnitude
        ax1.plot(
            x_values,
            gradients,
            color=self.colors["gradient"],
            linewidth=2,
            label="Gradient magnitude",
        )

        # Mark pole regions
        if true_poles:
            for pole in true_poles:
                # Shade near-pole region
                rect = Rectangle(
                    (pole - 0.1, 0), 0.2, max(gradients), color=self.colors["near_pole"], alpha=0.3
                )
                ax1.add_patch(rect)

        ax1.set_ylabel("|∇L|", fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Bottom: Q values
        ax2.plot(x_values, Q_values, color=self.colors["Q_curve"], linewidth=2, label="|Q(x)|")

        # Mark poles
        if true_poles:
            for pole in true_poles:
                ax2.axvline(x=pole, color=self.colors["true_pole"], linestyle="--", alpha=0.5)

        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("|Q(x)|", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()

    def plot_coverage_breakdown(
        self,
        x_values: List[float],
        y_tags: List[TRTag],
        true_poles: List[float],
        near_threshold: float = 0.1,
        mid_threshold: float = 0.5,
        title: str = "Coverage by Distance from Poles",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize coverage breakdown by distance from poles.

        Args:
            x_values: Input values
            y_tags: Output tags
            true_poles: True pole locations
            near_threshold: Near pole distance
            mid_threshold: Mid-range distance
            title: Plot title
            save_path: Optional path to save
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Classify points by distance
        near_x, near_y = [], []
        mid_x, mid_y = [], []
        far_x, far_y = [], []

        for x, tag in zip(x_values, y_tags):
            # Find min distance to pole
            if true_poles:
                min_dist = min(abs(x - pole) for pole in true_poles)
            else:
                min_dist = float("inf")

            # Convert tag to numeric (1 for REAL, 0 for non-REAL)
            is_real = 1 if tag == TRTag.REAL else 0

            if min_dist < near_threshold:
                near_x.append(x)
                near_y.append(is_real)
            elif min_dist < mid_threshold:
                mid_x.append(x)
                mid_y.append(is_real)
            else:
                far_x.append(x)
                far_y.append(is_real)

        # Create background regions
        x_min, x_max = min(x_values), max(x_values)

        for pole in true_poles:
            # Near region
            rect_near = Rectangle(
                (pole - near_threshold, -0.1),
                2 * near_threshold,
                1.2,
                color=self.colors["near_pole"],
                alpha=0.2,
                label="Near pole" if pole == true_poles[0] else None,
            )
            ax.add_patch(rect_near)

            # Mid region
            rect_mid_left = Rectangle(
                (pole - mid_threshold, -0.1),
                mid_threshold - near_threshold,
                1.2,
                color=self.colors["mid_range"],
                alpha=0.2,
                label="Mid-range" if pole == true_poles[0] else None,
            )
            ax.add_patch(rect_mid_left)

            rect_mid_right = Rectangle(
                (pole + near_threshold, -0.1),
                mid_threshold - near_threshold,
                1.2,
                color=self.colors["mid_range"],
                alpha=0.2,
            )
            ax.add_patch(rect_mid_right)

        # Plot points
        if near_x:
            ax.scatter(near_x, near_y, color="red", s=50, alpha=0.7, label="Near samples")
        if mid_x:
            ax.scatter(mid_x, mid_y, color="orange", s=50, alpha=0.7, label="Mid samples")
        if far_x:
            ax.scatter(far_x, far_y, color="green", s=50, alpha=0.7, label="Far samples")

        # Mark poles
        for pole in true_poles:
            ax.axvline(
                x=pole,
                color=self.colors["true_pole"],
                linestyle="--",
                alpha=0.5,
                label="Pole" if pole == true_poles[0] else None,
            )

        # Compute and display coverage stats
        near_coverage = np.mean(near_y) if near_y else 0
        mid_coverage = np.mean(mid_y) if mid_y else 1
        far_coverage = np.mean(far_y) if far_y else 1

        stats_text = f"Coverage:\nNear: {near_coverage:.1%}\nMid: {mid_coverage:.1%}\nFar: {far_coverage:.1%}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("REAL Output (1) vs Non-REAL (0)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()

    def plot_asymptotic_behavior(
        self,
        x_values: List[float],
        y_values: List[float],
        Q_values: List[float],
        near_pole_mask: Optional[List[bool]] = None,
        title: str = "Asymptotic Behavior Near Poles",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot log|y| vs -log|Q| to check asymptotic behavior.

        Args:
            x_values: Input values
            y_values: Output values (REAL only)
            Q_values: |Q(x)| values
            near_pole_mask: Boolean mask for near-pole points
            title: Plot title
            save_path: Optional path to save
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Filter for valid values
        valid_points = []
        for i, (y, q) in enumerate(zip(y_values, Q_values)):
            if abs(y) > 1e-10 and abs(q) > 1e-10:
                valid_points.append(
                    (np.log(abs(y)), -np.log(abs(q)), near_pole_mask[i] if near_pole_mask else True)
                )

        if not valid_points:
            logger.warning("No valid points for asymptotic plot")
            return

        log_y = [p[0] for p in valid_points]
        neg_log_q = [p[1] for p in valid_points]
        is_near = [p[2] for p in valid_points]

        # Left plot: log|y| vs -log|Q|
        colors = ["red" if near else "blue" for near in is_near]
        ax1.scatter(neg_log_q, log_y, c=colors, alpha=0.6, s=30)

        # Add ideal line (slope 1)
        min_val = min(min(neg_log_q), min(log_y))
        max_val = max(max(neg_log_q), max(log_y))
        ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Ideal (slope=1)")

        # Fit line to near-pole points
        near_indices = [i for i, near in enumerate(is_near) if near]
        if len(near_indices) > 1:
            near_x = [neg_log_q[i] for i in near_indices]
            near_y = [log_y[i] for i in near_indices]
            z = np.polyfit(near_x, near_y, 1)
            p = np.poly1d(z)
            ax1.plot(
                sorted(near_x), p(sorted(near_x)), "r-", alpha=0.7, label=f"Fit (slope={z[0]:.2f})"
            )

        ax1.set_xlabel("-log|Q|", fontsize=12)
        ax1.set_ylabel("log|y|", fontsize=12)
        ax1.set_title("Asymptotic Relationship", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Residuals
        residuals = [ly - nlq for ly, nlq in zip(log_y, neg_log_q)]
        ax2.scatter(neg_log_q, residuals, c=colors, alpha=0.6, s=30)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # Add legend for colors
        red_patch = mpatches.Patch(color="red", label="Near pole")
        blue_patch = mpatches.Patch(color="blue", label="Far from pole")
        ax2.legend(handles=[red_patch, blue_patch])

        ax2.set_xlabel("-log|Q|", fontsize=12)
        ax2.set_ylabel("log|y| - (-log|Q|)", fontsize=12)
        ax2.set_title("Residuals", fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()

    def create_summary_figure(
        self,
        metrics_history: List[Dict[str, float]],
        title: str = "Pole Learning Summary",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a summary figure showing metric evolution.

        Args:
            metrics_history: List of metric dictionaries over time
            title: Plot title
            save_path: Optional path to save
        """
        if not metrics_history:
            logger.warning("No metrics history to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Extract metrics over time
        epochs = list(range(len(metrics_history)))

        # 1. PLE over time
        ple = [m.get("ple", 0) for m in metrics_history]
        axes[0].plot(epochs, ple, "b-", linewidth=2)
        axes[0].set_title("Pole Localization Error")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("PLE")
        axes[0].grid(True, alpha=0.3)

        # 2. Sign consistency
        sign_cons = [m.get("sign_consistency", 0) for m in metrics_history]
        axes[1].plot(epochs, sign_cons, "g-", linewidth=2)
        axes[1].set_title("Sign Consistency")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Consistency")
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3)

        # 3. Coverage breakdown
        near_cov = [m.get("coverage_near", 0) for m in metrics_history]
        mid_cov = [m.get("coverage_mid", 1) for m in metrics_history]
        far_cov = [m.get("coverage_far", 1) for m in metrics_history]

        axes[2].plot(epochs, near_cov, "r-", label="Near", linewidth=2)
        axes[2].plot(epochs, mid_cov, "orange", label="Mid", linewidth=2)
        axes[2].plot(epochs, far_cov, "g-", label="Far", linewidth=2)
        axes[2].set_title("Coverage by Distance")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Coverage")
        axes[2].set_ylim(0, 1.1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. Precision/Recall
        precision = [m.get("pole_precision", 0) for m in metrics_history]
        recall = [m.get("pole_recall", 0) for m in metrics_history]

        axes[3].plot(epochs, precision, "b-", label="Precision", linewidth=2)
        axes[3].plot(epochs, recall, "r-", label="Recall", linewidth=2)
        axes[3].set_title("Pole Detection Accuracy")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Score")
        axes[3].set_ylim(0, 1.1)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # 5. Asymptotic error
        asymp_error = [m.get("asymptotic_slope_error", 0) for m in metrics_history]
        axes[4].plot(epochs, asymp_error, "purple", linewidth=2)
        axes[4].set_title("Asymptotic Slope Error")
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Error")
        axes[4].grid(True, alpha=0.3)

        # 6. Residual error
        residual = [m.get("residual_error", 0) for m in metrics_history]
        axes[5].plot(epochs, residual, "brown", linewidth=2)
        axes[5].set_title("Residual Consistency")
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("Mean |R(x)|")
        axes[5].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.show()
