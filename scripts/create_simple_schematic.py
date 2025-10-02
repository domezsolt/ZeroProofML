#!/usr/bin/env python3
"""
Create a clean, simple ZeroProofML schematic focused on the key innovation
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"


def create_clean_schematic():
    """Create a clean, focused schematic"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Define clean colors
    colors = {
        "input": "#E3F2FD",
        "layer": "#C8E6C9",
        "guard": "#FFF8E1",
        "real": "#FFEBEE",
        "output": "#F3E5F5",
    }

    # Input
    input_rect = FancyBboxPatch(
        (0.5, 2),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["input"],
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(input_rect)
    ax.text(
        1.5,
        2.5,
        "Input\\n$x \\in \\mathbb{R}^n$",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )

    # TR-Rational Layer
    tr_rect = FancyBboxPatch(
        (4, 1.5),
        3,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["layer"],
        edgecolor="#388E3C",
        linewidth=2,
    )
    ax.add_patch(tr_rect)
    ax.text(5.5, 2.8, "\\textbf{TR-Rational}", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(5.5, 2.4, "$P(x)/Q(x)$", ha="center", va="center", fontsize=11)
    ax.text(5.5, 2.0, "with tags", ha="center", va="center", fontsize=10, style="italic")

    # Decision point (simplified)
    decision_circle = Circle((8.5, 2.5), 0.4, facecolor="#FFEB3B", edgecolor="#F57C00", linewidth=2)
    ax.add_patch(decision_circle)
    ax.text(8.5, 2.5, "?", ha="center", va="center", fontsize=16, weight="bold")

    # Guard path
    guard_rect = FancyBboxPatch(
        (10, 3.5),
        2.5,
        0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors["guard"],
        edgecolor="#FF9800",
        linewidth=1.5,
    )
    ax.add_patch(guard_rect)
    ax.text(11.25, 3.9, "Guard: $P/Q$", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(11.25, 3.6, "(REAL)", ha="center", va="center", fontsize=9, color="blue")

    # Real path
    real_rect = FancyBboxPatch(
        (10, 1.2),
        2.5,
        0.8,
        boxstyle="round,pad=0.05",
        facecolor=colors["real"],
        edgecolor="#F44336",
        linewidth=1.5,
    )
    ax.add_patch(real_rect)
    ax.text(
        11.25,
        1.6,
        "Real: $\\pm\\infty/\\Phi$",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(11.25, 1.3, "(INF/NULL)", ha="center", va="center", fontsize=9, color="red")

    # Output
    output_rect = FancyBboxPatch(
        (14, 2),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["output"],
        edgecolor="#9C27B0",
        linewidth=2,
    )
    ax.add_patch(output_rect)
    ax.text(15, 2.5, "Output\\n$(y, tag)$", ha="center", va="center", fontsize=11, weight="bold")

    # Arrows
    # Input to TR layer
    ax.arrow(2.5, 2.5, 1.3, 0, head_width=0.1, head_length=0.1, fc="black", ec="black", lw=2)

    # TR layer to decision
    ax.arrow(7, 2.5, 1.2, 0, head_width=0.1, head_length=0.1, fc="black", ec="black", lw=2)

    # Decision to paths
    ax.arrow(8.8, 2.8, 0.8, 0.5, head_width=0.08, head_length=0.08, fc="orange", ec="orange", lw=2)
    ax.arrow(8.8, 2.2, 0.8, -0.5, head_width=0.08, head_length=0.08, fc="red", ec="red", lw=2)

    # Paths to output
    ax.arrow(
        12.5, 3.9, 1.2, -1.2, head_width=0.08, head_length=0.08, fc="orange", ec="orange", lw=1.5
    )
    ax.arrow(12.5, 1.6, 1.2, 0.7, head_width=0.08, head_length=0.08, fc="red", ec="red", lw=1.5)

    # Add condition text
    ax.text(
        8.5,
        1.5,
        "$|Q| > \\tau_{switch}$",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange"),
    )

    # Add key insight
    ax.text(
        8.5,
        0.5,
        "\\textbf{Key:} Singularities become well-defined computational states",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F8FF", edgecolor="blue", alpha=0.8),
    )

    # Title
    ax.text(
        8.5,
        4.8,
        "\\textbf{ZeroProofML: Transreal Computation Flow}",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )

    # Set limits and clean up
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    # Save clean schematic
    output_dir = Path("/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures")
    plt.savefig(output_dir / "figure6_clean_schematic.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "figure6_clean_schematic.pdf", dpi=300, bbox_inches="tight")

    print("✓ Clean schematic saved as figure6_clean_schematic.pdf")
    return True


if __name__ == "__main__":
    create_clean_schematic()
