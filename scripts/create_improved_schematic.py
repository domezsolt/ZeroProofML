#!/usr/bin/env python3
"""
Create an improved ZeroProofML architecture schematic
- Better proportions and layout
- Clearer flow arrows
- Professional appearance
- More readable text
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

# Set high-quality defaults
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 150


def create_professional_schematic():
    """Create a professional ZeroProofML architecture schematic"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define colors with better contrast
    colors = {
        "input": "#E3F2FD",  # Light blue
        "tr_layer": "#C8E6C9",  # Light green
        "guard": "#FFF3E0",  # Light orange
        "real": "#FFEBEE",  # Light red
        "output": "#F3E5F5",  # Light purple
        "control": "#FFF9C4",  # Light yellow
        "decision": "#FFECB3",  # Light amber
    }

    # Layer 1: Input
    input_box = FancyBboxPatch(
        (1, 8.5),
        3,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["input"],
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(input_box)
    ax.text(2.5, 9, "Input Layer", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(
        2.5, 8.7, "$x \\in \\mathbb{R}^n$ (joint angles)", ha="center", va="center", fontsize=10
    )

    # Layer 2: TR-Rational Layer
    tr_box = FancyBboxPatch(
        (6, 8),
        4,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["tr_layer"],
        edgecolor="#388E3C",
        linewidth=2,
    )
    ax.add_patch(tr_box)
    ax.text(8, 9.3, "TR-Rational Layer", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(8, 8.8, "$r(x) = P(x)/Q(x)$", ha="center", va="center", fontsize=11)
    ax.text(8, 8.3, "Learnable polynomials", ha="center", va="center", fontsize=10, style="italic")

    # Decision Diamond (larger and clearer)
    decision_points = np.array([[8, 6.5], [9.5, 5.5], [8, 4.5], [6.5, 5.5]])
    decision_diamond = plt.Polygon(
        decision_points, facecolor=colors["decision"], edgecolor="#F57C00", linewidth=2
    )
    ax.add_patch(decision_diamond)
    ax.text(8, 5.5, "$|Q(x)| >$", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(8, 5.1, "$\\tau_{switch}$?", ha="center", va="center", fontsize=11, weight="bold")

    # Guard Path (Left)
    guard_box = FancyBboxPatch(
        (1, 2.5),
        4,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["guard"],
        edgecolor="#F57C00",
        linewidth=2,
    )
    ax.add_patch(guard_box)
    ax.text(3, 3.8, "Guard Mode", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(3, 3.4, "High-Precision TR Arithmetic", ha="center", va="center", fontsize=10)
    ax.text(3, 3.0, "$y = P(x)/Q(x)$", ha="center", va="center", fontsize=11)
    ax.text(
        3,
        2.7,
        "Tag: REAL",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="darkgreen",
    )

    # Real Path (Right)
    real_box = FancyBboxPatch(
        (9, 2.5),
        4,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["real"],
        edgecolor="#D32F2F",
        linewidth=2,
    )
    ax.add_patch(real_box)
    ax.text(11, 3.8, "Real Mode", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(11, 3.4, "Masked Operations", ha="center", va="center", fontsize=10)
    ax.text(11, 3.0, "$y = \\infty$ or $\\Phi$", ha="center", va="center", fontsize=11)
    ax.text(
        11,
        2.7,
        "Tag: INF/NULL",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="darkred",
    )

    # Coverage Controller (Top Right)
    controller_box = FancyBboxPatch(
        (11, 7),
        3,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["control"],
        edgecolor="#689F38",
        linewidth=2,
    )
    ax.add_patch(controller_box)
    ax.text(12.5, 8.3, "Coverage", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(12.5, 7.9, "Controller", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(12.5, 7.5, "Maintains near-pole", ha="center", va="center", fontsize=9)
    ax.text(12.5, 7.2, "sampling focus", ha="center", va="center", fontsize=9)

    # Output Layer
    output_box = FancyBboxPatch(
        (6, 0.5),
        4,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["output"],
        edgecolor="#7B1FA2",
        linewidth=2,
    )
    ax.add_patch(output_box)
    ax.text(8, 1.5, "Output", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(8, 1.1, "$(y, tag) \\in \\mathbb{T}$", ha="center", va="center", fontsize=11)
    ax.text(
        8, 0.8, "Tagged transreal values", ha="center", va="center", fontsize=10, style="italic"
    )

    # Flow arrows with better styling
    arrow_props = dict(arrowstyle="->", lw=2.5, color="black")

    # Input to TR layer
    ax.annotate("", xy=(6, 9), xytext=(4, 9), arrowprops=arrow_props)

    # TR layer to decision
    ax.annotate("", xy=(8, 6.5), xytext=(8, 8), arrowprops=arrow_props)

    # Decision to Guard (YES)
    ax.annotate(
        "",
        xy=(4.5, 4.5),
        xytext=(6.8, 5.2),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="green"),
    )
    ax.text(
        5.5,
        5.2,
        "YES",
        fontsize=11,
        color="green",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green"),
    )

    # Decision to Real (NO)
    ax.annotate(
        "", xy=(9.5, 4.5), xytext=(9.2, 5.2), arrowprops=dict(arrowstyle="->", lw=2.5, color="red")
    )
    ax.text(
        9.8,
        5.2,
        "NO",
        fontsize=11,
        color="red",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
    )

    # Both paths to output
    ax.annotate(
        "", xy=(7, 2), xytext=(3, 2.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="orange")
    )
    ax.annotate(
        "", xy=(9, 2), xytext=(11, 2.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="red")
    )

    # Coverage controller feedback (dashed)
    ax.annotate(
        "",
        xy=(10, 8),
        xytext=(11, 7.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="brown", linestyle="--"),
    )
    ax.text(
        10.2,
        7.2,
        "Adaptive\\nFeedback",
        fontsize=9,
        color="brown",
        weight="bold",
        ha="center",
        va="center",
    )

    # Add mathematical details in boxes
    math_box1 = FancyBboxPatch(
        (0.5, 5.5),
        3.5,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor="#F5F5F5",
        edgecolor="gray",
        linewidth=1,
    )
    ax.add_patch(math_box1)
    ax.text(
        2.25, 6.6, "Mathematical Foundation", ha="center", va="center", fontsize=10, weight="bold"
    )
    ax.text(
        2.25,
        6.2,
        "$\\mathbb{T} = \\mathbb{R} \\cup \\{+\\infty, -\\infty, \\Phi\\}$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(2.25, 5.8, "Total arithmetic", ha="center", va="center", fontsize=9, style="italic")

    math_box2 = FancyBboxPatch(
        (10.5, 0.5),
        3.5,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor="#F5F5F5",
        edgecolor="gray",
        linewidth=1,
    )
    ax.add_patch(math_box2)
    ax.text(12.25, 1.6, "Key Properties", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(12.25, 1.2, "• Bounded updates", ha="center", va="center", fontsize=9)
    ax.text(12.25, 0.9, "• Finite switching", ha="center", va="center", fontsize=9)
    ax.text(12.25, 0.6, "• Deterministic", ha="center", va="center", fontsize=9)

    # Add threshold annotation
    ax.text(
        8,
        4,
        "$\\tau_{switch} = 10^{-6}$ (ULP-based)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.9),
    )

    # Title sections
    ax.text(2.5, 10.2, "\\textbf{Input Processing}", fontsize=13, weight="bold", color="#1976D2")
    ax.text(8, 10.5, "\\textbf{Transreal Computation}", fontsize=13, weight="bold", color="#388E3C")
    ax.text(12.5, 10.2, "\\textbf{Adaptive Control}", fontsize=13, weight="bold", color="#689F38")

    # Set limits and formatting
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add overall title
    plt.suptitle(
        "ZeroProofML: Architecture and Computational Flow", fontsize=16, fontweight="bold", y=0.95
    )

    # Add subtitle
    ax.text(
        7.5,
        10.8,
        "Singularity-Resilient Learning with Transreal Arithmetic",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    # Save improved schematic
    output_dir = Path("/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures")
    plt.savefig(output_dir / "figure6_zeroproofml_schematic.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "figure6_zeroproofml_schematic.pdf", dpi=300, bbox_inches="tight")

    print("✓ Improved ZeroProofML schematic saved")
    return True


def create_alternative_flow_diagram():
    """Create an alternative flow-focused diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define better colors
    colors = {
        "process": "#E8F5E8",
        "decision": "#FFF3E0",
        "guard": "#E3F2FD",
        "real": "#FFEBEE",
        "output": "#F3E5F5",
    }

    # Main processing flow (vertical)
    processes = [
        (6, 7.5, "Input $x$"),
        (6, 6.5, "Compute $P(x), Q(x)$"),
        (6, 4.5, "Tag Assignment"),
        (6, 1.5, "Output $(y, tag)$"),
    ]

    # Create process boxes
    for x, y, text in processes:
        if "Input" in text:
            color = colors["process"]
            edge_color = "#4CAF50"
        elif "Output" in text:
            color = colors["output"]
            edge_color = "#9C27B0"
        else:
            color = colors["process"]
            edge_color = "#2196F3"

        box = FancyBboxPatch(
            (x - 1, y - 0.3),
            2,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=11, weight="bold")

    # Decision diamond (better proportioned)
    decision_center = (6, 5.5)
    decision_size = 1.2
    diamond_points = np.array(
        [
            [decision_center[0], decision_center[1] + decision_size / 2],  # top
            [decision_center[0] + decision_size / 2, decision_center[1]],  # right
            [decision_center[0], decision_center[1] - decision_size / 2],  # bottom
            [decision_center[0] - decision_size / 2, decision_center[1]],  # left
        ]
    )

    diamond = plt.Polygon(
        diamond_points, facecolor=colors["decision"], edgecolor="#FF9800", linewidth=2
    )
    ax.add_patch(diamond)
    ax.text(6, 5.7, "$|Q(x)|$", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(6, 5.3, "$> \\tau_{switch}$?", ha="center", va="center", fontsize=10, weight="bold")

    # Guard mode (left branch)
    guard_box = FancyBboxPatch(
        (1.5, 4.8),
        3,
        1.4,
        boxstyle="round,pad=0.1",
        facecolor=colors["guard"],
        edgecolor="#2196F3",
        linewidth=2,
    )
    ax.add_patch(guard_box)
    ax.text(3, 5.8, "\\textbf{Guard Mode}", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(3, 5.4, "Standard division:", ha="center", va="center", fontsize=10)
    ax.text(3, 5.1, "$y = P(x)/Q(x)$", ha="center", va="center", fontsize=10)
    ax.text(3, 4.9, "Tag: REAL", ha="center", va="center", fontsize=9, color="blue", weight="bold")

    # Real mode (right branch)
    real_box = FancyBboxPatch(
        (9.5, 4.8),
        3,
        1.4,
        boxstyle="round,pad=0.1",
        facecolor=colors["real"],
        edgecolor="#F44336",
        linewidth=2,
    )
    ax.add_patch(real_box)
    ax.text(11, 5.8, "\\textbf{Real Mode}", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(11, 5.4, "Near singularity:", ha="center", va="center", fontsize=10)
    ax.text(11, 5.1, "$y = \\pm\\infty$ or $\\Phi$", ha="center", va="center", fontsize=10)
    ax.text(
        11, 4.9, "Tag: INF/NULL", ha="center", va="center", fontsize=9, color="red", weight="bold"
    )

    # Coverage controller (side panel)
    controller_box = FancyBboxPatch(
        (9, 7),
        4,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["control"],
        edgecolor="#8BC34A",
        linewidth=2,
    )
    ax.add_patch(controller_box)
    ax.text(
        11, 8, "\\textbf{Coverage Controller}", ha="center", va="center", fontsize=11, weight="bold"
    )
    ax.text(11, 7.6, "Prevents mode collapse", ha="center", va="center", fontsize=10)
    ax.text(11, 7.3, "Maintains $\\pi_{near} \\geq 15\\%$", ha="center", va="center", fontsize=10)

    # Flow arrows with labels
    # Main flow
    ax.annotate(
        "", xy=(6, 6.2), xytext=(6, 7.2), arrowprops=dict(arrowstyle="->", lw=3, color="black")
    )
    ax.annotate(
        "", xy=(6, 5.8), xytext=(6, 6.2), arrowprops=dict(arrowstyle="->", lw=3, color="black")
    )
    ax.annotate(
        "", xy=(6, 4.8), xytext=(6, 4.9), arrowprops=dict(arrowstyle="->", lw=3, color="black")
    )
    ax.annotate(
        "", xy=(6, 1.8), xytext=(6, 4.2), arrowprops=dict(arrowstyle="->", lw=3, color="black")
    )

    # Decision branches
    ax.annotate(
        "", xy=(4.5, 5.5), xytext=(5.4, 5.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="blue")
    )
    ax.text(4.8, 5.8, "YES", fontsize=10, color="blue", weight="bold")

    ax.annotate(
        "", xy=(7.5, 5.5), xytext=(6.6, 5.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="red")
    )
    ax.text(7.2, 5.8, "NO", fontsize=10, color="red", weight="bold")

    # Convergence arrows
    ax.annotate(
        "",
        xy=(5.5, 4),
        xytext=(3, 4.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="blue", connectionstyle="arc3,rad=0.3"),
    )
    ax.annotate(
        "",
        xy=(6.5, 4),
        xytext=(11, 4.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="red", connectionstyle="arc3,rad=-0.3"),
    )

    # Coverage feedback
    ax.annotate(
        "",
        xy=(9.5, 7.5),
        xytext=(9, 8.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="green", linestyle="--"),
    )
    ax.text(8.5, 8, "Feedback", fontsize=9, color="green", style="italic", rotation=45)

    # Add key insights as side notes
    ax.text(0.5, 8, "\\textbf{Key Innovation:}", fontsize=11, weight="bold", color="darkblue")
    ax.text(0.5, 7.6, "• Total arithmetic", fontsize=10)
    ax.text(0.5, 7.3, "• No $\\varepsilon$ parameters", fontsize=10)
    ax.text(0.5, 7.0, "• Deterministic behavior", fontsize=10)
    ax.text(0.5, 6.7, "• Bounded gradients", fontsize=10)

    ax.text(0.5, 3, "\\textbf{Advantages:}", fontsize=11, weight="bold", color="darkgreen")
    ax.text(0.5, 2.6, "• 29-47\\% error reduction", fontsize=10)
    ax.text(0.5, 2.3, "• 12× computational speedup", fontsize=10)
    ax.text(0.5, 2.0, "• Zero training failures", fontsize=10)
    ax.text(0.5, 1.7, "• Reproducible results", fontsize=10)

    # Set limits and remove axes
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    plt.tight_layout()

    # Save alternative version
    output_dir = Path("/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures")
    plt.savefig(output_dir / "figure6_flow_diagram.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "figure6_flow_diagram.pdf", dpi=300, bbox_inches="tight")

    print("✓ Alternative flow diagram saved")
    return True


if __name__ == "__main__":
    print("\nCreating improved schematic figures...")

    success_count = 0

    try:
        if create_professional_schematic():
            success_count += 1
    except Exception as e:
        print(f"Error creating professional schematic: {e}")

    try:
        if create_alternative_flow_diagram():
            success_count += 1
    except Exception as e:
        print(f"Error creating flow diagram: {e}")

    print(f"\n✓ {success_count}/2 improved schematics created!")

    if success_count >= 1:
        print("\nImproved schematics available:")
        print("  • figure6_zeroproofml_schematic.pdf (detailed architecture)")
        print("  • figure6_flow_diagram.pdf (simplified flow)")
        print("\nChoose the one that looks better in your paper!")
