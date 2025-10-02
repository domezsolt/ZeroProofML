#!/usr/bin/env python3
"""
Create the final, professional ZeroProofML schematic
- Clean design with proper text rendering
- Professional color scheme
- Clear flow visualization
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# Configure matplotlib for better text rendering
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = False  # Don't use LaTeX rendering


def create_professional_flow():
    """Create the final professional schematic"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # Professional color palette
    colors = {
        "input": "#E8F4FD",  # Very light blue
        "tr_layer": "#E8F5E9",  # Very light green
        "guard": "#FFF8E1",  # Very light yellow
        "real": "#FFEBEE",  # Very light red
        "output": "#F3E5F5",  # Very light purple
        "decision": "#FFE0B2",  # Light orange
        "text": "#212121",  # Dark gray for text
        "accent": "#1565C0",  # Blue accent
    }

    # Helper function for clean text
    def add_text(x, y, text, **kwargs):
        default_kwargs = {"ha": "center", "va": "center", "color": colors["text"]}
        default_kwargs.update(kwargs)
        return ax.text(x, y, text, **default_kwargs)

    # 1. Input Box
    input_box = FancyBboxPatch(
        (1, 3),
        2.5,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor=colors["input"],
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(input_box)
    add_text(2.25, 3.8, "Input", fontsize=13, fontweight="bold")
    add_text(2.25, 3.4, "x ∈ ℝⁿ", fontsize=11)

    # 2. TR-Rational Layer (central)
    tr_box = FancyBboxPatch(
        (5, 2.5),
        3.5,
        2,
        boxstyle="round,pad=0.08",
        facecolor=colors["tr_layer"],
        edgecolor="#388E3C",
        linewidth=2.5,
    )
    ax.add_patch(tr_box)
    add_text(6.75, 3.9, "TR-Rational Layer", fontsize=14, fontweight="bold")
    add_text(6.75, 3.45, "P(x) / Q(x)", fontsize=12)
    add_text(6.75, 3.0, "with transreal tags", fontsize=10, style="italic", color="#555")

    # 3. Decision Diamond
    diamond_x, diamond_y = 10, 3.5
    diamond_size = 0.7
    diamond = mpatches.FancyBboxPatch(
        (diamond_x - diamond_size, diamond_y - diamond_size),
        diamond_size * 2,
        diamond_size * 2,
        boxstyle="round,pad=0.02",
        transform=ax.transData,
        facecolor=colors["decision"],
        edgecolor="#FF6F00",
        linewidth=2,
    )
    # Rotate 45 degrees
    from matplotlib.transforms import Affine2D

    t = Affine2D().rotate_deg(45).translate(diamond_x, diamond_y) + ax.transData
    diamond.set_transform(t)
    ax.add_patch(diamond)
    add_text(diamond_x, diamond_y + 0.1, "|Q| >", fontsize=10, fontweight="bold")
    add_text(diamond_x, diamond_y - 0.2, "τ", fontsize=10, fontweight="bold")

    # 4. Guard Mode (top path)
    guard_box = FancyBboxPatch(
        (11.5, 5),
        3,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor=colors["guard"],
        edgecolor="#FF8F00",
        linewidth=2,
    )
    ax.add_patch(guard_box)
    add_text(13, 5.8, "Guard Mode", fontsize=12, fontweight="bold")
    add_text(13, 5.4, "Standard P/Q", fontsize=10)
    add_text(13, 5.1, "→ REAL tag", fontsize=9, color="#1565C0", style="italic")

    # 5. Real Mode (bottom path)
    real_box = FancyBboxPatch(
        (11.5, 1.5),
        3,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor=colors["real"],
        edgecolor="#D32F2F",
        linewidth=2,
    )
    ax.add_patch(real_box)
    add_text(13, 2.3, "Real Mode", fontsize=12, fontweight="bold")
    add_text(13, 1.9, "±∞ or Φ", fontsize=10)
    add_text(13, 1.6, "→ INF/NULL tags", fontsize=9, color="#D32F2F", style="italic")

    # 6. Output Box
    output_box = FancyBboxPatch(
        (16, 3),
        2.5,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor=colors["output"],
        edgecolor="#7B1FA2",
        linewidth=2,
    )
    ax.add_patch(output_box)
    add_text(17.25, 3.8, "Output", fontsize=13, fontweight="bold")
    add_text(17.25, 3.4, "(y, tag) ∈ 𝕋", fontsize=11)

    # Flow arrows
    arrow_style = "Simple, tail_width=0.5, head_width=6, head_length=8"
    arrow_kwargs = dict(arrowstyle=arrow_style, color="black", lw=2)

    # Input → TR-Rational
    ax.add_patch(FancyArrowPatch((3.5, 3.6), (5, 3.6), connectionstyle="arc3", **arrow_kwargs))

    # TR-Rational → Decision
    ax.add_patch(FancyArrowPatch((8.5, 3.5), (9.3, 3.5), connectionstyle="arc3", **arrow_kwargs))

    # Decision → Guard (YES)
    ax.add_patch(
        FancyArrowPatch(
            (10.3, 3.9),
            (11.5, 5.3),
            connectionstyle="arc3,rad=.3",
            arrowstyle=arrow_style,
            color="#388E3C",
            lw=2,
        )
    )
    add_text(
        10.5,
        4.5,
        "YES",
        fontsize=10,
        color="#388E3C",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#388E3C"),
    )

    # Decision → Real (NO)
    ax.add_patch(
        FancyArrowPatch(
            (10.3, 3.1),
            (11.5, 2.2),
            connectionstyle="arc3,rad=-.3",
            arrowstyle=arrow_style,
            color="#D32F2F",
            lw=2,
        )
    )
    add_text(
        10.5,
        2.5,
        "NO",
        fontsize=10,
        color="#D32F2F",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D32F2F"),
    )

    # Guard → Output
    ax.add_patch(
        FancyArrowPatch(
            (14.5, 5.5),
            (16.2, 3.8),
            connectionstyle="arc3,rad=-.3",
            arrowstyle=arrow_style,
            color="#FF8F00",
            lw=1.8,
        )
    )

    # Real → Output
    ax.add_patch(
        FancyArrowPatch(
            (14.5, 2),
            (16.2, 3.2),
            connectionstyle="arc3,rad=.3",
            arrowstyle=arrow_style,
            color="#D32F2F",
            lw=1.8,
        )
    )

    # Add key components labels
    add_text(
        6.75,
        1.5,
        "Computational Core",
        fontsize=11,
        fontweight="bold",
        color=colors["accent"],
        style="italic",
    )

    add_text(
        13,
        0.5,
        "Dual-Path Architecture",
        fontsize=11,
        fontweight="bold",
        color=colors["accent"],
        style="italic",
    )

    # Add threshold notation
    add_text(
        10,
        2.3,
        "τ_switch = 10⁻⁶",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="orange", alpha=0.9),
    )

    # Add mathematical domain info
    add_text(2.25, 5.5, "Domain Extension:", fontsize=10, fontweight="bold")
    add_text(2.25, 5.1, "𝕋 = ℝ ∪ {+∞, -∞, Φ}", fontsize=9)
    add_text(2.25, 4.7, "• Total arithmetic", fontsize=8)
    add_text(2.25, 4.4, "• No undefined ops", fontsize=8)

    # Add benefits box
    add_text(17.25, 1.8, "Benefits:", fontsize=10, fontweight="bold")
    add_text(17.25, 1.4, "• Bounded gradients", fontsize=8)
    add_text(17.25, 1.1, "• Stable training", fontsize=8)
    add_text(17.25, 0.8, "• Deterministic", fontsize=8)
    add_text(17.25, 0.5, "• Unbiased", fontsize=8)

    # Title and subtitle
    add_text(10, 6.5, "ZeroProofML: Transreal Computation Flow", fontsize=16, fontweight="bold")
    add_text(
        10,
        6.0,
        "Singularities become well-defined computational states",
        fontsize=11,
        style="italic",
        color="#555",
    )

    # Clean up plot
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    # Save final schematic
    output_dir = Path("/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "figure6_final_schematic.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        output_dir / "figure6_final_schematic.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    print("✓ Final professional schematic created!")
    return True


def create_minimalist_version():
    """Create an ultra-clean minimalist version"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Minimalist palette
    colors = {"box": "#FAFAFA", "guard": "#E8F5E9", "real": "#FFEBEE", "edge": "#424242"}

    # Simple boxes with minimal styling
    boxes = [
        # (x, y, width, height, text, color, edge_color)
        (1, 2, 2, 0.8, "Input\nx ∈ ℝⁿ", colors["box"], colors["edge"]),
        (4.5, 2, 2.5, 0.8, "P(x)/Q(x)", colors["box"], colors["edge"]),
        (8.5, 3, 2, 0.6, "Guard: P/Q", colors["guard"], "#4CAF50"),
        (8.5, 1, 2, 0.6, "Real: ∞/Φ", colors["real"], "#F44336"),
        (12, 2, 2, 0.8, "Output\n(y, tag)", colors["box"], colors["edge"]),
    ]

    for x, y, w, h, text, fcolor, ecolor in boxes:
        rect = Rectangle((x, y), w, h, facecolor=fcolor, edgecolor=ecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    # Decision point (simple circle)
    circle = Circle((7.5, 2.4), 0.3, facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=1.5)
    ax.add_patch(circle)
    ax.text(7.5, 2.4, "?", ha="center", va="center", fontsize=14, fontweight="bold")

    # Simple arrows
    arrows = [
        ((3, 2.4), (4.5, 2.4)),  # Input to P/Q
        ((7, 2.4), (7.2, 2.4)),  # P/Q to decision
        ((7.7, 2.6), (8.5, 3.2)),  # Decision to Guard
        ((7.7, 2.2), (8.5, 1.4)),  # Decision to Real
        ((10.5, 3.3), (12, 2.6)),  # Guard to Output
        ((10.5, 1.3), (12, 2.2)),  # Real to Output
    ]

    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="black")
        )

    # Labels
    ax.text(7.5, 3.2, "|Q| > τ", ha="center", va="center", fontsize=8)
    ax.text(8, 2.8, "YES", fontsize=8, color="green")
    ax.text(8, 1.8, "NO", fontsize=8, color="red")

    # Title
    ax.text(7, 4.2, "ZeroProofML Architecture", fontsize=14, fontweight="bold", ha="center")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")

    plt.tight_layout()

    # Save minimalist version
    output_dir = Path("/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures")
    plt.savefig(output_dir / "figure6_minimalist.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "figure6_minimalist.pdf", dpi=300, bbox_inches="tight")

    print("✓ Minimalist schematic created!")
    return True


if __name__ == "__main__":
    print("\nCreating final professional schematics...")

    try:
        create_professional_flow()
    except Exception as e:
        print(f"Error creating professional flow: {e}")

    try:
        create_minimalist_version()
    except Exception as e:
        print(f"Error creating minimalist version: {e}")

    print("\n✓ Final schematics complete!")
    print("  • figure6_final_schematic.pdf - Professional detailed version")
    print("  • figure6_minimalist.pdf - Clean minimalist version")
