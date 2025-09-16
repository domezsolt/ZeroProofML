#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch

def box(ax, xy, w, h, text):
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.2, facecolor="#f7f7f7")
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha='center', va='center', fontsize=9)
    return rect

def arrow(ax, p0, p1):
    arr = FancyArrowPatch(p0, p1, arrowstyle=ArrowStyle("-|>", head_length=6, head_width=3),
                          linewidth=1.0, color='black')
    ax.add_patch(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results/robotics/figures')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    box(ax, (0.5, 1.5), 2.0, 1.0, 'Inputs\n$[\\theta_1,\\theta_2,\\Delta x,\\Delta y]$')
    box(ax, (3.0, 1.5), 2.2, 1.0, 'TR ops\n(totalized $+,\\times,\\div$)')
    box(ax, (5.6, 1.5), 2.2, 1.0, 'Tag semantics\nREAL / $\\pm\\infty$ / $\\Phi$')
    box(ax, (8.2, 2.3), 2.2, 0.8, 'Mask‑REAL\n(REAL = classical)')
    box(ax, (8.2, 1.1), 2.2, 0.8, 'Saturating\n(near poles)')
    box(ax, (10.6, 1.5), 1.2, 1.0, 'Outputs\n$[\\Delta\\theta_1,\\Delta\\theta_2]$')

    arrow(ax, (2.6, 2.0), (3.0, 2.0))
    arrow(ax, (5.2, 2.0), (5.6, 2.0))
    arrow(ax, (7.8, 2.0), (8.2, 2.0))
    arrow(ax, (10.4, 2.0), (10.6, 2.0))
    ax.text(9.3, 2.05, 'Hybrid schedule', ha='center', va='bottom', fontsize=8)

    for name in ("tr_schematic.png", "tr_schematic.pdf"):
        out = os.path.join(args.outdir, name)
        plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved schematic to {args.outdir}/tr_schematic.(png|pdf)")

if __name__ == '__main__':
    main()

