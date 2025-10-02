#!/usr/bin/env python3
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- rendering defaults ----
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["figure.dpi"] = 220

BOX_FACE = "#f1f5ff"
BOX_EDGE = "#2d3e91"
TEXT_COLOR = "#0f172a"
ARROW_COLOR = "#1b1b1b"


# ---- helpers ---------------------------------------------------------------
def _data_size_in_pixels(ax, w_data, h_data):
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((w_data, h_data))
    return abs(x1 - x0), abs(y1 - y0)


def _fit_text_inside(ax, txt, w_data, h_data, pad_px=14, min_fs=8.0, max_fs=11.0):
    """Shrink (only) until text fits inside a (w_data, h_data) box."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    w_px, h_px = _data_size_in_pixels(ax, w_data, h_data)
    limit_w = max(w_px - 2 * pad_px, 8)
    limit_h = max(h_px - 2 * pad_px, 8)

    fs = max_fs
    txt.set_fontsize(fs)
    for _ in range(70):
        bbox = txt.get_window_extent(renderer=rend)
        if bbox.width <= limit_w and bbox.height <= limit_h:
            break
        fs -= 0.3
        if fs <= min_fs:
            fs = min_fs
            txt.set_fontsize(fs)
            break
        txt.set_fontsize(fs)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()


def add_box(ax, xy, w, h, text, max_fontsize=11.0):
    """Create rounded box and centered text. We fit text after limits are set."""
    rect = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.32,rounding_size=0.16",
        linewidth=1.3,
        facecolor=BOX_FACE,
        edgecolor=BOX_EDGE,
        zorder=1.0,
    )
    ax.add_patch(rect)
    t = ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=max_fontsize,
        color=TEXT_COLOR,
        linespacing=1.2,
        wrap=True,
        zorder=2.0,
        clip_on=True,
    )
    t.set_clip_path(rect)

    # convenience accessors
    box = {
        "rect": rect,
        "text": t,
        "w": w,
        "h": h,
        "x": xy[0],
        "y": xy[1],
    }
    box["left"] = box["x"]
    box["right"] = box["x"] + w
    box["bottom"] = box["y"]
    box["top"] = box["y"] + h
    box["mid_y"] = box["y"] + h / 2
    box["mid_x"] = box["x"] + w / 2
    box["max_fontsize"] = max_fontsize
    return box


def fit_all_text_after_limits(ax, boxes):
    for b in boxes:
        _fit_text_inside(
            ax, b["text"], b["w"], b["h"], pad_px=14, min_fs=8.2, max_fs=b["max_fontsize"]
        )


def arrow(ax, p0, p1, rad=0.0, mscale=9, lw=1.15, z=2.1):
    """Uniform, unfilled arrow; we pre-offset endpoints so no shrink is needed."""
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="->",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=mscale,
        lw=lw,
        ec=ARROW_COLOR,
        fc="none",
        capstyle="round",
        joinstyle="round",
        zorder=z,
        clip_on=True,
    )
    ax.add_patch(arr)


# ---- figure -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/robotics/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 5.1))
    ax.axis("off")

    # layout
    core_y, core_h, core_w, gap = 2.2, 2.30, 4.60, 1.60
    branch_w, branch_h = 4.60, 1.85

    # boxes
    inputs = add_box(
        ax, (0.5, core_y), core_w, core_h, r"Inputs" "\n" r"$[\theta_1,\theta_2,\Delta x,\Delta y]$"
    )
    ops = add_box(
        ax,
        (inputs["right"] + gap, core_y),
        core_w,
        core_h,
        r"Totalized TR ops" "\n" r"(no undefined $+,\times,\div$)",
    )
    tags = add_box(
        ax,
        (ops["right"] + gap, core_y),
        core_w,
        core_h,
        r"Tagged values" "\n" r"REAL / $\pm\infty$ / $\Phi$",
    )
    mask = add_box(
        ax,
        (tags["right"] + gap, core_y + core_h + 0.50),
        branch_w,
        branch_h,
        r"Mask-REAL gradients" "\n" r"REAL path $\Rightarrow$ classical derivatives",
    )
    sat = add_box(
        ax,
        (tags["right"] + gap, core_y - branch_h - 0.50),
        branch_w,
        branch_h,
        r"Saturating gradients" "\n" r"Bounded updates when $|Q(x)|\ll 1$",
    )
    outs = add_box(
        ax,
        (mask["right"] + gap, core_y),
        core_w,
        core_h,
        r"Outputs" "\n" r"$[\Delta\theta_1,\Delta\theta_2]$",
    )

    boxes = [inputs, ops, tags, mask, sat, outs]

    # headline
    top_y = max(mask["top"], core_y + core_h)
    ax.text(
        (tags["mid_x"] + outs["mid_x"]) / 2,
        top_y + 1.02,
        "Hybrid scheduler selects REAL vs. saturating branch",
        ha="center",
        va="center",
        fontsize=11.0,
        color=ARROW_COLOR,
        bbox=dict(facecolor="white", alpha=0.96, pad=2.2, edgecolor="none"),
        zorder=3.0,
        clip_on=False,
    )

    # final limits BEFORE text fitting
    left = inputs["left"]
    right = outs["right"]
    bottom = min(sat["bottom"] - 0.60, core_y - 0.90)
    top = top_y + 1.60
    ax.set_xlim(left - 0.9, right + 1.1)
    ax.set_ylim(bottom - 0.6, top + 0.6)

    # now shrink text to fit exactly
    fit_all_text_after_limits(ax, boxes)

    # ---- arrows with fixed margins so sizes match everywhere ----
    # margin offset (data units) between arrow tip and box side
    margin = 0.14
    arrow_y = core_y + core_h / 2

    # core row (left-to-right)
    arrow(ax, (inputs["right"] + margin, arrow_y), (ops["left"] - margin, arrow_y), rad=0.0)
    arrow(ax, (ops["right"] + margin, arrow_y), (tags["left"] - margin, arrow_y), rad=0.0)

    # split from tags -> (mask, sat) (same size both)
    arrow(ax, (tags["right"] + margin, arrow_y), (mask["left"] - margin, mask["mid_y"]), rad=0.10)
    arrow(ax, (tags["right"] + margin, arrow_y), (sat["left"] - margin, sat["mid_y"]), rad=-0.10)

    # join into outputs (same size both)
    join_up = arrow_y + 0.55
    join_dn = arrow_y - 0.55
    arrow(ax, (mask["right"] - margin, mask["mid_y"]), (outs["left"] + margin, join_up), rad=-0.07)
    arrow(ax, (sat["right"] - margin, sat["mid_y"]), (outs["left"] + margin, join_dn), rad=0.07)

    # save
    for name in ("tr_schematic.png", "tr_schematic.pdf"):
        plt.savefig(os.path.join(args.outdir, name), bbox_inches=None, pad_inches=0.30)
    plt.close(fig)
    print(f"Saved schematic to {args.outdir}/tr_schematic.(png|pdf)")


if __name__ == "__main__":
    main()
