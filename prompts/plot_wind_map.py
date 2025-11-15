#!/usr/bin/env python3
"""
plot_wind_map_final.py

Professional dark-gray wind speed heatmap with direction arrows:
 - Row 1 is top-left (y-axis flipped correctly).
 - Very dark gray background; very light gray text.
 - Every cell displays an arrow (same color) — no arrows are filtered out.
 - Arrows sized and styled for clarity on a 100x100 grid.

Usage:
    python plot_wind_map_final.py --out wind_final.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ---- Files (adjust if necessary) ----
META_FILE = "./prompts/map_100_meta.json"
WIND_SPEED_FILE = "./prompts/map_100_windSpeed.csv"
WIND_DIR_FILE = "./prompts/map_100_windDir.csv"

def load_csv_matrix(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception:
        rows = []
        with open(path, "r") as f:
            for line in f:
                vals = [v for v in line.strip().split(",") if v != ""]
                if vals:
                    rows.append([float(v) for v in vals])
        return np.array(rows)

def mark_cell(ax, rows, pos, marker, facecolor, label, fontsize, text_color):
    """Mark a cell (1-indexed row, col) with a compact label. Row 1 is top."""
    if not pos:
        return
    r, c = int(pos[0]) - 1, int(pos[1]) - 1
    # For our coordinate system (imshow with origin='lower' then inverted),
    # the center of data cell (r,c) is x=c+0.5, y=r+0.5 BEFORE inversion.
    x = c + 0.5
    y = r + 0.5
    ax.scatter([x], [y], s=110, marker=marker, edgecolor="#1a1a1a", linewidth=0.9,
               facecolor=facecolor, zorder=6)
    ax.text(x + 0.85, y - 0.28, f"{label} ({r+1},{c+1})",
            fontsize=fontsize, color=text_color,
            bbox=dict(boxstyle="round,pad=0.18", fc="#171717", ec="#333333", alpha=0.92),
            zorder=7)

def main():
    parser = argparse.ArgumentParser(description="Wind Map")
    parser.add_argument("--out", default="wind_map_final.png", help="Output image filename")
    parser.add_argument("--arrow-color", default="#d8d8d8", help="Hex color for all arrows (default light gray)")
    parser.add_argument("--arrow-scale", type=float, default=0.38,
                        help="Max arrow length (in cell units) for the strongest wind (default 0.38)")
    parser.add_argument("--no-show", action="store_true", help="Save but do not show")
    args = parser.parse_args()

    for p in (META_FILE, WIND_SPEED_FILE, WIND_DIR_FILE):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(META_FILE, "r") as f:
        meta = json.load(f)
    speed = load_csv_matrix(WIND_SPEED_FILE)
    winddir = load_csv_matrix(WIND_DIR_FILE)
    if speed.shape != winddir.shape:
        raise ValueError("Speed and direction matrices must have the same shape")

    rows, cols = speed.shape

    # Protect against invalid data
    if np.isnan(speed).all():
        raise ValueError("Speed matrix is all NaN")
    vmin = float(np.nanmin(speed))
    vmax = float(np.nanmax(speed))
    if vmax <= 0 or np.isnan(vmax):
        vmax = 1.0

    # ---------------- Vector components ----------------
    # Convention: winddir degrees: 0 = east (positive x), 90 = north (positive y), CCW.
    rad = np.deg2rad(winddir)
    u = np.cos(rad) * speed
    v = np.sin(rad) * speed

    # We will draw the heatmap using imshow with origin='lower' then invert the y-axis
    # to place row 1 at top. Because we invert the axis, flip the v component so arrows
    # visually point in the intended directions.
    v_plot = -v

    # Build per-cell center coordinates (every cell)
    x_centers = (np.arange(cols) + 0.5)
    y_centers = (np.arange(rows) + 0.5)
    XX, YY = np.meshgrid(x_centers, y_centers)

    # ---------------- Arrow scaling: ensure all arrows are drawn (minlength=0) -----------
    desired_max_length = max(1e-6, float(args.arrow_scale))
    quiver_scale = vmax / desired_max_length
    # width tuned for a 100x100 grid; scales down if grid larger
    base_width = 0.0038
    width = base_width * (100.0 / max(rows, cols))

    # ---------------- Style: very dark gray background, very light gray text -----------
    bg_color = "#121317"         # very dark gray (not pure black)
    text_color = "#e6e6e6"       # very light gray (not pure white)
    plt.style.use("dark_background")
    small = 9
    plt.rcParams.update({
        "font.size": small,
        "axes.titlesize": small + 2,
        "axes.labelsize": small,
        "xtick.labelsize": small - 1,
        "ytick.labelsize": small - 1,
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
    })

    fig_w = 8.4
    fig_h = max(5.0, fig_w * (rows / max(cols, 1)) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # ---------------- Heatmap: imshow origin='lower', then invert y-axis to make row 1 top-left ----
    extent = [0, cols, 0, rows]
    im = ax.imshow(speed, cmap="magma", origin="lower", extent=extent,
                   interpolation="bilinear", aspect="auto",
                   norm=colors.Normalize(vmin=vmin, vmax=vmax))

    # Invert the y-axis so row 1 is at the top-left
    ax.invert_yaxis()

    # Single, compact colorbar (heatmap only)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Wind speed", color=text_color, fontsize=small)
    cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=small - 1)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_color)

    # ---------------- Quiver (every cell), single uniform color ----------------
    # Pass color argument (single color) and minlength=0 to prevent Matplotlib from dropping tiny arrows.
    q = ax.quiver(
        XX, YY, u, v_plot,
        color=args.arrow_color,
        angles="xy", scale_units="xy", scale=quiver_scale,
        width=width, headwidth=3.0, headlength=3.6, headaxislength=2.7,
        minlength=0.0, alpha=0.92, pivot="mid", zorder=3
    )

    # ---------------- Ticks / labels aligned with cell centers ----------------
    xtick_step = max(1, int(cols / 8))
    ytick_step = max(1, int(rows / 8))
    ax.set_xticks(np.arange(0.5, cols, xtick_step))
    ax.set_xticklabels(np.arange(1, cols + 1, xtick_step), color=text_color)
    # For Y: because we've inverted the axis, labels 1..rows appear top-to-bottom if we set ticks at centers.
    y_indices = np.arange(0, rows, ytick_step)
    y_tick_positions = y_indices + 0.5
    y_tick_labels = (y_indices + 1).astype(int)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, color=text_color)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)   # axis inverted above keeps row1 at top

    # Minimal spines, subtle border color
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#333333")

    ax.set_xlabel("Column", color=text_color, fontsize=small)
    ax.set_ylabel("Row", color=text_color, fontsize=small)
    ax.set_title("Wind Speed", color=text_color, pad=6)

    # Start / Finish markers (compact)
    mark_cell(ax, rows, meta.get("startPos"), marker='o', facecolor="#2ca02c",
              label="Start", fontsize=small - 1, text_color=text_color)
    mark_cell(ax, rows, meta.get("finishPos"), marker='*', facecolor="#ff6b61",
              label="Finish", fontsize=small - 1, text_color=text_color)

    # Tidy tick colors
    ax.tick_params(colors=text_color, which="both")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved: {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
