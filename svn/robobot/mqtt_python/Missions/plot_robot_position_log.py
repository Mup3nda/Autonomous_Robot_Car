#!/usr/bin/env python3
"""Plot robot route from position log and save an image file.

Supports both formats:
- x,y
- timestamp,x,y,heading
"""

import argparse
import math
import os
from typing import List, Optional, Tuple


def _prepare_matplotlib():
    import matplotlib

    # Headless backend so plotting works over SSH and without GUI.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_samples(path: str) -> Tuple[List[Optional[float]], List[float], List[float], List[Optional[float]]]:
    times: List[Optional[float]] = []
    xs: List[float] = []
    ys: List[float] = []
    headings: List[Optional[float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]

            # New format: timestamp,x,y,heading
            if len(parts) >= 4:
                try:
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    h = float(parts[3])
                except ValueError:
                    continue
                times.append(t)
                xs.append(x)
                ys.append(y)
                headings.append(h)
                continue

            # Legacy format: x,y
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                times.append(None)
                xs.append(x)
                ys.append(y)
                headings.append(None)

    return times, xs, ys, headings


def plot_route(log_path: str, out_path: str, title: str, dpi: int = 160) -> str:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    _, xs, ys, headings = _load_samples(log_path)
    if not xs:
        raise RuntimeError(f"No valid samples found in: {log_path}")

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(xs, ys, color="tab:blue", linewidth=1.8, label="route")
    ax.scatter([xs[0]], [ys[0]], color="tab:green", s=80, label="start", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], color="tab:red", s=80, label="end", zorder=3)

    # Add sparse heading arrows when heading data is available.
    valid_idx = [i for i, h in enumerate(headings) if h is not None]
    if valid_idx:
        n_arrows = min(20, len(valid_idx))
        step = max(1, len(valid_idx) // n_arrows)
        picked = valid_idx[::step]
        if picked:
            arrow_len = 0.07
            for i in picked:
                h = headings[i]
                if h is None:
                    continue
                dx = arrow_len * math.cos(h)
                dy = arrow_len * math.sin(h)
                ax.arrow(
                    xs[i],
                    ys[i],
                    dx,
                    dy,
                    head_width=0.03,
                    head_length=0.04,
                    fc="tab:orange",
                    ec="tab:orange",
                    alpha=0.65,
                    length_includes_head=True,
                    zorder=2,
                )

    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot route from robot position log.")
    parser.add_argument("--input", default="/tmp/robot_position_log.txt", help="Input log file path")
    parser.add_argument("--output", default="/tmp/robot_position_route.png", help="Output image path (.png/.jpg)")
    parser.add_argument("--title", default="Robot Route", help="Plot title")
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI")
    args = parser.parse_args()

    out = plot_route(args.input, args.output, args.title, dpi=args.dpi)
    print(f"Saved route plot: {out}")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Missing plotting dependency: matplotlib")
        print("Install matplotlib and rerun.")
    except Exception as e:
        print(f"Plotting failed: {e}")
