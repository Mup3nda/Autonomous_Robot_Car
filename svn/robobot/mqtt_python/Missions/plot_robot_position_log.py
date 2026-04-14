#!/usr/bin/env python3
"""Plot robot route from position log and save an image file.

Supports both formats:
- x,y
- timestamp,x,y,heading
- timestamp,x,y,heading,objective_name
- timestamp,x,y,heading,objective_name,objective_index
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


def _load_samples(
    path: str,
) -> Tuple[
    List[Optional[float]],
    List[float],
    List[float],
    List[Optional[float]],
    List[Optional[str]],
    List[Optional[int]],
]:
    times: List[Optional[float]] = []
    xs: List[float] = []
    ys: List[float] = []
    headings: List[Optional[float]] = []
    objectives: List[Optional[str]] = []
    objective_indices: List[Optional[int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]

            # New format: timestamp,x,y,heading[,objective_name[,objective_index]]
            if len(parts) >= 4:
                try:
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    h = float(parts[3])
                except ValueError:
                    continue
                objective = ",".join(parts[4:]).strip() if len(parts) >= 5 else None
                objective_idx = None
                if len(parts) >= 6:
                    try:
                        objective_idx = int(parts[-1])
                        objective = ",".join(parts[4:-1]).strip()
                    except ValueError:
                        objective_idx = None
                times.append(t)
                xs.append(x)
                ys.append(y)
                headings.append(h)
                objectives.append(objective or None)
                objective_indices.append(objective_idx)
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
                objectives.append(None)
                objective_indices.append(None)

    return times, xs, ys, headings, objectives, objective_indices


def _objective_start_indices(
    objectives: List[Optional[str]],
    objective_indices: List[Optional[int]],
) -> List[int]:
    """Return sample indices where a new objective starts."""
    starts: List[int] = []

    has_index_info = any(idx is not None for idx in objective_indices)
    if has_index_info:
        prev_idx: Optional[int] = None
        for i, idx in enumerate(objective_indices):
            if idx is None:
                continue
            if prev_idx is None or idx != prev_idx:
                starts.append(i)
                prev_idx = idx
        return starts

    prev_label: Optional[str] = None
    for i, label in enumerate(objectives):
        if not label:
            continue
        if prev_label is None or label != prev_label:
            starts.append(i)
        prev_label = label
    return starts


def plot_route(log_path: str, out_path: str, title: str, dpi: int = 160) -> str:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    times, xs, ys, headings, objectives, objective_indices = _load_samples(log_path)
    if not xs:
        raise RuntimeError(f"No valid samples found in: {log_path}")

    plt = _prepare_matplotlib()
    fig, (ax_route, ax_heading) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Plot route using a single color.
    ax_route.plot(xs, ys, color="tab:blue", linewidth=1.5, alpha=0.9, label="route")
    ax_route.scatter(xs, ys, s=8, color="tab:blue", alpha=0.4, zorder=2)

    objective_start_idx = _objective_start_indices(objectives, objective_indices)
    if objective_start_idx:
        start_x = [xs[i] for i in objective_start_idx]
        start_y = [ys[i] for i in objective_start_idx]
        ax_route.scatter(
            start_x,
            start_y,
            color="tab:purple",
            s=42,
            alpha=0.95,
            label="objective start",
            zorder=4,
        )

    ax_route.scatter([xs[0]], [ys[0]], color="tab:green", s=75, label="start", zorder=5)
    ax_route.scatter([xs[-1]], [ys[-1]], color="tab:red", s=75, label="end", zorder=5)

    # Add sparse heading arrows when heading data is available.
    # Heading over time (or sample index if timestamps are unavailable).
    valid_heading_idx = [i for i, h in enumerate(headings) if h is not None]
    has_time = all(t is not None for t in times) and len(times) > 0
    if has_time:
        t0 = float(times[0])
        x_axis = [float(t) - t0 for t in times]
        x_label = "time (s)"
    else:
        x_axis = list(range(len(xs)))
        x_label = "sample index"

    if valid_heading_idx:
        heading_deg = [math.degrees(float(headings[i])) for i in valid_heading_idx]
        heading_x = [x_axis[i] for i in valid_heading_idx]
        ax_heading.plot(heading_x, heading_deg, color="tab:orange", linewidth=1.4, label="heading (deg)")

    if objective_start_idx:
        start_h_idx = [i for i in objective_start_idx if i in valid_heading_idx]
        if start_h_idx:
            ax_heading.scatter(
                [x_axis[i] for i in start_h_idx],
                [math.degrees(float(headings[i])) for i in start_h_idx],
                color="tab:purple",
                s=30,
                alpha=0.95,
                label="objective start",
                zorder=4,
            )

    ax_route.set_title(title)
    ax_route.set_xlabel("x (m)")
    ax_route.set_ylabel("y (m)")
    ax_route.grid(True, linestyle="--", alpha=0.4)
    ax_route.axis("equal")
    ax_route.legend(loc="best")

    ax_heading.set_title("Heading Over Time")
    ax_heading.set_xlabel(x_label)
    ax_heading.set_ylabel("heading (deg)")
    ax_heading.grid(True, linestyle="--", alpha=0.4)
    ax_heading.legend(loc="best")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mission_logs_dir = os.path.join(this_dir, "MissionLogs")
    default_input = os.path.join(mission_logs_dir, "robot_position_log.txt")
    default_output = os.path.join(mission_logs_dir, "robot_position_route.png")
    parser = argparse.ArgumentParser(description="Plot route from robot position log.")
    parser.add_argument("--input", default=default_input, help="Input log file path")
    parser.add_argument("--output", default=default_output, help="Output image path (.png/.jpg)")
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
