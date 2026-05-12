#!/usr/bin/env python3
"""Create separate plots from Nav_Platform_Kalman CSV logs.

This script writes one PNG per figure so you can compare them individually.
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_log(path: str) -> Dict[str, List[Optional[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    data: Dict[str, List[Optional[float]]] = defaultdict(list)
    states: List[str] = []

    for row in rows:
        states.append(row.get("state_machine", ""))
        for key, value in row.items():
            if key == "state_machine":
                continue
            data[key].append(_to_float(value) if value is not None else None)

    data["state_machine"] = states  # type: ignore[assignment]
    return data


def _time_axis(data: Dict[str, List[Optional[float]]]) -> List[float]:
    times = data.get("timestamp_s", [])
    valid_times = [t for t in times if t is not None]
    if not valid_times:
        return list(range(len(times)))
    t0 = float(valid_times[0])
    return [float(t - t0) if t is not None else math.nan for t in times]


def _save_fig(fig, path: str, dpi: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path


def _first_valid(values):
    for value in values:
        if value is not None:
            return value
    return None


def plot_positions(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, data["meas_global_x_m"], color="tab:blue", label="measured x", linewidth=1.2)
    ax.plot(t, data["kf_x_m"], color="tab:orange", label="filtered x", linewidth=1.4)
    ax.plot(t, data["meas_global_z_m"], color="tab:green", label="measured z", linewidth=1.2)
    ax.plot(t, data["kf_z_m"], color="tab:red", label="filtered z", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_velocities(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, data["kf_vx_mps"], color="tab:purple", label="v_x", linewidth=1.4)
    ax.plot(t, data["kf_vz_mps"], color="tab:brown", label="v_z", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (m/s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_path(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(data["meas_global_x_m"], data["meas_global_z_m"], color="tab:blue", alpha=0.55, label="measured path")
    ax.plot(data["kf_x_m"], data["kf_z_m"], color="tab:orange", linewidth=1.5, label="filtered path")
    start_x = _first_valid(data["meas_global_x_m"])
    start_z = _first_valid(data["meas_global_z_m"])
    if start_x is not None and start_z is not None:
        ax.scatter([start_x], [start_z], s=45, color="tab:green", label="start")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_distance(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, data["real_distance_to_platform_m"], color="tab:red", label="real distance", linewidth=1.5)
    ax.plot(t, data["drive_distance_m"], color="tab:blue", label="drive distance", linewidth=1.2)
    ax.axhline(0.31, color="black", linestyle="--", linewidth=1.0, label="safe stop")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_commands(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, data["linear_cmd_mps"], color="tab:green", label="linear cmd", linewidth=1.4)
    ax.plot(t, data["angular_cmd_radps"], color="tab:purple", label="angular cmd", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("command")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_innovations(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, data["innovation_x_m"], color="tab:blue", label="innovation x", linewidth=1.2)
    ax.plot(t, data["innovation_z_m"], color="tab:orange", label="innovation z", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("residual (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    return _save_fig(fig, out_path, dpi)


def plot_state_timeline(data, out_path, title, dpi=160):
    plt = _prepare_matplotlib()
    t = _time_axis(data)
    states = data["state_machine"]
    mapping: Dict[str, int] = {}
    y_values: List[int] = []
    labels: List[str] = []
    for state in states:
        if state not in mapping:
            mapping[state] = len(mapping)
            labels.append(state)
        y_values.append(mapping[state])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(t, y_values, where="post", color="tab:red", linewidth=1.5)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state")
    ax.grid(True, linestyle="--", alpha=0.35)
    return _save_fig(fig, out_path, dpi)


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(this_dir, "MissionLogs", "nav_platform_kalman_log.csv")
    default_out_dir = os.path.join(this_dir, "MissionLogs", "nav_platform_kalman_plots")

    parser = argparse.ArgumentParser(description="Create separate plots from Nav_Platform_Kalman logs.")
    parser.add_argument("--input", default=default_input, help="Input CSV log path")
    parser.add_argument("--output-dir", default=default_out_dir, help="Directory to write PNG files")
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI")
    args = parser.parse_args()

    data = _load_log(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    outputs = [
        plot_positions(data, os.path.join(args.output_dir, "01_positions.png"), "Kalman: Measured vs Filtered Positions", dpi=args.dpi),
        plot_velocities(data, os.path.join(args.output_dir, "02_velocities.png"), "Kalman: Velocity Estimates", dpi=args.dpi),
        plot_path(data, os.path.join(args.output_dir, "03_path.png"), "Kalman: Measured vs Filtered Path", dpi=args.dpi),
        plot_distance(data, os.path.join(args.output_dir, "04_distance.png"), "Kalman: Distance to Platform", dpi=args.dpi),
        plot_commands(data, os.path.join(args.output_dir, "05_commands.png"), "Kalman: Control Commands", dpi=args.dpi),
        plot_innovations(data, os.path.join(args.output_dir, "06_innovations.png"), "Kalman: Innovation / Residual", dpi=args.dpi),
        plot_state_timeline(data, os.path.join(args.output_dir, "07_state_timeline.png"), "Kalman: State Machine Timeline", dpi=args.dpi),
    ]

    for path in outputs:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Missing plotting dependency: matplotlib")
        print("Install matplotlib and rerun.")