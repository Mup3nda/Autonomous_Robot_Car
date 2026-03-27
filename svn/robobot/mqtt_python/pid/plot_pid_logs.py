#!/usr/bin/env python3
"""Plot PID telemetry logs written by sedge.py.

This utility reads:
- pid_slow.csv
- pid_medium.csv
- pid_fast.csv

and writes PNG plots to the same folder.
"""

import csv
import os
from typing import Dict, List, Optional


PROFILES = ("slow", "medium", "fast")


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def load_profile_csv(csv_path: str) -> Optional[Dict[str, List[float]]]:
    """Load one profile CSV into numeric lists.

    Returns None if file is missing or has no data rows.
    """
    if not os.path.exists(csv_path):
        return None

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="ascii", errors="ignore") as f:
        filtered = (line for line in f if line and not line.startswith("#"))
        reader = csv.DictReader(filtered)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    data: Dict[str, List[float]] = {
        "t": [],
        "target_velocity": [],
        "velocity": [],
        "error": [],
        "P": [],
        "I": [],
        "D": [],
        "Y": [],
        "line_integral": [],
        "line_deriv_filtered": [],
        "line_valid": [],
        "crossing": [],
        "edge_upd_cnt": [],
    }

    t0 = _to_float(rows[0].get("timestamp", "0"))
    for row in rows:
        ts = _to_float(row.get("timestamp", "0"))
        data["t"].append(ts - t0)
        data["target_velocity"].append(_to_float(row.get("target_velocity", "0")))
        data["velocity"].append(_to_float(row.get("velocity", "0")))
        data["error"].append(_to_float(row.get("error", "0")))
        data["P"].append(_to_float(row.get("P", "0")))
        data["I"].append(_to_float(row.get("I", "0")))
        data["D"].append(_to_float(row.get("D", "0")))
        data["Y"].append(_to_float(row.get("Y", "0")))
        data["line_integral"].append(_to_float(row.get("line_integral", "0")))
        data["line_deriv_filtered"].append(_to_float(row.get("line_deriv_filtered", "0")))
        data["line_valid"].append(_to_int(row.get("line_valid", "0")))
        data["crossing"].append(_to_int(row.get("crossing", "0")))
        data["edge_upd_cnt"].append(_to_int(row.get("edge_upd_cnt", "0")))

    return data


def _prepare_matplotlib():
    import matplotlib

    # Use non-interactive backend so plots can be rendered over SSH/headless.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_profile(data: Dict[str, List[float]], profile: str, out_dir: str) -> List[str]:
    """Create detailed plots for one profile.

    Returns created image paths.
    """
    plt = _prepare_matplotlib()
    created: List[str] = []
    t = data["t"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"PID Profile: {profile}")

    axes[0].plot(t, data["error"], label="error", color="tab:red", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, data["P"], label="P", linewidth=1.0)
    axes[1].plot(t, data["I"], label="I", linewidth=1.0)
    axes[1].plot(t, data["D"], label="D", linewidth=1.0)
    axes[1].set_ylabel("PID Terms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, data["Y"], label="turn cmd Y", color="tab:purple", linewidth=1.0)
    axes[2].set_ylabel("Y (turn)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(t, data["target_velocity"], label="target v", color="tab:gray", linewidth=1.0)
    axes[3].plot(t, data["velocity"], label="actual v", color="tab:green", linewidth=1.0)
    axes[3].set_ylabel("Velocity m/s")
    axes[3].set_xlabel("Time s")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"plot_pid_{profile}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    created.append(out_path)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig2.suptitle(f"State Profile: {profile}")

    axes2[0].plot(t, data["line_integral"], label="integral state", color="tab:blue", linewidth=1.0)
    axes2[0].plot(t, data["line_deriv_filtered"], label="filtered derivative state", color="tab:orange", linewidth=1.0)
    axes2[0].set_ylabel("Internal State")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(loc="upper right")

    axes2[1].plot(t, data["line_valid"], label="line_valid", linewidth=1.0)
    axes2[1].plot(t, data["crossing"], label="crossing", linewidth=1.0)
    axes2[1].set_ylabel("Flags")
    axes2[1].set_xlabel("Time s")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(loc="upper right")

    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, f"plot_state_{profile}.png")
    fig2.savefig(out_path2, dpi=130)
    plt.close(fig2)
    created.append(out_path2)

    return created


def plot_all_profiles(profile_data: Dict[str, Dict[str, List[float]]], out_dir: str) -> List[str]:
    """Create cross-profile comparison plots."""
    plt = _prepare_matplotlib()
    created: List[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("PID Profile Comparison")

    for profile, data in profile_data.items():
        t = data["t"]
        axes[0].plot(t, data["error"], label=profile, linewidth=1.0)
        axes[1].plot(t, data["Y"], label=profile, linewidth=1.0)
        axes[2].plot(t, data["velocity"], label=profile, linewidth=1.0)

    axes[0].set_ylabel("Error")
    axes[1].set_ylabel("Y (turn)")
    axes[2].set_ylabel("Velocity m/s")
    axes[2].set_xlabel("Time s")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "plot_comparison_profiles.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    created.append(out_path)

    return created


def generate_all_plots(pid_dir: Optional[str] = None) -> List[str]:
    """Load all profile CSVs and generate per-profile + comparison plots."""
    if pid_dir is None:
        pid_dir = os.path.dirname(os.path.abspath(__file__))

    profile_data: Dict[str, Dict[str, List[float]]] = {}
    for profile in PROFILES:
        csv_path = os.path.join(pid_dir, f"pid_{profile}.csv")
        data = load_profile_csv(csv_path)
        if data is not None:
            profile_data[profile] = data

    if not profile_data:
        print(f"No PID csv data found in {pid_dir}")
        return []

    created: List[str] = []
    for profile, data in profile_data.items():
        created.extend(plot_profile(data, profile, pid_dir))

    if len(profile_data) > 1:
        created.extend(plot_all_profiles(profile_data, pid_dir))

    print("Created plot files:")
    for path in created:
        print(path)
    return created


if __name__ == "__main__":
    try:
        generate_all_plots()
    except ImportError as exc:
        print("Missing plotting dependency.")
        print("Please install matplotlib, then run again.")
        print(f"Import error: {exc}")
