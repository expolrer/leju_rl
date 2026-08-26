#!/usr/bin/env python3
"""Export TensorBoard scalars and plot stable forward-descent training curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WINDOW = 200


def moving_average(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def load_scalars(run: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags()["scalars"])
    primary = accumulator.Scalars("Train/mean_reward")
    steps = np.asarray([event.step for event in primary], dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        by_step = {event.step: event.value for event in events}
        values[tag] = np.asarray([by_step.get(int(step), np.nan) for step in steps], dtype=np.float64)
    return steps, values


def summarize(steps: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, object]:
    selected = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_anchor_pos",
        "Metrics/motion/error_anchor_rot",
        "Metrics/motion/error_body_pos",
        "Episode_Reward/pre_descent_gate_position",
        "Episode_Reward/pre_descent_gate_orientation",
        "Episode_Reward/pre_descent_stability",
        "Episode_Reward/descent_feet_position",
        "Episode_Reward/descent_feet_slide",
        "Episode_Reward/descent_swing_foot_contact",
        "Episode_Reward/descent_feet_under_clearance",
        "Episode_Reward/feet_contact_forces",
        "Episode_Reward/feet_slide_vel",
        "Episode_Termination/anchor_ori",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/ee_body_pos",
    ]
    metrics: dict[str, object] = {}
    for tag in selected:
        array = values[tag]
        finite = np.isfinite(array)
        valid_steps, valid = steps[finite], array[finite]
        smooth = moving_average(valid)
        finite_smooth = np.isfinite(smooth)
        metrics[tag] = {
            "first": float(valid[0]),
            "last": float(valid[-1]),
            "first_sma_200": float(smooth[finite_smooth][0]),
            "final_sma_200": float(smooth[finite_smooth][-1]),
            "minimum": float(np.min(valid)),
            "minimum_iteration": int(valid_steps[np.argmin(valid)]),
            "maximum": float(np.max(valid)),
            "maximum_iteration": int(valid_steps[np.argmax(valid)]),
        }
    return {
        "run": str(run_path),
        "iterations": [int(steps[0]), int(steps[-1])],
        "points": int(len(steps)),
        "smoothing_window": WINDOW,
        "metrics": metrics,
    }


def panel(ax, steps, values, tags, title, ylabel=None, zero=False):
    for tag, label, color in tags:
        ax.plot(steps, moving_average(values[tag]), label=label, color=color, linewidth=1.7)
    if zero:
        ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)


def save_overview(output: Path, steps, values):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panel(axes[0, 0], steps, values, [("Train/mean_reward", "Mean reward", "#087f8c")], "Training reward", "Reward")
    panel(axes[0, 1], steps, values, [("Train/mean_episode_length", "Episode length", "#ca6702")], "Episode length", "Steps")
    panel(
        axes[1, 0], steps, values,
        [
            ("Metrics/motion/error_anchor_pos", "Anchor position", "#005f73"),
            ("Metrics/motion/error_body_pos", "Body position", "#0a9396"),
            ("Metrics/motion/error_anchor_rot", "Anchor rotation", "#bb3e03"),
        ],
        "Tracking errors", "Error",
    )
    panel(
        axes[1, 1], steps, values,
        [
            ("Episode_Termination/time_out", "Timeout", "#2a9d8f"),
            ("Episode_Termination/anchor_pos", "Anchor position", "#e9c46a"),
            ("Episode_Termination/anchor_ori", "Anchor orientation", "#f4a261"),
            ("Episode_Termination/ee_body_pos", "Feet/body position", "#e76f51"),
        ],
        "Episode termination rates", "Events / episode",
    )
    fig.suptitle("Kuavo S53 stable forward-descent training (SMA200)", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_gate(output: Path, steps, values):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panel(
        axes[0, 0], steps, values,
        [
            ("Episode_Reward/pre_descent_gate_position", "Gate position", "#0077b6"),
            ("Episode_Reward/pre_descent_gate_orientation", "Gate orientation", "#00b4d8"),
        ],
        "Fixed-coordinate gate rewards", "Reward / step",
    )
    panel(
        axes[0, 1], steps, values,
        [("Episode_Reward/pre_descent_stability", "Velocity stability penalty", "#d00000")],
        "Pre-descent stability", "Penalty / step", True,
    )
    panel(
        axes[1, 0], steps, values,
        [
            ("Episode_Reward/motion_forward_pos", "Forward position", "#005f73"),
            ("Episode_Reward/motion_height", "Root height", "#0a9396"),
            ("Episode_Reward/motion_forward_velocity", "Forward velocity", "#94d2bd"),
        ],
        "Task-space tracking", "Reward / step",
    )
    panel(
        axes[1, 1], steps, values,
        [
            ("Metrics/motion/error_anchor_lin_vel", "Anchor linear velocity", "#9b2226"),
            ("Metrics/motion/error_anchor_ang_vel", "Anchor angular velocity", "#ee9b00"),
        ],
        "Velocity tracking errors", "Error",
    )
    fig.suptitle("Fixed platform switch and stabilization gate (SMA200)", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_descent(output: Path, steps, values):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panel(
        axes[0, 0], steps, values,
        [
            ("Episode_Reward/descent_feet_position", "Descent feet position", "#007f5f"),
            ("Episode_Reward/motion_feet_pos", "Global feet position", "#55a630"),
            ("Episode_Reward/motion_feet_xy", "Global feet XY", "#aacc00"),
        ],
        "Foot placement tracking", "Reward / step",
    )
    panel(
        axes[0, 1], steps, values,
        [
            ("Episode_Reward/descent_feet_slide", "Descent contact slide", "#d00000"),
            ("Episode_Reward/descent_swing_foot_contact", "Swing-foot contact", "#e85d04"),
            ("Episode_Reward/descent_feet_under_clearance", "Under-clearance", "#ffba08"),
        ],
        "Descent contact penalties", "Penalty / step", True,
    )
    panel(
        axes[1, 0], steps, values,
        [
            ("Episode_Reward/feet_slide_vel", "All-phase slide", "#6a040f"),
            ("Episode_Reward/feet_contact_forces", "Contact force", "#9d0208"),
            ("Episode_Reward/action_rate_l2", "Action rate", "#dc2f02"),
        ],
        "Sim2real-oriented penalties", "Penalty / step", True,
    )
    panel(
        axes[1, 1], steps, values,
        [
            ("Episode_Reward/motion_feet_vel", "Feet linear velocity", "#4361ee"),
            ("Episode_Reward/motion_feet_ang_vel", "Feet angular velocity", "#7209b7"),
            ("Episode_Reward/motion_feet_under_clearance", "Global under-clearance", "#f72585"),
        ],
        "Foot motion quality", "Reward / penalty",
    )
    fig.suptitle("Forward descent stepping constraints (SMA200)", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    global run_path
    run_path = args.run
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps, values = load_scalars(args.run)

    tags = sorted(values)
    with (args.output_dir / "stable_training_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", *tags])
        for index, step in enumerate(steps):
            writer.writerow([int(step), *[values[tag][index] for tag in tags]])

    summary = summarize(steps, values)
    (args.output_dir / "stable_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_overview(args.output_dir / "stable_training_overview.png", steps, values)
    save_gate(args.output_dir / "platform_stability_gate_curves.png", steps, values)
    save_descent(args.output_dir / "descent_constraint_curves.png", steps, values)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
