#!/usr/bin/env python3
"""Plot Kuavo S53 ascent-platform-forward-descent TensorBoard metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OVERVIEW_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_anchor_rot",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Loss/value_function",
    "Loss/surrogate",
    "Policy/mean_noise_std",
    "Metrics/motion/sampling_entropy",
    "Metrics/motion/sampling_top1_prob",
]

REWARD_TAGS = [
    "Episode_Reward/motion_global_anchor_pos",
    "Episode_Reward/motion_global_anchor_ori",
    "Episode_Reward/motion_body_pos",
    "Episode_Reward/motion_body_ori",
    "Episode_Reward/motion_knee_pos",
    "Episode_Reward/motion_feet_pos",
    "Episode_Reward/motion_feet_vel",
    "Episode_Reward/motion_anchor_lateral_pos",
    "Episode_Reward/motion_feet_xy",
    "Episode_Reward/motion_late_body_pos",
    "Episode_Reward/motion_late_feet_pos",
    "Episode_Reward/motion_phase_progress",
    "Episode_Reward/motion_feet_under_clearance",
    "Episode_Reward/motion_forward_pos",
    "Episode_Reward/motion_height",
    "Episode_Reward/motion_forward_velocity",
    "Episode_Reward/backward_velocity",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/feet_slide_vel",
    "Episode_Reward/undesired_contacts",
]

TERMINATION_TAGS = [
    "Episode_Termination/time_out",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/ee_body_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smooth", type=int, default=200)
    return parser.parse_args()


def load_runs(run_dirs: list[Path]) -> dict[str, dict[int, float]]:
    merged: dict[str, dict[int, float]] = {}
    for run_dir in run_dirs:
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event file in {run_dir}")
        for event_file in event_files:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                points = merged.setdefault(tag, {})
                for event in accumulator.Scalars(tag):
                    # A resumed runner starts with empty episode buffers. Keep the earlier
                    # run in overlapping iterations so those warm-up zeros do not look
                    # like a real policy regression; use the resumed run only after it
                    # extends beyond the previous log.
                    points.setdefault(int(event.step), float(event.value))
    return merged


def series(data: dict[str, dict[int, float]], tag: str) -> tuple[np.ndarray, np.ndarray]:
    points = data.get(tag, {})
    steps = np.asarray(sorted(points), dtype=np.int64)
    values = np.asarray([points[int(step)] for step in steps], dtype=np.float64)
    return steps, values


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    width = min(window, values.size)
    kernel = np.ones(width, dtype=np.float64) / width
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.full(width - 1, np.nan)
    return np.concatenate((prefix, valid))


def plot_raw_and_smooth(ax, steps, values, label, window, color=None):
    if values.size == 0:
        return
    line = ax.plot(steps, values, alpha=0.14, linewidth=0.7, color=color)[0]
    ax.plot(
        steps,
        smooth(values, window),
        linewidth=1.8,
        label=label,
        color=line.get_color(),
    )


def style_axes(axes):
    for ax in np.asarray(axes).reshape(-1):
        ax.axvline(79997, color="#666666", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("Training iteration")


def plot_overview(data, output_dir: Path, window: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    steps, values = series(data, "Train/mean_reward")
    plot_raw_and_smooth(axes[0, 0], steps, values, "Mean reward", window, "#1976d2")
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].set_ylabel("Reward")

    steps, values = series(data, "Train/mean_episode_length")
    plot_raw_and_smooth(axes[0, 1], steps, values, "Episode length", window, "#00897b")
    axes[0, 1].set_title("Mean episode length")
    axes[0, 1].set_ylabel("Steps")

    for tag, label, color in [
        ("Metrics/motion/error_anchor_pos", "Root position", "#d32f2f"),
        ("Metrics/motion/error_body_pos", "Body position", "#7b1fa2"),
    ]:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[1, 0], steps, values, label, window, color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Tracking position error")
    axes[1, 0].set_ylabel("Error (m, log scale)")

    for tag, label, color in [
        ("Loss/value_function", "Value loss", "#f57c00"),
        ("Loss/surrogate", "Surrogate loss", "#455a64"),
    ]:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[1, 1], steps, values, label, window, color)
    axes[1, 1].set_title("PPO losses")
    axes[1, 1].set_ylabel("Loss")

    style_axes(axes)
    for ax in axes.reshape(-1):
        ax.legend(loc="best")
    fig.suptitle("Kuavo S53 ascent-platform-forward-descent training", fontsize=16)
    fig.savefig(output_dir / "forward_updown_training_overview.png", dpi=180)
    plt.close(fig)


def plot_components(data, output_dir: Path, window: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    positives = [
        ("Episode_Reward/motion_global_anchor_pos", "Root position"),
        ("Episode_Reward/motion_body_pos", "Body position"),
        ("Episode_Reward/motion_knee_pos", "Knee position"),
        ("Episode_Reward/motion_feet_pos", "Feet position"),
        ("Episode_Reward/motion_feet_vel", "Feet velocity"),
    ]
    for tag, label in positives:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[0, 0], steps, values, label, window)
    axes[0, 0].set_title("Positive reward components")
    axes[0, 0].set_ylabel("Episode reward contribution")

    for tag, label in [
        ("Episode_Reward/motion_forward_pos", "Forward position"),
        ("Episode_Reward/motion_height", "Root height"),
        ("Episode_Reward/motion_forward_velocity", "Forward velocity"),
        ("Episode_Reward/backward_velocity", "Backward penalty"),
        ("Episode_Reward/motion_phase_progress", "Phase progress"),
    ]:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[0, 1], steps, values, label, window)
    axes[0, 1].set_title("Task-space and phase rewards")
    axes[0, 1].set_ylabel("Episode reward contribution")

    for tag, label in [
        ("Episode_Reward/action_rate_l2", "Action-rate penalty"),
        ("Episode_Reward/feet_slide_vel", "Foot-slide penalty"),
        ("Episode_Reward/undesired_contacts", "Contact penalty"),
        ("Episode_Reward/motion_feet_under_clearance", "Under-clearance penalty"),
    ]:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[1, 0], steps, values, label, window)
    axes[1, 0].set_title("Penalty components")
    axes[1, 0].set_ylabel("Episode reward contribution")

    termination_values = []
    for tag in TERMINATION_TAGS:
        steps, values = series(data, tag)
        plot_raw_and_smooth(axes[1, 1], steps, values, tag.rsplit("/", 1)[-1], window)
        termination_values.extend(values[steps > 80000].tolist())
    axes[1, 1].set_title("Episode termination counters")
    axes[1, 1].set_ylabel("Logged count")
    if termination_values:
        upper = max(0.5, float(np.quantile(termination_values, 0.995)) * 1.15)
        axes[1, 1].set_ylim(-0.05, upper)

    style_axes(axes)
    for ax in axes.reshape(-1):
        ax.legend(loc="best")
    fig.suptitle("Kuavo S53 combined-task reward decomposition", fontsize=16)
    fig.savefig(output_dir / "forward_updown_reward_components.png", dpi=180)
    plt.close(fig)


def write_csv(data, output_dir: Path):
    tags = [tag for tag in OVERVIEW_TAGS + REWARD_TAGS + TERMINATION_TAGS if tag in data]
    steps = sorted({step for tag in tags for step in data[tag]})
    with (output_dir / "forward_updown_training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", *tags])
        for step in steps:
            writer.writerow([step, *[data[tag].get(step, "") for tag in tags]])


def metric_summary(data, tag: str, window: int) -> dict[str, float | int] | None:
    steps, values = series(data, tag)
    if values.size == 0:
        return None
    smoothed = smooth(values, window)
    valid = np.flatnonzero(~np.isnan(smoothed))
    result: dict[str, float | int] = {
        "first_iteration": int(steps[0]),
        "last_iteration": int(steps[-1]),
        "first_value": float(values[0]),
        "last_value": float(values[-1]),
        "minimum": float(np.min(values)),
        "minimum_iteration": int(steps[int(np.argmin(values))]),
        "maximum": float(np.max(values)),
        "maximum_iteration": int(steps[int(np.argmax(values))]),
    }
    if valid.size:
        result[f"final_sma_{window}"] = float(smoothed[valid[-1]])
        peak = valid[int(np.argmax(smoothed[valid]))]
        result[f"maximum_sma_{window}"] = float(smoothed[peak])
        result[f"maximum_sma_{window}_iteration"] = int(steps[peak])
    return result


def write_summary(data, run_dirs: list[Path], output_dir: Path, window: int):
    tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/error_anchor_pos",
        "Metrics/motion/error_anchor_rot",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_joint_pos",
        "Episode_Reward/motion_feet_pos",
        "Episode_Reward/motion_anchor_lateral_pos",
        "Episode_Reward/motion_feet_xy",
        "Episode_Reward/motion_late_body_pos",
        "Episode_Reward/motion_late_feet_pos",
        "Episode_Reward/motion_phase_progress",
        "Episode_Reward/motion_feet_under_clearance",
        "Episode_Reward/motion_forward_pos",
        "Episode_Reward/motion_height",
        "Episode_Reward/motion_forward_velocity",
        "Episode_Reward/backward_velocity",
        "Episode_Reward/action_rate_l2",
        "Episode_Reward/feet_slide_vel",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/anchor_ori",
        "Episode_Termination/ee_body_pos",
    ]
    payload = {
        "runs": [str(path) for path in run_dirs],
        "warm_start_iteration": 79997,
        "smoothing_window": window,
        "metrics": {tag: metric_summary(data, tag, window) for tag in tags},
    }
    (output_dir / "forward_updown_training_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_runs(args.runs)
    plot_overview(data, args.output_dir, args.smooth)
    plot_components(data, args.output_dir, args.smooth)
    write_csv(data, args.output_dir)
    write_summary(data, args.runs, args.output_dir, args.smooth)
    print(f"Wrote training analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
