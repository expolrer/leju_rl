#!/usr/bin/env python3
"""Render an Isaac policy rollout as a display-independent stair animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


STAIR_HEIGHT = 0.13
STAIR_TREAD = 0.28
FIRST_RISER_X = 0.14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def skeleton_edges(names: list[str]) -> list[tuple[int, int]]:
    index = {name: i for i, name in enumerate(names)}
    chains = [
        ["base_link", "waist_yaw_link"],
        ["base_link", "leg_l2_link", "leg_l4_link", "leg_l6_link"],
        ["base_link", "leg_r2_link", "leg_r4_link", "leg_r6_link"],
        ["waist_yaw_link", "zarm_l2_link", "zarm_l4_link", "zarm_l7_link"],
        ["waist_yaw_link", "zarm_r2_link", "zarm_r4_link", "zarm_r7_link"],
    ]
    edges = []
    for chain in chains:
        present = [index[name] for name in chain if name in index]
        edges.extend(zip(present[:-1], present[1:]))
    return edges


def draw_stairs_3d(ax) -> None:
    for level in range(1, 5):
        x = FIRST_RISER_X + (level - 1) * STAIR_TREAD
        length = STAIR_TREAD if level < 4 else STAIR_TREAD + 1.0
        ax.bar3d(
            x,
            -0.5,
            0.0,
            length,
            1.0,
            level * STAIR_HEIGHT,
            color="#d9dde3",
            edgecolor="#697386",
            linewidth=0.5,
            alpha=0.72,
            shade=True,
        )


def stair_profile() -> tuple[np.ndarray, np.ndarray]:
    x = [-0.4, FIRST_RISER_X]
    z = [0.0, 0.0]
    for level in range(1, 5):
        start = FIRST_RISER_X + (level - 1) * STAIR_TREAD
        end = start + (STAIR_TREAD if level < 4 else STAIR_TREAD + 1.0)
        x.extend([start, end])
        z.extend([level * STAIR_HEIGHT, level * STAIR_HEIGHT])
    return np.asarray(x), np.asarray(z)


def plot_skeleton_3d(ax, points, edges, color, alpha, linestyle, label):
    for a, b in edges:
        ax.plot(
            points[[a, b], 0],
            points[[a, b], 1],
            points[[a, b], 2],
            color=color,
            linewidth=3.0 if alpha > 0.8 else 1.5,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=22, alpha=alpha, label=label)


def plot_skeleton_side(ax, points, edges, color, alpha, linestyle, label):
    for a, b in edges:
        ax.plot(
            points[[a, b], 0],
            points[[a, b], 2],
            color=color,
            linewidth=3.0 if alpha > 0.8 else 1.5,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.scatter(points[:, 0], points[:, 2], color=color, s=22, alpha=alpha, label=label)


def configure_axes(ax3d, ax2d) -> None:
    draw_stairs_3d(ax3d)
    ax3d.set_xlim(-0.45, 1.75)
    ax3d.set_ylim(-0.8, 0.8)
    ax3d.set_zlim(0.0, 1.8)
    ax3d.set_box_aspect((2.2, 1.6, 1.8))
    ax3d.view_init(elev=18, azim=-62)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.grid(True, alpha=0.2)

    stair_x, stair_z = stair_profile()
    ax2d.fill_between(stair_x, stair_z, color="#d9dde3", step="pre", alpha=0.9)
    ax2d.plot(stair_x, stair_z, color="#697386", linewidth=1.3, drawstyle="steps-pre")
    ax2d.set_xlim(-0.45, 1.75)
    ax2d.set_ylim(0.0, 1.8)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x (m)")
    ax2d.set_ylabel("z (m)")
    ax2d.grid(True, alpha=0.2)


def render_animation(data, output_dir: Path, name: str, fps: int) -> None:
    actual = data["actual_body_pos"]
    reference = data["reference_body_pos"]
    names = [str(name) for name in data["body_names"]]
    edges = skeleton_edges(names)
    dt = float(data["dt"])
    stride = max(1, int(round(1.0 / (fps * dt))))
    frame_ids = np.arange(0, actual.shape[0], stride)

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    def update(animation_index: int):
        frame = int(frame_ids[animation_index])
        ax3d.clear()
        ax2d.clear()
        configure_axes(ax3d, ax2d)
        plot_skeleton_3d(ax3d, reference[frame], edges, "#6b7280", 0.45, "--", "Reference")
        plot_skeleton_3d(ax3d, actual[frame], edges, "#087f8c", 1.0, "-", "Policy")
        plot_skeleton_side(ax2d, reference[frame], edges, "#6b7280", 0.45, "--", "Reference")
        plot_skeleton_side(ax2d, actual[frame], edges, "#087f8c", 1.0, "-", "Policy")
        ax3d.set_title("3D state replay")
        ax2d.set_title("Side view against stair geometry")
        ax2d.legend(loc="upper left")
        fig.suptitle(
            f"Kuavo S53 model rollout | t={frame * dt:5.2f}s | motion frame={int(data['motion_frame'][frame])}"
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p"])
    animation.save(output_dir / f"{name}.mp4", writer=writer, dpi=130)
    plt.close(fig)


def render_metrics(data, output_dir: Path, name: str) -> None:
    dt = float(data["dt"])
    time = np.arange(data["actual_body_pos"].shape[0]) * dt
    names = [str(name) for name in data["body_names"]]
    left = names.index("leg_l6_link")
    right = names.index("leg_r6_link")
    actual = data["actual_body_pos"]
    reference = data["reference_body_pos"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(time, data["root_pos"][:, 0], label="Root x", color="#1976d2")
    axes[0, 0].plot(time, data["root_pos"][:, 2], label="Root z", color="#d32f2f")
    axes[0, 0].set_title("Root translation")
    axes[0, 0].set_ylabel("Position (m)")

    axes[0, 1].plot(time, actual[:, left, 2], label="Left foot", color="#00897b")
    axes[0, 1].plot(time, actual[:, right, 2], label="Right foot", color="#7b1fa2")
    axes[0, 1].plot(time, reference[:, left, 2], "--", alpha=0.55, color="#00897b", label="Left reference")
    axes[0, 1].plot(time, reference[:, right, 2], "--", alpha=0.55, color="#7b1fa2", label="Right reference")
    for level in range(1, 5):
        axes[0, 1].axhline(level * STAIR_HEIGHT, color="#777777", linewidth=0.6, alpha=0.4)
    axes[0, 1].set_title("Foot clearance and landing height")
    axes[0, 1].set_ylabel("Foot z (m)")

    anchor_error = data["anchor_error"].astype(float).copy()
    body_error = data["body_error"].astype(float).copy()
    anchor_error[:2] = np.nan
    body_error[:2] = np.nan
    axes[1, 0].plot(time, anchor_error, label="Root error", color="#d32f2f")
    axes[1, 0].plot(time, body_error, label="Mean body error", color="#7b1fa2")
    axes[1, 0].set_title("Tracking errors during rollout")
    axes[1, 0].set_ylabel("Error (m)")

    action_time = np.arange(data["actions"].shape[0]) * dt
    axes[1, 1].plot(action_time, np.linalg.norm(data["actions"], axis=1), color="#f57c00")
    done_indices = np.flatnonzero(data["dones"])
    done_indices = done_indices[done_indices > 0]
    for index in done_indices:
        axes[1, 1].axvline(index * dt, color="#c62828", linewidth=1.0, alpha=0.7)
    axes[1, 1].set_title(f"Action norm; resets={len(done_indices)}")
    axes[1, 1].set_ylabel("L2 norm")

    for ax in axes.reshape(-1):
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.22)
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")
    fig.suptitle("Kuavo S53 policy rollout diagnostics", fontsize=16)
    fig.savefig(output_dir / f"{name}_metrics.png", dpi=180)
    plt.close(fig)


def render_keyframes(data, output_dir: Path, name: str) -> None:
    actual = data["actual_body_pos"]
    reference = data["reference_body_pos"]
    names = [str(name) for name in data["body_names"]]
    edges = skeleton_edges(names)
    dt = float(data["dt"])
    frame_ids = np.linspace(0, actual.shape[0] - 1, 6, dtype=int)
    stair_x, stair_z = stair_profile()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, frame in zip(axes.reshape(-1), frame_ids):
        ax.fill_between(stair_x, stair_z, color="#d9dde3", step="pre", alpha=0.9)
        ax.plot(stair_x, stair_z, color="#697386", linewidth=1.2, drawstyle="steps-pre")
        plot_skeleton_side(ax, reference[frame], edges, "#6b7280", 0.45, "--", "Reference")
        plot_skeleton_side(ax, actual[frame], edges, "#087f8c", 1.0, "-", "Policy")
        ax.set_xlim(-0.45, 1.75)
        ax.set_ylim(0.0, 1.8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"t={frame * dt:.2f}s, ref={int(data['motion_frame'][frame])}")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Kuavo S53 stair-policy keyframes", fontsize=16)
    fig.savefig(output_dir / f"{name}_keyframes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input, allow_pickle=False)
    if not args.skip_video:
        render_animation(data, args.output_dir, args.name, args.fps)
    render_metrics(data, args.output_dir, args.name)
    render_keyframes(data, args.output_dir, args.name)
    print(f"Rendered rollout artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
