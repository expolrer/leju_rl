#!/usr/bin/env python3
"""Render a policy rollout using the original model_29999 two-panel style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
try:
    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


STAIR_HEIGHT = 0.13
STAIR_TREAD = 0.28
FIRST_RISER_X = 0.14
PLATFORM_LENGTH = 1.0
FOOT_LENGTH = 0.24
FOOT_WIDTH = 0.10
FOOT_SOLE_OFFSET = 0.0645
FOOT_COLORS = {"leg_l6_link": "#d62828", "leg_r6_link": "#0077b6"}
FOOT_LABELS = {"leg_l6_link": "L", "leg_r6_link": "R"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-frame", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def skeleton_edges(names: list[str]) -> list[tuple[int, int]]:
    index = {name: body_index for body_index, name in enumerate(names)}
    chains = [
        ["base_link", "waist_yaw_link"],
        ["base_link", "leg_l2_link", "leg_l4_link", "leg_l6_link"],
        ["base_link", "leg_r2_link", "leg_r4_link", "leg_r6_link"],
        ["waist_yaw_link", "zarm_l2_link", "zarm_l4_link", "zarm_l7_link"],
        ["waist_yaw_link", "zarm_r2_link", "zarm_r4_link", "zarm_r7_link"],
    ]
    edges: list[tuple[int, int]] = []
    for chain in chains:
        present = [index[name] for name in chain if name in index]
        edges.extend(zip(present[:-1], present[1:]))
    return edges


def draw_box(ax, x: float, length: float, height: float) -> None:
    ax.bar3d(
        x,
        -0.5,
        0.0,
        length,
        1.0,
        height,
        color="#d9dde3",
        edgecolor="#697386",
        linewidth=0.5,
        alpha=0.72,
        shade=True,
    )


def draw_stairs_3d(ax) -> None:
    for level in range(1, 5):
        start_x = FIRST_RISER_X + (level - 1) * STAIR_TREAD
        length = STAIR_TREAD if level < 4 else STAIR_TREAD + PLATFORM_LENGTH
        draw_box(ax, start_x, length, level * STAIR_HEIGHT)
    descent_start = FIRST_RISER_X + 4 * STAIR_TREAD + PLATFORM_LENGTH
    for index, level in enumerate((3, 2, 1)):
        draw_box(ax, descent_start + index * STAIR_TREAD, STAIR_TREAD, level * STAIR_HEIGHT)


def stair_profile() -> tuple[np.ndarray, np.ndarray]:
    x = [-0.4, FIRST_RISER_X]
    z = [0.0, 0.0]
    for level in range(1, 5):
        start = FIRST_RISER_X + (level - 1) * STAIR_TREAD
        end = start + (STAIR_TREAD if level < 4 else STAIR_TREAD + PLATFORM_LENGTH)
        x.extend([start, end])
        z.extend([level * STAIR_HEIGHT, level * STAIR_HEIGHT])
    descent_start = FIRST_RISER_X + 4 * STAIR_TREAD + PLATFORM_LENGTH
    for index, level in enumerate((3, 2, 1)):
        start = descent_start + index * STAIR_TREAD
        x.extend([start, start + STAIR_TREAD])
        z.extend([level * STAIR_HEIGHT, level * STAIR_HEIGHT])
    ground_start = descent_start + 3 * STAIR_TREAD
    x.extend([ground_start, 3.55])
    z.extend([0.0, 0.0])
    return np.asarray(x), np.asarray(z)


def plot_skeleton_3d(ax, points, edges, color, alpha, linestyle, label) -> None:
    for start, end in edges:
        ax.plot(
            points[[start, end], 0],
            points[[start, end], 1],
            points[[start, end], 2],
            color=color,
            linewidth=3.0 if alpha > 0.8 else 1.5,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=22, alpha=alpha, label=label)


def plot_skeleton_side(ax, points, edges, color, alpha, linestyle, label) -> None:
    for start, end in edges:
        ax.plot(
            points[[start, end], 0],
            points[[start, end], 2],
            color=color,
            linewidth=3.0 if alpha > 0.8 else 1.5,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.scatter(points[:, 0], points[:, 2], color=color, s=22, alpha=alpha, label=label)


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1.0e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def sole_corners(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    local = np.asarray(
        [
            [-0.5 * FOOT_LENGTH, -0.5 * FOOT_WIDTH, -FOOT_SOLE_OFFSET],
            [0.5 * FOOT_LENGTH, -0.5 * FOOT_WIDTH, -FOOT_SOLE_OFFSET],
            [0.5 * FOOT_LENGTH, 0.5 * FOOT_WIDTH, -FOOT_SOLE_OFFSET],
            [-0.5 * FOOT_LENGTH, 0.5 * FOOT_WIDTH, -FOOT_SOLE_OFFSET],
        ]
    )
    return position[None] + local @ quaternion_matrix(quaternion).T


def plot_foot_soles_3d(
    ax,
    points: np.ndarray,
    quaternions: np.ndarray,
    names: list[str],
    reference: bool = False,
) -> None:
    index = {name: body_index for body_index, name in enumerate(names)}
    for foot_name in FOOT_LABELS:
        if foot_name not in index:
            continue
        foot_index = index[foot_name]
        corners = sole_corners(points[foot_index], quaternions[foot_index])
        color = "#6b7280" if reference else FOOT_COLORS[foot_name]
        polygon = Poly3DCollection(
            [corners],
            facecolors=color,
            edgecolors=color,
            linewidths=1.0 if reference else 2.0,
            alpha=0.12 if reference else 0.62,
        )
        ax.add_collection3d(polygon)
        if not reference:
            center = corners.mean(axis=0)
            ax.text(center[0], center[1], center[2] + 0.035, FOOT_LABELS[foot_name], color=color, weight="bold")


def plot_foot_soles_side(
    ax,
    points: np.ndarray,
    quaternions: np.ndarray,
    names: list[str],
    reference: bool = False,
) -> None:
    index = {name: body_index for body_index, name in enumerate(names)}
    for foot_name in FOOT_LABELS:
        if foot_name not in index:
            continue
        foot_index = index[foot_name]
        corners = sole_corners(points[foot_index], quaternions[foot_index])
        color = "#6b7280" if reference else FOOT_COLORS[foot_name]
        polygon = Polygon(
            corners[:, [0, 2]],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.9 if reference else 2.0,
            alpha=0.10 if reference else 0.62,
        )
        ax.add_patch(polygon)
        if not reference:
            center = corners[:, [0, 2]].mean(axis=0)
            label_x_offset = -0.055 if foot_name == "leg_l6_link" else 0.055
            ax.text(
                center[0] + label_x_offset,
                center[1] + 0.045,
                FOOT_LABELS[foot_name],
                color=color,
                fontsize=11,
                fontweight="bold",
                ha="center",
            )


def load_body_quaternions(data, key: str, names: list[str]) -> np.ndarray:
    if key in data.files:
        return data[key]
    if key == "actual_body_quat" and "all_body_quat" in data.files:
        all_names = [str(name) for name in data["all_body_names"]]
        indexes = [all_names.index(name) for name in names]
        return data["all_body_quat"][:, indexes]
    identity = np.zeros((len(data["actual_body_pos"]), len(names), 4), dtype=np.float32)
    identity[:, :, 0] = 1.0
    return identity


def configure_axes(ax3d, ax2d) -> None:
    draw_stairs_3d(ax3d)
    ax3d.set_xlim(-0.45, 3.55)
    ax3d.set_ylim(-0.8, 0.8)
    ax3d.set_zlim(0.0, 2.05)
    ax3d.set_box_aspect((4.0, 1.6, 2.05))
    ax3d.view_init(elev=18, azim=-62)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.grid(True, alpha=0.2)

    stair_x, stair_z = stair_profile()
    ax2d.fill_between(stair_x, stair_z, color="#d9dde3", step="pre", alpha=0.9)
    ax2d.plot(stair_x, stair_z, color="#697386", linewidth=1.3, drawstyle="steps-pre")
    ax2d.set_xlim(-0.45, 3.55)
    ax2d.set_ylim(0.0, 2.05)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x (m)")
    ax2d.set_ylabel("z (m)")
    ax2d.grid(True, alpha=0.2)


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=False)
    actual = data["actual_body_pos"]
    reference = data["reference_body_pos"]
    names = [str(name) for name in data["body_names"]]
    actual_quat = load_body_quaternions(data, "actual_body_quat", names)
    reference_quat = load_body_quaternions(data, "reference_body_quat", names)
    edges = skeleton_edges(names)
    dt = float(data["dt"])
    stride = max(1, int(round(1.0 / (args.fps * dt))))
    frame_ids = np.arange(0, actual.shape[0], stride)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(1, 2, left=0.035, right=0.98, bottom=0.09, top=0.88, wspace=0.20)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid[0, 1])
    title = fig.suptitle("", y=0.965)

    def update(animation_index: int):
        frame = int(frame_ids[animation_index])
        ax3d.clear()
        ax2d.clear()
        configure_axes(ax3d, ax2d)
        plot_skeleton_3d(ax3d, reference[frame], edges, "#6b7280", 0.45, "--", "Reference")
        plot_skeleton_3d(ax3d, actual[frame], edges, "#087f8c", 1.0, "-", "Policy")
        plot_foot_soles_3d(ax3d, reference[frame], reference_quat[frame], names, reference=True)
        plot_foot_soles_3d(ax3d, actual[frame], actual_quat[frame], names)
        plot_skeleton_side(ax2d, reference[frame], edges, "#6b7280", 0.45, "--", "Reference")
        plot_skeleton_side(ax2d, actual[frame], edges, "#087f8c", 1.0, "-", "Policy")
        plot_foot_soles_side(ax2d, reference[frame], reference_quat[frame], names, reference=True)
        plot_foot_soles_side(ax2d, actual[frame], actual_quat[frame], names)
        ax3d.set_title("3D state replay")
        ax2d.set_title("Side view against stair geometry")
        ax2d.legend(loc="upper left")
        title.set_text(
            f"Kuavo S53 model rollout | t={frame * dt:5.2f}s | motion frame={int(data['motion_frame'][frame])}"
        )
        return []

    if args.preview is not None:
        preview_index = int(np.clip(args.preview_frame // stride, 0, len(frame_ids) - 1))
        update(preview_index)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=130)
        if args.preview_only:
            plt.close(fig)
            print(f"preview={args.preview}")
            return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p"])
    animation.save(args.output, writer=writer, dpi=130)
    plt.close(fig)
    print(f"video={args.output} frames={len(frame_ids)} fps={args.fps}")


if __name__ == "__main__":
    main()
