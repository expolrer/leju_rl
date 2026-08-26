#!/usr/bin/env python3
"""Render a recorded Isaac stair rollout as a simple articulated skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--contact-threshold", type=float, default=10.0)
    return parser.parse_args()


def skeleton_edges(names: list[str]) -> tuple[list[tuple[int, int]], np.ndarray]:
    index = {name: body_index for body_index, name in enumerate(names)}
    chains = [
        ["base_link", "waist_yaw_link"],
        ["base_link", *[f"leg_l{joint}_link" for joint in range(1, 7)]],
        ["base_link", *[f"leg_r{joint}_link" for joint in range(1, 7)]],
        ["waist_yaw_link", *[f"zarm_l{joint}_link" for joint in range(1, 8)]],
        ["waist_yaw_link", *[f"zarm_r{joint}_link" for joint in range(1, 8)]],
    ]
    colors = ("#263238", "#1976d2", "#d84315", "#1976d2", "#d84315")
    edges: list[tuple[int, int]] = []
    edge_colors: list[str] = []
    for chain, color in zip(chains, colors):
        present = [index[name] for name in chain if name in index]
        chain_edges = list(zip(present[:-1], present[1:]))
        edges.extend(chain_edges)
        edge_colors.extend([color] * len(chain_edges))
    return edges, np.asarray(edge_colors)


def main() -> None:
    args = parse_args()
    data = np.load(args.rollout, allow_pickle=False)
    body_names = [str(name) for name in data["all_body_names"]]
    body_pos = data["all_body_pos"]
    edges, edge_colors = skeleton_edges(body_names)
    edge_array = np.asarray(edges, dtype=np.int64)
    dt = float(data["dt"])
    frame_indices = np.arange(0, len(body_pos), args.stride)

    terrain_triangles = data["terrain_vertices"][data["terrain_faces"]]
    visible_terrain = terrain_triangles[
        (terrain_triangles[:, :, 2].max(axis=1) > -0.1)
        & (terrain_triangles[:, :, 2].min(axis=1) < 1.0)
    ]
    foot_indices = [body_names.index("leg_l6_link"), body_names.index("leg_r6_link")]
    foot_forces = data["foot_contact_force_w"]
    foot_surface_z = data["foot_surface_z"]
    checkpoint_name = Path(str(data["checkpoint"].item())).name
    task_name = str(data["task"].item())

    scene_points = np.concatenate((visible_terrain.reshape(-1, 3), body_pos.reshape(-1, 3)), axis=0)
    scene_min = scene_points.min(axis=0)
    scene_max = scene_points.max(axis=0)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#eef2f5")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#eef2f5")
    ax.set_xlim(scene_min[0] - 0.18, scene_max[0] + 0.18)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.03, max(1.85, scene_max[2] + 0.08))
    ax.set_box_aspect((3.9, 1.9, 1.9), zoom=1.25)
    ax.view_init(elev=18, azim=-112)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.91)

    terrain = Poly3DCollection(
        visible_terrain,
        facecolor="#aeb9c3",
        edgecolor="#66747f",
        linewidth=0.45,
        alpha=0.88,
    )
    ax.add_collection3d(terrain)
    segments = body_pos[0][edge_array]
    skeleton = Line3DCollection(segments, colors=edge_colors, linewidths=4.0)
    ax.add_collection3d(skeleton)
    joints = ax.scatter(
        body_pos[0, :, 0], body_pos[0, :, 1], body_pos[0, :, 2],
        s=22, c="#17212b", edgecolors="#ffffff", linewidths=0.35, depthshade=True,
    )
    contacts = ax.scatter([], [], [], s=95, c="#43a047", edgecolors="#1b5e20", linewidths=1.4)
    root_path = data["root_pos"]
    ax.plot(root_path[:, 0], root_path[:, 1], root_path[:, 2], color="#00695c", linewidth=1.2, alpha=0.4)

    fig.text(0.035, 0.945, "KUAVO S53  |  JOINT SKELETON ROLLOUT", fontsize=18, weight="bold", color="#17212b")
    fig.text(0.036, 0.907, f"{checkpoint_name}  |  {task_name}", fontsize=10.5, color="#4c5b68")
    clock = fig.text(0.82, 0.93, "", fontsize=11, family="monospace", color="#263746")
    status = fig.text(0.035, 0.065, "", fontsize=9.5, family="monospace", color="#263746")
    fig.text(
        0.035, 0.037,
        "Blue/red: left/right joint chains    Green: Isaac foot contact    Scene: recorded Isaac terrain mesh",
        fontsize=9.5, color="#52616d",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
    try:
        for output_index, frame_index in enumerate(frame_indices):
            frame_body = body_pos[frame_index]
            skeleton.set_segments(frame_body[edge_array])
            joints._offsets3d = (frame_body[:, 0], frame_body[:, 1], frame_body[:, 2])

            magnitudes = np.linalg.norm(foot_forces[frame_index], axis=-1)
            active = magnitudes > args.contact_threshold
            active_pos = frame_body[foot_indices][active].copy()
            active_pos[:, 2] = foot_surface_z[frame_index, active] + 0.008
            contacts._offsets3d = (
                active_pos[:, 0] if len(active_pos) else [],
                active_pos[:, 1] if len(active_pos) else [],
                active_pos[:, 2] if len(active_pos) else [],
            )
            clock.set_text(f"t = {frame_index * dt:05.2f} s")
            status.set_text(
                f"foot force  L {magnitudes[0]:6.1f} N @ {foot_surface_z[frame_index, 0]:.2f} m"
                f"   R {magnitudes[1]:6.1f} N @ {foot_surface_z[frame_index, 1]:.2f} m"
            )
            fig.canvas.draw()
            frame = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if output_index == 0 and args.preview is not None:
                args.preview.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(args.preview, frame)
            writer.append_data(frame)
            if output_index % 25 == 0:
                print(f"rendered {output_index + 1}/{len(frame_indices)}", flush=True)
    finally:
        writer.close()
        plt.close(fig)
    print(f"video={args.output} frames={len(frame_indices)} fps={args.fps}")


if __name__ == "__main__":
    main()
