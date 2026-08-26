#!/usr/bin/env python3
"""Render an Isaac physics rollout with the Kuavo URDF visual meshes."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation


@dataclass
class VisualMesh:
    body_index: int
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    color: np.ndarray


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a dense STL with vertex clustering while keeping a filled surface."""
    if len(faces) <= max_faces:
        return vertices, faces
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    best_vertices, best_faces = vertices, faces
    for fraction in (0.004, 0.006, 0.009, 0.013, 0.019, 0.027, 0.038, 0.054, 0.075):
        pitch = max(extent * fraction, 1.0e-5)
        cells = np.round(vertices / pitch).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        clustered = np.zeros((count, 3), dtype=np.float64)
        np.add.at(clustered, inverse, vertices)
        clustered /= np.bincount(inverse)[:, None]
        remapped = inverse[faces]
        valid = (
            (remapped[:, 0] != remapped[:, 1])
            & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2])
        )
        remapped = remapped[valid]
        if len(remapped) == 0:
            continue
        _, unique_indices = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
        remapped = remapped[np.sort(unique_indices)]
        best_vertices, best_faces = clustered, remapped
        if len(remapped) <= max_faces:
            break
    return best_vertices, best_faces


def parse_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, :3] = Rotation.from_euler("xyz", parse_vector(element.get("rpy"), (0, 0, 0))).as_matrix()
    transform[:3, 3] = parse_vector(element.get("xyz"), (0, 0, 0))
    return transform


def load_visual_meshes(urdf_path: Path, body_names: list[str], max_faces: int) -> list[VisualMesh]:
    root = ET.parse(urdf_path).getroot()
    body_index = {name: index for index, name in enumerate(body_names)}
    joints: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        joints[child.get("link")] = (parent.get("link"), origin_transform(joint.find("origin")))

    ancestor_cache: dict[str, tuple[str, np.ndarray]] = {}

    def ancestor(link_name: str) -> tuple[str, np.ndarray]:
        if link_name in ancestor_cache:
            return ancestor_cache[link_name]
        if link_name in body_index:
            result = (link_name, np.eye(4))
        else:
            parent_name, parent_to_link = joints[link_name]
            ancestor_name, ancestor_to_parent = ancestor(parent_name)
            result = (ancestor_name, ancestor_to_parent @ parent_to_link)
        ancestor_cache[link_name] = result
        return result

    visuals: list[VisualMesh] = []
    palette = {
        "base": np.array((0.20, 0.23, 0.27)),
        "left": np.array((0.16, 0.47, 0.72)),
        "right": np.array((0.88, 0.32, 0.27)),
        "waist": np.array((0.83, 0.85, 0.88)),
        "head": np.array((0.24, 0.27, 0.31)),
    }
    for link in root.findall("link"):
        link_name = link.get("name")
        try:
            ancestor_name, ancestor_to_link = ancestor(link_name)
        except KeyError:
            continue
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                continue
            mesh_path = (urdf_path.parent / mesh_element.get("filename")).resolve()
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            if not isinstance(loaded, trimesh.Trimesh):
                continue
            scale = parse_vector(mesh_element.get("scale"), (1, 1, 1))
            local = origin_transform(visual.find("origin"))
            local[:3, :3] = local[:3, :3] @ np.diag(scale)
            transform = ancestor_to_link @ local
            vertices = trimesh.transform_points(np.asarray(loaded.vertices), transform)
            faces = np.asarray(loaded.faces)
            vertices, faces = simplify_mesh(vertices, faces, max_faces)
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(norms, 1.0e-9)
            if link_name.startswith("leg_l") or link_name.startswith("zarm_l"):
                color = palette["left"]
            elif link_name.startswith("leg_r") or link_name.startswith("zarm_r"):
                color = palette["right"]
            elif "head" in link_name or "camera" in link_name:
                color = palette["head"]
            elif "waist" in link_name:
                color = palette["waist"]
            else:
                color = palette["base"]
            visuals.append(
                VisualMesh(body_index[ancestor_name], vertices, faces, normals, color)
            )
    return visuals


def box_faces(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size) * 0.5
    vertices = np.array(
        [
            [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
        ]
    )
    return vertices[
        np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
    ]


def legacy_stair_geometry() -> list[np.ndarray]:
    boxes: list[np.ndarray] = []
    for level in range(1, 5):
        start_x = 0.14 + (level - 1) * 0.28
        length = 0.28 if level < 4 else 1.28
        height = level * 0.13
        boxes.append(box_faces((start_x + 0.5 * length, 0.0, 0.5 * height), (length, 1.5, height)))
    descent_start = 0.14 + 4 * 0.28 + 1.0
    for index, level in enumerate((3, 2, 1)):
        start_x = descent_start + index * 0.28
        height = level * 0.13
        boxes.append(box_faces((start_x + 0.14, 0.0, 0.5 * height), (0.28, 1.5, height)))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_faces", type=int, default=700)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--contact_threshold", type=float, default=10.0)
    args = parser.parse_args()

    data = np.load(args.rollout, allow_pickle=False)
    body_names = [str(item) for item in data["all_body_names"]]
    body_pos = data["all_body_pos"]
    body_quat = data["root_quat"]
    if "all_body_quat" in data:
        body_quat = data["all_body_quat"]
    else:
        raise KeyError("rollout is missing all_body_quat")
    dt = float(data["dt"])
    visuals = load_visual_meshes(args.urdf, body_names, args.max_faces)
    checkpoint_name = Path(str(data["checkpoint"].item())).name
    task_name = str(data["task"].item())

    if "terrain_vertices" in data and "terrain_faces" in data:
        terrain_triangles = data["terrain_vertices"][data["terrain_faces"]]
        terrain_source = "recorded Isaac terrain mesh"
    else:
        terrain_triangles = np.concatenate(legacy_stair_geometry(), axis=0)
        terrain_source = "legacy reconstructed terrain"

    left_index = body_names.index("leg_l6_link")
    right_index = body_names.index("leg_r6_link")
    foot_indices = [left_index, right_index]
    foot_forces = data["foot_contact_force_w"] if "foot_contact_force_w" in data else None
    foot_surface_z = data["foot_surface_z"] if "foot_surface_z" in data else None
    reference_root = data["reference_body_pos"][:, 0] if "reference_body_pos" in data else None
    actual_root = data["root_pos"]

    frame_indices = np.arange(0, len(body_pos), args.stride)
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#e8edf2")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#e8edf2")
    visible_terrain = terrain_triangles[
        (terrain_triangles[:, :, 2].max(axis=1) > -0.1)
        & (terrain_triangles[:, :, 2].min(axis=1) < 1.0)
    ]
    scene_points = np.concatenate((visible_terrain.reshape(-1, 3), body_pos.reshape(-1, 3)), axis=0)
    scene_min = scene_points.min(axis=0)
    scene_max = scene_points.max(axis=0)
    ax.set_xlim(scene_min[0] - 0.15, scene_max[0] + 0.15)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.03, max(2.05, scene_max[2] + 0.1))
    ax.set_box_aspect((3.87, 1.9, 2.08), zoom=1.25)
    ax.view_init(elev=17, azim=-112)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.92)

    ground = Poly3DCollection(
        [np.array([[-0.5, -1.25, -0.015], [3.7, -1.25, -0.015], [3.7, 1.25, -0.015], [-0.5, 1.25, -0.015]])],
        facecolor="#c7d0d8",
        edgecolor="none",
    )
    ax.add_collection3d(ground)
    stairs = Poly3DCollection(
        visible_terrain,
        facecolor="#9aa8b5",
        edgecolor="#667480",
        linewidth=0.45,
    )
    ax.add_collection3d(stairs)

    robot_collection = Poly3DCollection([], edgecolor="none", linewidth=0.0)
    ax.add_collection3d(robot_collection)
    contact_markers = ax.scatter([], [], [], s=85, facecolors="#43a047", edgecolors="#1b5e20", linewidths=1.5)
    contact_lines = Line3DCollection(
        [np.zeros((2, 3))], colors="#2e7d32", linewidths=2.0, alpha=0.85
    )
    ax.add_collection3d(contact_lines)
    contact_lines.set_segments([])
    ax.plot(actual_root[:, 0], actual_root[:, 1], actual_root[:, 2], color="#1565c0", linewidth=1.2, alpha=0.55)
    if reference_root is not None:
        ax.plot(
            reference_root[:, 0], reference_root[:, 1], reference_root[:, 2],
            color="#f9a825", linewidth=1.1, linestyle="--", alpha=0.65,
        )

    title = fig.text(0.035, 0.94, "KUAVO S53  |  ISAAC PHYSICS ROLLOUT", fontsize=18, weight="bold", color="#17212b")
    subtitle = fig.text(0.036, 0.902, f"{checkpoint_name}  |  {task_name}", fontsize=10.5, color="#4c5b68")
    clock = fig.text(0.82, 0.925, "", fontsize=11, family="monospace", color="#263746")
    contact_text = fig.text(0.035, 0.084, "", fontsize=9.5, family="monospace", color="#263746")
    fig.text(
        0.035, 0.055,
        f"Blue/red: robot sides    Green rings: Isaac foot contact > {args.contact_threshold:g} N    Scene: {terrain_source}",
        fontsize=9.5, color="#52616d",
    )
    del title, subtitle

    light = np.array((-0.35, -0.45, 0.82))
    light /= np.linalg.norm(light)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
    try:
        for output_index, frame_index in enumerate(frame_indices):
            frame_triangles = []
            frame_colors = []
            for visual in visuals:
                q = body_quat[frame_index, visual.body_index]
                rotation = Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix()
                position = body_pos[frame_index, visual.body_index]
                transformed = visual.vertices @ rotation.T + position
                frame_triangles.append(transformed[visual.faces])
                world_normals = visual.normals @ rotation.T
                intensity = 0.62 + 0.38 * np.clip(world_normals @ light, 0.0, 1.0)
                colors = np.ones((len(intensity), 4))
                colors[:, :3] = np.clip(visual.color[None, :] * intensity[:, None], 0.0, 1.0)
                frame_colors.append(colors)
            robot_collection.set_verts(np.concatenate(frame_triangles, axis=0))
            robot_collection.set_facecolor(np.concatenate(frame_colors, axis=0))
            foot_pos = body_pos[frame_index, foot_indices]
            if foot_forces is not None:
                magnitudes = np.linalg.norm(foot_forces[frame_index], axis=-1)
                active = magnitudes > args.contact_threshold
                active_pos = foot_pos[active].copy()
                segments = []
                if foot_surface_z is not None:
                    active_pos[:, 2] = foot_surface_z[frame_index, active] + 0.006
                    for foot_index in np.flatnonzero(active):
                        surface = foot_pos[foot_index].copy()
                        surface[2] = foot_surface_z[frame_index, foot_index]
                        segments.append(np.stack((surface, foot_pos[foot_index])))
                contact_markers._offsets3d = (
                    active_pos[:, 0] if len(active_pos) else [],
                    active_pos[:, 1] if len(active_pos) else [],
                    active_pos[:, 2] if len(active_pos) else [],
                )
                contact_lines.set_segments(segments)
                if foot_surface_z is not None:
                    contact_text.set_text(
                        f"Isaac contact  L {magnitudes[0]:6.1f} N @ z={foot_surface_z[frame_index, 0]:.2f} m"
                        f"   R {magnitudes[1]:6.1f} N @ z={foot_surface_z[frame_index, 1]:.2f} m"
                    )
                else:
                    contact_text.set_text(f"foot force  L {magnitudes[0]:6.1f} N   R {magnitudes[1]:6.1f} N")
            else:
                contact_markers._offsets3d = ([], [], [])
                contact_lines.set_segments([])
                contact_text.set_text("foot force  unavailable in legacy rollout")
            clock.set_text(f"t = {frame_index * dt:05.2f} s")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = np.ascontiguousarray(rgba[..., :3])
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
