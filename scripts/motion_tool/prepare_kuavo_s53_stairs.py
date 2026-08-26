#!/usr/bin/env python3
"""Prepare KuavoS53 stair-tracking CSV files from TongVerse S45 captures.

The TongVerse challenge robot exposes 28 joints (including two head joints),
while LejuLab's KuavoS53 uses 27 joints (including one waist yaw joint).  This
tool performs that mapping explicitly, canonicalizes each trajectory to face
positive X, validates joint limits, and chooses a medoid demonstration without
averaging away swing-foot clearance.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_FILE = "processed/stairs_full_s45_50fps_raw.npz"

S53_JOINT_NAMES = (
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
)


@dataclass(frozen=True)
class Episode:
    name: str
    source_path: Path
    fps: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    stage: np.ndarray


def _yaw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Encountered a zero-norm root quaternion")
    quat /= norms
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat


def _circular_mean(values: np.ndarray) -> float:
    return math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean()))


def canonicalize_root(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    target_root_height: float,
    heading_window: int = 25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Center XY, rotate initial heading to +X, and set initial root height."""
    pos = np.asarray(root_pos, dtype=np.float64).copy()
    quat = _normalize_quaternions(root_quat)
    window = max(1, min(heading_window, len(quat)))
    initial_yaw = _circular_mean(_yaw_from_wxyz(quat[:window]))
    rotation_yaw = -initial_yaw

    pos[:, :2] -= pos[0, :2]
    cosine = math.cos(rotation_yaw)
    sine = math.sin(rotation_yaw)
    xy = pos[:, :2].copy()
    pos[:, 0] = cosine * xy[:, 0] - sine * xy[:, 1]
    pos[:, 1] = sine * xy[:, 0] + cosine * xy[:, 1]
    pos[:, 2] += target_root_height - pos[0, 2]

    yaw_quat = np.array(
        [math.cos(0.5 * rotation_yaw), 0.0, 0.0, math.sin(0.5 * rotation_yaw)],
        dtype=np.float64,
    )
    quat = _quat_mul_wxyz(np.broadcast_to(yaw_quat, quat.shape), quat)
    quat = _normalize_quaternions(quat)
    return pos, quat, initial_yaw


def retarget_joint_positions(
    source_joint_pos: np.ndarray,
    source_joint_names: Iterable[str],
) -> np.ndarray:
    source_names = tuple(str(name) for name in source_joint_names)
    if len(source_names) != source_joint_pos.shape[1]:
        raise ValueError("Source joint-name count does not match joint-position columns")
    if len(set(source_names)) != len(source_names):
        raise ValueError("Source joint names are not unique")

    source_index = {name: index for index, name in enumerate(source_names)}
    required = set(S53_JOINT_NAMES) - {"waist_yaw_joint"}
    missing = sorted(required - set(source_index))
    if missing:
        raise ValueError(f"Source motion is missing S53 joints: {missing}")

    result = np.zeros((source_joint_pos.shape[0], len(S53_JOINT_NAMES)), dtype=np.float64)
    for target_index, name in enumerate(S53_JOINT_NAMES):
        if name != "waist_yaw_joint":
            result[:, target_index] = source_joint_pos[:, source_index[name]]
    return result


def load_soft_joint_limits(urdf_path: Path, factor: float) -> dict[str, tuple[float, float]]:
    if not 0.0 < factor <= 1.0:
        raise ValueError("soft-limit factor must be in (0, 1]")
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "fixed" or limit is None:
            continue
        lower_text = limit.get("lower")
        upper_text = limit.get("upper")
        if lower_text is None or upper_text is None:
            continue
        lower = float(lower_text)
        upper = float(upper_text)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * factor * (upper - lower)
        limits[str(joint.get("name"))] = (midpoint - half_range, midpoint + half_range)
    return limits


def clip_to_joint_limits(
    joint_pos: np.ndarray,
    limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    clipped = np.asarray(joint_pos, dtype=np.float64).copy()
    report: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(S53_JOINT_NAMES):
        if name not in limits:
            raise ValueError(f"S53 URDF does not define a finite limit for {name}")
        lower, upper = limits[name]
        before = clipped[:, index].copy()
        clipped[:, index] = np.clip(before, lower, upper)
        changed = np.abs(clipped[:, index] - before) > 1.0e-10
        if np.any(changed):
            report[name] = {
                "samples": int(changed.sum()),
                "max_correction_rad": float(np.max(np.abs(clipped[:, index] - before))),
                "soft_lower_rad": lower,
                "soft_upper_rad": upper,
            }
    return clipped, report


def _load_episode(path: Path, target_root_height: float) -> tuple[Episode, tuple[str, ...], float]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "fps",
            "root_pos_w",
            "root_quat_wxyz",
            "joint_pos",
            "joint_names",
            "solver_stage_index",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        fps = float(np.asarray(data["fps"]).reshape(()))
        root_pos, root_quat, initial_yaw = canonicalize_root(
            data["root_pos_w"], data["root_quat_wxyz"], target_root_height
        )
        joint_names = tuple(str(value) for value in data["joint_names"])
        joint_pos = retarget_joint_positions(data["joint_pos"], joint_names)
        stage = np.asarray(data["solver_stage_index"], dtype=np.int64)

    arrays = (root_pos, root_quat, joint_pos, stage)
    frame_count = len(root_pos)
    if any(len(value) != frame_count for value in arrays):
        raise ValueError(f"{path} contains inconsistent frame counts")
    if not all(np.isfinite(value).all() for value in arrays[:3]):
        raise ValueError(f"{path} contains non-finite motion values")
    episode = Episode(path.parents[1].name, path, fps, root_pos, root_quat, joint_pos, stage)
    return episode, joint_names, initial_yaw


def choose_medoid(episodes: list[Episode]) -> tuple[int, np.ndarray]:
    """Choose the smooth real episode closest to all other demonstrations."""
    if not episodes:
        raise ValueError("At least one episode is required")
    features = []
    for episode in episodes:
        root = episode.root_pos - episode.root_pos[:1]
        root_scale = np.array([0.5, 0.2, 0.2], dtype=np.float64)
        feature = np.concatenate((root / root_scale, episode.joint_pos[:, :12]), axis=1)
        features.append(feature.reshape(-1))
    matrix = np.stack(features)
    pairwise = np.sqrt(np.mean((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2))
    return int(np.argmin(pairwise.mean(axis=1))), pairwise


def _write_csv(path: Path, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
    values = np.concatenate((root_pos, root_quat, joint_pos), axis=1)
    if values.shape[1] != 34:
        raise AssertionError(f"Expected a 34-column S53 CSV, got {values.shape[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.10f")


def _slice_with_hold(values: np.ndarray, start: int, end_inclusive: int, hold_frames: int) -> np.ndarray:
    result = values[start : end_inclusive + 1].copy()
    if hold_frames > 0:
        result = np.concatenate((result, np.repeat(result[-1:], hold_frames, axis=0)), axis=0)
    return result


def prepare(args: argparse.Namespace) -> dict[str, object]:
    source_paths = sorted(args.collection_root.glob(f"trajectory_*/{SOURCE_FILE}"))
    if len(source_paths) < args.min_episodes:
        raise ValueError(
            f"Found {len(source_paths)} valid processed trajectories; expected at least {args.min_episodes}"
        )
    limits = load_soft_joint_limits(args.s53_urdf, args.soft_limit_factor)

    episodes: list[Episode] = []
    source_orders: list[tuple[str, ...]] = []
    initial_yaws: list[float] = []
    per_episode_limit_reports: dict[str, dict[str, dict[str, float | int]]] = {}
    for source_path in source_paths:
        episode, source_order, initial_yaw = _load_episode(source_path, args.target_root_height)
        clipped, limit_report = clip_to_joint_limits(episode.joint_pos, limits)
        episode = Episode(
            episode.name,
            episode.source_path,
            episode.fps,
            episode.root_pos,
            episode.root_quat,
            clipped,
            episode.stage,
        )
        episodes.append(episode)
        source_orders.append(source_order)
        initial_yaws.append(initial_yaw)
        per_episode_limit_reports[episode.name] = limit_report

    frame_counts = {len(episode.root_pos) for episode in episodes}
    fps_values = {round(episode.fps, 9) for episode in episodes}
    if len(frame_counts) != 1 or len(fps_values) != 1:
        raise ValueError(f"Episodes are not aligned: frame_counts={frame_counts}, fps={fps_values}")
    if len(set(source_orders)) != 1:
        raise ValueError("Source episodes do not share one joint order")

    medoid_index, pairwise = choose_medoid(episodes)
    medoid = episodes[medoid_index]
    fps = medoid.fps
    stage_indices = np.flatnonzero(medoid.stage == args.stair_stage)
    if len(stage_indices) == 0:
        raise ValueError(f"Stage {args.stair_stage} does not appear in the medoid episode")
    stair_start = int(stage_indices[0])
    clip_start = max(0, stair_start - int(round(args.pre_roll_seconds * fps)))
    if args.step_end_seconds is None:
        baseline = float(np.median(medoid.root_pos[max(0, stair_start - 5) : stair_start + 5, 2]))
        gain = medoid.root_pos[:, 2] - baseline
        candidates = np.flatnonzero((np.arange(len(gain)) >= stair_start) & (gain >= 0.95 * args.stair_height))
        if len(candidates) == 0:
            raise ValueError("Could not detect completion of the first stair from root height")
        clip_end = int(candidates[0] + round(args.post_step_seconds * fps))
    else:
        clip_end = int(round(args.step_end_seconds * fps))
    clip_end = min(clip_end, len(medoid.root_pos) - 1)
    if clip_end <= clip_start:
        raise ValueError(f"Invalid first-step frame range: {clip_start}..{clip_end}")
    hold_frames = int(round(args.hold_seconds * fps))

    output_dir: Path = args.output_dir
    episode_dir = output_dir / "episodes"
    for index, episode in enumerate(episodes, start=1):
        stem = f"trajectory_{index:02d}"
        _write_csv(
            episode_dir / f"{stem}_full_s53_50fps.csv",
            episode.root_pos,
            episode.root_quat,
            episode.joint_pos,
        )
        _write_csv(
            episode_dir / f"{stem}_step1_s53_50fps.csv",
            _slice_with_hold(episode.root_pos, clip_start, clip_end, hold_frames),
            _slice_with_hold(episode.root_quat, clip_start, clip_end, hold_frames),
            _slice_with_hold(episode.joint_pos, clip_start, clip_end, hold_frames),
        )

    full_csv = output_dir / "kuavoS53_stairs_full_medoid_50fps.csv"
    step_csv = output_dir / "kuavoS53_stairs_step1_medoid_50fps.csv"
    _write_csv(full_csv, medoid.root_pos, medoid.root_quat, medoid.joint_pos)
    _write_csv(
        step_csv,
        _slice_with_hold(medoid.root_pos, clip_start, clip_end, hold_frames),
        _slice_with_hold(medoid.root_quat, clip_start, clip_end, hold_frames),
        _slice_with_hold(medoid.joint_pos, clip_start, clip_end, hold_frames),
    )

    metadata: dict[str, object] = {
        "format_version": 1,
        "method": "S45-to-S53 explicit joint mapping and real-episode medoid selection",
        "source_collection": str(args.collection_root.resolve()),
        "source_episode_count": len(episodes),
        "source_joint_order": list(source_orders[0]),
        "target_robot": "KuavoS53",
        "target_joint_order": list(S53_JOINT_NAMES),
        "mapping": {
            "copied_by_name": [name for name in S53_JOINT_NAMES if name != "waist_yaw_joint"],
            "constant_joints": {"waist_yaw_joint": 0.0},
            "discarded_source_joints": ["zhead_1_joint", "zhead_2_joint"],
        },
        "coordinate_canonicalization": {
            "initial_xy": [0.0, 0.0],
            "initial_heading": "+X",
            "initial_root_height_m": args.target_root_height,
            "source_initial_yaw_rad": initial_yaws,
        },
        "joint_limit_validation": {
            "urdf": str(args.s53_urdf.resolve()),
            "soft_limit_factor": args.soft_limit_factor,
            "per_episode_clipping": per_episode_limit_reports,
        },
        "medoid": {
            "episode_index": medoid_index + 1,
            "episode_name": medoid.name,
            "source_path": str(medoid.source_path.resolve()),
            "mean_distance_by_episode": pairwise.mean(axis=1).tolist(),
        },
        "fps": fps,
        "full_motion": {
            "csv": str(full_csv.resolve()),
            "frames": len(medoid.root_pos),
            "duration_seconds": (len(medoid.root_pos) - 1) / fps,
        },
        "first_step_motion": {
            "csv": str(step_csv.resolve()),
            "source_frame_start": clip_start,
            "source_frame_end_inclusive": clip_end,
            "stair_stage_start_frame": stair_start,
            "hold_frames": hold_frames,
            "frames": clip_end - clip_start + 1 + hold_frames,
            "duration_seconds": (clip_end - clip_start + hold_frames) / fps,
            "root_height_gain_m": float(medoid.root_pos[clip_end, 2] - medoid.root_pos[stair_start, 2]),
        },
        "stair_geometry_m": {
            "count": 4,
            "height": args.stair_height,
            "tread": args.stair_tread,
            "width": args.stair_width,
            "first_riser_x": args.first_riser_x,
        },
    }
    metadata_path = output_dir / "prepare_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s53-urdf", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=10)
    parser.add_argument("--target-root-height", type=float, default=0.925)
    parser.add_argument("--soft-limit-factor", type=float, default=0.95)
    parser.add_argument("--stair-stage", type=int, default=8)
    parser.add_argument("--pre-roll-seconds", type=float, default=0.5)
    parser.add_argument("--step-end-seconds", type=float, default=4.2)
    parser.add_argument("--post-step-seconds", type=float, default=0.25)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--stair-height", type=float, default=0.13)
    parser.add_argument("--stair-tread", type=float, default=0.28)
    parser.add_argument("--stair-width", type=float, default=1.0)
    parser.add_argument("--first-riser-x", type=float, default=0.14)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
