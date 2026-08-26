#!/usr/bin/env python3
"""Build a KuavoS53 up-platform-down reference from captured motions.

The ascent is the validated S53 reference. Platform walking comes from the
real stage-3 flat gait in the original S45 capture. The initial descent
demonstration is the time reversal of the ascent, translated forward so its
contacts line up with a symmetric four-step staircase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_kuavo_s53_stairs import (
    canonicalize_root,
    clip_to_joint_limits,
    load_soft_joint_limits,
    retarget_joint_positions,
)


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def smooth_blend(start: np.ndarray, end: np.ndarray, frames: int, quaternion: bool = False) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, frames + 2, dtype=np.float64)[1:-1]
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    if quaternion and np.dot(start, end) < 0.0:
        end = -end
    values = start[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None]
    if quaternion:
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def resample_sequence(values: np.ndarray, frames: int, quaternion: bool = False) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    phase = np.linspace(0.0, len(source) - 1, frames, dtype=np.float64)
    lower = np.floor(phase).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = phase - lower
    left = source[lower]
    right = source[upper].copy()
    if quaternion:
        right[np.sum(left * right, axis=1) < 0.0] *= -1.0
    result = left * (1.0 - alpha[:, None]) + right * alpha[:, None]
    return normalize_quaternions(result) if quaternion else result


def load_flat_stage(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    chunk_paths = sorted((args.capture_dir / "chunks").glob("chunk_*.npz"))
    if not chunk_paths:
        raise ValueError(f"No chunks found under {args.capture_dir}")

    fields = {
        "time": "core__sim_time",
        "stage": "core__solver_stage_index",
        "root_pos": "core__root_pos_w",
        "root_quat": "core__root_quat_wxyz",
        "joint_pos": "core__joint_pos_ordered",
    }
    parts: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    for path in chunk_paths:
        with np.load(path) as data:
            for name, key in fields.items():
                parts[name].append(np.asarray(data[key]))
    data = {name: np.concatenate(values) for name, values in parts.items()}

    indexes = np.flatnonzero(data["stage"] == args.flat_stage)
    if len(indexes) < 2:
        raise ValueError(f"Stage {args.flat_stage} has too few frames")
    # Captures run at 500 Hz. Keep exact samples at the requested output rate.
    capture_dt = float(np.median(np.diff(data["time"][indexes])))
    stride = max(1, int(round((1.0 / args.fps) / capture_dt)))
    indexes = indexes[::stride]
    if indexes[-1] != np.flatnonzero(data["stage"] == args.flat_stage)[-1]:
        indexes = np.append(indexes, np.flatnonzero(data["stage"] == args.flat_stage)[-1])

    metadata = json.loads((args.capture_dir / "metadata.json").read_text(encoding="utf-8"))
    joint_names = metadata["ordered_joint_names"]
    root_pos, root_quat, source_yaw = canonicalize_root(
        data["root_pos"][indexes], data["root_quat"][indexes], args.root_height
    )
    joint_pos = retarget_joint_positions(data["joint_pos"][indexes], joint_names)
    limits = load_soft_joint_limits(args.s53_urdf, args.soft_limit_factor)
    joint_pos, clipping = clip_to_joint_limits(joint_pos, limits)

    if root_pos[-1, 0] < root_pos[0, 0]:
        root_pos[:, :2] *= -1.0
        yaw_pi = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        w, x, y, z = np.moveaxis(root_quat, -1, 0)
        pw, px, py, pz = yaw_pi
        root_quat = np.stack(
            (
                pw * w - px * x - py * y - pz * z,
                pw * x + px * w + py * z - pz * y,
                pw * y - px * z + py * w + pz * x,
                pw * z + px * y - py * x + pz * w,
            ),
            axis=-1,
        )
    source_lateral_drift = float(root_pos[-1, 1] - root_pos[0, 1])
    source_vertical_drift = float(root_pos[-1, 2] - root_pos[0, 2])
    phase = np.linspace(0.0, 1.0, len(root_pos), dtype=np.float64)
    root_pos[:, 1] -= phase * source_lateral_drift
    root_pos[:, 2] -= phase * source_vertical_drift
    travel = float(root_pos[-1, 0] - root_pos[0, 0])
    if travel < 0.5:
        raise ValueError(f"Flat stage only travels {travel:.3f} m after canonicalization")
    root_pos[:, 0] *= args.platform_travel / travel
    return root_pos, normalize_quaternions(root_quat), joint_pos, {
        "capture_stride": stride,
        "source_frames": int(len(indexes)),
        "source_yaw_rad": source_yaw,
        "source_travel_m": travel,
        "removed_lateral_drift_m": source_lateral_drift,
        "removed_vertical_drift_m": source_vertical_drift,
        "target_travel_m": args.platform_travel,
        "joint_limit_clipping": clipping,
    }


def write_csv(path: Path, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
    values = np.concatenate((root_pos, root_quat, joint_pos), axis=1)
    if values.shape[1] != 34 or not np.isfinite(values).all():
        raise ValueError(f"Invalid output motion shape or values: {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.10f")


def build(args: argparse.Namespace) -> None:
    first_line = args.ascent_csv.read_text(encoding="utf-8").splitlines()[0]
    has_header = first_line.startswith("body_pos_x,")
    ascent = np.loadtxt(
        args.ascent_csv,
        delimiter=",",
        skiprows=1 if has_header else 0,
        usecols=range(34),
    )
    if ascent.ndim != 2 or ascent.shape[1] != 34:
        raise ValueError(f"Expected a 34-column ascent CSV, got {ascent.shape}")
    up_pos = ascent[:, :3]
    up_quat = normalize_quaternions(ascent[:, 3:7])
    up_joint = ascent[:, 7:]

    flat_pos, flat_quat, flat_joint, flat_metadata = load_flat_stage(args)
    top_pos = up_pos[-1].copy()
    flat_pos = flat_pos - flat_pos[0] + top_pos

    transition_in_pos = np.repeat(top_pos[None], args.blend_frames, axis=0)
    transition_in_quat = smooth_blend(up_quat[-1], flat_quat[0], args.blend_frames, quaternion=True)
    transition_in_joint = smooth_blend(up_joint[-1], flat_joint[0], args.blend_frames)

    down_pos = up_pos[::-1].copy()
    down_pos[:, 0] = flat_pos[-1, 0] + up_pos[-1, 0] - down_pos[:, 0]
    down_pos[:, 2] = flat_pos[-1, 2] + down_pos[:, 2] - up_pos[-1, 2]
    if args.descent_mode == "backward":
        down_pos[:, 1] = flat_pos[-1, 1] + up_pos[-1, 1] - down_pos[:, 1]
        yaw_pi = np.broadcast_to(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), up_quat.shape
        )
        down_quat = normalize_quaternions(quat_mul_wxyz(yaw_pi, up_quat[::-1]))
        down_joint = up_joint[::-1].copy()
        turn_pos = np.repeat(flat_pos[-1][None], args.turn_frames, axis=0)
        turn_quat = smooth_blend(up_quat[-1], down_quat[0], args.turn_frames, quaternion=True)
        turn_joint = np.repeat(up_joint[-1][None], args.turn_frames, axis=0)
    else:
        # Preserve a forward-facing gait prior while the task-space root follows
        # the validated stair height profile in the descending direction.
        down_flat_pos = resample_sequence(flat_pos - flat_pos[:1], len(up_pos))
        down_pos[:, 1] = flat_pos[-1, 1] + down_flat_pos[:, 1]
        down_quat = resample_sequence(flat_quat, len(up_pos), quaternion=True)
        down_joint = resample_sequence(flat_joint, len(up_pos))
        turn_pos = np.empty((0, 3), dtype=np.float64)
        turn_quat = np.empty((0, 4), dtype=np.float64)
        turn_joint = np.empty((0, up_joint.shape[1]), dtype=np.float64)

    transition_out_pos = np.repeat(flat_pos[-1][None], args.blend_frames, axis=0)
    transition_out_quat = smooth_blend(flat_quat[-1], down_quat[0], args.blend_frames, quaternion=True)
    transition_out_joint = smooth_blend(flat_joint[-1], down_joint[0], args.blend_frames)

    final_hold_pos = np.repeat(down_pos[-1:], args.final_hold_frames, axis=0)
    final_hold_quat = np.repeat(down_quat[-1:], args.final_hold_frames, axis=0)
    final_hold_joint = np.repeat(down_joint[-1:], args.final_hold_frames, axis=0)

    root_pos = np.concatenate(
        (up_pos, transition_in_pos, flat_pos, transition_out_pos, turn_pos, down_pos, final_hold_pos), axis=0
    )
    root_quat = normalize_quaternions(
        np.concatenate(
            (up_quat, transition_in_quat, flat_quat, transition_out_quat, turn_quat, down_quat, final_hold_quat),
            axis=0,
        )
    )
    joint_pos = np.concatenate(
        (up_joint, transition_in_joint, flat_joint, transition_out_joint, turn_joint, down_joint, final_hold_joint),
        axis=0,
    )
    write_csv(args.output_csv, root_pos, root_quat, joint_pos)

    up_end = len(up_pos) - 1
    platform_start = up_end + args.blend_frames + 1
    platform_end = platform_start + len(flat_pos) - 1
    turn_start = platform_end + args.blend_frames + 1
    descent_start = turn_start + len(turn_pos)
    segments = {
        "ascent": [0, up_end],
        "platform_walk": [platform_start, platform_end],
        "descent": [descent_start, descent_start + len(down_pos) - 1],
        "final_hold": [len(root_pos) - args.final_hold_frames, len(root_pos) - 1],
    }
    if len(turn_pos):
        segments["top_turn"] = [turn_start, descent_start - 1]
    report = {
        "format_version": 1,
        "descent_mode": args.descent_mode,
        "fps": args.fps,
        "frames": len(root_pos),
        "duration_seconds": (len(root_pos) - 1) / args.fps,
        "segments": segments,
        "geometry_m": {
            "step_count_each_side": 4,
            "step_height": 0.13,
            "step_tread": 0.28,
            "stair_width": 1.5,
            "platform_length": 1.0,
        },
        "root_start_xyz": root_pos[0].tolist(),
        "root_top_before_descent_xyz": down_pos[0].tolist(),
        "root_end_xyz": root_pos[-1].tolist(),
        "flat_source": flat_metadata,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascent-csv", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--s53-urdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--flat-stage", type=int, default=3)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--root-height", type=float, default=0.925)
    parser.add_argument("--platform-travel", type=float, default=0.84)
    parser.add_argument("--blend-frames", type=int, default=30)
    parser.add_argument("--turn-frames", type=int, default=100)
    parser.add_argument("--descent-mode", choices=("forward", "backward"), default="backward")
    parser.add_argument("--final-hold-frames", type=int, default=50)
    parser.add_argument("--soft-limit-factor", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
