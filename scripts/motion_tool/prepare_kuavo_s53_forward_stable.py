#!/usr/bin/env python3
"""Build a fixed-gate, forward-stepping up-platform-down reference.

The stable descent is the time reverse of the validated ascent joint motion,
translated forward across the symmetric descending staircase.  A stationary
hold at the platform exit gives the policy an explicit stabilization gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_kuavo_s53_updown import normalize_quaternions, smooth_blend, write_csv


def repeat(values: np.ndarray, frames: int) -> np.ndarray:
    return np.repeat(values[-1:], frames, axis=0)


def load_motion(path: Path) -> np.ndarray:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1 if first_line.startswith("body_pos_x,") else 0,
        usecols=range(34),
    )
    if data.ndim != 2 or data.shape[1] != 34 or len(data) < 1293:
        raise ValueError(f"Expected the 1343-frame, 34-column combined CSV, got {data.shape}")
    return data


def build(args: argparse.Namespace) -> None:
    source = load_motion(args.input_csv)
    up = source[0:550]
    flat = source[580:713]

    up_pos, up_quat, up_joint = up[:, :3], normalize_quaternions(up[:, 3:7]), up[:, 7:]
    flat_pos, flat_quat, flat_joint = flat[:, :3], normalize_quaternions(flat[:, 3:7]), flat[:, 7:]

    # Shorten ascent-to-platform settling from 30 to 10 frames.
    transition_in_pos = np.repeat(up_pos[-1:], args.ascent_blend_frames, axis=0)
    transition_in_quat = smooth_blend(
        up_quat[-1], flat_quat[0], args.ascent_blend_frames, quaternion=True
    )
    transition_in_joint = smooth_blend(up_joint[-1], flat_joint[0], args.ascent_blend_frames)

    # Reverse the demonstrated stair gait, but keep the robot facing and moving +X.
    down_pos = up_pos[::-1].copy()
    down_pos[:, 0] = flat_pos[-1, 0] + up_pos[-1, 0] - down_pos[:, 0]
    down_pos[:, 1] = flat_pos[-1, 1] + down_pos[:, 1] - up_pos[-1, 1]
    down_pos[:, 2] = flat_pos[-1, 2] + down_pos[:, 2] - up_pos[-1, 2]
    down_quat = normalize_quaternions(up_quat[::-1])
    down_joint = up_joint[::-1].copy()

    # Blend into the first descent pose at a fixed platform coordinate, then hold.
    transition_out_pos = np.repeat(flat_pos[-1:], args.descent_blend_frames, axis=0)
    transition_out_quat = smooth_blend(
        flat_quat[-1], down_quat[0], args.descent_blend_frames, quaternion=True
    )
    transition_out_joint = smooth_blend(flat_joint[-1], down_joint[0], args.descent_blend_frames)
    stable_pos = np.repeat(flat_pos[-1:], args.stable_hold_frames, axis=0)
    stable_quat = np.repeat(down_quat[:1], args.stable_hold_frames, axis=0)
    stable_joint = np.repeat(down_joint[:1], args.stable_hold_frames, axis=0)

    final_pos = repeat(down_pos, args.final_hold_frames)
    final_quat = repeat(down_quat, args.final_hold_frames)
    final_joint = repeat(down_joint, args.final_hold_frames)

    root_pos = np.concatenate(
        (up_pos, transition_in_pos, flat_pos, transition_out_pos, stable_pos, down_pos, final_pos)
    )
    root_quat = normalize_quaternions(
        np.concatenate(
            (up_quat, transition_in_quat, flat_quat, transition_out_quat, stable_quat, down_quat, final_quat)
        )
    )
    joint_pos = np.concatenate(
        (up_joint, transition_in_joint, flat_joint, transition_out_joint, stable_joint, down_joint, final_joint)
    )
    write_csv(args.output_csv, root_pos, root_quat, joint_pos)

    segments = {
        "ascent": [0, 549],
        "ascent_to_platform": [550, 559],
        "platform_walk": [560, 692],
        "platform_to_descent_blend": [693, 707],
        "pre_descent_stable_hold": [708, 742],
        "descent": [743, 1292],
        "final_hold": [1293, 1342],
    }
    if len(root_pos) != 1343:
        raise ValueError(f"Frame allocation changed checkpoint-compatible length: {len(root_pos)}")
    report = {
        "format_version": 2,
        "reference": "forward_time_reversed_ascent_with_fixed_platform_gate",
        "fps": args.fps,
        "frames": len(root_pos),
        "duration_seconds": (len(root_pos) - 1) / args.fps,
        "segments": segments,
        "fixed_descent_switch_root_xyz": flat_pos[-1].tolist(),
        "fixed_descent_switch_frame": segments["descent"][0],
        "stable_hold_seconds": args.stable_hold_frames / args.fps,
        "ascent_transition_seconds": args.ascent_blend_frames / args.fps,
        "geometry_m": {
            "step_count_each_side": 4,
            "step_height": 0.13,
            "step_tread": 0.28,
            "stair_width": 1.5,
            "platform_length": 1.0,
        },
        "source_csv": str(args.input_csv),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--ascent-blend-frames", type=int, default=10)
    parser.add_argument("--descent-blend-frames", type=int, default=15)
    parser.add_argument("--stable-hold-frames", type=int, default=35)
    parser.add_argument("--final-hold-frames", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
