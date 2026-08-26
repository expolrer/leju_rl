#!/usr/bin/env python3
"""Validate S53 foot-link landings against the configured four stair treads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[..., 1:]
    w = quat[..., :1]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def validate(
    motion_file: Path,
    stair_height: float,
    stair_tread: float,
    first_riser_x: float,
    z_tolerance: float,
) -> dict[str, object]:
    with np.load(motion_file, allow_pickle=True) as data:
        names = [str(value) for value in data["end_effector_names"]]
        left_index = names.index("leg_l6_link")
        right_index = names.index("leg_r6_link")
        root_pos = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
        foot_pos_b = np.asarray(data["end_effector_pos_b"][:, [left_index, right_index]], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    repeated_quat = np.repeat(root_quat, 2, axis=0)
    feet_w = root_pos[:, None, :] + _quat_rotate_wxyz(
        repeated_quat, foot_pos_b.reshape(-1, 3)
    ).reshape(-1, 2, 3)
    initial_link_z = np.median(feet_w[: min(40, len(feet_w)), :, 2], axis=0)

    landings = []
    previous_frame = 0
    for level, foot_index in enumerate((0, 1, 0, 1), start=1):
        lower_x = first_riser_x + (level - 1) * stair_tread
        upper_x = lower_x + stair_tread
        expected_z = initial_link_z[foot_index] + level * stair_height
        valid_x = (feet_w[:, foot_index, 0] >= lower_x) & (feet_w[:, foot_index, 0] <= upper_x)
        candidate_indices = np.flatnonzero(valid_x & (np.arange(len(feet_w)) >= previous_frame))
        if len(candidate_indices) == 0:
            raise ValueError(f"No level-{level} landing candidate falls inside its tread")
        errors = np.abs(feet_w[candidate_indices, foot_index, 2] - expected_z)
        frame = int(candidate_indices[int(np.argmin(errors))])
        z_error = float(abs(feet_w[frame, foot_index, 2] - expected_z))
        if z_error > z_tolerance:
            raise ValueError(
                f"Level-{level} landing z error {z_error:.4f} exceeds tolerance {z_tolerance:.4f}"
            )
        landings.append(
            {
                "level": level,
                "foot": "left" if foot_index == 0 else "right",
                "frame": frame,
                "time_seconds": frame / fps,
                "foot_link_xyz_m": feet_w[frame, foot_index].tolist(),
                "tread_x_range_m": [lower_x, upper_x],
                "expected_foot_link_z_m": float(expected_z),
                "z_error_m": z_error,
            }
        )
        previous_frame = frame

    return {
        "valid": True,
        "motion_file": str(motion_file.resolve()),
        "fps": fps,
        "frame_count": len(feet_w),
        "initial_foot_link_z_m": initial_link_z.tolist(),
        "stair_height_m": stair_height,
        "stair_tread_m": stair_tread,
        "first_riser_x_m": first_riser_x,
        "landings": landings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--stair-height", type=float, default=0.13)
    parser.add_argument("--stair-tread", type=float, default=0.28)
    parser.add_argument("--first-riser-x", type=float, default=0.14)
    parser.add_argument("--z-tolerance", type=float, default=0.025)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = validate(
        args.motion_file,
        args.stair_height,
        args.stair_tread,
        args.first_riser_x,
        args.z_tolerance,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
