#!/usr/bin/env python3
"""Build a forward step-to descent with explicit double-foot gates on every level."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prepare_kuavo_s53_updown import normalize_quaternions, smooth_blend, write_csv


JOINT_ORDER = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint",
    "zarm_l6_joint", "zarm_l7_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint",
    "zarm_r6_joint", "zarm_r7_joint",
]
LEG_INDICES = {
    "left": [JOINT_ORDER.index(f"leg_l{i}_joint") for i in range(1, 7)],
    "right": [JOINT_ORDER.index(f"leg_r{i}_joint") for i in range(1, 7)],
}


@dataclass
class JointGeometry:
    origin: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(axis * angle).as_matrix()


def load_leg_chains(urdf: Path) -> dict[str, list[JointGeometry]]:
    root = ET.parse(urdf).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    chains = {}
    for side, prefix in (("left", "leg_l"), ("right", "leg_r")):
        chain = []
        for index in range(1, 7):
            joint = joints[f"{prefix}{index}_joint"]
            origin_node = joint.find("origin")
            axis_node = joint.find("axis")
            limit_node = joint.find("limit")
            xyz = np.fromstring(origin_node.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin_node.attrib.get("rpy", "0 0 0"), sep=" ")
            axis = np.fromstring(axis_node.attrib["xyz"], sep=" ")
            axis /= np.linalg.norm(axis)
            chain.append(
                JointGeometry(
                    origin=xyz,
                    origin_rotation=rpy_matrix(rpy),
                    axis=axis,
                    lower=float(limit_node.attrib["lower"]),
                    upper=float(limit_node.attrib["upper"]),
                )
            )
        chains[side] = chain
    return chains


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def fk_leg(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_position: np.ndarray,
    chain: list[JointGeometry],
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(root_position, dtype=np.float64).copy()
    rotation = quat_wxyz_to_matrix(root_quaternion)
    for value, joint in zip(joint_position, chain):
        position = position + rotation @ joint.origin
        rotation = rotation @ joint.origin_rotation @ axis_rotation(joint.axis, float(value))
    return position, rotation


def solve_leg(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    initial: np.ndarray,
    chain: list[JointGeometry],
) -> tuple[np.ndarray, float, float]:
    lower = np.asarray([joint.lower for joint in chain]) + 1.0e-5
    upper = np.asarray([joint.upper for joint in chain]) - 1.0e-5
    initial = np.clip(initial, lower, upper)

    def residual(values: np.ndarray) -> np.ndarray:
        position, rotation = fk_leg(root_position, root_quaternion, values, chain)
        orientation_error = Rotation.from_matrix(target_rotation.T @ rotation).as_rotvec()
        return np.concatenate(((position - target_position) * 8.0, orientation_error, (values - initial) * 0.002))

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=120,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    position, rotation = fk_leg(root_position, root_quaternion, result.x, chain)
    position_error = float(np.linalg.norm(position - target_position))
    orientation_error = float(np.linalg.norm(Rotation.from_matrix(target_rotation.T @ rotation).as_rotvec()))
    return result.x, position_error, orientation_error


def smooth_alpha(frames: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def blend(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    alpha = smooth_alpha(frames)[:, None]
    return start[None] * (1.0 - alpha) + end[None] * alpha


def swing(start: np.ndarray, end: np.ndarray, frames: int, clearance: float) -> np.ndarray:
    alpha = smooth_alpha(frames)
    result = start[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None]
    result[:, 2] += clearance * 4.0 * alpha * (1.0 - alpha)
    return result


def solve_segment(
    root_positions: np.ndarray,
    root_quaternions: np.ndarray,
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    base_joint: np.ndarray,
    chains: dict[str, list[JointGeometry]],
    previous_left: np.ndarray,
    previous_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    joints = np.repeat(base_joint[None], len(root_positions), axis=0)
    errors = {"position": 0.0, "orientation": 0.0}
    foot_rotation = np.eye(3)
    for frame in range(len(root_positions)):
        previous_left, pos_error, ori_error = solve_leg(
            root_positions[frame], root_quaternions[frame], left_targets[frame], foot_rotation,
            previous_left, chains["left"],
        )
        errors["position"] = max(errors["position"], pos_error)
        errors["orientation"] = max(errors["orientation"], ori_error)
        previous_right, pos_error, ori_error = solve_leg(
            root_positions[frame], root_quaternions[frame], right_targets[frame], foot_rotation,
            previous_right, chains["right"],
        )
        errors["position"] = max(errors["position"], pos_error)
        errors["orientation"] = max(errors["orientation"], ori_error)
        joints[frame, LEG_INDICES["left"]] = previous_left
        joints[frame, LEG_INDICES["right"]] = previous_right
    return joints, previous_left, previous_right, errors


def build(args: argparse.Namespace) -> None:
    source = np.loadtxt(args.input_csv, delimiter=",")
    if source.shape != (1343, 34):
        raise ValueError(f"Expected the stable 1343x34 CSV, got {source.shape}")
    chains = load_leg_chains(args.urdf)

    prefix = source[:693]
    platform_end = source[692]
    source_gate = source[742]
    center_y = float(source_gate[1])
    top_root = source_gate[:3].copy()
    top_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    top_surface = 0.52
    foot_height = 0.0645
    root_surface_offset = float(top_root[2] - top_surface)
    foot_y = {"left": center_y + 0.105, "right": center_y - 0.105}
    initial_foot_x = 1.98
    left_foot = np.array([initial_foot_x, foot_y["left"], top_surface + foot_height])
    right_foot = np.array([initial_foot_x, foot_y["right"], top_surface + foot_height])

    base_joint = source_gate[7:].copy()
    previous_left = base_joint[LEG_INDICES["left"]].copy()
    previous_right = base_joint[LEG_INDICES["right"]].copy()
    initial_joint, previous_left, previous_right, initial_errors = solve_segment(
        top_root[None], top_quaternion[None], left_foot[None], right_foot[None], base_joint,
        chains, previous_left, previous_right,
    )
    initial_joint = initial_joint[0]

    transition_frames = args.stance_transition_frames
    transition_pos = blend(platform_end[:3], top_root, transition_frames)
    transition_quat = smooth_blend(
        normalize_quaternions(source[692:693, 3:7])[0], top_quaternion,
        transition_frames, quaternion=True,
    )
    transition_joint = blend(platform_end[7:], initial_joint, transition_frames)
    gate_pos = np.repeat(top_root[None], args.pre_descent_gate_frames, axis=0)
    gate_quat = np.repeat(top_quaternion[None], args.pre_descent_gate_frames, axis=0)
    gate_joint = np.repeat(initial_joint[None], args.pre_descent_gate_frames, axis=0)

    root_parts: list[np.ndarray] = []
    quat_parts: list[np.ndarray] = []
    joint_parts: list[np.ndarray] = []
    target_parts: list[tuple[np.ndarray, np.ndarray]] = []
    segment_report: dict[str, list[int]] = {}
    ik_position_error = initial_errors["position"]
    ik_orientation_error = initial_errors["orientation"]
    current_frame = len(prefix) + transition_frames + args.pre_descent_gate_frames
    current_root = top_root.copy()
    current_left = left_foot.copy()
    current_right = right_foot.copy()

    def append_phase(
        name: str,
        root_position: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> None:
        nonlocal previous_left, previous_right, current_frame, ik_position_error, ik_orientation_error
        root_quaternion = np.repeat(top_quaternion[None], len(root_position), axis=0)
        joints, previous_left, previous_right, errors = solve_segment(
            root_position, root_quaternion, left_target, right_target, base_joint,
            chains, previous_left, previous_right,
        )
        root_parts.append(root_position)
        quat_parts.append(root_quaternion)
        joint_parts.append(joints)
        target_parts.append((left_target, right_target))
        segment_report[name] = [current_frame, current_frame + len(root_position) - 1]
        current_frame += len(root_position)
        ik_position_error = max(ik_position_error, errors["position"])
        ik_orientation_error = max(ik_orientation_error, errors["orientation"])

    level_targets = [(2.40, 0.39), (2.68, 0.26), (2.96, 0.13), (3.24, 0.0)]
    double_gate_ranges = []
    left_swing_ranges = []
    right_swing_ranges = []
    for level, (target_x, target_surface) in enumerate(level_targets, start=1):
        target_left = np.array([target_x, foot_y["left"], target_surface + foot_height])
        target_right = np.array([target_x, foot_y["right"], target_surface + foot_height])
        double_root = np.array([target_x + 0.07, center_y, target_surface + root_surface_offset])
        split_root = np.array(
            [0.5 * (current_left[0] + target_x) + 0.07, center_y - 0.045,
             0.5 * (current_root[2] + double_root[2])]
        )

        root = blend(current_root, split_root, args.swing_frames)
        left = swing(current_left, target_left, args.swing_frames, args.swing_clearance)
        right = np.repeat(current_right[None], args.swing_frames, axis=0)
        phase_start = current_frame
        append_phase(f"level_{level}_left_down", root, left, right)
        left_swing_ranges.append([phase_start, current_frame - 1])

        support_root = np.array([target_x + 0.02, center_y + 0.045, double_root[2]])
        root = blend(split_root, support_root, args.single_support_hold_frames)
        left = np.repeat(target_left[None], args.single_support_hold_frames, axis=0)
        right = np.repeat(current_right[None], args.single_support_hold_frames, axis=0)
        append_phase(f"level_{level}_left_support_hold", root, left, right)

        root = blend(support_root, double_root, args.swing_frames)
        left = np.repeat(target_left[None], args.swing_frames, axis=0)
        right = swing(current_right, target_right, args.swing_frames, args.swing_clearance)
        phase_start = current_frame
        append_phase(f"level_{level}_right_join", root, left, right)
        right_swing_ranges.append([phase_start, current_frame - 1])

        root = np.repeat(double_root[None], args.double_gate_frames, axis=0)
        left = np.repeat(target_left[None], args.double_gate_frames, axis=0)
        right = np.repeat(target_right[None], args.double_gate_frames, axis=0)
        phase_start = current_frame
        append_phase(f"level_{level}_double_foot_gate", root, left, right)
        double_gate_ranges.append([phase_start, current_frame - 1])

        current_root, current_left, current_right = double_root, target_left, target_right

    root = np.repeat(current_root[None], args.descent_end_hold_frames, axis=0)
    left = np.repeat(current_left[None], args.descent_end_hold_frames, axis=0)
    right = np.repeat(current_right[None], args.descent_end_hold_frames, axis=0)
    append_phase("descent_end_stable", root, left, right)

    descent_pos = np.concatenate(root_parts)
    descent_quat = np.concatenate(quat_parts)
    descent_joint = np.concatenate(joint_parts)
    final_pos = np.repeat(descent_pos[-1:], args.final_hold_frames, axis=0)
    final_quat = np.repeat(descent_quat[-1:], args.final_hold_frames, axis=0)
    final_joint = np.repeat(descent_joint[-1:], args.final_hold_frames, axis=0)

    root_position = np.concatenate((prefix[:, :3], transition_pos, gate_pos, descent_pos, final_pos))
    root_quaternion = normalize_quaternions(
        np.concatenate((prefix[:, 3:7], transition_quat, gate_quat, descent_quat, final_quat))
    )
    joint_position = np.concatenate((prefix[:, 7:], transition_joint, gate_joint, descent_joint, final_joint))
    write_csv(args.output_csv, root_position, root_quaternion, joint_position)

    pre_gate_start = len(prefix) + transition_frames
    descent_start = pre_gate_start + args.pre_descent_gate_frames
    report = {
        "format_version": 3,
        "reference": "forward_left_first_step_to_descent_with_per_level_double_foot_gates",
        "fps": args.fps,
        "frames": int(len(root_position)),
        "duration_seconds": (len(root_position) - 1) / args.fps,
        "fixed_pre_descent_root_xyz": top_root.tolist(),
        "segments": {
            "ascent": [0, 549],
            "ascent_to_platform": [550, 559],
            "platform_walk": [560, 692],
            "platform_to_double_stance": [693, pre_gate_start - 1],
            "pre_descent_double_foot_gate": [pre_gate_start, descent_start - 1],
            **segment_report,
            "final_hold": [len(root_position) - args.final_hold_frames, len(root_position) - 1],
        },
        "gate_frames": [pre_gate_start + args.pre_descent_gate_frames - 1]
        + [end for _, end in double_gate_ranges],
        "double_foot_gate_ranges": [[pre_gate_start, descent_start - 1], *double_gate_ranges],
        "left_swing_ranges": left_swing_ranges,
        "right_swing_ranges": right_swing_ranges,
        "level_target_x_surface_z": [[x, z] for x, z in level_targets],
        "foot_target_y": foot_y,
        "maximum_ik_position_error_m": ik_position_error,
        "maximum_ik_orientation_error_rad": ik_orientation_error,
        "geometry_m": {
            "step_height": 0.13,
            "step_tread": 0.28,
            "platform_length": 1.0,
            "foot_rectangle_length": 0.24,
            "foot_rectangle_width": 0.10,
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--stance-transition-frames", type=int, default=35)
    parser.add_argument("--pre-descent-gate-frames", type=int, default=35)
    parser.add_argument("--swing-frames", type=int, default=45)
    parser.add_argument("--single-support-hold-frames", type=int, default=15)
    parser.add_argument("--double-gate-frames", type=int, default=25)
    parser.add_argument("--descent-end-hold-frames", type=int, default=30)
    parser.add_argument("--final-hold-frames", type=int, default=50)
    parser.add_argument("--swing-clearance", type=float, default=0.09)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
