#!/usr/bin/env python3
"""Compute phase, gate, contact, and sliding metrics from an Isaac rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEGMENTS = {
    "ascent": (0, 549),
    "ascent_to_platform": (550, 559),
    "platform_walk": (560, 692),
    "platform_to_descent_blend": (693, 707),
    "pre_descent_stable_hold": (708, 742),
    "descent": (743, 1292),
    "final_hold": (1293, 1341),
}


def quaternion_angular_speed(quaternions: np.ndarray, dt: float) -> np.ndarray:
    left = quaternions[:-1]
    right = quaternions[1:]
    dot = np.abs(np.sum(left * right, axis=-1)).clip(0.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return np.concatenate(([0.0], angle / dt))


def landing_sequence(contact: np.ndarray, levels: np.ndarray, mask: np.ndarray) -> list[float]:
    onset = contact & ~np.concatenate(([False], contact[:-1])) & mask
    sequence = []
    for level in levels[onset]:
        if not np.isfinite(level):
            continue
        rounded = float(np.round(level / 0.13) * 0.13)
        if not sequence or abs(sequence[-1] - rounded) > 0.03:
            sequence.append(rounded)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    dt = float(arrays["dt"])
    frames = arrays["motion_frame"]
    root = arrays["root_pos"]
    root_quat = arrays["root_quat"]
    actual = arrays["actual_body_pos"]
    reference = arrays["reference_body_pos"]
    names = [str(name) for name in arrays["body_names"]]
    foot_indices = [names.index("leg_l6_link"), names.index("leg_r6_link")]
    foot_names = ["left", "right"]
    forces = np.linalg.norm(arrays["foot_contact_force_w"], axis=-1)
    contact = forces > 10.0
    surface_z = arrays["foot_surface_z"]

    root_speed = np.linalg.norm(np.diff(root, axis=0, prepend=root[:1]), axis=-1) / dt
    root_ang_speed = quaternion_angular_speed(root_quat, dt)
    foot_xy_speed = np.linalg.norm(
        np.diff(actual[:, foot_indices, :2], axis=0, prepend=actual[:1, foot_indices, :2]), axis=-1
    ) / dt
    reference_foot_speed = np.linalg.norm(
        np.diff(reference[:, foot_indices], axis=0, prepend=reference[:1, foot_indices]), axis=-1
    ) / dt

    phases = {}
    for name, (start, end) in SEGMENTS.items():
        mask = (frames >= start) & (frames <= end)
        indexes = np.flatnonzero(mask)
        metric_mask = mask & (np.arange(len(frames)) >= 2)
        phases[name] = {
            "samples": int(len(indexes)),
            "root_start": root[indexes[0]].tolist(),
            "root_end": root[indexes[-1]].tolist(),
            "root_displacement": (root[indexes[-1]] - root[indexes[0]]).tolist(),
            "anchor_error_mean_m": float(np.mean(arrays["anchor_error"][metric_mask])),
            "anchor_error_max_m": float(np.max(arrays["anchor_error"][metric_mask])),
            "body_error_mean_m": float(np.mean(arrays["body_error"][metric_mask])),
            "body_error_max_m": float(np.max(arrays["body_error"][metric_mask])),
        }

    gate_mask = (frames >= 708) & (frames <= 742)
    descent_mask = (frames >= 743) & (frames <= 1292)
    gate_reference = reference[gate_mask, names.index("base_link")]
    gate_actual = actual[gate_mask, names.index("base_link")]
    gate_position_error = np.linalg.norm(gate_actual - gate_reference, axis=-1)

    foot_metrics = {}
    for foot, index in enumerate(foot_names):
        phase_contact = contact[:, foot] & descent_mask
        reference_swing = (reference_foot_speed[:, foot] > 0.06) & descent_mask
        swing_contact = reference_swing & contact[:, foot]
        foot_metrics[index] = {
            "descent_contact_ratio": float(np.mean(contact[descent_mask, foot])),
            "descent_contact_slide_mean_mps": float(np.mean(foot_xy_speed[phase_contact, foot]))
            if np.any(phase_contact) else 0.0,
            "descent_contact_slide_p95_mps": float(np.percentile(foot_xy_speed[phase_contact, foot], 95))
            if np.any(phase_contact) else 0.0,
            "descent_contact_slide_distance_m": float(np.sum(foot_xy_speed[phase_contact, foot]) * dt),
            "reference_swing_contact_ratio": float(np.sum(swing_contact) / max(1, np.sum(reference_swing))),
            "max_contact_force_n": float(np.max(forces[descent_mask, foot])),
            "descent_landing_surface_sequence_m": landing_sequence(
                contact[:, foot], surface_z[:, foot], descent_mask
            ),
        }

    action_delta = np.linalg.norm(np.diff(arrays["actions"], axis=0), axis=-1)
    report = {
        "input": str(args.input),
        "dt": dt,
        "completion_ratio": float(np.max(frames) / 1341.0),
        "failure_resets": int(np.count_nonzero(arrays["dones"])),
        "phases": phases,
        "fixed_gate": {
            "reference_mean_xyz": np.mean(gate_reference, axis=0).tolist(),
            "actual_mean_xyz": np.mean(gate_actual, axis=0).tolist(),
            "position_error_mean_m": float(np.mean(gate_position_error)),
            "position_error_max_m": float(np.max(gate_position_error)),
            "root_linear_speed_mean_mps": float(np.mean(root_speed[gate_mask])),
            "root_linear_speed_max_mps": float(np.max(root_speed[gate_mask])),
            "root_angular_speed_mean_radps": float(np.mean(root_ang_speed[gate_mask])),
            "root_angular_speed_max_radps": float(np.max(root_ang_speed[gate_mask])),
        },
        "descent_feet": foot_metrics,
        "action_delta": {
            "mean_l2": float(np.mean(action_delta)),
            "p95_l2": float(np.percentile(action_delta, 95)),
            "max_l2": float(np.max(action_delta)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
