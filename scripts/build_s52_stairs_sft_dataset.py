#!/usr/bin/env python3
"""Build S53-retention plus S52-correction data for residual-policy SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


S53_ACTION_SCALE = np.asarray(
    [
        0.2116666667,
        0.1183333333,
        0.165,
        0.2947368421,
        0.2081818182,
        0.1295454545,
        0.2116666667,
        0.1183333333,
        0.165,
        0.2947368421,
        0.2081818182,
        0.1295454545,
        0.255,
        0.33,
        0.375,
        0.285,
        0.375,
        0.0916666667,
        0.0916666667,
        0.0916666667,
        0.33,
        0.375,
        0.285,
        0.375,
        0.0916666667,
        0.0916666667,
        0.0916666667,
    ],
    dtype=np.float32,
)

S52_ACTION_SCALE = np.asarray(
    [
        0.15875,
        0.08875,
        0.20625,
        0.4375,
        0.2375,
        0.2375,
        0.15875,
        0.08875,
        0.20625,
        0.4375,
        0.2375,
        0.2375,
        0.425,
        0.275,
        0.3125,
        0.475,
        0.3125,
        0.1175,
        0.1175,
        0.1175,
        0.275,
        0.3125,
        0.475,
        0.3125,
        0.1175,
        0.1175,
        0.1175,
    ],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings(values: np.ndarray) -> list[str]:
    result = []
    for value in values.tolist():
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return result


def require_shape(name: str, value: np.ndarray, tail: tuple[int, ...]) -> None:
    if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail:
        raise ValueError(f"{name} must have shape [N, {', '.join(map(str, tail))}]")


def first_cycle_stop(rollout: np.lib.npyio.NpzFile) -> int:
    if "dones" not in rollout.files:
        return min(len(rollout["policy_observations"]), len(rollout["actions"]))
    done = np.flatnonzero(np.asarray(rollout["dones"], dtype=bool))
    return int(done[0] + 1) if len(done) else min(
        len(rollout["policy_observations"]), len(rollout["actions"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s53-success-rollout", type=Path, required=True)
    parser.add_argument("--s52-actual-rollout", type=Path, required=True)
    parser.add_argument("--s52-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--actual-frame-min", type=int, default=80)
    parser.add_argument("--actual-frame-max", type=int, default=320)
    parser.add_argument("--ascent-frame-min", type=int, default=60)
    parser.add_argument("--ascent-frame-max", type=int, default=380)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--ascent-retention-weight", type=float, default=2.0)
    parser.add_argument("--actual-weight", type=float, default=8.0)
    parser.add_argument("--position-gain", type=float, default=0.40)
    parser.add_argument("--velocity-gain", type=float, default=0.015)
    parser.add_argument("--max-leg-action-delta", type=float, default=0.35)
    parser.add_argument(
        "--action-scale",
        choices=("s52", "s53"),
        default="s53",
        help="Residual-action scale used by the target task.",
    )
    args = parser.parse_args()

    if args.actual_frame_min > args.actual_frame_max:
        raise ValueError("actual frame range is reversed")
    if args.max_leg_action_delta <= 0.0:
        raise ValueError("max leg action delta must be positive")
    action_scale = S52_ACTION_SCALE if args.action_scale == "s52" else S53_ACTION_SCALE

    s53 = np.load(args.s53_success_rollout, allow_pickle=False)
    actual = np.load(args.s52_actual_rollout, allow_pickle=False)
    reference = np.load(args.s52_reference, allow_pickle=False)
    require_shape("S53 policy_observations", s53["policy_observations"], (148,))
    require_shape("S53 actions", s53["actions"], (27,))
    require_shape("S52 policy_observations", actual["policy_observations"], (148,))
    require_shape("S52 actions", actual["actions"], (27,))
    require_shape("S52 reference joint_pos", reference["joint_pos"], (27,))
    require_shape("S52 reference joint_vel", reference["joint_vel"], (27,))

    reference_names = (
        strings(reference["joint_names"])
        if "joint_names" in reference.files
        else strings(s53["joint_names"])
    )
    actual_names = strings(actual["joint_names"])
    missing = sorted(set(reference_names).difference(actual_names))
    if missing:
        raise ValueError(f"S52 actual rollout is missing policy joints: {missing}")
    actual_indices = np.asarray([actual_names.index(name) for name in reference_names])
    if len(set(reference_names)) != 27:
        raise ValueError("S52 reference joint names must contain 27 unique names")

    s53_count = min(len(s53["actions"]), len(s53["policy_observations"]))
    retention_obs = np.asarray(s53["policy_observations"][:s53_count], dtype=np.float32)
    retention_target = np.asarray(s53["actions"][:s53_count], dtype=np.float32)
    retention_frames = np.asarray(s53["motion_frame"][:s53_count], dtype=np.int64)
    retention_weights = np.full(s53_count, args.retention_weight, dtype=np.float32)
    ascent_mask = (retention_frames >= args.ascent_frame_min) & (
        retention_frames <= args.ascent_frame_max
    )
    retention_weights[ascent_mask] = args.ascent_retention_weight

    stop = first_cycle_stop(actual)
    correction_obs: list[np.ndarray] = []
    correction_target: list[np.ndarray] = []
    correction_frames: list[int] = []
    correction_delta: list[np.ndarray] = []
    skipped_frames: list[int] = []
    for index in range(stop):
        frame = int(actual["motion_frame"][index])
        if frame < args.actual_frame_min or frame > args.actual_frame_max:
            continue
        if frame >= len(reference["joint_pos"]):
            skipped_frames.append(frame)
            continue
        q_actual = np.asarray(actual["joint_pos"][index, actual_indices], dtype=np.float32)
        v_actual = np.asarray(actual["joint_vel"][index, actual_indices], dtype=np.float32)
        q_error = np.asarray(reference["joint_pos"][frame], dtype=np.float32) - q_actual
        v_error = np.asarray(reference["joint_vel"][frame], dtype=np.float32) - v_actual
        raw = args.position_gain * q_error / action_scale
        raw += args.velocity_gain * v_error / action_scale
        delta = np.zeros(27, dtype=np.float32)
        delta[:12] = args.max_leg_action_delta * np.tanh(raw[:12])
        # The base action must be the teacher output at this *actual S52 state*.
        # A same-frame S53 action belongs to a different closed-loop state and can
        # create a large, unsafe label discontinuity even though frame IDs match.
        target = np.asarray(actual["actions"][index], dtype=np.float32).copy()
        target[:12] += delta[:12]
        correction_obs.append(np.asarray(actual["policy_observations"][index], dtype=np.float32))
        correction_target.append(target)
        correction_frames.append(frame)
        correction_delta.append(delta)
    if not correction_obs:
        raise ValueError("no S52 correction samples survived the requested frame range")

    correction_obs_array = np.stack(correction_obs)
    correction_target_array = np.stack(correction_target)
    correction_frames_array = np.asarray(correction_frames, dtype=np.int64)
    correction_delta_array = np.stack(correction_delta)
    observations = np.concatenate((retention_obs, correction_obs_array), axis=0)
    target_actions = np.concatenate((retention_target, correction_target_array), axis=0)
    sample_weights = np.concatenate(
        (
            retention_weights,
            np.full(len(correction_obs_array), args.actual_weight, dtype=np.float32),
        )
    )
    source = np.concatenate(
        (
            np.zeros(len(retention_obs), dtype=np.int8),
            np.ones(len(correction_obs_array), dtype=np.int8),
        )
    )
    motion_frame = np.concatenate((retention_frames, correction_frames_array), axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        policy_observations=observations,
        target_actions=target_actions,
        sample_weights=sample_weights,
        source=source,
        motion_frame=motion_frame,
        correction_action_delta=correction_delta_array,
        correction_motion_frame=correction_frames_array,
        source_labels=np.asarray(["s53_success_retention", "s52_actual_state_correction"]),
    )
    report_data = {
        "status": "dataset_only_not_an_accepted_policy",
        "action_semantics": "residual added to the S52 retargeted joint-position command",
        "correction_base": "model_92099 action recorded at the same actual S52 observation",
        "sources": {
            "s53_success_rollout": {
                "path": str(args.s53_success_rollout.resolve()),
                "sha256": sha256(args.s53_success_rollout),
            },
            "s52_actual_rollout": {
                "path": str(args.s52_actual_rollout.resolve()),
                "sha256": sha256(args.s52_actual_rollout),
            },
            "s52_reference": {
                "path": str(args.s52_reference.resolve()),
                "sha256": sha256(args.s52_reference),
            },
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "samples": int(len(observations)),
        "s53_retention_samples": int(len(retention_obs)),
        "s52_correction_samples": int(len(correction_obs_array)),
        "s52_correction_frame_min": int(np.min(correction_frames_array)),
        "s52_correction_frame_max": int(np.max(correction_frames_array)),
        "skipped_unaligned_frames": sorted(set(skipped_frames)),
        "action_delta_leg_abs_p50": float(np.percentile(np.abs(correction_delta_array[:, :12]), 50)),
        "action_delta_leg_abs_p95": float(np.percentile(np.abs(correction_delta_array[:, :12]), 95)),
        "action_delta_leg_abs_max": float(np.max(np.abs(correction_delta_array[:, :12]))),
        "parameters": {
            "retention_weight": args.retention_weight,
            "ascent_retention_weight": args.ascent_retention_weight,
            "actual_weight": args.actual_weight,
            "position_gain": args.position_gain,
            "velocity_gain": args.velocity_gain,
            "max_leg_action_delta": args.max_leg_action_delta,
            "action_scale": args.action_scale,
        },
    }
    report = args.report or args.output.with_suffix(".json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
