#!/usr/bin/env python3
"""Validate a Lab-trained Kuavo S52 velocity policy in official MuJoCo physics.

The script intentionally has no viewer, ROS, or controller-manager dependency.
It reproduces the S52 Lab policy interface and actuator model while retaining
the official S52 MJCF geometry, inertia, joints, and contacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


CONTROLLED_JOINTS = (
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
    "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
    "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
)

HEAD_JOINTS = ("zhead_1_joint", "zhead_2_joint")

DEFAULT_JOINT_POS = np.asarray(
    [
        0.0, 0.0, -0.25, 0.50, -0.25, 0.0,
        0.0, 0.0, -0.25, 0.50, -0.25, 0.0,
        0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

EFFORT_LIMITS = np.asarray(
    [
        127.0, 71.0, 132.0, 280.0, 57.0, 57.0,
        127.0, 71.0, 132.0, 280.0, 57.0, 57.0,
        102.0,
        66.0, 75.0, 57.0, 75.0, 14.1, 14.1, 14.1,
        66.0, 75.0, 57.0, 75.0, 14.1, 14.1, 14.1,
        1.5, 12.0,
    ],
    dtype=np.float64,
)

VELOCITY_LIMITS = np.asarray(
    [
        23.55, 41.3433, 41.3433, 23.55, 23.0267, 23.0267,
        23.55, 41.3433, 41.3433, 23.55, 23.0267, 23.0267,
        23.55,
        41.3433, 8.3733, 9.9433, 8.3733, 7.3267, 7.3267, 7.3267,
        41.3433, 8.3733, 9.9433, 8.3733, 7.3267, 7.3267, 7.3267,
        5.23, 5.23,
    ],
    dtype=np.float64,
)

EFFORT_WEAKEN_VELOCITY = np.asarray(
    [
        4.71, 8.2687, 8.2687, 4.71, 4.6053, 4.6053,
        4.71, 8.2687, 8.2687, 4.71, 4.6053, 4.6053,
        4.71,
        8.2687, 1.6747, 1.9887, 1.6747, 1.4653, 1.4653, 1.4653,
        8.2687, 1.6747, 1.9887, 1.6747, 1.4653, 1.4653, 1.4653,
        1.046, 1.046,
    ],
    dtype=np.float64,
)

KP = np.asarray(
    [
        100.0, 100.0, 80.0, 80.0, 30.0, 30.0,
        100.0, 100.0, 80.0, 80.0, 30.0, 30.0,
        30.0,
        30.0, 30.0, 15.0, 30.0, 15.0, 15.0, 15.0,
        30.0, 30.0, 15.0, 30.0, 15.0, 15.0, 15.0,
        10.0, 10.0,
    ],
    dtype=np.float64,
)

KD = np.asarray(
    [
        5.0, 5.0, 5.0, 6.0, 7.5, 7.5,
        5.0, 5.0, 5.0, 6.0, 7.5, 7.5,
        3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        1.0, 1.0,
    ],
    dtype=np.float64,
)

ARMATURE = np.asarray(
    [
        0.05, 0.025, 0.025, 0.05, 0.05, 0.05,
        0.05, 0.025, 0.025, 0.05, 0.05, 0.05,
        0.025,
        0.025, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
        0.025, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
        0.01, 0.01,
    ],
    dtype=np.float64,
)

FRICTION_STATIC = np.asarray(
    [
        1.0, 0.5, 0.5, 1.0, 0.2, 0.2,
        1.0, 0.5, 0.5, 1.0, 0.2, 0.2,
        0.2,
        0.5, 0.3, 0.2, 0.3, 0.1, 0.1, 0.1,
        0.5, 0.3, 0.2, 0.3, 0.1, 0.1, 0.1,
        0.1, 0.1,
    ],
    dtype=np.float64,
)

FRICTION_DYNAMIC = np.asarray(
    [
        *([0.2] * 12), 0.2,
        *([0.1] * 14),
        0.1, 0.1,
    ],
    dtype=np.float64,
)

ACTION_SCALE = 0.25
HISTORY_LENGTH = 5
GAIT_PERIOD_S = 0.8
FRICTION_ACTIVATION_VEL = 0.1

MIRROR_JOINT_PERMUTATION = np.asarray(
    [
        6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5,
        12,
        20, 21, 22, 23, 24, 25, 26,
        13, 14, 15, 16, 17, 18, 19,
    ],
    dtype=np.int32,
)

MIRROR_JOINT_SIGN = np.asarray(
    [
        -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
        -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
        -1.0,
        1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
        1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mode", choices=("stand", "walk"), required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command_x", type=float, default=None)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--root_height", type=float, default=0.955)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--zero_action", action="store_true")
    parser.add_argument("--contact_time_constant", type=float, default=None)
    parser.add_argument("--symmetry_ensemble", action="store_true")
    parser.add_argument("--symmetry_blend", type=float, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (quat[..., index] for index in range(4))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack((roll, pitch, yaw), axis=-1)


def object_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_type: mujoco.mjtObj,
    object_id: int,
    local: bool,
) -> np.ndarray:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, object_type, object_id, velocity, int(local))
    return velocity


def build_ids(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    all_joints = CONTROLLED_JOINTS + HEAD_JOINTS
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in all_joints],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise ValueError("The official S52 MJCF is missing one or more expected joints")
    qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int32)
    dof_ids = model.jnt_dofadr[joint_ids].astype(np.int32)
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            for name in all_joints
        ],
        dtype=np.int32,
    )
    if np.any(actuator_ids < 0):
        raise ValueError("The official S52 MJCF is missing one or more expected motors")
    return {
        "joint": joint_ids,
        "qpos": qpos_ids,
        "dof": dof_ids,
        "actuator": actuator_ids,
        "base": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
        "left_foot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_l6_link"),
        "right_foot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_r6_link"),
    }


def configure_lab_actuator_parity(model: mujoco.MjModel, ids: dict[str, np.ndarray | int]) -> None:
    joint_ids = np.asarray(ids["joint"])
    dof_ids = np.asarray(ids["dof"])
    actuator_ids = np.asarray(ids["actuator"])
    model.dof_armature[dof_ids] = ARMATURE
    model.dof_damping[dof_ids] = 0.2
    model.dof_frictionloss[dof_ids] = 0.0
    # Lab clips the explicit actuator output. Disable the older MJCF motor and
    # joint torque clips, then apply the current S52 limits in Python.
    model.actuator_ctrllimited[actuator_ids] = 0
    model.jnt_actfrclimited[joint_ids] = 0


def actuator_torque(target: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    effort = (
        KP * (target - joint_pos)
        - KD * joint_vel
        - FRICTION_STATIC * np.tanh(joint_vel / FRICTION_ACTIVATION_VEL)
        - FRICTION_DYNAMIC * joint_vel
    )
    linear_limit = np.maximum(
        0.0,
        (-2.0 * EFFORT_LIMITS / VELOCITY_LIMITS)
        * (np.abs(joint_vel) - VELOCITY_LIMITS / 2.0)
        + EFFORT_LIMITS,
    )
    max_effort = np.where(
        np.abs(joint_vel) < EFFORT_WEAKEN_VELOCITY,
        EFFORT_LIMITS,
        linear_limit,
    )
    same_direction = joint_vel * effort > 0.0
    max_effort = np.where(same_direction, max_effort, EFFORT_LIMITS)
    return np.clip(effort, -max_effort, max_effort)


def command(mode: str, command_x: float, step: int, control_dt: float) -> np.ndarray:
    if mode == "stand" or abs(command_x) <= 1.0e-6:
        return np.zeros(3, dtype=np.float32)
    phase = ((step * control_dt) % GAIT_PERIOD_S) / GAIT_PERIOD_S
    return np.asarray(
        [command_x, math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        dtype=np.float32,
    )


def mirror_joint_vector(values: np.ndarray) -> np.ndarray:
    """Mirror a 27-DoF vector across the sagittal plane."""

    values = np.asarray(values, dtype=np.float32)
    return MIRROR_JOINT_SIGN * values[MIRROR_JOINT_PERMUTATION]


def mirror_policy_observation(observation: np.ndarray) -> np.ndarray:
    """Mirror the term-major five-frame S52 policy observation."""

    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (450,):
        raise ValueError(f"Expected a 450-D observation, got {observation.shape}")
    mirrored = observation.copy()
    term_layout = (
        (0, 3, np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)),
        (15, 3, np.asarray([1.0, -1.0, 1.0], dtype=np.float32)),
        # The two reused command slots are sin/cos gait phase. Swapping the
        # legs requires a half-cycle shift, which negates both clock values.
        (30, 3, np.asarray([1.0, -1.0, -1.0], dtype=np.float32)),
    )
    for start, width, sign in term_layout:
        frames = observation[start : start + HISTORY_LENGTH * width].reshape(HISTORY_LENGTH, width)
        mirrored[start : start + HISTORY_LENGTH * width] = (frames * sign).reshape(-1)
    for start in (45, 180, 315):
        frames = observation[start : start + HISTORY_LENGTH * 27].reshape(HISTORY_LENGTH, 27)
        mirrored[start : start + HISTORY_LENGTH * 27] = np.stack(
            [mirror_joint_vector(frame) for frame in frames], axis=0
        ).reshape(-1)
    return mirrored


def frame_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, np.ndarray | int],
    gait_command: np.ndarray,
    last_action: np.ndarray,
) -> tuple[np.ndarray, ...]:
    base_id = int(ids["base"])
    qpos_ids = np.asarray(ids["qpos"])[:27]
    dof_ids = np.asarray(ids["dof"])[:27]
    base_velocity_local = object_velocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, local=True
    )
    rotation_world_from_body = data.xmat[base_id].reshape(3, 3)
    projected_gravity = rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0])
    return (
        (0.2 * base_velocity_local[:3]).astype(np.float32),
        projected_gravity.astype(np.float32),
        gait_command.astype(np.float32),
        (data.qpos[qpos_ids] - DEFAULT_JOINT_POS).astype(np.float32),
        (0.05 * data.qvel[dof_ids]).astype(np.float32),
        last_action.astype(np.float32),
    )


def foot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_body: int,
    right_body: int,
) -> np.ndarray:
    forces = np.zeros(2, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        foot_index = None
        if body_1 == left_body or body_2 == left_body:
            foot_index = 0
        elif body_1 == right_body or body_2 == right_body:
            foot_index = 1
        if foot_index is None:
            continue
        contact_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, contact_force)
        forces[foot_index] += abs(contact_force[0])
    return forces


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    symmetry_blend = 0.5 if args.symmetry_ensemble else args.symmetry_blend
    if symmetry_blend is None:
        symmetry_blend = 0.0
    if not 0.0 <= symmetry_blend <= 1.0:
        raise ValueError("symmetry_blend must be between 0 and 1")
    xml_path = args.xml.resolve()
    policy_path = args.policy.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command_x = args.command_x
    if command_x is None:
        command_x = 0.0 if args.mode == "stand" else 0.25

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError(
            f"Unexpected S52 dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    ratio = args.control_dt / float(model.opt.timestep)
    substeps = int(round(ratio))
    if substeps <= 0 or not math.isclose(substeps, ratio, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("control_dt must be an integer multiple of the MJCF timestep")

    ids = build_ids(model)
    configure_lab_actuator_parity(model, ids)
    if args.contact_time_constant is not None:
        if args.contact_time_constant <= 0.0:
            raise ValueError("contact_time_constant must be positive")
        model.geom_solref[:, 0] = float(args.contact_time_constant)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.asarray([0.0, 0.0, args.root_height])
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qpos[np.asarray(ids["qpos"])[:27]] = DEFAULT_JOINT_POS
    data.qpos[np.asarray(ids["qpos"])[27:]] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    policy = ort.InferenceSession(
        str(policy_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    input_meta = policy.get_inputs()[0]
    output_meta = policy.get_outputs()[0]
    if input_meta.shape[-1] != 450 or output_meta.shape[-1] != 27:
        raise ValueError(
            f"Unexpected policy shape: input={input_meta.shape}, output={output_meta.shape}"
        )

    history: list[deque[np.ndarray]] | None = None
    last_action = np.zeros(27, dtype=np.float32)
    initial_root_pos = data.qpos[:3].copy()
    initial_joint_pos = data.qpos[np.asarray(ids["qpos"])[:27]].copy()

    root_pos = []
    root_quat = []
    root_lin_vel_w = []
    root_lin_vel_b = []
    root_ang_vel_b = []
    joint_pos = []
    joint_vel = []
    actions = []
    applied_torque = []
    commands = []
    foot_pos = []
    foot_lin_vel = []
    foot_contact_force = []
    policy_observations = []
    failures = []

    for step in range(args.steps):
        gait_command = command(args.mode, float(command_x), step, args.control_dt)
        terms = frame_terms(model, data, ids, gait_command, last_action)
        if history is None:
            history = [deque((term.copy() for _ in range(HISTORY_LENGTH)), maxlen=HISTORY_LENGTH) for term in terms]
        else:
            for term_history, term in zip(history, terms, strict=True):
                term_history.append(term)
        observation = np.concatenate(
            [np.concatenate(tuple(term_history), axis=0) for term_history in history], axis=0
        ).astype(np.float32, copy=False)
        policy_observations.append(observation.copy())
        if args.zero_action:
            action = np.zeros(27, dtype=np.float32)
        else:
            action = policy.run([output_meta.name], {input_meta.name: observation[None, :]})[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if symmetry_blend > 0.0:
                mirrored_observation = mirror_policy_observation(observation)
                mirrored_action = policy.run(
                    [output_meta.name], {input_meta.name: mirrored_observation[None, :]}
                )[0]
                mirrored_action = mirror_joint_vector(np.asarray(mirrored_action).reshape(-1))
                action = (1.0 - symmetry_blend) * action + symmetry_blend * mirrored_action
        if action.shape != (27,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Invalid action at step {step}: shape={action.shape}")
        target = np.concatenate(
            (DEFAULT_JOINT_POS + ACTION_SCALE * action.astype(np.float64), np.zeros(2))
        )

        interval_contact = np.zeros(2, dtype=np.float64)
        interval_peak_contact = np.zeros(2, dtype=np.float64)
        interval_torque = np.zeros(29, dtype=np.float64)
        for _ in range(substeps):
            current_pos = data.qpos[np.asarray(ids["qpos"])]
            current_vel = data.qvel[np.asarray(ids["dof"])]
            torque = actuator_torque(target, current_pos, current_vel)
            data.ctrl[np.asarray(ids["actuator"])] = torque
            interval_torque = torque.copy()
            mujoco.mj_step(model, data)
            contact_force = foot_contacts(
                model, data, int(ids["left_foot"]), int(ids["right_foot"])
            )
            interval_contact = np.maximum(interval_contact, contact_force)
            interval_peak_contact = np.maximum(interval_peak_contact, contact_force)

        base_local = object_velocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(ids["base"]), local=True
        )
        base_world = object_velocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(ids["base"]), local=False
        )
        left_world = object_velocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(ids["left_foot"]), local=False
        )
        right_world = object_velocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(ids["right_foot"]), local=False
        )

        root_pos.append(data.qpos[:3].copy())
        root_quat.append(data.qpos[3:7].copy())
        root_lin_vel_w.append(base_world[3:].copy())
        root_lin_vel_b.append(base_local[3:].copy())
        root_ang_vel_b.append(base_local[:3].copy())
        joint_pos.append(data.qpos[np.asarray(ids["qpos"])[:27]].copy())
        joint_vel.append(data.qvel[np.asarray(ids["dof"])[:27]].copy())
        actions.append(action.copy())
        applied_torque.append(interval_torque[:27].copy())
        commands.append(np.asarray([command_x, 0.0, 0.0], dtype=np.float64))
        foot_pos.append(
            np.stack(
                (data.xpos[int(ids["left_foot"])], data.xpos[int(ids["right_foot"])]),
                axis=0,
            )
        )
        foot_lin_vel.append(np.stack((left_world[3:], right_world[3:]), axis=0))
        force_vectors = np.zeros((2, 3), dtype=np.float64)
        force_vectors[:, 2] = interval_peak_contact
        foot_contact_force.append(force_vectors)
        last_action = action

        rpy = quat_wxyz_to_rpy(data.qpos[3:7])
        failed = (
            not np.all(np.isfinite(data.qpos))
            or data.qpos[2] < (0.82 if args.mode == "stand" else 0.75)
            or max(abs(float(rpy[0])), abs(float(rpy[1]))) > math.radians(15.0)
        )
        failures.append(bool(failed))
        if failed:
            break

    arrays = {
        "root_pos": np.asarray(root_pos, dtype=np.float32),
        "root_quat_wxyz": np.asarray(root_quat, dtype=np.float32),
        "root_lin_vel_w": np.asarray(root_lin_vel_w, dtype=np.float32),
        "root_lin_vel_b": np.asarray(root_lin_vel_b, dtype=np.float32),
        "root_ang_vel_b": np.asarray(root_ang_vel_b, dtype=np.float32),
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "applied_torque": np.asarray(applied_torque, dtype=np.float32),
        "commands": np.asarray(commands, dtype=np.float32),
        "foot_pos": np.asarray(foot_pos, dtype=np.float32),
        "foot_lin_vel": np.asarray(foot_lin_vel, dtype=np.float32),
        "foot_contact_force": np.asarray(foot_contact_force, dtype=np.float32),
        "policy_observations": np.asarray(policy_observations, dtype=np.float32),
        "failures": np.asarray(failures, dtype=bool),
    }
    if len(arrays["failures"]) == 0:
        raise RuntimeError("No MuJoCo samples were recorded")

    np.savez_compressed(
        output_path,
        policy=np.asarray(str(policy_path)),
        policy_sha256=np.asarray(sha256(policy_path)),
        xml=np.asarray(str(xml_path)),
        xml_sha256=np.asarray(sha256(xml_path)),
        mode=np.asarray(args.mode),
        zero_action=np.asarray(args.zero_action),
        symmetry_ensemble=np.asarray(args.symmetry_ensemble),
        symmetry_blend=np.asarray(symmetry_blend),
        seed=np.asarray(args.seed),
        physics_dt=np.asarray(float(model.opt.timestep)),
        control_dt=np.asarray(float(args.control_dt)),
        controlled_joint_names=np.asarray(CONTROLLED_JOINTS),
        effort_limits=EFFORT_LIMITS[:27].astype(np.float32),
        kp=KP[:27].astype(np.float32),
        kd=KD[:27].astype(np.float32),
        action_scale=np.asarray(ACTION_SCALE),
        initial_root_pos=initial_root_pos.astype(np.float32),
        initial_joint_pos=initial_joint_pos.astype(np.float32),
        default_joint_pos=DEFAULT_JOINT_POS.astype(np.float32),
        **arrays,
    )

    failure_indices = np.flatnonzero(arrays["failures"])
    first_failure = int(failure_indices[0]) if failure_indices.size else None
    valid_count = first_failure if first_failure is not None else len(arrays["failures"])
    valid_count = max(valid_count, 1)
    valid = slice(0, valid_count)
    duration_s = float(valid_count * args.control_dt)
    rpy = quat_wxyz_to_rpy(arrays["root_quat_wxyz"][valid])
    roll_pitch_deg = np.rad2deg(np.abs(rpy[:, :2]))
    root_xy = arrays["root_pos"][valid, :2]
    displacement_xy = root_xy - root_xy[0]
    command_xy = arrays["commands"][valid, :2]
    velocity_xy = arrays["root_lin_vel_b"][valid, :2]
    velocity_error = np.linalg.norm(velocity_xy - command_xy, axis=-1)
    force_norm = np.linalg.norm(arrays["foot_contact_force"][valid], axis=-1)
    contact_mask = force_norm > 10.0
    foot_speed_xy = np.linalg.norm(arrays["foot_lin_vel"][valid, :, :2], axis=-1)
    sliding_speed = foot_speed_xy[contact_mask]
    torque_ratio = np.abs(arrays["applied_torque"][valid]) / EFFORT_LIMITS[None, :27]

    metrics = {
        "simulator": "MuJoCo",
        "mode": args.mode,
        "zero_action": bool(args.zero_action),
        "symmetry_ensemble": bool(args.symmetry_ensemble),
        "symmetry_blend": float(symmetry_blend),
        "policy": str(policy_path),
        "policy_sha256": sha256(policy_path),
        "xml": str(xml_path),
        "xml_sha256": sha256(xml_path),
        "seed": int(args.seed),
        "requested_steps": int(args.steps),
        "valid_steps_before_first_failure": int(valid_count),
        "duration_before_first_failure_s": duration_s,
        "reset_or_failure_count": int(failure_indices.size),
        "first_failure_step": first_failure,
        "physics_dt_s": float(model.opt.timestep),
        "control_dt_s": float(args.control_dt),
        "substeps": int(substeps),
        "contact_time_constant_s": float(model.geom_solref[0, 0]),
        "model_total_mass_kg": float(np.sum(model.body_mass)),
        "root_height_mean_m": float(np.mean(arrays["root_pos"][valid, 2])),
        "root_height_min_m": float(np.min(arrays["root_pos"][valid, 2])),
        "root_height_max_m": float(np.max(arrays["root_pos"][valid, 2])),
        "roll_abs_p95_deg": percentile(roll_pitch_deg[:, 0], 95.0),
        "pitch_abs_p95_deg": percentile(roll_pitch_deg[:, 1], 95.0),
        "roll_pitch_abs_max_deg": float(np.max(roll_pitch_deg)),
        "xy_endpoint_displacement_m": float(np.linalg.norm(displacement_xy[-1])),
        "xy_max_displacement_m": float(np.max(np.linalg.norm(displacement_xy, axis=-1))),
        "forward_displacement_m": float(displacement_xy[-1, 0]),
        "command_forward_mean_mps": float(np.mean(command_xy[:, 0])),
        "forward_velocity_mean_mps": float(np.mean(velocity_xy[:, 0])),
        "velocity_tracking_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "left_contact_duty": float(np.mean(contact_mask[:, 0])),
        "right_contact_duty": float(np.mean(contact_mask[:, 1])),
        "double_support_duty": float(np.mean(np.all(contact_mask, axis=1))),
        "no_support_fraction": float(np.mean(~np.any(contact_mask, axis=1))),
        "foot_slide_contact_p95_mps": percentile(sliding_speed, 95.0),
        "peak_foot_contact_force_n": float(np.max(force_norm)),
        "applied_torque_ratio_p95": percentile(torque_ratio, 95.0),
        "applied_torque_ratio_max": float(np.max(torque_ratio)),
    }
    if args.mode == "stand":
        checks = {
            "no_failure_60s": metrics["reset_or_failure_count"] == 0 and duration_s >= 59.9,
            "height_safe": metrics["root_height_min_m"] >= 0.82,
            "orientation_stable": max(metrics["roll_abs_p95_deg"], metrics["pitch_abs_p95_deg"]) <= 10.0,
            "low_xy_drift": metrics["xy_max_displacement_m"] <= 0.20,
            "low_velocity_error": metrics["velocity_tracking_rmse_mps"] <= 0.10,
        }
    else:
        expected_progress = max(0.0, metrics["command_forward_mean_mps"] * duration_s)
        checks = {
            "no_failure_60s": metrics["reset_or_failure_count"] == 0 and duration_s >= 59.9,
            "height_safe": metrics["root_height_min_m"] >= 0.75,
            "orientation_stable": max(metrics["roll_abs_p95_deg"], metrics["pitch_abs_p95_deg"]) <= 15.0,
            "forward_progress": metrics["forward_displacement_m"] >= 0.70 * expected_progress,
            "velocity_tracking": metrics["velocity_tracking_rmse_mps"] <= 0.15,
        }
    metrics["checks"] = checks
    metrics["passed"] = bool(all(checks.values()))
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
