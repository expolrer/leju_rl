#!/usr/bin/env python3
"""Validate an S52 stand/walk policy with deterministic headless Isaac physics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Validate a Kuavo S52 velocity policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--zero_action", action="store_true")
parser.add_argument("--export_dir", type=Path, default=None)
parser.add_argument(
    "--export_only",
    action="store_true",
    help="Load and export the checkpoint, then exit before stepping physics.",
)
parser.add_argument("--root_height", type=float, default=None)
parser.add_argument("--seed", type=int, default=131)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from leju_robot.rsl_rl_extensions.utils.exporter import export_policy_as_onnx
from leju_robot.rsl_rl_extensions.utils.on_policy_runner import RobotOnPolicyRunner  # noqa: F401


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

EFFORT_LIMITS = np.asarray(
    [
        127.0, 71.0, 132.0, 280.0, 57.0, 57.0,
        127.0, 71.0, 132.0, 280.0, 57.0, 57.0,
        102.0,
        66.0, 75.0, 57.0, 75.0, 14.1, 14.1, 14.1,
        66.0, 75.0, 57.0, 75.0, 14.1, 14.1, 14.1,
    ],
    dtype=np.float32,
)


def cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternions to roll, pitch, yaw in radians."""

    w, x, y, z = (quat[:, index] for index in range(4))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack((roll, pitch, yaw), axis=-1)


def percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.root_height is not None:
        init_pos = tuple(env_cfg.scene.robot.init_state.pos)
        env_cfg.scene.robot.init_state.pos = (
            init_pos[0],
            init_pos[1],
            float(args_cli.root_height),
        )
    # The validator owns the horizon. Time-out resets must not be mistaken for falls.
    env_cfg.episode_length_s = 1.0e9
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    raw_env = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)

    checkpoint = None
    runner = None
    if not args_cli.zero_action:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner_class = eval(agent_cfg.runner_class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=raw_env.device)
        if args_cli.export_dir is not None:
            export_dir = args_cli.export_dir.resolve()
            export_dir.mkdir(parents=True, exist_ok=True)
            export_policy_as_jit(
                runner.alg.policy,
                runner.obs_normalizer,
                path=str(export_dir),
                filename="policy.pt",
            )
            export_policy_as_onnx(
                raw_env,
                runner.alg.policy,
                normalizer=runner.obs_normalizer,
                path=str(export_dir),
                filename="policy.onnx",
            )
        if args_cli.export_only:
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "export_dir": str(args_cli.export_dir.resolve()) if args_cli.export_dir else None,
                        "export_only": True,
                    },
                    indent=2,
                )
            )
            env.close()
            return
    else:
        action_dim = int(raw_env.action_manager.total_action_dim)

        def policy(_obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((1, action_dim), dtype=torch.float32, device=raw_env.device)

    obs, _ = env.get_observations()
    robot = raw_env.scene["robot"]
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    command_term = raw_env.command_manager.get_term("base_velocity")
    env_origin = raw_env.scene.env_origins[0]

    controlled_ids = [robot.joint_names.index(name) for name in CONTROLLED_JOINTS]
    foot_names = ("leg_l6_link", "leg_r6_link")
    foot_body_ids = [robot.body_names.index(name) for name in foot_names]
    sensor_body_names = list(contact_sensor.body_names)
    foot_sensor_ids = [sensor_body_names.index(name) for name in foot_names]
    initial_root_pos = cpu(robot.data.root_pos_w[0] - env_origin)
    initial_joint_pos = cpu(robot.data.joint_pos[0, controlled_ids])
    default_joint_pos = cpu(robot.data.default_joint_pos[0, controlled_ids])

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
    rewards = []
    dones = []
    foot_pos = []
    foot_lin_vel = []
    foot_contact_force = []
    policy_observations = []

    for _ in range(int(args_cli.steps)):
        policy_observations.append(cpu(obs[0]))
        with torch.inference_mode():
            action = policy(obs)
            obs, reward, done, _ = env.step(action)

        command = command_term.command[0]
        forces = contact_sensor.data.net_forces_w_history[0, -1, foot_sensor_ids]
        root_pos.append(cpu(robot.data.root_pos_w[0] - env_origin))
        root_quat.append(cpu(robot.data.root_quat_w[0]))
        root_lin_vel_w.append(cpu(robot.data.root_lin_vel_w[0]))
        root_lin_vel_b.append(cpu(robot.data.root_lin_vel_b[0]))
        root_ang_vel_b.append(cpu(robot.data.root_ang_vel_b[0]))
        joint_pos.append(cpu(robot.data.joint_pos[0, controlled_ids]))
        joint_vel.append(cpu(robot.data.joint_vel[0, controlled_ids]))
        actions.append(cpu(action[0]))
        applied_torque.append(cpu(robot.data.applied_torque[0, controlled_ids]))
        commands.append(cpu(command))
        rewards.append(float(reward[0].item()))
        dones.append(bool(done[0].item()))
        foot_pos.append(cpu(robot.data.body_pos_w[0, foot_body_ids] - env_origin))
        foot_lin_vel.append(cpu(robot.data.body_lin_vel_w[0, foot_body_ids]))
        foot_contact_force.append(cpu(forces))

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
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "foot_pos": np.asarray(foot_pos, dtype=np.float32),
        "foot_lin_vel": np.asarray(foot_lin_vel, dtype=np.float32),
        "foot_contact_force": np.asarray(foot_contact_force, dtype=np.float32),
        "policy_observations": np.asarray(policy_observations, dtype=np.float32),
    }

    output = args_cli.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        checkpoint=np.asarray(checkpoint if checkpoint else "zero_action"),
        task=np.asarray(args_cli.task),
        seed=np.asarray(args_cli.seed),
        dt=np.asarray(float(raw_env.step_dt)),
        controlled_joint_names=np.asarray(CONTROLLED_JOINTS),
        effort_limits=EFFORT_LIMITS,
        foot_names=np.asarray(foot_names),
        initial_root_pos=initial_root_pos,
        initial_joint_pos=initial_joint_pos,
        default_joint_pos=default_joint_pos,
        **arrays,
    )

    done_indices = np.flatnonzero(arrays["dones"])
    first_done = int(done_indices[0]) if done_indices.size else None
    valid_count = first_done if first_done is not None else len(arrays["dones"])
    valid_count = max(valid_count, 1)
    valid = slice(0, valid_count)
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
    torque_ratio = np.abs(arrays["applied_torque"][valid]) / EFFORT_LIMITS[None, :]

    mode = "stand" if "Stand" in args_cli.task else "walk"
    duration_s = float(valid_count * raw_env.step_dt)
    metrics = {
        "task": args_cli.task,
        "mode": mode,
        "policy_source": "zero_action" if args_cli.zero_action else str(checkpoint),
        "seed": int(args_cli.seed),
        "requested_steps": int(args_cli.steps),
        "valid_steps_before_first_reset": int(valid_count),
        "duration_before_first_reset_s": duration_s,
        "reset_count": int(done_indices.size),
        "first_reset_step": first_done,
        "mean_reward": float(np.mean(arrays["rewards"][valid])),
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

    if mode == "stand":
        checks = {
            "no_reset_60s": metrics["reset_count"] == 0 and duration_s >= 59.9,
            "height_safe": metrics["root_height_min_m"] >= 0.82,
            "orientation_stable": max(metrics["roll_abs_p95_deg"], metrics["pitch_abs_p95_deg"]) <= 10.0,
            "low_xy_drift": metrics["xy_max_displacement_m"] <= 0.20,
            "low_velocity_error": metrics["velocity_tracking_rmse_mps"] <= 0.10,
        }
    else:
        expected_progress = max(0.0, metrics["command_forward_mean_mps"] * duration_s)
        checks = {
            "no_reset_60s": metrics["reset_count"] == 0 and duration_s >= 59.9,
            "height_safe": metrics["root_height_min_m"] >= 0.75,
            "orientation_stable": max(metrics["roll_abs_p95_deg"], metrics["pitch_abs_p95_deg"]) <= 15.0,
            "forward_progress": metrics["forward_displacement_m"] >= 0.70 * expected_progress,
            "velocity_tracking": metrics["velocity_tracking_rmse_mps"] <= 0.15,
        }
    metrics["checks"] = checks
    metrics["passed"] = bool(all(checks.values()))

    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
