#!/usr/bin/env python3
"""Probe whether an S52 PhysX rollout state can be replayed exactly enough for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--rollout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", type=int, nargs="+", default=[160, 180, 207, 220, 240])
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
from isaaclab_tasks.utils import parse_env_cfg

import leju_robot.tasks.tracking.mdp as tracking_mdp


OBSERVATION_SLICES = {
    "command": slice(0, 54),
    "target_z": slice(54, 55),
    "anchor_orientation": slice(55, 61),
    "projected_gravity": slice(61, 64),
    "base_angular_velocity": slice(64, 67),
    "joint_position": slice(67, 94),
    "joint_velocity": slice(94, 121),
    "previous_action": slice(121, 148),
}


def max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)))


def quaternion_max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    direct = np.max(np.abs(actual - expected), axis=-1)
    negated = np.max(np.abs(actual + expected), axis=-1)
    return float(np.max(np.minimum(direct, negated)))


def tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def main() -> None:
    rollout = np.load(args_cli.rollout, allow_pickle=True)
    required = {
        "joint_names",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "motion_frame",
        "actions",
        "policy_observations",
        "all_body_pos",
        "all_body_quat",
        "contact_body_names",
        "contact_force_w",
    }
    missing = sorted(required.difference(rollout.files))
    if missing:
        raise KeyError(f"rollout is missing required arrays: {missing}")
    if rollout["policy_observations"].shape[1] != 148:
        raise ValueError("expected 148 policy observations")
    if rollout["actions"].shape[1] != 27:
        raise ValueError("expected 27 policy actions")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42

    motion_cfg = env_cfg.commands.motion
    env_cfg.commands.motion = tracking_mdp.MotionCommandPlayCfg(
        motion_file=motion_cfg.motion_file,
        asset_name=motion_cfg.asset_name,
        anchor_body=motion_cfg.anchor_body,
        resampling_time_range=motion_cfg.resampling_time_range,
        debug_vis=False,
        pose_range=motion_cfg.pose_range,
        velocity_range=motion_cfg.velocity_range,
        joint_position_range=motion_cfg.joint_position_range,
        body_names=motion_cfg.body_names,
        start_hold_steps=motion_cfg.start_hold_steps,
        end_hold_steps=motion_cfg.end_hold_steps,
    )

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    raw_env = gym_env.unwrapped
    robot = raw_env.scene["robot"]
    command = raw_env.command_manager.get_term("motion")
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    env_origin = raw_env.scene.env_origins[0]

    saved_joint_names = [str(name) for name in rollout["joint_names"]]
    robot_joint_names = list(robot.joint_names)
    if set(saved_joint_names) != set(robot_joint_names):
        missing_robot = sorted(set(saved_joint_names).difference(robot_joint_names))
        missing_rollout = sorted(set(robot_joint_names).difference(saved_joint_names))
        raise ValueError(
            f"joint-name mismatch: only in rollout={missing_robot}, only in robot={missing_rollout}"
        )
    saved_joint_index = {name: index for index, name in enumerate(saved_joint_names)}
    saved_to_robot = np.asarray(
        [saved_joint_index[name] for name in robot_joint_names], dtype=np.int64
    )
    saved_body_names = [str(name) for name in rollout["all_body_names"]]
    robot_body_names = list(robot.body_names)
    if saved_body_names != robot_body_names:
        raise ValueError("rollout and live robot body order differ")

    saved_contact_names = [str(name) for name in rollout["contact_body_names"]]
    sensor_contact_names = list(contact_sensor.body_names)
    shared_contact_names = [name for name in sensor_contact_names if name in saved_contact_names]
    sensor_contact_indices = [sensor_contact_names.index(name) for name in shared_contact_names]
    saved_contact_indices = [saved_contact_names.index(name) for name in shared_contact_names]

    results: list[dict[str, object]] = []
    for row in args_cli.frames:
        if row <= 0 or row >= len(rollout["actions"]):
            raise ValueError(f"frame {row} must be in [1, {len(rollout['actions']) - 1}]")
        frame = int(rollout["motion_frame"][row])
        next_frame = int(rollout["motion_frame"][row + 1])
        if next_frame != frame + 1:
            raise ValueError(f"frame {row} crosses a reset or hold: {frame} -> {next_frame}")

        gym_env.reset()
        joint_pos = rollout["joint_pos"][row, saved_to_robot]
        joint_vel = rollout["joint_vel"][row, saved_to_robot]
        root_state = np.concatenate(
            (
                rollout["root_pos"][row] + env_origin.detach().cpu().numpy(),
                rollout["root_quat"][row],
                rollout["root_lin_vel"][row],
                rollout["root_ang_vel"][row],
            )
        )
        robot.write_joint_state_to_sim(
            tensor(joint_pos[None, :], raw_env.device),
            tensor(joint_vel[None, :], raw_env.device),
            env_ids=torch.tensor([0], dtype=torch.long, device=raw_env.device),
        )
        robot.write_root_state_to_sim(
            tensor(root_state[None, :], raw_env.device),
            env_ids=torch.tensor([0], dtype=torch.long, device=raw_env.device),
        )
        raw_env.sim.forward()
        raw_env.scene.update(0.0)

        command.time_steps[0] = frame - 1
        command.start_time[0] = command.cfg.start_hold_steps
        command.out_time[0] = 0
        command._update_command()
        if int(command.time_steps[0].item()) != frame:
            raise RuntimeError("command frame did not advance to the replay frame")

        previous_action = tensor(
            rollout["policy_observations"][row, OBSERVATION_SLICES["previous_action"]][None, :],
            raw_env.device,
        )
        raw_env.action_manager._action[0] = previous_action[0]
        raw_env.action_manager._prev_action[0] = previous_action[0]

        replay_observation = (
            raw_env.observation_manager.compute()["policy"][0].detach().cpu().numpy()
        )
        expected_observation = rollout["policy_observations"][row]
        observation_errors = {
            name: max_abs(replay_observation[index], expected_observation[index])
            for name, index in OBSERVATION_SLICES.items()
        }

        pre_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy() - env_origin.detach().cpu().numpy()
        pre_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()[
            [robot_joint_names.index(name) for name in saved_joint_names]
        ]
        pre_body_pos = robot.data.body_pos_w[0].detach().cpu().numpy() - env_origin.detach().cpu().numpy()
        pre_body_quat = robot.data.body_quat_w[0].detach().cpu().numpy()

        next_observation, _, terminated, truncated, _ = gym_env.step(
            tensor(rollout["actions"][row][None, :], raw_env.device)
        )
        replay_next_observation = next_observation["policy"][0].detach().cpu().numpy()
        expected_next_observation = rollout["policy_observations"][row + 1]
        next_observation_errors = {
            name: max_abs(replay_next_observation[index], expected_next_observation[index])
            for name, index in OBSERVATION_SLICES.items()
        }
        next_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy() - env_origin.detach().cpu().numpy()
        next_root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        next_root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        next_root_ang_vel = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
        next_joint_pos_robot = robot.data.joint_pos[0].detach().cpu().numpy()
        next_joint_vel_robot = robot.data.joint_vel[0].detach().cpu().numpy()
        robot_to_saved = [robot_joint_names.index(name) for name in saved_joint_names]
        next_body_pos = robot.data.body_pos_w[0].detach().cpu().numpy() - env_origin.detach().cpu().numpy()
        next_body_quat = robot.data.body_quat_w[0].detach().cpu().numpy()
        replay_contact = (
            contact_sensor.data.net_forces_w_history[0, -1]
            .detach()
            .cpu()
            .numpy()[sensor_contact_indices]
        )
        expected_contact = rollout["contact_force_w"][row + 1, saved_contact_indices]

        results.append(
            {
                "row": row,
                "motion_frame": frame,
                "next_motion_frame": int(command.time_steps[0].item()),
                "prewrite": {
                    "root_position_max_abs": max_abs(pre_root_pos, rollout["root_pos"][row]),
                    "joint_position_max_abs": max_abs(pre_joint_pos, rollout["joint_pos"][row]),
                    "body_position_max_abs": max_abs(pre_body_pos, rollout["all_body_pos"][row]),
                    "body_quaternion_max_abs_sign_invariant": quaternion_max_abs(
                        pre_body_quat, rollout["all_body_quat"][row]
                    ),
                    "observation_max_abs": max_abs(replay_observation, expected_observation),
                    "observation_group_max_abs": observation_errors,
                },
                "one_step": {
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "root_position_max_abs": max_abs(next_root_pos, rollout["root_pos"][row + 1]),
                    "root_quaternion_max_abs_sign_invariant": quaternion_max_abs(
                        next_root_quat[None, :], rollout["root_quat"][row + 1][None, :]
                    ),
                    "root_linear_velocity_max_abs": max_abs(
                        next_root_lin_vel, rollout["root_lin_vel"][row + 1]
                    ),
                    "root_angular_velocity_max_abs": max_abs(
                        next_root_ang_vel, rollout["root_ang_vel"][row + 1]
                    ),
                    "joint_position_max_abs": max_abs(
                        next_joint_pos_robot[robot_to_saved], rollout["joint_pos"][row + 1]
                    ),
                    "joint_velocity_max_abs": max_abs(
                        next_joint_vel_robot[robot_to_saved], rollout["joint_vel"][row + 1]
                    ),
                    "body_position_max_abs": max_abs(next_body_pos, rollout["all_body_pos"][row + 1]),
                    "body_quaternion_max_abs_sign_invariant": quaternion_max_abs(
                        next_body_quat, rollout["all_body_quat"][row + 1]
                    ),
                    "observation_max_abs": max_abs(
                        replay_next_observation, expected_next_observation
                    ),
                    "observation_group_max_abs": next_observation_errors,
                    "contact_force_max_abs": max_abs(replay_contact, expected_contact),
                },
            }
        )

    output = args_cli.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task": args_cli.task,
        "rollout": str(args_cli.rollout.resolve()),
        "joint_count": len(robot_joint_names),
        "policy_joint_count": len(command.npz_to_isaac_indices),
        "observation_dim": int(rollout["policy_observations"].shape[1]),
        "action_dim": int(rollout["actions"].shape[1]),
        "shared_contact_body_count": len(shared_contact_names),
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
