#!/usr/bin/env python3
"""Run a trained RSL-RL policy headlessly and record exact Isaac body states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record a Kuavo stair policy rollout without rendering.")
parser.add_argument("--steps", type=int, default=325)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
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
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from leju_robot.rsl_rl_extensions.utils.on_policy_runner import RobotOnPolicyRunner  # noqa: F401
import leju_robot.tasks.tracking.mdp as tracking_mdp


def cpu(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy()


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed if args_cli.seed is not None else 42)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    # The stairs PLAY configs inherit the training command term. Replace it here
    # so evaluation always begins at frame zero and advances deterministically.
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
    )

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    raw_env = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)

    obs, _ = env.get_observations()
    robot = raw_env.scene["robot"]
    command = raw_env.command_manager.get_term("motion")
    env_origin = raw_env.scene.env_origins[0]

    actual_body_pos = []
    reference_body_pos = []
    all_body_pos = []
    root_pos = []
    root_quat = []
    joint_pos = []
    joint_vel = []
    motion_frame = []
    anchor_error = []
    body_error = []
    actions = []
    rewards = []
    dones = []

    def capture() -> None:
        actual_body_pos.append(cpu(command.robot_body_pos_w[0] - env_origin))
        reference_body_pos.append(cpu(command.body_pos_relative_w[0] - env_origin))
        all_body_pos.append(cpu(robot.data.body_pos_w[0] - env_origin))
        root_pos.append(cpu(robot.data.root_pos_w[0] - env_origin))
        root_quat.append(cpu(robot.data.root_quat_w[0]))
        joint_pos.append(cpu(robot.data.joint_pos[0]))
        joint_vel.append(cpu(robot.data.joint_vel[0]))
        motion_frame.append(int(command.time_steps[0].item()))
        anchor_error.append(float(command.metrics["error_anchor_pos"][0].item()))
        body_error.append(float(command.metrics["error_body_pos"][0].item()))

    capture()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
        actions.append(cpu(action[0]))
        rewards.append(float(reward[0].item()))
        dones.append(bool(done[0].item()))
        capture()

    output = args_cli.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        checkpoint=np.asarray(checkpoint),
        task=np.asarray(args_cli.task),
        motion_file=np.asarray(env_cfg.commands.motion.motion_file),
        dt=np.asarray(float(raw_env.step_dt)),
        body_names=np.asarray(command.cfg.body_names),
        all_body_names=np.asarray(robot.body_names),
        joint_names=np.asarray(robot.joint_names),
        actual_body_pos=np.asarray(actual_body_pos),
        reference_body_pos=np.asarray(reference_body_pos),
        all_body_pos=np.asarray(all_body_pos),
        root_pos=np.asarray(root_pos),
        root_quat=np.asarray(root_quat),
        joint_pos=np.asarray(joint_pos),
        joint_vel=np.asarray(joint_vel),
        motion_frame=np.asarray(motion_frame),
        anchor_error=np.asarray(anchor_error),
        body_error=np.asarray(body_error),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        dones=np.asarray(dones),
    )

    left_idx = command.cfg.body_names.index("leg_l6_link")
    right_idx = command.cfg.body_names.index("leg_r6_link")
    actual = np.asarray(actual_body_pos)
    evaluation_start = min(2, len(anchor_error) - 1)
    summary = {
        "checkpoint": checkpoint,
        "task": args_cli.task,
        "motion_file": env_cfg.commands.motion.motion_file,
        "steps": args_cli.steps,
        "dt": float(raw_env.step_dt),
        "duration_s": args_cli.steps * float(raw_env.step_dt),
        "initialization_resets": int(bool(dones[0])) if dones else 0,
        "rollout_failure_resets": int(np.count_nonzero(dones[1:])),
        "root_start": np.asarray(root_pos)[0].tolist(),
        "root_end": np.asarray(root_pos)[-1].tolist(),
        "root_displacement": (np.asarray(root_pos)[-1] - np.asarray(root_pos)[0]).tolist(),
        "max_motion_frame": int(np.max(motion_frame)),
        "mean_anchor_error_m": float(np.mean(anchor_error[evaluation_start:])),
        "max_anchor_error_m": float(np.max(anchor_error[evaluation_start:])),
        "mean_body_error_m": float(np.mean(body_error[evaluation_start:])),
        "max_body_error_m": float(np.max(body_error[evaluation_start:])),
        "max_left_foot_z_m": float(actual[:, left_idx, 2].max()),
        "max_right_foot_z_m": float(actual[:, right_idx, 2].max()),
        "final_left_foot": actual[-1, left_idx].tolist(),
        "final_right_foot": actual[-1, right_idx].tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
