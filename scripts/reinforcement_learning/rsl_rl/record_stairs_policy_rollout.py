#!/usr/bin/env python3
"""Run a trained RSL-RL policy headlessly and record exact Isaac body states."""

from __future__ import annotations

import argparse
import copy
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
import trimesh

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from leju_robot.rsl_rl_extensions.utils.on_policy_runner import RobotOnPolicyRunner  # noqa: F401
import leju_robot.tasks.tracking.mdp as tracking_mdp


def cpu(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy()


def terrain_mesh_local(env_cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the exact configured sub-terrain in its environment-local frame."""
    generator_cfg = env_cfg.scene.terrain.terrain_generator
    if generator_cfg is None or len(generator_cfg.sub_terrains) != 1:
        raise RuntimeError("expected one generated sub-terrain")
    sub_cfg = copy.deepcopy(next(iter(generator_cfg.sub_terrains.values())))
    sub_cfg.size = generator_cfg.size
    meshes, terrain_origin = sub_cfg.function(0.0, sub_cfg)
    mesh = trimesh.util.concatenate(meshes)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) - np.asarray(terrain_origin, dtype=np.float32)
    return vertices, np.asarray(mesh.faces, dtype=np.int32), np.asarray(terrain_origin, dtype=np.float32)


def terrain_surface_height(
    vertices: np.ndarray, faces: np.ndarray, points_xy: np.ndarray
) -> np.ndarray:
    """Return the highest horizontal terrain surface below each XY point."""
    triangles = vertices[faces]
    horizontal = np.ptp(triangles[:, :, 2], axis=1) < 1.0e-6
    triangles = triangles[horizontal]
    heights = np.full(len(points_xy), np.nan, dtype=np.float32)
    tolerance = 1.0e-6
    for triangle in triangles:
        xy = triangle[:, :2]
        edge0 = xy[1] - xy[0]
        edge1 = xy[2] - xy[0]
        denominator = edge0[0] * edge1[1] - edge1[0] * edge0[1]
        if abs(denominator) < tolerance:
            continue
        relative = points_xy - xy[0]
        u = (relative[:, 0] * edge1[1] - relative[:, 1] * edge1[0]) / denominator
        v = (edge0[0] * relative[:, 1] - edge0[1] * relative[:, 0]) / denominator
        inside = (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1.0 + tolerance)
        heights[inside] = np.fmax(heights[inside], float(triangle[0, 2]))
    return heights


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
    play_cfg_type = tracking_mdp.MotionCommandPlayCfg
    play_cfg_kwargs = {}
    if hasattr(motion_cfg, "gate_frames"):
        play_cfg_type = tracking_mdp.StepToGatedMotionCommandPlayCfg
        play_cfg_kwargs = {
            "gate_frames": tuple(motion_cfg.gate_frames),
            "gate_body_names": tuple(motion_cfg.gate_body_names),
            "gate_position_tolerance": motion_cfg.gate_position_tolerance,
            "gate_foot_speed_tolerance": motion_cfg.gate_foot_speed_tolerance,
            "gate_anchor_speed_tolerance": motion_cfg.gate_anchor_speed_tolerance,
            "gate_anchor_angular_speed_tolerance": motion_cfg.gate_anchor_angular_speed_tolerance,
            "gate_stable_steps": motion_cfg.gate_stable_steps,
        }
    env_cfg.commands.motion = play_cfg_type(
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
        **play_cfg_kwargs,
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
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    contact_body_names = list(contact_sensor.body_names)
    foot_names = ("leg_l6_link", "leg_r6_link")
    foot_contact_indices = [contact_body_names.index(name) for name in foot_names]

    terrain_vertices, terrain_faces, terrain_origin = terrain_mesh_local(env_cfg)

    actual_body_pos = []
    actual_body_quat = []
    actual_body_lin_vel = []
    actual_body_ang_vel = []
    reference_body_pos = []
    reference_body_quat = []
    all_body_pos = []
    all_body_quat = []
    root_pos = []
    root_quat = []
    root_lin_vel = []
    root_ang_vel = []
    joint_pos = []
    joint_vel = []
    motion_frame = []
    anchor_error = []
    body_error = []
    actions = []
    rewards = []
    dones = []
    contact_force_w = []
    foot_contact_force_w = []
    gate_position_error = []
    gate_foot_speed = []
    gate_anchor_speed = []
    gate_anchor_angular_speed = []
    gate_stable_count = []
    shared_clearance_metric_names = (
        "shared_clearance_cost",
        "shared_clearance_active_fraction",
        "shared_clearance_actual_swing_fraction",
        "shared_clearance_planned_swing_fraction",
        "shared_clearance_violation_fraction",
        "shared_clearance_hard_fraction",
        "shared_clearance_min_distance",
        "left_shared_clearance_distance",
        "right_shared_clearance_distance",
    )
    shared_clearance_metrics = {
        name: [] for name in shared_clearance_metric_names
    }

    def capture() -> None:
        actual_body_pos.append(cpu(command.robot_body_pos_w[0] - env_origin))
        actual_body_quat.append(cpu(command.robot_body_quat_w[0]))
        actual_body_lin_vel.append(cpu(command.robot_body_lin_vel_w[0]))
        actual_body_ang_vel.append(cpu(command.robot_body_ang_vel_w[0]))
        reference_body_pos.append(cpu(command.body_pos_relative_w[0] - env_origin))
        reference_body_quat.append(cpu(command.body_quat_relative_w[0]))
        all_body_pos.append(cpu(robot.data.body_pos_w[0] - env_origin))
        all_body_quat.append(cpu(robot.data.body_quat_w[0]))
        root_pos.append(cpu(robot.data.root_pos_w[0] - env_origin))
        root_quat.append(cpu(robot.data.root_quat_w[0]))
        root_lin_vel.append(cpu(robot.data.root_lin_vel_w[0]))
        root_ang_vel.append(cpu(robot.data.root_ang_vel_w[0]))
        joint_pos.append(cpu(robot.data.joint_pos[0]))
        joint_vel.append(cpu(robot.data.joint_vel[0]))
        motion_frame.append(int(command.time_steps[0].item()))
        anchor_error.append(float(command.metrics["error_anchor_pos"][0].item()))
        body_error.append(float(command.metrics["error_body_pos"][0].item()))
        for name in shared_clearance_metric_names:
            metric = command.metrics.get(name)
            shared_clearance_metrics[name].append(
                float(metric[0].item()) if metric is not None else float("nan")
            )
        forces = contact_sensor.data.net_forces_w_history[0, -1]
        contact_force_w.append(cpu(forces))
        foot_contact_force_w.append(cpu(forces[foot_contact_indices]))
        if hasattr(command, "gate_body_indexes"):
            target_feet = command.body_pos_relative_w[0, command.gate_body_indexes]
            current_feet = command.robot_body_pos_w[0, command.gate_body_indexes]
            current_foot_velocity = command.robot_body_lin_vel_w[0, command.gate_body_indexes]
            gate_position_error.append(float(torch.linalg.vector_norm(current_feet - target_feet, dim=-1).amax().item()))
            gate_foot_speed.append(float(torch.linalg.vector_norm(current_foot_velocity, dim=-1).amax().item()))
            gate_anchor_speed.append(float(torch.linalg.vector_norm(command.robot_anchor_lin_vel_w[0]).item()))
            gate_anchor_angular_speed.append(
                float(torch.linalg.vector_norm(command.robot_anchor_ang_vel_w[0]).item())
            )
            gate_stable_count.append(int(command.gate_stable_count[0].item()))
        else:
            gate_position_error.append(float("nan"))
            gate_foot_speed.append(float("nan"))
            gate_anchor_speed.append(float("nan"))
            gate_anchor_angular_speed.append(float("nan"))
            gate_stable_count.append(-1)

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
        env_origin_w=cpu(env_origin),
        terrain_origin_in_patch=terrain_origin,
        terrain_vertices=terrain_vertices,
        terrain_faces=terrain_faces,
        contact_body_names=np.asarray(contact_body_names),
        foot_names=np.asarray(foot_names),
        actual_body_pos=np.asarray(actual_body_pos),
        actual_body_quat=np.asarray(actual_body_quat),
        actual_body_lin_vel=np.asarray(actual_body_lin_vel),
        actual_body_ang_vel=np.asarray(actual_body_ang_vel),
        reference_body_pos=np.asarray(reference_body_pos),
        reference_body_quat=np.asarray(reference_body_quat),
        all_body_pos=np.asarray(all_body_pos),
        all_body_quat=np.asarray(all_body_quat),
        root_pos=np.asarray(root_pos),
        root_quat=np.asarray(root_quat),
        root_lin_vel=np.asarray(root_lin_vel),
        root_ang_vel=np.asarray(root_ang_vel),
        joint_pos=np.asarray(joint_pos),
        joint_vel=np.asarray(joint_vel),
        motion_frame=np.asarray(motion_frame),
        anchor_error=np.asarray(anchor_error),
        body_error=np.asarray(body_error),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        dones=np.asarray(dones),
        contact_force_w=np.asarray(contact_force_w),
        foot_contact_force_w=np.asarray(foot_contact_force_w),
        gate_position_error=np.asarray(gate_position_error),
        gate_foot_speed=np.asarray(gate_foot_speed),
        gate_anchor_speed=np.asarray(gate_anchor_speed),
        gate_anchor_angular_speed=np.asarray(gate_anchor_angular_speed),
        gate_stable_count=np.asarray(gate_stable_count),
        **{
            f"metric_{name}": np.asarray(values, dtype=np.float32)
            for name, values in shared_clearance_metrics.items()
        },
    )

    left_idx = command.cfg.body_names.index("leg_l6_link")
    right_idx = command.cfg.body_names.index("leg_r6_link")
    actual = np.asarray(actual_body_pos)
    foot_forces = np.linalg.norm(np.asarray(foot_contact_force_w), axis=-1)
    foot_positions = actual[:, [left_idx, right_idx]]
    foot_surface_z = terrain_surface_height(
        terrain_vertices,
        terrain_faces,
        foot_positions[:, :, :2].reshape(-1, 2),
    ).reshape(len(foot_positions), 2)
    foot_center_clearance = foot_positions[:, :, 2] - foot_surface_z
    with np.load(env_cfg.commands.motion.motion_file) as motion_data:
        motion_frame_count = int(len(motion_data["joint_pos"]))
    contact_threshold_n = 10.0
    contact_mask = foot_forces > contact_threshold_n
    valid_contact = contact_mask & np.isfinite(foot_center_clearance)
    contact_levels = []
    for foot_index in range(2):
        levels = foot_surface_z[valid_contact[:, foot_index], foot_index]
        rounded = np.round(levels / 0.13) * 0.13
        unique, counts = np.unique(np.round(rounded, 3), return_counts=True)
        contact_levels.append({f"{level:.3f}": int(count) for level, count in zip(unique, counts)})
    center_clearance = foot_center_clearance[valid_contact]
    evaluation_start = min(2, len(anchor_error) - 1)
    npz_path = output
    with np.load(npz_path, allow_pickle=False) as saved:
        saved_arrays = {name: saved[name] for name in saved.files}
    saved_arrays["foot_surface_z"] = foot_surface_z
    saved_arrays["foot_center_clearance"] = foot_center_clearance
    np.savez_compressed(npz_path, **saved_arrays)
    summary = {
        "checkpoint": checkpoint,
        "task": args_cli.task,
        "motion_file": env_cfg.commands.motion.motion_file,
        "steps": args_cli.steps,
        "dt": float(raw_env.step_dt),
        "duration_s": args_cli.steps * float(raw_env.step_dt),
        "env_origin_w": cpu(env_origin).tolist(),
        "terrain_origin_in_patch": terrain_origin.tolist(),
        "initialization_resets": int(bool(dones[0])) if dones else 0,
        "rollout_failure_resets": int(np.count_nonzero(dones[1:])),
        "root_start": np.asarray(root_pos)[0].tolist(),
        "root_end": np.asarray(root_pos)[-1].tolist(),
        "root_displacement": (np.asarray(root_pos)[-1] - np.asarray(root_pos)[0]).tolist(),
        "max_motion_frame": int(np.max(motion_frame)),
        "motion_frame_count": motion_frame_count,
        "motion_completion_ratio": float(np.max(motion_frame) / max(1, motion_frame_count - 1)),
        "mean_anchor_error_m": float(np.mean(anchor_error[evaluation_start:])),
        "max_anchor_error_m": float(np.max(anchor_error[evaluation_start:])),
        "mean_body_error_m": float(np.mean(body_error[evaluation_start:])),
        "max_body_error_m": float(np.max(body_error[evaluation_start:])),
        "max_left_foot_z_m": float(actual[:, left_idx, 2].max()),
        "max_right_foot_z_m": float(actual[:, right_idx, 2].max()),
        "final_left_foot": actual[-1, left_idx].tolist(),
        "final_right_foot": actual[-1, right_idx].tolist(),
        "left_foot_contact_ratio": float(np.mean(foot_forces[:, 0] > contact_threshold_n)),
        "right_foot_contact_ratio": float(np.mean(foot_forces[:, 1] > contact_threshold_n)),
        "max_left_foot_contact_force_n": float(foot_forces[:, 0].max()),
        "max_right_foot_contact_force_n": float(foot_forces[:, 1].max()),
        "left_contact_surface_levels_m": contact_levels[0],
        "right_contact_surface_levels_m": contact_levels[1],
        "median_contact_foot_center_clearance_m": float(np.median(center_clearance))
        if center_clearance.size else None,
        "max_contact_foot_center_clearance_m": float(np.max(center_clearance))
        if center_clearance.size else None,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
