#!/usr/bin/env python3
"""Record a deterministic Isaac Sim video of a Kuavo stair policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record a rendered Kuavo stair policy rollout.")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--video_length", type=int, default=800)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
disable_viewport = "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true"
args_cli.kit_args = f"{args_cli.kit_args or ''} {disable_viewport}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from leju_robot.rsl_rl_extensions.utils.on_policy_runner import RobotOnPolicyRunner  # noqa: F401
import leju_robot.tasks.tracking.mdp as tracking_mdp


# Isaac Lab 2.1 assumes an active viewport even for headless RTX sensors. The
# real camera below does not need one, so keep this workaround local to replay.
carb.settings.get_settings().set_bool("/isaaclab/render/active_viewport", True)


class _OffscreenCameraController:
    def __init__(self, *_args, **_kwargs):
        pass


import isaaclab.envs.manager_based_env as manager_based_env

manager_based_env.ViewportCameraController = _OffscreenCameraController


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed if args_cli.seed is not None else 42)
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    env_cfg.scene.render_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=4.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(4.5, 3.2, 2.0),
            rot=(-0.4417051, -0.1399513, -0.0699757, 0.8834102),
            convention="world",
        ),
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

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

    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    camera = raw_env.scene["render_camera"]
    video_path = output_dir / "isaac_stairs_policy.mp4"
    fps = int(round(1.0 / float(raw_env.step_dt)))
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
    completed_steps = 0
    rendered_frames = 0
    reset_count = 0
    max_sim_steps = args_cli.video_length + 200
    try:
        while simulation_app.is_running() and rendered_frames < args_cli.video_length:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.ndim == 3 and rgb.shape[0] > 0 and rgb.shape[1] > 0:
                writer.append_data(rgb[..., :3])
                rendered_frames += 1
            reset_count += int(torch.count_nonzero(dones).item())
            completed_steps += 1
            if completed_steps >= max_sim_steps:
                raise RuntimeError(
                    f"RTX camera did not produce enough frames: {rendered_frames}/{args_cli.video_length}"
                )
    finally:
        writer.close()

    print(f"[INFO] checkpoint={checkpoint}")
    print(
        f"[INFO] simulation_steps={completed_steps} rendered_frames={rendered_frames} reset_count={reset_count}"
    )
    print(f"[INFO] video={video_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
