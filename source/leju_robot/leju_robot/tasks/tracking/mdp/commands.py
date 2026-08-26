from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.isaac_joint_pos = self.robot.data.default_joint_pos.clone()
        self.isaac_joint_vel = self.robot.data.default_joint_vel.clone()

        self.npz_to_isaac_indices = self.robot.find_joints(
            self.robot.cfg.preserve_joint_order.joint_names, 
            preserve_order=True)[0]
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.out_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count


    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        
        sampled_at_start = (self.time_steps[env_ids] == 0)
        self.start_time[env_ids] = torch.where(
            sampled_at_start, 
            torch.zeros_like(self.time_steps[env_ids]), 
            torch.full_like(self.time_steps[env_ids], self.cfg.start_hold_steps)
        )
        self.out_time[env_ids] = 0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        # joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        
        isaac_soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        npz_soft_joint_pos_limits = isaac_soft_joint_pos_limits[:, self.npz_to_isaac_indices, :]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], npz_soft_joint_pos_limits[:, :, 0], npz_soft_joint_pos_limits[:, :, 1]
        )
        
        isaac_joint_pos_env = self.isaac_joint_pos[env_ids].clone()
        isaac_joint_vel_env = self.isaac_joint_vel[env_ids].clone()        
        isaac_joint_pos_env[:, self.npz_to_isaac_indices] = joint_pos[env_ids]
        isaac_joint_vel_env[:, self.npz_to_isaac_indices] = joint_vel[env_ids]        
        self.isaac_joint_pos[env_ids] = isaac_joint_pos_env
        self.isaac_joint_vel[env_ids] = isaac_joint_vel_env
        self.robot.write_joint_state_to_sim(self.isaac_joint_pos[env_ids], self.isaac_joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _motion_advance_mask(self) -> torch.Tensor:
        """Return which environments may advance to the next reference frame."""
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_command(self):
        in_start_phase = (self.start_time < self.cfg.start_hold_steps) & (self.time_steps == 0)
        self.start_time = torch.where(in_start_phase, self.start_time + 1, self.start_time)
        in_motion_phase = self.start_time >= self.cfg.start_hold_steps
        advance_motion = in_motion_phase & self._motion_advance_mask()
        self.time_steps = torch.where(advance_motion, self.time_steps + 1, self.time_steps)
        self.time_steps = torch.where(
            self.time_steps >= self.motion.time_step_total - 1, 
            self.motion.time_step_total - 1, 
            self.time_steps
        )
        self.out_time = torch.where(
            self.time_steps == self.motion.time_step_total - 1, 
            self.out_time + 1, 
            0
        )        
        env_ids = torch.where(self.out_time > self.cfg.end_hold_steps)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization implementation."""
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback function."""
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


class TerrainBiasedMotionCommand(MotionCommand):
    """Mix full-course starts with terrain-transition focused starts."""

    cfg: "TerrainBiasedMotionCommandCfg"

    def __init__(self, cfg: "TerrainBiasedMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.zero_start_fraction + cfg.terrain_focus_fraction > 1.0:
            raise ValueError(
                "zero_start_fraction + terrain_focus_fraction must be <= 1.0"
            )
        if not cfg.terrain_focus_frames:
            raise ValueError("terrain_focus_frames must not be empty")
        if max(cfg.terrain_focus_frames) >= self.motion.time_step_total:
            raise ValueError(
                f"terrain focus frame exceeds motion length {self.motion.time_step_total}"
            )
        self.metrics["sampling_zero_start_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_terrain_focus_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        super()._adaptive_sampling(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        random_draw = torch.rand(len(env_ids), device=self.device)
        zero_mask = random_draw < self.cfg.zero_start_fraction
        focus_mask = (
            (random_draw >= self.cfg.zero_start_fraction)
            & (
                random_draw
                < self.cfg.zero_start_fraction + self.cfg.terrain_focus_fraction
            )
        )

        if torch.any(zero_mask):
            self.time_steps[env_ids_tensor[zero_mask]] = 0
        if torch.any(focus_mask):
            focus_env_ids = env_ids_tensor[focus_mask]
            focus_frames = torch.as_tensor(
                self.cfg.terrain_focus_frames, dtype=torch.long, device=self.device
            )
            selected = focus_frames[
                torch.randint(
                    0, len(focus_frames), (len(focus_env_ids),), device=self.device
                )
            ]
            approach_offsets = torch.randint(
                1,
                max(self.cfg.terrain_focus_approach_steps, 1) + 1,
                (len(focus_env_ids),),
                device=self.device,
            )
            self.time_steps[focus_env_ids] = torch.clamp(
                selected - approach_offsets, min=0
            )

        self.metrics["sampling_zero_start_fraction"][:] = zero_mask.float().mean()
        self.metrics["sampling_terrain_focus_fraction"][:] = focus_mask.float().mean()


class StepToGatedMotionCommand(MotionCommand):
    """Freeze the reference at each stair gate until both feet are stable on target."""

    cfg: "StepToGatedMotionCommandCfg"

    def __init__(self, cfg: "StepToGatedMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sampling_fraction = (
            cfg.zero_start_fraction
            + cfg.gate_approach_fraction
            + cfg.foothold_approach_fraction
        )
        if sampling_fraction > 1.0:
            raise ValueError(
                "zero_start_fraction + gate_approach_fraction + "
                "foothold_approach_fraction must be <= 1.0"
            )
        self.gate_frames = torch.tensor(cfg.gate_frames, dtype=torch.long, device=self.device)
        if torch.any(self.gate_frames >= self.motion.time_step_total):
            raise ValueError(
                f"gate frame exceeds motion length {self.motion.time_step_total}: {cfg.gate_frames}"
            )
        self.gate_body_indexes = torch.tensor(
            [self.cfg.body_names.index(name) for name in cfg.gate_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.left_gate_body_index = self.cfg.body_names.index(cfg.gate_body_names[0])
        self.right_gate_body_index = self.cfg.body_names.index(cfg.gate_body_names[1])
        if cfg.gate_foothold_edge_x_offsets and (
            len(cfg.gate_foothold_edge_x_offsets) != len(cfg.gate_frames)
            or len(cfg.gate_foothold_ground_heights) != len(cfg.gate_frames)
        ):
            raise ValueError(
                "gate foothold edge and height tables must match gate_frames"
            )
        self.gate_stable_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gate_wait_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gate_pass_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gate_stable_reset_fraction = torch.zeros(self.num_envs, device=self.device)
        self.gate_pass_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gate_completion_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["gate_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_foot_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_anchor_angular_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_stable_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_wait_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_stable_reset_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_pass_event"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_pass_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_foothold_ok"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["gate_footprint_margin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_sole_max_gap"] = torch.zeros(self.num_envs, device=self.device)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Keep a fixed share of full-task and gate-approach episodes in every reset batch."""
        super()._adaptive_sampling(env_ids)
        if len(env_ids) == 0:
            return

        random_draw = torch.rand(len(env_ids), device=self.device)
        zero_mask = random_draw < self.cfg.zero_start_fraction
        gate_mask = (
            (random_draw >= self.cfg.zero_start_fraction)
            & (random_draw < self.cfg.zero_start_fraction + self.cfg.gate_approach_fraction)
            & (len(self.cfg.gate_frames) > 0)
        )
        foothold_start = self.cfg.zero_start_fraction + self.cfg.gate_approach_fraction
        foothold_mask = (
            (random_draw >= foothold_start)
            & (random_draw < foothold_start + self.cfg.foothold_approach_fraction)
            & (len(self.cfg.foothold_approach_frames) > 0)
        )
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if torch.any(zero_mask):
            self.time_steps[env_ids_tensor[zero_mask]] = 0
        if torch.any(gate_mask):
            gate_env_ids = env_ids_tensor[gate_mask]
            gate_choices = torch.randint(
                0,
                len(self.cfg.gate_frames),
                (len(gate_env_ids),),
                device=self.device,
            )
            selected_gates = torch.as_tensor(
                self.cfg.gate_frames, dtype=torch.long, device=self.device
            )[gate_choices]
            approach_offsets = torch.randint(
                1,
                max(self.cfg.gate_approach_steps, 1) + 1,
                (len(gate_env_ids),),
                device=self.device,
            )
            self.time_steps[gate_env_ids] = torch.clamp(selected_gates - approach_offsets, min=0)
        if torch.any(foothold_mask):
            foothold_env_ids = env_ids_tensor[foothold_mask]
            foothold_choices = torch.randint(
                0,
                len(self.cfg.foothold_approach_frames),
                (len(foothold_env_ids),),
                device=self.device,
            )
            selected_frames = torch.as_tensor(
                self.cfg.foothold_approach_frames, dtype=torch.long, device=self.device
            )[foothold_choices]
            approach_offsets = torch.randint(
                1,
                max(self.cfg.foothold_approach_steps, 1) + 1,
                (len(foothold_env_ids),),
                device=self.device,
            )
            self.time_steps[foothold_env_ids] = torch.clamp(
                selected_frames - approach_offsets, min=0
            )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if hasattr(self, "gate_stable_count") and len(env_ids) > 0:
            self.gate_stable_count[env_ids] = 0
            self.gate_wait_count[env_ids] = 0
            self.gate_pass_count[env_ids] = 0
            self.gate_stable_reset_fraction[env_ids] = 0.0
            self.gate_pass_event[env_ids] = False
            self.gate_completion_event[env_ids] = False

    def _target_offset_alpha(self, ramp: tuple[int, int]) -> torch.Tensor:
        start, end = ramp
        if end <= start:
            return (self.time_steps >= end).float()
        return ((self.time_steps.float() - float(start)) / float(end - start)).clamp(0.0, 1.0)

    def _apply_foothold_target_offsets(self):
        """Move descending foot targets to fit the asymmetric physical sole on each tread."""
        if self.cfg.foothold_target_x_offset == 0.0:
            return
        left_alpha = self._target_offset_alpha(self.cfg.left_target_offset_ramp)
        right_alpha = self._target_offset_alpha(self.cfg.right_target_offset_ramp)
        self.body_pos_relative_w[:, self.left_gate_body_index, 0] += (
            left_alpha * self.cfg.foothold_target_x_offset
        )
        self.body_pos_relative_w[:, self.right_gate_body_index, 0] += (
            right_alpha * self.cfg.foothold_target_x_offset
        )

    def _update_command(self):
        super()._update_command()
        self._apply_foothold_target_offsets()

    def _gate_foothold_quality(
        self, at_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check that both physical soles lie on one horizontal tread at a gate."""
        if not self.cfg.gate_foothold_edge_x_offsets:
            ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            zeros = torch.zeros(self.num_envs, device=self.device)
            return ones, zeros, zeros

        gate_matches = self.time_steps[:, None] == self.gate_frames[None, :]
        gate_index = gate_matches.long().argmax(dim=1)
        edge_offsets = torch.as_tensor(
            self.cfg.gate_foothold_edge_x_offsets,
            dtype=torch.float32,
            device=self.device,
        )
        ground_heights = torch.as_tensor(
            self.cfg.gate_foothold_ground_heights,
            dtype=torch.float32,
            device=self.device,
        )
        edge_x = self._env.scene.env_origins[:, 0] + edge_offsets[gate_index]
        ground_z = self._env.scene.env_origins[:, 2] + ground_heights[gate_index]
        quality_active = at_gate & (edge_offsets[gate_index] >= 0.0)

        local_points = torch.as_tensor(
            self.cfg.gate_sole_local_points,
            dtype=torch.float32,
            device=self.device,
        )
        body_pos = self.robot_body_pos_w[:, self.gate_body_indexes]
        body_quat = self.robot_body_quat_w[:, self.gate_body_indexes]
        num_points = local_points.shape[0]
        expanded_quat = body_quat[:, :, None, :].expand(-1, -1, num_points, -1)
        expanded_points = local_points[None, None, :, :].expand(
            self.num_envs, len(self.gate_body_indexes), -1, -1
        )
        rotated_points = quat_apply(
            expanded_quat.reshape(-1, 4), expanded_points.reshape(-1, 3)
        ).reshape(self.num_envs, len(self.gate_body_indexes), num_points, 3)
        sole_points_w = body_pos[:, :, None, :] + rotated_points

        sole_min_x = sole_points_w[..., 0].amin(dim=-1)
        sole_max_x = sole_points_w[..., 0].amax(dim=-1)
        rear_margin = sole_min_x - edge_x[:, None]
        front_margin = (
            edge_x[:, None] + self.cfg.gate_foothold_tread_length - sole_max_x
        )
        footprint_margin = torch.minimum(rear_margin, front_margin).amin(dim=-1)
        sole_max_gap = torch.abs(sole_points_w[..., 2] - ground_z[:, None, None]).amax(
            dim=(1, 2)
        )
        foothold_ok = (~quality_active) | (
            (footprint_margin >= self.cfg.gate_foothold_margin)
            & (sole_max_gap <= self.cfg.gate_sole_max_gap)
        )
        footprint_margin = torch.where(quality_active, footprint_margin, 0.0)
        sole_max_gap = torch.where(quality_active, sole_max_gap, 0.0)
        return foothold_ok, footprint_margin, sole_max_gap

    def _motion_advance_mask(self) -> torch.Tensor:
        if self.gate_frames.numel() == 0:
            return super()._motion_advance_mask()

        at_gate = torch.any(self.time_steps[:, None] == self.gate_frames[None, :], dim=1)
        # Match the target frame used by policy observations and tracking rewards.
        target_feet = self.body_pos_relative_w[:, self.gate_body_indexes]
        actual_feet = self.robot_body_pos_w[:, self.gate_body_indexes]
        position_error = torch.linalg.vector_norm(actual_feet - target_feet, dim=-1).amax(dim=-1)
        foot_speed = torch.linalg.vector_norm(
            self.robot_body_lin_vel_w[:, self.gate_body_indexes], dim=-1
        ).amax(dim=-1)
        anchor_speed = torch.linalg.vector_norm(self.robot_anchor_lin_vel_w, dim=-1)
        anchor_angular_speed = torch.linalg.vector_norm(self.robot_anchor_ang_vel_w, dim=-1)
        foothold_ok, footprint_margin, sole_max_gap = self._gate_foothold_quality(at_gate)
        stable = (
            (position_error <= self.cfg.gate_position_tolerance)
            & (foot_speed <= self.cfg.gate_foot_speed_tolerance)
            & (anchor_speed <= self.cfg.gate_anchor_speed_tolerance)
            & (anchor_angular_speed <= self.cfg.gate_anchor_angular_speed_tolerance)
            & foothold_ok
        )
        previous_stable_count = self.gate_stable_count.clone()
        next_stable_count = torch.where(
            at_gate & stable,
            self.gate_stable_count + 1,
            torch.zeros_like(self.gate_stable_count),
        )
        self.gate_stable_reset_fraction = torch.where(
            at_gate & ~stable & (previous_stable_count > 0),
            previous_stable_count.float() / float(max(self.cfg.gate_stable_steps, 1)),
            torch.zeros_like(self.gate_stable_reset_fraction),
        )
        self.gate_stable_count = next_stable_count
        self.gate_wait_count = torch.where(
            at_gate,
            self.gate_wait_count + 1,
            torch.zeros_like(self.gate_wait_count),
        )
        self.gate_pass_event = (
            at_gate
            & (previous_stable_count < self.cfg.gate_stable_steps)
            & (self.gate_stable_count >= self.cfg.gate_stable_steps)
        )
        next_pass_count = self.gate_pass_count + self.gate_pass_event.long()
        self.gate_completion_event = self.gate_pass_event & (
            next_pass_count >= len(self.cfg.gate_frames)
        )
        self.gate_pass_count = next_pass_count
        stable_fraction = self.gate_stable_count.float() / float(max(self.cfg.gate_stable_steps, 1))
        self.metrics["gate_active"] = at_gate.float()
        self.metrics["gate_position_error"] = torch.where(at_gate, position_error, 0.0)
        self.metrics["gate_foot_speed"] = torch.where(at_gate, foot_speed, 0.0)
        self.metrics["gate_anchor_speed"] = torch.where(at_gate, anchor_speed, 0.0)
        self.metrics["gate_anchor_angular_speed"] = torch.where(at_gate, anchor_angular_speed, 0.0)
        self.metrics["gate_stable_fraction"] = torch.where(
            at_gate, stable_fraction.clamp(max=1.0), 0.0
        )
        self.metrics["gate_wait_fraction"] = torch.where(
            at_gate,
            (
                self.gate_wait_count.float()
                / float(max(self.cfg.gate_max_wait_steps, 1))
            ).clamp(max=1.0),
            0.0,
        )
        self.metrics["gate_stable_reset_fraction"] = self.gate_stable_reset_fraction
        self.metrics["gate_pass_event"] = self.gate_pass_event.float()
        self.metrics["gate_pass_fraction"] = self.gate_pass_count.float() / float(
            max(len(self.cfg.gate_frames), 1)
        )
        self.metrics["gate_foothold_ok"] = torch.where(
            at_gate, foothold_ok.float(), 1.0
        )
        self.metrics["gate_footprint_margin"] = footprint_margin
        self.metrics["gate_sole_max_gap"] = sole_max_gap
        return (~at_gate) | (self.gate_stable_count >= self.cfg.gate_stable_steps)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    
    start_hold_steps: int = 50
    end_hold_steps: int = 50
    
    anchor_pos_threshold: float = 0.25
    anchor_ori_threshold: float = 0.3

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class TerrainBiasedMotionCommandCfg(MotionCommandCfg):
    """Motion command with explicit full-course and terrain-transition sampling."""

    class_type: type = TerrainBiasedMotionCommand
    zero_start_fraction: float = 0.30
    terrain_focus_fraction: float = 0.45
    terrain_focus_frames: tuple[int, ...] = ()
    terrain_focus_approach_steps: int = 40


@configclass
class StepToGatedMotionCommandCfg(MotionCommandCfg):
    """Motion command with fixed double-foot gates between descending levels."""

    class_type: type = StepToGatedMotionCommand
    gate_frames: tuple[int, ...] = ()
    gate_body_names: tuple[str, str] = ("leg_l6_link", "leg_r6_link")
    gate_position_tolerance: float = 0.045
    gate_foot_speed_tolerance: float = 0.10
    gate_anchor_speed_tolerance: float = 0.10
    gate_anchor_angular_speed_tolerance: float = 0.30
    gate_stable_steps: int = 10
    gate_max_wait_steps: int = 100
    zero_start_fraction: float = 0.0
    gate_approach_fraction: float = 0.0
    gate_approach_steps: int = 45
    foothold_approach_fraction: float = 0.0
    foothold_approach_frames: tuple[int, ...] = ()
    foothold_approach_steps: int = 30
    foothold_target_x_offset: float = 0.0
    left_target_offset_ramp: tuple[int, int] = (0, 0)
    right_target_offset_ramp: tuple[int, int] = (0, 0)
    gate_foothold_edge_x_offsets: tuple[float, ...] = ()
    gate_foothold_ground_heights: tuple[float, ...] = ()
    gate_foothold_tread_length: float = 0.28
    gate_foothold_margin: float = 0.012
    gate_sole_max_gap: float = 0.025
    gate_sole_local_points: tuple[tuple[float, float, float], ...] = (
        (-0.063, -0.040, -0.062),
        (-0.063, 0.040, -0.062),
        (0.165, -0.040, -0.062),
        (0.165, 0.040, -0.062),
    )


class MotionCommandPlay(MotionCommand):
    """Simplified motion command for PLAY mode - always starts from frame 0 and resets from beginning on failure."""
    
    cfg: "MotionCommandPlayCfg"
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Simplified resampling - always starts from frame 0."""
        if len(env_ids) == 0:
            return
        
        self.time_steps[env_ids] = 0
        self.start_time[env_ids] = 0
        self.out_time[env_ids] = 0

        # NOTE:
        # - Motion data positions are stored in a common world frame.
        # - In a vectorized env, each env instance is offset by env_origins in world.
        # Therefore we must add per-env origins to avoid all envs spawning at the same place.
        env_origins = self._env.scene.env_origins[env_ids].to(self.device)

        root_pos = self.motion.body_pos_w[0, self.motion_anchor_body_index].clone().to(self.device)
        root_ori = self.motion.body_quat_w[0, self.motion_anchor_body_index].clone().to(self.device)
        root_lin_vel = self.motion.body_lin_vel_w[0, self.motion_anchor_body_index].clone().to(self.device)
        root_ang_vel = self.motion.body_ang_vel_w[0, self.motion_anchor_body_index].clone().to(self.device)

        root_pos = root_pos.unsqueeze(0).expand(len(env_ids), -1) + env_origins
        root_ori = root_ori.unsqueeze(0).expand(len(env_ids), -1)
        root_lin_vel = root_lin_vel.unsqueeze(0).expand(len(env_ids), -1)
        root_ang_vel = root_ang_vel.unsqueeze(0).expand(len(env_ids), -1)

        # motion joint arrays are in NPZ joint order; map into Isaac joint order using npz_to_isaac_indices
        joint_pos = self.motion.joint_pos[0].clone().to(self.device).unsqueeze(0).expand(len(env_ids), -1)
        joint_vel = self.motion.joint_vel[0].clone().to(self.device).unsqueeze(0).expand(len(env_ids), -1)
        
        isaac_soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        npz_soft_joint_pos_limits = isaac_soft_joint_pos_limits[:, self.npz_to_isaac_indices, :]
        joint_pos = torch.clip(joint_pos, npz_soft_joint_pos_limits[:, :, 0], npz_soft_joint_pos_limits[:, :, 1])
        isaac_joint_pos_env = self.isaac_joint_pos[env_ids].clone()
        isaac_joint_vel_env = self.isaac_joint_vel[env_ids].clone()        
        isaac_joint_pos_env[:, self.npz_to_isaac_indices] = joint_pos[env_ids]
        isaac_joint_vel_env[:, self.npz_to_isaac_indices] = joint_vel[env_ids]        
        self.isaac_joint_pos[env_ids] = isaac_joint_pos_env
        self.isaac_joint_vel[env_ids] = isaac_joint_vel_env
        self.robot.write_joint_state_to_sim(self.isaac_joint_pos[env_ids], self.isaac_joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Simplified update - directly increment time_steps, reset to frame 0 when reaching the end."""
        self.time_steps = self.time_steps + 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if len(env_ids) > 0:
            self.time_steps[env_ids] = 0
            self.start_time[env_ids] = 0
            self.out_time[env_ids] = 0
            self._resample_command(env_ids)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()


@configclass
class MotionCommandPlayCfg(MotionCommandCfg):
    """Configuration for the simplified motion command in PLAY mode."""

    class_type: type = MotionCommandPlay


class StepToGatedMotionCommandPlay(StepToGatedMotionCommand):
    """Deterministic PLAY command that keeps the training-time stair gates."""

    cfg: "StepToGatedMotionCommandPlayCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        MotionCommandPlay._resample_command(self, env_ids)
        if hasattr(self, "gate_stable_count") and len(env_ids) > 0:
            self.gate_stable_count[env_ids] = 0
            self.gate_wait_count[env_ids] = 0
            self.gate_pass_count[env_ids] = 0
            self.gate_stable_reset_fraction[env_ids] = 0.0
            self.gate_pass_event[env_ids] = False
            self.gate_completion_event[env_ids] = False


@configclass
class StepToGatedMotionCommandPlayCfg(StepToGatedMotionCommandCfg):
    """Configuration for deterministic gated playback."""

    class_type: type = StepToGatedMotionCommandPlay
