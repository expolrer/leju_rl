from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass

from leju_robot.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class ActualStateReplayMotionCommand(MotionCommand):
    """Reset selected environments to states visited by the S52 teacher."""

    cfg: "ActualStateReplayMotionCommandCfg"

    def __init__(self, cfg: "ActualStateReplayMotionCommandCfg", env):
        super().__init__(cfg, env)
        if cfg.zero_start_fraction + cfg.actual_replay_fraction > 1.0:
            raise ValueError("zero_start_fraction + actual_replay_fraction must be <= 1")
        if not cfg.actual_replay_rows:
            raise ValueError("actual_replay_rows must not be empty")

        replay = np.load(cfg.actual_replay_file, allow_pickle=True)
        required = {
            "joint_names",
            "all_body_names",
            "root_pos",
            "root_quat",
            "root_lin_vel",
            "root_ang_vel",
            "joint_pos",
            "joint_vel",
            "motion_frame",
            "policy_observations",
        }
        missing = sorted(required.difference(replay.files))
        if missing:
            raise KeyError(f"actual replay file is missing: {missing}")

        saved_joint_names = [str(name) for name in replay["joint_names"]]
        robot_joint_names = list(self.robot.joint_names)
        if set(saved_joint_names) != set(robot_joint_names):
            raise ValueError("actual replay joint names do not match the S52 asset")
        saved_index = {name: index for index, name in enumerate(saved_joint_names)}
        robot_order = [saved_index[name] for name in robot_joint_names]
        if [str(name) for name in replay["all_body_names"]] != list(self.robot.body_names):
            raise ValueError("actual replay body order does not match the S52 asset")
        if replay["policy_observations"].shape[1] != 148:
            raise ValueError("actual replay must contain 148-D policy observations")

        replay_rows = np.asarray(cfg.actual_replay_rows, dtype=np.int64)
        if np.any(replay_rows < 1) or np.any(replay_rows >= len(replay["root_pos"])):
            raise ValueError("actual replay row is outside the dataset")
        replay_frames = replay["motion_frame"][replay_rows]
        if np.any(replay_frames != replay_rows):
            raise ValueError("actual replay rows must map one-to-one to command frames")

        self._actual_rows = torch.as_tensor(
            replay_rows, dtype=torch.long, device=self.device
        )
        self._actual_motion_frame = torch.as_tensor(
            replay["motion_frame"], dtype=torch.long, device=self.device
        )
        self._actual_root_pos = torch.as_tensor(
            replay["root_pos"], dtype=torch.float32, device=self.device
        )
        self._actual_root_quat = torch.as_tensor(
            replay["root_quat"], dtype=torch.float32, device=self.device
        )
        self._actual_root_lin_vel = torch.as_tensor(
            replay["root_lin_vel"], dtype=torch.float32, device=self.device
        )
        self._actual_root_ang_vel = torch.as_tensor(
            replay["root_ang_vel"], dtype=torch.float32, device=self.device
        )
        self._actual_joint_pos = torch.as_tensor(
            replay["joint_pos"][:, robot_order], dtype=torch.float32, device=self.device
        )
        self._actual_joint_vel = torch.as_tensor(
            replay["joint_vel"][:, robot_order], dtype=torch.float32, device=self.device
        )
        self._actual_previous_action = torch.as_tensor(
            replay["policy_observations"][:, 121:148],
            dtype=torch.float32,
            device=self.device,
        )
        self._actual_row_by_env = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.metrics["sampling_zero_start_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_actual_replay_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_actual_replay_frame"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        super()._adaptive_sampling(env_ids)
        if len(env_ids) == 0:
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._actual_row_by_env[ids] = -1
        draw = torch.rand(len(ids), device=self.device)
        actual_mask = draw < self.cfg.actual_replay_fraction
        zero_mask = (
            (draw >= self.cfg.actual_replay_fraction)
            & (
                draw
                < self.cfg.actual_replay_fraction + self.cfg.zero_start_fraction
            )
        )
        if torch.any(zero_mask):
            self.time_steps[ids[zero_mask]] = 0
        if torch.any(actual_mask):
            actual_ids = ids[actual_mask]
            row_choices = torch.randint(
                0, len(self._actual_rows), (len(actual_ids),), device=self.device
            )
            rows = self._actual_rows[row_choices]
            self._actual_row_by_env[actual_ids] = rows
            self.time_steps[actual_ids] = self._actual_motion_frame[rows]

        self.metrics["sampling_zero_start_fraction"][:] = zero_mask.float().mean()
        self.metrics["sampling_actual_replay_fraction"][:] = actual_mask.float().mean()
        self.metrics["sampling_actual_replay_frame"].zero_()
        if torch.any(actual_mask):
            self.metrics["sampling_actual_replay_frame"][ids[actual_mask]] = (
                self.time_steps[ids[actual_mask]].float()
                / float(max(self.motion.time_step_total - 1, 1))
            )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        replay_mask = self._actual_row_by_env[ids] >= 0
        if not torch.any(replay_mask):
            return
        replay_ids = ids[replay_mask]
        rows = self._actual_row_by_env[replay_ids]
        root_pos = self._actual_root_pos[rows] + self._env.scene.env_origins[replay_ids]
        root_state = torch.cat(
            (
                root_pos,
                self._actual_root_quat[rows],
                self._actual_root_lin_vel[rows],
                self._actual_root_ang_vel[rows],
            ),
            dim=-1,
        )
        self.robot.write_joint_state_to_sim(
            self._actual_joint_pos[rows],
            self._actual_joint_vel[rows],
            env_ids=replay_ids,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=replay_ids)

        previous_action = self._actual_previous_action[rows]
        self._env.action_manager._action[replay_ids] = previous_action
        self._env.action_manager._prev_action[replay_ids] = previous_action


@configclass
class ActualStateReplayMotionCommandCfg(MotionCommandCfg):
    class_type: type = ActualStateReplayMotionCommand
    actual_replay_file: str = MISSING
    actual_replay_rows: tuple[int, ...] = ()
    actual_replay_fraction: float = 0.45
    zero_start_fraction: float = 0.55
