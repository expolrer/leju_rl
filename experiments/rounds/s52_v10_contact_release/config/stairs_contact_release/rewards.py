from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from leju_robot.tasks.tracking.mdp.commands import MotionCommand
from leju_robot.tasks.tracking.mdp.rewards import (
    _get_body_indexes,
    _scheduled_foot_mask,
    _scheduled_foothold_targets,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def scheduled_swing_path_l1_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    horizontal_scale: float,
    vertical_scale: float,
    foot_frame_ranges: tuple[tuple[tuple[int, int], ...], ...],
    body_names: list[str],
) -> torch.Tensor:
    """Keep useful path error credit after exponential tracking has saturated."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    active = _scheduled_foot_mask(command, foot_frame_ranges).float()
    error = torch.abs(
        command.robot_body_pos_w[:, body_indexes]
        - command.body_pos_w[:, body_indexes]
    )
    normalized = (
        error[..., :2].sum(dim=-1) / (2.0 * float(horizontal_scale))
        + error[..., 2] / float(vertical_scale)
    )
    return (normalized * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)


def scheduled_future_foothold_l1_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    horizontal_scale: float,
    foot_frame_ranges: tuple[tuple[tuple[int, int], ...], ...],
    body_names: list[str],
) -> torch.Tensor:
    """Pull each swing foot toward its fixed landing point without an exp tail."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    targets, active, progress = _scheduled_foothold_targets(
        env, command, body_indexes, foot_frame_ranges
    )
    error = torch.abs(
        command.robot_body_pos_w[:, body_indexes, :2] - targets[..., :2]
    ).sum(dim=-1) / (2.0 * float(horizontal_scale))
    weight = active.float() * torch.square(progress)
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def scheduled_swing_log_contact_force_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float,
    contact_force_scale: float,
    release_grace_frames: int,
    touchdown_grace_frames: int,
    foot_frame_ranges: tuple[tuple[tuple[int, int], ...], ...],
    body_names: list[str],
) -> torch.Tensor:
    """Penalize loaded swing feet monotonically without clipping high forces."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    active = _scheduled_foot_mask(
        command,
        foot_frame_ranges,
        start_trim=release_grace_frames,
        end_trim=touchdown_grace_frames,
    ).float()
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    if contact_force.shape[-1] != len(body_indexes):
        raise ValueError("contact sensor body order must match scheduled feet")
    excess = torch.relu(contact_force - float(contact_force_threshold))
    normalized = torch.log1p(excess / float(contact_force_scale))
    return (normalized * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
