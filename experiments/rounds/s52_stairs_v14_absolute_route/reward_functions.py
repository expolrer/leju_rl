from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def _body_indexes(command: MotionCommand, body_names: list[str]) -> torch.Tensor:
    indexes = [command.cfg.body_names.index(name) for name in body_names]
    return torch.tensor(indexes, dtype=torch.long, device=command.device)


def _frame_mask(command: MotionCommand, frame_start: int, frame_end: int) -> torch.Tensor:
    return (
        (command.time_steps >= int(frame_start))
        & (command.time_steps <= int(frame_end))
    ).float()


def _clipped_smooth_l1(normalized_error: torch.Tensor, cap: float) -> torch.Tensor:
    absolute = torch.abs(normalized_error)
    loss = torch.where(absolute < 1.0, 0.5 * torch.square(absolute), absolute - 0.5)
    return torch.clamp(loss, max=float(cap))


def absolute_anchor_route_huber_penalty(
    env,
    command_name: str,
    forward_scale: float,
    lateral_scale: float,
    height_scale: float,
    cap: float = 6.0,
) -> torch.Tensor:
    """Penalize un-reanchored pelvis route error without exponential saturation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    scales = torch.tensor(
        [forward_scale, lateral_scale, height_scale],
        dtype=command.robot_anchor_pos_w.dtype,
        device=command.device,
    )
    normalized = (command.robot_anchor_pos_w - command.anchor_pos_w) / scales
    return _clipped_smooth_l1(normalized, cap).mean(dim=-1)


def absolute_feet_route_huber_penalty(
    env,
    command_name: str,
    horizontal_scale: float,
    vertical_scale: float,
    body_names: list[str],
    frame_start: int = 0,
    frame_end: int = 10_000,
    cap: float = 6.0,
) -> torch.Tensor:
    """Track both S52 feet in world coordinates so route drift cannot be reanchored away."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    indexes = _body_indexes(command, body_names)
    error = command.robot_body_pos_w[:, indexes] - command.body_pos_w[:, indexes]
    horizontal = _clipped_smooth_l1(error[..., :2] / float(horizontal_scale), cap).mean(dim=-1)
    vertical = _clipped_smooth_l1(error[..., 2] / float(vertical_scale), cap)
    return (horizontal + vertical).mean(dim=-1) * _frame_mask(command, frame_start, frame_end)


def forward_route_overshoot_penalty(
    env,
    command_name: str,
    allowed_ahead: float,
    scale: float,
    frame_start: int,
    cap: float = 6.0,
) -> torch.Tensor:
    """Charge only motion ahead of the fixed route, the v13 false-progress mode."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ahead = torch.relu(
        command.robot_anchor_pos_w[:, 0]
        - command.anchor_pos_w[:, 0]
        - float(allowed_ahead)
    )
    normalized = ahead / float(scale)
    return _clipped_smooth_l1(normalized, cap) * _frame_mask(
        command, frame_start, command.motion.time_step_total - 1
    )


def terminal_double_support_route_reward(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    position_std: float,
    speed_std: float,
    frame_start: int,
) -> torch.Tensor:
    """Reward the final fixed landing only when both feet support a route-aligned pelvis."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    if contact_force.shape[-1] != 2:
        raise ValueError("terminal support reward requires exactly two foot bodies")
    support = torch.clamp(contact_force / float(support_force_threshold), 0.0, 1.0).min(dim=-1)[0]
    position_error = torch.sum(
        torch.square(command.robot_anchor_pos_w - command.anchor_pos_w), dim=-1
    )
    speed_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)
    route_score = torch.exp(-position_error / float(position_std) ** 2)
    settle_score = torch.exp(-speed_error / float(speed_std) ** 2)
    return support * route_score * settle_score * _frame_mask(
        command, frame_start, command.motion.time_step_total - 1
    )
