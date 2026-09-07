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


def _pseudo_huber(normalized_error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 + torch.square(normalized_error)) - 1.0


def absolute_anchor_route_pseudo_huber_penalty(
    env,
    command_name: str,
    forward_scale: float,
    lateral_scale: float,
    height_scale: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    scales = torch.tensor(
        [forward_scale, lateral_scale, height_scale],
        dtype=command.robot_anchor_pos_w.dtype,
        device=command.device,
    )
    normalized = (command.robot_anchor_pos_w - command.anchor_pos_w) / scales
    return _pseudo_huber(normalized).mean(dim=-1)


def absolute_feet_route_pseudo_huber_penalty(
    env,
    command_name: str,
    horizontal_scale: float,
    vertical_scale: float,
    body_names: list[str],
    frame_start: int = 0,
    frame_end: int = 10_000,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    indexes = _body_indexes(command, body_names)
    error = command.robot_body_pos_w[:, indexes] - command.body_pos_w[:, indexes]
    horizontal = _pseudo_huber(error[..., :2] / float(horizontal_scale)).mean(dim=-1)
    vertical = _pseudo_huber(error[..., 2] / float(vertical_scale))
    return (horizontal + vertical).mean(dim=-1) * _frame_mask(
        command, frame_start, frame_end
    )


def forward_route_overshoot_pseudo_huber_penalty(
    env,
    command_name: str,
    allowed_ahead: float,
    scale: float,
    frame_start: int,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ahead = torch.relu(
        command.robot_anchor_pos_w[:, 0]
        - command.anchor_pos_w[:, 0]
        - float(allowed_ahead)
    )
    return _pseudo_huber(ahead / float(scale)) * _frame_mask(
        command, frame_start, command.motion.time_step_total - 1
    )


def route_velocity_feedback_penalty(
    env,
    command_name: str,
    position_gain: float,
    max_position_correction: float,
    velocity_scale: float,
) -> torch.Tensor:
    """Track reference velocity plus bounded route-error feedback in world XY."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    route_error = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    correction = torch.clamp(
        float(position_gain) * route_error,
        min=-float(max_position_correction),
        max=float(max_position_correction),
    )
    desired_velocity = command.anchor_lin_vel_w[:, :2] + correction
    velocity_error = command.robot_anchor_lin_vel_w[:, :2] - desired_velocity
    return _pseudo_huber(velocity_error / float(velocity_scale)).mean(dim=-1)


def platform_stop_speed_penalty(
    env,
    command_name: str,
    linear_scale: float,
    yaw_scale: float,
    frame_start: int,
    frame_end: int,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    linear = _pseudo_huber(command.robot_anchor_lin_vel_w[:, :2] / float(linear_scale)).mean(
        dim=-1
    )
    yaw = _pseudo_huber(command.robot_anchor_ang_vel_w[:, 2] / float(yaw_scale))
    return (linear + 0.25 * yaw) * _frame_mask(command, frame_start, frame_end)


def platform_stationary_feet_penalty(
    env,
    command_name: str,
    body_names: list[str],
    speed_scale: float,
    frame_start: int,
    frame_end: int,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    indexes = _body_indexes(command, body_names)
    foot_speed = torch.linalg.vector_norm(
        command.robot_body_lin_vel_w[:, indexes, :], dim=-1
    )
    return _pseudo_huber(foot_speed / float(speed_scale)).mean(dim=-1) * _frame_mask(
        command, frame_start, frame_end
    )


def platform_double_support_hold_reward(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    position_std: float,
    speed_std: float,
    frame_start: int,
    frame_end: int,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    if contact_force.shape[-1] != 2:
        raise ValueError("platform hold reward requires exactly two foot bodies")
    support = torch.clamp(
        contact_force / float(support_force_threshold), 0.0, 1.0
    ).min(dim=-1)[0]
    route_error = torch.sum(
        torch.square(command.robot_anchor_pos_w - command.anchor_pos_w), dim=-1
    )
    speed_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)
    return (
        support
        * torch.exp(-route_error / float(position_std) ** 2)
        * torch.exp(-speed_error / float(speed_std) ** 2)
        * _frame_mask(command, frame_start, frame_end)
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
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    support = torch.clamp(
        contact_force / float(support_force_threshold), 0.0, 1.0
    ).min(dim=-1)[0]
    route_error = torch.sum(
        torch.square(command.robot_anchor_pos_w - command.anchor_pos_w), dim=-1
    )
    speed_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)
    return (
        support
        * torch.exp(-route_error / float(position_std) ** 2)
        * torch.exp(-speed_error / float(speed_std) ** 2)
        * _frame_mask(command, frame_start, command.motion.time_step_total - 1)
    )
