from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_rotate_inverse
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_yaw
from .commands import MotionCommand
from .stairs_safety import rigid_foot_riser_clearance_cost

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _motion_phase_mask(command: MotionCommand, phase_start: float, phase_end: float) -> torch.Tensor:
    denominator = float(max(command.motion.time_step_total - 1, 1))
    phase = command.time_steps.float() / denominator
    return ((phase >= phase_start) & (phase <= phase_end)).float()


def _motion_frame_ranges_mask(
    command: MotionCommand, frame_ranges: list[tuple[int, int]]
) -> torch.Tensor:
    """Select exact reference ranges without normalized-phase rounding."""
    mask = torch.zeros_like(command.time_steps, dtype=torch.bool)
    for start, end in frame_ranges:
        mask |= (command.time_steps >= start) & (command.time_steps <= end)
    return mask.float()


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_feet_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward


def motion_global_anchor_lateral_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the reference corridor without over-constraining forward progress."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    lateral_error = torch.square(command.anchor_pos_w[:, 1] - command.robot_anchor_pos_w[:, 1])
    return torch.exp(-lateral_error / std**2)


def motion_global_anchor_forward_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track forward task progress independently from lateral and height errors."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 0] - command.robot_anchor_pos_w[:, 0])
    return torch.exp(-error / std**2)


def motion_global_anchor_height_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the stair-dependent pelvis height without coupling it to XY drift."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)


def motion_anchor_forward_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward the demonstrated positive-X speed, including stopping at the end."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_lin_vel_w[:, 0] - command.robot_anchor_lin_vel_w[:, 0])
    return torch.exp(-error / std**2)


def motion_backward_velocity_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize moving toward negative X during the forward-descent curriculum."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.relu(-command.robot_anchor_lin_vel_w[:, 0])


def motion_feet_horizontal_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track foot placement in the stair plane separately from foot clearance."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes, :2]
            - command.robot_body_pos_w[:, body_indexes, :2]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_phase_progress_reward(
    env: ManagerBasedRLEnv, command_name: str, exponent: float = 2.0
) -> torch.Tensor:
    """Reward surviving into later reference frames while preserving dense gradients."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    denominator = float(max(command.motion.time_step_total - 1, 1))
    progress = command.time_steps.float() / denominator
    return torch.pow(progress.clamp(0.0, 1.0), exponent)


def motion_phase_weighted_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float = 0.25,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Add tracking pressure gradually after the already-solved early stair phase."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    tracking_reward = torch.exp(-error.mean(-1) / std**2)
    denominator = float(max(command.motion.time_step_total - 1, 1))
    progress = command.time_steps.float() / denominator
    phase_weight = ((progress - phase_start) / max(1.0 - phase_start, 1.0e-6)).clamp(0.0, 1.0)
    return tracking_reward * phase_weight


def motion_phase_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
) -> torch.Tensor:
    """Track the fixed platform gate only during its stabilization window."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2) * _motion_phase_mask(command, phase_start, phase_end)


def motion_phase_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w))
    return torch.exp(-error / std**2) * _motion_phase_mask(command, phase_start, phase_end)


def motion_phase_anchor_stability_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
    phase_start: float,
    phase_end: float,
) -> torch.Tensor:
    """Require low pelvis velocity before the first descending step."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    linear = torch.clamp(
        torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=-1) / linear_velocity_scale,
        max=1.0,
    )
    angular = torch.clamp(
        torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1) / angular_velocity_scale,
        max=1.0,
    )
    return (torch.square(linear) + 0.5 * torch.square(angular)) * _motion_phase_mask(
        command, phase_start, phase_end
    )


def motion_phase_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Apply strict body tracking only in the selected motion phase."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward * _motion_phase_mask(command, phase_start, phase_end)


def motion_frame_ranges_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Track fixed world-frame foot targets during selected step-to phases."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_body_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Require support and double-stance feet to settle before phase changes."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    speed = torch.linalg.vector_norm(command.robot_body_lin_vel_w[:, body_indexes], dim=-1)
    normalized = torch.clamp(speed / speed_scale, max=1.0)
    return torch.square(normalized).mean(dim=-1) * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_anchor_stability_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
    frame_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """Penalize pelvis motion at every per-level double-foot localization gate."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    linear = torch.clamp(
        torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=-1) / linear_velocity_scale,
        max=1.0,
    )
    angular = torch.clamp(
        torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1) / angular_velocity_scale,
        max=1.0,
    )
    penalty = torch.square(linear) + 0.5 * torch.square(angular)
    return penalty * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_feet_same_level_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str],
) -> torch.Tensor:
    """Reward the demonstrated left/right X and Z separation at double-foot gates."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    if len(body_indexes) != 2:
        raise ValueError("feet_same_level requires exactly two body names")
    target_delta = command.body_pos_w[:, body_indexes[0], [0, 2]] - command.body_pos_w[
        :, body_indexes[1], [0, 2]
    ]
    actual_delta = command.robot_body_pos_w[:, body_indexes[0], [0, 2]] - command.robot_body_pos_w[
        :, body_indexes[1], [0, 2]
    ]
    error = torch.sum(torch.square(actual_delta - target_delta), dim=-1)
    return torch.exp(-error / std**2) * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_backward_body_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_scale: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str],
) -> torch.Tensor:
    """Suppress the observed backward kick of the active descending foot."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    backward_speed = torch.relu(-command.robot_body_lin_vel_w[:, body_indexes, 0])
    normalized = torch.clamp(backward_speed / velocity_scale, max=1.0)
    return torch.square(normalized).mean(dim=-1) * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_backward_body_velocity_log_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_scale: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str],
) -> torch.Tensor:
    """Keep a gradient for severe backward kicks without over-weighting small corrections."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    backward_speed = torch.relu(-command.robot_body_lin_vel_w[:, body_indexes, 0])
    normalized = backward_speed / velocity_scale
    return torch.log1p(torch.square(normalized)).mean(dim=-1) * _motion_frame_ranges_mask(
        command, frame_ranges
    )


def motion_frame_ranges_contact_impact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    frame_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """Penalize high landing impulses while preserving normal support forces."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    violation = torch.clamp((force - force_threshold) / force_threshold, min=0.0, max=3.0)
    return torch.square(violation).mean(dim=-1) * _motion_frame_ranges_mask(command, frame_ranges)


def motion_frame_ranges_body_under_reference_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_scale: float,
    allowed_below: float,
    frame_ranges: list[tuple[int, int]],
    body_names: list[str],
) -> torch.Tensor:
    """Keep a swing foot above the demonstrated clearance near stair nosings."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    target_z = command.body_pos_relative_w[:, body_indexes, 2]
    actual_z = command.robot_body_pos_w[:, body_indexes, 2]
    shortage = torch.relu(target_z - actual_z - allowed_below)
    normalized = torch.clamp(shortage / height_scale, max=3.0)
    return torch.square(normalized).mean(dim=-1) * _motion_frame_ranges_mask(
        command, frame_ranges
    )


def motion_frame_ranges_contact_force_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    force_scale: float,
    frame_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """Penalize contact while the demonstrated swing foot is clearing a nosing."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    violation = torch.clamp(
        (force - force_threshold) / force_scale, min=0.0, max=3.0
    )
    return torch.square(violation).mean(dim=-1) * _motion_frame_ranges_mask(
        command, frame_ranges
    )


def _body_sample_points_w(
    command: MotionCommand,
    body_indexes: list[int],
    local_points: list[tuple[float, float, float]],
) -> torch.Tensor:
    """Transform fixed foot-frame sample points into world coordinates."""
    points = torch.as_tensor(
        local_points, dtype=torch.float32, device=command.robot_body_pos_w.device
    )
    body_pos = command.robot_body_pos_w[:, body_indexes]
    body_quat = command.robot_body_quat_w[:, body_indexes]
    num_envs, num_bodies = body_pos.shape[:2]
    num_points = points.shape[0]
    expanded_quat = body_quat[:, :, None, :].expand(-1, -1, num_points, -1)
    expanded_points = points[None, None, :, :].expand(
        num_envs, num_bodies, -1, -1
    )
    rotated = quat_apply(
        expanded_quat.reshape(-1, 4), expanded_points.reshape(-1, 3)
    ).reshape(num_envs, num_bodies, num_points, 3)
    return body_pos[:, :, None, :] + rotated


def _fixed_stairs_predictive_sweep_distance(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    body_indexes: list[int],
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    local_points: list[tuple[float, float, float]],
    lookahead_time: float,
    approach_distance: float,
    release_distance: float,
    min_forward_speed: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the rigid sole's current and near-future sweep past descent risers."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    sample_points = _body_sample_points_w(command, body_indexes, local_points)
    body_pos = command.robot_body_pos_w[:, body_indexes]
    point_offsets = sample_points - body_pos[:, :, None, :]
    body_ang_vel = command.robot_body_ang_vel_w[:, body_indexes, None, :].expand_as(
        point_offsets
    )
    point_velocity = (
        command.robot_body_lin_vel_w[:, body_indexes, None, :]
        + torch.cross(body_ang_vel, point_offsets, dim=-1)
    )
    horizons = torch.as_tensor(
        (0.0, 0.5 * float(lookahead_time), float(lookahead_time)),
        dtype=torch.float32,
        device=env.device,
    )
    swept_points = (
        sample_points[:, :, :, None, :]
        + point_velocity[:, :, :, None, :] * horizons[None, None, None, :, None]
    )

    num_feet = len(body_indexes)
    predicted_min = torch.full(
        (env.num_envs, num_feet), 1.0e6, dtype=torch.float32, device=env.device
    )
    current_min = torch.full_like(predicted_min, 1.0e6)
    active = torch.zeros_like(predicted_min, dtype=torch.bool)
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
    forward_speed = command.robot_body_lin_vel_w[:, body_indexes, 0]
    moving_forward = forward_speed > float(min_forward_speed)
    leading_x = sample_points[..., 0].amax(dim=-1)
    trailing_x = sample_points[..., 0].amin(dim=-1)

    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        in_crossing_window = (
            (leading_x >= edge_x[:, None] - float(approach_distance))
            & (trailing_x <= edge_x[:, None] + float(release_distance))
        )
        edge_active = phase_active & moving_forward & in_crossing_window

        swept_z = swept_points[..., 2]
        swept_closest_z = torch.minimum(
            torch.maximum(swept_z, lower_z[:, None, None, None]),
            upper_z[:, None, None, None],
        )
        swept_dx = swept_points[..., 0] - edge_x[:, None, None, None]
        swept_dz = swept_z - swept_closest_z
        edge_predicted_min = torch.sqrt(
            torch.square(swept_dx) + torch.square(swept_dz) + 1.0e-9
        ).amin(dim=(2, 3))

        current_z = sample_points[..., 2]
        current_closest_z = torch.minimum(
            torch.maximum(current_z, lower_z[:, None, None]),
            upper_z[:, None, None],
        )
        current_dx = sample_points[..., 0] - edge_x[:, None, None]
        current_dz = current_z - current_closest_z
        edge_current_min = torch.sqrt(
            torch.square(current_dx) + torch.square(current_dz) + 1.0e-9
        ).amin(dim=2)

        predicted_min = torch.where(
            edge_active,
            torch.minimum(predicted_min, edge_predicted_min),
            predicted_min,
        )
        current_min = torch.where(
            edge_active,
            torch.minimum(current_min, edge_current_min),
            current_min,
        )
        active |= edge_active

    return predicted_min, current_min, active, forward_speed


def motion_predictive_riser_sweep_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    safety_distance: float,
    hard_distance: float,
    lookahead_time: float,
    approach_distance: float,
    release_distance: float,
    min_forward_speed: float,
    speed_scale: float,
    near_contact_force_threshold: float,
    contact_force_scale: float,
    hard_penalty_scale: float,
    contact_penalty_scale: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
    tread_heights: list[float] | None = None,
    reference_swing_speed_threshold: float = 0.0,
    reference_swing_contrast_threshold: float = 0.0,
    settled_contact_force_threshold: float = 0.0,
    settled_speed_threshold: float = 0.0,
    settled_max_gap: float = 0.0,
    tail_fraction: float = 0.0,
    tail_extra_weight: float = 0.0,
    tail_min_active_envs: int = 16,
) -> torch.Tensor:
    """Penalize predicted rigid-foot sweeps and emphasize low-clearance tails."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    predicted_min, current_min, active, forward_speed = (
        _fixed_stairs_predictive_sweep_distance(
            env=env,
            command=command,
            body_indexes=body_indexes,
            frame_ranges=frame_ranges,
            riser_x_offsets=riser_x_offsets,
            upper_heights=upper_heights,
            lower_heights=lower_heights,
            local_points=local_points,
            lookahead_time=lookahead_time,
            approach_distance=approach_distance,
            release_distance=release_distance,
            min_forward_speed=min_forward_speed,
        )
    )

    warning_shortage = torch.relu(
        (float(safety_distance) - predicted_min) / float(safety_distance)
    )
    hard_shortage = torch.relu(
        (float(hard_distance) - predicted_min) / float(hard_distance)
    )
    speed_weight = 0.5 + 0.5 * torch.clamp(
        (forward_speed - float(min_forward_speed)) / float(speed_scale),
        min=0.0,
        max=1.0,
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    reference_swing = torch.ones_like(active, dtype=torch.bool)
    if float(reference_swing_speed_threshold) > 0.0:
        reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
        reference_swing = reference_speed >= float(reference_swing_speed_threshold)
        if float(reference_swing_contrast_threshold) > 0.0:
            speed_contrast = reference_speed - reference_speed.amin(
                dim=-1, keepdim=True
            )
            reference_swing &= (
                speed_contrast >= float(reference_swing_contrast_threshold)
            )
        active &= reference_swing

    settled_support = torch.zeros_like(active, dtype=torch.bool)
    if tread_heights is not None:
        sole_samples_w = _body_sample_points_w(command, body_indexes, local_points)
        relative_x = (
            sole_samples_w[..., 0] - env.scene.env_origins[:, None, None, 0]
        )
        ground_z = _fixed_stairs_height(
            relative_x, riser_x_offsets, tread_heights
        )
        ground_z = ground_z + env.scene.env_origins[:, None, None, 2]
        max_gap = torch.abs(sole_samples_w[..., 2] - ground_z).amax(dim=-1)
        foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
        settled_support = (
            (contact_force >= float(settled_contact_force_threshold))
            & (foot_speed <= float(settled_speed_threshold))
            & (max_gap <= float(settled_max_gap))
        )
        active &= ~settled_support

    normalized_contact = torch.clamp(
        (contact_force - float(near_contact_force_threshold))
        / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    near_contact = (
        active
        & (predicted_min < float(safety_distance))
        & (contact_force >= float(near_contact_force_threshold))
    )

    per_foot = (
        torch.square(warning_shortage)
        + float(hard_penalty_scale) * torch.square(hard_shortage)
        + float(contact_penalty_scale)
        * warning_shortage
        * normalized_contact
        * near_contact.float()
    )
    per_foot = torch.clamp(per_foot * speed_weight * active.float(), max=4.0)
    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    unsafe = active & (predicted_min < float(safety_distance))
    reported_predicted_min = torch.where(
        active,
        predicted_min,
        torch.full_like(predicted_min, float(safety_distance)),
    )
    reported_current_min = torch.where(
        active,
        current_min,
        torch.full_like(current_min, float(safety_distance)),
    )
    per_env = per_foot.sum(dim=-1) / active_count
    tail_selected = torch.zeros_like(per_env, dtype=torch.bool)
    tail_threshold = torch.full_like(per_env, float(safety_distance))
    valid_env = active.any(dim=-1)
    if float(tail_fraction) > 0.0 and float(tail_extra_weight) > 0.0:
        env_min_distance = reported_predicted_min.amin(dim=-1).detach()
        active_distances = env_min_distance[valid_env]
        if active_distances.numel() >= int(tail_min_active_envs):
            threshold = torch.quantile(
                active_distances,
                min(max(float(tail_fraction), 0.0), 1.0),
            )
            tail_selected = valid_env & (env_min_distance <= threshold)
            tail_threshold = torch.where(
                valid_env,
                threshold.expand_as(per_env),
                tail_threshold,
            )
            per_env = per_env * (
                1.0 + float(tail_extra_weight) * tail_selected.float()
            )

    command.metrics["sweep_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["sweep_reference_swing_fraction"] = (
        reference_swing.float().mean(dim=-1)
    )
    command.metrics["sweep_settled_support_fraction"] = (
        settled_support.float().mean(dim=-1)
    )
    command.metrics["sweep_violation_fraction"] = (
        unsafe.float().sum(dim=-1) / active_count
    )
    command.metrics["sweep_near_contact_fraction"] = (
        near_contact.float().sum(dim=-1) / active_count
    )
    command.metrics["sweep_predicted_min_distance"] = reported_predicted_min.amin(
        dim=-1
    )
    command.metrics["sweep_current_min_distance"] = reported_current_min.amin(dim=-1)
    command.metrics["sweep_tail_selected_fraction"] = tail_selected.float()
    command.metrics["sweep_tail_clearance_threshold"] = tail_threshold
    command.metrics["sweep_tail_penalty"] = per_env
    if len(body_indexes) >= 2:
        command.metrics["left_sweep_active"] = active[:, 0].float()
        command.metrics["right_sweep_active"] = active[:, 1].float()
        command.metrics["left_sweep_predicted_min"] = reported_predicted_min[:, 0]
        command.metrics["right_sweep_predicted_min"] = reported_predicted_min[:, 1]
        command.metrics["left_sweep_near_contact"] = near_contact[:, 0].float()
        command.metrics["right_sweep_near_contact"] = near_contact[:, 1].float()
    return per_env


def motion_reference_swing_toe_riser_barrier_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    tread_heights: list[float],
    safety_distance: float,
    hard_distance: float,
    near_contact_force_threshold: float,
    contact_force_scale: float,
    hard_penalty_scale: float,
    contact_penalty_scale: float,
    reference_swing_speed_threshold: float,
    reference_swing_contrast_threshold: float,
    settled_contact_force_threshold: float,
    settled_speed_threshold: float,
    settled_max_gap: float,
    local_points: list[tuple[float, float, float]],
    sole_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Apply a direct toe barrier throughout reference swing, including contact."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    toe_points = _body_sample_points_w(command, body_indexes, local_points)
    min_distance = torch.full(
        (env.num_envs, len(body_indexes)),
        1.0e6,
        dtype=torch.float32,
        device=env.device,
    )
    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        point_z = toe_points[..., 2]
        closest_z = torch.minimum(
            torch.maximum(point_z, lower_z[:, None, None]),
            upper_z[:, None, None],
        )
        distance = torch.sqrt(
            torch.square(toe_points[..., 0] - edge_x[:, None, None])
            + torch.square(point_z - closest_z)
            + 1.0e-9
        )
        min_distance = torch.minimum(min_distance, distance.amin(dim=-1))

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
    reference_contrast = reference_speed - reference_speed.amin(
        dim=-1, keepdim=True
    )
    reference_swing = (
        (reference_speed >= float(reference_swing_speed_threshold))
        & (reference_contrast >= float(reference_swing_contrast_threshold))
    )

    sole_samples_w = _body_sample_points_w(command, body_indexes, sole_points)
    relative_x = sole_samples_w[..., 0] - env.scene.env_origins[:, None, None, 0]
    ground_z = _fixed_stairs_height(relative_x, riser_x_offsets, tread_heights)
    ground_z = ground_z + env.scene.env_origins[:, None, None, 2]
    max_gap = torch.abs(sole_samples_w[..., 2] - ground_z).amax(dim=-1)
    foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
    settled_support = (
        (contact_force >= float(settled_contact_force_threshold))
        & (foot_speed <= float(settled_speed_threshold))
        & (max_gap <= float(settled_max_gap))
    )
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
    active = phase_active & reference_swing & ~settled_support

    warning_shortage = torch.relu(
        (float(safety_distance) - min_distance) / float(safety_distance)
    )
    hard_shortage = torch.relu(
        (float(hard_distance) - min_distance) / float(hard_distance)
    )
    normalized_contact = torch.clamp(
        (contact_force - float(near_contact_force_threshold))
        / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    near_contact = (
        active
        & (min_distance < float(safety_distance))
        & (contact_force >= float(near_contact_force_threshold))
    )
    per_foot = (
        torch.square(warning_shortage)
        + float(hard_penalty_scale) * torch.square(hard_shortage)
        + float(contact_penalty_scale)
        * warning_shortage
        * normalized_contact
        * near_contact.float()
    )
    per_foot = torch.clamp(per_foot * active.float(), max=6.0)
    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    reported_min = torch.where(
        active,
        min_distance,
        torch.full_like(min_distance, float(safety_distance)),
    )
    violation = active & (min_distance < float(safety_distance))

    command.metrics["toe_barrier_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["toe_barrier_violation_fraction"] = (
        violation.float().sum(dim=-1) / active_count
    )
    command.metrics["toe_barrier_near_contact_fraction"] = (
        near_contact.float().sum(dim=-1) / active_count
    )
    command.metrics["toe_barrier_min_distance"] = reported_min.amin(dim=-1)
    if len(body_indexes) >= 2:
        command.metrics["left_toe_barrier_min_distance"] = reported_min[:, 0]
        command.metrics["right_toe_barrier_min_distance"] = reported_min[:, 1]
    return per_foot.sum(dim=-1) / active_count


class motion_swing_running_min_toe_riser_barrier_penalty(ManagerTermBase):
    """Retain the worst toe-riser clearance over each reference swing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.safety_distance = float(cfg.params["safety_distance"])
        self.running_min: torch.Tensor | None = None
        self.previous_active: torch.Tensor | None = None
        self.previous_frame: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        if self.running_min is None:
            return
        if env_ids is None:
            self.running_min.fill_(self.safety_distance)
            self.previous_active.zero_()
            self.previous_frame.fill_(-1)
        else:
            self.running_min[env_ids] = self.safety_distance
            self.previous_active[env_ids] = False
            self.previous_frame[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        frame_ranges: list[tuple[int, int]],
        riser_x_offsets: list[float],
        upper_heights: list[float],
        lower_heights: list[float],
        tread_heights: list[float],
        safety_distance: float,
        hard_distance: float,
        near_contact_force_threshold: float,
        contact_force_scale: float,
        hard_penalty_scale: float,
        contact_penalty_scale: float,
        persistence_scale: float,
        new_minimum_scale: float,
        terminal_scale: float,
        reference_swing_speed_threshold: float,
        reference_swing_contrast_threshold: float,
        settled_contact_force_threshold: float,
        settled_speed_threshold: float,
        settled_max_gap: float,
        local_points: list[tuple[float, float, float]],
        sole_points: list[tuple[float, float, float]],
        body_names: list[str],
    ) -> torch.Tensor:
        if not (
            len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
        ):
            raise ValueError("riser geometry tables must have identical lengths")

        command: MotionCommand = env.command_manager.get_term(command_name)
        body_indexes = _get_body_indexes(command, body_names)
        toe_points = _body_sample_points_w(command, body_indexes, local_points)
        current_min = torch.full(
            (env.num_envs, len(body_indexes)),
            1.0e6,
            dtype=torch.float32,
            device=env.device,
        )
        for edge_offset, upper_height, lower_height in zip(
            riser_x_offsets, upper_heights, lower_heights
        ):
            edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
            upper_z = env.scene.env_origins[:, 2] + float(upper_height)
            lower_z = env.scene.env_origins[:, 2] + float(lower_height)
            point_z = toe_points[..., 2]
            closest_z = torch.minimum(
                torch.maximum(point_z, lower_z[:, None, None]),
                upper_z[:, None, None],
            )
            distance = torch.sqrt(
                torch.square(toe_points[..., 0] - edge_x[:, None, None])
                + torch.square(point_z - closest_z)
                + 1.0e-9
            )
            current_min = torch.minimum(current_min, distance.amin(dim=-1))

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact_force = contact_sensor.data.net_forces_w[
            :, sensor_cfg.body_ids, :
        ].norm(dim=-1)
        reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
        reference_contrast = reference_speed - reference_speed.amin(
            dim=-1, keepdim=True
        )
        reference_swing = (
            (reference_speed >= float(reference_swing_speed_threshold))
            & (reference_contrast >= float(reference_swing_contrast_threshold))
        )

        sole_samples_w = _body_sample_points_w(command, body_indexes, sole_points)
        relative_x = sole_samples_w[..., 0] - env.scene.env_origins[:, None, None, 0]
        ground_z = _fixed_stairs_height(relative_x, riser_x_offsets, tread_heights)
        ground_z = ground_z + env.scene.env_origins[:, None, None, 2]
        max_gap = torch.abs(sole_samples_w[..., 2] - ground_z).amax(dim=-1)
        foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
        settled_support = (
            (contact_force >= float(settled_contact_force_threshold))
            & (foot_speed <= float(settled_speed_threshold))
            & (max_gap <= float(settled_max_gap))
        )
        phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
        active = phase_active & reference_swing & ~settled_support

        if self.running_min is None or self.running_min.shape != current_min.shape:
            self.running_min = torch.full_like(current_min, self.safety_distance)
            self.previous_active = torch.zeros_like(active)
            self.previous_frame = torch.full_like(command.time_steps, -1)

        assert self.previous_active is not None
        assert self.previous_frame is not None
        sequence_reset = command.time_steps < self.previous_frame
        swing_start = active & (
            ~self.previous_active | sequence_reset[:, None]
        )
        running_before = torch.where(
            swing_start,
            torch.full_like(self.running_min, float(safety_distance)),
            self.running_min,
        )
        running_after = torch.where(
            active,
            torch.minimum(running_before, current_min),
            running_before,
        )
        new_minimum = torch.relu(
            (running_before - running_after) / float(safety_distance)
        )

        terminal = self.previous_active & ~active & ~sequence_reset[:, None]
        terminal_min = torch.where(terminal, self.running_min, running_after)
        active_shortage = torch.relu(
            (float(safety_distance) - running_after) / float(safety_distance)
        )
        active_hard_shortage = torch.relu(
            (float(hard_distance) - running_after) / float(hard_distance)
        )
        terminal_shortage = torch.relu(
            (float(safety_distance) - terminal_min) / float(safety_distance)
        )
        terminal_hard_shortage = torch.relu(
            (float(hard_distance) - terminal_min) / float(hard_distance)
        )
        active_risk = torch.square(active_shortage) + float(hard_penalty_scale) * torch.square(
            active_hard_shortage
        )
        terminal_risk = torch.square(terminal_shortage) + float(hard_penalty_scale) * torch.square(
            terminal_hard_shortage
        )

        normalized_contact = torch.clamp(
            (contact_force - float(near_contact_force_threshold))
            / float(contact_force_scale),
            min=0.0,
            max=3.0,
        )
        near_contact = (
            active
            & (current_min < float(safety_distance))
            & (contact_force >= float(near_contact_force_threshold))
        )
        per_foot = active.float() * (
            float(persistence_scale) * active_risk
            + float(new_minimum_scale) * new_minimum
            + float(contact_penalty_scale)
            * active_shortage
            * normalized_contact
            * near_contact.float()
        )
        per_foot = per_foot + terminal.float() * float(terminal_scale) * terminal_risk
        per_foot = torch.clamp(per_foot, max=8.0)

        event = active | terminal
        event_count = event.float().sum(dim=-1).clamp(min=1.0)
        reported_running = torch.where(
            event,
            torch.where(active, running_after, terminal_min),
            torch.full_like(running_after, float(safety_distance)),
        )
        violation = event & (reported_running < float(safety_distance))
        command.metrics["running_min_barrier_active_fraction"] = active.float().mean(dim=-1)
        command.metrics["running_min_barrier_terminal_fraction"] = terminal.float().mean(dim=-1)
        command.metrics["running_min_barrier_new_minimum_fraction"] = (
            ((new_minimum > 0.0) & active).float().sum(dim=-1) / event_count
        )
        command.metrics["running_min_barrier_violation_fraction"] = (
            violation.float().sum(dim=-1) / event_count
        )
        command.metrics["running_min_barrier_near_contact_fraction"] = (
            near_contact.float().sum(dim=-1) / event_count
        )
        command.metrics["running_min_barrier_current_distance"] = torch.where(
            active,
            current_min,
            torch.full_like(current_min, float(safety_distance)),
        ).amin(dim=-1)
        command.metrics["running_min_barrier_min_distance"] = reported_running.amin(dim=-1)
        if len(body_indexes) >= 2:
            command.metrics["left_running_min_barrier_distance"] = reported_running[:, 0]
            command.metrics["right_running_min_barrier_distance"] = reported_running[:, 1]

        self.running_min = torch.where(
            active,
            running_after,
            torch.full_like(running_after, float(safety_distance)),
        ).detach()
        self.previous_active = active.detach().clone()
        self.previous_frame = command.time_steps.detach().clone()
        return per_foot.sum(dim=-1) / event_count


def motion_time_to_riser_cone_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    approach_distance: float,
    prediction_horizon: float,
    time_scale: float,
    safety_height: float,
    hard_height: float,
    minimum_forward_speed: float,
    reference_swing_speed_threshold: float,
    reference_swing_contrast_threshold: float,
    minimum_toe_up: float,
    toe_pitch_scale: float,
    near_contact_force_threshold: float,
    contact_force_scale: float,
    contact_penalty_scale: float,
    forefoot_points: list[tuple[float, float, float]],
    heel_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Penalize a forefoot's time-to-collision cone before it reaches a riser."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    forefoot_w = _body_sample_points_w(command, body_indexes, forefoot_points)
    heel_w = _body_sample_points_w(command, body_indexes, heel_points)
    body_pos = command.robot_body_pos_w[:, body_indexes]
    point_offsets = forefoot_w - body_pos[:, :, None, :]
    body_ang_vel = command.robot_body_ang_vel_w[:, body_indexes, None, :].expand_as(
        point_offsets
    )
    point_velocity = (
        command.robot_body_lin_vel_w[:, body_indexes, None, :]
        + torch.cross(body_ang_vel, point_offsets, dim=-1)
    )

    reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
    reference_contrast = reference_speed - reference_speed.amin(
        dim=-1, keepdim=True
    )
    reference_swing = (
        (reference_speed >= float(reference_swing_speed_threshold))
        & (reference_contrast >= float(reference_swing_contrast_threshold))
    )
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]

    per_foot_risk = torch.zeros(
        (env.num_envs, len(body_indexes)), dtype=torch.float32, device=env.device
    )
    active = torch.zeros_like(per_foot_risk, dtype=torch.bool)
    predicted_clearance = torch.full_like(per_foot_risk, float(safety_height))

    point_x = forefoot_w[..., 0]
    point_z = forefoot_w[..., 2]
    point_vx = point_velocity[..., 0]
    point_vz = point_velocity[..., 2]
    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        dx = edge_x[:, None, None] - point_x
        safe_vx = torch.clamp(point_vx, min=float(minimum_forward_speed))
        time_to_edge = dx / safe_vx
        candidate = (
            phase_active[:, :, None]
            & reference_swing[:, :, None]
            & (point_vx >= float(minimum_forward_speed))
            & (dx >= 0.0)
            & (dx <= float(approach_distance))
            & (time_to_edge <= float(prediction_horizon))
        )

        crossing_z = point_z + point_vz * time_to_edge
        clearance_above_riser = crossing_z - upper_z[:, None, None]
        warning_shortage = torch.clamp(
            torch.relu(float(safety_height) - clearance_above_riser)
            / float(safety_height),
            max=3.0,
        )
        hard_shortage = torch.clamp(
            torch.relu(float(hard_height) - clearance_above_riser)
            / float(hard_height),
            max=3.0,
        )
        urgency = torch.exp(
            -torch.clamp(time_to_edge, min=0.0) / float(time_scale)
        )
        point_risk = urgency * (
            torch.square(warning_shortage) + 2.0 * torch.square(hard_shortage)
        )
        point_risk = point_risk * candidate.float()
        edge_risk = point_risk.amax(dim=-1)
        edge_active = candidate.any(dim=-1)
        edge_clearance = torch.where(
            candidate,
            clearance_above_riser,
            torch.full_like(clearance_above_riser, float(safety_height)),
        ).amin(dim=-1)
        per_foot_risk = torch.maximum(per_foot_risk, edge_risk)
        predicted_clearance = torch.where(
            edge_active,
            torch.minimum(predicted_clearance, edge_clearance),
            predicted_clearance,
        )
        active |= edge_active

    toe_z = forefoot_w[..., 2].amin(dim=-1)
    heel_z = heel_w[..., 2].amin(dim=-1)
    toe_up = toe_z - heel_z
    toe_up_shortage = torch.clamp(
        torch.relu(float(minimum_toe_up) - toe_up) / float(minimum_toe_up),
        max=2.0,
    )
    per_foot_risk = per_foot_risk + (
        float(toe_pitch_scale) * torch.square(toe_up_shortage) * active.float()
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    near_contact = (
        active
        & (predicted_clearance < float(safety_height))
        & (contact_force >= float(near_contact_force_threshold))
    )
    normalized_contact = torch.clamp(
        (contact_force - float(near_contact_force_threshold))
        / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    per_foot_risk = per_foot_risk + (
        float(contact_penalty_scale)
        * normalized_contact
        * near_contact.float()
    )
    per_foot_risk = torch.clamp(per_foot_risk, max=8.0)
    active_count = active.float().sum(dim=-1).clamp(min=1.0)

    command.metrics["ttc_riser_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["ttc_riser_violation_fraction"] = (
        (active & (predicted_clearance < float(safety_height))).float().sum(dim=-1)
        / active_count
    )
    command.metrics["ttc_riser_hard_fraction"] = (
        (active & (predicted_clearance < float(hard_height))).float().sum(dim=-1)
        / active_count
    )
    command.metrics["ttc_riser_near_contact_fraction"] = (
        near_contact.float().sum(dim=-1) / active_count
    )
    command.metrics["ttc_riser_predicted_clearance"] = predicted_clearance.amin(
        dim=-1
    )
    command.metrics["ttc_riser_toe_up_shortage"] = (
        toe_up_shortage * active.float()
    ).sum(dim=-1) / active_count
    if len(body_indexes) >= 2:
        command.metrics["left_ttc_riser_clearance"] = predicted_clearance[:, 0]
        command.metrics["right_ttc_riser_clearance"] = predicted_clearance[:, 1]
    return per_foot_risk.sum(dim=-1) / active_count


def motion_spatial_riser_corridor_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    window_before: float,
    window_after: float,
    corridor_sigma: float,
    base_clearance: float,
    peak_clearance: float,
    hard_clearance: float,
    reference_swing_speed_threshold: float,
    reference_swing_contrast_threshold: float,
    minimum_toe_up: float,
    toe_pitch_scale: float,
    near_contact_force_threshold: float,
    contact_force_scale: float,
    contact_penalty_scale: float,
    forefoot_points: list[tuple[float, float, float]],
    heel_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Shape a velocity-independent spatial safety corridor around each riser."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    forefoot_w = _body_sample_points_w(command, body_indexes, forefoot_points)
    heel_w = _body_sample_points_w(command, body_indexes, heel_points)
    reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
    reference_contrast = reference_speed - reference_speed.amin(
        dim=-1, keepdim=True
    )
    reference_swing = (
        (reference_speed >= float(reference_swing_speed_threshold))
        & (reference_contrast >= float(reference_swing_contrast_threshold))
    )
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]

    per_foot_risk = torch.zeros(
        (env.num_envs, len(body_indexes)), dtype=torch.float32, device=env.device
    )
    active = torch.zeros_like(per_foot_risk, dtype=torch.bool)
    corridor_clearance = torch.full_like(per_foot_risk, float(peak_clearance))
    point_x = forefoot_w[..., 0]
    point_z = forefoot_w[..., 2]

    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        del lower_height
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        relative_x = point_x - edge_x[:, None, None]
        candidate = (
            phase_active[:, :, None]
            & reference_swing[:, :, None]
            & (relative_x >= -float(window_before))
            & (relative_x <= float(window_after))
        )
        gaussian = torch.exp(
            -0.5 * torch.square(relative_x / float(corridor_sigma))
        )
        required_clearance = float(base_clearance) + float(peak_clearance) * gaussian
        actual_clearance = point_z - upper_z[:, None, None]
        shortage = torch.clamp(
            torch.relu(required_clearance - actual_clearance)
            / float(peak_clearance),
            max=3.0,
        )
        hard_shortage = torch.clamp(
            torch.relu(float(hard_clearance) - actual_clearance)
            / float(hard_clearance),
            max=3.0,
        )
        spatial_weight = 0.25 + 0.75 * gaussian
        point_risk = spatial_weight * (
            torch.square(shortage) + 2.0 * gaussian * torch.square(hard_shortage)
        )
        point_risk = point_risk * candidate.float()
        edge_risk = point_risk.amax(dim=-1)
        edge_active = candidate.any(dim=-1)
        edge_clearance = torch.where(
            candidate,
            actual_clearance,
            torch.full_like(actual_clearance, float(peak_clearance)),
        ).amin(dim=-1)
        per_foot_risk = torch.maximum(per_foot_risk, edge_risk)
        corridor_clearance = torch.where(
            edge_active,
            torch.minimum(corridor_clearance, edge_clearance),
            corridor_clearance,
        )
        active |= edge_active

    toe_z = forefoot_w[..., 2].amin(dim=-1)
    heel_z = heel_w[..., 2].amin(dim=-1)
    toe_up = toe_z - heel_z
    toe_up_shortage = torch.clamp(
        torch.relu(float(minimum_toe_up) - toe_up) / float(minimum_toe_up),
        max=2.0,
    )
    per_foot_risk = per_foot_risk + (
        float(toe_pitch_scale) * torch.square(toe_up_shortage) * active.float()
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    near_contact = (
        active
        & (corridor_clearance < float(base_clearance))
        & (contact_force >= float(near_contact_force_threshold))
    )
    normalized_contact = torch.clamp(
        (contact_force - float(near_contact_force_threshold))
        / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    per_foot_risk = per_foot_risk + (
        float(contact_penalty_scale)
        * normalized_contact
        * near_contact.float()
    )
    per_foot_risk = torch.clamp(per_foot_risk, max=8.0)
    active_count = active.float().sum(dim=-1).clamp(min=1.0)

    command.metrics["spatial_corridor_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["spatial_corridor_violation_fraction"] = (
        (active & (corridor_clearance < float(peak_clearance))).float().sum(dim=-1)
        / active_count
    )
    command.metrics["spatial_corridor_hard_fraction"] = (
        (active & (corridor_clearance < float(hard_clearance))).float().sum(dim=-1)
        / active_count
    )
    command.metrics["spatial_corridor_near_contact_fraction"] = (
        near_contact.float().sum(dim=-1) / active_count
    )
    command.metrics["spatial_corridor_clearance"] = corridor_clearance.amin(dim=-1)
    command.metrics["spatial_corridor_toe_up_shortage"] = (
        toe_up_shortage * active.float()
    ).sum(dim=-1) / active_count
    if len(body_indexes) >= 2:
        command.metrics["left_spatial_corridor_clearance"] = corridor_clearance[:, 0]
        command.metrics["right_spatial_corridor_clearance"] = corridor_clearance[:, 1]
    return per_foot_risk.sum(dim=-1) / active_count


class motion_riser_clearance_lagrangian_cost(ManagerTermBase):
    """Penalize rigid-foot riser risk with an episode-preserving dual budget."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dual_multiplier = torch.tensor(
            float(cfg.params["initial_dual_multiplier"]),
            dtype=torch.float32,
            device=env.device,
        )
        self.cost_ema = torch.tensor(
            float(cfg.params["cost_budget"]),
            dtype=torch.float32,
            device=env.device,
        )
        self.update_step = 0

    def reset(self, env_ids=None) -> None:
        # The dual variable is global optimization state and must span episodes.
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        frame_ranges: list[tuple[int, int]],
        riser_x_offsets: list[float],
        upper_heights: list[float],
        lower_heights: list[float],
        window_before: float,
        window_after: float,
        safety_distance: float,
        hard_distance: float,
        hard_cost_scale: float,
        cost_budget: float,
        dual_learning_rate: float,
        dual_ema_decay: float,
        min_dual_multiplier: float,
        max_dual_multiplier: float,
        update_period_steps: int,
        initial_dual_multiplier: float,
        reference_swing_speed_threshold: float,
        reference_swing_contrast_threshold: float,
        local_points: list[tuple[float, float, float]],
        body_names: list[str],
    ) -> torch.Tensor:
        del initial_dual_multiplier
        if not (
            len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
        ):
            raise ValueError("riser geometry tables must have identical lengths")
        if hard_distance < 0.0 or safety_distance <= hard_distance:
            raise ValueError("safety_distance must be greater than hard_distance >= 0")
        if not 0.0 <= dual_ema_decay < 1.0:
            raise ValueError("dual_ema_decay must be in [0, 1)")
        if update_period_steps <= 0:
            raise ValueError("update_period_steps must be positive")

        command: MotionCommand = env.command_manager.get_term(command_name)
        body_indexes = _get_body_indexes(command, body_names)
        sample_points = _body_sample_points_w(command, body_indexes, local_points)
        reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
        reference_contrast = reference_speed - reference_speed.amin(
            dim=-1, keepdim=True
        )
        reference_swing = (
            (reference_speed >= float(reference_swing_speed_threshold))
            & (reference_contrast >= float(reference_swing_contrast_threshold))
        )
        phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]

        num_feet = len(body_indexes)
        active = torch.zeros(
            (env.num_envs, num_feet), dtype=torch.bool, device=env.device
        )
        min_distance = torch.full(
            (env.num_envs, num_feet),
            float(safety_distance),
            dtype=torch.float32,
            device=env.device,
        )
        point_x = sample_points[..., 0]
        point_z = sample_points[..., 2]

        for edge_offset, upper_height, lower_height in zip(
            riser_x_offsets, upper_heights, lower_heights
        ):
            edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
            upper_z = env.scene.env_origins[:, 2] + float(upper_height)
            lower_z = env.scene.env_origins[:, 2] + float(lower_height)
            relative_x = point_x - edge_x[:, None, None]
            candidate = (
                phase_active[:, :, None]
                & reference_swing[:, :, None]
                & (relative_x >= -float(window_before))
                & (relative_x <= float(window_after))
            )
            closest_z = torch.minimum(
                torch.maximum(point_z, lower_z[:, None, None]),
                upper_z[:, None, None],
            )
            point_distance = torch.sqrt(
                torch.square(relative_x)
                + torch.square(point_z - closest_z)
                + 1.0e-12
            )
            edge_active = candidate.any(dim=-1)
            edge_distance = torch.where(
                candidate,
                point_distance,
                torch.full_like(point_distance, float(safety_distance)),
            ).amin(dim=-1)
            min_distance = torch.where(
                edge_active,
                torch.minimum(min_distance, edge_distance),
                min_distance,
            )
            active |= edge_active

        safety_span = float(safety_distance) - float(hard_distance)
        normalized_shortage = torch.clamp(
            torch.relu(float(safety_distance) - min_distance) / safety_span,
            max=2.0,
        )
        hard_shortage = torch.clamp(
            torch.relu(float(hard_distance) - min_distance)
            / max(float(hard_distance), 1.0e-6),
            max=2.0,
        )
        per_foot_cost = active.float() * (
            torch.square(normalized_shortage)
            + float(hard_cost_scale) * torch.square(hard_shortage)
        )
        active_count = active.float().sum(dim=-1).clamp(min=1.0)
        raw_cost = per_foot_cost.sum(dim=-1) / active_count

        global_active_count = active.float().sum()
        if bool(global_active_count > 0):
            active_cost_mean = per_foot_cost.sum() / global_active_count
            decay = float(dual_ema_decay)
            self.cost_ema = (
                decay * self.cost_ema
                + (1.0 - decay) * active_cost_mean.detach()
            )
            self.update_step += 1
            if self.update_step % int(update_period_steps) == 0:
                dual_update = float(dual_learning_rate) * (
                    self.cost_ema - float(cost_budget)
                )
                self.dual_multiplier = torch.clamp(
                    self.dual_multiplier + dual_update,
                    min=float(min_dual_multiplier),
                    max=float(max_dual_multiplier),
                ).detach()

        violation = active & (min_distance < float(safety_distance))
        hard_violation = active & (min_distance < float(hard_distance))
        command.metrics["lagrangian_clearance_active_fraction"] = active.float().mean(
            dim=-1
        )
        command.metrics["lagrangian_clearance_violation_fraction"] = (
            violation.float().sum(dim=-1) / active_count
        )
        command.metrics["lagrangian_clearance_hard_fraction"] = (
            hard_violation.float().sum(dim=-1) / active_count
        )
        command.metrics["lagrangian_clearance_raw_cost"] = raw_cost
        command.metrics["lagrangian_clearance_min_distance"] = min_distance.amin(
            dim=-1
        )
        command.metrics["lagrangian_clearance_dual_multiplier"] = (
            torch.ones_like(raw_cost) * self.dual_multiplier
        )
        command.metrics["lagrangian_clearance_cost_ema"] = (
            torch.ones_like(raw_cost) * self.cost_ema
        )
        if num_feet >= 2:
            command.metrics["left_lagrangian_clearance_distance"] = min_distance[:, 0]
            command.metrics["right_lagrangian_clearance_distance"] = min_distance[:, 1]
        return self.dual_multiplier.detach() * raw_cost


class motion_shared_riser_clearance_cost(ManagerTermBase):
    """Expose a shared actual/planned-swing rigid-foot cost without reward coupling."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        frame_ranges: list[tuple[int, int]],
        riser_x_offsets: list[float],
        upper_heights: list[float],
        lower_heights: list[float],
        window_before: float,
        window_after: float,
        safety_distance: float,
        hard_distance: float,
        hard_cost_scale: float,
        actual_swing_speed_threshold: float,
        support_force_threshold: float,
        reference_swing_speed_threshold: float,
        reference_swing_contrast_threshold: float,
        local_points: list[tuple[float, float, float]],
        body_names: list[str],
        require_low_contact_for_swing: bool = False,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        body_indexes = _get_body_indexes(command, body_names)
        sample_points = _body_sample_points_w(command, body_indexes, local_points)
        reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
        reference_contrast = reference_speed - reference_speed.amin(
            dim=-1, keepdim=True
        )
        planned_swing = (
            (reference_speed >= float(reference_swing_speed_threshold))
            & (reference_contrast >= float(reference_swing_contrast_threshold))
        )
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(
            dim=-1
        )
        result = rigid_foot_riser_clearance_cost(
            sample_points_w=sample_points,
            foot_lin_vel_w=command.robot_body_lin_vel_w[:, body_indexes],
            contact_force_n=contact_force,
            env_origins=env.scene.env_origins,
            phase_active=_motion_frame_ranges_mask(command, frame_ranges),
            planned_swing=planned_swing,
            riser_x_offsets=riser_x_offsets,
            upper_heights=upper_heights,
            lower_heights=lower_heights,
            window_before=window_before,
            window_after=window_after,
            safety_distance=safety_distance,
            hard_distance=hard_distance,
            hard_cost_scale=hard_cost_scale,
            actual_swing_speed_threshold=actual_swing_speed_threshold,
            support_force_threshold=support_force_threshold,
            require_low_contact_for_swing=require_low_contact_for_swing,
        )
        active = result["active"]
        active_count = active.float().sum(dim=-1).clamp(min=1.0)
        command.metrics["shared_clearance_cost"] = result["cost"]
        command.metrics["shared_clearance_active_fraction"] = active.float().mean(
            dim=-1
        )
        command.metrics["shared_clearance_actual_swing_fraction"] = result[
            "actual_swing"
        ].float().mean(dim=-1)
        command.metrics["shared_clearance_planned_swing_fraction"] = result[
            "planned_swing"
        ].float().mean(dim=-1)
        command.metrics["shared_clearance_violation_fraction"] = result[
            "violation"
        ].float().sum(dim=-1) / active_count
        command.metrics["shared_clearance_hard_fraction"] = result[
            "hard_violation"
        ].float().sum(dim=-1) / active_count
        command.metrics["shared_clearance_min_distance"] = result[
            "reported_min_distance"
        ].amin(dim=-1)
        command.metrics["left_shared_clearance_distance"] = result[
            "reported_min_distance"
        ][:, 0]
        command.metrics["right_shared_clearance_distance"] = result[
            "reported_min_distance"
        ][:, 1]
        return result["cost"]


def motion_pre_touchdown_soft_landing_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    tread_heights: list[float],
    approach_height: float,
    below_tolerance: float,
    safe_downward_speed: float,
    downward_speed_scale: float,
    reference_swing_speed_threshold: float,
    reference_swing_contrast_threshold: float,
    contact_force_threshold: float,
    contact_force_scale: float,
    impact_scale: float,
    sole_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Densely slow descending feet before touchdown, then penalize hard contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    sole_w = _body_sample_points_w(command, body_indexes, sole_points)

    relative_x = sole_w[..., 0] - env.scene.env_origins[:, None, None, 0]
    ground_z = _fixed_stairs_height(
        relative_x, riser_x_offsets, tread_heights
    ) + env.scene.env_origins[:, None, None, 2]
    foot_gap = (sole_w[..., 2] - ground_z).amin(dim=-1)

    reference_speed = command.body_lin_vel_w[:, body_indexes].norm(dim=-1)
    reference_contrast = reference_speed - reference_speed.amin(
        dim=-1, keepdim=True
    )
    reference_swing = (
        (reference_speed >= float(reference_swing_speed_threshold))
        & (reference_contrast >= float(reference_swing_contrast_threshold))
    )
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
    active = (
        phase_active
        & reference_swing
        & (foot_gap <= float(approach_height))
        & (foot_gap >= -float(below_tolerance))
    )

    downward_speed = torch.relu(
        -command.robot_body_lin_vel_w[:, body_indexes, 2]
        - float(safe_downward_speed)
    )
    proximity = torch.clamp(
        (float(approach_height) - torch.clamp(foot_gap, min=0.0))
        / float(approach_height),
        min=0.0,
        max=1.0,
    )
    velocity_penalty = torch.square(
        torch.clamp(downward_speed / float(downward_speed_scale), max=3.0)
    ) * proximity * active.float()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    impact = (
        active
        & (foot_gap <= 0.015)
        & (contact_force >= float(contact_force_threshold))
    )
    normalized_force = torch.clamp(
        (contact_force - float(contact_force_threshold))
        / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    impact_penalty = (
        float(impact_scale) * torch.square(normalized_force) * impact.float()
    )
    active_count = active.float().sum(dim=-1).clamp(min=1.0)

    command.metrics["soft_landing_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["soft_landing_downward_speed"] = (
        downward_speed * active.float()
    ).sum(dim=-1) / active_count
    command.metrics["soft_landing_velocity_penalty"] = (
        velocity_penalty.sum(dim=-1) / active_count
    )
    command.metrics["soft_landing_impact_fraction"] = (
        impact.float().sum(dim=-1) / active_count
    )
    command.metrics["soft_landing_contact_force"] = (
        contact_force * impact.float()
    ).sum(dim=-1) / impact.float().sum(dim=-1).clamp(min=1.0)
    if len(body_indexes) >= 2:
        command.metrics["left_soft_landing_downward_speed"] = downward_speed[:, 0]
        command.metrics["right_soft_landing_downward_speed"] = downward_speed[:, 1]
        command.metrics["left_soft_landing_impact"] = impact[:, 0].float()
        command.metrics["right_soft_landing_impact"] = impact[:, 1].float()
    return (velocity_penalty + impact_penalty).sum(dim=-1) / active_count


def motion_riser_risk_gated_feet_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    safety_distance: float,
    lookahead_time: float,
    approach_distance: float,
    release_distance: float,
    min_forward_speed: float,
    minimum_tracking_weight: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Relax unsafe foot imitation locally while retaining full-course tracking."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    per_foot_reward = torch.exp(-error / float(std) ** 2)
    predicted_min, _, active, _ = _fixed_stairs_predictive_sweep_distance(
        env=env,
        command=command,
        body_indexes=body_indexes,
        frame_ranges=frame_ranges,
        riser_x_offsets=riser_x_offsets,
        upper_heights=upper_heights,
        lower_heights=lower_heights,
        local_points=local_points,
        lookahead_time=lookahead_time,
        approach_distance=approach_distance,
        release_distance=release_distance,
        min_forward_speed=min_forward_speed,
    )
    risk = torch.relu(
        (float(safety_distance) - predicted_min) / float(safety_distance)
    ) * active.float()
    multiplier = 1.0 - (1.0 - float(minimum_tracking_weight)) * risk
    command.metrics["sweep_feet_tracking_multiplier"] = multiplier.mean(dim=-1)
    return (per_foot_reward * multiplier).mean(dim=-1)


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute mechanical joint power."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(power), dim=1)


class action_smoothness_l2(ManagerTermBase):
    """Penalize the second finite difference of policy actions."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.prev_prev_action: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        if self.prev_prev_action is None:
            return
        if env_ids is None:
            self.prev_prev_action.zero_()
        else:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        curvature = (
            env.action_manager.action
            - 2.0 * env.action_manager.prev_action
            + self.prev_prev_action
        )
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return torch.sum(torch.square(curvature), dim=1)


def motion_fixed_stairs_toe_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    safety_distance: float,
    swing_speed_threshold: float,
    contact_force_threshold: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Apply T-GMP's cubic toe-distance cost to both swing feet and all risers."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    toe_points = _body_sample_points_w(command, body_indexes, local_points)
    min_distance = torch.full(
        (env.num_envs, len(body_indexes)),
        1.0e6,
        dtype=torch.float32,
        device=env.device,
    )

    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        point_z = toe_points[..., 2]
        closest_z = torch.minimum(
            torch.maximum(point_z, lower_z[:, None, None]),
            upper_z[:, None, None],
        )
        dx = toe_points[..., 0] - edge_x[:, None, None]
        dz = point_z - closest_z
        distance = torch.sqrt(torch.square(dx) + torch.square(dz) + 1.0e-9)
        min_distance = torch.minimum(min_distance, distance.amin(dim=-1))

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
    swing = (
        (contact_force < float(contact_force_threshold))
        & (foot_speed > float(swing_speed_threshold))
    )
    shortage = torch.relu(
        (float(safety_distance) - min_distance) / float(safety_distance)
    )
    per_foot_penalty = torch.pow(shortage, 3.0) * swing.float()
    active_count = swing.float().sum(dim=-1).clamp(min=1.0)

    command.metrics["tgmp_toe_min_distance"] = min_distance.amin(dim=-1)
    command.metrics["tgmp_toe_safe_fraction"] = torch.where(
        swing,
        (min_distance >= float(safety_distance)).float(),
        torch.ones_like(min_distance),
    ).mean(dim=-1)
    return per_foot_penalty.sum(dim=-1) / active_count


def motion_fixed_stairs_continuous_riser_safety_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    tread_heights: list[float],
    safety_distance: float,
    near_contact_force_threshold: float,
    settled_contact_force_threshold: float,
    settled_speed_threshold: float,
    settled_max_gap: float,
    contact_force_scale: float,
    contact_penalty_scale: float,
    forefoot_points: list[tuple[float, float, float]],
    sole_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Keep riser clearance active until a foot is settled on one tread."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    toe_points = _body_sample_points_w(command, body_indexes, forefoot_points)
    min_distance = torch.full(
        (env.num_envs, len(body_indexes)),
        1.0e6,
        dtype=torch.float32,
        device=env.device,
    )
    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        point_z = toe_points[..., 2]
        closest_z = torch.minimum(
            torch.maximum(point_z, lower_z[:, None, None]),
            upper_z[:, None, None],
        )
        dx = toe_points[..., 0] - edge_x[:, None, None]
        dz = point_z - closest_z
        distance = torch.sqrt(torch.square(dx) + torch.square(dz) + 1.0e-9)
        min_distance = torch.minimum(min_distance, distance.amin(dim=-1))

    sole_samples_w = _body_sample_points_w(command, body_indexes, sole_points)
    relative_x = (
        sole_samples_w[..., 0] - env.scene.env_origins[:, None, None, 0]
    )
    ground_z = _fixed_stairs_height(
        relative_x, riser_x_offsets, tread_heights
    )
    ground_z = ground_z + env.scene.env_origins[:, None, None, 2]
    max_gap = torch.abs(sole_samples_w[..., 2] - ground_z).amax(dim=-1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
    settled = (
        (contact_force >= float(settled_contact_force_threshold))
        & (foot_speed <= float(settled_speed_threshold))
        & (max_gap <= float(settled_max_gap))
    )
    active = phase_active & ~settled

    shortage = torch.relu(
        (float(safety_distance) - min_distance) / float(safety_distance)
    )
    clearance_penalty = torch.square(shortage) * active.float()
    near_contact = (
        phase_active
        & ~settled
        & (min_distance < float(safety_distance))
        & (contact_force >= float(near_contact_force_threshold))
    )
    normalized_contact = torch.clamp(
        (
            contact_force - float(near_contact_force_threshold)
        ) / float(contact_force_scale),
        min=0.0,
        max=3.0,
    )
    contact_penalty = (
        float(contact_penalty_scale)
        * shortage
        * normalized_contact
        * near_contact.float()
    )
    active_count = active.float().sum(dim=-1).clamp(min=1.0)

    unsafe = active & (min_distance < float(safety_distance))
    command.metrics["descent_toe_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["descent_toe_violation_fraction"] = (
        unsafe.float().sum(dim=-1) / active_count
    )
    command.metrics["descent_near_riser_contact_fraction"] = (
        near_contact.float().sum(dim=-1) / active_count
    )
    command.metrics["descent_toe_min_distance"] = torch.where(
        active,
        min_distance,
        torch.full_like(min_distance, float(safety_distance)),
    ).amin(dim=-1)
    if len(body_indexes) >= 2:
        command.metrics["left_descent_toe_active"] = active[:, 0].float()
        command.metrics["right_descent_toe_active"] = active[:, 1].float()
        command.metrics["left_descent_toe_violation"] = unsafe[:, 0].float()
        command.metrics["right_descent_toe_violation"] = unsafe[:, 1].float()
        command.metrics["left_near_riser_contact"] = near_contact[:, 0].float()
        command.metrics["right_near_riser_contact"] = near_contact[:, 1].float()

    per_foot = clearance_penalty + contact_penalty
    return per_foot.sum(dim=-1) / active_count


def motion_fixed_stairs_bezier_riser_corridor_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    approach_distance: float,
    landing_distance: float,
    arc_height: float,
    clearance_scale: float,
    min_toe_up: float,
    max_toe_up: float,
    swing_speed_threshold: float,
    near_contact_force_threshold: float,
    settled_contact_force_threshold: float,
    contact_force_scale: float,
    forefoot_points: list[tuple[float, float, float]],
    heel_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Shape a toe-up Bezier swing corridor across descending stair risers."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    forefoot_w = _body_sample_points_w(command, body_indexes, forefoot_points)
    heel_w = _body_sample_points_w(command, body_indexes, heel_points)
    leading_x = forefoot_w[..., 0].amax(dim=-1)
    toe_z = forefoot_w[..., 2].amin(dim=-1)
    heel_z = heel_w[..., 2].amin(dim=-1)
    toe_up = toe_z - heel_z

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    foot_speed = command.robot_body_lin_vel_w[:, body_indexes].norm(dim=-1)
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]
    moving = foot_speed > float(swing_speed_threshold)
    unsettled = (
        (contact_force < float(settled_contact_force_threshold)) | moving
    )

    penalty = torch.zeros_like(contact_force)
    active = torch.zeros_like(contact_force, dtype=torch.bool)
    height_shortage_metric = torch.zeros_like(contact_force)
    toe_pitch_shortage_metric = torch.zeros_like(contact_force)
    early_contact = torch.zeros_like(contact_force, dtype=torch.bool)
    span = float(approach_distance + landing_distance)

    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        progress = torch.clamp(
            (leading_x - (edge_x[:, None] - float(approach_distance))) / span,
            min=0.0,
            max=1.0,
        )
        smooth_progress = torch.square(progress) * (3.0 - 2.0 * progress)
        target_z = (
            upper_z[:, None]
            + (lower_z - upper_z)[:, None] * smooth_progress
            + float(arc_height) * 4.0 * progress * (1.0 - progress)
        )
        in_window = (
            (leading_x >= edge_x[:, None] - float(approach_distance))
            & (leading_x <= edge_x[:, None] + float(landing_distance))
        )
        edge_active = phase_active & unsettled & in_window
        moving_active = edge_active & moving
        height_shortage = torch.clamp(
            torch.relu(target_z - toe_z) / float(clearance_scale), max=2.0
        )

        tangent_window = (
            (leading_x >= edge_x[:, None] - 0.060)
            & (leading_x <= edge_x[:, None] + 0.040)
        )
        tangent_active = moving_active & tangent_window
        toe_up_shortage = torch.clamp(
            torch.relu(float(min_toe_up) - toe_up) / float(min_toe_up), max=2.0
        )
        toe_up_excess = torch.clamp(
            torch.relu(toe_up - float(max_toe_up)) / float(max_toe_up), max=2.0
        )
        contact_active = (
            edge_active
            & (height_shortage > 0.0)
            & (contact_force >= float(near_contact_force_threshold))
        )
        normalized_contact = torch.clamp(
            (contact_force - float(near_contact_force_threshold))
            / float(contact_force_scale),
            min=0.0,
            max=2.0,
        )

        edge_penalty = (
            torch.square(height_shortage) * moving_active.float()
            + 0.5
            * (torch.square(toe_up_shortage) + torch.square(toe_up_excess))
            * tangent_active.float()
            + height_shortage * normalized_contact * contact_active.float()
        )
        penalty = penalty + edge_penalty
        active |= edge_active
        height_shortage_metric = torch.maximum(
            height_shortage_metric, height_shortage * moving_active.float()
        )
        toe_pitch_shortage_metric = torch.maximum(
            toe_pitch_shortage_metric,
            toe_up_shortage * tangent_active.float(),
        )
        early_contact |= contact_active

    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    penalty = torch.clamp(penalty, max=3.0)
    command.metrics["bezier_corridor_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["bezier_height_shortage"] = (
        height_shortage_metric.sum(dim=-1) / active_count
    )
    command.metrics["bezier_toe_up_shortage"] = (
        toe_pitch_shortage_metric.sum(dim=-1) / active_count
    )
    command.metrics["bezier_early_riser_contact"] = (
        early_contact.float().sum(dim=-1) / active_count
    )
    return penalty.sum(dim=-1) / active_count


def motion_fixed_stairs_foothold_sequence_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    tread_edges: list[float],
    tread_heights: list[float],
    foot_center_offset: float,
    ankle_height: float,
    std_x: float,
    std_y: float,
    std_z: float,
    contact_force_threshold: float,
    body_names: list[str],
) -> torch.Tensor:
    """Reward the next pair of three-dimensional footholds without hard gating."""
    if not (len(frame_ranges) == len(tread_edges) == len(tread_heights)):
        raise ValueError("foothold sequence tables must have identical lengths")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    foot_pos = command.robot_body_pos_w[:, body_indexes]
    reference_y = command.body_pos_w[:, body_indexes, 1]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    contact_confidence = torch.clamp(
        contact_force / float(contact_force_threshold), min=0.0, max=1.0
    )

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    target_error = torch.zeros_like(reward)
    target_success = torch.zeros_like(reward)
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for frame_range, tread_edge, tread_height in zip(
        frame_ranges, tread_edges, tread_heights
    ):
        in_range = (
            (command.time_steps >= int(frame_range[0]))
            & (command.time_steps <= int(frame_range[1]))
        )
        target_x = env.scene.env_origins[:, 0] + float(tread_edge + foot_center_offset)
        target_z = env.scene.env_origins[:, 2] + float(tread_height + ankle_height)
        dx = foot_pos[..., 0] - target_x[:, None]
        dy = foot_pos[..., 1] - reference_y
        dz = foot_pos[..., 2] - target_z[:, None]
        normalized_error = (
            torch.square(dx / float(std_x))
            + torch.square(dy / float(std_y))
            + torch.square(dz / float(std_z))
        )
        per_foot_reward = torch.exp(-normalized_error) * contact_confidence
        contact_count = contact_confidence.sum(dim=-1).clamp(min=1.0)
        distance = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz))
        success = (
            (torch.abs(dx) <= float(std_x))
            & (torch.abs(dy) <= float(std_y))
            & (torch.abs(dz) <= float(std_z))
            & (contact_force >= float(contact_force_threshold))
        )
        reward = reward + per_foot_reward.mean(dim=-1) * in_range.float()
        target_error = target_error + (
            (distance * contact_confidence).sum(dim=-1) / contact_count
        ) * in_range.float()
        target_success = target_success + success.float().mean(dim=-1) * in_range.float()
        active |= in_range

    command.metrics["foothold_target_error"] = target_error
    command.metrics["foothold_target_success"] = target_success
    command.metrics["foothold_sequence_active"] = active.float()
    return reward


def motion_fixed_stairs_support_margin_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    tread_edges: list[float],
    tread_heights: list[float],
    tread_length: float,
    edge_margin: float,
    ankle_height: float,
    height_tolerance: float,
    margin_scale: float,
    contact_force_threshold: float,
    sole_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Keep the complete contacting sole inside each short descending tread."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    foot_pos = command.robot_body_pos_w[:, body_indexes]
    sole_w = _body_sample_points_w(command, body_indexes, sole_points)
    sole_min_x = sole_w[..., 0].amin(dim=-1)
    sole_max_x = sole_w[..., 0].amax(dim=-1)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ].norm(dim=-1)
    phase_active = _motion_frame_ranges_mask(command, frame_ranges).bool()[:, None]

    penalty = torch.zeros_like(contact_force)
    active = torch.zeros_like(contact_force, dtype=torch.bool)
    for tread_edge, tread_height in zip(tread_edges, tread_heights):
        edge_x = env.scene.env_origins[:, 0] + float(tread_edge)
        target_z = env.scene.env_origins[:, 2] + float(tread_height + ankle_height)
        near_tread = (
            (foot_pos[..., 0] >= edge_x[:, None] - 0.08)
            & (foot_pos[..., 0] <= edge_x[:, None] + float(tread_length) + 0.08)
            & (torch.abs(foot_pos[..., 2] - target_z[:, None]) <= float(height_tolerance))
        )
        support = (
            phase_active
            & near_tread
            & (contact_force >= float(contact_force_threshold))
        )
        left_shortage = torch.relu(
            edge_x[:, None] + float(edge_margin) - sole_min_x
        )
        right_shortage = torch.relu(
            sole_max_x
            - (edge_x[:, None] + float(tread_length - edge_margin))
        )
        normalized = torch.clamp(
            (left_shortage + right_shortage) / float(margin_scale), max=1.5
        )
        penalty = penalty + torch.square(normalized) * support.float()
        active |= support

    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    command.metrics["support_margin_bad_fraction"] = (
        ((penalty > 0.0) & active).float().sum(dim=-1) / active_count
    )
    command.metrics["support_margin_penalty"] = penalty.sum(dim=-1) / active_count
    return penalty.sum(dim=-1) / active_count


def motion_frame_ranges_foot_nosing_distance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    upper_heights: list[float],
    lower_heights: list[float],
    safety_distance: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """T-GMP-style cubic distance penalty around each descending stair nosing."""
    if not (
        len(frame_ranges)
        == len(riser_x_offsets)
        == len(upper_heights)
        == len(lower_heights)
    ):
        raise ValueError("nosing geometry tables must have identical lengths")
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    sample_points = _body_sample_points_w(command, body_indexes, local_points)
    min_distance = torch.full(
        (env.num_envs,), 1.0e6, dtype=torch.float32, device=env.device
    )
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for frame_range, edge_offset, upper_height, lower_height in zip(
        frame_ranges, riser_x_offsets, upper_heights, lower_heights
    ):
        in_range = (
            (command.time_steps >= frame_range[0])
            & (command.time_steps <= frame_range[1])
        )
        edge_x = env.scene.env_origins[:, 0] + float(edge_offset)
        upper_z = env.scene.env_origins[:, 2] + float(upper_height)
        lower_z = env.scene.env_origins[:, 2] + float(lower_height)
        point_z = sample_points[..., 2]
        closest_z = torch.minimum(
            torch.maximum(point_z, lower_z[:, None, None]), upper_z[:, None, None]
        )
        dx = sample_points[..., 0] - edge_x[:, None, None]
        dz = point_z - closest_z
        distance = torch.sqrt(torch.square(dx) + torch.square(dz) + 1.0e-9).amin(
            dim=(1, 2)
        )
        min_distance = torch.where(in_range, distance, min_distance)
        active |= in_range

    shortage = torch.relu((float(safety_distance) - min_distance) / float(safety_distance))
    penalty = torch.pow(shortage, 3.0) * active.float()
    command.metrics["left_nosing_min_distance"] = torch.where(
        active, min_distance, torch.zeros_like(min_distance)
    )
    command.metrics["left_nosing_safe_fraction"] = torch.where(
        active, (min_distance >= safety_distance).float(), torch.ones_like(min_distance)
    )
    return penalty


def _fixed_stairs_height(
    relative_x: torch.Tensor,
    riser_x_offsets: list[float],
    tread_heights: list[float],
) -> torch.Tensor:
    if len(tread_heights) != len(riser_x_offsets) + 1:
        raise ValueError("tread_heights must contain one more value than riser_x_offsets")
    height = torch.full_like(relative_x, float(tread_heights[0]))
    for edge_x, next_height in zip(riser_x_offsets, tread_heights[1:]):
        height = torch.where(relative_x >= float(edge_x), float(next_height), height)
    return height


def motion_fixed_stairs_sole_support_distance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    frame_ranges: list[tuple[int, int]],
    riser_x_offsets: list[float],
    tread_heights: list[float],
    max_gap_threshold: float,
    gap_scale: float,
    contact_force_threshold: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Penalize contact when sole samples span an edge or hang above the tread."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    sole_points = _body_sample_points_w(command, body_indexes, local_points)
    relative_x = sole_points[..., 0] - env.scene.env_origins[:, None, None, 0]
    ground_z = _fixed_stairs_height(relative_x, riser_x_offsets, tread_heights)
    ground_z = ground_z + env.scene.env_origins[:, None, None, 2]
    max_gap = torch.abs(sole_points[..., 2] - ground_z).amax(dim=-1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    contact = contact_force >= float(contact_force_threshold)
    normalized = torch.clamp(
        torch.relu(max_gap - float(max_gap_threshold)) / float(gap_scale), max=3.0
    )
    per_foot_penalty = torch.square(normalized) * contact.float()
    contact_count = contact.float().sum(dim=-1).clamp(min=1.0)
    penalty = per_foot_penalty.sum(dim=-1) / contact_count
    active = _motion_frame_ranges_mask(command, frame_ranges)

    contacted_gap = torch.where(contact, max_gap, torch.zeros_like(max_gap)).amax(dim=-1)
    command.metrics["sole_support_max_gap"] = contacted_gap * active
    command.metrics["sole_support_bad_fraction"] = (
        (contact & (max_gap > max_gap_threshold)).float().mean(dim=-1) * active
    )
    return penalty * active


def motion_gate_consecutive_stability_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Reward progress toward the consecutive stable samples required by a gate."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    stable_count = getattr(command, "gate_stable_count", None)
    gate_frames = tuple(getattr(command.cfg, "gate_frames", ()))
    if stable_count is None or not gate_frames:
        return torch.zeros_like(command.time_steps, dtype=torch.float)

    at_gate = torch.zeros_like(command.time_steps, dtype=torch.bool)
    for frame in gate_frames:
        at_gate |= command.time_steps == frame
    stable_steps = float(max(getattr(command.cfg, "gate_stable_steps", 1), 1))
    progress = torch.clamp(stable_count.float() / stable_steps, min=0.0, max=1.0)
    return torch.pow(progress, exponent) * at_gate.float()


def motion_gate_anchor_angular_excess_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_speed: float,
    speed_scale: float,
) -> torch.Tensor:
    """Penalize gate angular speed without saturating above the hard threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gate_wait_count = getattr(command, "gate_wait_count", None)
    if gate_wait_count is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)

    angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1)
    normalized_excess = torch.relu(angular_speed - target_speed) / max(speed_scale, 1.0e-6)
    # Smooth-L1 stays quadratic near the target and keeps a linear gradient for large violations.
    penalty = torch.where(
        normalized_excess < 1.0,
        0.5 * torch.square(normalized_excess),
        normalized_excess - 0.5,
    )
    return penalty * (gate_wait_count > 0).float()


def motion_gate_wait_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    grace_steps: int,
    max_wait_steps: int,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Increase the cost of waiting at a gate after the nominal settling window."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gate_wait_count = getattr(command, "gate_wait_count", None)
    if gate_wait_count is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)

    denominator = float(max(max_wait_steps - grace_steps, 1))
    wait_fraction = torch.clamp(
        (gate_wait_count.float() - float(grace_steps)) / denominator,
        min=0.0,
        max=1.0,
    )
    return torch.pow(wait_fraction, exponent)


def motion_gate_stable_reset_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalize losing consecutive stable samples instead of rewarding partial cycles."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reset_fraction = getattr(command, "gate_stable_reset_fraction", None)
    if reset_fraction is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)
    return reset_fraction


def motion_gate_pass_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return a one-step event when any double-foot gate is passed."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pass_event = getattr(command, "gate_pass_event", None)
    if pass_event is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)
    return pass_event.float()


def motion_gate_task_complete_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return a one-step event after all gates were passed in the same episode."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    completion_event = getattr(command, "gate_completion_event", None)
    if completion_event is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)
    return completion_event.float()


def motion_gate_timeout_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_wait_steps: int,
) -> torch.Tensor:
    """Apply a terminal cost on the step that exceeds the gate wait budget."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gate_wait_count = getattr(command, "gate_wait_count", None)
    if gate_wait_count is None:
        return torch.zeros_like(command.time_steps, dtype=torch.float)
    return (gate_wait_count >= max_wait_steps).float()


def motion_feet_under_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    margin: float = 0.01,
    swing_velocity_threshold: float = 0.05,
    body_names: list[str] | None = None,
    phase_start: float = 0.0,
    phase_end: float = 1.0,
) -> torch.Tensor:
    """Penalize only reference-swing feet that pass below demonstrated clearance."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    target_z = command.body_pos_relative_w[:, body_indexes, 2]
    actual_z = command.robot_body_pos_w[:, body_indexes, 2]
    target_speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, body_indexes], dim=-1)
    swing_mask = target_speed > swing_velocity_threshold
    clearance_shortage = torch.relu(target_z - actual_z - margin)
    normalized_shortage = torch.clamp(clearance_shortage / std, max=1.0)
    penalty = torch.mean(torch.square(normalized_shortage) * swing_mask.float(), dim=-1)
    return penalty * _motion_phase_mask(command, phase_start, phase_end)

def motion_feet_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel

def hand_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when contact force times velocity exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    velocity_norm = torch.norm(body_vel, dim=-1)
    force_velocity_product = contact_force_norm * velocity_norm
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)
    return reward

def body_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 300.0,
) -> torch.Tensor:
    """Penalize when historical maximum contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    max_contact_force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_force_violation = torch.clamp(max_contact_force_norm - force_threshold, min=0.0)
    max_force_penalty = torch.sum(max_force_violation / force_threshold, dim=1)
    return max_force_penalty

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when historical maximum contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] 
    velocity_norm = torch.norm(body_vel, dim=-1) 
    force_velocity_product = contact_force_norm * velocity_norm  
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  
    reward = torch.sum(violation / force_velocity_threshold, dim=1) 
    return reward

def body_contact_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize velocity when contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    max_contact_force = torch.max(net_contact_forces[:, :, sensor_cfg.body_ids, 2], dim=1)[0]
    is_contact = max_contact_force > force_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    reward = torch.sum(torch.square(body_vel_z) * is_contact, dim=1)
    return reward

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when contact force times velocity exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    velocity_norm = torch.norm(body_vel, dim=-1)  # (num_envs, num_feet)
    force_velocity_product = contact_force_norm * velocity_norm  # (num_envs, num_feet)
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  # (num_envs, num_feet)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)  # (num_envs,)
    return reward


def motion_default_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma = float,
    delta = float,
    start_frames = int,
    end_frames = int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""Reward for maintaining default pose at trajectory start and end using exponential kernel.

    .. math:: K(x, \sigma, \delta) = \exp\left(-\left(\frac{\max(0, \|x\| - \delta)}{\sigma}\right)^2\right)
    """
    joint_pos = env.command_manager.get_command(command_name)
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    asset: Articulation = env.scene[asset_cfg.name]
    time_steps = command.time_steps
    total_steps = command.motion.time_step_total
    in_start = time_steps < start_frames
    in_end = time_steps > (total_steps - 1 - end_frames)
    in_boundary = in_start | in_end
    default_joint_pos = asset.data.default_joint_pos_nominal
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error_vec = current_joint_pos - default_joint_pos
    error_norm = torch.linalg.vector_norm(error_vec, dim=-1)
    clipped_error = torch.clamp(error_norm - delta, min=0.0)
    scaled_error_squared = torch.square(clipped_error / sigma)
    reward = torch.exp(-scaled_error_squared)
    reward = reward * in_boundary.float()
    return reward

def com_balance_when_stand(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward COM balance based on support condition."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_com_pos_w = asset.data.body_com_pos_w
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True)
    system_com = (body_com_pos_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]

    feet_max_forces = contact_forces.norm(dim=-1).max(dim=1)[0]

    left_foot_contact = (feet_max_forces[:, 0] > total_mass.squeeze(-1) * 9.8 * 0.6)
    right_foot_contact = (feet_max_forces[:, 1] > total_mass.squeeze(-1) * 9.8 * 0.6)

    total_contact_force = feet_max_forces[:, 0] + feet_max_forces[:, 1]
    both_feet_support = (total_contact_force > total_mass.squeeze(-1) * 9.8 * 0.8)

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    waist_quat = asset.data.body_quat_w[:, 0, :]
    left_foot_pos = feet_pos[:, 0, :]
    right_foot_pos = feet_pos[:, 1, :]

    foot_offset_local = torch.tensor([0.02, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    foot_offset_world = quat_apply_yaw(waist_quat, foot_offset_local)
    left_foot_center = left_foot_pos + foot_offset_world
    right_foot_center = right_foot_pos + foot_offset_world

    only_left_contact = left_foot_contact & (~right_foot_contact) & (~both_feet_support)
    only_right_contact = (~left_foot_contact) & right_foot_contact & (~both_feet_support)
    no_contact = (~left_foot_contact) & (~right_foot_contact) & (~both_feet_support)
    has_contact = ~no_contact

    both_feet_target = (left_foot_center + right_foot_center) / 2.0

    target_com = torch.where(
        both_feet_support.unsqueeze(-1),
        both_feet_target,
        torch.where(
            only_left_contact.unsqueeze(-1),
            left_foot_center,
            torch.where(
                only_right_contact.unsqueeze(-1),
                right_foot_center,
                both_feet_target
            )
        )
    )

    com_offset = torch.norm(system_com[:, :2] - target_com[:, :2], dim=1)

    reward = torch.exp(-torch.square(1.5 * com_offset / std))

    reward = torch.where(has_contact, reward, torch.zeros_like(reward))

    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # (N, history, 2, 3)
    
    # left_foot_contact = contact_forces[0] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # right_foot_contact = contact_forces[1] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # both_feet_support = left_foot_contact & right_foot_contact
    
    # left_foot_air_time = torch.where(left_foot_contact & (~right_foot_contact), left_foot_air_time + env.step_dt, 0.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    total_contact_force = torch.sum(contact_forces, dim=1)
    body_weight = total_mass * 9.8
    stand_mask = (total_contact_force > body_weight * 0.8) & (total_contact_force < body_weight * 1.2)
    # left_foot_contact = contact_forces[0] > contact_forces[1] * 2.0
    # right_foot_contact = contact_forces[1] > contact_forces[0] * 2.0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command: MotionCommand = env.command_manager.get_term(command_name)
    # mask = torch.where(command.start_time < 50, 0, 1)
    # mask = torch.where(command.out_time > 0, 1, mask)
    reward *= stand_mask # * mask
    return reward

def body_slide_vel(
        env, sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_threshold: float = 110.0
    ) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > contact_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel


def motion_phase_body_slide_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 110.0,
    phase_start: float = 0.0,
    phase_end: float = 1.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    penalty = body_slide_vel(env, sensor_cfg, asset_cfg, contact_threshold)
    return penalty * _motion_phase_mask(command, phase_start, phase_end)


def motion_phase_swing_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    swing_velocity_threshold: float,
    contact_force_threshold: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize dragging when the reference foot should be in swing."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    target_speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, body_indexes], dim=-1)
    swing_mask = target_speed > swing_velocity_threshold
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    normalized_force = torch.clamp(contact_force / contact_force_threshold, max=1.0)
    penalty = torch.mean(normalized_force * swing_mask.float(), dim=-1)
    return penalty * _motion_phase_mask(command, phase_start, phase_end)

def penalize_feet_dragging(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    drag_threshold: float = 5.0,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize dragging of swing foot during single foot support."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]

    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    body_weight = total_mass * 9.8

    foot_in_contact = contact_forces_z > contact_threshold

    num_feet_in_contact = torch.sum(foot_in_contact.int(), dim=1)
    single_foot_support = num_feet_in_contact == 1

    left_foot_force = contact_forces_z[:, 0]
    right_foot_force = contact_forces_z[:, 1]

    left_is_support = contact_forces_z[:, 0] > contact_forces_z[:, 1] * 2

    swing_foot_force = torch.where(
        left_is_support,
        right_foot_force,
        left_foot_force
    )

    drag_penalty = torch.clamp(swing_foot_force - drag_threshold, min=0.0)

    no_foot_contact = num_feet_in_contact == 0

    both_feet_drag_penalty = (
        torch.clamp(left_foot_force - drag_threshold, min=0.0) +
        torch.clamp(right_foot_force - drag_threshold, min=0.0)
    )

    reward = torch.where(
        single_foot_support,
        drag_penalty,
        torch.where(
            no_foot_contact,
            both_feet_drag_penalty,
            torch.zeros_like(drag_penalty)
        )
    )
    reward = torch.clamp(reward, max=threshold)

    return reward

def penalize_feet_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    swing_force_ratio: float = 0.15,
    support_force_ratio: float = 0.5,
    single_support_target: float = 0.11,
    single_support_scale: float = 5.0,
) -> torch.Tensor:
    """Penalize insufficient swing foot height during single foot support."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True).squeeze(-1)
    body_weight = total_mass * 9.8

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]

    swing_threshold = body_weight * swing_force_ratio
    support_threshold = body_weight * support_force_ratio

    left_force = contact_forces_z[:, 0]
    right_force = contact_forces_z[:, 1]

    left_is_swing = (left_force < swing_threshold) & (right_force > support_threshold)
    right_is_swing = (right_force < swing_threshold) & (left_force > support_threshold)
    single_foot_support = left_is_swing | right_is_swing

    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    swing_foot_height = torch.where(
        left_is_swing,
        feet_pos_z[:, 0],
        torch.where(
            right_is_swing,
            feet_pos_z[:, 1],
            torch.zeros_like(feet_pos_z[:, 0]),
        )
    )

    height_penalty = torch.clamp((single_support_target - swing_foot_height) * single_support_scale, min=0.0)

    height_reward = torch.where(
        single_foot_support,
        height_penalty,
        torch.zeros_like(height_penalty),
    )

    return height_reward

def twisted_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shank_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    horizontal_threshold: float = 0.3,
    shank_upright_threshold: float = 0.7,
) -> torch.Tensor:
    """Penalize twisted foot landing (contact force not perpendicular to foot surface)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces_w = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]

    shank_quat_w = asset.data.body_quat_w[:, shank_cfg.body_ids, :]

    num_envs = contact_forces_w.shape[0]
    num_shanks = shank_quat_w.shape[1]
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_envs * num_shanks, 3)

    shank_quat_flat = shank_quat_w.reshape(-1, 4)
    local_z = quat_rotate_inverse(shank_quat_flat, world_z)
    local_z = local_z.reshape(num_envs, num_shanks, 3)

    shank_uprightness = local_z[:, :, 2]

    is_shank_upright = shank_uprightness > shank_upright_threshold

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_feet = contact_forces_w.shape[1]

    contact_forces_w_flat = contact_forces_w.reshape(-1, 3)
    foot_quat_w_flat = foot_quat_w.reshape(-1, 4)

    contact_forces_local_flat = quat_rotate_inverse(foot_quat_w_flat, contact_forces_w_flat)

    contact_forces_local = contact_forces_local_flat.reshape(num_envs, num_feet, 3)

    force_local_x = contact_forces_local[:, :, 0]
    force_local_y = contact_forces_local[:, :, 1]
    force_local_z = contact_forces_local[:, :, 2]

    force_horizontal = torch.sqrt(force_local_x**2 + force_local_y**2)

    force_total = torch.norm(contact_forces_local, dim=-1)

    horizontal_ratio = force_horizontal / (force_total + 1e-6)

    contact_mag = torch.norm(contact_forces_w, dim=-1)
    is_in_contact = contact_mag > contact_threshold

    penalty_per_foot = torch.clamp(horizontal_ratio - horizontal_threshold, min=0.0)

    penalty_per_foot = torch.where(
        is_in_contact & is_shank_upright,
        penalty_per_foot,
        torch.zeros_like(penalty_per_foot)
    )

    total_penalty = torch.sum(penalty_per_foot, dim=1)

    return total_penalty
