from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MotionCommand
from .rewards import (
    _body_sample_points_w,
    _get_body_indexes,
    _motion_frame_ranges_mask,
)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ret = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    # if ret[0] == True:
    #     a = 1 
    return ret

def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    ret = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    # if ret[0] == True:
    #         a = 1 
    return ret


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    ret = torch.any(error > threshold, dim=-1)
    # if ret[0] == True:
    #     a = 1 
    return ret


def motion_gate_wait_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_wait_steps: int,
) -> torch.Tensor:
    """Fail an episode that stalls at a hard gate beyond its settling budget."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gate_wait_count = getattr(command, "gate_wait_count", None)
    if gate_wait_count is None:
        return torch.zeros(command.time_steps.shape, dtype=torch.bool, device=command.time_steps.device)
    return gate_wait_count >= max_wait_steps


def motion_riser_clearance_cat(
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
    max_termination_probability: float,
    risk_exponent: float,
    reference_swing_speed_threshold: float,
    reference_swing_contrast_threshold: float,
    local_points: list[tuple[float, float, float]],
    body_names: list[str],
) -> torch.Tensor:
    """Apply a local Constraints-as-Terminations barrier at descent risers."""
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")
    if not 0.0 <= max_termination_probability <= 1.0:
        raise ValueError("max_termination_probability must be in [0, 1]")
    if hard_distance < 0.0 or safety_distance <= hard_distance:
        raise ValueError("safety_distance must be greater than hard_distance >= 0")

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

    normalized_risk = torch.clamp(
        (float(safety_distance) - min_distance)
        / (float(safety_distance) - float(hard_distance)),
        min=0.0,
        max=1.0,
    )
    termination_probability = (
        float(max_termination_probability)
        * torch.pow(normalized_risk, float(risk_exponent))
        * active.float()
    )
    hard_violation = active & (min_distance <= float(hard_distance))
    sampled_termination = torch.rand_like(termination_probability) < termination_probability
    terminated_feet = hard_violation | (active & sampled_termination)
    terminated = terminated_feet.any(dim=-1)

    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    command.metrics["cat_riser_active_fraction"] = active.float().mean(dim=-1)
    command.metrics["cat_riser_violation_fraction"] = (
        (active & (min_distance < float(safety_distance))).float().sum(dim=-1)
        / active_count
    )
    command.metrics["cat_riser_hard_fraction"] = (
        hard_violation.float().sum(dim=-1) / active_count
    )
    command.metrics["cat_riser_min_distance"] = min_distance.amin(dim=-1)
    command.metrics["cat_riser_termination_probability"] = (
        termination_probability.amax(dim=-1)
    )
    command.metrics["cat_riser_terminated"] = terminated.float()
    if num_feet >= 2:
        command.metrics["left_cat_riser_min_distance"] = min_distance[:, 0]
        command.metrics["right_cat_riser_min_distance"] = min_distance[:, 1]
    return terminated
