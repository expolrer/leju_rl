from __future__ import annotations

import torch


def rigid_foot_riser_clearance_cost(
    sample_points_w: torch.Tensor,
    foot_lin_vel_w: torch.Tensor,
    contact_force_n: torch.Tensor,
    env_origins: torch.Tensor,
    phase_active: torch.Tensor,
    planned_swing: torch.Tensor,
    riser_x_offsets: tuple[float, ...] | list[float],
    upper_heights: tuple[float, ...] | list[float],
    lower_heights: tuple[float, ...] | list[float],
    window_before: float,
    window_after: float,
    safety_distance: float,
    hard_distance: float,
    hard_cost_scale: float,
    actual_swing_speed_threshold: float,
    support_force_threshold: float,
    require_low_contact_for_swing: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute one shared rigid-foot clearance cost for training and rollout audits."""
    if sample_points_w.ndim != 4 or sample_points_w.shape[-1] != 3:
        raise ValueError("sample_points_w must have shape [N, F, P, 3]")
    if not (
        len(riser_x_offsets) == len(upper_heights) == len(lower_heights)
    ):
        raise ValueError("riser geometry tables must have identical lengths")
    if hard_distance < 0.0 or safety_distance <= hard_distance:
        raise ValueError("safety_distance must be greater than hard_distance >= 0")

    num_envs, num_feet = sample_points_w.shape[:2]
    device = sample_points_w.device
    actual_speed = torch.linalg.vector_norm(foot_lin_vel_w, dim=-1)
    low_contact = contact_force_n < float(support_force_threshold)
    if require_low_contact_for_swing:
        actual_swing = low_contact & (
            actual_speed >= float(actual_swing_speed_threshold)
        )
        swing_active = low_contact & (actual_swing | planned_swing.bool())
    else:
        actual_swing = (actual_speed >= float(actual_swing_speed_threshold)) | low_contact
        swing_active = actual_swing | planned_swing.bool()
    phase_active = phase_active.bool().reshape(num_envs, 1)

    active = torch.zeros((num_envs, num_feet), dtype=torch.bool, device=device)
    min_distance = torch.full(
        (num_envs, num_feet), torch.inf, dtype=sample_points_w.dtype, device=device
    )
    point_x = sample_points_w[..., 0]
    point_z = sample_points_w[..., 2]

    for edge_offset, upper_height, lower_height in zip(
        riser_x_offsets, upper_heights, lower_heights
    ):
        edge_x = env_origins[:, 0] + float(edge_offset)
        upper_z = env_origins[:, 2] + float(upper_height)
        lower_z = env_origins[:, 2] + float(lower_height)
        relative_x = point_x - edge_x[:, None, None]
        candidate = (
            phase_active[:, :, None]
            & swing_active[:, :, None]
            & (relative_x >= -float(window_before))
            & (relative_x <= float(window_after))
        )
        closest_z = torch.minimum(
            torch.maximum(point_z, lower_z[:, None, None]),
            upper_z[:, None, None],
        )
        distance = torch.sqrt(
            torch.square(relative_x) + torch.square(point_z - closest_z) + 1.0e-12
        )
        edge_active = candidate.any(dim=-1)
        edge_distance = torch.where(
            candidate, distance, torch.full_like(distance, torch.inf)
        ).amin(dim=-1)
        min_distance = torch.minimum(min_distance, edge_distance)
        active |= edge_active

    safe_report = torch.where(
        active, min_distance, torch.full_like(min_distance, float(safety_distance))
    )
    safety_span = float(safety_distance) - float(hard_distance)
    shortage = torch.clamp(
        torch.relu(float(safety_distance) - safe_report) / safety_span, max=2.0
    )
    hard_shortage = torch.clamp(
        torch.relu(float(hard_distance) - safe_report)
        / max(float(hard_distance), 1.0e-6),
        max=2.0,
    )
    per_foot_cost = active.float() * (
        torch.square(shortage) + float(hard_cost_scale) * torch.square(hard_shortage)
    )
    active_count = active.float().sum(dim=-1).clamp(min=1.0)
    return {
        "cost": per_foot_cost.sum(dim=-1) / active_count,
        "active": active,
        "actual_swing": actual_swing,
        "planned_swing": planned_swing.bool(),
        "min_distance": min_distance,
        "reported_min_distance": safe_report,
        "violation": active & (min_distance < float(safety_distance)),
        "hard_violation": active & (min_distance < float(hard_distance)),
    }
