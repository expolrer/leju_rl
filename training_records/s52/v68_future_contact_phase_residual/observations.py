from __future__ import annotations

import torch

from isaaclab.utils.math import quat_rotate_inverse

from leju_robot.tasks.tracking.mdp import rewards as base_rewards
from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def future_foot_contacts_b(
    env,
    command_name: str,
    body_names: list[str],
    foot_frame_ranges: list[list[tuple[int, int]]],
    time_horizon_frames: int,
) -> torch.Tensor:
    """Return each foot's next target XY and normalized time-to-contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = base_rewards._get_body_indexes(command, body_names)
    if len(body_indexes) != 2:
        raise ValueError("future contact observation requires exactly two feet")

    targets = torch.zeros(
        (env.num_envs, 2, 3),
        dtype=command.robot_anchor_pos_w.dtype,
        device=command.robot_anchor_pos_w.device,
    )
    contact_frames = torch.zeros(
        (env.num_envs, 2), dtype=torch.long, device=command.time_steps.device
    )
    for foot_index, frame_ranges in enumerate(foot_frame_ranges):
        selected = torch.zeros(env.num_envs, dtype=torch.bool, device=targets.device)
        motion_index = body_indexes[foot_index]
        for _, end in frame_ranges:
            pending = (~selected) & (command.time_steps <= int(end))
            world_target = (
                command.motion.body_pos_w[int(end), motion_index].unsqueeze(0)
                + env.scene.env_origins
            )
            targets[:, foot_index] = torch.where(
                pending.unsqueeze(-1), world_target, targets[:, foot_index]
            )
            contact_frames[:, foot_index] = torch.where(
                pending,
                torch.full_like(contact_frames[:, foot_index], int(end)),
                contact_frames[:, foot_index],
            )
            selected |= pending

        final_end = int(frame_ranges[-1][1])
        world_target = (
            command.motion.body_pos_w[final_end, motion_index].unsqueeze(0)
            + env.scene.env_origins
        )
        targets[:, foot_index] = torch.where(
            (~selected).unsqueeze(-1), world_target, targets[:, foot_index]
        )
        contact_frames[:, foot_index] = torch.where(
            ~selected,
            torch.full_like(contact_frames[:, foot_index], final_end),
            contact_frames[:, foot_index],
        )

    delta_w = targets - command.robot_anchor_pos_w[:, None, :]
    anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
    delta_b = quat_rotate_inverse(
        anchor_quat.reshape(-1, 4), delta_w.reshape(-1, 3)
    ).reshape(env.num_envs, 2, 3)
    time_to_contact = torch.clamp(
        (contact_frames - command.time_steps[:, None]).float()
        / float(time_horizon_frames),
        min=0.0,
        max=1.0,
    )
    return torch.cat((delta_b[..., :2], time_to_contact.unsqueeze(-1)), dim=-1).reshape(
        env.num_envs, -1
    )


def future_foot_contacts_with_phase_b(
    env,
    command_name: str,
    body_names: list[str],
    foot_frame_ranges: list[list[tuple[int, int]]],
    time_horizon_frames: int,
    total_motion_frames: int,
) -> torch.Tensor:
    contacts = future_foot_contacts_b(
        env,
        command_name=command_name,
        body_names=body_names,
        foot_frame_ranges=foot_frame_ranges,
        time_horizon_frames=time_horizon_frames,
    )
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = torch.clamp(
        command.time_steps.float() / float(total_motion_frames), min=0.0, max=1.0
    ).unsqueeze(-1)
    return torch.cat((contacts, phase), dim=-1)


__all__ = ["future_foot_contacts_b", "future_foot_contacts_with_phase_b"]

