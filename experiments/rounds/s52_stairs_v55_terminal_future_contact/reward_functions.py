from __future__ import annotations

import torch

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def terminal_future_contact_residual_reward(
    env,
    command_name: str,
    body_names: list[str],
    frame_start: int,
    target_frame: int,
    forward_std: float,
    lateral_std: float,
    height_std: float,
    speed_std: float,
) -> torch.Tensor:
    """Track both final foot contacts with urgency set by remaining contact time."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    indexes = torch.tensor(
        [command.cfg.body_names.index(name) for name in body_names],
        dtype=torch.long,
        device=command.device,
    )
    target = command.motion.body_pos_w[int(target_frame), indexes]
    target = target.unsqueeze(0) + env.scene.env_origins[:, None, :]
    error = command.robot_body_pos_w[:, indexes] - target
    scaled_error = (
        torch.square(error[..., 0] / float(forward_std))
        + torch.square(error[..., 1] / float(lateral_std))
        + torch.square(error[..., 2] / float(height_std))
    )
    position_score = torch.exp(-scaled_error).mean(dim=-1)

    foot_speed = torch.linalg.vector_norm(
        command.robot_body_lin_vel_w[:, indexes], dim=-1
    ).mean(dim=-1)
    speed_score = torch.exp(-torch.square(foot_speed / float(speed_std)))

    frame = command.time_steps.float()
    active = ((frame >= float(frame_start)) & (frame <= float(target_frame))).float()
    progress = torch.clamp(
        (frame - float(frame_start)) / max(float(target_frame - frame_start), 1.0),
        min=0.0,
        max=1.0,
    )
    urgency = torch.square(progress)
    reward = (0.75 * position_score + 0.25 * speed_score) * urgency * active

    command.metrics["s52_terminal_future_contact_error"] = torch.linalg.vector_norm(
        error, dim=-1
    ).mean(dim=-1)
    command.metrics["s52_terminal_future_contact_speed"] = foot_speed
    command.metrics["s52_terminal_future_contact_progress"] = progress * active
    command.metrics["s52_terminal_future_contact_active"] = active
    return reward


__all__ = ["terminal_future_contact_residual_reward"]
