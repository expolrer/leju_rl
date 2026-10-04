from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def terminal_contact_time_gate_reward(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str],
    frame_start: int,
    target_frame: int,
    frame_end: int,
    forward_std: float,
    lateral_std: float,
    height_std: float,
    velocity_std: float,
    stable_speed_std: float,
    max_desired_speed: float,
    contact_force_threshold: float,
    touchdown_window_s: float,
) -> torch.Tensor:
    """Coordinate terminal foot arrival with per-foot contact time."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = torch.tensor(
        [command.cfg.body_names.index(name) for name in body_names],
        dtype=torch.long,
        device=command.device,
    )
    target = command.motion.body_pos_w[int(target_frame), body_indexes]
    target = target.unsqueeze(0) + env.scene.env_origins[:, None, :]
    position_error = target - command.robot_body_pos_w[:, body_indexes]
    normalized_position_error = (
        torch.square(position_error[..., 0] / float(forward_std))
        + torch.square(position_error[..., 1] / float(lateral_std))
        + torch.square(position_error[..., 2] / float(height_std))
    )
    position_score = torch.exp(-normalized_position_error)

    frame = command.time_steps.float()
    active = (frame >= float(frame_start)) & (frame <= float(frame_end))
    remaining_s = torch.clamp(
        (float(target_frame) - frame) * float(env.step_dt), min=float(env.step_dt)
    )
    desired_velocity = position_error / remaining_s[:, None, None]
    desired_speed = torch.linalg.vector_norm(desired_velocity, dim=-1, keepdim=True)
    desired_velocity = desired_velocity * torch.clamp(
        float(max_desired_speed) / torch.clamp(desired_speed, min=1.0e-6), max=1.0
    )
    velocity_error = command.robot_body_lin_vel_w[:, body_indexes] - desired_velocity
    arrival_velocity_score = torch.exp(
        -torch.square(
            torch.linalg.vector_norm(velocity_error, dim=-1) / float(velocity_std)
        )
    )
    foot_speed = torch.linalg.vector_norm(
        command.robot_body_lin_vel_w[:, body_indexes], dim=-1
    )
    stable_speed_score = torch.exp(
        -torch.square(foot_speed / float(stable_speed_std))
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[
        :, :, sensor_cfg.body_ids, :
    ].norm(dim=-1).max(dim=1)[0]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact = contact_force >= float(contact_force_threshold)
    touchdown = contact & (contact_time <= float(touchdown_window_s))
    stable_contact = contact & (contact_time > float(touchdown_window_s))
    pre_contact = ~contact

    # Position cannot compensate for unsafe speed at touchdown: both gates multiply.
    pre_contact_score = arrival_velocity_score * (0.25 + 0.75 * position_score)
    touchdown_score = position_score * stable_speed_score
    stable_score = stable_speed_score * (0.40 + 0.60 * position_score)
    per_foot_reward = (
        0.45 * pre_contact_score * pre_contact.float()
        + touchdown_score * touchdown.float()
        + 0.60 * stable_score * stable_contact.float()
    )
    reward = per_foot_reward.mean(dim=-1) * active.float()

    command.metrics["s52_terminal_contact_time_gate_active"] = active.float()
    command.metrics["s52_terminal_contact_time_gate_position_error"] = (
        torch.linalg.vector_norm(position_error, dim=-1).mean(dim=-1)
    )
    command.metrics["s52_terminal_contact_time_gate_speed"] = foot_speed.mean(dim=-1)
    command.metrics["s52_terminal_contact_time_gate_pre_contact_fraction"] = (
        pre_contact.float().mean(dim=-1) * active.float()
    )
    command.metrics["s52_terminal_contact_time_gate_touchdown_fraction"] = (
        touchdown.float().mean(dim=-1) * active.float()
    )
    command.metrics["s52_terminal_contact_time_gate_stable_fraction"] = (
        stable_contact.float().mean(dim=-1) * active.float()
    )
    return reward


__all__ = ["terminal_contact_time_gate_reward"]
