from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

VELOCITY_SMALL_RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}


def update_push_with_entropy_and_eposidelength(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    entropy_threshold: float = 0.9,
    target_episode_length: float = 490.0,
) -> float:
    """Enable push disturbances only after the agent reaches sufficient sampling entropy and episode length."""
    motion_term = env.command_manager.get_term("motion")

    mean_entropy = motion_term.metrics["sampling_entropy"].mean().item()

    if len(env_ids) > 0:
        died_episode_length = env.episode_length_buf[env_ids].float().mean().item()
    else:
        died_episode_length = 0.0

    if not hasattr(env, "smooth_episode_length"):
        env.smooth_episode_length = 0.0

    if len(env_ids) > 0:
        alpha = 0.05
        env.smooth_episode_length = (1 - alpha) * env.smooth_episode_length + alpha * died_episode_length

    can_push = (mean_entropy > entropy_threshold) and (env.smooth_episode_length > target_episode_length)

    if "push_robot" in env.event_manager.active_terms:
        push_event = env.event_manager.get_term_cfg("push_robot")
        if can_push:
            push_event.params["velocity_range"] = VELOCITY_SMALL_RANGE
        else:
            push_event.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
        env.event_manager.set_term_cfg("push_robot", push_event)

    if "base_external_force_torque" in env.event_manager.active_terms:
        force_event = env.event_manager.get_term_cfg("base_external_force_torque")
        if can_push:
            force_event.params["force_range"] = {
                "x": (-600.0, 600.0), "y": (-600.0, 600.0), "z": (-350.0, 350.0),
            }
        else:
            force_event.params["force_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        env.event_manager.set_term_cfg("base_external_force_torque", force_event)

    return 1.0 if can_push else 0.0
