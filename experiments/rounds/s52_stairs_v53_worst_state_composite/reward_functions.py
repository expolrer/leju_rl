from __future__ import annotations

import torch

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def worst_state_composite_constraint_probe(
    env,
    command_name: str,
    window_start_frame: int,
    window_end_frame: int,
) -> torch.Tensor:
    """Hold the worst route-safety cost across the stair-descent segment."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    current = command.metrics.get("s52_route_safety_composite_constraint_cost")
    if current is None:
        raise RuntimeError("route safety composite must be evaluated before worst-state probe")

    frame = command.time_steps
    previous = getattr(command, "_s52_worst_state_composite", None)
    previous_frame = getattr(command, "_s52_worst_state_frame", None)
    if previous is None or previous.shape != current.shape:
        previous = torch.zeros_like(current)
    if previous_frame is None or previous_frame.shape != frame.shape:
        previous_frame = torch.zeros_like(frame)

    reset = (frame < int(window_start_frame)) | (frame < previous_frame)
    running_worst = torch.where(reset, current, torch.maximum(previous, current))
    active = (
        (frame >= int(window_start_frame)) & (frame <= int(window_end_frame))
    ).to(current.dtype)
    held_cost = running_worst * active

    command._s52_worst_state_composite = running_worst.detach().clone()
    command._s52_worst_state_frame = frame.detach().clone()
    command.metrics["s52_worst_state_composite_constraint_cost"] = held_cost
    command.metrics["s52_worst_state_composite_constraint_active"] = active
    command.metrics["s52_worst_state_composite_current_cost"] = current
    command.metrics["s52_worst_state_composite_running_max"] = running_worst
    return held_cost


__all__ = ["worst_state_composite_constraint_probe"]
