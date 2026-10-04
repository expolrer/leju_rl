from __future__ import annotations

import torch

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def finite_memory_normalized_composite_probe(
    env,
    command_name: str,
    window_start_frame: int,
    window_end_frame: int,
    clearance_normalizer: float,
    impact_normalizer: float,
    memory_decay: float,
) -> torch.Tensor:
    """Apply short, bounded memory to separately normalized route-safety risks."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    clearance = command.metrics.get("shared_clearance_cost")
    impact = command.metrics.get("s52_instant_impact_constraint_cost")
    if clearance is None or impact is None:
        raise RuntimeError("clearance and impact probes must run before finite-memory probe")

    clearance_risk = torch.clamp(
        clearance / float(clearance_normalizer), min=0.0, max=1.0
    )
    impact_risk = torch.clamp(
        impact / float(impact_normalizer), min=0.0, max=1.0
    )
    current = torch.maximum(clearance_risk, impact_risk)

    frame = command.time_steps
    previous = getattr(command, "_s52_finite_memory_risk", None)
    previous_frame = getattr(command, "_s52_finite_memory_frame", None)
    if previous is None or previous.shape != current.shape:
        previous = torch.zeros_like(current)
    if previous_frame is None or previous_frame.shape != frame.shape:
        previous_frame = torch.zeros_like(frame)

    in_window = (frame >= int(window_start_frame)) & (frame <= int(window_end_frame))
    reset = (~in_window) | (frame < previous_frame)
    decayed = previous * float(memory_decay)
    memory = torch.where(reset, torch.zeros_like(current), torch.maximum(current, decayed))
    active = in_window.to(current.dtype)
    bounded_cost = memory * active

    command._s52_finite_memory_risk = memory.detach().clone()
    command._s52_finite_memory_frame = frame.detach().clone()
    command.metrics["s52_finite_memory_composite_constraint_cost"] = bounded_cost
    command.metrics["s52_finite_memory_composite_constraint_active"] = active
    command.metrics["s52_finite_memory_clearance_risk"] = clearance_risk
    command.metrics["s52_finite_memory_impact_risk"] = impact_risk
    command.metrics["s52_finite_memory_current_risk"] = current
    command.metrics["s52_finite_memory_decayed_risk"] = memory
    return bounded_cost


__all__ = ["finite_memory_normalized_composite_probe"]
