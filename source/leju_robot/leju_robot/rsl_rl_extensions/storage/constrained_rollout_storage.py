from __future__ import annotations

import torch

from rsl_rl.storage import RolloutStorage


class ConstrainedRolloutStorage(RolloutStorage):
    """Rollout storage with a second return stream for physical constraint costs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.num_transitions_per_env, self.num_envs, 1)
        self.costs = torch.zeros(*shape, device=self.device)
        self.raw_costs = torch.zeros(*shape, device=self.device)
        self.cost_active = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self.cost_values = torch.zeros(*shape, device=self.device)
        self.cost_returns = torch.zeros(*shape, device=self.device)
        self.cost_advantages = torch.zeros(*shape, device=self.device)

    def add_transitions(self, transition):
        step = self.step
        if getattr(transition, "costs", None) is None:
            raise ValueError("constraint costs are missing from rollout transition")
        if getattr(transition, "cost_values", None) is None:
            raise ValueError("constraint values are missing from rollout transition")
        if getattr(transition, "cost_active", None) is None:
            raise ValueError("constraint activity is missing from rollout transition")
        super().add_transitions(transition)
        self.costs[step].copy_(transition.costs.view(-1, 1))
        self.raw_costs[step].copy_(transition.costs.view(-1, 1))
        self.cost_active[step].copy_(transition.cost_active.view(-1, 1).bool())
        self.cost_values[step].copy_(transition.cost_values.view(-1, 1))

    def aggregate_active_segment_max(self, replay_scale=1.0):
        """Blend each swing segment's worst cost back into its active samples."""
        if not 0.0 <= replay_scale <= 1.0:
            raise ValueError("segment replay scale must be in [0, 1]")
        self.costs.copy_(self.raw_costs)
        for env_index in range(self.num_envs):
            start = None
            for step in range(self.num_transitions_per_env + 1):
                active = (
                    step < self.num_transitions_per_env
                    and bool(self.cost_active[step, env_index, 0].item())
                    and not bool(self.dones[step, env_index, 0].item())
                )
                if active and start is None:
                    start = step
                if start is not None and (not active or step == self.num_transitions_per_env):
                    end = step
                    segment_max = self.raw_costs[start:end, env_index].amax()
                    segment_costs = self.costs[start:end, env_index]
                    segment_costs.add_(
                        float(replay_scale) * (segment_max - segment_costs)
                    )
                    start = None

    def compute_cost_returns(self, last_cost_values, gamma, lam):
        advantage = torch.zeros_like(last_cost_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_cost_values
                if step == self.num_transitions_per_env - 1
                else self.cost_values[step + 1]
            )
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.costs[step]
                + next_is_not_terminal * gamma * next_values
                - self.cost_values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]
        self.cost_advantages = self.cost_returns - self.cost_values
