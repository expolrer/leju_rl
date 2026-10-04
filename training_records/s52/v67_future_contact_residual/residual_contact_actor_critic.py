from __future__ import annotations

import torch
from torch import nn

from rsl_rl.modules import ActorCritic


class FrozenBaseResidualActor(nn.Module):
    """Keep the verified actor intact and learn only a bounded goal adapter."""

    def __init__(
        self,
        base: nn.Module,
        num_actor_obs: int,
        base_obs_dim: int,
        num_actions: int,
        adapter_hidden_dims: list[int],
        adapter_scale: float,
    ) -> None:
        super().__init__()
        if num_actor_obs <= base_obs_dim:
            raise ValueError("residual actor requires appended contact-goal observations")
        if not adapter_hidden_dims:
            raise ValueError("adapter_hidden_dims must not be empty")
        self.base = base
        self.num_actor_obs = int(num_actor_obs)
        self.base_obs_dim = int(base_obs_dim)
        self.adapter_scale = float(adapter_scale)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        layers: list[nn.Module] = []
        input_dim = num_actor_obs
        for hidden_dim in adapter_hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions))
        self.adapter = nn.Sequential(*layers)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base_observations = observations[..., : self.base_obs_dim].contiguous()
        base_actions = self.base(base_observations)
        residual = self.adapter_scale * torch.tanh(self.adapter(observations))
        return base_actions + residual


class ResidualContactActorCritic(ActorCritic):
    """ActorCritic with a frozen 148-D actor plus a small contact-goal adapter."""

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
        init_noise_std: float,
        noise_std_type: str = "scalar",
        base_obs_dim: int = 148,
        adapter_hidden_dims: list[int] = [64, 32],
        adapter_scale: float = 0.02,
        **kwargs,
    ) -> None:
        full_actor_obs = int(num_actor_obs)
        super().__init__(
            num_actor_obs=int(base_obs_dim),
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.actor = FrozenBaseResidualActor(
            base=self.actor,
            num_actor_obs=full_actor_obs,
            base_obs_dim=base_obs_dim,
            num_actions=num_actions,
            adapter_hidden_dims=adapter_hidden_dims,
            adapter_scale=adapter_scale,
        )


__all__ = ["ResidualContactActorCritic"]

