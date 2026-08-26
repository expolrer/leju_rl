from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.algorithms import PPO
from leju_robot.rsl_rl_extensions.storage.constrained_rollout_storage import (
    ConstrainedRolloutStorage,
)


class TeacherRegularizedPPO(PPO):
    """PPO that keeps the actor close to a fixed, verified teacher policy."""

    def __init__(
        self,
        policy,
        teacher_checkpoint: str,
        teacher_action_coef: float = 2.0,
        teacher_kl_coef: float = 0.1,
        student_init_std: float = 0.12,
        student_std_coef: float = 0.1,
        freeze_observation_normalizer: bool = True,
        teacher_action_weights: tuple[float, ...] = (),
        teacher_hard_action_rmse_limit: float = 0.0,
        teacher_projection_iterations: int = 4,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("TeacherRegularizedPPO currently requires RND and symmetry to be disabled")
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(f"teacher checkpoint not found: {teacher_checkpoint}")

        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_action_coef = teacher_action_coef
        self.teacher_kl_coef = teacher_kl_coef
        self.student_init_std = student_init_std
        self.student_std_coef = student_std_coef
        self.freeze_observation_normalizer = freeze_observation_normalizer
        self.teacher_hard_action_rmse_limit = teacher_hard_action_rmse_limit
        self.teacher_projection_iterations = teacher_projection_iterations
        if teacher_hard_action_rmse_limit < 0.0:
            raise ValueError("teacher_hard_action_rmse_limit must be non-negative")
        if teacher_projection_iterations < 1:
            raise ValueError("teacher_projection_iterations must be positive")
        num_actions = int(policy.std.numel())
        if teacher_action_weights and len(teacher_action_weights) != num_actions:
            raise ValueError(
                f"teacher_action_weights has {len(teacher_action_weights)} entries, "
                f"expected {num_actions}"
            )
        if teacher_action_weights:
            action_weights = torch.as_tensor(
                teacher_action_weights, dtype=torch.float32, device=self.device
            )
        else:
            action_weights = torch.ones(num_actions, dtype=torch.float32, device=self.device)
        if torch.any(action_weights <= 0.0):
            raise ValueError("teacher_action_weights must be positive")
        self.teacher_action_weights = action_weights.view(1, -1)
        self.teacher_relaxed_mask = action_weights < 0.999
        self.teacher_preserved_mask = ~self.teacher_relaxed_mask

        checkpoint = torch.load(teacher_checkpoint, map_location=self.device, weights_only=False)
        self.teacher_policy = copy.deepcopy(policy).to(self.device)
        self.teacher_policy.load_state_dict(checkpoint["model_state_dict"])
        self.teacher_policy.eval()
        self.teacher_policy.requires_grad_(False)
        print(f"[INFO] Fixed action teacher loaded from: {teacher_checkpoint}")

    @torch.no_grad()
    def reset_student_action_std(self):
        """Use low exploration noise so the verified mean policy can pass strict gates."""
        if self.policy.noise_std_type == "scalar":
            self.policy.std.fill_(self.student_init_std)
        elif self.policy.noise_std_type == "log":
            self.policy.log_std.fill_(torch.log(torch.tensor(self.student_init_std)).item())
        else:
            raise ValueError(f"unsupported student noise type: {self.policy.noise_std_type}")

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_teacher_action_loss = 0.0
        mean_teacher_action_rmse = 0.0
        mean_teacher_relaxed_rmse = 0.0
        mean_teacher_preserved_rmse = 0.0
        mean_student_std_loss = 0.0
        mean_teacher_kl = 0.0
        mean_policy_kl = 0.0
        mean_teacher_projected_rmse = 0.0
        mean_teacher_projection_scale = 0.0
        mean_teacher_projection_active = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            _rnd_state_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / (advantages_batch.std() + 1.0e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            with torch.no_grad():
                teacher_mu_batch = self.teacher_policy.act_inference(obs_batch)
                policy_kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    dim=-1,
                ).mean()

            teacher_square_error = torch.square(mu_batch - teacher_mu_batch)
            teacher_kl = torch.sum(
                teacher_square_error
                * self.teacher_action_weights
                / (2.0 * self.student_init_std**2),
                dim=-1,
            ).mean()

            if self.desired_kl is not None and self.schedule == "adaptive":
                kl_mean = policy_kl
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-6, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-4, self.learning_rate * 1.5)
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = torch.square(value_batch - returns_batch)
                value_losses_clipped = torch.square(value_clipped - returns_batch)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = torch.square(returns_batch - value_batch).mean()

            teacher_action_loss = (
                teacher_square_error * self.teacher_action_weights
            ).sum(dim=-1).mean() / self.teacher_action_weights.sum()
            teacher_action_rmse = torch.sqrt(
                torch.mean(teacher_square_error.detach())
            )
            if torch.any(self.teacher_relaxed_mask):
                teacher_relaxed_rmse = torch.sqrt(
                    teacher_square_error.detach()[:, self.teacher_relaxed_mask].mean()
                )
            else:
                teacher_relaxed_rmse = torch.zeros((), device=self.device)
            if torch.any(self.teacher_preserved_mask):
                teacher_preserved_rmse = torch.sqrt(
                    teacher_square_error.detach()[:, self.teacher_preserved_mask].mean()
                )
            else:
                teacher_preserved_rmse = torch.zeros((), device=self.device)
            student_std_loss = F.mse_loss(
                sigma_batch,
                torch.full_like(sigma_batch, self.student_init_std),
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.teacher_action_coef * teacher_action_loss
                + self.teacher_kl_coef * teacher_kl
                + self.student_std_coef * student_std_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            projected_rmse = torch.zeros((), device=self.device)
            projection_scale = 1.0
            projection_active = 0.0
            if self.teacher_hard_action_rmse_limit > 0.0:
                with torch.no_grad():
                    for _ in range(self.teacher_projection_iterations):
                        projected_mu = self.policy.act_inference(obs_batch)
                        projected_square_error = torch.square(
                            projected_mu - teacher_mu_batch
                        )
                        projected_rmse = torch.sqrt(
                            (
                                projected_square_error
                                * self.teacher_action_weights
                            ).sum(dim=-1).mean()
                            / self.teacher_action_weights.sum()
                        )
                        if projected_rmse <= self.teacher_hard_action_rmse_limit:
                            break
                        step_scale = min(
                            0.95,
                            self.teacher_hard_action_rmse_limit
                            / (projected_rmse.item() + 1.0e-8),
                        )
                        projection_scale *= step_scale
                        projection_active = 1.0
                        for student_param, teacher_param in zip(
                            self.policy.actor.parameters(),
                            self.teacher_policy.actor.parameters(),
                        ):
                            student_param.copy_(
                                teacher_param
                                + step_scale * (student_param - teacher_param)
                            )
                    projected_mu = self.policy.act_inference(obs_batch)
                    projected_rmse = torch.sqrt(
                        (
                            torch.square(projected_mu - teacher_mu_batch)
                            * self.teacher_action_weights
                        ).sum(dim=-1).mean()
                        / self.teacher_action_weights.sum()
                    )

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_teacher_action_loss += teacher_action_loss.item()
            mean_teacher_action_rmse += teacher_action_rmse.item()
            mean_teacher_relaxed_rmse += teacher_relaxed_rmse.item()
            mean_teacher_preserved_rmse += teacher_preserved_rmse.item()
            mean_student_std_loss += student_std_loss.item()
            mean_teacher_kl += teacher_kl.detach().item()
            mean_policy_kl += policy_kl.item()
            mean_teacher_projected_rmse += projected_rmse.item()
            mean_teacher_projection_scale += projection_scale
            mean_teacher_projection_active += projection_active

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "teacher_action": mean_teacher_action_loss / num_updates,
            "teacher_action_rmse": mean_teacher_action_rmse / num_updates,
            "teacher_relaxed_action_rmse": mean_teacher_relaxed_rmse / num_updates,
            "teacher_preserved_action_rmse": mean_teacher_preserved_rmse / num_updates,
            "student_std_anchor": mean_student_std_loss / num_updates,
            "teacher_kl": mean_teacher_kl / num_updates,
            "policy_kl": mean_policy_kl / num_updates,
            "teacher_projected_action_rmse": mean_teacher_projected_rmse / num_updates,
            "teacher_projection_scale": mean_teacher_projection_scale / num_updates,
            "teacher_projection_active": mean_teacher_projection_active / num_updates,
        }


class TeacherRegularizedConstrainedPPO(TeacherRegularizedPPO):
    """Teacher-regularized PPO with a distinct cost critic and primal-dual actor update."""

    def __init__(
        self,
        policy,
        constraint_cost_metric: str,
        constraint_active_metric: str = "shared_clearance_active_fraction",
        constraint_command_name: str = "motion",
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        cost_budget: float = 0.001,
        segment_replay_scale: float = 1.0,
        dual_proportional_gain: float = 25.0,
        dual_integral_gain: float = 0.50,
        dual_derivative_gain: float = 5.0,
        initial_dual_multiplier: float = 0.50,
        min_dual_multiplier: float = 0.0,
        max_dual_multiplier: float = 5.0,
        cost_value_loss_coef: float = 1.0,
        cost_critic_learning_rate: float = 3.0e-4,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if not constraint_cost_metric:
            raise ValueError("constraint_cost_metric must be non-empty")
        if cost_budget < 0.0:
            raise ValueError("cost_budget must be non-negative")
        if not 0.0 <= segment_replay_scale <= 1.0:
            raise ValueError("segment_replay_scale must be in [0, 1]")
        if not min_dual_multiplier <= initial_dual_multiplier <= max_dual_multiplier:
            raise ValueError("initial dual multiplier must lie within configured bounds")

        self.constraint_cost_metric = constraint_cost_metric
        self.constraint_active_metric = constraint_active_metric
        self.constraint_command_name = constraint_command_name
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.cost_budget = cost_budget
        self.segment_replay_scale = segment_replay_scale
        self.dual_proportional_gain = dual_proportional_gain
        self.dual_integral_gain = dual_integral_gain
        self.dual_derivative_gain = dual_derivative_gain
        self.min_dual_multiplier = min_dual_multiplier
        self.max_dual_multiplier = max_dual_multiplier
        self.cost_value_loss_coef = cost_value_loss_coef
        self.dual_multiplier = torch.tensor(
            float(initial_dual_multiplier), dtype=torch.float32, device=self.device
        )
        self.dual_integral = self.dual_multiplier.clone()
        self.dual_previous_error = torch.zeros((), device=self.device)

        self.cost_critic = copy.deepcopy(policy.critic).to(self.device)
        linear_layers = [m for m in self.cost_critic.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            nn.init.zeros_(linear_layers[-1].weight)
            nn.init.zeros_(linear_layers[-1].bias)
        self.cost_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=float(cost_critic_learning_rate)
        )

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        if training_type != "rl":
            raise ValueError("constrained PPO only supports reinforcement learning")
        self.storage = ConstrainedRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, critic_obs):
        actions = super().act(obs, critic_obs)
        self.transition.cost_values = self.cost_critic(critic_obs).detach()
        return actions

    def process_env_step(self, rewards, dones, infos):
        if "constraint_cost" not in infos:
            raise KeyError("runner did not provide infos['constraint_cost']")
        self.transition.costs = infos["constraint_cost"].to(self.device).clone()
        if "constraint_cost_active" not in infos:
            raise KeyError("runner did not provide infos['constraint_cost_active']")
        self.transition.cost_active = infos["constraint_cost_active"].to(
            self.device
        ).clone()
        if "time_outs" in infos:
            self.transition.costs += self.cost_gamma * torch.squeeze(
                self.transition.cost_values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        super().process_env_step(rewards, dones, infos)

    def compute_returns(self, last_critic_obs):
        super().compute_returns(last_critic_obs)
        self.storage.aggregate_active_segment_max(self.segment_replay_scale)
        last_cost_values = self.cost_critic(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values, self.cost_gamma, self.cost_lam
        )

    def update(self):
        mean_raw_step_cost = self.storage.raw_costs.mean().detach()
        mean_step_cost = self.storage.costs.mean().detach()
        cost_error = mean_step_cost - float(self.cost_budget)
        self.dual_integral = torch.clamp(
            self.dual_integral + float(self.dual_integral_gain) * cost_error,
            min=float(self.min_dual_multiplier),
            max=float(self.max_dual_multiplier),
        ).detach()
        cost_derivative = cost_error - self.dual_previous_error
        self.dual_multiplier = torch.clamp(
            self.dual_integral
            + float(self.dual_proportional_gain) * cost_error
            + float(self.dual_derivative_gain) * cost_derivative,
            min=float(self.min_dual_multiplier),
            max=float(self.max_dual_multiplier),
        ).detach()
        self.dual_previous_error = cost_error.detach()

        critic_obs = self.storage.privileged_observations
        if critic_obs is None:
            critic_obs = self.storage.observations
        critic_obs = critic_obs.flatten(0, 1)
        cost_returns = self.storage.cost_returns.flatten(0, 1)
        batch_size = critic_obs.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches
        mean_cost_value_loss = 0.0
        updates = 0
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for batch_index in range(self.num_mini_batches):
                start = batch_index * mini_batch_size
                end = batch_size if batch_index == self.num_mini_batches - 1 else (
                    batch_index + 1
                ) * mini_batch_size
                selected = indices[start:end]
                predicted = self.cost_critic(critic_obs[selected])
                cost_value_loss = torch.square(
                    predicted - cost_returns[selected]
                ).mean()
                self.cost_optimizer.zero_grad()
                (float(self.cost_value_loss_coef) * cost_value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.cost_critic.parameters(), self.max_grad_norm
                )
                self.cost_optimizer.step()
                mean_cost_value_loss += cost_value_loss.item()
                updates += 1

        cost_advantages = self.storage.cost_advantages
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (
            cost_advantages.std() + 1.0e-8
        )
        combined_advantages = (
            self.storage.advantages
            - self.dual_multiplier * cost_advantages
        ) / (1.0 + self.dual_multiplier)
        # Rollout storage is populated under the runner's inference context.
        # Replace the tensor instead of mutating that inference tensor in-place.
        self.storage.advantages = (
            (combined_advantages - combined_advantages.mean())
            / (combined_advantages.std() + 1.0e-8)
        ).clone()

        metrics = super().update()
        metrics.update(
            {
                "constraint_cost_value": mean_cost_value_loss / max(1, updates),
                "constraint_mean_raw_step_cost": mean_raw_step_cost.item(),
                "constraint_mean_step_cost": mean_step_cost.item(),
                "constraint_mean_return": self.storage.cost_returns.mean().item(),
                "constraint_dual_multiplier": self.dual_multiplier.item(),
                "constraint_dual_integral": self.dual_integral.item(),
                "constraint_cost_error": cost_error.item(),
                "constraint_cost_budget": float(self.cost_budget),
                "constraint_segment_replay_scale": float(self.segment_replay_scale),
            }
        )
        return metrics
