import os

import torch

from rsl_rl.env import VecEnv
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner_module
from leju_robot.rsl_rl_extensions.algorithms.teacher_regularized_ppo import (
    TeacherRegularizedConstrainedPPO,
    TeacherRegularizedPPO,
)
from leju_robot.rsl_rl_extensions.utils.exporter import attach_onnx_metadata, export_policy_as_onnx


class _ClippedEmpiricalNormalization(EmpiricalNormalization):
    """EmpiricalNormalization with output clipping to prevent extreme values."""

    def forward(self, x):
        # Inline parent forward to avoid super() / class method calls that
        # TorchScript cannot resolve after __class__ swap.
        if self.training:
            self.update(x)
        return torch.clamp((x - self._mean) / (self._std + self.eps), min=-5.0, max=5.0)


class RobotOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with ONNX export and NaN reward protection."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        use_constrained_ppo = bool(
            train_cfg.get("algorithm", {}).get("constraint_cost_metric")
        )
        use_teacher_regularization = bool(
            train_cfg.get("algorithm", {}).get("teacher_checkpoint")
        )
        if use_constrained_ppo or use_teacher_regularization:
            original_ppo_class = rsl_on_policy_runner_module.PPO
            rsl_on_policy_runner_module.PPO = (
                TeacherRegularizedConstrainedPPO
                if use_constrained_ppo
                else TeacherRegularizedPPO
            )
            try:
                super().__init__(env, train_cfg, log_dir, device)
            finally:
                rsl_on_policy_runner_module.PPO = original_ppo_class
        else:
            super().__init__(env, train_cfg, log_dir, device)

        if isinstance(self.alg, TeacherRegularizedConstrainedPPO):
            original_constraint_step = self.env.step

            def _constraint_step(actions):
                obs, rewards, dones, infos = original_constraint_step(actions)
                command = self.env.unwrapped.command_manager.get_term(
                    self.alg.constraint_command_name
                )
                if self.alg.constraint_cost_metric not in command.metrics:
                    raise KeyError(
                        "constraint metric was not produced by the environment: "
                        f"{self.alg.constraint_cost_metric}"
                    )
                cost = command.metrics[self.alg.constraint_cost_metric]
                infos["constraint_cost"] = torch.nan_to_num(
                    cost.detach().clone(), nan=0.0, posinf=10.0, neginf=0.0
                )
                if self.alg.constraint_active_metric not in command.metrics:
                    raise KeyError(
                        "constraint activity metric was not produced: "
                        f"{self.alg.constraint_active_metric}"
                    )
                infos["constraint_cost_active"] = (
                    command.metrics[self.alg.constraint_active_metric]
                    .detach()
                    .clone()
                    .gt(0.0)
                )
                return obs, rewards, dones, infos

            self.env.step = _constraint_step

        # Safeguards (NaN reward handling, reward clamp, normalizer output clamp)
        # are opt-in via cfg flag. Only Velocity-KuavoS54 enables them; tracking
        # and other tasks early-return here so behavior matches master exactly.
        if not train_cfg.get("enable_runner_safeguards", False):
            return

        # Wrap env.step to replace NaN rewards with 0, preventing NaN from
        # corrupting PPO's GAE computation. NaN rewards arise because Isaac Lab
        # computes rewards before checking termination conditions — when
        # invalid_state terminates an environment, its reward for that step
        # is already NaN.
        _original_step = self.env.step

        def _safe_step(actions):
            obs, rewards, dones, infos = _original_step(actions)
            rewards = torch.nan_to_num(rewards, nan=0.0)
            rewards = torch.clamp(rewards, min=-1000.0, max=1000.0)
            return obs, rewards, dones, infos

        self.env.step = _safe_step

        # Swap normalizer class to clip extreme values after normalization.
        # EmpiricalNormalization has no output clipping — when _std is very small
        # (e.g., push_force is zero most of the time), normalized values can be
        # extreme, causing the critic to output inf → inf value loss.
        # Using __class__ swap instead of monkey-patching forward, because the
        # ONNX exporter deepcopies the normalizer — closures don't survive that.
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.__class__ = _ClippedEmpiricalNormalization
        if isinstance(self.privileged_obs_normalizer, EmpiricalNormalization):
            self.privileged_obs_normalizer.__class__ = _ClippedEmpiricalNormalization

    def train_mode(self):
        super().train_mode()
        if isinstance(self.alg, TeacherRegularizedConstrainedPPO):
            self.alg.cost_critic.train()
        if isinstance(self.alg, TeacherRegularizedPPO) and self.alg.freeze_observation_normalizer:
            # model_113994 and its teacher must continue to see the same normalized observations.
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer=load_optimizer)
        if isinstance(self.alg, TeacherRegularizedConstrainedPPO):
            loaded = torch.load(path, map_location=self.device, weights_only=False)
            if "cost_critic_state_dict" in loaded:
                self.alg.cost_critic.load_state_dict(loaded["cost_critic_state_dict"])
            if load_optimizer and "cost_optimizer_state_dict" in loaded:
                self.alg.cost_optimizer.load_state_dict(
                    loaded["cost_optimizer_state_dict"]
                )
            if "constraint_dual_multiplier" in loaded:
                self.alg.dual_multiplier.fill_(
                    float(loaded["constraint_dual_multiplier"])
                )
            if "constraint_dual_integral" in loaded:
                self.alg.dual_integral.fill_(float(loaded["constraint_dual_integral"]))
            if "constraint_dual_previous_error" in loaded:
                self.alg.dual_previous_error.fill_(
                    float(loaded["constraint_dual_previous_error"])
                )
        if isinstance(self.alg, TeacherRegularizedPPO) and not load_optimizer:
            self.alg.reset_student_action_std()
            print(f"[INFO] Student exploration std reset to: {self.alg.student_init_std}")
        return infos

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if isinstance(self.alg, TeacherRegularizedConstrainedPPO):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved["cost_critic_state_dict"] = self.alg.cost_critic.state_dict()
            saved["cost_optimizer_state_dict"] = self.alg.cost_optimizer.state_dict()
            saved["constraint_dual_multiplier"] = self.alg.dual_multiplier.item()
            saved["constraint_dual_integral"] = self.alg.dual_integral.item()
            saved["constraint_dual_previous_error"] = (
                self.alg.dual_previous_error.item()
            )
            torch.save(saved, path)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        export_policy_as_onnx(
            self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
        )
        try:
            attach_onnx_metadata(self.env.unwrapped, "none", path=policy_path, filename=filename)
        except Exception:
            pass
