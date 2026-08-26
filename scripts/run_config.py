#!/usr/bin/env python3
"""Turn documented YAML files into reproducible train, sim and export commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    return data


def hydra_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flatten(prefix: str, value, output: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    else:
        output.append(f"{prefix}={hydra_value(value)}")


def stage_checkpoint(checkpoint: Path, experiment: str, run_name: str) -> tuple[str, str]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    run_dir = ROOT / "logs" / "rsl_rl" / experiment / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    link = run_dir / checkpoint.name
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoint:
            raise RuntimeError(f"引导 checkpoint 冲突: {link}")
    else:
        link.symlink_to(checkpoint)
    return run_name, checkpoint.name


def show_and_run(command: list[str], dry_run: bool) -> int:
    print("[leju_rl]", shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def train(config: dict, dry_run: bool) -> int:
    task = config["task"]
    runtime = config["runtime"]
    command = [sys.executable, "scripts/reinforcement_learning/rsl_rl/train.py"]
    command += ["--task", task["id"]]
    command += ["--num_envs", str(runtime["num_envs"])]
    command += ["--max_iterations", str(runtime["max_iterations"])]
    command += ["--seed", str(runtime["seed"])]
    command += ["--device", str(runtime["device"])]
    command += ["--logger", str(runtime["logger"])]
    command += ["--run_name", str(task["run_name"])]
    if runtime.get("headless", True):
        command.append("--headless")

    warm = config.get("warm_start", {})
    if warm.get("enabled", False):
        load_run, checkpoint = stage_checkpoint(
            ROOT / warm["checkpoint"], task["experiment"], warm["bootstrap_run"]
        )
        command += ["--resume", "True", "--load_run", load_run, "--checkpoint", checkpoint]
        if warm.get("reset_optimizer", True):
            command.append("--reset_optimizer")

    runner_map = {
        "save_interval": "agent.save_interval",
        "num_steps_per_env": "agent.num_steps_per_env",
        "empirical_normalization": "agent.empirical_normalization",
    }
    for key, hydra_key in runner_map.items():
        if key in config.get("runner", {}):
            command.append(f"{hydra_key}={hydra_value(config['runner'][key])}")

    ppo_map = {
        "learning_rate": "agent.algorithm.learning_rate",
        "num_learning_epochs": "agent.algorithm.num_learning_epochs",
        "num_mini_batches": "agent.algorithm.num_mini_batches",
        "clip_param": "agent.algorithm.clip_param",
        "value_loss_coef": "agent.algorithm.value_loss_coef",
        "entropy_coef": "agent.algorithm.entropy_coef",
        "gamma": "agent.algorithm.gamma",
        "lam": "agent.algorithm.lam",
        "desired_kl": "agent.algorithm.desired_kl",
        "max_grad_norm": "agent.algorithm.max_grad_norm",
    }
    for key, hydra_key in ppo_map.items():
        command.append(f"{hydra_key}={hydra_value(config['ppo'][key])}")

    teacher_map = {
        "action_coef": "agent.algorithm.teacher_action_coef",
        "kl_coef": "agent.algorithm.teacher_kl_coef",
        "student_init_std": "agent.algorithm.student_init_std",
        "student_std_coef": "agent.algorithm.student_std_coef",
        "freeze_observation_normalizer": "agent.algorithm.freeze_observation_normalizer",
    }
    teacher_checkpoint = (ROOT / config["teacher"]["checkpoint"]).resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint 不存在: {teacher_checkpoint}")
    command.append(
        f"agent.algorithm.teacher_checkpoint={hydra_value(str(teacher_checkpoint))}"
    )
    for key, hydra_key in teacher_map.items():
        command.append(f"{hydra_key}={hydra_value(config['teacher'][key])}")

    constraint_map = {
        "metric": "agent.algorithm.constraint_cost_metric",
        "active_metric": "agent.algorithm.constraint_active_metric",
        "cost_budget": "agent.algorithm.cost_budget",
        "cost_gamma": "agent.algorithm.cost_gamma",
        "cost_lam": "agent.algorithm.cost_lam",
        "segment_replay_scale": "agent.algorithm.segment_replay_scale",
        "proportional_gain": "agent.algorithm.dual_proportional_gain",
        "integral_gain": "agent.algorithm.dual_integral_gain",
        "derivative_gain": "agent.algorithm.dual_derivative_gain",
        "initial_multiplier": "agent.algorithm.initial_dual_multiplier",
        "min_multiplier": "agent.algorithm.min_dual_multiplier",
        "max_multiplier": "agent.algorithm.max_dual_multiplier",
        "cost_value_loss_coef": "agent.algorithm.cost_value_loss_coef",
        "cost_critic_learning_rate": "agent.algorithm.cost_critic_learning_rate",
    }
    for key, hydra_key in constraint_map.items():
        command.append(f"{hydra_key}={hydra_value(config['constraint'][key])}")

    reward_map = {
        "motion_feet_position.weight": "env.rewards.motion_feet_pos.weight",
        "motion_feet_position.minimum_tracking_weight": "env.rewards.motion_feet_pos.params.minimum_tracking_weight",
        "motion_feet_velocity.weight": "env.rewards.motion_feet_vel.weight",
        "spatial_riser_corridor.weight": "env.rewards.spatial_riser_corridor.weight",
        "pre_touchdown_soft_landing.weight": "env.rewards.pre_touchdown_soft_landing.weight",
        "feet_slide_velocity.weight": "env.rewards.feet_slide_vel.weight",
        "feet_contact_force.weight": "env.rewards.feet_contact_forces.weight",
        "contact_gated_clearance.weight": "env.rewards.shared_riser_clearance_cost.weight",
        "contact_gated_clearance.safety_distance": "env.rewards.shared_riser_clearance_cost.params.safety_distance",
        "contact_gated_clearance.hard_distance": "env.rewards.shared_riser_clearance_cost.params.hard_distance",
        "contact_gated_clearance.require_low_contact_for_swing": "env.rewards.shared_riser_clearance_cost.params.require_low_contact_for_swing",
    }
    for yaml_path, hydra_key in reward_map.items():
        value = config["rewards"]
        for part in yaml_path.split("."):
            value = value[part]
        command.append(f"{hydra_key}={hydra_value(value)}")
    return show_and_run(command, dry_run)


def simulate(config: dict, dry_run: bool) -> int:
    load_run, checkpoint = stage_checkpoint(
        ROOT / config["checkpoint"], config["experiment"], config["bootstrap_run"]
    )
    output = ROOT / config["rollout_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/reinforcement_learning/rsl_rl/record_stairs_policy_rollout.py",
        "--task", config["task"], "--output", str(output), "--steps", str(config["steps"]),
        "--seed", str(config["seed"]), "--device", str(config["device"]),
        "--resume", "True", "--load_run", load_run, "--checkpoint", checkpoint,
    ]
    result = show_and_run(command, dry_run)
    if result or dry_run or not config.get("render_video", True):
        return result
    video = ROOT / config["video_output"]
    video.parent.mkdir(parents=True, exist_ok=True)
    return show_and_run([
        sys.executable, "scripts/motion_tool/render_model29999_style_rollout.py",
        "--input", str(output), "--output", str(video),
    ], False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "sim"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    return train(config, args.dry_run) if args.mode == "train" else simulate(config, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
