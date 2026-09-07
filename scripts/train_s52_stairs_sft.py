#!/usr/bin/env python3
"""Conservatively fine-tune model_92099 on S53/S52 residual-action supervision."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(148, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 27),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_from_checkpoint(checkpoint: dict) -> Actor:
    actor = Actor()
    state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.actor.load_state_dict(state)
    return actor


def rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean((actual - expected) ** 2)))


def metrics(
    actor: Actor,
    normalized: torch.Tensor,
    targets: torch.Tensor,
    teacher_actions: torch.Tensor,
    source: torch.Tensor,
) -> dict[str, float]:
    actor.eval()
    with torch.inference_mode():
        predicted = actor(normalized)
    retention = source == 0
    correction = source == 1
    return {
        "target_rmse_all": rmse(predicted, targets),
        "target_rmse_retention": rmse(predicted[retention], targets[retention]),
        "target_rmse_s52_correction": rmse(predicted[correction], targets[correction]),
        "teacher_drift_rmse_all": rmse(predicted, teacher_actions),
        "teacher_drift_rmse_retention": rmse(
            predicted[retention], teacher_actions[retention]
        ),
        "teacher_drift_abs_max": float(torch.max(torch.abs(predicted - teacher_actions))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="model_92099_s52_stairs_sft_v12")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--teacher-anchor", type=float, default=0.50)
    parser.add_argument("--max-parameter-delta", type=float, default=0.025)
    parser.add_argument("--train-layers", choices=("final", "last-two"), default="final")
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.max_parameter_delta <= 0.0:
        raise ValueError("max parameter delta must be positive")
    if args.validation_stride < 2:
        raise ValueError("validation stride must be at least 2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = actor_from_checkpoint(checkpoint).eval()
    student = actor_from_checkpoint(checkpoint)
    source_actor_state = copy.deepcopy(student.actor.state_dict())
    mean = checkpoint["obs_norm_state_dict"]["_mean"].float().reshape(1, 148)
    std = checkpoint["obs_norm_state_dict"]["_std"].float().reshape(1, 148)

    dataset = np.load(args.dataset, allow_pickle=False)
    observations = torch.from_numpy(dataset["policy_observations"].astype(np.float32))
    targets = torch.from_numpy(dataset["target_actions"].astype(np.float32))
    weights = torch.from_numpy(dataset["sample_weights"].astype(np.float32)).reshape(-1, 1)
    source = torch.from_numpy(dataset["source"].astype(np.int64))
    frames = torch.from_numpy(dataset["motion_frame"].astype(np.int64))
    if observations.shape[1:] != (148,) or targets.shape != (len(observations), 27):
        raise ValueError("dataset must contain [N,148] observations and [N,27] actions")
    if len(weights) != len(observations) or len(source) != len(observations):
        raise ValueError("dataset arrays have inconsistent row counts")

    normalized = (observations - mean) / (std + 0.01)
    with torch.inference_mode():
        teacher_actions = teacher(normalized)

    indices = torch.arange(len(observations))
    validation_mask = torch.zeros(len(observations), dtype=torch.bool)
    for source_id in (0, 1):
        source_indices = indices[source == source_id]
        validation_mask[source_indices[:: args.validation_stride]] = True
    train_mask = ~validation_mask
    if not torch.any(train_mask) or not torch.any(validation_mask):
        raise ValueError("train/validation split is empty")

    train_dataset = TensorDataset(
        normalized[train_mask],
        targets[train_mask],
        teacher_actions[train_mask],
        weights[train_mask],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    train_module_indices = (6,) if args.train_layers == "final" else (4, 6)
    for module_index in train_module_indices:
        for parameter in student.actor[module_index].parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-6)
    student.to(device)

    bounded_parameters: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module_index in train_module_indices:
        for suffix in ("weight", "bias"):
            name = f"{module_index}.{suffix}"
            parameter = dict(student.actor.named_parameters())[name]
            bounded_parameters.append((parameter, source_actor_state[name].to(device)))

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(student.actor.state_dict())
    best_score = float("inf")
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        total_target = 0.0
        total_anchor = 0.0
        sample_count = 0
        for obs_batch, target_batch, teacher_batch, weight_batch in loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            teacher_batch = teacher_batch.to(device)
            weight_batch = weight_batch.to(device)
            predicted = student(obs_batch)
            target_loss = torch.mean(
                weight_batch * torch.mean((predicted - target_batch) ** 2, dim=1, keepdim=True)
            )
            anchor_loss = torch.mean((predicted - teacher_batch) ** 2)
            loss = target_loss + args.teacher_anchor * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            with torch.no_grad():
                for parameter, initial in bounded_parameters:
                    parameter.copy_(
                        torch.maximum(
                            torch.minimum(parameter, initial + args.max_parameter_delta),
                            initial - args.max_parameter_delta,
                        )
                    )
            count = len(obs_batch)
            sample_count += count
            total_loss += float(loss.detach()) * count
            total_target += float(target_loss.detach()) * count
            total_anchor += float(anchor_loss.detach()) * count

        student.cpu()
        validation = metrics(
            student,
            normalized[validation_mask],
            targets[validation_mask],
            teacher_actions[validation_mask],
            source[validation_mask],
        )
        score = validation["target_rmse_s52_correction"] + 2.0 * validation[
            "teacher_drift_rmse_retention"
        ]
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / sample_count,
            "target_loss": total_target / sample_count,
            "anchor_loss": total_anchor / sample_count,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(student.actor.state_dict())
        student.to(device)

    student.cpu()
    student.actor.load_state_dict(best_state)
    final_metrics = metrics(student, normalized, targets, teacher_actions, source)
    parameter_deltas = {}
    for name, value in student.actor.state_dict().items():
        delta = value - source_actor_state[name]
        parameter_deltas[name] = {
            "l2": float(torch.linalg.vector_norm(delta)),
            "abs_max": float(torch.max(torch.abs(delta))),
        }

    output_checkpoint = copy.deepcopy(checkpoint)
    for name, value in student.actor.state_dict().items():
        output_checkpoint["model_state_dict"][f"actor.{name}"] = value.cpu()
    metadata = {
        "purpose": "S53 successful stairs retention plus S52 dynamics-correction SFT",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_sha256": sha256(args.teacher_checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "observation_dimension": 148,
        "action_dimension": 27,
        "action_semantics": "residual added to the S52 retargeted joint-position command",
        "train_layers": args.train_layers,
        "reset_optimizer_before_rl": True,
        "acceptance_status": "candidate_until_complete_S52_Isaac_PhysX_rollout_passes",
    }
    output_checkpoint["s52_stairs_sft"] = metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.name}.pt"
    torch.save(output_checkpoint, checkpoint_path)
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    metadata["checkpoint_sha256"] = sha256(checkpoint_path)

    history_path = args.output_dir / f"{args.name}_training.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    curve_path = args.output_dir / f"{args.name}_training.png"
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot([row["loss"] for row in history], label="total")
    axes[0].plot([row["target_loss"] for row in history], label="target")
    axes[0].plot([row["anchor_loss"] for row in history], label="teacher anchor")
    axes[0].set_yscale("log")
    axes[0].set_title("S52 stairs SFT training losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        [row["validation_target_rmse_s52_correction"] for row in history],
        label="S52 correction target",
    )
    axes[1].plot(
        [row["validation_teacher_drift_rmse_retention"] for row in history],
        label="S53 retention drift",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validation residual-action RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    report = {
        **metadata,
        "device": str(device),
        "samples": int(len(observations)),
        "training_samples": int(torch.count_nonzero(train_mask)),
        "validation_samples": int(torch.count_nonzero(validation_mask)),
        "s53_retention_samples": int(torch.count_nonzero(source == 0)),
        "s52_correction_samples": int(torch.count_nonzero(source == 1)),
        "frame_min": int(torch.min(frames)),
        "frame_max": int(torch.max(frames)),
        "epochs": args.epochs,
        "best_validation_score": best_score,
        "final_metrics": final_metrics,
        "parameter_deltas": parameter_deltas,
        "history_csv": str(history_path.resolve()),
        "training_curve": str(curve_path.resolve()),
    }
    report_path = args.output_dir / f"{args.name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
