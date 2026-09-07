#!/usr/bin/env python3
"""SFT a phase-gated S52 delta-action adapter while freezing model_92099."""

from __future__ import annotations

import argparse
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
            nn.Linear(148, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 27),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


class DeployAdapter(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        max_output: float,
    ) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("mean", mean.reshape(1, 148).float())
        self.register_buffer("std", std.reshape(1, 148).float())
        self.max_output = float(max_output)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.mean) / (self.std + 0.01)
        leg_delta = self.max_output * torch.tanh(self.network(normalized))
        other_delta = torch.zeros(
            (leg_delta.shape[0], 15), dtype=leg_delta.dtype, device=leg_delta.device
        )
        return torch.cat((leg_delta, other_delta), dim=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_from_checkpoint(checkpoint: dict) -> Actor:
    actor = Actor()
    actor.actor.load_state_dict(
        {
            key.removeprefix("actor."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
    )
    return actor


def rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((actual - expected) ** 2))) if actual.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="s52_stairs_delta_adapter_v12c")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-output", type=float, default=0.10)
    parser.add_argument("--retention-coef", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()
    if args.hidden_dim <= 0 or args.max_output <= 0.0:
        raise ValueError("hidden dimension and max output must be positive")
    if args.validation_stride < 2:
        raise ValueError("validation stride must be at least 2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = actor_from_checkpoint(checkpoint).eval()
    mean = checkpoint["obs_norm_state_dict"]["_mean"].float().reshape(1, 148)
    std = checkpoint["obs_norm_state_dict"]["_std"].float().reshape(1, 148)

    data = np.load(args.dataset, allow_pickle=False)
    observations = torch.from_numpy(data["policy_observations"].astype(np.float32))
    target_actions = torch.from_numpy(data["target_actions"].astype(np.float32))
    sample_weights = torch.from_numpy(data["sample_weights"].astype(np.float32)).reshape(-1, 1)
    source = torch.from_numpy(data["source"].astype(np.int64))
    frames = torch.from_numpy(data["motion_frame"].astype(np.int64))
    normalized = (observations - mean) / (std + 0.01)
    with torch.inference_mode():
        teacher_actions = teacher(normalized)
    target_delta = target_actions[:, :12] - teacher_actions[:, :12]

    indices = torch.arange(len(observations))
    validation = torch.zeros(len(observations), dtype=torch.bool)
    for source_id in (0, 1):
        selected = indices[source == source_id]
        validation[selected[:: args.validation_stride]] = True
    training = ~validation

    network = nn.Sequential(
        nn.Linear(148, args.hidden_dim),
        nn.Tanh(),
        nn.Linear(args.hidden_dim, 12),
    )
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(
            normalized[training], target_delta[training], sample_weights[training], source[training]
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    for epoch in range(args.epochs):
        network.train()
        totals = {"loss": 0.0, "target": 0.0, "retention": 0.0}
        count = 0
        for obs_batch, target_batch, weight_batch, source_batch in loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            weight_batch = weight_batch.to(device)
            source_batch = source_batch.to(device)
            predicted = args.max_output * torch.tanh(network(obs_batch))
            target_loss = torch.mean(
                weight_batch * torch.mean((predicted - target_batch) ** 2, dim=1, keepdim=True)
            )
            retention_mask = source_batch == 0
            retention_loss = (
                torch.mean(predicted[retention_mask] ** 2)
                if torch.any(retention_mask)
                else torch.zeros((), device=device)
            )
            loss = target_loss + args.retention_coef * retention_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            batch_count = len(obs_batch)
            count += batch_count
            totals["loss"] += float(loss.detach()) * batch_count
            totals["target"] += float(target_loss.detach()) * batch_count
            totals["retention"] += float(retention_loss.detach()) * batch_count

        network.cpu().eval()
        with torch.inference_mode():
            validation_pred = args.max_output * torch.tanh(network(normalized[validation]))
        validation_source = source[validation]
        actual_mask = validation_source == 1
        retention_mask = validation_source == 0
        actual_rmse = rmse(validation_pred[actual_mask], target_delta[validation][actual_mask])
        retention_rmse = rmse(
            validation_pred[retention_mask], torch.zeros_like(validation_pred[retention_mask])
        )
        score = actual_rmse + 5.0 * retention_rmse
        history.append(
            {
                "epoch": epoch + 1,
                "loss": totals["loss"] / count,
                "target_loss": totals["target"] / count,
                "retention_loss": totals["retention"] / count,
                "validation_actual_target_rmse": actual_rmse,
                "validation_retention_rmse": retention_rmse,
            }
        )
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone() for name, value in network.state_dict().items()
            }
        network.to(device)

    network.cpu().eval()
    network.load_state_dict(best_state)
    with torch.inference_mode():
        predicted_delta = args.max_output * torch.tanh(network(normalized))
    retention = source == 0
    correction = source == 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripted_path = args.output_dir / f"{args.name}.pt"
    deploy = DeployAdapter(network, mean, std, args.max_output).eval()
    torch.jit.script(deploy).save(str(scripted_path))
    state_path = args.output_dir / f"{args.name}_state.pt"
    torch.save(
        {
            "state_dict": network.state_dict(),
            "mean": mean,
            "std": std,
            "max_output": args.max_output,
            "teacher_sha256": sha256(args.teacher_checkpoint),
            "dataset_sha256": sha256(args.dataset),
        },
        state_path,
    )

    history_path = args.output_dir / f"{args.name}_training.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    curve_path = args.output_dir / f"{args.name}_training.png"
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot([row["loss"] for row in history], label="total")
    axes[0].plot([row["target_loss"] for row in history], label="target")
    axes[0].plot([row["retention_loss"] for row in history], label="retention")
    axes[0].set_yscale("log")
    axes[0].set_title("Delta-adapter SFT losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        [row["validation_actual_target_rmse"] for row in history], label="S52 target"
    )
    axes[1].plot(
        [row["validation_retention_rmse"] for row in history], label="S53 zero delta"
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validation delta-action RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    report = {
        "name": args.name,
        "status": "candidate_adapter_until_complete_S52_PhysX_rollout_passes",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_sha256": sha256(args.teacher_checkpoint),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "samples": int(len(observations)),
        "s53_retention_samples": int(torch.count_nonzero(retention)),
        "s52_correction_samples": int(torch.count_nonzero(correction)),
        "frame_min": int(torch.min(frames)),
        "frame_max": int(torch.max(frames)),
        "hidden_dim": args.hidden_dim,
        "max_output": args.max_output,
        "best_validation_score": best_score,
        "actual_target_delta_rmse": rmse(predicted_delta[correction], target_delta[correction]),
        "retention_delta_rmse": rmse(
            predicted_delta[retention], torch.zeros_like(predicted_delta[retention])
        ),
        "retention_delta_abs_max": float(torch.max(torch.abs(predicted_delta[retention]))),
        "actual_delta_abs_p95": float(
            np.percentile(np.abs(predicted_delta[correction].numpy()), 95)
        ),
        "actual_delta_abs_max": float(torch.max(torch.abs(predicted_delta[correction]))),
        "scripted_adapter": str(scripted_path.resolve()),
        "scripted_adapter_sha256": sha256(scripted_path),
        "state": str(state_path.resolve()),
        "history_csv": str(history_path.resolve()),
        "training_curve": str(curve_path.resolve()),
    }
    report_path = args.output_dir / f"{args.name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
