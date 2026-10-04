#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[5]
    / "rsl_rl_extensions"
    / "residual_contact_actor_critic.py"
)
SPEC = importlib.util.spec_from_file_location("residual_contact_actor_critic", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load residual policy module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ResidualContactActorCritic = MODULE.PhaseGatedResidualContactActorCritic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    policy = ResidualContactActorCritic(
        num_actor_obs=155,
        num_critic_obs=280,
        num_actions=27,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.02,
        base_obs_dim=148,
        adapter_hidden_dims=[64, 32],
        adapter_scale=0.02,
        phase_obs_index=-1,
        phase_gate_start=0.82,
        phase_gate_width=0.04,
    )
    target = policy.state_dict()
    for key, value in source["model_state_dict"].items():
        if key.startswith("actor."):
            mapped = "actor.base." + key.removeprefix("actor.")
        elif key.startswith("critic.") or key in {"std", "log_std"}:
            mapped = key
        else:
            continue
        target[mapped] = value.detach().clone()
    policy.load_state_dict(target)

    checkpoint = dict(source)
    checkpoint["model_state_dict"] = policy.state_dict()
    obs_norm = dict(source["obs_norm_state_dict"])
    for key in ("_mean", "_var", "_std"):
        old = obs_norm[key]
        fill = 0.0 if key == "_mean" else 1.0
        obs_norm[key] = torch.cat(
            (old, torch.full((old.shape[0], 7), fill, dtype=old.dtype)), dim=1
        )
    checkpoint["obs_norm_state_dict"] = obs_norm
    checkpoint["optimizer_state_dict"] = {}
    checkpoint.setdefault("infos", {})
    checkpoint["infos"] = dict(checkpoint["infos"] or {})
    checkpoint["infos"].update(
        {
            "source_checkpoint": str(args.source.resolve()),
            "source_sha256": sha256(args.source),
            "migration": "frozen_148d_base_plus_zero_initialized_7d_phase_gated_contact_adapter",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)
    print(sha256(args.output))


if __name__ == "__main__":
    main()

