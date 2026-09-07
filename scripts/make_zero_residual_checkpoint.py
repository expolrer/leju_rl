#!/usr/bin/env python3
"""Create an inference-only checkpoint whose actor outputs zero residuals."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import torch


ACTOR_WEIGHT = re.compile(r"^actor\.(\d+)\.weight$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError("checkpoint has no model_state_dict")

    candidates: list[tuple[int, str]] = []
    for key, value in state.items():
        match = ACTOR_WEIGHT.match(key)
        if match and value.ndim == 2:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise KeyError("checkpoint has no actor linear weights")

    layer_index, weight_key = max(candidates)
    bias_key = f"actor.{layer_index}.bias"
    if bias_key not in state:
        raise KeyError(f"missing final actor bias: {bias_key}")
    weight = state[weight_key]
    bias = state[bias_key]
    if tuple(weight.shape)[0] != 27 or tuple(bias.shape) != (27,):
        raise ValueError(
            f"expected a 27-D actor output, got weight={tuple(weight.shape)}, "
            f"bias={tuple(bias.shape)}"
        )

    output_checkpoint = copy.deepcopy(checkpoint)
    output_state = output_checkpoint["model_state_dict"]
    output_state[weight_key] = torch.zeros_like(weight)
    output_state[bias_key] = torch.zeros_like(bias)
    metadata = {
        "purpose": "S52 direct retargeted-joint-NPZ zero-residual physics diagnostic",
        "source_checkpoint": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "final_actor_layer": layer_index,
        "action_dimension": 27,
        "training_candidate": False,
        "acceptance_rule": "diagnostic_only_until_full Isaac/PhysX rollout passes",
    }
    output_checkpoint["zero_residual_diagnostic"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    metadata["output_checkpoint"] = str(args.output.resolve())
    metadata["output_sha256"] = sha256(args.output)

    report = args.report or args.output.with_suffix(".json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
