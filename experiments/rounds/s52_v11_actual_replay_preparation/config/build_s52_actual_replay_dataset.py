#!/usr/bin/env python3
"""Build a compact, provenance-stamped S52 actual-state replay dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--last-row", type=int, default=264)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parser.parse_args()
    source = np.load(args.input, allow_pickle=True)
    required = {
        "checkpoint",
        "task",
        "joint_names",
        "all_body_names",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "motion_frame",
        "policy_observations",
        "actions",
        "contact_body_names",
        "contact_force_w",
    }
    missing = sorted(required.difference(source.files))
    if missing:
        raise KeyError(f"source rollout is missing: {missing}")
    if source["joint_pos"].shape[1] != 29:
        raise ValueError("expected 29 S52 simulator joints")
    if source["policy_observations"].shape[1] != 148:
        raise ValueError("expected 148 policy observations")
    if source["actions"].shape[1] != 27:
        raise ValueError("expected 27 policy actions")
    stop = args.last_row + 1
    if stop >= len(source["root_pos"]):
        raise ValueError("last row exceeds the source rollout")
    if np.any(source["motion_frame"][:stop] != np.arange(stop)):
        raise ValueError("source prefix is not a continuous frame-zero rollout")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_path=np.asarray(str(args.input.resolve())),
        source_sha256=np.asarray(sha256(args.input)),
        source_checkpoint=source["checkpoint"],
        source_task=source["task"],
        source_rows=np.arange(stop, dtype=np.int64),
        joint_names=source["joint_names"],
        all_body_names=source["all_body_names"],
        root_pos=source["root_pos"][:stop],
        root_quat=source["root_quat"][:stop],
        root_lin_vel=source["root_lin_vel"][:stop],
        root_ang_vel=source["root_ang_vel"][:stop],
        joint_pos=source["joint_pos"][:stop],
        joint_vel=source["joint_vel"][:stop],
        motion_frame=source["motion_frame"][:stop],
        policy_observations=source["policy_observations"][:stop],
        actions=source["actions"][:stop],
        contact_body_names=source["contact_body_names"],
        contact_force_w=source["contact_force_w"][:stop],
    )
    print(f"wrote {args.output} with {stop} rows; sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
