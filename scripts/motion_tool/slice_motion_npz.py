#!/usr/bin/env python3
"""Slice all frame-aligned arrays in a LejuLab motion NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="Inclusive frame index")
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=True) as source:
        frame_count = int(source["joint_pos"].shape[0])
        if not 0 <= args.start <= args.end < frame_count:
            raise ValueError(f"Invalid range {args.start}..{args.end} for {frame_count} frames")
        output = {}
        for key in source.files:
            value = source[key]
            output[key] = value[args.start : args.end + 1] if value.ndim and value.shape[0] == frame_count else value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_frames": frame_count,
        "source_range_inclusive": [args.start, args.end],
        "output_frames": args.end - args.start + 1,
        "fps": int(np.asarray(output["fps"]).reshape(-1)[0]),
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
