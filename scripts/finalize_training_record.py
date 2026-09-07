#!/usr/bin/env python3
"""Validate and hash a completed training-record directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    archive = args.archive_dir.resolve(strict=True)
    archive.relative_to(root)

    requirements = {
        "failure_report": list(archive.glob("*FAILURE_REPORT*.md")),
        "config": [path for path in (archive / "config").glob("*") if path.is_file()],
        "curves": [path for path in (archive / "curves").glob("*") if path.is_file()],
        "rollout_json": [path for path in (archive / "rollouts").glob("*.json") if path.is_file()],
    }
    missing = [name for name, files in requirements.items() if not files]
    if missing:
        raise RuntimeError(f"training record is incomplete: {missing}")

    excluded = {"archive_manifest.sha256", "deletion_manifest.json"}
    files = [
        path
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    empty = [str(path) for path in files if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"training record contains empty files: {empty}")

    manifest = archive / "archive_manifest.sha256"
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "archive": str(archive),
        "manifest": str(manifest),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "requirements": {name: len(paths) for name, paths in requirements.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
