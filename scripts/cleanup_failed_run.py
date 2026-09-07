#!/usr/bin/env python3
"""Archive-gated cleanup for checkpoints from one rejected training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def verify_archive(project_root: Path, archive_dir: Path) -> int:
    required = [
        archive_dir / "FAILURE_REPORT_ZH.md",
        archive_dir / "curves" / "complete_training_diagnostics.png",
        archive_dir / "config",
        archive_dir / "archive_manifest.sha256",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"archive is incomplete: {missing}")

    checked = 0
    manifest = archive_dir / "archive_manifest.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        archived_file = (project_root / relative.strip()).resolve()
        if not within(archived_file, archive_dir):
            raise RuntimeError(f"manifest path escapes archive: {archived_file}")
        if not archived_file.is_file() or sha256(archived_file) != expected:
            raise RuntimeError(f"archive hash mismatch: {archived_file}")
        checked += 1
    if checked == 0:
        raise RuntimeError("archive manifest is empty")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    if not root.is_dir() or not within(run_dir, root) or not within(archive_dir, root):
        raise RuntimeError("project, run, or archive path is outside the project root")
    if not run_dir.is_dir() or not archive_dir.is_dir():
        raise RuntimeError("run or archive directory does not exist")

    verified_entries = verify_archive(root, archive_dir)
    protected = {Path(path).resolve() for path in args.protect}
    candidates = sorted(run_dir.glob("model_*.pt"))
    if not candidates:
        raise RuntimeError("no model_*.pt files found directly inside the rejected run")
    conflicts = [str(path) for path in candidates if path.resolve() in protected]
    if conflicts:
        raise RuntimeError(f"candidate intersects protected set: {conflicts}")

    now = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "absolute_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in candidates
    ]
    manifest_path = archive_dir / "deletion_manifest.json"
    payload = {
        "schema_version": 1,
        "status": "planned",
        "planned_at": now,
        "project_root": str(root),
        "rejected_run": str(run_dir),
        "archive_dir": str(archive_dir),
        "archive_entries_verified": verified_entries,
        "protected_paths": sorted(str(path) for path in protected),
        "files": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.execute:
        for record in records:
            target = Path(record["absolute_path"])
            if not within(target, run_dir) or target.parent != run_dir:
                raise RuntimeError(f"refusing unexpected deletion target: {target}")
            if sha256(target) != record["sha256"]:
                raise RuntimeError(f"checkpoint changed after manifest creation: {target}")
            target.unlink()
        remaining = [record["absolute_path"] for record in records if Path(record["absolute_path"]).exists()]
        payload["status"] = "deleted" if not remaining else "verification_failed"
        payload["deleted_at"] = datetime.now(timezone.utc).isoformat()
        payload["remaining_files"] = remaining
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if remaining:
            raise RuntimeError(f"files remain after cleanup: {remaining}")

    print(json.dumps({"manifest": str(manifest_path), "status": payload["status"], "count": len(records)}))


if __name__ == "__main__":
    main()
