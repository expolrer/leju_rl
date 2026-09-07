#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-config/train_s52_stairs_sft_v12.yaml}"

cd "${ROOT_DIR}"
exec python scripts/run_config.py sft "${CONFIG}"
