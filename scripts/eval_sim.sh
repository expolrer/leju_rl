#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/activate_env.sh"
exec python "${ROOT_DIR}/scripts/run_config.py" sim "${1:-${ROOT_DIR}/config/sim.yaml}" "${@:2}"

