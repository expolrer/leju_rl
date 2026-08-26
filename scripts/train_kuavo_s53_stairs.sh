#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/activate_lejulab_isaac.sh" >/dev/null 2>&1
set -u

MODE="${MODE:-step1}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
SEED="${SEED:-42}"
LOGGER="${LOGGER:-tensorboard}"

case "${MODE}" in
  step1)
    TASK="Tracking-Stairs-Step1-KuavoS53"
    DEFAULT_RUN_NAME="s45_medoid10_step1_mesh_warmstart_v1"
    ;;
  full)
    TASK="Tracking-Stairs-Full-KuavoS53"
    DEFAULT_RUN_NAME="s45_medoid10_full_mesh_tracking_v1"
    ;;
  updown)
    TASK="Tracking-Stairs-UpDown-KuavoS53"
    DEFAULT_RUN_NAME="s53_up_platform_down_warm59998_v1"
    ;;
  forward_down)
    TASK="Tracking-Stairs-ForwardDown-KuavoS53"
    DEFAULT_RUN_NAME="s53_forward_down_warm59998_v1"
    ;;
  forward_updown)
    TASK="Tracking-Stairs-ForwardUpDown-KuavoS53"
    DEFAULT_RUN_NAME="s53_forward_updown_v1"
    ;;
  forward_updown_stable)
    TASK="Tracking-Stairs-ForwardUpDownStable-KuavoS53"
    DEFAULT_RUN_NAME="s53_forward_updown_stable_gate_v1"
    ;;
  step_to_down)
    TASK="Tracking-Stairs-StepToDown-KuavoS53"
    DEFAULT_RUN_NAME="s53_step_to_down_gated_v1"
    ;;
  step_to_down_gate_fixed)
    TASK="Tracking-Stairs-StepToDownGateFixed-KuavoS53"
    DEFAULT_RUN_NAME="s53_step_to_down_gate_fixed_v2"
    ;;
  *)
    echo "MODE must be 'step1', 'full', 'updown', 'forward_down', 'forward_updown', 'forward_updown_stable', 'step_to_down', or 'step_to_down_gate_fixed', got: ${MODE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

exec python scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --max_iterations "${MAX_ITERATIONS}" \
  --logger "${LOGGER}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  "$@"
