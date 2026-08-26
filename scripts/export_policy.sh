#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/activate_env.sh"

EXPERIMENT="kuavoS53_stairs_contact_gated_margin_cmdp_tracking"
RUN_NAME="published_model_92150"
CHECKPOINT="model_92150.pt"
RUN_DIR="${ROOT_DIR}/logs/rsl_rl/${EXPERIMENT}/${RUN_NAME}"
SOURCE_MODEL="${ROOT_DIR}/checkpoints/kuavo_s53_stairs/${CHECKPOINT}"
OUTPUT_DIR="${ROOT_DIR}/exported/model_92150"

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"
ln -sfn "${SOURCE_MODEL}" "${RUN_DIR}/${CHECKPOINT}"

# play.py 在进入仿真循环前导出 JIT 和 ONNX；只录制 1 帧使其自动退出。
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Tracking-Stairs-ContactGatedMarginCMDP-KuavoS53-Play \
  --num_envs 1 \
  --device cuda:0 \
  --resume True \
  --load_run "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --video \
  --video_length 1

cp "${RUN_DIR}/exported/policy.onnx" "${OUTPUT_DIR}/policy.onnx"
cp "${RUN_DIR}/exported/policy.pt" "${OUTPUT_DIR}/policy.pt"
echo "策略已导出到 ${OUTPUT_DIR}"

