#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/config/lab.yaml}"
CONFIRM="${2:-}"

read_yaml() {
  python - "$CONFIG" "$1" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
value = data
for key in sys.argv[2].split('.'):
    value = value[key]
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

DEPLOY_REPO="$(read_yaml deploy_repo)"
MODEL="${ROOT_DIR}/$(read_yaml policy_onnx)"
TARGET="${DEPLOY_REPO}/$(read_yaml controller_policy_path)"
DRY_RUN="$(read_yaml dry_run)"
ROBOT_VERSION="$(read_yaml robot_version)"
LAUNCH_COMMAND="$(read_yaml launch_command)"

[[ -d "$DEPLOY_REPO" ]] || { echo "LejuLab-Deploy 不存在: $DEPLOY_REPO" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ONNX 模型不存在: $MODEL，请先导出策略" >&2; exit 2; }
mkdir -p "$(dirname "$TARGET")"
install -m 0644 "$MODEL" "$TARGET"
echo "模型已部署到: $TARGET"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "dry_run=true：未连接仿真或实机。"
  exit 0
fi
if [[ "$CONFIRM" != "--confirm-real-robot" ]]; then
  echo "拒绝启动实机。确认急停、吊架、限位和人员撤离后追加 --confirm-real-robot。" >&2
  exit 3
fi

export ROBOT_VERSION
cd "$DEPLOY_REPO"
# shellcheck disable=SC1091
source devel/setup.bash
exec bash -lc "$LAUNCH_COMMAND"

