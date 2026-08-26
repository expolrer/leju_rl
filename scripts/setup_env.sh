#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lejulab_isaac}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "未找到 Conda 初始化脚本: ${CONDA_SH}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

python -m pip install -e "${ROOT_DIR}/source/leju_robot"
python -m pip install -r "${ROOT_DIR}/requirements-tools.txt"

echo "环境安装完成。进入仓库后运行：source scripts/activate_env.sh"

