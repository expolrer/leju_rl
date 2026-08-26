#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lejulab_isaac}"
REAL_HOME="${REAL_HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
CONDA_SH="${CONDA_SH:-${REAL_HOME}/miniconda3/etc/profile.d/conda.sh}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export LEJU_RL_ROOT="${ROOT_DIR}"
export REAL_HOME
export HOME="${LEJU_RL_OMNI_HOME:-${ROOT_DIR}/.omni_home}"
export XDG_CACHE_HOME="${HOME}/.cache"
export XDG_CONFIG_HOME="${HOME}/.config"
export XDG_DATA_HOME="${HOME}/.local/share"

# Some workstations contain the same NVIDIA ICD in both /etc and /usr/share.
# Limit this process to one manifest so Isaac Sim does not enumerate one GPU
# twice. This does not modify the host driver or other users' processes.
DEFAULT_NVIDIA_ICD="/etc/vulkan/icd.d/nvidia_icd.json"
if [[ -f "${DEFAULT_NVIDIA_ICD}" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${DEFAULT_NVIDIA_ICD}}"
  export VK_DRIVER_FILES="${VK_DRIVER_FILES:-${VK_ICD_FILENAMES}}"
fi

ISAACLAB_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export ISAACLAB_SOURCE_DIR="${ISAACLAB_SOURCE_DIR:-${ISAACLAB_SITE}/isaaclab/source}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAACLAB_SITE}/isaacsim}"
export PYTHONPATH="${ROOT_DIR}/source/leju_robot:${ISAACLAB_SOURCE_DIR}/isaaclab:${ISAACLAB_SOURCE_DIR}/isaaclab_assets:${ISAACLAB_SOURCE_DIR}/isaaclab_mimic:${ISAACLAB_SOURCE_DIR}/isaaclab_rl:${ISAACLAB_SOURCE_DIR}/isaaclab_tasks:${PYTHONPATH:-}"

# Isaac Sim wheel contains shared libraries in several nested directories.
# Populate LD_LIBRARY_PATH once so headless Vulkan/PhysX can locate the same
# runtime libraries used by the validated training environment.
if [[ -z "${ISAACSIM_LD_READY:-}" ]]; then
  ISAACSIM_LD_DIRS="$(find "${ISAACSIM_ROOT}" -type f -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)"
  export LD_LIBRARY_PATH="${ISAACSIM_LD_DIRS}:${ISAACLAB_SITE}/omni:${LD_LIBRARY_PATH:-}"
  export ISAACSIM_LD_READY=1
fi

cd "${ROOT_DIR}"
