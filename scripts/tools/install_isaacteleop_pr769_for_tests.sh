#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Test-only pin for the public dVRK retargeter API in NVIDIA/IsaacTeleop PR 769.
# Remove this source build once the first IsaacTeleop release containing that
# API satisfies IsaacLab's normal version constraint.
set -euo pipefail

readonly ISAAC_TELEOP_REPOSITORY="https://github.com/NVIDIA/IsaacTeleop.git"
readonly ISAAC_TELEOP_PR769_HEAD_SHA="ca175df7afc8198cbba0592cd1b447b11a4f3165"
readonly UV_VERSION="0.11.29"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ISAACLAB_ROOT="${ISAACLAB_PATH:-$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)}"

selected_python=""
python_source=""
if [[ -n "${ISAACLAB_PYTHON:-}" ]]; then
    selected_python="${ISAACLAB_PYTHON}"
    python_source="ISAACLAB_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    selected_python="${VIRTUAL_ENV}/bin/python"
    python_source="VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    selected_python="${CONDA_PREFIX}/bin/python"
    python_source="CONDA_PREFIX"
elif [[ -x "${ISAACLAB_ROOT}/env_isaaclab/bin/python" ]]; then
    selected_python="${ISAACLAB_ROOT}/env_isaaclab/bin/python"
    python_source="${ISAACLAB_ROOT}/env_isaaclab"
elif [[ -x /isaac-sim/python.sh ]]; then
    selected_python="/isaac-sim/python.sh"
    python_source="/isaac-sim"
elif [[ -x "${ISAACLAB_ROOT}/_isaac_sim/python.sh" ]]; then
    selected_python="${ISAACLAB_ROOT}/_isaac_sim/python.sh"
    python_source="${ISAACLAB_ROOT}/_isaac_sim"
elif command -v python3 >/dev/null 2>&1; then
    selected_python="$(command -v python3)"
    python_source="PATH"
else
    echo "Unable to find a Python interpreter for the IsaacTeleop source build." >&2
    echo "Set ISAACLAB_PYTHON to an executable interpreter or launcher." >&2
    exit 1
fi

if [[ ! -x "${selected_python}" ]]; then
    echo "Python interpreter selected from ${python_source} is not executable: ${selected_python}" >&2
    exit 1
fi
readonly ISAACLAB_PYTHON="${selected_python}"

if ! selected_python_version="$(
    "${ISAACLAB_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"; then
    echo "Failed to run Python interpreter selected from ${python_source}: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi
if [[ ! "${selected_python_version}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Selected Python interpreter returned an invalid version: ${selected_python_version}" >&2
    exit 1
fi
readonly ISAAC_TELEOP_PYTHON_VERSION="${selected_python_version}"

if ! selected_python_scripts="$(
    "${ISAACLAB_PYTHON}" -c 'import sysconfig; path = sysconfig.get_path("scripts"); assert path; print(path)'
)"; then
    echo "Failed to locate the console-script directory for ${ISAACLAB_PYTHON}." >&2
    exit 1
fi
readonly ISAACLAB_PYTHON_SCRIPTS="${selected_python_scripts}"
readonly BUILD_ROOT="${ISAACLAB_TELEOP_TEST_CACHE:-/tmp/isaacteleop-pr769-${ISAAC_TELEOP_PR769_HEAD_SHA}}"
readonly SOURCE_DIR="${BUILD_ROOT}/source"
readonly CMAKE_BUILD_DIR="${BUILD_ROOT}/build-python-${ISAAC_TELEOP_PYTHON_VERSION}"
readonly SOURCE_REVISION_FILE="${BUILD_ROOT}/source-revision"

# The selected Python environment's console-script directory is not always on
# ``PATH`` in a test container. CMake finds the same ``uv`` executable that
# the interpreter installs only after this export.
export PATH="${ISAACLAB_PYTHON_SCRIPTS}:${PATH}"

assert_source_checkout_clean() {
    local source_status
    if ! source_status="$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)"; then
        echo "Failed to inspect IsaacTeleop source checkout: ${SOURCE_DIR}" >&2
        return 1
    fi
    if [[ -n "${source_status}" ]]; then
        echo "IsaacTeleop source checkout is not clean: ${SOURCE_DIR}" >&2
        printf '%s\n' "${source_status}" >&2
        return 1
    fi
}

if ! command -v git >/dev/null || ! dpkg-query --show --showformat='${db:Status-Status}' libx11-dev 2>/dev/null | grep -qx installed; then
    # IsaacTeleop's CMake dependencies are fetched through Git and its static
    # OpenXR loader selects the Xlib backend.  The minimal Isaac Sim runtime
    # image used by the test action includes neither build prerequisite.
    apt_get=(apt-get)
    if (( EUID != 0 )); then
        if ! command -v sudo >/dev/null; then
            echo "Installing IsaacTeleop build prerequisites requires root or sudo." >&2
            exit 1
        fi
        apt_get=(sudo apt-get)
    fi
    "${apt_get[@]}" update
    DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" install --yes --no-install-recommends git libx11-dev
fi

installed_uv_version="$(
    "${ISAACLAB_PYTHON}" -c 'import importlib.metadata; print(importlib.metadata.version("uv"))' 2>/dev/null || true
)"
if [[ "${installed_uv_version}" != "${UV_VERSION}" ]]; then
    "${ISAACLAB_PYTHON}" -m pip install "uv==${UV_VERSION}"
fi
test "$("${ISAACLAB_PYTHON}" -c 'import importlib.metadata; print(importlib.metadata.version("uv"))')" = "${UV_VERSION}"

source_cache_valid=false
source_cache_head=""
source_cache_status=""
if [[ -f "${SOURCE_REVISION_FILE}" ]] \
    && [[ "$(<"${SOURCE_REVISION_FILE}")" = "${ISAAC_TELEOP_PR769_HEAD_SHA}" ]] \
    && [[ -d "${SOURCE_DIR}/.git" ]] \
    && source_cache_head="$(git -C "${SOURCE_DIR}" rev-parse HEAD 2>/dev/null)" \
    && [[ "${source_cache_head}" = "${ISAAC_TELEOP_PR769_HEAD_SHA}" ]] \
    && source_cache_status="$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all 2>/dev/null)" \
    && [[ -z "${source_cache_status}" ]]; then
    source_cache_valid=true
fi

if [[ "${source_cache_valid}" != "true" ]]; then
    rm -rf "${SOURCE_DIR}" "${CMAKE_BUILD_DIR}"
    mkdir -p "${BUILD_ROOT}"
    git init --quiet "${SOURCE_DIR}"
    git -C "${SOURCE_DIR}" remote add origin "${ISAAC_TELEOP_REPOSITORY}"
    git -C "${SOURCE_DIR}" fetch --depth 1 origin "${ISAAC_TELEOP_PR769_HEAD_SHA}"
    git -C "${SOURCE_DIR}" checkout --quiet --detach FETCH_HEAD
    test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${ISAAC_TELEOP_PR769_HEAD_SHA}"
    assert_source_checkout_clean
    printf '%s\n' "${ISAAC_TELEOP_PR769_HEAD_SHA}" > "${SOURCE_REVISION_FILE}"
fi

test -f "${SOURCE_DIR}/CMakeLists.txt"
test "$(<"${SOURCE_REVISION_FILE}")" = "${ISAAC_TELEOP_PR769_HEAD_SHA}"
test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${ISAAC_TELEOP_PR769_HEAD_SHA}"
assert_source_checkout_clean

cmake -S "${SOURCE_DIR}" -B "${CMAKE_BUILD_DIR}" \
    -DISAAC_TELEOP_PYTHON_VERSION="${ISAAC_TELEOP_PYTHON_VERSION}" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PLUGINS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_VIZ=OFF \
    -DENABLE_CLANG_FORMAT_CHECK=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${CMAKE_BUILD_DIR}" --target python_wheel --parallel

wheel_path="$(find "${CMAKE_BUILD_DIR}/wheels" -maxdepth 1 -name 'isaacteleop-*.whl' -print -quit)"
test -n "${wheel_path}"
"${ISAACLAB_PYTHON}" -m pip install --force-reinstall --no-deps "${wheel_path}"
"${ISAACLAB_PYTHON}" - <<'PY'
from isaacteleop.retargeters import (
    DVRKPSMClutchConfig,
    DVRKPSMClutchRetargeter,
    DVRKPSMGripperConfig,
    DVRKPSMGripperRetargeter,
)

assert DVRKPSMClutchConfig is not None
assert DVRKPSMGripperConfig is not None
assert DVRKPSMClutchRetargeter.OUTPUT_POSE == "ee_pose"
assert DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS == "jaw_targets"
PY
