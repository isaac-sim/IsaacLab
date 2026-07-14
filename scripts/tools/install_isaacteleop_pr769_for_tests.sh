#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Test-only pin: this merge revision includes NVIDIA/IsaacTeleop PR 769.  This
# PR is part of the dVRK Teleop Extended Universe and, once merged, this pin
# will be updated to align with the corresponding released IsaacTeleop version.
set -euo pipefail

readonly ISAAC_TELEOP_REPOSITORY="https://github.com/NVIDIA/IsaacTeleop.git"
readonly ISAAC_TELEOP_PR769_MERGE_SHA="790d6cb4e948de377975c76ed1e9cbf5098e10fc"
readonly ISAACLAB_PYTHON="${ISAACLAB_PATH:?ISAACLAB_PATH must be set}/_isaac_sim/python.sh"
readonly ISAACLAB_PYTHON_SCRIPTS="$("${ISAACLAB_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
readonly BUILD_ROOT="${ISAACLAB_TELEOP_TEST_CACHE:-/tmp/isaacteleop-pr769-${ISAAC_TELEOP_PR769_MERGE_SHA}}"
readonly SOURCE_DIR="${BUILD_ROOT}/source"
readonly CMAKE_BUILD_DIR="${BUILD_ROOT}/build"
readonly SOURCE_REVISION_FILE="${BUILD_ROOT}/source-revision"

# The Isaac Sim launcher owns a Python installation whose console-script
# directory is not always on ``PATH`` in a test container.  CMake finds the
# same ``uv`` executable that the launcher installs only after this export.
export PATH="${ISAACLAB_PYTHON_SCRIPTS}:${PATH}"

if ! command -v git >/dev/null || ! dpkg-query --show --showformat='${db:Status-Status}' libx11-dev 2>/dev/null | grep -qx installed; then
    # IsaacTeleop's CMake dependencies are fetched through Git and its static
    # OpenXR loader selects the Xlib backend.  The minimal Isaac Sim runtime
    # image used by the test action includes neither build prerequisite.
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends git libx11-dev
fi

if ! command -v uv >/dev/null; then
    "${ISAACLAB_PYTHON}" -m pip install uv
fi

if [[ ! -f "${SOURCE_REVISION_FILE}" ]] \
    || [[ "$(<"${SOURCE_REVISION_FILE}")" != "${ISAAC_TELEOP_PR769_MERGE_SHA}" ]]; then
    rm -rf "${SOURCE_DIR}" "${CMAKE_BUILD_DIR}"
    mkdir -p "${BUILD_ROOT}"
    if command -v git >/dev/null; then
        git init --quiet "${SOURCE_DIR}"
        git -C "${SOURCE_DIR}" remote add origin "${ISAAC_TELEOP_REPOSITORY}"
        git -C "${SOURCE_DIR}" fetch --depth 1 origin "${ISAAC_TELEOP_PR769_MERGE_SHA}"
        git -C "${SOURCE_DIR}" checkout --quiet --detach FETCH_HEAD
        test "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${ISAAC_TELEOP_PR769_MERGE_SHA}"
    else
        # Isaac Sim runtime images do not all include Git.  GitHub's archive
        # endpoint names the immutable commit directly, so this fallback pins
        # the identical merge revision rather than installing a branch tip.
        archive_path="$(mktemp "${BUILD_ROOT}/isaacteleop-${ISAAC_TELEOP_PR769_MERGE_SHA}.XXXXXX.tar.gz")"
        curl --fail --location --retry 3 --retry-delay 2 \
            "https://github.com/NVIDIA/IsaacTeleop/archive/${ISAAC_TELEOP_PR769_MERGE_SHA}.tar.gz" \
            --output "${archive_path}"
        mkdir -p "${SOURCE_DIR}"
        tar --extract --gzip --file "${archive_path}" --strip-components=1 --directory "${SOURCE_DIR}"
        rm -f "${archive_path}"
    fi
    printf '%s\n' "${ISAAC_TELEOP_PR769_MERGE_SHA}" > "${SOURCE_REVISION_FILE}"
fi

test -f "${SOURCE_DIR}/CMakeLists.txt"
test "$(<"${SOURCE_REVISION_FILE}")" = "${ISAAC_TELEOP_PR769_MERGE_SHA}"

cmake -S "${SOURCE_DIR}" -B "${CMAKE_BUILD_DIR}" \
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
"${ISAACLAB_PYTHON}" -c 'from isaacteleop.retargeters.DVRK.control import DVRKPSMCartesianClutchStateMachine'
