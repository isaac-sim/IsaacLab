#!/bin/bash

# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

isaacsim_root_path="${1:?Usage: install_carb_env_shim.sh ISAACSIM_ROOT_PATH [DESTINATION] [PRELOAD_FILE]}"
destination="${2:-/usr/local/lib/libcarb.env.shim.so}"
preload_file="${3:-/etc/ld.so.preload}"
source_shim=""

candidate_paths=(
    "${isaacsim_root_path}/kit/kernel/plugins/libcarb.env.shim.so"
    "${isaacsim_root_path}/kit/libcarb.env.shim.so"
    "${isaacsim_root_path}/_build/linux-x86_64/release/kit/kernel/plugins/libcarb.env.shim.so"
    "${isaacsim_root_path}/_build/target-deps/carb_sdk_plugins/_build/linux-x86_64/release/libcarb.env.shim.so"
)

for candidate_path in "${candidate_paths[@]}"; do
    if [[ -f "${candidate_path}" ]]; then
        source_shim="${candidate_path}"
        break
    fi
done

if [[ -z "${source_shim}" ]]; then
    source_shim="$(find "${isaacsim_root_path}" -name libcarb.env.shim.so -type f -print -quit 2>/dev/null || true)"
fi

if [[ -z "${source_shim}" || ! -f "${source_shim}" ]]; then
    echo "no carb env shim found under ${isaacsim_root_path}; skipping ${preload_file}" >&2
    exit 0
fi

install -D -m 0755 "${source_shim}" "${destination}"
mkdir -p "$(dirname "${preload_file}")"
touch "${preload_file}"

if ! grep -Fqx -- "${destination}" "${preload_file}"; then
    if [[ -s "${preload_file}" && -n "$(tail -c 1 "${preload_file}")" ]]; then
        printf '\n' >> "${preload_file}"
    fi
    printf '%s\n' "${destination}" >> "${preload_file}"
fi

echo "installed carb env shim from ${source_shim}"
