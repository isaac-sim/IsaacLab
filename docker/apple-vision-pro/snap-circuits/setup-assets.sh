#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ASSET_DIR="${ISAACLAB_SNAP_CIRCUITS_ASSET_ROOT:-${SCRIPT_DIR}/assets}"
readonly SOURCE_DIR="${ASSET_DIR}/source"
readonly UNPACKED_DIR="${ASSET_DIR}/unpacked"
readonly PREPARED_DIR="${ASSET_DIR}/prepared"
readonly SHARPA_DIR="${ASSET_DIR}/sharpa-urdf-usd-xml"
readonly SHARPA_REPOSITORY="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
readonly SHARPA_REF="6eea427eb24189519f32b9f21674cd534d3f973c"
readonly PROHAND_DIR="${ASSET_DIR}/pro-models"
readonly PROHAND_REPOSITORY="https://github.com/Proception-AI/pro-models.git"
readonly PROHAND_REF="eb8bd682d1ab1a40b8dfbd9d293665165d5519ce"

usage() {
    cat <<'EOF'
Usage: setup-assets.sh [--rclone-source REMOTE:PATH] [--rclone-config FILE]
                       [--s3-scale SCALE] [--asset-set demo|catalog]

Copies the private Snap Circuits prefix with rclone (when --rclone-source is
provided), unpacks the two local ZIP bundles, fetches the pinned Sharpa Wave
and ProHand models, and converts/arranges the curated demo set into a table-top
USD. Pass --asset-set catalog to place every discovered object instead.

Run this script from the host. The Isaac Lab base container must already be up.
EOF
}

rclone_source="${RCLONE_SNAP_CIRCUITS_SOURCE:-}"
rclone_config="${RCLONE_CONFIG:-}"
s3_scale="${ISAACLAB_SNAP_CIRCUITS_S3_SCALE:-1.0}"
asset_set="${ISAACLAB_SNAP_CIRCUITS_ASSET_SET:-demo}"

while (($#)); do
    case "$1" in
        --rclone-source)
            rclone_source="$2"
            shift 2
            ;;
        --rclone-config)
            rclone_config="$2"
            shift 2
            ;;
        --s3-scale)
            s3_scale="$2"
            shift 2
            ;;
        --asset-set)
            asset_set="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${asset_set}" != "demo" && "${asset_set}" != "catalog" ]]; then
    echo "Unknown asset set '${asset_set}'; expected demo or catalog." >&2
    exit 2
fi

mkdir -p "${SOURCE_DIR}" "${UNPACKED_DIR}" "${PREPARED_DIR}"
# The container runs as uid 1000 while the robolab host checkout may use a
# different uid. This ignored generated-output directory must be writable
# through the bind mount; source assets remain host-owned and read-only.
chmod 0777 "${PREPARED_DIR}"

if [[ -n "${rclone_source}" ]]; then
    rclone_args=()
    if [[ -n "${rclone_config}" ]]; then
        rclone_args+=(--config "${rclone_config}")
    fi
    rclone "${rclone_args[@]}" copy "${rclone_source}" "${SOURCE_DIR}/s3-snap-circuits" --progress
fi

for archive in sc100_mesh_bundle.zip test_tube_16mm_rack_18mm_compatible.zip; do
    if [[ ! -f "${SOURCE_DIR}/${archive}" ]]; then
        echo "Missing ${SOURCE_DIR}/${archive}" >&2
        echo "Copy the archive there before running this script." >&2
        exit 1
    fi
    unzip -q -o "${SOURCE_DIR}/${archive}" -d "${UNPACKED_DIR}"
done

if [[ ! -d "${SHARPA_DIR}/.git" ]]; then
    git clone --filter=blob:none "${SHARPA_REPOSITORY}" "${SHARPA_DIR}"
fi
git -C "${SHARPA_DIR}" fetch --depth 1 origin "${SHARPA_REF}"
git -C "${SHARPA_DIR}" checkout --detach "${SHARPA_REF}"

if [[ ! -d "${PROHAND_DIR}/.git" ]]; then
    git clone --filter=blob:none "${PROHAND_REPOSITORY}" "${PROHAND_DIR}"
fi
git -C "${PROHAND_DIR}" fetch --depth 1 origin "${PROHAND_REF}"
git -C "${PROHAND_DIR}" checkout --detach "${PROHAND_REF}"
python3 "${SCRIPT_DIR}/prepare_prohand_urdf.py" --repository "${PROHAND_DIR}"

if [[ -d "${SOURCE_DIR}/s3-snap-circuits" ]]; then
    mkdir -p "${UNPACKED_DIR}/s3-snap-circuits"
    cp -R "${SOURCE_DIR}/s3-snap-circuits/." "${UNPACKED_DIR}/s3-snap-circuits/"
fi

docker_command=(docker exec isaac-lab-base bash -lc)
if ! docker info >/dev/null 2>&1; then
    docker_command=(sudo docker exec isaac-lab-base bash -lc)
fi

container_script="/workspace/isaaclab/docker/apple-vision-pro/snap-circuits/prepare_assets.py"
container_source="/workspace/isaaclab/docker/apple-vision-pro/snap-circuits/assets/unpacked"
container_output="/workspace/isaaclab/docker/apple-vision-pro/snap-circuits/assets/prepared"
"${docker_command[@]}" \
    "cd /workspace/isaaclab && ./isaaclab.sh -p ${container_script} --source-dir ${container_source} --output-dir ${container_output} --s3-scale ${s3_scale} --asset-set ${asset_set}"

echo "Snap Circuits demo assets are ready in ${PREPARED_DIR}"
