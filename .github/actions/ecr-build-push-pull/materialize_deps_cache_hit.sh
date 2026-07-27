#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

DEPS_ECR_IMAGE="${1:?dependency-cache ECR image is required}"
ECR_IMAGE="${2:?commit ECR image is required}"
LOCAL_IMAGE="${3:?local image tag is required}"

echo "🔵 Tagging dependency-cache image as commit image ${ECR_IMAGE}..."
docker buildx imagetools create -t "${ECR_IMAGE}" "${DEPS_ECR_IMAGE}"

echo "🔵 Pulling ${ECR_IMAGE} from ECR..."
docker pull "${ECR_IMAGE}"
docker tag "${ECR_IMAGE}" "${LOCAL_IMAGE}"

echo "🟢 Materialized ${DEPS_ECR_IMAGE} as ${ECR_IMAGE} and ${LOCAL_IMAGE}"
