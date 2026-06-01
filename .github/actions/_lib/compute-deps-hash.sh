#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Shared deps-hash computation for the docker-build and ecr-build-push-pull
# composite actions. Both invoke this script so a registry-side cache hit and
# a local-store cache hit always agree on the same `deps-<hash>` tag.
#
# Usage: compute-deps-hash.sh <dockerfile-path> <isaacsim-base-image> <isaacsim-version>
# Prints the 16-character deps-hash to stdout. Diagnostic output goes to stderr.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "compute-deps-hash: expected 3 args (dockerfile-path, isaacsim-base-image, isaacsim-version)" >&2
  exit 2
fi

dockerfile_path="$1"
isaacsim_base_image="$2"
isaacsim_version="$3"

# Exact files/dirs whose full content is hashed. The Dockerfile is first.
deps_files=(
  "${dockerfile_path}"
  isaaclab.sh
  environment.yml
  source/isaaclab/isaaclab/cli
)
deps_manifest_pattern='(setup\.py|pyproject\.toml|setup\.cfg|extension\.toml|requirements[^/]*\.txt|uv\.lock)$'

# Resolve the actual base image digest so a new push of a mutable tag
# (e.g. latest-develop) invalidates the deps cache automatically.
base_image_digest=$(docker buildx imagetools inspect \
  "${isaacsim_base_image}:${isaacsim_version}" \
  --format '{{json .Manifest.Digest}}' 2>/dev/null | tr -d '"' || true)
if [ -n "${base_image_digest}" ]; then
  base_image_uniq_id="${isaacsim_base_image}:${isaacsim_version}:${base_image_digest}"
else
  echo "🟠 Could not resolve base image digest, falling back to tag string" >&2
  base_image_uniq_id="${isaacsim_base_image}:${isaacsim_version}"
fi

mapfile -t manifest_files < <(git ls-files | grep -E "${deps_manifest_pattern}" || true)
file_hash=$(git ls-files -s "${deps_files[@]}" "${manifest_files[@]}" 2>/dev/null \
  | sha256sum | cut -c1-16)
deps_hash=$(printf '%s %s' "${file_hash}" "${base_image_uniq_id}" | sha256sum | cut -c1-16)

printf '%s\n' "${deps_hash}"
