# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Shared deps-hash computation for the docker-build and ecr-build-push-pull
# composite actions. Both invoke this script so a registry-side cache hit and
# a local-store cache hit always agree on the same `deps-<hash>` tag.
#
# Source this script (do not exec) — it sets DEPS_HASH in the caller's
# environment. Caller must export DOCKERFILE_PATH, ISAACSIM_BASE_IMAGE,
# ISAACSIM_VERSION before sourcing. Diagnostic output goes to stderr so the
# caller's stdout stays usable.

: "${DOCKERFILE_PATH:?compute-deps-hash: DOCKERFILE_PATH must be set}"
: "${ISAACSIM_BASE_IMAGE:?compute-deps-hash: ISAACSIM_BASE_IMAGE must be set}"
: "${ISAACSIM_VERSION:?compute-deps-hash: ISAACSIM_VERSION must be set}"

# Exact files/dirs whose full content is hashed. The Dockerfile is first.
_DEPS_FILES=(
  "${DOCKERFILE_PATH}"
  isaaclab.sh
  environment.yml
  source/isaaclab/isaaclab/cli
)
# Manifest files matched repo-wide via git ls-files.
_DEPS_MANIFEST_PATTERN='(setup\.py|pyproject\.toml|setup\.cfg|extension\.toml|requirements[^/]*\.txt|uv\.lock)$'

# Resolve the actual base image digest so a new push of a mutable tag
# (e.g. latest-develop) invalidates the deps cache automatically.
_BASE_IMAGE_DIGEST=$(docker buildx imagetools inspect \
  "${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}" \
  --format '{{json .Manifest.Digest}}' 2>/dev/null | tr -d '"' || true)
if [ -n "${_BASE_IMAGE_DIGEST}" ]; then
  _BASE_IMAGE_UNIQ_ID="${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}:${_BASE_IMAGE_DIGEST}"
else
  echo "🟠 Could not resolve base image digest, falling back to tag string" >&2
  _BASE_IMAGE_UNIQ_ID="${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}"
fi

_MANIFEST_FILES=$(git ls-files | grep -E "${_DEPS_MANIFEST_PATTERN}" || true)
# shellcheck disable=SC2086  # word-splitting MANIFEST_FILES is intentional
_FILE_HASH=$(git ls-files -s "${_DEPS_FILES[@]}" ${_MANIFEST_FILES} 2>/dev/null \
  | sha256sum | cut -c1-16)
DEPS_HASH=$(printf '%s %s' "${_FILE_HASH}" "${_BASE_IMAGE_UNIQ_ID}" | sha256sum | cut -c1-16)

unset _DEPS_FILES _DEPS_MANIFEST_PATTERN _BASE_IMAGE_DIGEST _BASE_IMAGE_UNIQ_ID _MANIFEST_FILES _FILE_HASH
