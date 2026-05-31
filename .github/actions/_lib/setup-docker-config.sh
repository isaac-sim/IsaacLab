# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Idempotent docker-config + nvcr.io login setup for the docker-build and
# ecr-build-push-pull composite actions. Both composites source this; the
# second invocation in a job is a no-op so callers don't need to coordinate.
#
# Source this script (do not exec) — it sets DOCKER_CONFIG in the caller's
# environment and writes it to $GITHUB_ENV so subsequent steps inherit it.
# Expects NGC_API_KEY in the environment (optional; warns when missing).

# The runner's credential helper backend is broken ("not implemented") and
# causes docker login calls to fail unless we point DOCKER_CONFIG at a temp
# config with credsStore disabled.

if [ -n "${DOCKER_CONFIG:-}" ] && [ -f "${DOCKER_CONFIG}/config.json" ]; then
  echo "🟢 Docker config already set up at ${DOCKER_CONFIG}, skipping" >&2
  return 0 2>/dev/null || exit 0
fi

DOCKER_CONFIG_DIR=$(mktemp -d)
if [ -f "${HOME}/.docker/config.json" ]; then
  python3 -c "import json; cfg=json.load(open('${HOME}/.docker/config.json')); cfg['credsStore']=''; cfg.pop('credHelpers',None); json.dump(cfg,open('${DOCKER_CONFIG_DIR}/config.json','w'))"
else
  echo '{"credsStore":""}' > "${DOCKER_CONFIG_DIR}/config.json"
fi
export DOCKER_CONFIG="${DOCKER_CONFIG_DIR}"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "DOCKER_CONFIG=${DOCKER_CONFIG_DIR}" >> "${GITHUB_ENV}"
fi

if [ -n "${NGC_API_KEY:-}" ]; then
  echo "🔵 Logging into nvcr.io..." >&2
  docker login -u '$oauthtoken' -p "${NGC_API_KEY}" nvcr.io
else
  echo "🟠 NGC_API_KEY not set - skipping nvcr.io login (normal for fork PRs)" >&2
fi

unset DOCKER_CONFIG_DIR
