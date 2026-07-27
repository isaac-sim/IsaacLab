<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# ECR Cache-Hit Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that an ECR dependency-cache hit leaves the requested local Docker image tag available for the following test action.

**Architecture:** Move the cache-hit alias/pull/tag sequence into a strict shell helper invoked by the composite action. A fake-Docker regression test will execute the helper and assert the registry alias is created first, then pulled, then tagged locally.

**Tech Stack:** Bash with `set -euo pipefail`, Docker CLI, pytest, GitHub composite actions.

## Global Constraints

- Do not change rendering-job `continue-on-error`.
- Keep ECR cache-hit failures fatal.
- Do not alter full-build or exact-image-hit behavior.
- Add no required or optional dependency.
- Verify the regression test fails before adding the helper and passes afterward.
- Run commands through `./isaaclab.sh -p` and run `./isaaclab.sh -f` before committing.

---

### Task 1: Materialize dependency-cache hits locally

**Files:**

- Create: `.github/actions/ecr-build-push-pull/materialize_deps_cache_hit.sh`
- Create: `.github/actions/ecr-build-push-pull/test_materialize_deps_cache_hit.py`
- Modify: `.github/actions/ecr-build-push-pull/action.yml`
- Modify: `.github/actions/ecr-build-push-pull/README.md`

**Interfaces:**

- Produces: `materialize_deps_cache_hit.sh <deps-ecr-image> <commit-ecr-image> <local-image>`.
- Consumes: Docker CLI subcommands `buildx imagetools create`, `pull`, and `tag`.

- [ ] **Step 1: Write the failing fake-Docker regression**

Create a fake `docker` executable that appends `"$*"` to a log, prepend its directory to `PATH`, and run:

```python
result = subprocess.run(
    [
        "bash",
        str(_SCRIPT_PATH),
        "registry.example/repo:deps-abc",
        "registry.example/repo:commit-123",
        "isaac-lab-ci:develop-123",
    ],
    cwd=_REPO_ROOT,
    env=env,
    check=False,
    capture_output=True,
    text=True,
)
```

Assert:

```python
assert result.returncode == 0, result.stderr
assert docker_log.read_text(encoding="utf-8").splitlines() == [
    "buildx imagetools create -t registry.example/repo:commit-123 registry.example/repo:deps-abc",
    "pull registry.example/repo:commit-123",
    "tag registry.example/repo:commit-123 isaac-lab-ci:develop-123",
]
```

- [ ] **Step 2: Run the regression before the helper exists**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/ecr-build-push-pull/test_materialize_deps_cache_hit.py -v
```

Expected: FAIL because `materialize_deps_cache_hit.sh` does not exist.

- [ ] **Step 3: Implement the strict materialization helper**

Create an executable file with the current-year license:

```bash
#!/usr/bin/env bash
set -euo pipefail

DEPS_ECR_IMAGE="${1:?dependency-cache ECR image is required}"
ECR_IMAGE="${2:?commit ECR image is required}"
LOCAL_IMAGE="${3:?local image tag is required}"

echo "🔵 Tagging dependency cache as commit image ${ECR_IMAGE}..."
docker buildx imagetools create -t "${ECR_IMAGE}" "${DEPS_ECR_IMAGE}"
echo "🔵 Pulling commit image ${ECR_IMAGE}..."
docker pull "${ECR_IMAGE}"
echo "🔵 Tagging commit image as local image ${LOCAL_IMAGE}..."
docker tag "${ECR_IMAGE}" "${LOCAL_IMAGE}"
echo "🟢 Materialized ${DEPS_ECR_IMAGE} as ${LOCAL_IMAGE}"
```

- [ ] **Step 4: Call the helper from the cache-hit branch**

Replace the inline `docker buildx imagetools create` call in `Check deps cache` with:

```bash
bash "${GITHUB_ACTION_PATH}/materialize_deps_cache_hit.sh" \
  "${DEPS_ECR_IMAGE}" \
  "${ECR_IMAGE}" \
  "${{ inputs.image-tag }}"
```

Only write `deps-cache-hit=true` after the helper succeeds.

- [ ] **Step 5: Document cache-hit local availability**

Update the README to state that both exact-image hits and dependency-cache hits pull and tag `inputs.image-tag` locally before the action succeeds.

- [ ] **Step 6: Run the regression**

Run the Step 2 command.

Expected: PASS with the exact three-command Docker log.

- [ ] **Step 7: Commit the ECR fix**

```bash
git add \
  .github/actions/ecr-build-push-pull/action.yml \
  .github/actions/ecr-build-push-pull/README.md \
  .github/actions/ecr-build-push-pull/materialize_deps_cache_hit.sh \
  .github/actions/ecr-build-push-pull/test_materialize_deps_cache_hit.py
git commit -m "Materialize ECR dependency cache hits"
```

### Task 2: Verify the ECR deliverable

**Files:**

- Verify only; no planned modifications.

**Interfaces:**

- Consumes: Task 1.
- Produces: action-test evidence for the PR.

- [ ] **Step 1: Run all adjacent composite-action tests**

```bash
./isaaclab.sh -p -m pytest --ignore=tools/conftest.py \
  .github/actions/ecr-build-push-pull/test_materialize_deps_cache_hit.py \
  .github/actions/run-package-tests/test_cleanup_docker_storage.py -v
```

Expected: PASS.

- [ ] **Step 2: Validate shell syntax and executable mode**

```bash
bash -n .github/actions/ecr-build-push-pull/materialize_deps_cache_hit.sh
test -x .github/actions/ecr-build-push-pull/materialize_deps_cache_hit.sh
```

Expected: both commands succeed.
