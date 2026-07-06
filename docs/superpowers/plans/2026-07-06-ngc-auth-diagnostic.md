<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# NGC Authentication Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a temporary pull-request workflow that proves which authentication source lets CI reach NVCR containers, ECR-cached containers, and NGC artifacts.

**Architecture:** Create one standalone workflow containing an ambient-auth job and a same-repository secret-backed job. Each job isolates client configuration, suppresses command output that could contain credentials, and publishes only boolean outcomes to the GitHub job summary. Existing production workflows and actions remain unchanged.

**Tech Stack:** GitHub Actions YAML, Bash, Docker CLI, AWS CLI, NGC CLI 4.20.0, repository pre-commit hooks.

---

### Task 1: Add the diagnostic workflow

**Files:**
- Create: `.github/workflows/ngc-auth-diagnostic.yaml`
- Modify: `docs/superpowers/plans/2026-07-06-ngc-auth-diagnostic.md`

- [x] **Step 1: Verify the workflow does not exist**

Run:

```bash
test -f .github/workflows/ngc-auth-diagnostic.yaml
```

Expected: exit status 1 because the diagnostic workflow has not been created.

- [x] **Step 2: Create the workflow skeleton**

Create `.github/workflows/ngc-auth-diagnostic.yaml` with the 2026 SPDX header,
the name `NGC Authentication Diagnostic`, and only this trigger:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [develop]
```

Grant only `contents: read`, use pull-request concurrency, and define two jobs
on `[self-hosted, gpu]`, each with a 30-minute timeout:

```yaml
jobs:
  ambient-auth:
    name: Ambient runner authentication
    if: github.event.pull_request.head.repo.full_name == github.repository
  secret-backed-auth:
    name: Explicit NGC secret authentication
    if: github.event.pull_request.head.repo.full_name == github.repository
```

Neither job may use `pull_request_target`. Guard both jobs to same-repository pull
requests so fork-authored code never runs on persistent self-hosted runners. Only
`secret-backed-auth` may refer to `${{ secrets.NGC_API_KEY }}`.

- [x] **Step 3: Implement the ambient probe**

Checkout the repository, load `isaacsim_image_name`, `isaacsim_image_tag`, and
`ovphysx_wheelhouse_resource` from `.github/workflows/config.yaml` with a
quote-aware `sed` parser and no `yq` dependency, then run one
`bash` probe with `set -uo pipefail` and temporary directories cleaned by a
`trap`. Check that `mktemp` returns a non-empty path, install the cleanup trap
immediately, and fail cleanly if temporary subdirectory creation fails. The probe
must collect these booleans without printing configuration or captured command
output:

```text
Timeout tool is available
Python 3 is available
Docker CLI is available
AWS CLI is available
NGC CLI download, checksum, and unpack are ready
default Docker config exists
default Docker config names an nvcr.io auth/helper entry
default Docker config can inspect the configured Isaac Sim image
empty isolated Docker config can inspect the configured Isaac Sim image
AWS caller identity is available
ECR cache URL is available from ECR_CACHE_URL or runner SSM
authenticated ECR cache-base manifest is inspectable
isolated NGC CLI without an API key can download the configured resource
```

Resolve the ECR URL with the same precedence used by
`.github/actions/ecr-build-push-pull/action.yml`: `ECR_CACHE_URL`, then EC2 IMDS
plus `/github-runner/<instance-id>/ecr-cache-url`. Authenticate to ECR in an
isolated Docker config using `aws ecr get-login-password` before inspecting
`${ECR_URL}:cache-base`.

Install NGC CLI 4.20.0 from the public NVIDIA URL and verify SHA-256
`5cf084c88998c58ad8abf7849d2d1b41d578423886eb03018df10194e341d35b`.
Run the ambient artifact probe with `NGC_API_KEY` and `NGC_CLI_API_KEY` unset,
an isolated `HOME`/`NGC_CLI_HOME`, org `nvidian`, and team `no-team`.

Bound Docker, AWS, unzip, and NGC commands with `timeout`; bound every `curl`
with connection and total timeouts; and close NGC CLI stdin with `/dev/null`.
Report `timeout` readiness, separate ECR login success from manifest inspection,
and record attempted and timed-out booleans for NVCR, ECR, and NGC operations.
Append a Markdown table of the booleans to `$GITHUB_STEP_SUMMARY`. Probe failures
must remain visible as `false` but must not stop later probes.

- [x] **Step 4: Implement the secret-backed probe**

Set only this job environment mapping:

```yaml
env:
  NGC_API_KEY: ${{ secrets.NGC_API_KEY }}
```

Fail immediately with a clear error if the secret is empty. In isolated Docker
configuration, authenticate using:

```bash
printf '%s' "$NGC_API_KEY" |
  timeout 60s docker login --username '$oauthtoken' --password-stdin nvcr.io
```

Suppress login and inspect output, inspect the configured Isaac Sim image, then
use the verified NGC CLI with `NGC_CLI_API_KEY="$NGC_API_KEY"` and isolated
NGC home and closed stdin to download the configured OVPhysX resource. Bound
all external client operations as in the ambient probe. Append a Markdown table
containing only these booleans:

```text
Timeout tool is available
Docker CLI is available
AWS CLI is available
NGC CLI download, checksum, and unpack are ready
NGC_API_KEY is present
NVCR login attempted, timed out, and accepted
NVCR image inspection attempted, timed out, and succeeded
NGC artifact download attempted, timed out, and succeeded
```

Exit nonzero when either secret-backed service probe fails so a trusted PR
cannot look green when a service check fails. The error must direct readers to
readiness and timeout rows before attributing the failure to credentials.

- [x] **Step 5: Validate the workflow locally**

Run:

```bash
test -f .github/workflows/ngc-auth-diagnostic.yaml
./isaaclab.sh -f
```

Expected: the file-existence check exits 0 and every pre-commit hook passes.
If pre-commit modifies files, review and stage them, then run `./isaaclab.sh -f`
again.

Regression evidence: CI run `28798383155` failed in configuration loading with
exit `127` and `yq: command not found`. A structural RED check must show the old
workflow contains `yq`; the GREEN check must show none and verify that the
dependency-free parser preserves the full quoted resource value after its colon.

- [x] **Step 6: Review and commit**

Confirm `git diff --check` passes, verify no production workflow/action changed,
and inspect the workflow for secret output, `set -x`, `pull_request_target`, and
unguarded secret usage. Force-add this ignored plan file, stage the workflow,
and commit:

```bash
git add .github/workflows/ngc-auth-diagnostic.yaml
git add -f docs/superpowers/plans/2026-07-06-ngc-auth-diagnostic.md
git commit -m "Add NGC authentication diagnostic"
```

Do not push or open the PR; the coordinating agent performs those steps after
independent review.
