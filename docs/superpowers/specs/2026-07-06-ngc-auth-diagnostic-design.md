<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# NGC Authentication Diagnostic Design

## Objective

Determine why fork pull-request CI can obtain the Isaac Sim container while it
cannot download the OVPhysX wheelhouse from NGC. The diagnostic must distinguish
anonymous registry access, persistent self-hosted-runner credentials, the AWS
ECR cache, and explicitly injected GitHub Actions secrets.

This is a temporary diagnostic only. It must not modify production build or test
behavior and its pull request must be marked `[DO NOT MERGE]`.

## Approach

Add one standalone pull-request workflow that runs on the same self-hosted GPU
runner class as the Docker and OVPhysX jobs. The workflow contains independent
ambient and secret-backed probes so a result from one authentication source
cannot mask another.

The ambient probe does not reference `secrets.NGC_API_KEY`. It will:

1. Report whether the runner has a Docker configuration, Docker credential
   helper, NGC CLI configuration, and AWS identity, without printing their
   contents.
2. Test the configured Isaac Sim NVCR image with the runner's default Docker
   configuration.
3. Test the same image with a new empty `DOCKER_CONFIG` to determine whether the
   image is anonymously readable.
4. Test the configured ECR cache path using the runner's AWS identity.
5. Attempt to inspect or download the configured OVPhysX NGC resource with an
   isolated NGC home and no API key.

The secret-backed probe is restricted to a pull request whose head repository is
`isaac-sim/IsaacLab`. It explicitly injects `secrets.NGC_API_KEY` and repeats the
NVCR and OVPhysX resource operations with isolated client configuration. This
confirms whether the same secret authorizes both NGC services.

## Safety

- Never print API keys, Docker auth records, configuration contents, or command
  lines containing credentials.
- Report only credential/configuration presence, registry/resource identifiers,
  exit status, and redacted success/failure summaries.
- Use temporary Docker and NGC configuration directories and remove them with
  `trap` cleanup.
- Restrict secret-backed execution to same-repository pull requests.
- Do not use `pull_request_target` and do not execute fork code with secrets.
- Keep diagnostic failures non-destructive while making each failed probe visible
  in the job summary.

## Expected Interpretation

- Default Docker succeeds and empty-config Docker fails: persistent runner Docker
  credentials provide NVCR access.
- Both Docker probes succeed: the Isaac Sim image is anonymously readable.
- ECR succeeds while both NVCR probes fail: the normal Docker path is satisfied
  by the runner-authenticated ECR cache rather than NGC.
- Ambient artifact access fails while secret-backed access succeeds: artifacts
  require the GitHub Actions secret and have no runner-side fallback.
- Both secret-backed probes succeed: one NGC API key can authenticate both NVCR
  and the artifact registry; the difference is credential delivery, not service
  capability.

## Verification and Deliverable

Validate workflow syntax and run all repository pre-commit hooks before pushing.
Push the branch to the upstream repository so the pull request is same-repository,
then open a draft PR named `[DO NOT MERGE] Probe NGC authentication paths`.
Interpret the resulting logs and job summary; do not convert the diagnostic into
a production fix in the same PR.
