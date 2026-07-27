<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Rendering CI Flake Fixes Design

## Objective

Stop three independent rendering CI failure modes without changing the rendering jobs'
`continue-on-error` policy:

1. deterministic Franka cloth/soft Newton-Warp golden-image failures;
2. intermittent Shadow Hand hangs within a multi-backend pytest process;
3. Docker startup failures after an ECR dependency-cache hit.

## Newton deformable rendering

The Newton/Warp upgrade in commit `a625084538` changed the deformable render output.
The introducing PR failed the same fourteen cloth/soft Newton-Warp cases now seen on
unrelated PRs, while the committed goldens remained unchanged.

Add a focused helper that restores visibility for viewport-visible procedural
colliders that Newton hides when a model also contains visual-only shapes. Invoke it
after every relevant USD import path. Cover the shape-selection behavior with unit
tests, including dynamic and static shapes and geometry that must remain hidden.

Refresh the sixteen Newton-renderer cloth/soft golden images from the previously
validated output produced with this visibility correction. Keep the existing pixel
and SSIM thresholds unchanged.

## Shadow Hand process isolation

Replace the single standard Shadow Hand rendering module with three modules, one per
backend/renderer slice:

- PhysX with Isaac RTX;
- Newton with Isaac RTX;
- PhysX with Newton Warp.

Each module launches one simulator process and contains only its own AOV parameters,
so the test no longer switches physics or renderer backends inside a long-lived Kit
process. The workflow continues to run the same complete parameter matrix.

Install a standard-library `faulthandler.dump_traceback_later` watchdog in these
modules. A healthy slice finishes before the watchdog fires; a stalled slice emits
all Python thread stacks before its hard timeout. Give only these files one
fresh-process retry after a hard timeout. Do not restore global timeout retries.

## ECR cache-hit image availability

On a dependency-cache hit, the ECR action currently creates the commit tag in the
registry but can leave the runner without the local short image tag consumed by
`docker run`.

After creating the registry alias, explicitly pull the commit-tagged ECR image and
tag it as the requested local image name. Add a regression test for the action script
that proves the cache-hit path performs both operations in order.

## Error handling and observability

- Image comparisons remain hard failures; deterministic mismatches are never retried.
- Shadow Hand retries apply only to hard timeouts without a completed JUnit report.
- Timeout watchdogs write stack traces to captured stderr and have no third-party
  dependency.
- Docker cache-hit failures remain fatal, but logs identify pull or local-tag
  failures directly.

## Validation

- Run Newton clone/visibility unit tests.
- Run test-runner unit tests covering per-file timeout retries.
- Run CI-action regression tests covering the ECR cache-hit local tag.
- Verify the old behavior fails each new regression before applying its fix.
- Run the available focused rendering tests if the local simulator environment
  supports the current Newton/OV packages; otherwise validate the exact image
  artifacts with the repository comparison functions and rely on the PR GPU jobs for
  simulator execution.
- Run `./isaaclab.sh -f` twice if the first run modifies files.

## Scope

The PR will include the Newton visibility fix, refreshed deformable goldens, Shadow
Hand test isolation and diagnostics, the ECR cache-hit fix, focused tests, and
required changelog fragments. It will not change `continue-on-error`, image
thresholds, renderer dependency pins, or unrelated rendering tests.
