# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register.
A `workflow_dispatch:` workflow must also exist on the default branch,
although manual dispatch can select another branch or tag. `pull_request:`
and `push:` triggers use the event branch normally.

## Installation coverage

Installation checks use the environments already prepared by Docker + Tests,
ARM CI, Windows CI, and Kit-less Docker Image. Build PIP Wheel checks workspace
lock consistency and unlocked dependency resolution, dry-runs wheel extras
resolution (including Isaac Sim on Windows and Linux aarch64), and installs the
built base wheel into a fresh environment outside the checkout. It reuses the
installation suite's import, CLI, and packaged-resource
probe on a Linux CPU runner. There is no separate Installation Tests workflow.

This removes three GPU-runner jobs and their repeated fresh environments and
wheel builds, at the cost of one base-wheel install in the existing CPU wheel job.
It deliberately drops automatic validation of fresh ARM/Windows wheel installs,
fresh Isaac Sim wheel installs, minimal workspace installs, and training from an
installed wheel. Existing architecture jobs retain runtime and training coverage.

The isolated installation suite remains available for packaging and release work:

```bash
uv run --no-project --with pytest python tools/run_install_ci.py docker --build-wheel --gpu
```

For native runs and narrower selections, see the
[installation test instructions](../../source/isaaclab/test/install_ci/README.md).
The architecture jobs still run the Cartpole probes from that directory in their
prepared environments, so keep the suite out of the general test orchestrator.

When retiring the workflow, remove `Installation Tests (x86)`,
`Installation Tests (ARM)`, and `Installation Tests (Windows)` from any required
status-check rules that reference them.

## Multi-GPU training smoke routing

The multi-GPU pytest workflow matches runtime package directories and package
manifests for its training smoke jobs. Core and backend test-only or documentation
changes still run the applicable unit-test lane, but do not start the unrelated
Cartpole training smoke lanes. Changes to the smoke tests themselves, their
conftest hooks, runtime test helpers, training entry points, dependencies, and
shared actions retain training coverage.

For a core/backend unit-test-only change, this avoids three multi-GPU runner
allocations: the kit-less image builder and both training smoke jobs. The Kit
image builder and unit-test job remain. This reduces total runner work without
serializing independent tests or dropping any training configuration.

## Test-only Docker PRs

Docker + Tests selects owning suites when every changed executable file is a
recognized package test. Core test edits run the three core shards; backend test
edits run their backend suite. Mixed test edits take the union. Shared fixtures,
runtime changes, unknown paths, incomplete file lists, and API errors retain the
full suite. Required job names remain present and unrelated jobs report skipped.
This selection changes neither the cases within a suite nor post-merge coverage.

## Compatibility cadence and runner reuse

Latest Released Isaac Sim Compatibility runs daily at 07:23 UTC and on manual
dispatch, rather than provisioning a second integration pipeline on every main
merge. The current runtime still receives post-merge validation. Released-runtime
regressions are detected on the next daily run; manual dispatch is available for
release-sensitive changes. Scheduled runs retain the post-merge job selection and
rendering subsets.

The kit-less multi-GPU training job builds or pulls its own image, eliminating a
separate builder runner for a single consumer. Both training smoke jobs restore
the shared compatible Warp kernel cache. ARM configuration loading shares the
change detector's hosted runner instead of provisioning another one.
