# CI Workflows

`schedule:` triggers run only from the repository's current default branch.
A scheduled workflow YAML must live on that branch for its cron to register.
A `workflow_dispatch:` workflow must also exist on the default branch,
although manual dispatch can select another branch or tag. `pull_request:`
and `push:` triggers use the event branch normally.

## Installation coverage

Installation checks use the environments already prepared by Docker + Tests,
ARM CI, Windows CI, and Kit-less Docker Image. Build PIP Wheel builds the
distribution and dry-runs dependency resolution; it does not install or execute
the wheel. There is no separate Installation Tests workflow.

This removes three GPU-runner jobs and their repeated fresh environments and
wheel builds. It deliberately drops automatic validation of fresh wheel installs,
packaged resources at runtime, minimal workspace installs, and unlocked workspace
dependency resolution. Passing workspace tests does not guarantee those paths work.

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
