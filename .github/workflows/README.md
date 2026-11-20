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

## Kitless OVStage rendering coverage

The full kitless OVStage rendering suite (`-k ovstage` across
`test_rendering_kitless_newton_ovrtx.py` and
`test_rendering_kitless_ovphysx_ovrtx.py`) runs in four places:

- **Post-merge (`build.yaml`)**: runs on pushes to protected branches.
- **Nightly (`nightly-ovstage.yml`)**: runs once daily against the default
  branch independently of code merges and Isaac Sim image updates, reusing the
  ECR CI image, Warp kernel cache, OVRTX shader cache, and `run-package-tests`
  action without launching the rest of `Docker + Tests`. It also supports manual
  runs via `workflow_dispatch`:

  ```bash
  gh workflow run nightly-ovstage.yml --ref main
  ```

- **Isaac Sim image updates (`nightly-isaacsim-image.yml`)**: comments
  `run-ci ovstage` when opening or refreshing the rolling image-bump PR.
- **Other pull requests (`run-docker-ci.yml`)**: opt in per run by commenting
  `run-ci ovstage`.
