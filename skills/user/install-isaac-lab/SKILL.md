---
name: isaaclab-installing-isaac-lab
description: Routes fresh Isaac Lab installations to the correct install page (pip, binary, source, kit-less, or Docker) for the Isaac Lab checkout and target platform (Linux x86_64, Linux aarch64, or Windows 11). Use when installing Isaac Lab for the first time, picking between install combinations, or asking for install commands for a specific platform.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Installing Isaac Lab

## When To Use

Use this skill when a user wants to install Isaac Lab from scratch and either has not yet picked an install method or wants the exact commands from the current docs for the method they picked.

This skill operates on the currently-checked-out Isaac Lab ref. If the user wants to install a different ref (a specific branch or tag), have them check that ref out first, then invoke the skill.

Do not use this skill for post-install issues. Use `isaaclab-setup-troubleshooting` for import failures, launch failures, verification failures, and other diagnostic questions after the install completed.

Do not vendor install commands, version pins, or troubleshooting steps into this skill. The install pages under `docs/source/setup/installation/index.rst` and their siblings are the source of truth.

## Workflow

1. Identify the target platform: Linux x86_64, Linux aarch64, or Windows 11.
2. Identify the intended install method. Read `docs/source/setup/installation/index.rst` from the Isaac Lab checkout and route by user context:
   - First-time full-feature user on a supported Linux distro or Windows 11: pip installation with uv.
   - Older Linux distro (GLIBC below the pip Isaac Sim minimum): binary Isaac Sim installation.
   - Isaac Sim contributor: source build.
   - External Isaac Lab extension author who only needs the Isaac Lab pip package: pip-only installation.
   - Newton physics only, no Isaac Sim required: kit-less installation.
   - Containerized deployment: Docker.
   - Experimental zero-env workflow: `uv run` path.
3. Confirm platform-specific prerequisites before prescribing commands: NVIDIA driver, GLIBC on Linux, Python 3.12, disk headroom, and any method-specific extras (Visual Studio Build Tools for Windows source builds; Docker Engine, Docker Compose, and NVIDIA Container Toolkit for the Docker path).
4. Read the install page from the checkout and prescribe its commands verbatim. Do not paraphrase, reorder, or substitute steps.
5. Prefer the docs-declared Recommended path for new users unless the user has a documented reason to pick another combination.
6. Run the docs-defined minimal verification command for the chosen method before running examples, training, or rendering workflows.

## Validation

Use this checklist:

1. Confirm the user is on a supported OS documented in `docs/source/setup/installation/index.rst`.
2. Confirm the NVIDIA driver meets or exceeds the documented minimum for the target platform in `docs/source/setup/installation/index.rst`.
3. On Linux, confirm GLIBC meets the pip Isaac Sim minimum before recommending a pip path. If it does not, route to a binary or kit-less path instead.
4. Ensure the Python environment matches the checkout's required Python version for the current Isaac Sim series.
5. Run the docs-defined minimal verification command before running examples, training, or rendering:

```bash
uv run python -c "import isaaclab; print('ok')"
```

6. If the verification command fails, hand off to `isaaclab-setup-troubleshooting`.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the following install docs. If commands or version pins change in the docs, update the docs, not this skill:

- `docs/source/setup/installation/index.rst` — comparison table, driver minimums, OS support matrix.
- `docs/source/setup/installation/pip_installation.rst` — recommended path.
- `docs/source/setup/installation/isaaclab_pip_installation.rst` — Isaac Lab as a pip package.
- `docs/source/setup/installation/binaries_installation.rst` — binary Isaac Sim.
- `docs/source/setup/installation/source_installation.rst` — source Isaac Sim build.
- `docs/source/setup/installation/kitless_installation.rst` — kit-less path.
- `docs/source/setup/installation/uv_run.rst` — experimental uv-run path.
- `docs/source/deployment/docker.rst` — Docker.
- `docs/source/refs/troubleshooting.rst` — hand-off target for post-install diagnostics.

This skill is a router, not a copy of the install pages. Adding install methods, changing version pins, or updating command sequences belongs in the docs above, not in this file.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Examples](examples.md)
- Installation index: `docs/source/setup/installation/index.rst`
- Pip installation (docs-Recommended): `docs/source/setup/installation/pip_installation.rst`
- Isaac Lab pip installation: `docs/source/setup/installation/isaaclab_pip_installation.rst`
- Binary installation: `docs/source/setup/installation/binaries_installation.rst`
- Source installation: `docs/source/setup/installation/source_installation.rst`
- Kit-less installation: `docs/source/setup/installation/kitless_installation.rst`
- `uv run` experimental: `docs/source/setup/installation/uv_run.rst`
- Docker deployment: `docs/source/deployment/docker.rst`
- Cross-skill hand-off for post-install issues: `isaaclab-setup-troubleshooting`.
