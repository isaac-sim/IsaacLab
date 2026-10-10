---
name: isaaclab-installing-isaac-lab
description: Installs Isaac Lab end-to-end with minimal user interaction. Auto-detects the system with read-only checks, picks the right install method (uv checkout, uv-managed environment, Isaac Sim source build, Isaac Lab wheel, or Docker) from the install docs, applies the documented China Asset Region Profile when requested, shows one consolidated plan, and after a single confirmation executes the docs-prescribed commands unattended through verification. Use when installing Isaac Lab for the first time, picking between install combinations, configuring China asset access during installation, or asking for install commands for a specific platform.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Installing Isaac Lab

## When To Use

Use this skill when a user wants to install Isaac Lab from scratch. The default mode is the express flow: auto-detect, auto-pick, one confirmation, then unattended execution. Users should not have to answer setup questions unless the system genuinely forces a choice.

This skill operates on the currently-checked-out Isaac Lab ref. If the user wants to install a different ref (a specific branch or tag), have them check that ref out first, then invoke the skill.

Do not use this skill for post-install issues. Use `isaaclab-setup-troubleshooting` for import failures, launch failures, verification failures, and other diagnostic questions after the install completed.

Do not vendor install commands, version pins, or troubleshooting steps into this skill. The install pages under `docs/source/setup/installation/index.rst` and their siblings are the source of truth for commands and minimums alike.

## Workflow

1. Run the read-only preflight in [reference.md](reference.md). Check the current checkout's
   installation documentation for platform, driver, disk, and native build prerequisites.
2. Use uv for every installation. Default to the source checkout's automatic environment.
   Honor an explicit environment directory through `UV_PROJECT_ENVIRONMENT`. Route wheel,
   source-build, container, and cloud workflows through the corresponding documented uv instructions.
3. Report unmet prerequisites before changing the environment. For existing installations,
   use `isaaclab-setup-troubleshooting` instead of reinstalling over them.
4. Read the selected installation section and use its current commands. For China storage,
   also apply the Asset Region Profile rules in [reference.md](reference.md).
5. Show the detected system, selected workflow, exact commands, and download requirements in
   one consolidated plan. Ask for confirmation before executing an installation unless the
   user has already authorized it.
6. Execute the approved commands and log output to `~/.isaaclab/logs/install-<timestamp>.log`.
   On failure, apply at most one documented fix and retry once; then hand off with the log.
7. Run the selected workflow's documented verification. Report the environment path, launch
   command, and log path; save the installation facts to `~/.isaaclab/install_profile.yaml`.

## Validation

- Commands and prerequisites match the current installation documentation.
- uv owns dependency resolution and environment creation.
- The selected workflow's verification succeeded before larger training jobs.
- China asset availability is checked against the current manifest when requested.
- Skill edits pass `uv run --no-project python tools/skills/cli.py check`.

## Maintenance

Keep this skill synchronized with the following install docs. If commands or version pins change in the docs, update the docs, not this skill:

- `docs/source/setup/installation/index.rst` — the installation entrypoint. Every install method is a section on this page with a stable ref anchor. Contains "System requirements" (driver minimums, GLIBC, Python, OS support), the method-picker cards, and the per-method command sequences.
- `docs/source/setup/installation/index.rst` — includes the automatic uv setup steps under `installation-method-uv`.
- `docs/source/setup/installation/index.rst` — `installation-method-uv` steps (Newton-only default without Isaac Sim).
- `docs/source/setup/installation/index.rst` — `installation-method-python-env` steps (managed uv sync with the isaacsim extra).
- `docs/source/setup/installation/index.rst` — `installation-method-wheel` steps (Isaac Lab Python package for external projects).
- `docs/source/setup/installation/index.rst` — `installation-method-source` steps (Isaac Sim source build).
- `docs/source/how-to/manage_asset_downloads.rst` — the `installation-asset-region-profiles` workflow.
- `docs/source/setup/installation/asset_caching_details.inc` — asset caching notes.
- `docs/source/workflows/docker/index.rst` — Docker and cloud-workstation deep dive; complements `installation-method-container` and `installation-method-cloud` in `index.rst`.
- `docs/source/refs/troubleshooting.rst` — hand-off target for post-install diagnostics.

This skill is a router and executor, not a copy of the install pages. Adding install methods, changing version pins, or updating command sequences belongs in the docs above, not in this file.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Examples](examples.md)
- Installation entrypoint: `docs/source/setup/installation/index.rst`
- Automatic setup with uv (docs-Recommended): section `installation-method-uv` in `index.rst`
- Python environment with Isaac Sim (uv sync): section `installation-method-python-env` in `index.rst`
- Isaac Lab Python package (external projects): section `installation-method-wheel` in `index.rst`
- Isaac Sim source build: section `installation-method-source` in `index.rst`
- Asset Region Profiles: section `installation-asset-region-profiles` in
  `docs/source/how-to/manage_asset_downloads.rst`
- Docker and HPC clusters: section `installation-method-container` in `index.rst`, deep-dive in `docs/source/workflows/docker/index.rst`
- Cloud workstations: section `installation-method-cloud` in `index.rst`
- Troubleshooting: `docs/source/refs/troubleshooting.rst`
- Cross-skill hand-off for post-install issues: `isaaclab-setup-troubleshooting`.
