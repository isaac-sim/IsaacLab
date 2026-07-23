---
name: isaaclab-installing-isaac-lab
description: Installs Isaac Lab end-to-end with minimal user interaction. Auto-detects the system with read-only checks, picks the right install method (automatic uv, downloaded Isaac Sim, source build, Isaac Lab wheel, legacy isaaclab.sh, managed Python env, or Docker) from the install docs, shows one consolidated plan, and after a single confirmation executes the docs-prescribed commands unattended through verification. Use when installing Isaac Lab for the first time, picking between install combinations, or asking for install commands for a specific platform.
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

The express flow asks the user at most one question: the final go/no-go. Do not interview the user about use case, env manager, or install method unless a rule below explicitly says to ask.

1. Run the read-only preflight detection commands listed in [reference.md](reference.md) to gather OS, arch, GLIBC, GPU and driver, Python, env managers, RAM, disk, and existing install artifacts. Nothing in this step changes system state.
2. Read the "System requirements" section and the install-method comparison from `docs/source/setup/installation/index.rst` in the checkout — never from memory — compare against the detected facts, and route to the correct section anchor in `index.rst` using the mapping in [reference.md](reference.md).
3. If a hard blocker exists (no NVIDIA GPU or driver, driver below the documented minimum, insufficient disk), stop before any state-changing command. Report each blocker with its fix and the documented alternative (the legacy Newton-only installer at `installation-legacy-installer` for no-Isaac-Sim machines). Do not attempt driver installs unattended. If existing install artifacts were found, hand off to `isaaclab-setup-troubleshooting` instead of reinstalling over them.
4. Auto-pick the remaining choices, honoring stated preferences. A preference the user already stated (conda, Docker, source build, Newton-only, a specific env name or directory) always wins and must not be re-asked. Otherwise pick without asking: the docs-Recommended method (automatic uv from the checkout); uv if present, else conda if present, else the docs' uv install step; install into the current checkout with the docs-default env name.
5. Read the routed section of `index.rst` (and any `.inc` fragments it includes) from the checkout and extract its commands verbatim for this platform. Do not paraphrase, reorder, or substitute steps.
6. Show one consolidated confirmation: detected system in two or three lines, chosen method and why, the exact commands in order, which steps need sudo, and rough download size. Ask one go/no-go question. This is the only question in the flow.
7. On yes, execute every step in order without further prompts, streaming output and appending everything to `~/.isaaclab/logs/install-<timestamp>.log`. Announce sudo steps as they run; the password prompt is expected, not a question. On the "Downloaded Isaac Sim package" route only, pause at the manual Isaac Sim download step with the URL from the docs section and resume when the user confirms — the one unavoidable manual step.
8. On a step failure, check the failure routing table in [reference.md](reference.md), apply at most one documented fix, and retry the step once. If it still fails, stop and hand off to `isaaclab-setup-troubleshooting` with the log path.
9. Run the docs-defined minimal verification command for the chosen method, then hand over: how to activate the env, how to run a first demo from the docs quickstart, and the log file path. Save a short summary of facts, route, and commands run to `~/.isaaclab/install_profile.yaml` for reproducibility.

## Validation

Use this checklist:

1. Preflight detection ran before any state-changing command, and blockers were empty or resolved by the user.
2. The chosen route matches the comparison table in `docs/source/setup/installation/index.rst` for the detected platform, driver, and GLIBC — with minimums read from the checkout docs, not memory.
3. Any preference the user stated in their request was honored without re-asking.
4. Exactly one confirmation question was asked before execution (plus the binary-download pause when on the binary route).
5. Commands came verbatim from the routed install page in the checkout.
6. The docs-defined minimal verification command ran before any examples, training, or rendering. Use the command documented by the chosen install page — for binary installs, that's the bundled-Python verification on the binary page (not `uv run`); for Docker installs, run the documented verification inside the container.
7. If verification fails, hand off to `isaaclab-setup-troubleshooting` with the install log path.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the following install docs. If commands or version pins change in the docs, update the docs, not this skill:

- `docs/source/setup/installation/index.rst` — the installation entrypoint. Every install method is a section on this page with a stable ref anchor. Contains "System requirements" (driver minimums, GLIBC, Python, OS support), the method-picker cards, and the per-method command sequences.
- `docs/source/setup/installation/uv_run_details.inc` — steps included by `installation-method-uv`.
- `docs/source/setup/installation/legacy_installer_details.inc` — steps included by `installation-legacy-installer` (Newton-only default without Isaac Sim).
- `docs/source/setup/installation/pip_details.inc` — steps included by `installation-method-python-env` (managed venv/conda + pip Isaac Sim).
- `docs/source/setup/installation/wheel_details.inc` — steps included by `installation-method-wheel` (Isaac Lab Python package for external projects).
- `docs/source/setup/installation/binaries_details.inc` — steps included by `installation-method-binary` (downloaded Isaac Sim package).
- `docs/source/setup/installation/source_details.inc` — steps included by `installation-method-source` (Isaac Sim source build).
- `docs/source/setup/installation/asset_caching_details.inc` — asset caching notes.
- `docs/source/setup/installation/include/` — verification and shared helper snippets.
- `docs/source/features/docker_cloud.rst` — Docker and cloud-workstation deep dive; complements `installation-method-container` and `installation-method-cloud` in `index.rst`.
- `docs/source/refs/troubleshooting.rst` — hand-off target for post-install diagnostics.

This skill is a router and executor, not a copy of the install pages. Adding install methods, changing version pins, or updating command sequences belongs in the docs above, not in this file.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Examples](examples.md)
- Installation entrypoint: `docs/source/setup/installation/index.rst`
- Automatic setup with uv (docs-Recommended): section `installation-method-uv` in `index.rst`
- Legacy isaaclab.sh installer (Newton-only default): section `installation-legacy-installer` in `index.rst`
- Python environment with Isaac Sim (venv/conda + pip): section `installation-method-python-env` in `index.rst`
- Isaac Lab Python package (external projects): section `installation-method-wheel` in `index.rst`
- Downloaded Isaac Sim package (older distros): section `installation-method-binary` in `index.rst`
- Isaac Sim source build: section `installation-method-source` in `index.rst`
- Docker and HPC clusters: section `installation-method-container` in `index.rst`, deep-dive in `docs/source/features/docker_cloud.rst`
- Cloud workstations: section `installation-method-cloud` in `index.rst`
- Troubleshooting: `docs/source/refs/troubleshooting.rst`
- Cross-skill hand-off for post-install issues: `isaaclab-setup-troubleshooting`.
