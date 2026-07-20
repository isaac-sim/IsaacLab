---
name: isaaclab-installing-isaac-lab
description: Installs Isaac Lab end-to-end with minimal user interaction. Auto-detects the system with read-only checks, picks the right install method (pip, binary, source, kit-less, or Docker) from the install docs, shows one consolidated plan, and after a single confirmation executes the docs-prescribed commands unattended through verification. Use when installing Isaac Lab for the first time, picking between install combinations, or asking for install commands for a specific platform.
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
2. Read the comparison table and minimums (driver, GLIBC, Python, disk) from `docs/source/setup/installation/index.rst` in the checkout — never from memory — compare against the detected facts, and route using the mapping in [reference.md](reference.md).
3. If a hard blocker exists (no NVIDIA GPU or driver, driver below the documented minimum, insufficient disk), stop before any state-changing command. Report each blocker with its fix and the documented alternative (kit-less for no-RTX machines). Do not attempt driver installs unattended. If existing install artifacts were found, hand off to `isaaclab-setup-troubleshooting` instead of reinstalling over them.
4. Auto-pick the remaining choices, honoring stated preferences. A preference the user already stated (conda, Docker, kit-less, source build, a specific env name or directory) always wins and must not be re-asked. Otherwise pick without asking: the docs-Recommended method; uv if present, else conda if present, else the docs' uv install step; install into the current checkout with the docs-default env name.
5. Read the routed install page from the checkout and extract its commands verbatim for this platform. Do not paraphrase, reorder, or substitute steps.
6. Show one consolidated confirmation: detected system in two or three lines, chosen method and why, the exact commands in order, which steps need sudo, and rough download size. Ask one go/no-go question. This is the only question in the flow.
7. On yes, execute every step in order without further prompts, streaming output and appending everything to `~/.isaaclab/logs/install-<timestamp>.log`. Announce sudo steps as they run; the password prompt is expected, not a question. On the binary route only, pause at the manual Isaac Sim download step with the URL from the docs page and resume when the user confirms — the one unavoidable manual step.
8. On a step failure, check the failure routing table in [reference.md](reference.md), apply at most one documented fix, and retry the step once. If it still fails, stop and hand off to `isaaclab-setup-troubleshooting` with the log path.
9. Run the docs-defined minimal verification command for the chosen method, then hand over: how to activate the env, how to run a first demo from the docs quickstart, and the log file path. Save a short summary of facts, route, and commands run to `~/.isaaclab/install_profile.yaml` for reproducibility.

## Validation

Use this checklist:

1. Preflight detection ran before any state-changing command, and blockers were empty or resolved by the user.
2. The chosen route matches the comparison table in `docs/source/setup/installation/index.rst` for the detected platform, driver, and GLIBC — with minimums read from the checkout docs, not memory.
3. Any preference the user stated in their request was honored without re-asking.
4. Exactly one confirmation question was asked before execution (plus the binary-download pause when on the binary route).
5. Commands came verbatim from the routed install page in the checkout.
6. The docs-defined minimal verification command ran before any examples, training, or rendering:

```bash
uv run python -c "import isaaclab; print('ok')"
```

7. If verification fails, hand off to `isaaclab-setup-troubleshooting` with the install log path.

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

This skill is a router and executor, not a copy of the install pages. Adding install methods, changing version pins, or updating command sequences belongs in the docs above, not in this file.

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
