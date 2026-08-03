---
name: isaaclab-setup-troubleshooting
description: Routes Isaac Lab installation, verification, and common troubleshooting issues to official docs and canonical commands. Use when installing Isaac Lab, verifying setup, debugging launch failures, or diagnosing environment problems.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Setup Troubleshooting

## When To Use

Use this skill when a user asks for help installing Isaac Lab, verifying a local setup, or diagnosing common setup and launch failures.

Do not duplicate installation or troubleshooting docs in this skill. The official docs are the source of truth.

## Workflow

1. Identify the install mode: automatic uv, legacy installer script, managed Python environment, Python package, downloaded Isaac Sim package, source build, Docker, cloud, or backend-specific setup. For a new full-feature Isaac Sim setup, prefer the automatic uv installation guide.
2. Identify OS, Python environment, GPU/driver context, Isaac Sim source, and target backend.
3. Read the matching installation guide and troubleshooting reference before prescribing commands.
4. From the Isaac Lab checkout, use documented uv commands such as `uv run python`, `uv run isaaclab train`, and `uv run isaaclab play` for Python, verification, and RL entry points.
5. Use suffixless task names in verification and training commands.
6. Ask for the smallest relevant error output when the failure mode is unclear.
7. Prefer a minimal verification command before running examples, training, or rendering workflows.
8. Route backend-specific setup to the relevant PhysX or Newton docs.
9. If the docs are incomplete or stale, update the docs rather than expanding this skill.

## Validation

Use this checklist:

1. Confirm the user is following one supported install path.
2. Confirm Python and package commands run from the intended Isaac Lab checkout and uv-managed environment when applicable.
3. Run a minimal import or verification command before larger tests.
4. Check troubleshooting docs for the observed error class.
5. Escalate to environment-specific debugging only after the documented checks are exhausted.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the unified installation guide, the Docker/Cloud feature guide, quick installation docs, backend installation docs, and `docs/source/refs/troubleshooting.rst`. Setup guidance changes often, so keep this skill as a router to official docs and minimal verification steps.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Quickstart](../../../docs/source/setup/quickstart.rst)
- [Installation](../../../docs/source/setup/installation/index.rst)
- [Docker/Cloud](../../../docs/source/features/docker_cloud.rst)
- [PhysX installation](../../../docs/source/overview/core-concepts/physical-backends/physx/installation.rst)
- [Newton installation](../../../docs/source/overview/core-concepts/physical-backends/newton/installation.rst)
- [Troubleshooting](../../../docs/source/refs/troubleshooting.rst)
