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
4. From the Isaac Lab checkout, use documented uv commands such as `uv run python`, `uv run isaaclab train`, and `uv run isaaclab play` for Python, verification, and RL entry points. XR teleoperation entry points are `uv run --extra teleop isaaclab teleop run|record|replay`; `teleop` cannot be combined with the `mimic` or `all` extras in one command.
5. Use suffixless task names in verification and training commands.
6. Ask for the smallest relevant error output when the failure mode is unclear.
7. Prefer a minimal verification command before running examples, training, or rendering workflows.
8. Route backend-specific setup to the unified installation guide and backend
   choice questions to the physics-backends concept.
9. For XR teleoperation setup, which is a separate workflow from the base installation, route to the CloudXR how-to rather than the installation guide.
10. If the docs are incomplete or stale, update the docs rather than expanding this skill.

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

Keep this skill synchronized with the unified installation guide, the physics-backends concept, the Docker/Cloud feature guide, quick installation docs, and `docs/source/refs/troubleshooting.rst`. Setup guidance changes often, so keep this skill as a router to official docs and minimal verification steps.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Quickstart](../../../docs/source/setup/quickstart.rst)
- [Installation](../../../docs/source/setup/installation/index.rst)
- [XR teleoperation setup](../../../docs/source/how-to/cloudxr_teleoperation.rst)
- [Docker/Cloud](../../../docs/source/features/docker_cloud.rst)
- [Physics backends](../../../docs/source/concepts/physics_backends.rst)
- [Troubleshooting](../../../docs/source/refs/troubleshooting.rst)
