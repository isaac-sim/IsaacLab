---
name: isaaclab-setup-troubleshooting
description: Use when installing Isaac Lab, verifying setup, debugging launch failures, or diagnosing environment and simulation-performance problems.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Setup Troubleshooting

## When To Use

Use this skill when a user asks for help installing Isaac Lab, verifying a local setup, or diagnosing common setup, launch, and simulation-performance problems.

Do not duplicate installation or troubleshooting docs in this skill. The official docs are the source of truth.

## Workflow

1. Identify the install mode: automatic uv, legacy installer script, managed Python environment, Python package, downloaded Isaac Sim package, source build, Docker, cloud, or backend-specific setup. For a new full-feature Isaac Sim setup, prefer the automatic uv installation guide.
2. Identify OS, Python environment, GPU/driver context, Isaac Sim source, and target backend.
3. Read the matching installation guide and troubleshooting reference before prescribing commands.
4. From the Isaac Lab checkout, use documented uv commands such as `uv run python`, `uv run isaaclab train`, and `uv run isaaclab play` for Python, verification, and RL entry points. The `all` extra is the curated `ov`, `rl-games`, `sb3`, `skrl`, `rsl-rl`, `rerun`, and `viser` list; it excludes Isaac Sim and the standalone URDF/MJCF importers. Install the `importers` extra with the documented resolver overrides. XR teleoperation entry points are `uv run --extra teleop isaaclab teleop run|record|replay`; `teleop` cannot be combined with the `mimic` or `all` extras in one command.
5. Use suffixless task names in verification and training commands.
6. Ask for the smallest relevant error output when the failure mode is unclear.
7. When the failure looks environmental rather than code-level, or when a report has to be reproduced on another machine, ask for a bundle from `uv run --no-project python tools/capture_env.py capture`; it runs on an installation too broken to import Isaac Lab, and its `REPRODUCE.md` carries the steps to rebuild the environment locally. Tell the user to read `REPRODUCE.md` and `env/environment.txt` from the bundle before attaching it to a public issue: the capture records the hostname, the command they name, and allowlisted path variables. The bundle records the commit the checkout sits on and no remotes and no source diff, but it does copy `pyproject.toml`, `uv.lock`, `pyvenv.cfg` and the `.pth` files verbatim from the working tree, dirty or not.
8. Prefer a minimal verification command before running examples, training, or rendering workflows.
9. Route backend-specific setup to the unified installation guide and backend choice questions to the physics-backends concept.
10. For unexpectedly slow simulation or training, route to the performance section in the troubleshooting reference and profile a representative workload before prescribing tuning changes.
11. For XR teleoperation setup, which is a separate workflow from the base installation, route to the CloudXR how-to rather than the installation guide.
12. If the docs are incomplete or stale, update the docs rather than expanding this skill.

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

Keep this skill synchronized with the unified installation guide, the Docker/Cloud feature guide, quick installation docs, the physics-backends concept, and `docs/source/refs/troubleshooting.rst`. Setup guidance changes often, so keep this skill as a router to official docs and minimal verification steps.

## References

- [Evaluations](evaluations.md)
- [Reference](reference.md)
- [Quickstart](../../../docs/source/setup/quickstart.rst)
- [Installation](../../../docs/source/setup/installation/index.rst)
- [XR teleoperation setup](../../../docs/source/how-to/cloudxr_teleoperation.rst)
- [Docker/Cloud](../../../docs/source/features/docker_cloud.rst)
- [Physics backends](../../../docs/source/concepts/physics_backends.rst)
- [Troubleshooting](../../../docs/source/refs/troubleshooting.rst)
