# Setup Troubleshooting Reference

## Contents

- Install path routing
- Minimal verification
- Common failure routing
- Escalation checklist

## Install Path Routing

Ask which install path the user is following before prescribing commands. For a new full-feature Isaac Sim setup, route to the automatic uv guide first.

| User context | First reference |
| --- | --- |
| Source checkout | `docs/source/setup/installation/index.rst` |
| uv-managed environment | `docs/source/setup/quickstart.rst` and `docs/source/setup/installation/index.rst` |
| Pip package | `docs/source/setup/installation/index.rst` |
| Isaac Lab pip package | `docs/source/setup/installation/index.rst` |
| Binary package | `docs/source/setup/installation/index.rst` |
| Cloud setup | `docs/source/features/docker_cloud.rst` |
| Legacy installer or setup without Isaac Sim | `docs/source/setup/installation/index.rst` |
| Newton setup | `docs/source/overview/core-concepts/physical-backends/newton/installation.rst` |
| PhysX setup | `docs/source/overview/core-concepts/physical-backends/physx/installation.rst` |

## Minimal Verification

Use the smallest command that exercises the failing layer:

```bash
uv run python -c "import isaaclab; print('ok')"
```

For task import and stepping:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole --num_envs 4
```

For training entry points:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --max_iterations 1
```

## Common Failure Routing

| Symptom | First check |
| --- | --- |
| Import fails | Active Python environment and Isaac Lab checkout |
| `isaaclab_tasks` import fails | Run through `uv run python` from the intended Isaac Lab checkout, then confirm the source packages or external task package are installed with the intended `uv pip` environment |
| App launch fails | Isaac Sim, display, driver, and launcher docs |
| Task registration fails | Gym registration and task package import |
| Backend preset fails | `uv run python scripts/environments/list_envs.py --show_presets` |
| Camera or renderer fails | Renderer selection and sensor docs |
| Training starts but shapes fail | Environment reset/step smoke test before runner |

## Escalation Checklist

Before suggesting broad reinstall steps:

1. Capture the exact command and traceback.
2. Confirm the install path and Python executable.
3. Run a minimal import check.
4. Run a small random-agent task check.
5. Check `docs/source/refs/troubleshooting.rst`.
6. Escalate only after the documented checks do not match the failure.
