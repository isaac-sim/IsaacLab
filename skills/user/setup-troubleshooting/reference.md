# Setup Troubleshooting Reference

## Contents

- Install path routing
- Minimal verification
- Common failure routing
- Escalation checklist

## Install Path Routing

Ask which install path the user is following before prescribing commands:

| User context | First reference |
| --- | --- |
| Source checkout | `docs/source/setup/installation/source_installation.rst` |
| Pip package | `docs/source/setup/installation/pip_installation.rst` |
| Isaac Lab pip package | `docs/source/setup/installation/isaaclab_pip_installation.rst` |
| Binary package | `docs/source/setup/installation/binaries_installation.rst` |
| Cloud setup | `docs/source/setup/installation/cloud_installation.rst` |
| Kit-less setup | `docs/source/setup/installation/kitless_installation.rst` |
| Newton setup | `docs/source/overview/core-concepts/physical-backends/newton/installation.rst` |
| PhysX setup | `docs/source/overview/core-concepts/physical-backends/physx/installation.rst` |

## Minimal Verification

Use the smallest command that exercises the failing layer:

```bash
./isaaclab.sh -p -c "import isaaclab; print('ok')"
```

For task import and stepping:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 4
```

For training entry points:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless --max_iterations 1
```

## Common Failure Routing

| Symptom | First check |
| --- | --- |
| Import fails | Active Python environment and wrapper usage |
| `isaaclab_tasks` import fails | Run through `./isaaclab.sh -p`, then re-run `./isaaclab.sh -i` if needed |
| App launch fails | Isaac Sim, display, driver, and launcher docs |
| Task registration fails | Gym registration and task package import |
| Backend preset fails | `scripts/environments/list_envs.py --show_presets` |
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
