# Environment Templates

Rather than maintaining duplicate templates here, refer to the real examples in the
codebase — they are always up to date.

## Manager-Based Examples

Look at existing task environments for working templates:

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/
```

Each environment directory contains:
- `__init__.py` — Gymnasium registration (`gym.register()`)
- `*_env_cfg.py` — Full environment config (scene, observations, actions, rewards, terminations, events)
- `agents/` — Per-framework RL configs (YAML files for SB3, SKRL, RSL-RL, RL-Games)

## Direct Examples

```
source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/
```

## Creating From Template

Use the built-in template generator:
```bash
./isaaclab.sh -n
```

## Gymnasium Registration Pattern

See any task `__init__.py` for the registration pattern, e.g.:
```bash
grep -A 10 'gym.register' source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py
```
