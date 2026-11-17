---
name: isaaclab-converting-direct-to-manager
description: Converts validated Isaac Lab direct workflow environments into manager-based task configurations. Use when a direct Isaac Lab environment or Isaac Gym migration already resets, steps, and trains, and the user wants reusable observations, rewards, commands, events, curricula, or terminations.
license: Apache-2.0
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
  tags: [environments, manager-based, refactoring]
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Converting Direct Environments To Manager-Based

## When To Use

Use this skill when a direct Isaac Lab environment is already runnable and the next goal is a reusable manager-based task. This is a follow-up workflow, not the first parity pass for Isaac Gym migration.

Do not use this skill when the direct environment still fails to construct, reset, step, or start training. Fix the direct baseline first with `isaaclab-building-environments` or `isaaclab-migrating-from-isaac-gym`.

## Workflow

1. Keep the direct environment as a parity baseline until the manager-based task passes equivalent smoke tests.
2. Read the task workflow overview, manager-based RL tutorial, and the closest direct/manager pair before proposing code.
   If a new package scaffold is needed, follow the [environment-building skill](../isaaclab-building-environments/SKILL.md#scaffold-a-new-task) and select **Manager-based | single-agent** in `uv run isaaclab --new`. For an existing project, add the manager variant within its layout; generate any reference scaffold in a fresh directory to preserve the direct baseline.
3. Split scene construction into an `InteractiveSceneCfg` subclass with assets, terrain, sensors, and lights.
4. Move action application into action configs such as joint position, velocity, or effort action terms.
5. Move observation assembly into `ObservationGroupCfg` and `ObservationTermCfg` entries, preserving block order and scaling unless the user accepts a policy interface change.
6. Move reward helpers into standalone MDP functions or `ManagerTermBase` classes, then wire them with `RewardTermCfg` weights and params.
7. Move reset logic and randomization into `EventTermCfg` entries with the correct mode (`prestartup`, `startup`, `reset`, or `interval`).
8. Move done logic into `TerminationTermCfg` entries and preserve timeout versus failure semantics.
9. Move command sampling into command configs when goals, velocities, poses, or targets are part of the observation/reward loop.
10. Keep backend-specific physics, sensor, and schema variants in `PresetCfg` classes rather than runtime conditionals.
11. Register the manager-based task under the suffixless task name when it is intended as the canonical task, and keep direct variants with a `-Direct` suffix.
12. Validate import, reset, random actions, and a short training run before removing or de-emphasizing the direct baseline.

## Examples

Representative requests this skill handles (worked patterns in [examples.md](examples.md)):

- "My direct Ant task runs now. Convert it to the manager workflow so rewards and observations are reusable."
- "My direct environment still has shape errors, but I want it manager-based."

## Validation

Use the same gates as the direct baseline:

1. Import the task module.
2. Instantiate a small number of environments.
3. Reset and step with random actions.
4. Compare observation shape, reward signs, termination rates, and reset behavior against the direct baseline.
5. Run a short training command after random-agent validation passes.

For skill changes, run the repository skill validator; the exact `check` command is in the [Skills README](../../README.md#authoring-rules).

## Maintenance

Keep this skill synchronized with [task workflows guide](../../../docs/source/concepts/task_workflows.rst), [manager-based RL environment guide](../../../docs/source/how-to/create_manager_rl_env.rst), direct/manager paired examples under the core `isaaclab_tasks` task packages, and shared MDP terms in each task pack's mdp package.

## References

- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Environment building skill](../isaaclab-building-environments/SKILL.md)
- [Isaac Gym migration skill](../isaaclab-migrating-from-isaac-gym/SKILL.md)
- [Task workflows](../../../docs/source/concepts/task_workflows.rst)
- [Create manager-based RL environment tutorial](../../../docs/source/how-to/create_manager_rl_env.rst)
- [Register Gym environment tutorial](../../../docs/source/how-to/register_rl_env_gym.rst)
- [Ant direct environment](../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_direct_env.py)
- [Ant direct config](../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_direct_env_cfg.py)
- [Ant manager config](../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py)
