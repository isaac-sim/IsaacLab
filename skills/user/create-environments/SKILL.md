---
name: isaaclab-building-environments
description: Builds complete Isaac Lab environments from task requirements, with manager-based environments preferred and direct environments reserved for migrations, custom control flow, or performance-specialized prototypes. Use when creating or registering a new task, choosing between direct and manager-based workflows, or adapting an entire maintained example into a new environment. Do not use for targeted sensor, sensor-derived observation or reward, contact-history, foot-contact, air-time, or actuator changes in an existing task; use isaaclab-using-sensors-actuators.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Building Environments

## When To Use

Use this skill when a user wants to create or register a complete Isaac Lab environment, choose the right task workflow, or adapt an entire existing example into a new task.

For a targeted change to an existing task's sensors, actuators, sensor-derived observations, rewards, or terminations, use `isaaclab-using-sensors-actuators` instead. A request to add foot contacts, contact history, touchdown timing, or air-time rewards is a sensor workflow even when it also changes an observation or reward config.

Default to manager-based environments for new Isaac Lab tasks because reusable scene, action, observation, reward, command, event, curriculum, and termination terms are the main benefit of the Isaac Lab task framework. Do not use this skill for migrating Isaac Gym tasks. Use the `isaaclab-migrating-from-isaac-gym` skill when the source is an Isaac Gym or IsaacGymEnvs task. For contact-rich manipulation task staging, pair this with `isaaclab-planning-manipulation-tasks`.

## Workflow

1. Identify the task type, assets, action space, observation needs, rewards, resets, termination conditions, sensors, training framework, and target backend.
2. Read the task workflow overview and the relevant manager-based tutorial before proposing code.
3. Choose manager-based workflow first for new Isaac Lab tasks. Express the task as scene, action, observation, reward, command, event, curriculum, and termination configs using existing MDP terms where possible.
4. Choose direct workflow only when the task is an Isaac Gym migration, needs bespoke step/reset/control flow that does not fit managers, requires monolithic parity with a source task, or is intentionally a short-lived performance prototype.
5. When a request mentions custom commands or rewards, try to make them manager terms first. Ask one clarifying question only when the behavior could be either reusable task logic or low-level control flow.
6. Start from the closest maintained source example under `source/isaaclab_tasks/isaaclab_tasks/`.
7. Define the scene and asset configs before adding rewards or training configuration.
8. Add observations, actions, rewards, resets, and terminations incrementally.
9. Register the environment and connect the smallest compatible agent config.
10. Use suffixless task names in smoke-test and training commands.
11. Run a random-action or short training smoke test before scaling environment count.
12. If a direct prototype becomes reusable, route the user to `isaaclab-converting-direct-to-manager` and move stable logic into shared MDP terms.

## Validation

Check the environment in this order:

1. Import the task module without launching a long training job.
2. Instantiate a small number of environments.
3. Reset and step with random actions.
4. Verify action, observation, reward, reset, and termination shapes.
5. Run a short training command only after the smoke test passes.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/overview/core-concepts/task_workflows.rst`, the environment tutorials under `docs/source/tutorials/03_envs/`, and maintained task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If workflow documentation is missing or stale, update the docs or examples first and keep this skill focused on choosing the right path.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Manipulation planning skill](../plan-manipulation-tasks/SKILL.md)
- [Task workflows](../../../docs/source/overview/core-concepts/task_workflows.rst)
- [Create manager-based base environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_base_env.rst)
- [Create manager-based RL environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_rl_env.rst)
- [Register Gym environment tutorial](../../../docs/source/tutorials/03_envs/register_rl_env_gym.rst)
- [Direct to manager conversion skill](../convert-direct-to-manager/SKILL.md)
- [Create direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Modify direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/modify_direct_rl_env.rst)
