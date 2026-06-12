---
name: isaaclab-building-environments
description: Builds Isaac Lab direct and manager-based environments from task requirements. Use when creating a new environment, choosing between direct and manager-based workflows, registering Gym environments, or adapting existing task examples.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Building Environments

## When To Use

Use this skill when a user wants to create a new Isaac Lab environment, choose the right task workflow, or adapt an existing task example.

Do not use this skill for migrating Isaac Gym tasks. Use the `isaaclab-migrating-from-isaac-gym` skill when the source is an Isaac Gym or IsaacGymEnvs task. For contact-rich manipulation task staging, pair this with `isaaclab-planning-manipulation-tasks`.

## Workflow

1. Identify the task type, assets, action space, observation needs, rewards, resets, termination conditions, sensors, training framework, and target backend.
2. Read the task workflow overview and the relevant direct or manager-based tutorial before proposing code.
3. Choose direct workflow when the task needs custom control flow, close parity with a monolithic source task, or rapid prototyping.
4. Choose manager-based workflow when the task benefits from reusable observation, reward, command, event, termination, or curriculum terms.
5. Start from the closest maintained source example under `source/isaaclab_tasks/isaaclab_tasks/`.
6. Define the scene and asset configs before adding rewards or training configuration.
7. Add observations, actions, rewards, resets, and terminations incrementally.
8. Register the environment and connect the smallest compatible agent config.
9. Use suffixless task names in smoke-test and training commands.
10. Run a random-action or short training smoke test before scaling environment count.
11. Move reusable logic into shared MDP terms only after the behavior is stable.

## Validation

Check the environment in this order:

1. Import the task module without launching a long training job.
2. Instantiate a small number of environments.
3. Reset and step with random actions.
4. Verify action, observation, reward, reset, and termination shapes.
5. Run a short training command only after the smoke test passes.

For skill changes, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/overview/core-concepts/task_workflows.rst`, the environment tutorials under `docs/source/tutorials/03_envs/`, and maintained task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If workflow documentation is missing or stale, update the docs or examples first and keep this skill focused on choosing the right path.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Manipulation planning skill](../plan-manipulation-tasks/SKILL.md)
- [Task workflows](../../../docs/source/overview/core-concepts/task_workflows.rst)
- [Create direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Modify direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/modify_direct_rl_env.rst)
- [Create manager-based base environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_base_env.rst)
- [Create manager-based RL environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_rl_env.rst)
- [Register Gym environment tutorial](../../../docs/source/tutorials/03_envs/register_rl_env_gym.rst)
