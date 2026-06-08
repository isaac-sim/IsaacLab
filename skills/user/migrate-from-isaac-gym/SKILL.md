---
name: isaaclab-migrating-from-isaac-gym
description: Migrates Isaac Gym tasks, assets, vectorized environments, and training workflows to Isaac Lab. Use when porting Isaac Gym environments, legacy Gym tasks, or Isaac Gym reinforcement learning code to Isaac Lab.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Migrating From Isaac Gym

## When To Use

Use this skill when a user wants to port an Isaac Gym task, asset workflow, observation/reward implementation, or training setup into Isaac Lab.

Do not use this skill for Isaac Lab 2.x to 3.x migration. Use the `isaaclab-migrating-2x-to-3x` skill for that workflow.

## Workflow

1. Identify the Isaac Gym task structure: assets, environment state tensors, observations, rewards, resets, and training runner.
2. Read the IsaacGymEnvs migration guide and direct workflow docs before proposing edits.
3. Migrate to a direct workflow first by default. This preserves the single-class structure that most Isaac Gym tasks already use.
4. Choose the initial backend target. Start with PhysX when matching Isaac Gym behavior; add Newton only after the direct PhysX migration is validated or if the user explicitly targets Newton.
5. Map Isaac Gym PhysX parameters through the schema cfg docs: first to Isaac Lab PhysX cfgs, then to backend-portable base cfgs or Newton/MuJoCo cfgs where an equivalent exists.
6. Map assets to Isaac Lab asset configs and scene entities.
7. Move action application, observation assembly, reward computation, termination checks, and reset logic into a `DirectRLEnv` or `DirectMARLEnv` implementation.
8. Port training configuration to the selected Isaac Lab reinforcement learning workflow.
9. Run a small smoke test before scaling training.
10. Only convert to a manager-based environment if the user asks for modular managers or the task benefits from reusable observation, reward, command, curriculum, or event terms.
11. Iterate through the validation loop until the environment resets, steps, and trains without shape or device errors.

## Validation

Use this feedback loop:

```bash
./isaaclab.sh -p -m pytest PATH_TO_MIGRATION_TEST
```

For manual smoke testing, run the smallest training or random-action entry point available for the migrated task.

For skill changes, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/migration/migrating_from_isaacgymenvs.rst`, `docs/source/overview/core-concepts/task_workflows.rst`, `docs/source/overview/core-concepts/multi_backend_architecture.rst`, `docs/source/overview/core-concepts/schema_cfgs.rst`, the direct environment tutorial, and direct task examples such as `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/` and `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/humanoid/`. If the migration requires documentation-level details, update `docs/source/` or the maintained examples first and keep this skill as a workflow router.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Rough locomotion validation](validation-rough-locomotion.md)
- [Evaluations](evaluations.md)
- [IsaacGymEnvs migration guide](../../../docs/source/migration/migrating_from_isaacgymenvs.rst)
- [Task workflows](../../../docs/source/overview/core-concepts/task_workflows.rst)
- [Multi-backend architecture](../../../docs/source/overview/core-concepts/multi_backend_architecture.rst)
- [Schema cfgs](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Environments overview](../../../docs/source/overview/environments.rst)
- [Create direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Create manager-based environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_rl_env.rst)
