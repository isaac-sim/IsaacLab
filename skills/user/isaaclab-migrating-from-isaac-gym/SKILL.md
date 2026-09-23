---
name: isaaclab-migrating-from-isaac-gym
description: Migrates Isaac Gym tasks, assets, vectorized environments, and training workflows to Isaac Lab. Use when porting Isaac Gym environments, legacy Gym tasks, or Isaac Gym reinforcement learning code to Isaac Lab.
license: BSD-3-Clause
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Migrating From Isaac Gym

## When To Use

Use this skill when a user wants to port an Isaac Gym task, asset workflow, observation/reward implementation, or training setup into Isaac Lab.

Do not use this skill for Isaac Lab 2.x to 3.x migration. Use the `isaaclab-migrating-2x-to-3x` skill for that workflow.
Port Isaac Gym code directly to the current Isaac Lab APIs; do not introduce an intermediate port to an older Isaac Lab release.

## Workflow

1. Identify the Isaac Gym task structure: assets, environment state tensors, observations, rewards, resets, and training runner.
2. If the user needs a new full-feature Isaac Sim setup, point them to the automatic uv installation guide first. If the user expects PhysX or Kit execution, run a runtime preflight from the Isaac Lab checkout before a long port: verify `uv run --extra isaacsim python` uses the intended Python environment and checkout, imports `isaacsim` and `omni`, and provides the requested RL library.
3. Read the IsaacGymEnvs migration guide and direct workflow docs before proposing edits. Treat the guide as a direct mapping to the current Isaac Lab APIs, including physics presets, `ProxyArray` access, XYZW quaternions, and explicit indexed or masked write methods.
4. For a new scratch or external migration project, follow the [template scaffolding workflow](../create-environments/SKILL.md#scaffold-a-new-task). Use `uv run isaaclab --new`, choose External, a parent path outside Isaac Lab, project/task-family/robot names, direct single-agent for ordinary Isaac Gym tasks, and the needed RL library. Preserve existing project layouts when migrating into an established package.
5. Migrate to a direct workflow first by default. This preserves the single-class structure that most Isaac Gym tasks already use.
6. Choose the initial backend target. Start with PhysX when matching Isaac Gym behavior; add Newton only after the direct PhysX migration is validated or if the user explicitly targets Newton.
7. Map Isaac Gym PhysX parameters through the schema cfg docs: first to Isaac Lab PhysX cfgs, then to backend-portable base cfgs or Newton/MuJoCo cfgs where an equivalent exists.
8. Map assets to Isaac Lab asset configs and scene entities.
9. Move action application, observation assembly, reward computation, termination checks, and reset logic into a `DirectRLEnv` or `DirectMARLEnv` implementation.
10. Port training configuration to the selected Isaac Lab reinforcement learning workflow.
11. Run a small smoke test before scaling training. Omit `--viz` for headless execution, or use `--viz none` only when a config or command would otherwise enable a visualizer.
12. For locomotion migrations, run the policy-success validation loop in [Reference](reference.md#policy-success-validation-loop). Validate a flat walking policy before rough-terrain curriculum training; rough terrain can start and still be unhealthy if episodes terminate immediately. If the legacy task's command range is broad, use a staged command curriculum or a simpler flat source config such as IsaacGymEnvs `Anymal.yaml` before claiming policy success.
13. After the direct migration resets, steps, and trains, recommend a manager-based follow-up when the task has reusable observation, reward, command, curriculum, termination, or event logic.
14. Use the `isaaclab-converting-direct-to-manager` skill for that follow-up instead of mixing manager conversion into the first parity pass.
15. Iterate through the validation loop until the environment resets, steps, trains, and reaches a task-appropriate policy metric. Do not claim policy success from a completed training command, checkpoint file, or improving scalar alone; parse training metrics and run a bounded checkpoint rollout.

## Validation

Use this feedback loop:

```bash
uv run --with pytest python -m pytest PATH_TO_MIGRATION_TEST
```

For manual smoke testing, run the smallest random-action entry point before training. For generated external projects, run `uv sync` from the project root and use the generated `isaaclab.tasks` entry point for CLI discovery; see [External Template Projects](reference.md#external-template-projects). Standalone validation scripts must import the task registration module before Gym lookup. When matching Isaac Gym PhysX behavior, retain `--extra isaacsim` and the `physics=isaacsim_physx` preset on simulation commands.

For policy validation, follow the policy-success loop in [Reference](reference.md#policy-success-validation-loop). The loop must import/register the migrated task, smoke-test reset and random steps, train to a useful budget, parse TensorBoard or equivalent scalars, evaluate the saved checkpoint in a bounded rollout, then adjust the migration and rerun the shortest affected gate until the policy succeeds or a concrete blocker is identified.

Runtime preflight for execution/training requests:

```bash
uv run --extra isaacsim python -c "import importlib.util, sys; print(sys.executable); print(sys.version); print(importlib.util.find_spec('isaacsim')); print(importlib.util.find_spec('omni'))"
```

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the Isaac Gym section in `docs/source/migration/migrating_to_isaaclab_3-0.rst`, `docs/source/setup/installation/index.rst`, `docs/source/concepts/task_workflows.rst`, `docs/source/concepts/backend_architecture.rst`, `docs/source/concepts/schema_cfgs.rst`, the direct environment tutorial, and direct task examples such as `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/`, `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/`, and `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/`. If the migration requires documentation-level details, update `docs/source/` or the maintained examples first and keep this skill as a workflow router.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Rough locomotion validation](validation-rough-locomotion.md)
- [Initial Ant smoke validation](validation-ant-fresh-agent.md)
- [Evaluations](evaluations.md)
- [Direct to manager conversion skill](../isaaclab-converting-direct-to-manager/SKILL.md)
- [Migration guide: Isaac Gym section](../../../docs/source/migration/migrating_to_isaaclab_3-0.rst#migration-from-isaac-gym-and-isaacgymenvs)
- [Task workflows](../../../docs/source/concepts/task_workflows.rst)
- [Backend architecture](../../../docs/source/concepts/backend_architecture.rst)
- [Schema cfgs](../../../docs/source/concepts/schema_cfgs.rst)
- [Environment browser](../../../docs/source/setup/environments.rst)
- [Create direct workflow environment tutorial](../../../docs/source/how-to/create_direct_rl_env.rst)
- [Create manager-based environment tutorial](../../../docs/source/how-to/create_manager_rl_env.rst)
