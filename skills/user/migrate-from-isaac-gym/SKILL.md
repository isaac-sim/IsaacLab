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
2. If the user needs a new full-feature Isaac Sim setup, point them to the pip/uv installation docs first. If the user expects execution or training, run a runtime preflight from the Isaac Lab checkout before a long port: verify `uv run python` uses the intended Python environment and checkout, imports `isaacsim` and `omni`, and provides the requested RL library.
3. Read the IsaacGymEnvs migration guide and direct workflow docs before proposing edits.
4. For a scratch or external migration project, start from the Isaac Lab template generator instead of hand-rolling package scaffolding. From the Isaac Lab checkout, use `uv run isaaclab -n`, choose an external project, choose the scratch path, choose the direct single-agent workflow for Isaac Gym style tasks, and select the needed RL library such as `rsl_rl`.
5. Migrate to a direct workflow first by default. This preserves the single-class structure that most Isaac Gym tasks already use.
6. Choose the initial backend target. Start with PhysX when matching Isaac Gym behavior; add Newton only after the direct PhysX migration is validated or if the user explicitly targets Newton.
7. Map Isaac Gym PhysX parameters through the schema cfg docs: first to Isaac Lab PhysX cfgs, then to backend-portable base cfgs or Newton/MuJoCo cfgs where an equivalent exists.
8. Map assets to Isaac Lab asset configs and scene entities.
9. Move action application, observation assembly, reward computation, termination checks, and reset logic into a `DirectRLEnv` or `DirectMARLEnv` implementation.
10. Port training configuration to the selected Isaac Lab reinforcement learning workflow.
11. Run a small smoke test before scaling training. Do not use deprecated `--headless` examples; omit `--viz` for headless execution, or use `--viz none` only when a config or command would otherwise enable a visualizer.
12. For locomotion migrations, run the policy-success validation loop in [Reference](reference.md#policy-success-validation-loop). Validate a flat walking policy before rough-terrain curriculum training; rough terrain can start and still be unhealthy if episodes terminate immediately. If the legacy task's command range is broad, use a staged command curriculum or a simpler flat source config such as IsaacGymEnvs `Anymal.yaml` before claiming policy success.
13. After the direct migration resets, steps, and trains, recommend a manager-based follow-up when the task has reusable observation, reward, command, curriculum, termination, or event logic.
14. Use the `isaaclab-converting-direct-to-manager` skill for that follow-up instead of mixing manager conversion into the first parity pass.
15. Iterate through the validation loop until the environment resets, steps, trains, and reaches a task-appropriate policy metric. Do not claim policy success from a completed training command, checkpoint file, or improving scalar alone; parse training metrics and run a bounded checkpoint rollout.

## Validation

Use this feedback loop:

```bash
uv run --with pytest python -m pytest PATH_TO_MIGRATION_TEST
```

For manual smoke testing, run the smallest random-action entry point available for the migrated task before training. For external scratch work, prefer a template-generated external project and install its extension in editable mode, or put the generated project extension and every package under the Isaac Lab checkout's `source/` directory at the front of `PYTHONPATH`; this avoids accidentally importing `isaaclab_tasks` or extension packages from another checkout or installed wheel. Ensure the task package is imported before Gym lookup; use a small wrapper for scripts without `--external_callback`, and use the callback option when a training script exposes one.

For policy validation, follow the policy-success loop in [Reference](reference.md#policy-success-validation-loop). The loop must import/register the migrated task, smoke-test reset and random steps, train to a useful budget, parse TensorBoard or equivalent scalars, evaluate the saved checkpoint in a bounded rollout, then adjust the migration and rerun the shortest affected gate until the policy succeeds or a concrete blocker is identified.

Runtime preflight for execution/training requests:

```bash
uv run python -c "import importlib.util, sys; print(sys.executable); print(sys.version); print(importlib.util.find_spec('isaacsim')); print(importlib.util.find_spec('omni'))"
```

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/migration/migrating_from_isaacgymenvs.rst`, `docs/source/setup/installation/pip_installation.rst`, `docs/source/overview/core-concepts/task_workflows.rst`, `docs/source/overview/core-concepts/multi_backend_architecture.rst`, `docs/source/overview/core-concepts/schema_cfgs.rst`, the direct environment tutorial, and direct task examples such as `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/`, `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/`, and `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/`. If the migration requires documentation-level details, update `docs/source/` or the maintained examples first and keep this skill as a workflow router.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Rough locomotion validation](validation-rough-locomotion.md)
- [Initial Ant smoke validation](validation-ant-fresh-agent.md)
- [Evaluations](evaluations.md)
- [Direct to manager conversion skill](../convert-direct-to-manager/SKILL.md)
- [IsaacGymEnvs migration guide](../../../docs/source/migration/migrating_from_isaacgymenvs.rst)
- [Task workflows](../../../docs/source/overview/core-concepts/task_workflows.rst)
- [Multi-backend architecture](../../../docs/source/overview/core-concepts/multi_backend_architecture.rst)
- [Schema cfgs](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Environments overview](../../../docs/source/overview/environments.rst)
- [Create direct workflow environment tutorial](../../../docs/source/tutorials/03_envs/create_direct_rl_env.rst)
- [Create manager-based environment tutorial](../../../docs/source/tutorials/03_envs/create_manager_rl_env.rst)
