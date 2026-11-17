---
name: isaaclab-building-environments
description: Designs and registers complete Isaac Lab task architectures from scratch, including template-generated external projects and internal tasks, by composing scene assets, wiring MDP term managers, and configuring the environment lifecycle. Use when creating or registering a new task, choosing between direct and manager-based workflows, or adapting a maintained example into a new environment. For targeted sensor signal pipeline, actuator hardware abstraction, or sensor-derived term changes in an existing task, use isaaclab-using-sensors-actuators instead.
license: BSD-3-Clause
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
  tags: [environments, manager-based, direct-workflow, task-registration]
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Building Environments

## When To Use

Use this skill when a user wants to design a new task architecture: compose scene assets, wire MDP term managers, register the environment in Gym, or select between manager-based and direct workflows. This skill covers the full environment lifecycle from asset config through termination conditions.

Use manager-based environments by default for new single-agent tasks: custom commands and rewards usually fit reusable manager terms. Respect an explicitly requested workflow. Use direct environments for bespoke step/reset control, multi-agent tasks, or a migration that needs parity with a monolithic implementation.

For a targeted hardware change to an existing task — adding a sensor signal pipeline, configuring actuator dynamics, or wiring sensor-derived observations — use `isaaclab-using-sensors-actuators` instead. A request to add foot contacts, contact history, touchdown timing, or air-time rewards is a sensor signal pipeline change even when it also changes an observation or reward config.

Route Isaac Gym ports to `isaaclab-migrating-from-isaac-gym`, validated direct-to-manager conversions to `isaaclab-converting-direct-to-manager`, and targeted sensor, sensor-derived reward or observation, or actuator changes to `isaaclab-using-sensors-actuators`. For contact-rich manipulation, also read `isaaclab-planning-manipulation-tasks`.

Infer assets, actions, observations, rewards, resets, termination conditions, training framework, and backend from the request and existing code. Ask only for missing choices that materially affect implementation.

## Workflow

### Scaffold a new task

Read the [template generator guide](../../../docs/source/developer-tools/template_generator.rst) before creating package scaffolding. From the Isaac Lab checkout or an installed Isaac Lab uv project, run the interactive generator:

```bash
uv run isaaclab --new
```

`-n` is equivalent. Use a terminal for the prompts; do not invent non-interactive flags. Choose:

- **External** for a standalone project; supply a parent directory outside Isaac Lab, a Python-compatible project name, task family, and robot/config name. The generator appends the project name and initializes a Git repository with an initial commit. Leave the optional Isaac Sim UI extension off unless needed.
- **Internal** for a contribution under isaaclab tasks package; this option requires a source checkout and is unavailable in installed wheels.
- The chosen workflow and only the RL libraries/algorithms needed. Direct multi-agent tasks use the supported multi-agent choices shown by the prompts.

For an existing project, preserve its layout and registration. If a scaffold is useful, generate into a fresh location and integrate the needed pieces instead of generating over existing work.

External projects use a `src` layout: tasks live under `<project>/tasks/<family>/config/<robot>`, with shared family terms in an `mdp` package. The root project metadata provides an `isaaclab` tasks entry point for CLI discovery. Do not recreate the legacy extension layout or add registration callbacks to a normally installed generated project. Source-generated projects record editable checkout paths; preserve those paths or update the uv sources table in the project metadata if directories move.

From the generated project root, sync dependencies and run a random-action smoke test. Then use the generated list-envs script with `--show_presets` and the generated registration test to confirm discovery:

```bash
uv sync
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16
```

Use the task ID listed by the project, rather than guessing its spelling or suffix. Default dependencies provide Newton without Isaac Sim. For a PhysX target, use `uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx --num_envs 16`; keep the extra on subsequent commands that need it. See the generator guide for other backend extras.

For internal tasks, run commands from the Isaac Lab root and list tasks by running the [list envs script](../../../scripts/environments/list_envs.py) through `uv run python` with `--show_presets`.

### Implement the task

Use the generated Cartpole as a packaging and execution baseline, then adapt the closest maintained task under isaaclab tasks package; see [examples](examples.md) for starting points. Read the relevant workflow tutorial before changing environment interfaces.

Define the scene and assets, then implement actions, observations, resets, rewards, and terminations using existing MDP terms where appropriate. Add commands, curricula, and backend presets when the task requires them. Keep agent configuration consistent with the chosen RL library.

## Examples

Representative requests this skill handles (worked patterns in [examples.md](examples.md)):

- "I want to build a quadruped locomotion task with custom command sampling and custom rewards."
- "I created a new environment config and need to train it with Gym registration."

## Validation

Verify registration, instantiate a small environment batch, and exercise reset and random steps. Check action/observation shapes, finite rewards, reset behavior, and timeout versus failure semantics. Registration tests alone do not establish simulation correctness. Run a short training smoke test after these checks when training is in scope; report task behavior separately from runner execution.

For internal task registrations or preset changes, use `isaaclab-updating-environment-docs` to synchronize the environment browser. For external projects, maintain their generated README and project-local tests.

## Maintenance

Keep generator instructions aligned with [cli](../../../tools/template/cli.py), its templates, and the maintained generator guide. Validate skill edits with the repository skill validator; the exact `check` command is in the [Skills README](../../README.md#authoring-rules).

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Manipulation planning skill](../isaaclab-planning-manipulation-tasks/SKILL.md)
- [Task workflows](../../../docs/source/concepts/task_workflows.rst)
- [Create manager-based base environment tutorial](../../../docs/source/how-to/create_manager_base_env.rst)
- [Create manager-based RL environment tutorial](../../../docs/source/how-to/create_manager_rl_env.rst)
- [Register Gym environment tutorial](../../../docs/source/how-to/register_rl_env_gym.rst)
- [Direct to manager conversion skill](../isaaclab-converting-direct-to-manager/SKILL.md)
- [Create direct workflow environment tutorial](../../../docs/source/how-to/create_direct_rl_env.rst)
- [Modify direct workflow environment tutorial](../../../docs/source/how-to/modify_direct_rl_env.rst)
