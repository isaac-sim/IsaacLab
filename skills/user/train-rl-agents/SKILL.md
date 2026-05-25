---
name: isaaclab-training-rl-agents
description: Configures and runs Isaac Lab reinforcement learning workflows. Use when selecting RL frameworks, wiring agent configs, launching training, resuming runs, or troubleshooting training setup for Isaac Lab tasks.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Training RL Agents

## When To Use

Use this skill when a user wants to train, resume, evaluate, or configure reinforcement learning for an Isaac Lab task.

Do not use this skill to design environment observations, rewards, or resets from scratch. Use `isaaclab-building-environments` (`skills/user/create-environments/`) for environment construction first.

## Workflow

1. Identify the registered task name, workflow type, action space, observation space, target backend, device, and desired RL framework.
2. Read the RL training guide and the training tutorial before writing commands or configs.
3. Start from an existing agent config under `source/isaaclab_tasks/isaaclab_tasks/` that matches the framework and task family.
4. Keep framework-specific config formats separate. Do not mix RSL-RL Python configs with RL-Games, SKRL, or SB3 YAML/config files.
5. Use `./isaaclab.sh -p` entry points and documented scripts rather than ad hoc Python invocations.
6. Run a small smoke training job before scaling environment count, horizon, network size, or logging integrations.
7. For visual observations, confirm the sensor pipeline and renderer requirements before enabling large environment counts.
8. For multi-backend tasks, validate training on one backend before introducing backend presets.
9. Record the exact task, framework, backend, seed, and config overrides needed to reproduce the result.

## Validation

Use this checklist:

1. Confirm the task can reset and step without the training runner.
2. Confirm the selected agent config belongs to the intended RL framework.
3. Run a short training command with a small number of environments.
4. Resume or load a checkpoint only after the initial run writes expected artifacts.
5. Check logs for device, observation shape, action shape, and backend errors.

For skill changes, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/overview/reinforcement-learning/training_guide.rst`, RL training tutorials under `docs/source/tutorials/03_envs/`, and agent configs under `source/isaaclab_tasks/isaaclab_tasks/`. If framework commands or config formats change, update the official training docs or maintained examples first.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [RL training guide](../../../docs/source/overview/reinforcement-learning/training_guide.rst)
- [Configure RL training tutorial](../../../docs/source/tutorials/03_envs/configuring_rl_training.rst)
- [Run RL training tutorial](../../../docs/source/tutorials/03_envs/run_rl_training.rst)
- [Task examples](../../../source/isaaclab_tasks/isaaclab_tasks)
