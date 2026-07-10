---
name: isaaclab-debugging-rl-training
description: Diagnoses Isaac Lab reinforcement learning behavior, rewards, metrics, checkpoints, and training experiments. Use when reward curves look wrong, policies fail despite training, checkpoints mismatch, or RL changes need focused ablations.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Debugging RL Training

## When To Use

Use this skill when a user needs to debug learned behavior, reward hacking, checkpoint compatibility, unstable training, or training-result trustworthiness.

Do not use this skill for first-time training commands. Use `isaaclab-training-rl-agents` for launch commands and agent config wiring.

## Workflow

1. Identify task name, workflow type, RL library, agent config, seed, backend, and exact launch command.
2. Confirm the environment contract: action space, observation space, reward terms, termination terms, reset logic, and success metric.
3. Run the smallest reproduction: import, reset/step, one-iteration training, or deterministic playback depending on where the failure appears.
4. Change one variable per training experiment. Mark multi-variable runs as exploratory.
5. Compare reward curves against task metrics. Reward increases are not proof that the task behavior improved.
6. For reward issues, map every reward term to a named task phase and check that success reward, termination, and evaluation metric use consistent geometry.
7. For checkpoint issues, compare current observation/action dimensions with the saved training configuration before editing policy code.
8. For contact-rich tasks, collect state traces for controlled-frame pose, object pose, contacts, gripper state, per-term rewards, and termination flags.
9. Select checkpoints by task metrics, rollout behavior, and stability, not reward alone.

## Validation

Use this checklist:

1. The exact command and failing symptom are recorded.
2. The failed layer is classified as environment, reward, reset, physics, runner, or checkpoint compatibility.
3. A focused reproduction isolates one variable.
4. Reward terms and task metrics are inspected together.
5. A deterministic rollout or state trace confirms the behavior change.
6. Any recommended next run changes only one variable.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `skills/user/train-rl-agents/`, `docs/source/overview/reinforcement-learning/training_guide.rst`, the uv-based `train` and `play` entry points, and task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If recurring reward or checkpoint guidance belongs in user docs, update `docs/source/` first.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [RL training skill](../train-rl-agents/SKILL.md)
- [RL training guide](../../../docs/source/overview/reinforcement-learning/training_guide.rst)
- [Task examples](../../../source/isaaclab_tasks/isaaclab_tasks)
