---
name: isaaclab-debugging-rl-training
description: Investigates post-launch RL pathologies in Isaac Lab tasks. Use when reward curves are pathological, policies collapse or fail to converge, reward hacking masks task failure, checkpoints are incompatible after config changes, or training stability is in question. Focuses on root cause isolation, failure mode classification, and controlled ablations — not initial job provisioning or framework setup.
license: BSD-3-Clause
metadata:
  author: Isaac Lab Team <Isaac-Lab@exchange.nvidia.com>
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Debugging RL Training

## When To Use

Use this skill when a training job is already running (or has run) and the results show a pathology: reward hacking that masks task failure, policy collapse, reward-metric divergence, checkpoint weight shape mismatches after config changes, or unexplained behavioral regression between runs.

Do not use this skill for first-time job provisioning or framework selection. Use `isaaclab-training-rl-agents` for initial launch commands, agent config wiring, and training setup.

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

Keep this skill synchronized with `skills/user/isaaclab-training-rl-agents/`, `docs/source/concepts/reinforcement_learning.rst`, the uv-based `train` and `play` entry points, and task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If recurring reward or checkpoint guidance belongs in user docs, update `docs/source/` first.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [RL training skill](../isaaclab-training-rl-agents/SKILL.md)
- [RL training guide](../../../docs/source/concepts/reinforcement_learning.rst)
- [Task examples](../../../source/isaaclab_tasks/isaaclab_tasks)
