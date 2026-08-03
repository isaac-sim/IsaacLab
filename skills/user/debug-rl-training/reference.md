# RL Debugging Reference

## Contents

- Experiment discipline
- Reward audit
- Checkpoint compatibility
- State traces
- Decision rules

## Experiment Discipline

Record these before a run:

- Task name and environment config.
- RL library and agent config.
- Backend, renderer, device, seed, number of environments, and iteration count.
- Checkpoint path if resuming.
- Single variable being tested.
- Primary success metric and stop condition.

One-iteration training proves only runner plumbing. Use short deterministic rollout or task metrics before judging behavior.

## Reward Audit

Reward is a training signal, not a success metric. For every reward term, record:

- The task phase it is intended to teach.
- The state variables it reads.
- The scale and units.
- Whether it saturates before real success.
- Whether it shares geometry with success termination and evaluation metrics.

Common reward failures:

- Reward increases while success stays flat.
- Dense shaping reward can be maximized without completing the task.
- Success reward, termination, and metric use different thresholds.
- The policy cannot observe the state needed to optimize the reward.
- Stateful reward buffers are not reset for the correct `env_ids`.

## Checkpoint Compatibility

Before replaying or resuming:

- Compare current observation and action dimensions with the checkpoint's training config.
- Confirm task ID and agent config match the run.
- Confirm the backend and sensor presets are compatible.
- Treat shape mismatches as environment-contract changes, not runner bugs.

## State Traces

When reward and behavior disagree, inspect task state directly. Useful trace fields include:

- Raw and processed policy actions.
- Observation term names and shapes.
- Per-term reward values.
- Termination and truncation flags.
- Robot root, joint, and controlled-frame pose.
- Object pose, velocity, goal error, and contact state.
- Reset state for robot, object, and goal.

## Decision Rules

- Reward rises and success rises: continue or scale.
- Reward rises and success stays flat: inspect reward saturation, observations, and success geometry.
- Reward is flat from the start: check action interface, target reachability, reset state, and observation coverage.
- Entropy collapses early: inspect exploration settings and overly strong penalties.
- Physics warnings appear: fix assets, contacts, or buffers before interpreting RL curves.
