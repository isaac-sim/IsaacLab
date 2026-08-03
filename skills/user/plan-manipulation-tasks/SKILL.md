---
name: isaaclab-planning-manipulation-tasks
description: Plans Isaac Lab manipulation tasks through phase gates for reaching, grasping, lifting, placing, insertion, and contact-rich workflows. Use when building or debugging manipulation environments where scene setup, reset geometry, action contracts, rewards, and validation must be staged.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Planning Manipulation Tasks

## When To Use

Use this skill when a user is creating, migrating, or debugging manipulation tasks such as reach, grasp, lift, place, insertion, or tool-use environments.

Do not use this skill as a replacement for environment construction details. Pair it with `isaaclab-building-environments`, `isaaclab-debugging-rl-training`, and `isaaclab-using-sensors-actuators` as needed.

## Workflow

1. Define the task objective as measurable phases: approach, align, pre-contact, contact or grasp, transport or lift, and final goal.
2. Build the minimum scene needed for the first phase: robot, support surfaces, task object, goal, lights, and required sensors.
3. Validate asset physics before training: collision geometry, mass, inertia, joint limits, friction, and contact materials.
4. Validate reset geometry before stepping: robot pose, object pose, goal pose, reachability, and absence of interpenetration.
5. Prove the action contract with scripted or zero-action probes before interpreting PPO results.
6. Add observations that expose the state required by the current phase, especially controlled-frame pose for end-effector or contact tasks.
7. Add rewards for one phase at a time. Keep success reward, termination, and metric geometry consistent.
8. Run a training smoke only after scene, reset, action, observation, and reward checks pass.
9. Validate behavior using deterministic rollout state metrics, not just total reward.

## Validation

Use these gates before calling a manipulation task ready:

1. Environment launches with the intended backend and task ID.
2. Scene assets spawn with usable collision and support geometry.
3. Reset state is physically valid and reachable.
4. Action dimensions and controlled joints or bodies match the agent config.
5. Observation terms expose the task frames needed by rewards and actions.
6. Reward, termination, and metric geometry agree.
7. A deterministic rollout or scripted probe satisfies the current phase.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with manipulation examples such as `source/isaaclab_tasks/isaaclab_tasks/core/lift/` and `source/isaaclab_tasks/isaaclab_tasks/contrib/stack/`, environment authoring docs, and RL debugging guidance. Put project-specific history and experiment logs in the project, not in this public skill.

## References

- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Create environments skill](../create-environments/SKILL.md)
- [Debug RL training skill](../debug-rl-training/SKILL.md)
- [Use sensors and actuators skill](../use-sensors-actuators/SKILL.md)
- [Manipulation task example (lift)](../../../source/isaaclab_tasks/isaaclab_tasks/core/lift)
