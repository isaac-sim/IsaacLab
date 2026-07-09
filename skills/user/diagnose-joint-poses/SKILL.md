---
name: isaaclab-diagnosing-joint-poses
description: Diagnoses and validates Isaac Lab robot initial joint poses from semantic or visual requests. Use when a robot starts with the wrong wrist, gripper, tool, camera, or end-effector orientation and the fix should be measured from articulation state instead of guessed from screenshots.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Diagnosing Joint Poses

## When To Use

Use this skill when a user asks to set, fix, or validate an initial robot pose from language such as "gripper fingers down", "tool points forward", "camera looks at the object", or "end effector aligns with the target".

## Workflow

1. Translate the visual request into a measurable body-axis or target-pose condition.
2. Inspect the actual articulation `body_names` and `joint_names`; do not rely only on USD prim names.
3. Measure current body position and orientation from simulation state.
4. Sweep candidate joints one variable at a time, respecting joint limits.
5. Choose the pose that satisfies the axis target while preserving reachability and avoiding self-collision.
6. Patch the task's initial joint positions or reset configuration.
7. Validate with compile checks and a one-env reset or zero-action smoke.
8. Report the measured before/after axis alignment, not just visual appearance.

## Validation

Use these acceptance checks:

1. The body or frame being measured is named explicitly.
2. The requested local axis and target world axis are documented.
3. Body-axis dot product is reported before and after the change.
4. Joint values are inside joint limits.
5. The pose remains reachable for the task target.
6. Observation and action dimensions remain unchanged unless the user requested a contract change.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with robot and articulation APIs under `source/isaaclab/isaaclab/assets/`, environment smoke-test patterns, and examples that use `ArticulationCfg.InitialStateCfg`. Do not include generated screenshots, private assets, or local machine paths in the skill.

## References

- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Articulation assets](../../../source/isaaclab/isaaclab/assets/articulation)
- [Asset configuration APIs](../../../source/isaaclab/isaaclab/assets/asset_base_cfg.py)
- [Zero-agent script](../../../scripts/environments/zero_agent.py)
