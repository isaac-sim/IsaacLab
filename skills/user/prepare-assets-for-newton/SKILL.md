---
name: isaaclab-preparing-assets-for-newton
description: Validates and prepares PhysX-compatible USD assets for Isaac Lab Newton workflows. Use when an asset runs under PhysX but Newton reports missing mass or inertia, placeholder inertials, unsupported collision or joint topology, unstable control, or task-level action and actuator mismatches.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Preparing Assets For Newton

## When To Use

Use this skill when a user needs to validate or prepare a USD robot, object, or scene asset for Newton after it already works or partially works under PhysX.

Do not use this skill for choosing a backend at the environment level. Use `isaaclab-selecting-backends` for backend selection and `isaaclab-using-presets` for backend-specific configuration variants.

## Workflow

1. Establish a PhysX baseline for the same asset path and task spawn path.
2. Classify the asset as PhysX-compatible, Newton-runnable, or Newton-clean.
3. Audit authored physics metadata: rigid bodies, colliders, mass, inertia, center of mass, joint topology, and material properties.
4. Fix authored USD physics data instead of hiding Newton warnings with task code.
5. If runtime-resolved mass properties are needed, produce a local package or authored layer with explicit mass, diagonal inertia, and center of mass.
6. When fixing task-side schema overrides, import universal schema fragments from `isaaclab.sim.schemas` and Newton or MuJoCo-specific cfgs from `isaaclab_newton.sim.schemas` instead of relying on deprecated core forwarding imports.
7. Re-audit the converted asset under Newton.
8. Validate the asset inside the target Isaac Lab task, not only in a standalone USD viewer.
9. Check actuator joint patterns, controller body names, action dimensions, and zero-action rollout stability.
10. Record source path, converted path, audit verdicts, smoke command, and residual warnings in project documentation.

## Validation

An asset is Newton-clean only when:

1. All rigid bodies have intentional mass properties.
2. Runtime mass and inertia values are finite and positive.
3. Collision geometry is parseable by Newton.
4. Joint topology is accepted by Newton.
5. The target task can spawn and reset the asset under Newton.
6. Zero-action rollout has finite observations, rewards, positions, and velocities.
7. Actuator and controller names still resolve after any USD conversion.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with Newton backend documentation, asset conversion utilities, and backend-specific examples. Avoid storing converted USD packages, generated audit logs, or private asset paths in this skill.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Backend selection skill](../select-backends/SKILL.md)
- [Preset skill](../use-presets/SKILL.md)
- [Newton documentation](../../../docs/source/overview/core-concepts/physical-backends/newton)
