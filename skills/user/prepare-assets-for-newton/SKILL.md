---
name: isaaclab-preparing-assets-for-newton
description: Migrates PhysX-compatible robot, object, and scene assets to Isaac Lab's Newton backend with MJWarp. Use when handling multi-physics conversion, mass and inertia warnings, collision or topology differences, mimic joints, actuator and armature calibration, velocity-limit semantics, MJWarp solver presets, reset validity, or task-level PhysX/MJWarp parity.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Prepare Assets For Newton With MJWarp

## When To Use

Read the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) before editing an asset. Read the [MJWarp solver page](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst) before tuning a preset. This skill routes the work; the documentation owns parameter semantics and starting values.

Use `isaaclab-transferring-policies-sim-to-sim` only after the task resets and steps correctly in both backends.

## Workflow

1. **Freeze the PhysX baseline.** Record the source asset and revision, conversion command, ordered bodies and joints, actions and observations, `dt`, decimation, resets, warnings, and zero-action behavior.
2. **Check MJWarp support and scope.** Keep shared public configs unchanged when a task-local converted asset avoids breaking other tasks or checkpoints.
3. **Convert or layer the asset.** Prefer the URDF/MJCF converters with asset transformation and multi-physics conversion enabled. Keep common USD Physics properties in base schema cfgs and route backend-only fields through `Mujoco*`, `Newton*`, or `Physx*PropertiesCfg`.
4. **Validate the mechanical model.** Fix mass, center of mass, inertia and frames, collision geometry, materials, scale, articulation roots, fixed joints, axes, limits, gravity flags, self-collision, and references. Treat placeholder inertia or fallback collision as model failures.
5. **Rebuild control deliberately.** Resolve actuator and controller names after conversion. Source effort and rated speed from hardware data, then calibrate stiffness, damping, friction, and armature. Drive a coupled mechanism's leader, keep its follower passive, and reset both from one sample.
6. **Apply MJWarp-specific rules.** MJWarp enforces neither `velocity_limit` nor `velocity_limit_sim`; check required speed bounds in task or control logic. Use armature only for articulated coordinates, not plain rigid objects. Increase it only from physical data or controlled response tests, then retune damping to avoid bang-bang control.
7. **Create explicit backend presets.** Preserve `dt * decimation`. Start from the nearest documented MJWarp profile instead of translating PhysX numbers. Size contact and constraint capacity before changing convergence, friction, or collision settings.
8. **Validate the exact task.** Run zero and random agents under `physics=physx` and `physics=newton_mjwarp` through multiple resets. Reject penetrations and invalid coupled states before stepping. Reproduce the first divergence with fixed state and commands, change one parameter family at a time, and record the result.

## Validation

Require:

- intentional finite mass properties and collision geometry;
- matching topology, resolved names, timing, and policy interface, or an explicit retraining boundary;
- finite reset, zero-action, small-action, impulse, and contact behavior in both backends;
- physically plausible actuator response without unexplained velocity spikes or bang-bang control;
- sufficient MJWarp capacity with no unresolved importer or solver warning; and
- a migration record containing sources, commands, presets, evidence, and remaining differences.

## Maintenance

Keep this skill synchronized with the asset migration guide and MJWarp solver page. Update those pages first when guidance changes.

## References

- Read [reference.md](reference.md) for the compact audit tables, smoke commands, solver profiles, and failure triage.
- Read [examples.md](examples.md) when the symptom matches a worked migration case.
- Use [evaluations.md](evaluations.md) when maintaining or testing this skill.
- Use the [schema cfg guide](../../../docs/source/overview/core-concepts/schema_cfgs.rst) to confirm per-solver property routing.
- Use Newton's [Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html) for current diagnose-first guidance.
