---
name: isaaclab-preparing-assets-for-newton
description: Migrates PhysX-compatible robot, object, and scene assets to Isaac Lab's Newton backend with MJWarp. Use when handling multi-physics conversion, per-solver asset configuration, mass or inertia warnings, collision and topology differences, velocity-limit behavior, armature and damping, MJWarp solver profiles, or task-level reset and smoke validation.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Prepare Assets For Newton With MJWarp

## When To Use

Read the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) first. This skill follows that page in the same order and targets Newton with MJWarp, not Newton solvers generally. Use the sim-to-sim skill only after the asset and task run in both backends.

## Workflow

1. **Multi-backend Asset Importing Pipeline.** Convert URDF or MJCF assets with the updated importers. Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True` so the layered asset contains neutral, PhysX, and MuJoCo payloads. Account for the nested rigid-body structure produced by the new converter.
2. **Use per-solver asset configuration classes.** Put common USD Physics properties in solver-common base cfgs. Put MJWarp-specific fields in `Mujoco*PropertiesCfg`, Newton-native fields in `Newton*PropertiesCfg`, and PhysX-only fields in `Physx*PropertiesCfg`. Confirm support in the generated schema APIs.
3. **Audit the authored mechanical model.** Check every dynamic link and contact-relevant object for intentional mass, COM, inertia and frames, collision geometry, approximation and scale, materials, articulation root, fixed-base and fixed-joint representation, joint axes and limits, self-collision, and gravity overrides.
4. **Velocity limits distinction.** `velocity_limit` is a rated speed and `velocity_limit_sim` is a requested solver clamp. MJWarp enforces neither. Add task or control checks for required speed bounds and use per-joint `effort_limit_sim`.
5. **Why MJWarp often needs more armature.** Use reflected rotor inertia or controlled response tests for articulated coordinates. A plain rigid object has body inertia, not actuator armature. Correct mass, inertia, units, reset penetration, effort, action scale, control period, and contact capacity before changing armature.
6. **Retune damping with armature.** Increasing armature changes effective inertia and damping ratio. Tune armature, stiffness, and damping together from a step response, use conservative action scales, and keep targets away from hard stops to prevent bang-bang control.
7. **Choose an MJWarp starting profile.** Do not translate PhysX parameters numerically. Start from the nearest profile on the MJWarp solver page, keep the documented convergence defaults initially, enable `debug_mode`, and use MuJoCo contacts unless the task requires Newton's collision pipeline. Then run zero and random agents in PhysX and MJWarp through multiple resets. Check non-finite state, first-step impulses, saturation, excessive angular velocity, contact loss, and warnings. Reject penetrations, impossible mimic states, and invalid randomized geometry before stepping.

## Validation

Require a multi-physics asset with supported per-solver fields, intentional mass and collision data, physically justified actuator parameters, finite task behavior in both backends, and valid resets. Do not classify an asset as ready while importer or solver warnings remain unexplained.

## Maintenance

Keep this skill synchronized section-for-section with the asset migration guide and use the MJWarp solver page for current profile values.

## References

- [Compact reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [MJWarp solver page](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst)
- [Schema configuration classes](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html)
