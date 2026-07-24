---
name: isaaclab-preparing-assets-for-newton
description: Migrates PhysX-compatible robot, object, and scene assets to Isaac Lab's Newton backend with MJWarp. Use when handling existing Isaac Lab assets or new multi-physics conversion, per-solver asset configuration, mass or inertia warnings, contact slip and friction, velocity-limit behavior, armature and damping, MJWarp solver profiles, solver-specific failures, or task-level validation.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Prepare Assets For Newton With MJWarp

## When To Use

Read the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) first. This skill follows that page in the same order and targets Newton with MJWarp, not Newton solvers generally. Use the sim-to-sim skill only after the asset and task run in both backends.

## Workflow

1. **Multi-backend Asset Importing Pipeline.** Use provided Isaac Lab assets directly in PhysX and MJWarp; Newton parses their supported authored USD Physics and PhysX properties. For a new URDF or MJCF, keep `run_asset_transformer=True` and `run_multi_physics_conversion=True` so the importer creates neutral, PhysX, and MuJoCo payloads. Account for its nested rigid-body structure.
2. **Use per-solver asset configuration classes.** Put common USD Physics properties in solver-common base cfgs. Put MJWarp-specific fields in `Mujoco*Cfg`, Newton-native fields in `Newton*Cfg`, and PhysX-only fields in `Physx*Cfg`. Confirm support in the generated schema APIs.
3. **Audit the authored mechanical model.** Check every dynamic link and contact-relevant object for intentional mass, COM, inertia and frames, collision geometry, approximation and scale, materials, articulation root, fixed-base and fixed-joint representation, joint axes and limits, self-collision, and gravity overrides.
4. **Match contact and friction behavior.** Expect more default slip in MJWarp. Validate colliders, material bindings, contact locations and gripper force. Set per-shape `condim` with `MujocoCollisionCfg`, tune material friction, then set global `MJWarpSolverCfg(cone=..., impratio=...)`. Treat `priority`, `solmix`, `solref`, and `solimp` as expert per-collider overrides. Use fixed-grasp displacement, contact count, effort, penetration, and success metrics.
5. **Velocity limits distinction.** `velocity_limit` is a rated speed and `velocity_limit_sim` is a requested solver clamp. MJWarp enforces neither. Add task or control checks for required speed bounds and use per-joint `effort_limit_sim`.
6. **Why MJWarp often needs more armature.** Use reflected rotor inertia or controlled response tests for articulated coordinates. A plain rigid object has body inertia, not actuator armature. Correct mass, inertia, units, reset penetration, effort, action scale, control period, and contact capacity before changing armature.
7. **Retune damping with armature.** Increasing armature changes effective inertia and damping ratio. Tune armature, stiffness, and damping together from a step response, use conservative action scales, and keep targets away from hard stops to prevent bang-bang control.
8. **Choose an MJWarp starting profile.** Do not translate PhysX parameters numerically. Start from the nearest profile on the MJWarp solver page, keep the documented convergence defaults initially, enable `debug_mode`, and use MuJoCo contacts unless the task requires Newton's collision pipeline. Then run zero and random agents in PhysX and MJWarp through multiple resets. Check non-finite state, first-step impulses, saturation, excessive angular velocity, contact loss, and warnings. Reject penetrations, impossible mimic states, and invalid randomized geometry before stepping.
9. **Diagnose MJWarp-only failures.** Reproduce the first bad step with one environment, a fixed state, no randomization, and identical actions. Classify initialization/model, contact/capacity, control, or dense-scene failures before tuning. Raise overflowing capacity first; change convergence settings only after the asset, reset, controller, contact model, and capacities are valid.

## Validation

Require an asset that parses into both backends with supported per-solver fields, intentional mass, collision and contact data, physically justified actuator parameters, finite task behavior, and valid resets. Do not classify an asset as ready while importer or solver warnings remain unexplained.

## Maintenance

Keep this skill synchronized section-for-section with the asset migration guide and use the MJWarp solver page for current profile values.

## References

- [Compact reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [MJWarp solver page](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst)
- [Schema configuration classes](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html)
