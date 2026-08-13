---
name: isaaclab-preparing-assets-for-newton
description: Prepares PhysX-compatible robot, object, and scene assets for Isaac Lab's Newton backend with MJWarp. Use when handling multi-physics conversion, per-solver configuration, mechanical-model audits, contact and friction behavior, actuator limits, paired backend smoke tests, or Newton-only failures.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Prepare Assets For Newton With MJWarp

## When To Use

Read the [Prepare an Asset for Newton with MJWarp how-to](../../../docs/source/how-to/prepare_asset_for_newton.rst) first. It is the authoritative source for this skill and targets Newton with MJWarp, not Newton solvers generally. Use the sim-to-sim skill only after the asset and task run in both backends.

1D cable / rod assets are out of scope: author them as Newton deformables (Newton + VBD only). See the [Using Cables guide](../../../docs/source/overview/core-concepts/physical-backends/newton/using-cables.rst). Implicit MPM particle assets and rigid-MPM coupling are also out of scope; see the [Using Implicit MPM guide](../../../docs/source/overview/core-concepts/physical-backends/newton/using-mpm.rst).

## Workflow

1. **Prerequisites.** Select the explicit `newton_mjwarp` backend preset and use the backend-and-preset documentation to identify intentional task differences.
2. **Import a multi-physics asset.** Use existing assets in both backends where supported. For new URDF or MJCF assets, retain `run_asset_transformer=True` and `run_multi_physics_conversion=True` to create neutral physics, PhysX, and MuJoCo payloads.
3. **Separate common and solver-specific properties.** Keep common USD Physics fields in base cfgs, MJWarp fields in `Mujoco*Cfg`, Newton-native fields in `Newton*Cfg`, and PhysX-only fields in `Physx*Cfg`. Confirm fields in the schema APIs.
4. **Audit the mechanical model.** Verify intentional positive mass, COM, inertia and frames, collision geometry, approximation and scale, material binding, topology, joint axes and limits, self-collision, and gravity overrides.
5. **Match collision, contact, and friction behavior.** Verify colliders, bindings, contact locations/counts, and normal force before changing friction. Inspect `condim`, then tune material friction. Reserve `priority`, `solmix`, `solref`, and `solimp` for measured per-collider needs; use the Tune MJWarp how-to for global tuning.
6. **Validate actuators and limits.** Audit per-joint effort, gains, friction, armature, action scale, and control period. Armature is for articulated coordinates only; retune damping after changing it. MJWarp does not enforce `velocity_limit` or `velocity_limit_sim`, so enforce required speed bounds in task or control logic.
7. **Run paired smoke tests.** Run the same fixed task state in PhysX and MJWarp across multiple resets. Compare displacement, contacts, effort, penetration, success, finite state, impulses, saturation, angular velocity, contact loss, and warnings. Reject invalid resets before stepping.
8. **Validate solver differences before porting.** Use the [Solver Differences concept comparison](../../../docs/source/concepts/solver_differences.rst) and the [Tune MJWarp how-to](../../../docs/source/how-to/tune_mjwarp.rst) to revalidate contact behavior, friction, restitution, timestep, and substeps in the target solver. Follow their focused procedure instead of copying PhysX settings numerically.
9. **Diagnose Newton-only failures.** Reproduce the first bad step with one environment, fixed state, no randomization, and identical actions. Classify initialization/model, contact/capacity, control, or dense-scene failures. Raise overflowing capacity before changing convergence settings.

## Validation

Require an asset that parses into both backends with supported per-solver fields, intentional mechanical and contact data, physically justified actuator parameters, finite task behavior, and valid resets. Do not classify an asset as ready while importer or solver warnings remain unexplained.

## Maintenance

Keep this skill synchronized with the authoritative how-to. Use the generated Newton configuration API for current solver defaults and the Tune MJWarp how-to for tuning guidance.

## References

- [Compact reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Tune MJWarp how-to](../../../docs/source/how-to/tune_mjwarp.rst)
- [Schema configuration classes](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html)
