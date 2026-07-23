---
name: isaaclab-preparing-assets-for-newton
description: Migrates and validates PhysX-compatible robot, object, and scene assets for Isaac Lab's Newton backend with the MJWarp solver. Use when converting URDF, MJCF, or USD assets; adding the newton_mjwarp preset to a PhysX task; resolving mass, inertia, collision, topology, mimic-joint, actuator, damping, armature, action, reset, or sensor incompatibilities; or preserving a policy interface while changing the underlying asset.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Preparing Assets For Newton With MJWarp

## When To Use

Use this skill when a user needs to migrate a USD robot, object, or scene asset from a working PhysX task to Newton with MJWarp, or when a converted asset parses but behaves differently with `physics=newton_mjwarp`. Do not report compatibility with Kamino or another Newton solver from this workflow.

Read the [official migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) before changing the asset. Use `isaaclab-selecting-backends` for backend selection and `isaaclab-using-presets` for preset mechanics. After asset and task parity are established, use `isaaclab-transferring-policies-sim-to-sim` for checkpoint evaluation in both directions.

## Workflow

1. Read the official migration guide and check Newton/MJWarp's supported features for the required joints, constraints, sensors, and deformables.
2. Freeze a seeded PhysX baseline. Run `scripts/tools/inspect_task_asset.py` for the robot and each contact-relevant object with `physics=physx`; save the JSON and simulator warnings. Record the source revision, conversion command, reset distribution, and zero-action behavior.
3. Protect compatibility. If the original config is public or shared, copy it into a task-local migration config instead of replacing the asset globally.
4. Prefer reconversion from URDF or MJCF with `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py`, the asset transformer, and multi-physics conversion enabled. Save the exact command. For hand-authored USD, preserve neutral physics in the common layer and backend-specific attributes in the appropriate payloads. Use solver-common schema cfg classes for portable properties, `Mujoco*PropertiesCfg` for MJWarp-supported properties, `Newton*PropertiesCfg` for supported Newton-native properties, and `Physx*PropertiesCfg` only for PhysX properties.
5. Audit every dynamic body for intentional mass, center of mass, positive-definite inertia, explicit collision geometry, material bindings, articulation roots, joint axes, fixed-joint structure, and resolvable references. Fix authored data rather than suppressing Newton/MJWarp warnings.
6. Compare resolved names and order with the baseline. Update actuator expressions, actions, sensors, controller frames, resets, rewards, and terminations for every intentional rename.
7. Treat mimic/equality-coupled joints as one mechanism: author one coupling, drive the leader, make the follower passive, reset both from one shared sample, and avoid independent follower randomization. Preserve checkpoint action width/order until retraining is allowed.
8. Confirm active joints have authored drives, then build a per-joint actuator table. Source torque and speed from the hardware data sheet, gains from a maintained controller or identification, and armature from reflected rotor inertia or measured response. `JointDrivePropertiesCfg.ensure_drives_exist` is asset-wide and activates every fully zero-gain drive; do not use it when a mimic follower or another joint must remain passive. Author a targeted active drive instead.
9. Calibrate armature, stiffness, and damping together. MJWarp can expose very large velocities when contact or drive impulses act on low-inertia generalized coordinates; armature adds diagonal reflected inertia and conditions the mass matrix. Zero-gravity training makes unsupported low-inertia coordinates especially susceptible. Use the smallest physically justified increase, then retune damping to avoid bang-bang control. A plain rigid object has body inertia, not actuator armature; do not assume PhysX angular-damping or velocity-clamp attributes are consumed by MJWarp.
10. Keep `velocity_limit` (rated joint speed used by actuator/task logic) separate from `velocity_limit_sim` (requested solver clamp). MJWarp does not parse `velocity_limit` into its solver model, and its MuJoCo solver drops the imported `joint_velocity_limit` behind `velocity_limit_sim`; neither value is enforced during MJWarp stepping. Check rated-speed boundaries explicitly. PhysX can enforce its supported clamp, so do not let a tight PhysX clamp hide a termination or create a transfer difference.
11. Add explicit `physx` and `newton_mjwarp` physics presets and preserve `dt * decimation` across backends. Read the official MJWarp solver page and select the nearest checked-in task profile instead of translating PhysX values numerically. Start with `integrator="implicitfast"`; size per-environment `njmax`/`nconmax` before tuning convergence; and use Newton `collision_cfg` only with `use_mujoco_contacts=False`.
12. Make reset states valid before the first step. Reject robot/object and support-surface penetrations, shared-coupling violations, and invalid geometry variants rather than relying on solver depenetration.
13. Run `inspect_task_asset.py` again with `physics=newton_mjwarp` and diff the two JSON reports. Classify every difference. Then run PhysX and MJWarp zero-action, small-step, impulse, reset, and contact rollouts.
14. Diagnose before tuning. Reproduce the first divergence with fixed state and commands; classify it as initialization/geometry, control, model, capacity, or contact/solver behavior. Validate the model first. For contact-dominated failures, set contact representation, `dt`, substeps, and capacity before bounded convergence sweeps and contact tuning. Enable `debug_mode` to inspect iteration usage; do not reduce the default line-search budget or raise convergence work without a measured reason. For drive symptoms, tune the actuator and controller path first. Change one parameter family at a time and require improved target metrics without regressions in non-finite states, penetration/residuals, or runtime. Confirm supported knobs and defaults in the installed Newton version and the Newton Simulation Tuning guide.
15. Record the converted path, source revision, conversion options, physical-data sources, contract diffs, solver preset, commands, residual warnings, and final classification. Hand off checkpoint evaluation to the sim-to-sim skill.

## Validation

An asset is MJWarp-clean only when:

1. All rigid bodies have intentional mass properties.
2. Runtime mass and inertia values are finite, positive, and expressed in the intended frames.
3. Only intended collision geometry is parsed and materials are bound consistently.
4. Joint, fixed-base, self-collision, and mimic/equality topology are accepted by both backends.
5. Resolved body/joint order and the action/observation interface are unchanged or explicitly versioned for retraining.
6. The target task can spawn, reset, sense contacts, and step under MJWarp and PhysX.
7. Zero-action and small-action rollouts have finite observations, rewards, positions, velocities, and forces.
8. Actuator, controller, sensor, reward, and reset names resolve after conversion.
9. Armature, stiffness, and damping produce finite, plausible step and impulse responses without unexplained MJWarp angular-velocity spikes or bang-bang control.
10. Control period, rated limits, nominal step response, and reset validity are comparable across backends.
11. The PhysX and MJWarp runtime JSON reports have no unexplained contract differences.
12. Remaining differences and warnings are documented before policy transfer.

For skill changes, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with the official PhysX-to-Newton/MJWarp migration guide, the MJWarp solver page, `source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager_cfg.py`, `scripts/tools/inspect_task_asset.py`, `docs/source/how-to/import_new_asset.rst`, converter configs under `source/isaaclab/isaaclab/sim/converters/`, actuator semantics under `source/isaaclab/isaaclab/actuators/`, and the Franka/Kuka Dexsuite example under `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/`. Update the official docs first when guidance changes. Avoid storing converted USD packages, generated audit logs, or private asset paths in this skill.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Official PhysX-to-Newton/MJWarp migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst)
- [MJWarp solver configuration and task profiles](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst)
- [Schema configuration classes](../../../docs/source/overview/core-concepts/schema_cfgs.rst)
- [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html)
- [Sim-to-sim policy transfer skill](../isaaclab-transferring-policies-sim-to-sim/SKILL.md)
- [Backend selection skill](../select-backends/SKILL.md)
- [Preset skill](../use-presets/SKILL.md)
- [Importing a new asset](../../../docs/source/how-to/import_new_asset.rst)
- [Newton documentation](../../../docs/source/overview/core-concepts/physical-backends/newton/index.rst)
