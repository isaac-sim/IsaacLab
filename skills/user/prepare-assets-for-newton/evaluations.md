# Newton/MJWarp Asset Migration Evaluations

## Scenario 1: Existing And Newly Converted Assets

Query: "Will Isaac Lab's existing asset work in MJWarp, and how should I convert a new URDF?"

Expected behavior:

- Explain that provided assets run in both backends because Newton parses supported authored USD Physics and PhysX properties.
- For a new URDF, use asset transformation and multi-physics conversion.
- Account for the layered payloads and nested rigid bodies produced by the new converter.

Known failure modes: claiming every existing PhysX field is consumed by MJWarp or requiring a
Newton-only copy of a provided asset.

## Scenario 2: Solver-Specific Properties

Query: "Can I reuse all PhysX rigid-body fields in MJWarp?"

Expected behavior:

- Route common, MuJoCo/MJWarp, Newton-native, and PhysX-only fields through their matching cfg classes.
- Verify support in the generated schema APIs.

Known failure modes: assuming an authored or imported value is consumed by every solver.

## Scenario 3: Contact Slip

Query: "Why does the object slip from my gripper in MJWarp but not PhysX?"

Expected behavior:

- Validate collision shapes, contacts, material bindings, and gripper force first.
- Set per-shape `mjc:condim` through `MujocoCollisionCfg` in `spawn.collision_props`, tune material
  friction, then set global `MJWarpSolverCfg(cone=..., impratio=...)` and compare fixed-grasp metrics.

Known failure modes: treating `condim` as a global solver field, treating `impratio` as an asset
field, recursively changing unintended colliders, or hiding missing contacts or insufficient effort.

## Scenario 4: Velocity Limits

Query: "Why does MJWarp exceed my joint velocity limit?"

Expected behavior:

- Explain that MJWarp enforces neither `velocity_limit` nor `velocity_limit_sim`.
- Recommend explicit task checks and physically justified control limits.

Known failure modes: treating either field as an MJWarp safety clamp.

## Scenario 5: Zero-Gravity Angular Velocity

Query: "Should I add armature to a spinning object?"

Expected behavior:

- Distinguish articulated-coordinate armature from plain rigid-body inertia.
- Correct the model first, use the smallest justified armature, and retune damping.

Known failure modes: adding arbitrary armature to a plain rigid object or claiming gravity damps rotation.

## Scenario 6: MJWarp Starting Profile

Query: "What solver values should I start with for dexterous manipulation?"

Expected behavior:

- Use the documented `200`/`70`, elliptic cone, `impratio=10`, two-substep profile.
- Keep initial convergence defaults and enable `debug_mode`.

Known failure modes: translating PhysX numbers directly or treating the profile as a guarantee.

## Scenario 7: Task Validation

Query: "How do I validate the migrated asset?"

Expected behavior:

- Run zero and random agents with both physics presets through multiple resets.
- Reject invalid reset geometry and check state, impulses, saturation, angular velocity, contacts, and warnings.

Known failure modes: stopping after successful parsing or relying on depenetration.

## Scenario 8: MJWarp-Only Failure

Query: "MJWarp produces NaNs, but the same scene runs in PhysX. Which solver value should I change?"

Expected behavior:

- Reproduce the first failure with one environment, a fixed state, no randomization, and identical actions.
- Classify initialization/model, contact/capacity, control, or dense-scene causes.
- Raise overflowing capacity before changing convergence settings.

Known failure modes: assuming PhysX success proves the asset and reset are valid for MJWarp or
increasing iterations before locating the first non-finite quantity.
