# Newton/MJWarp Asset Migration Reference

This reference follows the sections in the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst).

## Multi-Backend Asset Importing Pipeline

- Use provided Isaac Lab assets directly in PhysX and MJWarp. Newton parses their supported authored USD Physics and PhysX properties; verify support rather than assuming every authored field is used.
- For a new asset, use `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py`.
- Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True` so new conversions contain neutral physics, PhysX, and MuJoCo payloads.
- Expect the new converter's nested rigid-body structure.

## Use Per-Solver Asset Configuration Classes

| Property | Configuration |
| --- | --- |
| Common USD Physics | `RigidBodyBaseCfg`, `JointDriveBaseCfg`, and other base cfgs |
| MJWarp-specific | `MujocoRigidBodyPropertiesCfg`, `MujocoJointDrivePropertiesCfg` |
| Newton-native | matching `Newton*PropertiesCfg` |
| PhysX-only | matching `Physx*PropertiesCfg` |

A field present in an asset or imported model is not proof that MJWarp consumes it. Check the Newton/MuJoCo and PhysX schema APIs.

## Audit The Authored Mechanical Model

Check:

- positive mass, COM, and positive-definite inertia in the intended frames;
- explicit intended colliders rather than render-mesh fallback;
- collision approximation, scale, margins or offsets, materials, restitution, and filters;
- articulation root, fixed base, fixed-joint merging, joint types, axes, and limits; and
- intentional body-level gravity behavior.

## Match Contact And Friction Behavior

MJWarp can slip more than PhysX under nominally similar material settings. Tune in this order:

1. Verify colliders, material bindings, contact locations and counts, and available gripper force.
2. Inspect resolved `condim`: its default `3` adds tangential friction, `1` is frictionless, `4` adds torsional friction, and `6` adds rolling friction.
3. Tune material friction against measured tangential slip; do not map PhysX static/dynamic settings numerically.
4. Compare `cone="elliptic"` with `"pyramidal"` and use `impratio=10` as a grasping starting point only after the contact model is valid.

For copy-ready code, use the guide's task-local `MujocoCondimCfg(CollisionFragment)` and set `spawn.collision_props=[UsdPhysicsCollisionCfg(...), MujocoCondimCfg(condim=4)]`. This authors per-shape `mjc:condim`; the file-spawner override is recursive. Set the global options with `NewtonCfg(solver_cfg=MJWarpSolverCfg(cone="elliptic", impratio=10.0))`.

Track fixed-grasp displacement, contact count, effort, penetration, success, convergence, and runtime. Do not hide missing contacts, bad collision geometry, or insufficient effort with friction, `condim`, or `impratio`.

## Velocity Limits Distinction

- `velocity_limit` is the actuator's rated speed; MJWarp does not parse or enforce it.
- `velocity_limit_sim` requests a solver clamp; MJWarp drops the imported value and does not enforce it.
- Check required speed bounds in observations or terminations. Use effort limits, damping, armature, action scaling, rate limits, or controller clipping for well-behaved response.
- PhysX can enforce its supported clamp, so a tight PhysX clamp can hide a task termination.

## Why MJWarp Often Needs More Armature

Use `armature ~= rotor_inertia * gear_ratio^2` as a physical starting point for geared joints. Armature adds diagonal generalized inertia, so it reduces the velocity produced by drive, contact, and constraint impulses. This is why low-inertia fingers, distal joints, and articulated free coordinates often need larger per-joint armature in MJWarp.

Zero gravity exposes the issue because unsupported objects do not settle and repeated impulses can excite low-inertia coordinates. Add armature only to articulated coordinates that expose it. For a plain rigid object, correct body mass and inertia, model physical losses through contact or explicit drag, and enforce any required angular-speed bound in task or control logic.

Do not use armature to hide bad inertials, units, resets, effort, timing, or capacity.

## Retune Damping With Armature

Retune damping after armature changes. Increasing effective inertia with fixed damping lowers the damping ratio. Use a non-oscillatory step response, conservative action scale, and targets away from hard stops. Do not mask a bad inertia tensor with extreme damping.

## Choose An MJWarp Starting Profile

Keep `solver="newton"`, `integrator="implicitfast"`, `iterations=100`, `ls_iterations=50`, and `tolerance=1e-6` for the first explicit baseline. Enable `debug_mode` while tuning.

| Profile | `njmax` | `nconmax` | Cone / `impratio` | Substeps |
| --- | ---: | ---: | --- | ---: |
| Simple articulation/reach | 50 | 20 | pyramidal / 1 | 1 |
| Locomotion | 100 | 40 | pyramidal / 1 | 1 |
| Dexterous manipulation | 200 | 70 | elliptic / 10 | 2 |
| Dense manipulation | 300 | 200 | task-dependent | 2 |

These are starting budgets, not fidelity guarantees. Use MuJoCo contacts by default. Set `use_mujoco_contacts=False` and configure `collision_cfg` only when the task needs Newton's collision pipeline.

### Task-Level Smoke And Reset Validation

```bash
uv run python scripts/environments/zero_agent.py --task TASK --num_envs 4 --headless physics=physx
uv run python scripts/environments/zero_agent.py --task TASK --num_envs 4 --headless physics=newton_mjwarp
uv run python scripts/environments/random_agent.py --task TASK --num_envs 4 --headless physics=physx
uv run python scripts/environments/random_agent.py --task TASK --num_envs 4 --headless physics=newton_mjwarp
```

Let each agent run through multiple resets. Reject robot-object and robot-support penetration, impossible mimic states, and invalid geometry before stepping. For cached valid states, inspect explicit colliders, cover each heterogeneous group, exclude fixed bases from ground-clearance tests, use positions relative to environment origins, and rebuild after topology or geometry changes.

## Diagnose MJWarp-Only Failures

Reproduce the first bad step in one environment with a fixed state, no randomization, and identical
actions. Classify it before tuning:

- initialization or first step: asset data, scale, reset overlap, topology, drives, unsupported features;
- contact onset: contact locations/counts, capacity warnings, margins, `condim`, friction, cone, mass ratios;
- controlled motion: effort, gains, action scale, `dt`, substeps, damping, armature, limits; or
- dense scenes: busiest-environment `nconmax` and `njmax` demand.

Use `debug_mode` for iteration-cap evidence. Raise overflowing capacity first and change
convergence work only after the model, reset, controller, contact path, and capacities are valid.
