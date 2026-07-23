# Newton/MJWarp Asset Migration Reference

This reference follows the sections in the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst).

## Multi-Backend Asset Importing Pipeline

- Use `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py`.
- Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True`.
- Expect neutral physics, PhysX, and MuJoCo payloads and a nested rigid-body structure.

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
python scripts/environments/zero_agent.py \
  --task TASK --num_envs 4 --headless physics=physx
python scripts/environments/zero_agent.py \
  --task TASK --num_envs 4 --headless physics=newton_mjwarp
python scripts/environments/random_agent.py \
  --task TASK --num_envs 4 --headless physics=physx
python scripts/environments/random_agent.py \
  --task TASK --num_envs 4 --headless physics=newton_mjwarp
```

Let each agent run through multiple resets. Reject robot-object and robot-support penetration, impossible mimic states, and invalid geometry before stepping. For cached valid states, inspect explicit colliders, cover each heterogeneous group, exclude fixed bases from ground-clearance tests, use positions relative to environment origins, and rebuild after topology or geometry changes.
