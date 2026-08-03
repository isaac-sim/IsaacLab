# Newton/MJWarp Asset Migration Examples

## Existing Isaac Lab Asset

Use the asset in both backends without creating a Newton-only copy. Newton parses supported
authored USD Physics and PhysX properties; audit unsupported or ignored fields before relying on
them. Use the multi-physics importers when converting a new URDF or MJCF asset.

## Placeholder Inertia

Audit and correct authored mass, COM, inertia, units, and frames. Reconvert when needed, then rerun the exact task. Do not compensate with solver iterations.

## Excessive Grasp Slip

Verify collision geometry, contacts, material bindings, and gripper force. Author per-shape
`mjc:condim` with `MujocoCollisionCfg` in `spawn.collision_props`, tune material
friction, then set global `MJWarpSolverCfg(cone="elliptic", impratio=10.0)` and compare fixed-grasp
metrics. Limit recursive spawner overrides to assets whose colliders should all use that `condim`.

## Velocity Limit Is Exceeded

Treat `velocity_limit` as rated speed and `velocity_limit_sim` as a solver request. Because MJWarp enforces neither, add the required observation or termination check and tune effort, damping, armature, action scaling, rate limits, or controller clipping.

## Zero-Gravity Spin-Up

Correct mass, inertia, units, and reset penetration first. Add the smallest justified armature only when the unstable coordinate is articulated. A plain rigid object needs correct body inertia and a physical loss model or explicit speed bound, not actuator armature.

## Bang-Bang Control

Tune armature, stiffness, and damping together from an open-loop step response. Use enough damping for a plausible non-oscillatory response, conservative action scales, and targets away from hard stops.

## Selecting A Solver Profile

Choose the nearest documented profile, keep the initial convergence defaults, enable `debug_mode`, and size `njmax` and `nconmax` for the task. Use Newton contacts only when the task needs that collision pipeline.

## MJWarp-Only NaN

Reproduce the first failing step with one environment, fixed state and actions, and no
randomization. Classify model/reset, contact/capacity, control, or dense-scene causes before
changing convergence settings.
