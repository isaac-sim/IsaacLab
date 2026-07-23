# Newton/MJWarp Asset Migration Examples

## Placeholder Inertia

Audit and correct authored mass, COM, inertia, units, and frames. Reconvert when needed, then rerun the exact task. Do not compensate with solver iterations.

## Velocity Limit Is Exceeded

Treat `velocity_limit` as rated speed and `velocity_limit_sim` as a solver request. Because MJWarp enforces neither, add the required observation or termination check and tune effort, damping, armature, action scaling, rate limits, or controller clipping.

## Zero-Gravity Spin-Up

Correct mass, inertia, units, and reset penetration first. Add the smallest justified armature only when the unstable coordinate is articulated. A plain rigid object needs correct body inertia and a physical loss model or explicit speed bound, not actuator armature.

## Bang-Bang Control

Tune armature, stiffness, and damping together from an open-loop step response. Use enough damping for a plausible non-oscillatory response, conservative action scales, and targets away from hard stops.

## Selecting A Solver Profile

Choose the nearest documented profile, keep the initial convergence defaults, enable `debug_mode`, and size `njmax` and `nconmax` for the task. Use Newton contacts only when the task needs that collision pipeline.
