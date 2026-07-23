# Newton/MJWarp Asset Migration Examples

## Placeholder Inertia

For an asset that runs in PhysX but reports placeholder inertia in MJWarp:

1. inspect authored mass, COM, inertia, units, and frames;
2. fix the reusable asset layer or reconvert it;
3. rerun the exact task in MJWarp; and
4. tune armature or the solver only after the model is valid.

## Coupled Franka Fingers

For coupled fingers that snap or exert different force:

1. keep one authored mimic relation;
2. drive `panda_finger_joint1`;
3. make `panda_finger_joint2` passive with zero stiffness and damping;
4. reset both fingers from one sample; and
5. preserve any existing checkpoint action contract until retraining.

## Bang-Bang Joint Control

For saturated alternating actions in MJWarp:

1. verify `dt * decimation`, action scale, inertia, and hard-stop use;
2. source plausible armature and compare an open-loop step response;
3. retune damping after changing armature; and
4. add domain randomization only after nominal behavior is credible.

Do not rely on `velocity_limit` or `velocity_limit_sim` as an MJWarp clamp.

## Zero-Gravity Spin-Up

Locate the first reset, drive, constraint, or contact impulse. Correct mass and inertia first. Increase armature only if the unstable coordinate is articulated and supports armature; use the smallest justified value and retune damping. A plain rigid object needs correct body inertia and a physically modeled loss or explicit task bound, not actuator armature.

## Standalone Asset Passes, Task Fails

Validate task-level collision and material overrides, support surfaces, sensors, heterogeneous groups, and reset geometry. Reject penetration before stepping instead of relying on backend depenetration.
