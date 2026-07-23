# Newton/MJWarp Asset Migration Examples

## Missing Mass Properties

Input: Newton/MJWarp reports placeholder inertia for a robot that works in PhysX.

Expected workflow:

1. Confirm the same asset path works in a PhysX baseline.
2. Inspect whether rigid bodies have authored mass, inertia, and center of mass.
3. Author or bake explicit physical metadata in a reusable USD layer or package.
4. Re-run an MJWarp smoke in the target task.
5. Record residual warnings and modeling choices.

Do not compensate for missing inertia by increasing solver iterations. First author physically meaningful properties, then tune armature and damping from the nominal model.

## Preserve a Shared PhysX Asset

Input: a public articulation config is used by many PhysX tasks, but a converted multi-physics asset is needed for one Newton/MJWarp migration.

Expected workflow:

1. Freeze the public config and its legacy asset path.
2. Copy it into a task-local config.
3. Change only the task-local asset path and calibrated actuator groups.
4. Compare body/joint/action/observation order before loading a checkpoint.
5. Validate both presets in the migrated task without changing unrelated tasks.

This mirrors the Franka Dexsuite approach: the stock config stayed on the relocated legacy asset while the Menagerie-converted asset was scoped to Dexsuite.

For the migrated Franka actuator table, take effort and rated speed from the Franka data sheet, use the maintained libfranka impedance controller as a gain reference, and record armature or identification assumptions separately. Do not copy the table to a different robot without re-sourcing every joint.

## Converted Robot Does Not Move

Input: the converted asset spawns under Newton/MJWarp, but actions do not move the robot.

Expected workflow:

1. Check action dimensions and actuator joint name patterns.
2. Check controller body or frame names after conversion.
3. Run zero-action and small nonzero-action rollouts.
4. Fix task config names or actuator gains rather than assuming asset import is enough.

## Coupled Gripper Snaps at Reset

Input: two finger joints are equality-coupled, and MJWarp produces a large impulse at the first step.

Expected workflow:

1. Confirm the coupling is authored once and identify the driven joint.
2. Make the follower passive instead of giving it an independent drive.
3. Reset both joint positions with one shared random sample.
4. Randomize damping or closing speed only on the driver.
5. Preserve the checkpoint action contract; retrain if changing from two action entries to one.

## MJWarp Joint Uses Bang-Bang Control

Input: a policy alternates saturated position actions in MJWarp but looks smoother in PhysX.

Expected workflow:

1. Verify the policy period from `dt * decimation`.
2. Compare open-loop step responses and action scaling.
3. Audit mass, inertia, and per-joint armature before solver tuning.
4. Increase damping to a physically plausible response and keep targets away from hard stops.
5. Randomize gains around the corrected nominal model and re-evaluate both backends.

## Object Spins Up During Zero-Gravity Training

Input: an object or low-inertia articulated coordinate develops very large angular velocity in MJWarp during the zero-gravity curriculum.

Expected workflow:

1. Confirm the object starts collision-free and has correct mass, center of mass, inertia, scale, and units.
2. Locate the first impulse and determine whether it comes from contact, a constraint, a drive, or a reset correction.
3. If the object is represented as an articulation with free/generalized joints, calculate or identify a plausible armature and increase it by the smallest amount that produces a finite, credible impulse response.
4. Retune damping after changing armature and repeat the same controlled impulse under PhysX and MJWarp.
5. If it is a plain rigid object, do not look for actuator armature. Correct body inertia first, and do not assume PhysX angular-damping or maximum-velocity attributes are consumed by MJWarp.
6. Re-test at zero gravity and full nominal gravity. Zero gravity exposes the issue by keeping the object unsupported; gravity is not rotational damping.

## Object Works Alone But Fails In Task

Input: a standalone object audit passes, but the task fails under MJWarp.

Expected workflow:

1. Validate the exact task spawn path.
2. Check task-level material or collision overrides.
3. Inspect support surfaces and contact-relevant rigid objects separately.
4. Confirm observations and rewards remain finite during reset and first steps.

Do not let the target solver depenetrate an invalid reset. Use explicit collision geometry to reject robot/object and support-surface intersections before the first step.
