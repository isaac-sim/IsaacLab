# Newton Asset Preparation Examples

## Missing Mass Properties

Input: Newton reports placeholder inertia for a robot that works in PhysX.

Expected workflow:

1. Confirm the same asset path works in a PhysX baseline.
2. Inspect whether rigid bodies have authored mass, inertia, and center of mass.
3. Author or bake explicit physical metadata in a reusable USD layer or package.
4. Re-run a Newton smoke in the target task.
5. Record residual warnings and modeling choices.

## Converted Robot Does Not Move

Input: the converted asset spawns under Newton, but actions do not move the robot.

Expected workflow:

1. Check action dimensions and actuator joint name patterns.
2. Check controller body or frame names after conversion.
3. Run zero-action and small nonzero-action rollouts.
4. Fix task config names or actuator gains rather than assuming asset import is enough.

## Object Works Alone But Fails In Task

Input: a standalone object audit passes, but the task fails under Newton.

Expected workflow:

1. Validate the exact task spawn path.
2. Check task-level material or collision overrides.
3. Inspect support surfaces and contact-relevant rigid objects separately.
4. Confirm observations and rewards remain finite during reset and first steps.
