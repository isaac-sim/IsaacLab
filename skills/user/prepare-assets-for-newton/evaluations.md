# Newton/MJWarp Asset Migration Evaluations

## Scenario 1: Multi-Physics Conversion

Query: "How should I convert this URDF for both PhysX and MJWarp?"

Expected behavior:

- Use the updated converter with asset transformation and multi-physics conversion enabled.
- Account for neutral, PhysX, and MuJoCo payloads and nested rigid bodies.

Known failure modes:

- Produce a single-backend asset or assume the old flat hierarchy.

## Scenario 2: Solver-Specific Properties

Query: "Can I reuse all PhysX rigid-body fields in MJWarp?"

Expected behavior:

- Route common, MuJoCo/MJWarp, Newton-native, and PhysX-only fields through their matching cfg classes.
- Verify support in the generated schema APIs.

Known failure modes:

- Assume an authored or imported value is consumed by every solver.

## Scenario 3: Velocity Limits

Query: "Why does MJWarp exceed my joint velocity limit?"

Expected behavior:

- Explain that MJWarp enforces neither `velocity_limit` nor `velocity_limit_sim`.
- Recommend explicit task checks and physically justified control limits.

Known failure modes:

- Treat either field as an MJWarp safety clamp.

## Scenario 4: Zero-Gravity Angular Velocity

Query: "Should I add armature to a spinning object?"

Expected behavior:

- Distinguish articulated-coordinate armature from plain rigid-body inertia.
- Correct the model first, use the smallest justified armature, and retune damping.

Known failure modes:

- Add arbitrary armature to a plain rigid object or claim gravity damps rotation.

## Scenario 5: MJWarp Starting Profile

Query: "What solver values should I start with for dexterous manipulation?"

Expected behavior:

- Use the documented `200`/`70`, elliptic cone, `impratio=10`, two-substep profile.
- Keep initial convergence defaults and enable `debug_mode`.

Known failure modes:

- Translate PhysX numbers directly or treat the profile as a guarantee.

## Scenario 6: Task Validation

Query: "How do I validate the migrated asset?"

Expected behavior:

- Run zero and random agents with both physics presets through multiple resets.
- Reject invalid reset geometry and check state, impulses, saturation, angular velocity, contacts, and warnings.

Known failure modes:

- Stop after successful parsing or rely on depenetration.
