# Newton/MJWarp Asset Migration Evaluations

## Scenario 1: Placeholder Inertia

Query: "My PhysX robot runs, but MJWarp reports placeholder inertia."

Expected behavior:

- Audit authored mass, COM, inertia, units, and frames.
- Correct or reconvert the asset, then validate the exact task.

Known failure modes:

- Suppress the warning or tune the solver around invalid inertials.

## Scenario 2: Control Does Not Move

Query: "The converted robot spawns, but actions do nothing."

Expected behavior:

- Check actuator joint matches, drives, controller frames, action dimensions, and small-action behavior.

Known failure modes:

- Treat successful import as control readiness.

## Scenario 3: Mimic Gripper

Query: "My Franka fingers snap and the action count changed."

Expected behavior:

- Use one coupling, a driven leader, passive follower, shared reset, and an explicit checkpoint-compatibility decision.

Known failure modes:

- Drive or randomize the follower independently, or silently remove an action.

## Scenario 4: Bang-Bang Control

Query: "The robot is smooth in PhysX but alternates full actions in MJWarp."

Expected behavior:

- Check timing, action scale, inertials, armature, damping, and hard stops before convergence tuning.
- State that MJWarp enforces neither velocity-limit field.

Known failure modes:

- Hide oscillation with a clamp or solver iterations.

## Scenario 5: Concrete Audit

Query: "Give me exact steps to compare this task in PhysX and MJWarp."

Expected behavior:

- Run zero and random agents in both presets.
- Save contract and warning records, locate the first divergence, and cover contact objects and heterogeneous groups.

Known failure modes:

- Offer only a generic USD checklist or accept a successful spawn as parity.

## Scenario 6: Zero-Gravity Angular Velocity

Query: "My object spins extremely fast in MJWarp at zero gravity. Should I add armature?"

Expected behavior:

- Diagnose the first impulse and correct mass, inertia, and reset geometry.
- Distinguish articulated-coordinate armature from plain rigid-body inertia, then retune damping.

Known failure modes:

- Add arbitrary armature or claim gravity damps rotation.
