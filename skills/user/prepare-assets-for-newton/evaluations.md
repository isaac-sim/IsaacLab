# Newton/MJWarp Asset Migration Evaluations

## Contents

- Placeholder inertia
- Asset imports but control fails
- Task-level failure
- Shared config compatibility
- Mimic gripper migration
- Bang-bang MJWarp control
- Concrete migration audit
- Zero-gravity angular velocity

## Scenario 1: Placeholder Inertia

Query: "My PhysX robot runs, but Newton/MJWarp reports placeholder inertia."

Expected behavior:

- Establishes a PhysX baseline.
- Audits authored mass, inertia, and center of mass.
- Recommends fixing authored USD physics metadata or producing a local package.
- Requires task-level MJWarp validation after the asset audit.

Known failure modes:

- Treats PhysX runtime success as proof of MJWarp readiness.
- Suppresses warnings without fixing asset metadata.

## Scenario 2: Asset Imports But Control Fails

Query: "The converted robot spawns under Newton/MJWarp, but the policy actions do nothing."

Expected behavior:

- Checks actuator joint patterns, controller body names, and action dimensions.
- Runs zero-action and small nonzero-action rollouts.
- Separates asset import success from control readiness.

Known failure modes:

- Keeps changing USD mass properties when the task action config is stale.
- Declares the asset ready after standalone import only.

## Scenario 3: Task-Level Failure

Query: "The object passes a standalone MJWarp check but fails inside my environment."

Expected behavior:

- Validates the exact task spawn path and overrides.
- Audits support collision, contact materials, and nested references.
- Checks reset and first-step finite state in the target task.

Known failure modes:

- Assumes standalone USD parsing covers task-level material and collision overrides.
- Ignores support geometry and contact-relevant scene assets.

## Scenario 4: Shared Config Compatibility

Query: "Replace our stock Franka USD with the MJWarp-ready asset so every task gets it."

Expected behavior:

- Identifies the blast radius for public tasks and checkpoints.
- Preserves the shared legacy config and creates a task-local converted config by default.
- Compares resolved names, order, action width, and observation width.

Known failure modes:

- Globally swaps the asset without a deprecation or compatibility audit.
- Assumes identical visual geometry implies checkpoint compatibility.

## Scenario 5: Mimic Gripper Migration

Query: "My two Franka fingers are coupled, but MJWarp snaps them together on reset and the action count changed."

Expected behavior:

- Treats the pair as one mechanical DOF.
- Checks whether the coupling is already authored.
- Uses a driven leader, passive follower, shared reset sample, and leader-only gain randomization.
- Preserves action width/order or requires retraining.

Known failure modes:

- Independently resets or randomizes both coupled joints.
- Silently drops a follower action when deploying an existing checkpoint.

## Scenario 6: Bang-Bang MJWarp Control

Query: "The converted robot is stable in PhysX but oscillates between full actions in MJWarp."

Expected behavior:

- Compares `dt * decimation`, action scaling, step response, and hard-stop use.
- Audits mass/inertia and increases physically plausible armature and damping before solver tuning.
- Separates rated `velocity_limit` from `velocity_limit_sim`.
- Revalidates deterministic open-loop behavior before adding randomization.

Known failure modes:

- Only increases MJWarp solver iterations.
- Uses an unrealistically tight solver velocity clamp to hide the oscillation.

## Scenario 7: Concrete Migration Audit

Query: "Give me exact tooling to prove this task's Franka asset matches between PhysX and MJWarp."

Expected behavior:

- Runs `zero_agent.py` and `random_agent.py` under both physics presets.
- Disables startup randomization for the nominal comparison and keeps importer and solver logs.
- Records and compares ordered body, joint, actuator, action, and observation names, plus mass/inertia,
  limits, timing, and policy-interface behavior.
- Classifies every difference instead of treating successful task construction as proof of parity.

Known failure modes:

- Offers only a generic USD checklist with no reproducible commands.
- Treats positive inertia diagnostics as proof that units and frames are correct.

## Scenario 8: Zero-Gravity Angular Velocity

Query: "My grasped object spins extremely fast in MJWarp at zero gravity. Should I increase armature?"

Expected behavior:

- Finds the first drive, contact, constraint, or reset impulse and audits mass, inertia, scale, and penetration first.
- Explains that armature conditions articulated generalized coordinates and bounds the velocity response to impulses.
- Distinguishes articulated/free-joint armature from a plain rigid object's body inertia and avoids assuming PhysX damping/clamp attributes work in MJWarp.
- Uses the smallest justified increase, retunes damping, and tests zero and nominal gravity in both backends.

Known failure modes:

- Claims gravity directly damps rotation.
- Adds arbitrary armature or damping to hide invalid inertials or reset penetration.
