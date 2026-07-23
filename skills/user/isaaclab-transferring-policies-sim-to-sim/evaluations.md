# Sim-To-Sim Policy Transfer Evaluations

## Scenario 1: PhysX To MJWarp

Query: "Deploy this PhysX-trained Franka checkpoint in Newton."

Expected behavior:

- Require a Newton-clean task, exact contract match, deterministic parity, PP baseline, and explicit PN checkpoint.
- Report nominal and randomized metrics separately.

Known failure modes:

- Tune policy or solver settings before establishing the source baseline and contract.

## Scenario 2: MJWarp To PhysX

Query: "Can a policy trained with Newton run under PhysX?"

Expected behavior:

- Measure NN and NP with matched seeds, goals, normalizer, timing, and play config.

Known failure modes:

- Assume forward transfer proves reverse transfer.

## Scenario 3: Mimic-Drive Mismatch

Query: "My checkpoint has one gripper command, but PhysX applies more finger effort."

Expected behavior:

- Explain that MJWarp drives the constrained leader once while PhysX leaves the follower driveable.
- Make the follower passive without changing checkpoint width or order silently.

Known failure modes:

- Attribute the difference only to action dimensions or drive both fingers.

## Scenario 4: Bang-Bang Control

Query: "The policy alternates full actions only in MJWarp."

Expected behavior:

- Check timing, action scale, inertia, armature, damping, and hard stops before solver tuning.

Known failure modes:

- Hide the problem with an unenforced MJWarp velocity clamp or broad randomization.

## Scenario 5: Observation Contract Changed

Query: "I removed body velocities; can I deploy the old checkpoint?"

Expected behavior:

- Require retraining or a validated adapter and preserve the old normalizer/history for the old contract.

Known failure modes:

- Pad, truncate, or reorder observations without semantic validation.

## Scenario 6: Transfer Randomization

Query: "What should I randomize for both transfer directions?"

Expected behavior:

- Match the nominal model first, then add friction, payload, actuator, gravity, reset, and noise families incrementally.
- Keep mimic joints coherent and report nominal/randomized PP, PN, NN, and NP separately.

Known failure modes:

- Widen all distributions at once or randomize around a known model bug.
