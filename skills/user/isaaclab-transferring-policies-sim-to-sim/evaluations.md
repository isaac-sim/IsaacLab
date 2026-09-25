# Sim-To-Sim Policy Transfer Evaluations

## Scenario 1: Checkpoint Compatibility

Query: "Can I run this PhysX checkpoint in Newton?"

Expected behavior:

- State that the same registered task should retain one MDP.
- Resolve backend `PresetCfg` alternatives and flag any preset that changes a policy-facing MDP term.
- Require successful training of the same task in both engines.
- Compare the exact action, observation, policy-state, timing, mechanism, and episode contracts.

Known failure modes:

- Treat backend presets as permission to change the checkpoint contract.
- Treat matching tensor shapes alone as compatibility.

## Scenario 2: Mimic-Drive Difference

Query: "Why does PhysX apply more Franka finger effort than MJWarp?"

Expected behavior:

- Explain the one-drive MJWarp equality and the driveable PhysX follower.
- Make the second PhysX drive passive with zero stiffness and damping.

Known failure modes:

- Remove the follower coordinate or describe the issue only as action width.

## Scenario 3: Scrambled Joint Axis

Query: "My ANYmal-D PhysX policy falls over instantly in Newton."

Expected behavior:

- Recognize immediate collapse (rather than graceful degradation) as an ordering symptom, not a solver-dynamics one.
- Add `env.scene.robot.joint_ordering=physx env.scene.robot.body_ordering=physx` to the Newton replay, naming the source checkpoint's convention.
- Confirm by comparing `joint_names` with `backend_joint_names` and the body equivalents.

Known failure modes:

- Set only `joint_ordering` and leave body-indexed observations scrambled.
- Name the target backend instead of the source checkpoint's convention.
- Attribute the collapse to friction, contacts, or actuator tuning and start tuning solver parameters.

## Scenario 4: Bang-Bang Control

Query: "The policy alternates saturated actions in MJWarp."

Expected behavior:

- Match per-joint actuator response, timing, action hold, limits, armature, and damping.
- Retune damping after armature changes and keep targets away from hard stops.

Known failure modes:

- Tune the policy before matching nominal control behavior.

## Scenario 5: Domain Randomization

Query: "What should I randomize for sim-to-sim transfer?"

Expected behavior:

- Use the documented friction, mass/inertia, joint, armature, gravity, actuator, reset, and observation families.
- Keep ranges plausible and coupled mechanisms coherent.

Known failure modes:

- Use extreme randomization to hide an incorrect nominal model.

## Scenario 6: Curriculum

Query: "The final randomization distribution prevents learning."

Expected behavior:

- Start from easier gravity, noise, reset, or termination settings and promote to final difficulty.
- Keep a deterministic nominal evaluation.

Known failure modes:

- Evaluate only the easier curriculum stage.

## Scenario 7: Full Matrix

Query: "How do I demonstrate transfer in both directions with Franka?"

Expected behavior:

- Run PP, PN, NN, and NP with `Isaac-Lift-Franka` for both training and inference, relying on the play entry point to apply `play_mode` overrides.
- Reproduce each same-backend baseline and deploy the exact checkpoint cross-backend.

Known failure modes:

- Report only one transfer direction or use different checkpoints.
