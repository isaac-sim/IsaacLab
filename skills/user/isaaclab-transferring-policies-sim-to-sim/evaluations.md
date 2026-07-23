# Sim-To-Sim Policy Transfer Evaluations

## Scenario 1: PhysX to Newton

Query: "Deploy this PhysX-trained Franka checkpoint in Newton and tell me whether it transfers."

Expected behavior:

- Requires a Newton-clean asset and matching policy contract before playback.
- Reproduces the PhysX source baseline.
- Runs deterministic open-loop parity before the PN closed-loop evaluation.
- Uses an explicit checkpoint and changes only the physics preset.
- Reports nominal and randomized multi-seed metrics separately.

Known failure modes:

- Starts tuning PPO before checking the target task and asset.
- Reports one successful video without a PP baseline or quantitative evidence.

## Scenario 2: Newton to PhysX

Query: "Can a policy trained with Newton run under PhysX?"

Expected behavior:

- Treats transfer as bidirectional and evaluates NN plus NP.
- Preserves observation normalization, action order, timing, commands, and play config.
- Checks contact timing, actuator response, terminations, and saturation.
- Avoids assuming that success in the forward direction proves reverse transfer.

Known failure modes:

- Only discusses PhysX-to-Newton.
- Compares different seeds, goals, checkpoints, or evaluation distributions.

## Scenario 3: Mimic Joint Mismatch

Query: "The target backend has one gripper DOF but my checkpoint outputs two finger actions."

Expected behavior:

- Identifies a checkpoint action-contract incompatibility.
- Recommends one driver plus passive follower for future portable training.
- Preserves legacy width/order and `last_action` semantics when the follower was already passive; reproduces active/active training semantics or requires retraining otherwise.
- Uses shared reset and leader-only randomization for the coupled pair.

Known failure modes:

- Silently drops or reorders one action.
- Drives and randomizes both sides independently.

## Scenario 4: Bang-Bang Target Behavior

Query: "The policy works in PhysX but alternates full positive and negative actions in Newton."

Expected behavior:

- Checks control period, action scale, armature, damping, inertia, and hard-stop reliance before solver iterations.
- Separates `velocity_limit` from `velocity_limit_sim`.
- Measures action sign changes and open-loop step response.
- Adds randomization only after correcting the nominal model.

Known failure modes:

- Hides the behavior with a tight solver clamp.
- Widens domain randomization around a known bad nominal model.

## Scenario 5: Observation Shape Changed

Query: "I removed body velocities for transfer robustness; can I deploy the old checkpoint?"

Expected behavior:

- States that observation width/order changes require retraining or an explicitly validated compatibility adapter.
- Preserves the old checkpoint's normalizer and history when evaluating the old contract.
- Recommends pose plus history for the new portable contract when velocities are unnecessary.

Known failure modes:

- Claims the checkpoint is backend-independent without qualifying the environment contract.
- Pads or truncates observations without validating semantics.

## Scenario 6: Domain Randomization For Transfer

Query: "What domain randomization should I add so policies transfer in both directions?"

Expected behavior:

- Requires a matched nominal model before randomization.
- Adds friction, payload mass/inertia, joint gains/friction/armature, actuator response, gravity, reset diversity, and observation noise one family at a time.
- Keeps mimic mechanisms coherent, avoids stacked gripper gain randomization, and accounts for different friction parameterizations.
- Replays the same effective parameter vector for paired failures instead of relying on equal seeds.
- Uses curriculum to reach final deployment difficulty and reports nominal and randomized four-cell results separately.

Known failure modes:

- Widens all distributions simultaneously without checking effective sampled values.
- Uses randomization to cover a known action, inertia, collision, or reset bug.
