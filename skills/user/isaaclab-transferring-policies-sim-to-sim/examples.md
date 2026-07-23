# Sim-To-Sim Policy Transfer Examples

## PhysX-Trained Franka To MJWarp

1. Validate the task and asset in both backends.
2. Match the policy contract and make the second PhysX finger drive passive.
3. Reproduce deterministic open-loop and PP checkpoint behavior.
4. Run PN with the same checkpoint and only the physics preset changed.
5. Report PP and PN over the same seeds before adding randomization.

## MJWarp-Trained Locomotion To PhysX

Preserve commands, observation history, joint order, action scale, and policy period. Compare open-loop response, then contact timing. Reproduce NN and run NP with an explicit checkpoint; report saturation, speed violations, terminations, and tracking.

## Finger Action Contract Differs

The drive graph and checkpoint width are separate. Disable the second PhysX finger drive, but preserve the checkpoint's ordered action tensor and `last_action` semantics. If matching semantics requires dropping or reordering an action, retrain or implement and validate an explicit compatibility adapter.

## Nominal Passes, Randomization Fails

Re-enable one family at a time, compare effective backend parameters, and check friction mapping, stacked gripper/gain events, coupled-joint coherence, and randomized reset geometry. Fix model or event bugs before widening distributions.
