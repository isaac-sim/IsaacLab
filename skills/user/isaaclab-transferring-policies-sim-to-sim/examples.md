# Sim-To-Sim Policy Transfer Examples

## PhysX-Trained Manipulation Policy to Newton

Input: a PhysX-trained gripper policy must run in Newton.

Expected workflow:

1. Use the asset migration skill to validate inertials, colliders, coupled fingers, gains, damping, and armature.
2. Confirm the play task has the same observation/action descriptors and normalizer state as training.
3. Disable randomization and compare shared-reset open-loop trajectories through first contact.
4. Reproduce the PhysX checkpoint baseline, then run the same checkpoint with `physics=newton_mjwarp`.
5. Diagnose reset/contact/control gaps before broadening domain randomization.
6. Report PP and PN results over the same seeds.

## Newton-Trained Locomotion Policy to PhysX

Input: a Newton-trained locomotion policy must be evaluated in PhysX.

Expected workflow:

1. Preserve the same command distribution, observation history, action scale, policy period, and joint order.
2. Compare nominal open-loop joint responses without terrain contacts, then with flat-ground contacts.
3. Reproduce the Newton baseline and evaluate the explicit checkpoint with `physics=physx`.
4. Check action saturation, joint-speed violations, foot contact timing, termination causes, and success/velocity tracking.
5. Report NN and NP results, then add randomized terrain/material tests separately.

## Mimic-Joint Action Count Differs

Input: PhysX training exposed two gripper joint actions, but the Newton asset models one coupled mechanical DOF.

Expected workflow:

1. Stop direct checkpoint transfer; the action contract differs.
2. Prefer one driven action plus a passive follower for newly trained portable policies.
3. If the follower was already passive during training, keep both action entries and make the follower entry consistently ineffective in both backends. If both targets were active, reproduce that contract or declare the checkpoint incompatible.
4. Keep ordered descriptors identical, preserve any `last_action` observation width, and document the compatibility adapter.
5. Retrain before removing the legacy follower action.

## Nominal Transfer Works but Randomized Transfer Fails

Input: the same checkpoint succeeds in both backends nominally but fails under training randomization in Newton.

Expected workflow:

1. Re-enable one randomization family at a time.
2. Inspect effective sampled parameters in both backends.
3. Check Newton's single-friction-coefficient behavior and whether generic gain DR stacks with gripper-specific damping DR.
4. Validate every randomized geometry against the reset-clearance criteria.
5. Separate a modeling bug from insufficient policy robustness before widening distributions.
