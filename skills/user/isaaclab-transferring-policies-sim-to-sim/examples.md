# Sim-To-Sim Policy Transfer Examples

## PhysX-Trained Franka To MJWarp

Keep one MDP for the registered task, resolve and audit each backend's `PresetCfg`, and ensure the task can train in both engines. Disable the second PhysX finger drive, match nominal actuator behavior, reproduce PP with the explicit checkpoint, and then run PN with only the physics backend changed.

## MJWarp-Trained Franka To PhysX

Use the same contract and control settings, reproduce NN with the explicit checkpoint, and run NP in PhysX. Keep the training and inference task IDs from the how-to.

## Duplicate PhysX Finger Drive

If one logical gripper command produces more effort in PhysX, check whether both fingers received nonzero stiffness and damping. Drive `panda_finger_joint1`; make `panda_finger_joint2` passive while retaining it for the mimic constraint.

## Domain Randomization

Randomize the documented friction, payload, joint, armature, gravity, actuator-response, reset, and observation families around a corrected nominal model. Use curriculum only to reach the final deployment distribution and keep deterministic nominal evaluation separate.
