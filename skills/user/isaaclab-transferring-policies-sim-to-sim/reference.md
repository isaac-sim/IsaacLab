# Sim-To-Sim Policy Transfer Reference

Use this after reading the [official how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst).

## Environment Contract

| Area | Must match |
| --- | --- |
| Actions | term order, ordered targets, width, type, scale, offset, clipping |
| Observations | group/term order, width, history, units, frames, corruption |
| Policy state | normalizer, recurrent state and reset, commands, actor inputs |
| Timing | `dt`, decimation, policy period, action hold |
| Mechanism | ordered bodies/joints, active DOFs, mimic/equality coupling |
| Episode | reset/command distributions, rewards, terminations, horizon, success |

Changing tensor order or meaning is a retraining migration even when the checkpoint loads.

## Franka Mimic-Drive Rule

| Backend | Representation | Consequence |
| --- | --- | --- |
| Newton MJWarp | Mimic relation becomes an equality; the leader has the active drive. | One logical gripper command contributes through one PD drive. |
| PhysX | Native mimic coupling keeps the follower driveable. | A wildcard actuator with gains on both fingers contributes through two PD drives. |

For a portable model, drive `panda_finger_joint1` and give `panda_finger_joint2` zero stiffness and damping. The action configuration—not the number of joint coordinates or constraints—sets checkpoint width. Preserve the exact legacy width, order, and any `last_action` meaning in both backends; otherwise retrain. Never silently drop or reorder a finger action.

## Four-Cell Experiment

| Training | Deployment | Label |
| --- | --- | --- |
| PhysX | PhysX | PP source baseline |
| PhysX | MJWarp | PN transfer |
| MJWarp | MJWarp | NN source baseline |
| MJWarp | PhysX | NP transfer |

Use explicit checkpoint paths, not `latest`. Keep seeds, goals, episodes, evaluation mode, and metric code fixed. Use `Isaac-Lift-Franka` for training and `Isaac-Lift-Franka-Play` for inference; copy the exact PP/PN/NN/NP commands from the official how-to.

## Deterministic Parity

1. Disable randomization and corruption.
2. Reproduce reset state, commands, history, previous action, and recurrent state.
3. Replay zero actions, isolated action steps, then a fixed sequence through contact.
4. Compare control-rate joint/object state, targets, efforts, contacts, rewards, and terminations.
5. Locate the first material divergence.

Step-zero divergence points to reset, topology, defaults, or contract. First-contact divergence points to collision, friction, capacity, or sensing. Gradual closed-loop divergence after open-loop parity points to observations, normalization, or policy robustness.

## Domain Randomization

Add one plausible family at a time:

- robot/object friction and restitution;
- payload mass and inertia;
- joint stiffness, damping, friction, and armature;
- actuator response, including gripper closing speed;
- gravity, commands, reset pose, and geometry; and
- observation noise.

Keep nominal values inside each distribution. Newton and PhysX friction events do not map one-to-one, so compare effective parameters rather than seeds alone. Share coupled-joint reset samples, randomize the driven side, and avoid stacking generic gain randomization with dedicated gripper response randomization. Evaluate a deterministic nominal setting separately from the final randomized distribution.

## Metrics And Triage

Report success, return, episode length, termination causes, action saturation/sign changes, rated-speed violations, contact statistics, object drops, reset rejection, and non-finite state.

| Symptom | First checks |
| --- | --- |
| Checkpoint shape error | action/observation width, history, actor config |
| Loads but controls wrong joints | ordered action targets and converted joint order |
| Gripper force differs | active follower drive, wildcard actuator, shared mimic reset |
| Bang-bang actions | damping, armature, action scale, policy period, hard stops |
| Speed termination differs | explicit task check; PhysX clamp versus unenforced MJWarp limits |
| First contact explodes | collision, scale, reset penetration, inertia, capacity |
| Nominal passes, randomization fails | effective distribution mapping, stacked events, invalid geometry |
| Open loop matches, policy drifts | observations, body/contact features, normalizer |
| Only one direction transfers | compare PP with PN and NN with NP; transfer is not symmetric |

Treat large MJWarp velocity as a model/control diagnostic before a robustness problem. Return to the asset migration guide for armature, damping, zero-gravity, and rigid-object rules.
