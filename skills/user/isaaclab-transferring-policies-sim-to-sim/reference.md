# Sim-To-Sim Policy Transfer Reference

## Contents

- Source-of-truth rule
- Preconditions
- Contract snapshot
- Four-cell experiment
- Deterministic parity protocol
- Domain randomization protocol
- Metrics and acceptance criteria
- Failure triage
- Lessons from PR 6457

## Source-of-Truth Rule

Read the [official sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst) before planning the experiment. Use the separate [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) when the physical model is not yet Newton-clean.

## Preconditions

Do not start checkpoint playback until:

- the exact task can spawn, reset, sense, and step in PhysX and Newton;
- the target asset is MJWarp-clean under `isaaclab-preparing-assets-for-newton`;
- deterministic zero-action and small-action rollouts remain finite;
- the play configuration preserves the training action and observation spaces;
- a source-backend checkpoint baseline is reproducible.

## Contract Snapshot

Compare the resolved values, not only config source text:

| Contract | Must match |
| --- | --- |
| Actions | term names, ordered joints, width, scale, offset, clipping, target type |
| Observations | group/term order, width, history, units, frames, clipping, corruption |
| Policy state | normalization statistics, recurrent weights, hidden-state initialization/reset, command inputs |
| Timing | `dt`, decimation, policy period, action hold behavior |
| Mechanism | ordered bodies/joints, active DOFs, mimic/equality constraints |
| Actuation | effort, rated velocity, solver clamp, stiffness, damping, friction, armature |
| Episode | reset distribution, commands, rewards, terminations, horizon |

Changing any tensor width or order is a retraining migration. A checkpoint that happens to load after semantic reordering is still incompatible.

PR 6457's Franka action term selected all joints, including both coupled fingers. The follower target remained in the action tensor but was ineffective because its actuator had zero stiffness and damping. Preserve that apparently redundant entry for checkpoints trained with this passive-follower contract, including its effect on a `last_action` observation. If both targets were active during training, reproduce that behavior or declare incompatibility. Use one driver action only when defining a new contract and retraining.

## Four-Cell Experiment

Always measure:

| Training | Deployment | Label |
| --- | --- | --- |
| PhysX | PhysX | PP source baseline |
| PhysX | Newton | PN transfer |
| Newton | Newton | NN source baseline |
| Newton | PhysX | NP transfer |

Use the same seed set, goal set, episode count, nominal/randomized evaluation mode, and metric code for all cells. Compare PN to PP and NP to NN; also compare PP to NN to understand how source training differs.

Example commands:

```bash
./isaaclab.sh train --rl_library rsl_rl --task TRAIN_TASK physics=physx
./isaaclab.sh play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/physx_checkpoint.pt physics=physx
./isaaclab.sh play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/physx_checkpoint.pt physics=newton_mjwarp

./isaaclab.sh train --rl_library rsl_rl --task TRAIN_TASK physics=newton_mjwarp
./isaaclab.sh play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/newton_checkpoint.pt physics=newton_mjwarp
./isaaclab.sh play --rl_library rsl_rl --task PLAY_TASK \
  --checkpoint /absolute/path/to/newton_checkpoint.pt physics=physx
```

Do not use `latest` in a comparison report. It can select a different run.

For the maintained Franka example, train with `Isaac-Lift-Franka` and infer with
`Isaac-Lift-Franka-Play`. Follow the concrete PP, PN, NN, and NP commands in the
[official sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst).
Both source runs use the `dexsuite_franka` experiment directory, so select an explicit
`model_<iteration>.pt` path instead of `--checkpoint latest`.

## Deterministic Parity Protocol

1. Disable startup/reset/interval randomization and observation corruption.
2. Select one valid reset state and reproduce positions, velocities, commands, observation history, previous actions, terminations, and recurrent hidden-state initialization; express world positions relative to the environment origin.
3. Apply zero actions for several policy steps.
4. Apply isolated small steps to each action dimension.
5. Replay a fixed multi-joint action sequence through first contact.
6. Log the same arrays at the same policy-step boundary.
7. Locate the first material divergence instead of comparing only episode return.

Log joint position/velocity, action target, applied effort, link/object pose, contact count/force, reward components, termination flags, and non-finite masks. Compare control-rate samples; solver substeps need not align one-to-one.

Large divergence at step 0 usually indicates reset, topology, default-state, or action-contract mismatch. Large divergence at the first contact usually indicates collision, friction, solver capacity, or contact-sensor differences. Gradual closed-loop divergence with similar open-loop response points toward observations, normalization, or policy robustness.

## Domain Randomization Protocol

Add one family at a time after nominal parity:

1. object mass and inertia;
2. robot/object friction and restitution;
3. joint stiffness, damping, friction, and armature;
4. actuator response such as gripper closing speed;
5. gravity and command/reset distributions;
6. observation noise.

Keep nominal values inside every distribution. Check effective backend behavior: current Newton material randomization uses one friction coefficient, while PhysX exposes separate static/dynamic values and buckets. Use backend presets when a parameter does not map directly. For a paired failure reproduction, save and apply the same effective parameter vector; a shared seed is insufficient when event ordering, devices, or backend mappings differ.

For coupled joints, sample shared reset positions and randomize the driven side. Do not randomize passive follower damping independently. Exclude a gripper from generic gain randomization if a dedicated closing-speed term already owns its damping.

Curriculum can interpolate gravity, observation noise, or termination bounds from an easy distribution to the final one. Evaluate at the final difficulty. Run a separate nominal setting so random draws do not obscure engine differences.

## Metrics and Acceptance Criteria

Report distributions, not only averages:

- success rate and confidence interval;
- return and per-term reward statistics;
- episode length and termination-cause histogram;
- action saturation and consecutive sign-change rate;
- joint-speed limit violations and hard-stop contacts;
- contact count, peak/mean force, penetration indicators;
- object drops and reset rejection rate;
- non-finite observations, rewards, actions, or state;
- time-to-first trajectory tolerance violation.

Set acceptance thresholds before running the target backend. A suitable threshold depends on task risk and baseline variance; do not invent a universal percentage.

## Failure Triage

| Symptom | First checks |
| --- | --- |
| Checkpoint shape error | action/observation width, history, actor config |
| Loads but acts on wrong joints | ordered action descriptor, converted joint order |
| Immediate gripper snap | independent mimic reset, duplicate equality, active follower drive |
| Bang-bang control | damping, armature, action scale, policy period, hard stops |
| Target never triggers speed termination | `velocity_limit_sim` clamps below rated `velocity_limit` |
| First contact explodes | collision mesh/scale, invalid reset, inertia, contact capacity |
| Nominal works, randomized fails | distribution mapping, stacked gain DR, invalid geometry bank |
| One direction transfers, reverse does not | compare source-policy strategies and source baselines; do not assume symmetry |
| Policy drifts despite open-loop parity | observations, body velocities, contact features, normalizer state |

Large MJWarp velocity is a physical-model diagnostic before it is a robustness problem. Low generalized inertia amplifies drive, constraint, and contact impulses; physically justified armature conditions articulated coordinates. Retune damping after changing armature to avoid bang-bang control. For zero-gravity object behavior and the distinction between articulated-object armature and plain rigid-body inertia/damping, return to the asset migration guide.

## Lessons from PR 6457

The Franka/Kuka Dexsuite transfer combined several changes:

- a task-local multi-physics Franka asset with identified inertials and authored finger coupling;
- calibrated per-joint effort, velocity, stiffness, damping, friction, and armature;
- driven/passive finger roles, shared coupled-joint reset, and driver-only closing-speed randomization;
- increased armature and damping to reduce engine-sensitive acceleration and bang-bang behavior;
- collision-valid reset banking so solvers did not repair different penetrations;
- pose-only fingertip state with history rather than body velocities;
- material, payload, gain, friction, gravity, actuator-response, and observation randomization;
- an adaptive curriculum and final-difficulty play configuration;
- Newton contact capacity, contact formulation, iterations, CCD, and substeps tuned for manipulation;
- rated velocity limits separated from solver clamps by dependency PR 6481.

Treat asset, actuator, timing, action, observation, reset, command, reward, termination, and training-randomization semantic changes as possible retraining boundaries even when tensor shapes match. The transferable artifact is the policy trained after the unified environment contract is established.
