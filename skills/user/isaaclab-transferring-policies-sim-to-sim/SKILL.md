---
name: isaaclab-transferring-policies-sim-to-sim
description: Transfers and evaluates Isaac Lab policies between PhysX and Newton MJWarp in both directions. Use when handling checkpoint compatibility, action or observation contract mismatches, mimic-joint drive differences, cross-backend control tuning, transfer-oriented domain randomization, four-cell PhysX/MJWarp experiments, or diagnosing reset, contact, normalization, and success gaps.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Transfer Policies Between PhysX And Newton

## When To Use

Read the [sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst) before running an experiment. It owns the Franka commands, environment-contract requirements, mimic-joint behavior, domain-randomization guidance, and four-cell protocol.

If the task cannot reset and step reliably in both backends, stop and use `isaaclab-preparing-assets-for-newton`.

## Workflow

1. **Freeze the source run.** Record the explicit checkpoint, training and play task IDs, RL library and config, seed, normalizer, backend, and same-backend result.
2. **Compare the resolved environment contract.** Require identical ordered actions and observations, tensor widths, history, frames, units, clipping, normalization, recurrent state, commands, `dt * decimation`, mechanism topology, resets, rewards, terminations, and success meaning. Treat a semantic change as a retraining boundary even when shapes match.
3. **Resolve mimic drives before playback.** MJWarp lowers the Franka mimic relation to an equality and drives the leader once. PhysX leaves the follower driveable; assigning gains to both fingers makes one logical gripper command produce two PD-drive contributions. Drive the leader and set follower stiffness/damping to zero. Preserve a legacy checkpoint's action width and order in both backends or retrain.
4. **Establish deterministic parity.** Disable randomization, reproduce one valid reset and command, and replay zero, isolated small, and fixed open-loop actions through first contact. Diagnose the first material divergence.
5. **Evaluate the source baseline and transfer.** Use the exact checkpoint and play config. Change only `physics=physx` or `physics=newton_mjwarp`.
6. **Complete the reverse direction.** Train with the same finalized contract in the other backend, then measure PP, PN, NN, and NP with the same seeds, goals, episode counts, and metrics.
7. **Correct control before adding robustness.** Preserve the policy period and compare actuator response, saturation, hard stops, damping, armature, effort, friction, and speed violations. Retune damping after any armature increase; do not hide bang-bang control with a solver clamp.
8. **Add domain randomization around the corrected nominal model.** Introduce plausible friction, payload mass/inertia, joint gains/friction/armature, gravity, actuator response, reset geometry, and observation noise one family at a time. Keep coupled mechanisms coherent and use curriculum only to reach the final deployment distribution.
9. **Report comparable results.** Separate deterministic nominal and randomized evaluations. Include success, return, episode length, terminations, saturation/sign changes, speed violations, contacts, non-finite counts, seeds, and resolved configs.

## Validation

Require:

- both backends pass reset and step smokes;
- the checkpoint and normalizer load without contract adaptation;
- deterministic replays have no unexplained first-step or first-contact divergence;
- PP, PN, NN, and NP use explicit checkpoints and matched evaluation settings; and
- nominal and randomized results are reported separately.

## Maintenance

Keep this skill synchronized with the sim-to-sim how-to. Update the how-to first when transfer guidance or Franka commands change.

## References

- Read [reference.md](reference.md) for compact contract, mimic-drive, experiment, and triage tables.
- Read [examples.md](examples.md) for common transfer cases.
- Use [evaluations.md](evaluations.md) when maintaining or testing this skill.
- Use the [asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) for physical-model failures.
- Use the [RL train and play guide](../../../docs/source/overview/reinforcement-learning/rl_existing_scripts.rst) for entry-point details.
