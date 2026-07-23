---
name: isaaclab-transferring-policies-sim-to-sim
description: Transfers and evaluates Isaac Lab reinforcement-learning policies between PhysX and Newton in both directions. Use when deploying a PhysX-trained checkpoint in Newton, deploying a Newton-trained checkpoint in PhysX, designing a sim-to-sim experiment, preserving checkpoint action and observation contracts, adding transfer-oriented domain randomization, or diagnosing cross-backend success, contact, control, reset, or normalization gaps.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Transferring Policies Sim-To-Sim

## When To Use

Use this skill after the task and assets can reset and step under both PhysX and Newton. It covers PhysX-to-Newton and Newton-to-PhysX policy transfer, including nominal parity, domain-randomized robustness, checkpoint compatibility, and result reporting.

Read the [official sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst) first. If the target backend cannot parse or control the asset, use `isaaclab-preparing-assets-for-newton` and the separate asset migration guide before evaluating a policy.

## Workflow

1. Freeze the source run. Record the explicit checkpoint, task and play-task IDs, RL library, seed, resolved agent config, observation normalizer, training backend, and source success rate.
2. Prove the policy interface is identical across backends: action width/order/scale/offset, observation width/order/history/frames/units, command interface, control period, recurrent weights, and hidden-state initialization/reset. Stop and retrain or version an adapter if any contract differs.
3. Audit mimic/equality joints. Use one portable action per mechanical DOF by default. If a legacy checkpoint contains an already-passive follower action, preserve the same width, ordering, and `last_action` semantics in both backends until retraining. If both joints were active during training, reproduce that contract or declare incompatibility; do not silently make one passive.
4. Disable evaluation randomization and replay the same valid reset state plus open-loop actions in PhysX and Newton. Investigate first-step, first-contact, hard-stop, non-finite, or termination divergence before closed-loop playback.
5. Evaluate the source checkpoint in its training backend. Confirm the recorded baseline can be reproduced with the same play config and normalizer.
6. Change only `physics=physx` or `physics=newton_mjwarp` and evaluate the same checkpoint in the target backend. Use an explicit checkpoint path.
7. Train a policy in the other backend with the same finalized environment contract, RL configuration, training budget, and evaluation protocol, then repeat the same-backend and cross-backend evaluations. Complete all four train/deploy cells rather than reporting one successful direction. Preserve legacy results separately if actuator, timing, reset, reward, termination, or randomization semantics changed.
8. Diagnose gaps in this order: contract mismatch, invalid resets, asset mass/collision/topology, mimic-joint behavior, control period, actuator limits/gains/armature/damping, contacts and solver capacity, observations/normalization, rewards/terminations, then policy robustness. Low-inertia MJWarp coordinates can amplify impulses into large velocities; use only physically justified inertia or armature changes and retune damping before treating the behavior as a policy failure.
9. Add transfer-oriented domain randomization only around a corrected nominal model. Cover friction, payload mass/inertia, joint gains/friction/armature, gravity, actuator response or gripper closing speed, reset geometry, and observation noise as applicable. For paired failures, replay the same effective parameter vector; identical seeds do not guarantee identical draws or backend mappings.
10. Keep coupled mechanisms coherent during randomization. Do not stack generic gain randomization on a separately randomized gripper or randomize a passive follower independently.
11. Use curriculum when the full transfer distribution blocks learning: promote gravity, observation noise, reset difficulty, and bounds toward the final deployment distribution. Evaluate both the final robustness distribution, including actuator uncertainty when applicable, and a separate deterministic nominal setting.
12. Report source and target success rate, return, episode length, termination causes, action saturation/sign changes, joint-speed violations, contact statistics, non-finite counts, seeds, and resolved backend configs. Separate deterministic nominal results from randomized robustness results.

## Validation

Require all of these gates:

1. Both backends pass small random-agent reset/step smoke tests.
2. The checkpoint loads with the original observation-normalizer state.
3. Ordered action and observation descriptors match exactly.
4. Seeded zero-action and open-loop action replays remain finite and have no unexplained first-step impulse.
5. Mimic joints share reset state and leader/follower actuation semantics.
6. Rated velocity limits remain task-level limits; solver clamps do not mask violations.
7. The same checkpoint is evaluated in both backends with only the physics preset changed.
8. Both cross-backend directions are measured against same-backend baselines over multiple seeds.
9. Nominal and randomized evaluations are reported separately.

For skill changes, run:

```bash
./isaaclab.sh -p tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `docs/source/how-to/transfer_policies_between_physx_and_newton.rst`, `docs/source/overview/reinforcement-learning/rl_existing_scripts.rst`, the train/play entry points under `source/isaaclab_rl/isaaclab_rl/entrypoints/`, actuator semantics under `source/isaaclab/isaaclab/actuators/`, and the bidirectional-transfer patterns under `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/`. Update the official how-to first when transfer guidance changes.

## References

- [Reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Official sim-to-sim transfer how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst)
- [Official PhysX-to-Newton asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst)
- [Prepare assets for Newton skill](../prepare-assets-for-newton/SKILL.md)
- [Domain randomization skill](../domain-randomization-events/SKILL.md)
- [Training RL agents skill](../train-rl-agents/SKILL.md)
- [Debug RL training skill](../debug-rl-training/SKILL.md)
- [RL train and play guide](../../../docs/source/overview/reinforcement-learning/rl_existing_scripts.rst)
