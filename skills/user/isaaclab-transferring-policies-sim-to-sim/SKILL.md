---
name: isaaclab-transferring-policies-sim-to-sim
description: Transfers Isaac Lab policies between PhysX and Newton MJWarp in both directions. Use when checking PresetCfg-resolved task and checkpoint compatibility, handling Franka mimic-joint drive differences, matching cross-backend actuator behavior, adding transfer-oriented domain randomization, or running PhysX/PhysX, PhysX/Newton, Newton/Newton, and Newton/PhysX evaluations.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Transfer Policies Between PhysX And Newton

## When To Use

Read the [sim-to-sim how-to](../../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst) first. This skill follows that page in the same order. Before transfer, make the asset and task MJWarp-ready with `isaaclab-preparing-assets-for-newton`.

## Workflow

1. **Task readiness and checkpoint compatibility.** Keep one MDP for the same registered task. Resolve the explicit `isaacsim_physx` and `newton_mjwarp` backend alternatives from each `PresetCfg`, using them for intentional physics, asset, and control differences without silently changing policy-facing MDP terms. If an MDP-term preset differs, restore one checkpoint contract or treat it as a different task and retrain. Require successful training in both engines and exact action, observation, policy-state, timing, mechanism, and episode contracts.
2. **Mimic-joint action nuance.** Newton MJWarp preserves both Franka finger coordinates but creates one active drive: the leader is driven and an equality moves the follower. PhysX preserves the mimic coupling while leaving the follower driveable. If one logical command targets both fingers and both have nonzero gains, PhysX applies two PD-drive contributions. Drive `panda_finger_joint1` and set `panda_finger_joint2` stiffness and damping to zero.
3. **Transferring control behavior.** Match nominal actuator response before tuning the policy. Distinguish rated and solver velocity limits, use per-joint effort, gains, friction, and armature, preserve `dt * decimation`, keep targets away from hard stops, and monitor saturation and action sign changes. Increase damping to prevent bang-bang control and retune it after increasing armature.
4. **Introducing domain randomization.** Randomize plausible robot/object friction, object mass/inertia, joint gains/friction, joint armature, gravity, actuator response, reset pose/geometry, and observation noise. Keep inertia valid and coupled mechanisms coherent. If transfer needs extreme ranges, revisit the nominal model. Use curriculum when the final distribution blocks learning, promote to final deployment difficulty, and keep a separate deterministic nominal evaluation.
5. **Validate the full matrix.** Evaluate PP, PN, NN, and NP. For each source policy, reproduce the same-backend baseline and deploy the exact checkpoint in the other backend.
6. **Run the Franka lift transfer.** Use `Isaac-Lift-Franka` for both training and inference, selecting `isaacsim_physx` for the PhysX runs. The play entry point applies the environment's `play_mode` overrides automatically. Follow the PhysX-to-MJWarp and MJWarp-to-PhysX commands in the how-to.

## Validation

Require a task trainable in both backends, exact environment-contract equality, one active Franka finger drive in each backend, matched nominal control behavior, plausible randomization, and all four training/deployment combinations.

## Maintenance

Keep this skill synchronized section-for-section with the sim-to-sim how-to, including its Franka commands.

## References

- [Compact reference](reference.md)
- [Examples](examples.md)
- [Evaluations](evaluations.md)
- [Asset migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst)
- [RL train and play guide](../../../docs/source/overview/reinforcement-learning/rl_existing_scripts.rst)
