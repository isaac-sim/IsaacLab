<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka Pour

This package provides one artifact-owned reset-dataset learning workflow,
`IsaacContrib-Franka-Pour`.

The reset-dataset task follows the same separation of concerns as Franka Stack:

- the actor receives a terminal-sparse task reward;
- one offline cache supplies a connected path through the task;
- a separate Boolean local-progress signal trains the reset sampler; and
- terminal particle transfer remains the policy-success and evaluation metric.

The actor receives one current timestep of robot, cup, target, gripper, and previous-action state,
plus compact permutation-invariant MPM summaries. Its 84 inputs contain no temporal stack or
constant padding. The asymmetric critic adds two task-state inputs for a total of 86.

The artifact is still required for the reset-dataset task because it is the reset curriculum, not
a cache of simulated fluid outcomes. It amortizes NewtonIK and static collision rejection, gives
every physical start a stable row ID for outcome-aware replay, and binds training runs to one
reproducible state distribution. Runtime reconstructs the same source-local particle lattice
for each row and resets the complete MPM solver state; it does not replay cached stress or particle
trajectories.

## Generate and train

The artifact is generated locally and is intentionally not stored in Git. From the repository
root, generate it once and then train with the task defaults:

```bash
uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0

uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 --max_iterations 3000
```

The first command takes about two minutes on an L40S-class GPU and creates the default
`datasets/franka_pour/reset_dataset.pt` artifact with 20,000 rows. Runtime resolves that relative
path from the repository root, so launching training from another working directory does not
change which artifact is loaded. The script exposes the same 14 connected reset phases and phase
quotas as the reference distribution. Newton IK, joint-limit, table/object, self-collision,
grasp-seating, and particle-workspace checks reject unsafe proposals; there is no second validation
or artifact-promotion command.

The generator and task load the Franka Pour cups USD from the standard Isaac Lab Nucleus asset
root. Set `ISAACLAB_FRANKA_POUR_CUPS_USD_PATH` only to use a compatible local copy instead. The
generated database itself does not need to be published.

The `datasets/` directory is ignored by Git. The payload always validates its stored content digest,
so no digest override is needed for ordinary training or playback. For an exactly reproducible run,
the generator also prints the optional `env.reset_dataset_content_sha256` pin. Publishing the
database is not required.

## Play

The canonical task is registered with import-light string entry points. Registration does not load
the artifact; environment startup validates it when creating the environment. Play mode uses one
bounded sparse-grid world, disables observation corruption, freezes the sampler, and moves the
interactive viewer closer:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --checkpoint /path/to/model.pt \
  --device cuda:0 --visualizer newton_gl
```

## Learning contract

The actor reward contains only:

- `+5` on terminal particle-transfer success;
- `-1` on a non-timeout safety failure;
- `-1e-4` action magnitude; and
- `-1e-4` action-rate regularization.

An ordinary timeout is neutral. Nonfinite state, extreme rigid state, source/receiver overlap,
spill, particle out-of-bounds, and the configured collision guards remain terminal safety checks.
Success terminates on the first policy step with at least 70% of particles delivered to the
receiver.

The sampler does not use terminal success as its learning signal. For each restored row, a
non-terminating context asks whether the policy made meaningful forward physical progress from
that starting state. Near the end of the manifold it requires actual particle-transfer success.
The result latches after a safe milestone crossing and is recorded when that episode finishes;
spill, overlap, out-of-bounds, collision, or abnormal terminal transitions cannot create positive
curriculum evidence. This curriculum-only Boolean never contributes policy reward.

Each row retains its exact 50 most recent local-progress outcomes. Adaptive priority is
`sqrt(p * (1 - p)) + epsilon`, so uncertain rows near 50% progress are emphasized without starving
rows at 0% or 100%. A shuffled cyclic stream receives 50% of assignments throughout training,
which keeps persistently difficult reaching and grasp-acquisition rows represented alongside the
adaptive competence frontier. There are no runtime stages or region probabilities.

The sampler reports its effective pool size, concentration, and cyclic-replay fraction. Offline
reset regions remain in the artifact as generation provenance, but runtime does not maintain a
second set of per-region counters. Evaluate frozen policies separately rather than inferring
performance from training-time samples.

Each worker applies completed outcomes immediately to its own sampler. In distributed training,
the curriculum therefore adapts independently on each rank while PPO still learns from all rank
rollouts. Actor and critic empirical observation normalization is disabled because task
observations are physically scaled at their source.
