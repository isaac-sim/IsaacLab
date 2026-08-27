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
reproducible rigid-state distribution. Runtime reuses one configured source-local particle lattice
across reset rows and resets the complete MPM solver state; it does not replay cached stress or
particle trajectories.

## Train and regenerate

The task downloads the canonical 20,000-row artifact from the standard Isaac Lab asset root on
first use and reuses the local asset cache afterward. Training therefore needs no artifact setup:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 --max_iterations 3000
```

The checked-in generator remains the executable reference for creating or modifying the reset
distribution. It takes about two minutes on an L40S-class GPU and writes a 20,000-row local artifact:

```bash
uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0

uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 \
  env.reset_dataset_path=datasets/franka_pour/reset_dataset.pt
```

Relative local paths resolve from the repository root. The generator exposes the same 14 connected
reset phases and phase quotas as the published distribution. Newton IK, joint-limit, table/object,
self-collision, grasp-seating, and particle-workspace checks reject unsafe proposals; there is no
second validation or artifact-promotion command.

The generator and task load the Franka Pour cups USD from the standard Isaac Lab Nucleus asset
root. Set `ISAACLAB_FRANKA_POUR_CUPS_USD_PATH` only to use a compatible local copy instead. The
published reset artifact lives alongside those assets at `Contrib/MPM/Pour/reset_dataset.pt`.
Setting `ISAACSIM_ASSET_ROOT` redirects all three dependencies to a compatible local or self-hosted
asset tree.

The `datasets/` directory is ignored by Git. Both downloaded and locally generated payloads validate
their stored content digest, so no digest override is needed for ordinary training or playback. For
an exactly reproducible custom run, the generator also prints the optional
`env.reset_dataset_content_sha256` pin.

## Fill and success levels

`env.source_fill_level` is the initial media height as a fraction of the source-cup cavity. The
default `0.70` creates a 7×7×15 lattice (735 particles); values are quantized to complete lattice
layers and must lie in `(0, 1]`. This analytic fill volume is non-colliding. The visible cup mesh
remains a particle-only collider so it can physically contain the media. The receiver uses a solid
analytic box only for robot contact; it is invisible to particles, which continue to collide with
the hollow receiver mesh.

The particle generator and MPM solver both use a 15 mm voxel. Setting
`particles_per_cell=3` targets three particles per solver cell along each axis (27 per 3D cell),
while the bounded sparse grid, particle-backed automatic warm start, and two MPM entry substeps keep
the configuration CUDA-graph compatible. Collider projection remains disabled because manager-level
post-step projection is not supported inside a coupled MPM entry; contact is resolved by the MPM
solve.

The proxy presents the supported or grasp-constrained 50 g source cup as 50 kg to the MPM collision
view. This prevents the roughly 140 g payload from producing unphysical rigid-cup recoil without
changing the cup's authored mass or the robot dynamics in MJWarp.

`env.pour_target_frac` independently sets the fraction of the initial payload that must be inside
the receiver. Success is normalized by the live particle count, so changing the source fill does
not require changing the success check. For example, this starts half-full and requires 80% of that
payload to be delivered:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour --device cuda:0 \
  env.source_fill_level=0.50 env.pour_target_frac=0.80
```

## Play

The canonical task is registered with import-light string entry points. Registration does not
download the artifact; environment startup retrieves and validates it when creating the environment.
Play mode uses one bounded sparse-grid world, disables observation corruption, freezes the sampler,
and moves the interactive viewer closer:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --device cuda:0 --visualizer kit
```

Use `--visualizer newton_gl` instead for the lightweight Newton viewer. No external callback or
particle-setting override is required for either visualizer.

## Learning contract

The actor reward contains only:

- `+5` on terminal particle-transfer success;
- `-1` on a non-timeout safety failure;
- `-1e-4` action magnitude; and
- `-1e-4` action-rate regularization.

An ordinary timeout is neutral. Nonfinite state, extreme rigid state, source/receiver overlap,
spill, particle out-of-bounds, and the configured collision guards remain terminal safety checks.
Success terminates on the first policy step with the configured fraction of particles delivered to
the receiver (70% by default).

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
