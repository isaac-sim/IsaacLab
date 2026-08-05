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

## Reset dataset artifact

The calibrated 20,000-row v12 production artifact is generated and supplied externally. Its
expected content digest is
`93e1f86a5145812a412c4e3c0d5873bff765bce9d236e4e2f0893e190c4a75bf`. Place the artifact at
`datasets/franka_pour/reset_dataset.pt`; the environment validates its schema, provenance, pinned
content digest, and task contract when it starts. The `datasets/` directory is intentionally
ignored by Git.

The canonical artifact download location has not yet been published. Consequently, a clean
checkout cannot run this task until the artifact is supplied separately. Publishing that location
is required before treating the task as a reproducible user-facing workflow.

## Train or play

The canonical task is registered with import-light string entry points. Registration does not load
the artifact; environment startup validates the configured dataset path and content hash. The
configuration can also be constructed directly:

```python
from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaPourResetDatasetPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg

env_cfg = FrankaPourResetDatasetEnvCfg()
env_cfg.reset_dataset_path = "datasets/franka_pour/reset_dataset.pt"
agent_cfg = FrankaPourResetDatasetPPORunnerCfg()
```

The configuration pins the production digest by default. Set
`env_cfg.reset_dataset_content_sha256` explicitly only when intentionally using another trusted,
matching artifact.

Reset-dataset play mode uses one bounded sparse-grid world, disables observation corruption,
freezes the sampler, and moves the interactive viewer closer.

Train and play the canonical task through the unified Isaac Lab commands:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 \
  --max_iterations 3000 \
  env.reset_dataset_path="$PWD/datasets/franka_pour/reset_dataset.pt"

uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --checkpoint /path/to/model.pt \
  --device cuda:0 --viz newton \
  env.reset_dataset_path="$PWD/datasets/franka_pour/reset_dataset.pt"
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

The sampler reports its effective pool size, concentration, cyclic-replay fraction, first-sweep
coverage, and the assignment fraction for each offline reset region. Completed episodes also log
terminal success, local progress, reaching, bilateral grasp, and lift rates by reset region. The
regions are diagnostics for the sampled distribution; they do not gate policy behavior or change
the sampler objective. Evaluate frozen policies separately rather than inferring performance from
training-time samples.

Each worker applies completed outcomes immediately to its own sampler. In distributed training,
the curriculum therefore adapts independently on each rank while PPO still learns from all rank
rollouts. Actor and critic empirical observation normalization is disabled because task
observations are physically scaled at their source.
