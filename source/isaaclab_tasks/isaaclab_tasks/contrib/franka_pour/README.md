<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka Pour reset datasets

The reset-dataset task uses a two-step offline pipeline. Generation proposes and statically screens
an oversampled pool of 14,000 grasping states (including 2,000 near-pour states) and 12,000
non-grasping states. Validation restores every candidate in the real task, simulates it, requires
near-pour states to deliver at least 30% without excessive spill, and selects exactly 10,000 valid
states from each category. The normal training environment rejects the intermediate candidate file.
Generation and validation default to the same reset-dataset task configuration; custom ``--task``
overrides must match in both commands because the cache records a strict physics contract.

From the repository root, run:

```bash
./isaaclab.sh -p scripts/tools/generate_franka_pour_reset_dataset.py \
  --device cuda:0 --viz none
./isaaclab.sh -p scripts/tools/validate_franka_pour_reset_dataset.py \
  --device cuda:0 --viz none
```

If validation reports an insufficient quota, rerun generation with larger
`--grasping_count`, `--non_grasping_count`, or `--near_pour_grasp_count` values.

This produces:

- `datasets/franka_pour/reset_dataset_candidates.pt`: oversampled intermediate proposals; never
  use for training.
- `datasets/franka_pour/reset_dataset.pt`: dynamically validated production data.
- `datasets/franka_pour/reset_dataset_validation.json`: compact validation report.

`datasets/` is intentionally ignored by Git. Treat datasets as external build artifacts, preserve
the content SHA-256 printed by the validator with experiment metadata, and distribute the `.pt`
file separately from source changes.

Train the reset-dataset task and pin the exact artifact with:

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Pour-Franka-Reset-Dataset-v0 \
  --num_envs 2048 --logger wandb --video \
  env.reset_dataset_content_sha256=<content-sha256>
```

`--video` records one environment on rank zero. The traditional procedural task remains available
as `Isaac-Pour-Franka-v0` and does not require a dataset.

The former `Reset-Mixture` task and Python names remain as deprecated lookup aliases for one
release. They select this reset-dataset implementation; they do not make its 8-action
relative-joint policy compatible with older 7-action Cartesian-IK checkpoints.

The adaptive reset sampler is intentionally process-local during distributed training: each rank
learns from its own completed episodes and rank zero supplies the logged curriculum metrics. RSL-RL
checkpoints currently restore the policy and optimizer, not this transient sampling history, so a
resumed run starts the sampler again from its configured initial frontier. The reusable sampler
still exposes `state_dict()` for a future generic environment-state checkpoint integration.
