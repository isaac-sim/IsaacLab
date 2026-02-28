# MDP Warp-First Migration: Gap Analysis

> Updated 2026-02-18. Tracks the stable -> experimental (warp-first) conversion status of every
> public MDP term and every manager-based task.

---

## Table of Contents

1. [Testing Requirements](#testing-requirements)
2. [Shared MDP Term Catalog](#shared-mdp-term-catalog)
3. [Per-Task Gym ID Migration Table](#per-task-gym-id-migration-table)
4. [Per-Task MDP Term Usage](#per-task-mdp-term-usage)
5. [Custom (Task-Local) MDP Terms](#custom-task-local-mdp-terms)
6. [Shared Terms Not Used by Any Migrated Task](#shared-terms-not-used-by-any-migrated-task)
7. [Cross-Cutting Notes](#cross-cutting-notes)
8. [Key Warp Conversion Patterns](#key-warp-conversion-patterns)

---

## Testing Requirements

Every migrated MDP term **must** pass the following checks before it is considered complete:

### a) Parity Check: `stable == warp == warp-captured`

For each converted term, verify numerical equivalence across three execution modes:

1. **Stable** -- original torch-based term from `isaaclab.envs.mdp`
2. **Warp** -- experimental warp-first term from `isaaclab_experimental.envs.mdp` (eager launch)
3. **Warp Captured** -- same warp term executed inside a `wp.ScopedCapture` / `wp.capture_launch` graph

All three modes must produce identical results (within floating-point tolerance) for the same inputs.

### b) Dynamic Dependency Update Check

For the **warp-captured** execution path, verify that the captured graph produces correct results
even after upstream data changes between replay invocations. Specifically:

- Sim state buffers (joint_pos, root_vel, etc.) update between steps -- the captured graph must
  read the latest values from persistent pointers, not stale data baked into the graph.
- When a dependency (e.g., action buffer, sensor data, command output) is updated externally,
  the next `wp.capture_launch` must reflect the change.
- Resetting a subset of environments (via `env_mask`) must not corrupt state of non-reset environments.

---

## Shared MDP Term Catalog

Legend: **S** = Shared library (`isaaclab.envs.mdp`), **W** = Warp override exists in `isaaclab_experimental.envs.mdp`

### Observations (22 stable terms)

| # | Function/Class | Warp | Notes |
|---|---|---|---|
| 1 | `base_pos_z` | YES | Pure warp kernel |
| 2 | `base_lin_vel` | YES | Pure warp kernel |
| 3 | `base_ang_vel` | YES | Pure warp kernel |
| 4 | `projected_gravity` | YES | Pure warp kernel |
| 5 | `root_pos_w` | YES | Pure warp kernel |
| 6 | `root_quat_w` | YES | Pure warp kernel |
| 7 | `root_lin_vel_w` | YES | Pure warp kernel |
| 8 | `root_ang_vel_w` | YES | Pure warp kernel |
| 9 | `body_pose_w` | YES | Pure warp kernel |
| 10 | `body_projected_gravity_b` | YES | Pure warp kernel |
| 11 | `joint_pos` | YES | Pure warp kernel |
| 12 | `joint_pos_rel` | YES | Pure warp kernel with joint_mask |
| 13 | `joint_pos_limit_normalized` | YES | Pure warp kernel |
| 14 | `joint_vel` | YES | Pure warp kernel |
| 15 | `joint_vel_rel` | YES | Pure warp kernel with joint_mask |
| 16 | `joint_effort` | YES | Pure warp kernel |
| 17 | `last_action` | YES | Pure warp kernel |
| 18 | `generated_commands` | YES | `wp.from_torch` bridge (zero-copy) |
| 19 | `current_time_s` | YES | Pure warp kernel |
| 20 | `remaining_time_s` | YES | Pure warp kernel |
| 21 | `image` | **NO** | 4D tensor, per-type normalization. Deferred. |
| 22 | `image_features` | **NO** | PyTorch NN inference (ResNet/Theia). Not convertible. |

**Coverage: 20/22 (91%)**

### Rewards (22 stable terms)

| # | Function/Class | Warp | Notes |
|---|---|---|---|
| 1 | `is_alive` | YES | Pure warp kernel |
| 2 | `is_terminated` | YES | Pure warp kernel |
| 3 | `is_terminated_term` | YES | Class-based, reads `_term_dones_wp` + `time_outs_wp` |
| 4 | `lin_vel_z_l2` | YES | Pure warp kernel |
| 5 | `ang_vel_xy_l2` | YES | Pure warp kernel |
| 6 | `flat_orientation_l2` | YES | Pure warp kernel |
| 7 | `base_height_l2` | YES | Pure warp kernel |
| 8 | `body_lin_acc_l2` | YES | Pure warp kernel |
| 9 | `joint_torques_l2` | YES | Pure warp kernel |
| 10 | `joint_vel_l1` | YES | Pure warp kernel |
| 11 | `joint_vel_l2` | YES | Pure warp kernel |
| 12 | `joint_acc_l2` | YES | Pure warp kernel |
| 13 | `joint_deviation_l1` | YES | Pure warp kernel |
| 14 | `joint_pos_limits` | YES | Pure warp kernel |
| 15 | `joint_vel_limits` | YES | Pure warp kernel |
| 16 | `applied_torque_limits` | YES | Pure warp kernel |
| 17 | `action_rate_l2` | YES | Pure warp kernel |
| 18 | `action_l2` | YES | Pure warp kernel |
| 19 | `undesired_contacts` | YES | `wp.from_torch` bridge for sensor data |
| 20 | `desired_contacts` | YES | `wp.from_torch` bridge for sensor data |
| 21 | `contact_forces` | YES | `wp.from_torch` bridge for sensor data |
| 22 | `track_lin_vel_xy_exp` | YES | `wp.from_torch` bridge for commands |
| 23 | `track_ang_vel_z_exp` | YES | `wp.from_torch` bridge for commands |

**Coverage: 22/22 (100%)** (note: `track_*` counted as shared rewards)

### Terminations (10 stable terms)

| # | Function/Class | Warp | Notes |
|---|---|---|---|
| 1 | `time_out` | YES | Pure warp kernel |
| 2 | `command_resample` | YES | Pure warp kernel |
| 3 | `bad_orientation` | YES | Pure warp kernel |
| 4 | `root_height_below_minimum` | YES | Pure warp kernel |
| 5 | `joint_pos_out_of_limit` | YES | Pure warp kernel |
| 6 | `joint_pos_out_of_manual_limit` | YES | Pure warp kernel |
| 7 | `joint_vel_out_of_limit` | YES | Pure warp kernel |
| 8 | `joint_vel_out_of_manual_limit` | YES | Pure warp kernel |
| 9 | `joint_effort_out_of_limit` | YES | Pure warp kernel |
| 10 | `illegal_contact` | YES | `wp.from_torch` bridge for sensor data |

**Coverage: 10/10 (100%)**

### Events (20 stable terms)

| # | Function/Class | Warp | Notes |
|---|---|---|---|
| 1 | `randomize_rigid_body_material` | YES | Class-based, warp kernel for mu sampling |
| 2 | `randomize_rigid_body_mass` | YES | Class-based, `_scale_inertia_kernel` |
| 3 | `randomize_rigid_body_com` | YES | Warp kernel |
| 4 | `randomize_actuator_gains` | YES | Class-based, writes directly to warp arrays |
| 5 | `randomize_joint_parameters` | YES | Class-based, warp kernels with clamp |
| 6 | `apply_external_force_torque` | YES | Warp kernel |
| 7 | `push_by_setting_velocity` | YES | Warp kernel |
| 8 | `reset_root_state_uniform` | YES | Warp kernel |
| 9 | `reset_root_state_with_random_orientation` | YES | Warp kernel |
| 10 | `reset_root_state_from_terrain` | YES | Warp kernel |
| 11 | `reset_joints_by_scale` | YES | Warp kernel |
| 12 | `reset_joints_by_offset` | YES | Warp kernel |
| 13 | `reset_scene_to_default` | YES | Warp kernel |
| 14 | `randomize_rigid_body_collider_offsets` | **NO** | Stub (`NotImplementedError`) in stable |
| 15 | `randomize_physics_scene_gravity` | **NO** | Class-based, per-env gravity. Low priority. |
| 16 | `randomize_fixed_tendon_parameters` | **NO** | Stub (`NotImplementedError`) in stable |
| 17 | `reset_nodal_state_uniform` | **NO** | Stub (`NotImplementedError`) in stable |
| 18 | `randomize_rigid_body_scale` | **NO** | USD `pxr` API, pre-sim only. Not convertible. |
| 19 | `randomize_visual_texture_material` | **NO** | Omni Replicator API. Not convertible. |
| 20 | `randomize_visual_color` | **NO** | Omni Replicator API. Not convertible. |

**Coverage: 13/20 (65%)** -- remaining are stubs, USD/Replicator APIs, or low-priority

### Actions (10 stable classes)

| # | Class | Warp | Notes |
|---|---|---|---|
| 1 | `JointPositionActionCfg` | YES | Warp-first process_actions/apply_actions |
| 2 | `RelativeJointPositionActionCfg` | YES | |
| 3 | `JointVelocityActionCfg` | YES | |
| 4 | `JointEffortActionCfg` | YES | |
| 5 | `BinaryJointPositionActionCfg` | YES | |
| 6 | `BinaryJointVelocityActionCfg` | YES | |
| 7 | `JointPositionToLimitsActionCfg` | YES | |
| 8 | `EMAJointPositionToLimitsActionCfg` | YES | |
| 9 | `NonHolonomicActionCfg` | YES | |
| 10 | (IK-based actions) | N/A | Not used by current tasks |

**Coverage: 10/10 (100%)**

### Commands (6 stable classes)

| # | Class | Warp | Notes |
|---|---|---|---|
| 1 | `NullCommand` / `NullCommandCfg` | NO | Bridged via `wp.from_torch` (zero-copy) |
| 2 | `UniformVelocityCommand` / Cfg | NO | Bridged via `wp.from_torch` |
| 3 | `NormalVelocityCommand` / Cfg | NO | Bridged via `wp.from_torch` |
| 4 | `UniformPoseCommand` / Cfg | NO | Bridged via `wp.from_torch` |
| 5 | `UniformPose2dCommand` / Cfg | NO | Bridged via `wp.from_torch` |
| 6 | `TerrainBasedPose2dCommand` / Cfg | NO | Bridged via `wp.from_torch` |

**Coverage: 0/6 (0%)** -- **NOT a blocker** (see [Cross-Cutting Notes](#cross-cutting-notes))

### Curriculums (3 stable classes)

| # | Class | Warp | Notes |
|---|---|---|---|
| 1 | `modify_reward_weight` | NO | Runs at reset, not per-step. Low priority. |
| 2 | `modify_env_param` | NO | Runs at reset, not per-step. Low priority. |
| 3 | `modify_term_cfg` | NO | Inherits from `modify_env_param`. Low priority. |

**Coverage: 0/3 (0%)** -- Low priority (not in hot loop)

### Overall Shared Library Coverage

| Category | Stable | Warp | Coverage |
|---|---|---|---|
| **Actions** | 10 | 10 | **100%** |
| **Observations** | 22 | 20 | **91%** |
| **Rewards** | 22 | 22 | **100%** |
| **Terminations** | 10 | 10 | **100%** |
| **Events** | 20 | 13 | **65%** |
| **Commands** | 6 | 0 | **0%** (bridged) |
| **Curriculums** | 3 | 0 | **0%** (low priority) |
| **Total** | **93** | **75** | **~81%** |

---

## Per-Task Gym ID Migration Table

All stable manager-based tasks and their experimental `-Warp` counterpart status.

### Classic Tasks

| Stable Gym ID | Exp Gym ID | Status | Custom Terms | Blockers |
|---|---|---|---|---|
| `Isaac-Cartpole-v0` | `Isaac-Cartpole-Warp-v0` | **MIGRATED** | 1 reward | None |
| `Isaac-Humanoid-v0` | `Isaac-Humanoid-Warp-v0` | **MIGRATED** | 4 obs, 5 rewards | None |
| `Isaac-Ant-v0` | `Isaac-Ant-Warp-v0` | **MIGRATED** | Reuses humanoid mdp | None |

### Locomotion Velocity -- Standard Robots (use base velocity MDP)

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-Velocity-Flat-Unitree-A1-v0` | `Isaac-Velocity-Flat-Unitree-A1-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Unitree-A1-Play-v0` | `Isaac-Velocity-Flat-Unitree-A1-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Unitree-Go1-v0` | `Isaac-Velocity-Flat-Unitree-Go1-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Unitree-Go1-Play-v0` | `Isaac-Velocity-Flat-Unitree-Go1-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Unitree-Go2-v0` | `Isaac-Velocity-Flat-Unitree-Go2-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` | `Isaac-Velocity-Flat-Unitree-Go2-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-B-v0` | `Isaac-Velocity-Flat-Anymal-B-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-B-Play-v0` | `Isaac-Velocity-Flat-Anymal-B-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-C-v0` | `Isaac-Velocity-Flat-Anymal-C-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-C-Play-v0` | `Isaac-Velocity-Flat-Anymal-C-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-D-v0` | `Isaac-Velocity-Flat-Anymal-D-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Anymal-D-Play-v0` | `Isaac-Velocity-Flat-Anymal-D-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Rough-Anymal-D-v0` | `Isaac-Velocity-Rough-Anymal-D-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Rough-Anymal-D-Play-v0` | `Isaac-Velocity-Rough-Anymal-D-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Cassie-v0` | `Isaac-Velocity-Flat-Cassie-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-Cassie-Play-v0` | `Isaac-Velocity-Flat-Cassie-Warp-Play-v0` | **MIGRATED** | None |

### Locomotion Velocity -- Biped Robots (use base + biped rewards)

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-Velocity-Flat-G1-v0` | `Isaac-Velocity-Flat-G1-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-G1-Play-v0` | `Isaac-Velocity-Flat-G1-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-G1-v1` | `Isaac-Velocity-Flat-G1-Warp-v1` | **MIGRATED** | None (29-DOF) |
| `Isaac-Velocity-Flat-G1-Play-v1` | `Isaac-Velocity-Flat-G1-Warp-Play-v1` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-H1-v0` | `Isaac-Velocity-Flat-H1-Warp-v0` | **MIGRATED** | None |
| `Isaac-Velocity-Flat-H1-Play-v0` | `Isaac-Velocity-Flat-H1-Warp-Play-v0` | **MIGRATED** | None |

### Locomotion Velocity -- Rough Terrain (biped)

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-Velocity-Rough-G1-v0` | `Isaac-Velocity-Rough-G1-Warp-v0` | **MIGRATED** | None (stable has this commented out) |
| `Isaac-Velocity-Rough-G1-Play-v0` | `Isaac-Velocity-Rough-G1-Warp-Play-v0` | **MIGRATED** | None (stable has this commented out) |
| `Isaac-Velocity-Rough-H1-v0` | `Isaac-Velocity-Rough-H1-Warp-v0` | **MIGRATED** | None (stable has this commented out) |
| `Isaac-Velocity-Rough-H1-Play-v0` | `Isaac-Velocity-Rough-H1-Warp-Play-v0` | **MIGRATED** | None (stable has this commented out) |

### Locomotion Velocity -- Spot (custom MDP)

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-Velocity-Flat-Spot-v0` | -- | **NOT MIGRATED** | 14 custom reward fns + 1 event fn + `GaitReward` class need conversion |
| `Isaac-Velocity-Flat-Spot-Play-v0` | -- | **NOT MIGRATED** | Same as above |

### Locomotion Velocity -- Distillation Variants

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Velocity-G1-Distillation-v1` | -- | **NOT MIGRATED** | Teacher-student pipeline not in scope |
| `Velocity-G1-Student-Finetune-v1` | -- | **NOT MIGRATED** | Teacher-student pipeline not in scope |

### Manipulation -- Reach

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-Reach-Franka-v0` | `Isaac-Reach-Franka-Warp-v0` | **MIGRATED** | None |
| `Isaac-Reach-Franka-Play-v0` | `Isaac-Reach-Franka-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Reach-UR10-v0` | `Isaac-Reach-UR10-Warp-v0` | **MIGRATED** | None |
| `Isaac-Reach-UR10-Play-v0` | `Isaac-Reach-UR10-Warp-Play-v0` | **MIGRATED** | None |
| `Isaac-Reach-Franka-IK-Abs-v0` | -- | **NOT MIGRATED** | IK action not in scope |
| `Isaac-Reach-Franka-IK-Rel-v0` | -- | **NOT MIGRATED** | IK action not in scope |
| `Isaac-Reach-UR10-IK-Abs-v0` | -- | **NOT MIGRATED** | IK action not in scope |
| `Isaac-Reach-UR10-IK-Rel-v0` | -- | **NOT MIGRATED** | IK action not in scope |

### Manipulation -- Dexsuite

| Stable Gym ID | Exp Gym ID | Status | Blockers |
|---|---|---|---|
| `Isaac-DexsuiteKukaAllegroReorient-v0` | -- | **NOT MIGRATED** | Custom command class, 7 obs fns, 5 reward fns, 3 term fns, ADR curriculum |
| `Isaac-DexsuiteKukaAllegroReorient-Play-v0` | -- | **NOT MIGRATED** | Same as above |
| `Isaac-DexsuiteKukaAllegroLift-v0` | -- | **NOT MIGRATED** | Same + lift-specific overrides |
| `Isaac-DexsuiteKukaAllegroLift-Play-v0` | -- | **NOT MIGRATED** | Same as above |
| `Isaac-DexsuiteKukaAllegroReorientVision-v0` | -- | **NOT MIGRATED** | Same + `image`/`vision_camera` obs (not convertible) |
| `Isaac-DexsuiteKukaAllegroLiftVision-v0` | -- | **NOT MIGRATED** | Same as above |

### Migration Summary

| Category | Total Stable | Migrated | Not Migrated | % |
|---|---|---|---|---|
| Classic | 3 | 3 | 0 | **100%** |
| Velocity (flat, quadruped) | 16 | 16 | 0 | **100%** |
| Velocity (flat, biped) | 6 | 6 | 0 | **100%** |
| Velocity (rough) | 6 | 6 | 0 | **100%** |
| Velocity (Spot) | 2 | 0 | 2 | **0%** |
| Velocity (distillation) | 2 | 0 | 2 | **0%** |
| Reach (joint-space) | 4 | 4 | 0 | **100%** |
| Reach (IK) | 4 | 0 | 4 | **0%** |
| Dexsuite | 6 | 0 | 6 | **0%** |
| **Total** | **49** | **35** | **14** | **71%** |

---

## Per-Task MDP Term Usage

Shows which shared and custom MDP terms each task group uses. Terms from `isaaclab.envs.mdp` (shared)
are marked **S**. Task-local custom terms are marked **C**.

### Cartpole

| Manager | Term | Source | Warp |
|---|---|---|---|
| Actions | `JointEffortActionCfg` | S | YES |
| Obs | `joint_pos_rel` | S | YES |
| Obs | `joint_vel_rel` | S | YES |
| Rewards | `is_alive` | S | YES |
| Rewards | `is_terminated` | S | YES |
| Rewards | `joint_pos_target_l2` | **C** | YES |
| Rewards | `joint_vel_l1` | S | YES |
| Terms | `time_out` | S | YES |
| Terms | `joint_pos_out_of_manual_limit` | S | YES |
| Events | `reset_joints_by_offset` | S | YES |

### Humanoid / Ant

| Manager | Term | Source | Warp |
|---|---|---|---|
| Actions | `JointEffortActionCfg` | S | YES |
| Obs | `base_pos_z` | S | YES |
| Obs | `base_lin_vel` | S | YES |
| Obs | `base_ang_vel` | S | YES |
| Obs | `base_yaw_roll` | **C** | YES |
| Obs | `base_angle_to_target` | **C** | YES |
| Obs | `base_up_proj` | **C** | YES |
| Obs | `base_heading_proj` | **C** | YES |
| Obs | `joint_pos_limit_normalized` | S | YES |
| Obs | `joint_vel_rel` | S | YES |
| Obs | `last_action` | S | YES |
| Rewards | `progress_reward` | **C** (class) | YES |
| Rewards | `is_alive` | S | YES |
| Rewards | `upright_posture_bonus` | **C** | YES |
| Rewards | `move_to_target_bonus` | **C** | YES |
| Rewards | `action_l2` | S | YES |
| Rewards | `power_consumption` | **C** (class) | YES |
| Rewards | `joint_pos_limits_penalty_ratio` | **C** (class) | YES |
| Terms | `time_out` | S | YES |
| Terms | `root_height_below_minimum` | S | YES |
| Events | `reset_root_state_uniform` | S | YES |
| Events | `reset_joints_by_offset` | S | YES |

### Velocity Locomotion (base config -- all non-Spot robots)

| Manager | Term | Source | Warp |
|---|---|---|---|
| Commands | `UniformVelocityCommandCfg` | S | bridged |
| Actions | `JointPositionActionCfg` | S | YES |
| Obs | `base_lin_vel` | S | YES |
| Obs | `base_ang_vel` | S | YES |
| Obs | `projected_gravity` | S | YES |
| Obs | `generated_commands` | S | YES |
| Obs | `joint_pos_rel` | S | YES |
| Obs | `joint_vel_rel` | S | YES |
| Obs | `last_action` | S | YES |
| Rewards | `track_lin_vel_xy_exp` | S | YES |
| Rewards | `track_ang_vel_z_exp` | S | YES |
| Rewards | `lin_vel_z_l2` | S | YES |
| Rewards | `ang_vel_xy_l2` | S | YES |
| Rewards | `joint_torques_l2` | S | YES |
| Rewards | `joint_acc_l2` | S | YES |
| Rewards | `action_rate_l2` | S | YES |
| Rewards | `feet_air_time` | **C** | YES |
| Rewards | `undesired_contacts` | S | YES |
| Rewards | `flat_orientation_l2` | S | YES |
| Rewards | `joint_pos_limits` | S | YES |
| Terms | `time_out` | S | YES |
| Terms | `illegal_contact` | S | YES |
| Terms | `terrain_out_of_bounds` | **C** | YES |
| Events | `randomize_rigid_body_com` | S | YES |
| Events | `apply_external_force_torque` | S | YES |
| Events | `reset_root_state_uniform` | S | YES |
| Events | `reset_joints_by_scale` | S | YES |
| Events | `push_by_setting_velocity` | S | YES |
| Curriculum | `terrain_levels_vel` | **C** | YES (torch, reset-only) |

Biped robots (G1, G1-29, H1) additionally use:

| Manager | Term | Source | Warp |
|---|---|---|---|
| Rewards | `feet_air_time_positive_biped` | **C** | YES |
| Rewards | `feet_slide` | **C** | YES |
| Rewards | `track_lin_vel_xy_yaw_frame_exp` | **C** | YES |
| Rewards | `track_ang_vel_z_world_exp` | **C** | YES |
| Rewards | `joint_deviation_l1` | S | YES |
| Rewards | `is_terminated` | S | YES |

### Velocity Locomotion -- Spot (custom MDP, NOT migrated)

Additional terms beyond the base velocity config:

| Manager | Term | Source | Warp |
|---|---|---|---|
| Events | `reset_joints_around_default` | **C** | **NO** |
| Rewards | `air_time_reward` | **C** | **NO** |
| Rewards | `base_angular_velocity_reward` | **C** | **NO** |
| Rewards | `base_linear_velocity_reward` | **C** | **NO** |
| Rewards | `GaitReward` | **C** (class) | **NO** |
| Rewards | `foot_clearance_reward` | **C** | **NO** |
| Rewards | `action_smoothness_penalty` | **C** | **NO** |
| Rewards | `air_time_variance_penalty` | **C** | **NO** |
| Rewards | `base_motion_penalty` | **C** | **NO** |
| Rewards | `base_orientation_penalty` | **C** | **NO** |
| Rewards | `foot_slip_penalty` | **C** | **NO** |
| Rewards | `joint_acceleration_penalty` | **C** | **NO** |
| Rewards | `joint_position_penalty` | **C** | **NO** |
| Rewards | `joint_torques_penalty` | **C** | **NO** |
| Rewards | `joint_velocity_penalty` | **C** | **NO** |

**Total Spot custom terms to convert: 15** (1 event + 14 rewards including 1 class)

### Reach (Franka, UR10)

| Manager | Term | Source | Warp |
|---|---|---|---|
| Commands | `UniformPoseCommandCfg` | S | bridged |
| Actions | `JointPositionActionCfg` | S | YES |
| Obs | `joint_pos_rel` | S | YES |
| Obs | `joint_vel_rel` | S | YES |
| Obs | `generated_commands` | S | YES |
| Obs | `last_action` | S | YES |
| Rewards | `position_command_error` | **C** | YES |
| Rewards | `position_command_error_tanh` | **C** | YES |
| Rewards | `orientation_command_error` | **C** | YES |
| Rewards | `action_rate_l2` | S | YES |
| Rewards | `joint_vel_l2` | S | YES |
| Terms | `time_out` | S | YES |
| Events | `reset_root_state_uniform` | S | YES |
| Events | `reset_joints_by_scale` | S | YES |
| Curriculum | `modify_reward_weight` | S | forwarded from stable |

### Dexsuite (NOT migrated)

| Manager | Term | Source | Warp |
|---|---|---|---|
| Commands | `ObjectUniformPoseCommandCfg` | **C** | **NO** |
| Actions | `RelativeJointPositionActionCfg` | S | YES |
| Obs | `object_quat_b` | **C** | **NO** |
| Obs | `generated_commands` | S | YES |
| Obs | `last_action` | S | YES |
| Obs | `time_left` | **C** | **NO** |
| Obs | `joint_pos` | S | YES |
| Obs | `joint_vel` | S | YES |
| Obs | `body_state_b` | **C** | **NO** |
| Obs | `object_point_cloud_b` | **C** (class) | **NO** |
| Obs | `fingers_contact_force_b` | **C** | **NO** |
| Obs | `vision_camera` | **C** (class) | **NO** (not convertible) |
| Rewards | `action_l2_clamped` | **C** | **NO** |
| Rewards | `action_rate_l2_clamped` | **C** | **NO** |
| Rewards | `object_ee_distance` | **C** | **NO** |
| Rewards | `position_command_error_tanh` | S | YES |
| Rewards | `orientation_command_error_tanh` | S | YES |
| Rewards | `success_reward` | **C** | **NO** |
| Rewards | `is_terminated_term` | S | YES |
| Terms | `time_out` | S | YES |
| Terms | `out_of_bound` | **C** (class) | **NO** |
| Terms | `object_spinning_too_fast` | **C** | **NO** |
| Terms | `abnormal_robot_state` | **C** | **NO** |
| Events | `reset_root_state_uniform` | S | YES |
| Events | `reset_joints_by_offset` | S | YES |
| Events | `randomize_physics_scene_gravity` | S | **NO** |
| Curriculum | ADR | **C** (class) | **NO** |

**Total Dexsuite custom terms to convert: 15** (1 command class, 6 obs, 4 rewards, 3 terms, 1 curriculum)
Plus `randomize_physics_scene_gravity` from shared events.
Plus `vision_camera` / `image` are not convertible (PyTorch NN).

---

## Custom (Task-Local) MDP Terms

All task-specific MDP functions and classes, grouped by task, with conversion status.

### Cartpole -- `isaaclab_tasks.manager_based.classic.cartpole.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `joint_pos_target_l2` | reward fn | YES | Pure warp kernel `_joint_pos_target_l2_kernel` |

### Humanoid -- `isaaclab_tasks.manager_based.classic.humanoid.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `base_yaw_roll` | obs fn | YES | Warp kernel `_base_yaw_roll_kernel` |
| `base_up_proj` | obs fn | YES | Warp kernel `_base_up_proj_kernel` |
| `base_heading_proj` | obs fn | YES | Warp kernel `_base_heading_proj_kernel` |
| `base_angle_to_target` | obs fn | YES | Warp kernel |
| `upright_posture_bonus` | reward fn | YES | Warp kernel `_upright_posture_bonus_kernel` |
| `move_to_target_bonus` | reward fn | YES | Warp kernel `_move_to_target_bonus_kernel` |
| `progress_reward` | reward class | YES | Warp kernels `_progress_reward_kernel`, `_progress_reward_reset_kernel` |
| `joint_pos_limits_penalty_ratio` | reward class | YES | Warp kernel `_joint_pos_limits_penalty_ratio_kernel`, gear ratios cached in `__init__` |
| `power_consumption` | reward class | YES | Warp kernel `_power_consumption_kernel`, gear ratios cached in `__init__` |

### Velocity Locomotion -- `isaaclab_tasks.manager_based.locomotion.velocity.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `feet_air_time` | reward fn | YES | Warp kernel, sensor data cached via `wp.from_torch` |
| `feet_air_time_positive_biped` | reward fn | YES | Warp kernel, sensor data cached via `wp.from_torch` |
| `feet_slide` | reward fn | YES | Warp kernel, force history cached via `wp.from_torch` |
| `track_lin_vel_xy_yaw_frame_exp` | reward fn | YES | Warp kernel with yaw-frame rotation |
| `track_ang_vel_z_world_exp` | reward fn | YES | Warp kernel with exponential error |
| `stand_still_joint_deviation_l1` | reward fn | YES | Warp kernel with command gating |
| `terrain_out_of_bounds` | term fn | YES | Warp kernel, terrain config cached on first call |
| `terrain_levels_vel` | curriculum fn | YES | Torch-based (runs at reset, not per-step) |

### Velocity Locomotion / Spot -- `isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `reset_joints_around_default` | event fn | **NO** | Torch-based |
| `air_time_reward` | reward fn | **NO** | Torch-based |
| `base_angular_velocity_reward` | reward fn | **NO** | Torch-based |
| `base_linear_velocity_reward` | reward fn | **NO** | Torch-based |
| `GaitReward` | reward class | **NO** | Torch-based, stateful (gait phase tracking) |
| `foot_clearance_reward` | reward fn | **NO** | Torch-based |
| `action_smoothness_penalty` | reward fn | **NO** | Torch-based |
| `air_time_variance_penalty` | reward fn | **NO** | Torch-based |
| `base_motion_penalty` | reward fn | **NO** | Torch-based |
| `base_orientation_penalty` | reward fn | **NO** | Torch-based |
| `foot_slip_penalty` | reward fn | **NO** | Torch-based |
| `joint_acceleration_penalty` | reward fn | **NO** | Torch-based |
| `joint_position_penalty` | reward fn | **NO** | Torch-based |
| `joint_torques_penalty` | reward fn | **NO** | Torch-based |
| `joint_velocity_penalty` | reward fn | **NO** | Torch-based |

### Reach -- `isaaclab_tasks.manager_based.manipulation.reach.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `position_command_error` | reward fn | YES | Warp kernel `_position_command_error_kernel` with frame transform |
| `position_command_error_tanh` | reward fn | YES | Warp kernel `_position_command_error_tanh_kernel` |
| `orientation_command_error` | reward fn | YES | Warp kernel `_orientation_command_error_kernel` with quat math |

### Dexsuite -- `isaaclab_tasks.manager_based.manipulation.dexsuite.mdp`

| Term | Type | Warp | Implementation |
|---|---|---|---|
| `ObjectUniformPoseCommandCfg` | command class | **NO** | Torch-based |
| `object_pos_b` | obs fn | **NO** | Torch-based |
| `object_quat_b` | obs fn | **NO** | Torch-based |
| `body_state_b` | obs fn | **NO** | Torch-based |
| `object_point_cloud_b` | obs class | **NO** | Torch-based + USD point cloud |
| `fingers_contact_force_b` | obs fn | **NO** | Torch-based |
| `vision_camera` | obs class | **NO** | Not convertible (image pipeline) |
| `time_left` | obs fn | **NO** | Torch-based (simple) |
| `action_l2_clamped` | reward fn | **NO** | Torch-based |
| `action_rate_l2_clamped` | reward fn | **NO** | Torch-based |
| `object_ee_distance` | reward fn | **NO** | Torch-based |
| `success_reward` | reward fn | **NO** | Torch-based |
| `contacts` | reward fn | **NO** | Torch-based |
| `out_of_bound` | term class | **NO** | Torch-based |
| `abnormal_robot_state` | term fn | **NO** | Torch-based |
| `object_spinning_too_fast` | term fn | **NO** | Torch-based |
| ADR curriculum | curriculum class | **NO** | Torch-based |

---

## Shared Terms Not Used by Any Migrated Task

These shared MDP terms have warp overrides but are **not referenced** by any of the 31 currently
migrated gym IDs. They are available for future tasks.

### Observations (not used by any migrated task)

| Term | Warp | Potential Users |
|---|---|---|
| `root_pos_w` | YES | Navigation, locomanipulation |
| `root_quat_w` | YES | Navigation, locomanipulation |
| `root_lin_vel_w` | YES | Navigation |
| `root_ang_vel_w` | YES | Navigation |
| `body_pose_w` | YES | Manipulation (end-effector tracking) |
| `body_projected_gravity_b` | YES | Manipulation, dexterity |
| `joint_pos` | YES | Dexsuite (uses it, but not migrated) |
| `joint_vel` | YES | Dexsuite (uses it, but not migrated) |
| `joint_effort` | YES | Future tasks |
| `current_time_s` | YES | Future tasks |
| `remaining_time_s` | YES | Future tasks |

### Rewards (not used by any migrated task)

| Term | Warp | Potential Users |
|---|---|---|
| `base_height_l2` | YES | Locomotion with height tracking |
| `body_lin_acc_l2` | YES | Smooth motion tasks |
| `joint_vel_limits` | YES | Safety-critical tasks |
| `applied_torque_limits` | YES | Torque-limited robots |
| `desired_contacts` | YES | Gait-specific tasks |
| `contact_forces` | YES | Force control tasks |

### Terminations (not used by any migrated task)

| Term | Warp | Potential Users |
|---|---|---|
| `command_resample` | YES | Command-driven tasks with resampling |
| `bad_orientation` | YES | Tasks needing orientation limits |
| `joint_pos_out_of_limit` | YES | Safety-critical tasks |
| `joint_vel_out_of_limit` | YES | Safety-critical tasks |
| `joint_vel_out_of_manual_limit` | YES | Safety-critical tasks |
| `joint_effort_out_of_limit` | YES | Torque-limited tasks |

### Events (not used by any migrated task)

| Term | Warp | Potential Users |
|---|---|---|
| `randomize_rigid_body_material` | YES | Sim-to-real tasks |
| `randomize_rigid_body_mass` | YES | Sim-to-real tasks |
| `randomize_actuator_gains` | YES | Sim-to-real tasks |
| `randomize_joint_parameters` | YES | Sim-to-real tasks |
| `reset_root_state_with_random_orientation` | YES | Manipulation reset |
| `reset_root_state_from_terrain` | YES | Rough terrain locomotion |
| `reset_scene_to_default` | YES | General reset |

---

## Cross-Cutting Notes

### Command Manager -- NOT a Blocker

The command manager classes themselves remain torch-based, but this does **not** block
downstream MDP terms that consume command data. The command manager's output tensors
(`get_command()`, `get_term().time_left`, etc.) have stable pointers that can be wrapped
with `wp.from_torch()` on the first call. Subsequent calls reuse the zero-copy view,
so the warp kernel always reads the latest command values with no conversion overhead.

Pattern used in all command-dependent terms:
```python
def some_term(env, out, command_name: str, ...) -> None:
    if not hasattr(some_term, "_cmd_wp"):
        cmd_torch = env.command_manager.get_command(command_name)
        some_term._cmd_wp = wp.from_torch(cmd_torch.contiguous())
    wp.launch(kernel=_some_kernel, ..., inputs=[some_term._cmd_wp, ...], ...)
```

### Integrator Setting

All experimental configs use `integrator="implicit"` (standard Newton solver) instead of
the stable `"implicitfast"` variant. This is required for the MJWarp solver backend.

### Graphability: No Torch in the Hot Path

All per-step MDP terms must be CUDA-graph capturable. This means:

- **No `torch.*` ops** in function bodies (only in `__init__` or `reset`)
- **No Python conditionals on changing values** in the per-step path
- Cross-manager torch tensors (commands, sensors) are cached as zero-copy warp views
  via `wp.from_torch()` on first call using the `hasattr` pattern (see below)
- After warmup, every per-step call is a pure `wp.launch()` chain

### Non-Capturable MDP Terms (`@warp_capturable(False)`)

Some MDP terms access `ArticulationData` properties backed by `TimestampedWarpBuffer`
(lazy derived properties). These are incompatible with `wp.ScopedCapture` because the
timestamp guard (`if timestamp < sim_timestamp`) is a Python branch — the warmup call
updates the timestamp, causing the capture call to skip the kernel entirely. The graph
then replays with stale data. See `GRAPH_CAPTURE_MIGRATION.md` in the Newton
articulation package for the full Tier 1/2/3 property analysis.

Affected terms are marked `@warp_capturable(False)`, which causes the owning manager to
fall back to mode=1 (warp not captured) automatically via `register_manager_capturability`.

**Observations** (`isaaclab_experimental/envs/mdp/observations.py`):

| Term | Accesses | Status |
|------|----------|--------|
| `base_lin_vel` | `root_lin_vel_b` → `root_com_vel_b` (Tier 2) | Applied |
| `base_ang_vel` | `root_ang_vel_b` → `root_com_vel_b` (Tier 2) | Applied |
| `projected_gravity` | `projected_gravity_b` (Tier 2) | Applied |
| `body_projected_gravity_b` | `projected_gravity_b` (Tier 2) | Pending (body-level, not yet in experimental module) |

**Rewards — base** (`isaaclab_experimental/envs/mdp/rewards.py`):

| Term | Accesses | Status |
|------|----------|--------|
| `lin_vel_z_l2` | `root_lin_vel_b` → `root_com_vel_b` (Tier 2) | Applied |
| `ang_vel_xy_l2` | `root_ang_vel_b` → `root_com_vel_b` (Tier 2) | Applied |
| `flat_orientation_l2` | `projected_gravity_b` (Tier 2) | Applied |
| `track_lin_vel_xy_exp` | `root_lin_vel_b` → `root_com_vel_b` (Tier 2) | Applied |
| `track_ang_vel_z_exp` | `root_ang_vel_b` → `root_com_vel_b` (Tier 2) | Applied |

**Rewards — humanoid** (`isaaclab_tasks_experimental/.../humanoid/mdp/rewards.py`):

| Term | Accesses | Status |
|------|----------|--------|
| `upright_posture_bonus` | `projected_gravity_b` (Tier 2) | Applied |

**Rewards — safe** (no Tier 2 access, fully capturable):

- `track_lin_vel_xy_yaw_frame_exp` → `root_quat_w`, `root_lin_vel_w` (Tier 3 from Tier 1)
- `track_ang_vel_z_world_exp` → `root_ang_vel_w` (Tier 3 from Tier 1)
- `feet_slide` → `body_lin_vel_w` → `body_com_lin_vel_w` (Tier 3 from Tier 1)
- `feet_air_time`, `feet_air_time_positive_biped` → contact sensor data
- `joint_torques_l2`, `joint_acc_l2`, `joint_vel_l2`, etc. → `joint_pos`, `joint_vel` (Tier 1)
- `is_alive`, `is_terminated`, `action_rate_l2`, `action_l2` → no articulation data

**Applied fix (Phase 1):** Affected MDP kernels were rewritten to consume Tier 1 compound
types directly (`root_link_pose_w` as `wp.transformf`, `root_com_vel_w` as
`wp.spatial_vectorf`) and perform the body-frame rotation inline, eliminating the Tier 2
dependency entirely. The `@warp_capturable(False)` annotations have been removed and these
terms are now fully capturable. Shared `@wp.func` helpers (`body_lin_vel_from_root`,
`body_ang_vel_from_root`, `rotate_vec_to_body_frame`) live in
`isaaclab_newton.kernels.state_kernels`. See `GRAPH_CAPTURE_MIGRATION.md` in the Newton
articulation package for Phase 2 plans (making Tier 2 lazy update itself graph-safe).

### Resolved Cross-Cutting Blockers

| Blocker | Resolution |
|---|---|
| Class-based term support | All 5 class-based terms converted (see Events table) |
| Command Manager dependency | `wp.from_torch` bridge (zero-copy) |
| ContactSensor dependency | `net_forces_w_history` wrapped via `wp.from_torch` on first call |
| `body_mask` pattern | body_ids cached as `wp.array(dtype=wp.int32)` on first call |

### Remaining Gaps (shared events)

| Term | Why Deferred |
|---|---|
| `randomize_rigid_body_collider_offsets` | Stub (`NotImplementedError`) in stable |
| `randomize_physics_scene_gravity` | Class-based, per-env gravity. Low priority. |
| `randomize_fixed_tendon_parameters` | Stub (`NotImplementedError`) in stable |
| `reset_nodal_state_uniform` | Stub (`NotImplementedError`) in stable |
| `randomize_rigid_body_scale` | USD `pxr` API, pre-sim only. Not convertible. |
| `randomize_visual_texture_material` | Omni Replicator API. Not convertible. |
| `randomize_visual_color` | Omni Replicator API. Not convertible. |

### Not Convertible to Pure Warp

| Term | Reason |
|---|---|
| `image` | 4D image tensor with per-type normalization, depth conversion. |
| `image_features` | PyTorch NN inference (ResNet/Theia); must remain hybrid. |
| `vision_camera` (dexsuite) | Image pipeline, same limitation as `image`. |
| `randomize_rigid_body_scale` | USD `pxr` API, no tensor math. |
| `randomize_visual_texture_material` | Omni Replicator API. |
| `randomize_visual_color` | Omni Replicator API. |

---

## Migration Pattern

### How to create an `_exp` task copy

For each stable manager-based task, the experimental copy follows this structure:

```
isaaclab_tasks_experimental/manager_based/<category>/<task>/
├── __init__.py          # gym.register with -Warp suffix
├── <task>_env_cfg.py    # Copy of stable, change imports
└── mdp/
    ├── __init__.py      # from isaaclab_experimental.envs.mdp import *; from .custom import *
    └── <custom>.py      # Warp-first versions of task-specific terms
```

No `agents/` directory -- reuse stable agent configs via import.

### `__init__.py` registration pattern

```python
import gymnasium as gym

# Reuse agent configs from the stable task package.
from isaaclab_tasks.manager_based.<category>.<task> import agents

gym.register(
    id="Isaac-<Task>-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<task>_env_cfg:<Task>EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:<Task>PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # ... all agent configs from stable
    },
)
```

Key rules:
- Use `f"{__name__}"` for `env_cfg_entry_point` (local experimental config)
- Use `f"{agents.__name__}"` for all agent configs (points to stable)
- Include ALL agent configs that stable has (rsl_rl, skrl, sb3, rl_games, symmetry)
- Entry point is always `isaaclab_experimental.envs:ManagerBasedRLEnvWarp`

### env_cfg.py import changes

```python
# Stable imports:
from isaaclab.managers import ...
import isaaclab_tasks.manager_based.<path>.mdp as mdp

# Experimental imports:
from isaaclab_experimental.managers import ...
import isaaclab_tasks_experimental.manager_based.<path>.mdp as mdp
```

### MDP term signature change

```python
# Stable (torch): returns tensor
def term(env, **params) -> torch.Tensor:

# Experimental (warp): writes to pre-allocated output
def term(env, out: wp.array, **params) -> None:
```

### Kernel + function co-location

Every warp kernel must be placed directly above the function that launches it:

```python
@wp.kernel
def _my_reward_kernel(data: ..., out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = ...

def my_reward(env, out, asset_cfg=SceneEntityCfg("robot")) -> None:
    asset = env.scene[asset_cfg.name]
    wp.launch(kernel=_my_reward_kernel, dim=env.num_envs, inputs=[..., out], device=env.device)
```

### Cross-manager reference caching (commands, sensors)

Torch tensors from other managers (commands, contact sensors) are cached as zero-copy
warp views on first call. The `hasattr` guard only executes during warmup (before graph
capture). After warmup, only the `wp.launch` runs.

```python
def some_reward(env, out, command_name: str, ...) -> None:
    fn = some_reward
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(kernel=_some_kernel, ..., inputs=[fn._cmd_wp, ...], ...)
```

### Observation dimension inference (`out_dim`)

The observation manager infers output buffer dimensions from decorator metadata.
No `term_dim` parameter is needed in env_cfg `params`.

Resolution order:
1. `out_dim` on `@generic_io_descriptor_warp` decorator (for body/command/action/time obs)
2. `axes` on decorator (for root-state obs: `len(axes)` gives dimension)
3. `asset_cfg.joint_ids` count (for joint-state obs)

```python
# Root state: dimension derived from axes (no out_dim needed)
@generic_io_descriptor_warp(axes=["X", "Y", "Z"], observation_type="RootState", ...)
def base_lin_vel(env, out, asset_cfg=...): ...

# Body state: out_dim required (per-body component count varies)
@generic_io_descriptor_warp(out_dim="body:7", observation_type="BodyState", ...)
def body_pose_w(env, out, asset_cfg=...): ...

# Cross-manager: out_dim sentinel queries manager at init time
@generic_io_descriptor_warp(out_dim="command", observation_type="Command", ...)
def generated_commands(env, out, command_name: str): ...

# Custom task obs: explicit int
@generic_io_descriptor_warp(out_dim=2, observation_type="RootState")
def base_yaw_roll(env, out, asset_cfg=...): ...
```

Supported `out_dim` values: `int`, `"joint"`, `"body:N"`, `"command"`, `"action"`.

### Class-based term pattern

```python
class my_reward(ManagerTermBase):
    def __init__(self, env, cfg):
        # Torch ops OK here (one-time setup)
        # Cache persistent warp arrays
        self._gear_wp = wp.from_torch(gear_tensor)

    def reset(self, env_mask=None):
        # Warp kernel for reset
        wp.launch(kernel=_reset_kernel, ...)

    def __call__(self, env, out, **params):
        # Pure wp.launch only -- no torch ops
        wp.launch(kernel=_compute_kernel, ..., inputs=[self._gear_wp, ...])
```

### Joint subset: mask vs ids

```python
# Stable: asset_cfg.joint_ids (int list)
# Experimental: asset_cfg.joint_mask (wp.array(dtype=wp.bool))
```

### Action class changes

```python
# process_actions: (actions: torch.Tensor) -> (actions: wp.array, action_offset: int)
# reset: (env_ids: Sequence[int]) -> (mask: wp.array(dtype=wp.bool))
# joint targeting: joint_ids= -> joint_mask=
```

### Buffer management

- Pre-allocate all output buffers in `__init__` (persistent pointers for graph capture)
- No dynamic tensor creation in per-step functions
- Per-joint constants stored as 1D arrays, indexed by `j` inside kernels
