# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "update_task_stage",
    "lift_trocars_reward",
    "trocar_tip_alignment_reward",
    "trocar_insertion_reward",
    "trocar_placement_reward",
]


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get or initialize task stage tracker for each environment.

    Stage 0: Initial state (need to lift)
    Stage 1: Lifted (need to find hole - tip alignment)
    Stage 2: Hole found (need to insert - push in)
    Stage 3: Inserted (need to place)
    Stage 4: Placed (task complete)

    Returns:
        torch.Tensor: Current stage for each environment (num_envs,)
    """
    if not hasattr(env, "_task_stage"):
        env._task_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._task_stage


def should_print_debug(env: ManagerBasedRLEnv, print_interval: int = 50, print_log: bool = True) -> bool:
    """Check if debug info should be printed based on episode step counter.

    Uses the environment's built-in episode_length_buf to track steps,
    and ensures each step only prints once (first call).

    Args:
        env: Environment instance
        print_interval: Print every N steps

    Returns:
        bool: True if should print (only on first call per step)
    """
    # Hard gate: allow callers to disable all logs from this module.
    if not print_log:
        return False

    # Use environment's episode step counter (standard in Isaac Lab)
    if not hasattr(env, "episode_length_buf"):
        return False

    current_step = env.episode_length_buf[0].item()

    # Skip step 0 and non-interval steps
    if current_step == 0 or current_step % print_interval != 0:
        return False

    # Track last printed step to avoid duplicate prints in same step
    if not hasattr(env, "_last_debug_print_step"):
        env._last_debug_print_step = -1

    # Only print once per step (on first function call)
    if env._last_debug_print_step == current_step:
        return False  # Already printed this step

    # Mark this step as printed and return True
    env._last_debug_print_step = current_step
    return True


def update_task_stage(
    env: ManagerBasedRLEnv,
    asset_cfg1: SceneEntityCfg,
    asset_cfg2: SceneEntityCfg,
    table_height: float = 0.85483,
    lift_threshold: float = 0.05,
    tip_align_threshold: float = 0.015,
    insertion_dist_threshold: float = 0.03,
    insertion_angle_threshold: float = 0.15,
    placement_x_min: float = -1.8,
    placement_x_max: float = -1.4,
    placement_y_min: float = 1.5,
    placement_y_max: float = 1.8,
    placement_z_min: float = 0.9,
    print_log: bool = False,
) -> torch.Tensor:
    """Update task stage based on current state.

    This function checks conditions and advances stages automatically.
    Once a stage is completed, it never goes back.
    """
    stage = get_task_stage(env)

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    pos1 = wp.to_torch(obj1.data.root_pos_w)
    pos2 = wp.to_torch(obj2.data.root_pos_w)
    quat1 = wp.to_torch(obj1.data.root_quat_w)
    quat2 = wp.to_torch(obj2.data.root_quat_w)

    # Store old stage to detect changes (BEFORE any stage transitions)
    old_stage = stage.clone()

    # Stage 0 -> 1: Check if lifted
    target_z = table_height + lift_threshold
    is_lifted_1 = pos1[:, 2] > target_z
    is_lifted_2 = pos2[:, 2] > target_z
    both_lifted = is_lifted_1 & is_lifted_2
    stage = torch.where((stage == 0) & both_lifted, torch.ones_like(stage), stage)

    # Stage 1 -> 2: Check if tips are aligned (hole found)
    # Get tip positions
    tip_pos1 = get_trocar_tip_position(env, asset_cfg1)
    tip_pos2 = get_trocar_tip_position(env, asset_cfg2)
    tip_dist = torch.norm(tip_pos1 - tip_pos2, dim=-1)

    # Tip alignment success
    tip_aligned = tip_dist < tip_align_threshold
    stage = torch.where((stage == 1) & tip_aligned, torch.full_like(stage, 2), stage)

    # Stage 2 -> 3: Check if inserted (parallel + center close)
    # Get center distance
    center_dist = torch.norm(pos1 - pos2, dim=-1)

    # Check alignment
    target_axis1 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    target_axis2 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    axis1 = quat_apply(quat1, target_axis1)
    axis2 = quat_apply(quat2, target_axis2)
    dot_prod = torch.sum(axis1 * axis2, dim=-1)
    abs_dot = torch.clamp(torch.abs(dot_prod), max=1.0)
    angle = torch.acos(abs_dot)

    # Insertion success: parallel + center close
    is_parallel = angle < insertion_angle_threshold
    center_close = center_dist < insertion_dist_threshold
    is_inserted = is_parallel & center_close

    stage = torch.where((stage == 2) & is_inserted, torch.full_like(stage, 3), stage)

    # Stage 3 -> 4: Check if placed in target zone
    # Get environment origins to handle multi-env spatial offsets
    env_origins = env.scene.env_origins  # shape: (num_envs, 3)

    # Adjust target zone relative to each environment's origin
    curr_x_min = env_origins[:, 0] + min(placement_x_min, placement_x_max)  # (num_envs,)
    curr_x_max = env_origins[:, 0] + max(placement_x_min, placement_x_max)
    curr_y_min = env_origins[:, 1] + min(placement_y_min, placement_y_max)
    curr_y_max = env_origins[:, 1] + max(placement_y_min, placement_y_max)

    in_zone_1 = (
        (pos1[:, 0] >= curr_x_min)
        & (pos1[:, 0] <= curr_x_max)
        & (pos1[:, 1] >= curr_y_min)
        & (pos1[:, 1] <= curr_y_max)
        & (pos1[:, 2] < placement_z_min)
    )
    in_zone_2 = (
        (pos2[:, 0] >= curr_x_min)
        & (pos2[:, 0] <= curr_x_max)
        & (pos2[:, 1] >= curr_y_min)
        & (pos2[:, 1] <= curr_y_max)
        & (pos2[:, 2] < placement_z_min)
    )
    both_in_zone = in_zone_1 & in_zone_2
    stage = torch.where((stage == 3) & both_in_zone, torch.full_like(stage, 4), stage)

    # Print stage transitions (AFTER all stage transitions - always print when stage changes)
    if print_log and (stage != old_stage).any():
        for env_id in range(env.num_envs):
            if stage[env_id] != old_stage[env_id]:
                print(f"Env {env_id}: Stage {old_stage[env_id].item()} → {stage[env_id].item()}")

    env._task_stage = stage
    return stage


def lift_trocars_reward(
    env: ManagerBasedRLEnv,
    table_height: float = 0.85483,
    lift_threshold: float = 0.05,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    tip_align_threshold: float = 0.015,
    insertion_dist_threshold: float = 0.035,
    insertion_angle_threshold: float = 0.17,
    placement_x_min: float = -1.8,
    placement_x_max: float = -1.4,
    placement_y_min: float = 1.5,
    placement_y_max: float = 1.8,
    placement_z_min: float = 0.9,
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for lifting both trocars above the table.

    Only active in Stage 0. Once completed, this reward is locked at the achieved value.

    Args:
        use_sparse_reward: If True, only give reward (1.0) when stage transitions from 0->1.
                          If False, give continuous reward based on current state.
        print_log: If True, print debug information.
    """
    # Update task stage first - check ALL stage transitions once per step
    stage = update_task_stage(
        env,
        asset_cfg1,
        asset_cfg2,
        table_height,
        lift_threshold,
        tip_align_threshold,
        insertion_dist_threshold,
        insertion_angle_threshold,
        placement_x_min,
        placement_x_max,
        placement_y_min,
        placement_y_max,
        placement_z_min,
        print_log=print_log,
    )

    # Get the rigid objects from the scene
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    # Get positions (num_envs, 3)
    pos1 = wp.to_torch(obj1.data.root_pos_w)
    pos2 = wp.to_torch(obj2.data.root_pos_w)

    target_z = table_height + lift_threshold

    # Check if lifted
    is_lifted_1 = pos1[:, 2] > target_z
    is_lifted_2 = pos2[:, 2] > target_z
    both_lifted = is_lifted_1 & is_lifted_2

    if use_sparse_reward:
        # Sparse reward mode: give 1.0 ONLY when stage transitions from 0 to 1
        # Track previous stage
        if not hasattr(env, "_prev_stage_lift"):
            # Initialize prev_stage to current stage to avoid false positives on first call
            env._prev_stage_lift = stage.clone()

        # Reward = 1.0 only on transition step (prev_stage=0, curr_stage=1)
        stage_just_completed = (env._prev_stage_lift == 0) & (stage >= 1)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )

        # Update previous stage for next step
        env._prev_stage_lift = stage.clone()
    else:
        # Dense reward mode: continuous reward with locking for continuity
        # Initialize locked reward cache
        if not hasattr(env, "_lift_reward_locked"):
            env._lift_reward_locked = torch.zeros(env.num_envs, device=env.device)

        # Current reward value
        current_reward = both_lifted.float()

        # Lock the reward when transitioning to stage 1
        env._lift_reward_locked = torch.where(
            (stage >= 1) & (env._lift_reward_locked == 0),
            current_reward,  # Lock at current value when stage changes
            env._lift_reward_locked,
        )

        # Stage 0: give reward based on current state
        # Stage >= 1: return locked value (preserves continuity)
        reward = torch.where(stage == 0, current_reward, env._lift_reward_locked)

    # Print debug info periodically (every 50 steps)
    if should_print_debug(env, print_log=print_log):
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        print(
            f" Stage: {stage[0].item()}"
            f" | Lift ({mode_str}): {reward[0].item():.2f}"
            f" | z1: {pos1[0, 2]:.3f}"
            f" | z2: {pos2[0, 2]:.3f}"
        )

    return reward


def get_trocar_tip_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("trocar_1"),
) -> torch.Tensor:
    """Get trocar tip position (White_pos or Red_pos) in world coordinates.

    Calculates tip world position using trocar root's dynamic position and rotation,
    plus the tip's relative offset.

    Args:
        env: Environment instance
        asset_cfg: Trocar asset configuration (trocar_1 or trocar_2)

    Returns:
        torch.Tensor: Shape (num_envs, 3) - Position in world coordinates
    """
    from pxr import Gf, Usd, UsdGeom

    import isaaclab.utils.math as math_utils

    # Cache the tip offset to avoid recalculating every step
    cache_key = f"_tip_offset_{asset_cfg.name}"
    if not hasattr(env, cache_key):
        # Get tip's local offset relative to root (only calculate once, from USD)
        # Note: Local offset is the same in all environments (same asset structure), so get from env_0
        stage = env.scene.stage

        if asset_cfg.name == "trocar_1":
            tip_path = "/World/envs/env_0/trocar_1/Trocar002/White_pos"
            root_path = "/World/envs/env_0/trocar_1"
        elif asset_cfg.name == "trocar_2":
            tip_path = "/World/envs/env_0/trocar_2/DisposableLaparoscopicPunctureDevice001/Red_pos"
            root_path = "/World/envs/env_0/trocar_2"
        else:
            raise ValueError(f"Invalid asset configuration: {asset_cfg.name}")

        tip_prim = stage.GetPrimAtPath(tip_path)
        root_prim = stage.GetPrimAtPath(root_path)

        if not tip_prim.IsValid():
            print(f"Warning: Tip prim not found at {tip_path}, using zero offset")
            tip_offset_local = torch.zeros(3, dtype=torch.float32, device=env.device)
        else:
            tip_xform = UsdGeom.Xformable(tip_prim)
            root_xform = UsdGeom.Xformable(root_prim)

            tip_world_transform = tip_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            root_world_transform = root_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

            tip_world_pos = tip_world_transform.ExtractTranslation()
            root_world_pos = root_world_transform.ExtractTranslation()

            root_rotation_mat = root_world_transform.ExtractRotationMatrix()
            root_rotation_quat = root_rotation_mat.ExtractRotation().GetQuat()

            tip_offset_world = Gf.Vec3d(
                tip_world_pos[0] - root_world_pos[0],
                tip_world_pos[1] - root_world_pos[1],
                tip_world_pos[2] - root_world_pos[2],
            )

            # Convert world coordinate offset to root's local coordinate system
            # Using inverse of root rotation: local_offset = root_quat^{-1} * world_offset
            root_quat_inv = root_rotation_quat.GetInverse()
            tip_offset_local_gf = root_quat_inv.Transform(tip_offset_world)

            tip_offset_local = torch.tensor(
                [tip_offset_local_gf[0], tip_offset_local_gf[1], tip_offset_local_gf[2]],
                dtype=torch.float32,
                device=env.device,
            )

            print(f"Cached tip offset for {asset_cfg.name}: {tip_offset_local}")

        # Cache the offset
        setattr(env, cache_key, tip_offset_local)

    tip_offset_local = getattr(env, cache_key)

    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w = wp.to_torch(obj.data.root_pos_w)  # Shape: (num_envs, 3)
    root_quat_w = wp.to_torch(obj.data.root_quat_w)  # Shape: (num_envs, 4) XYZW

    tip_offset_local_batch = tip_offset_local.unsqueeze(0).repeat(env.num_envs, 1)

    tip_offset_world = math_utils.quat_apply(root_quat_w, tip_offset_local_batch)
    tip_pos_world = root_pos_w + tip_offset_world

    return tip_pos_world  # Shape: (num_envs, 3)


def trocar_tip_alignment_reward(
    env: ManagerBasedRLEnv,
    tip_dist_std: float = 0.02,  # Std for tip distance reward
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for aligning trocar tips (Stage 1: Finding the hole).

    Reward based on tip distance - encourages bringing tips close together.

    Only active in Stage 1. Once completed (stage >= 2), this reward is locked at the achieved value.

    Args:
        env: Environment instance
        tip_dist_std: Standard deviation for tip distance reward shaping
        asset_cfg1: Configuration for trocar 1
        asset_cfg2: Configuration for trocar 2
        use_sparse_reward: If True, only give reward (1.0) when stage >= 2.
                          If False, give continuous reward based on tip distance.
        print_log: If True, print debug information.

    Returns:
        torch.Tensor: Reward tensor (num_envs,)
    """
    # Read current stage
    stage = get_task_stage(env)

    # Get tip positions
    tip_pos1 = get_trocar_tip_position(env, asset_cfg1)
    tip_pos2 = get_trocar_tip_position(env, asset_cfg2)

    # Calculate tip distance
    tip_dist = torch.norm(tip_pos1 - tip_pos2, dim=-1)  # (num_envs,)

    if use_sparse_reward:
        # Sparse reward mode: give 1.0 ONLY when stage transitions from 1 to 2
        # Track previous stage
        if not hasattr(env, "_prev_stage_tip"):
            # Initialize prev_stage to current stage to avoid false positives on first call
            env._prev_stage_tip = stage.clone()

        # Reward = 1.0 only on transition step (prev_stage=1, curr_stage=2)
        stage_just_completed = (env._prev_stage_tip == 1) & (stage >= 2)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )

        # Update previous stage for next step
        env._prev_stage_tip = stage.clone()
    else:
        # Dense reward mode: continuous reward with locking for continuity
        # Reward: exponential decay based on tip distance
        tip_reward = torch.exp(-torch.square(tip_dist) / (2 * tip_dist_std**2))

        # Initialize locked reward cache
        if not hasattr(env, "_tip_reward_locked"):
            env._tip_reward_locked = torch.zeros(env.num_envs, device=env.device)

        # Lock the reward when transitioning to stage 2
        env._tip_reward_locked = torch.where(
            (stage >= 2) & (env._tip_reward_locked == 0),
            tip_reward,  # Lock at current value when stage changes
            env._tip_reward_locked,
        )

        # Stage 0: no reward (not lifted yet)
        # Stage 1: give reward based on tip distance
        # Stage >= 2: return locked value (preserves continuity)
        reward = torch.where(
            stage < 1,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 1, tip_reward, env._tip_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 1:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        print(
            f"   └─ Stage 1 (Find Hole, {mode_str}):"
            f" tip_pos_1=({tip_pos1[0, 0]:.3f}, {tip_pos1[0, 1]:.3f}, {tip_pos1[0, 2]:.3f})"
            f" | tip_pos_2=({tip_pos2[0, 0]:.3f}, {tip_pos2[0, 1]:.3f}, {tip_pos2[0, 2]:.3f})"
            f" | tip_d={tip_dist[0].item():.4f}"
            f" | reward={reward[0].item():.3f}"
        )

    return reward


def trocar_insertion_reward(
    env: ManagerBasedRLEnv,
    angle_std: float = 0.2,  # Std for angle alignment reward
    angle_threshold: float = 0.15,  # Tolerance for parallelism (radians)
    center_dist_std: float = 0.05,  # Std for center distance reward
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for inserting trocar_2 into trocar_1 (Stage 2: Pushing in).

    Reward based on:
    1. Orientation alignment (parallelism)
    2. Center distance (pushing in)

    Only active in Stage 2. Once completed (stage >= 3), this reward is locked at the achieved value.

    Args:
        env: Environment instance
        angle_std: Standard deviation for angle reward shaping
        angle_threshold: Angle threshold for parallelism (radians)
        center_dist_std: Standard deviation for center distance reward shaping
        asset_cfg1: Configuration for trocar 1
        asset_cfg2: Configuration for trocar 2
        use_sparse_reward: If True, only give reward (1.0) when stage >= 3.
                          If False (default), give continuous reward based on alignment and distance.
        print_log: If True, print debug information.
    Returns:
        torch.Tensor: Reward tensor (num_envs,)
    """
    # Read current stage
    stage = get_task_stage(env)

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    # Positions and Rotations
    pos1 = wp.to_torch(obj1.data.root_pos_w)
    quat1 = wp.to_torch(obj1.data.root_quat_w)
    pos2 = wp.to_torch(obj2.data.root_pos_w)
    quat2 = wp.to_torch(obj2.data.root_quat_w)

    # Calculate center distance
    center_dist = torch.norm(pos1 - pos2, dim=-1)  # (num_envs,)

    # Calculate alignment (parallelism)
    target_axis1 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    target_axis2 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)

    axis1 = quat_apply(quat1, target_axis1)
    axis2 = quat_apply(quat2, target_axis2)

    dot_prod = torch.sum(axis1 * axis2, dim=-1)
    abs_dot = torch.clamp(torch.abs(dot_prod), max=1.0)
    angle = torch.acos(abs_dot)

    is_parallel = angle < angle_threshold

    if use_sparse_reward:
        # Sparse reward mode: give 1.0 ONLY when stage transitions from 2 to 3
        # Track previous stage
        if not hasattr(env, "_prev_stage_insert"):
            # Initialize prev_stage to current stage to avoid false positives on first call
            env._prev_stage_insert = stage.clone()

        # Reward = 1.0 only on transition step (prev_stage=2, curr_stage=3)
        stage_just_completed = (env._prev_stage_insert == 2) & (stage >= 3)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )

        # Update previous stage for next step
        env._prev_stage_insert = stage.clone()
    else:
        # Dense reward mode: continuous reward with locking for continuity
        # Reward component 1: Alignment (parallelism)
        excess_angle = torch.clamp(angle - angle_threshold, min=0.0)
        align_reward = torch.exp(-torch.square(excess_angle) / (2 * angle_std**2))

        # Reward component 2: Center distance (pushing in)
        # Only reward center distance if already parallel
        center_reward = torch.exp(-torch.square(center_dist) / (2 * center_dist_std**2))
        center_reward = torch.where(is_parallel, center_reward, torch.zeros_like(center_reward))

        # Combined reward: alignment * center_distance
        insertion_reward = align_reward * center_reward

        # Initialize locked reward cache
        if not hasattr(env, "_insertion_reward_locked"):
            env._insertion_reward_locked = torch.zeros(env.num_envs, device=env.device)

        # Lock the reward when transitioning to stage 3
        env._insertion_reward_locked = torch.where(
            (stage >= 3) & (env._insertion_reward_locked == 0),
            insertion_reward,  # Lock at current value when stage changes
            env._insertion_reward_locked,
        )

        # Stage < 2: no reward (not ready yet)
        # Stage 2: give reward based on current state
        # Stage >= 3: return locked value (preserves continuity)
        reward = torch.where(
            stage < 2,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 2, insertion_reward, env._insertion_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 2:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        print(
            f"   └─ Stage 2 (Push In, {mode_str}): angle={angle[0].item():.3f} | "
            f"center_d={center_dist[0].item():.4f} | "
            f"is_parallel={is_parallel.item()} | reward={reward[0].item():.3f}"
        )

    return reward


def trocar_placement_reward(
    env: ManagerBasedRLEnv,
    x_min: float = -1.8,
    x_max: float = -1.4,
    y_min: float = 1.5,
    y_max: float = 1.8,
    z_min: float = 0.9,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for placing both trocars in the target tray region (Stage 3).

    Only active in Stage 3. Once completed (stage >= 4), this reward is locked at the achieved value.

    Args:
        env: Environment instance
        x_min, x_max: X bounds of target zone (relative to env origin)
        y_min, y_max: Y bounds of target zone (relative to env origin)
        z_min: Z threshold (below this is considered placed)
        asset_cfg1: Configuration for trocar 1
        asset_cfg2: Configuration for trocar 2
        use_sparse_reward: If True, only give reward (1.0) when stage >= 4.
                          If False (default), give continuous reward based on placement status.
        print_log: If True, print debug information.

    Returns:
        torch.Tensor: Reward tensor (num_envs,)
    """
    # Read current stage
    stage = get_task_stage(env)

    # Get rigid objects
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    # Get root positions (num_envs, 3)
    pos1 = wp.to_torch(obj1.data.root_pos_w)
    pos2 = wp.to_torch(obj2.data.root_pos_w)

    # Get environment origins to handle multi-env spatial offsets
    env_origins = env.scene.env_origins  # shape: (num_envs, 3)

    # Adjust target zone relative to each environment's origin
    curr_x_min = env_origins[:, 0] + min(x_min, x_max)  # shape: (num_envs,)
    curr_x_max = env_origins[:, 0] + max(x_min, x_max)
    curr_y_min = env_origins[:, 1] + min(y_min, y_max)
    curr_y_max = env_origins[:, 1] + max(y_min, y_max)

    # Check bounds for object 1
    in_x_1 = (pos1[:, 0] >= curr_x_min) & (pos1[:, 0] <= curr_x_max)
    in_y_1 = (pos1[:, 1] >= curr_y_min) & (pos1[:, 1] <= curr_y_max)
    in_z_1 = pos1[:, 2] < z_min
    in_zone_1 = in_x_1 & in_y_1 & in_z_1

    # Check bounds for object 2
    in_x_2 = (pos2[:, 0] >= curr_x_min) & (pos2[:, 0] <= curr_x_max)
    in_y_2 = (pos2[:, 1] >= curr_y_min) & (pos2[:, 1] <= curr_y_max)
    in_z_2 = pos2[:, 2] < z_min
    in_zone_2 = in_x_2 & in_y_2 & in_z_2

    both_in_zone = in_zone_1 & in_zone_2

    if use_sparse_reward:
        # Sparse reward mode: give 1.0 ONLY when stage transitions from 3 to 4
        # Track previous stage
        if not hasattr(env, "_prev_stage_place"):
            # Initialize prev_stage to current stage to avoid false positives on first call
            env._prev_stage_place = stage.clone()

        # Reward = 1.0 only on transition step (prev_stage=3, curr_stage=4)
        stage_just_completed = (env._prev_stage_place == 3) & (stage >= 4)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )

        # Update previous stage for next step
        env._prev_stage_place = stage.clone()
    else:
        # Dense reward mode: continuous reward with locking for continuity
        placement_reward = both_in_zone.float()

        # Initialize locked reward cache
        if not hasattr(env, "_placement_reward_locked"):
            env._placement_reward_locked = torch.zeros(env.num_envs, device=env.device)

        # Lock the reward when transitioning to stage 4
        env._placement_reward_locked = torch.where(
            (stage >= 4) & (env._placement_reward_locked == 0),
            placement_reward,  # Lock at current value when stage changes
            env._placement_reward_locked,
        )

        # Stage < 3: no reward (not inserted yet)
        # Stage 3: give reward based on current state
        # Stage >= 4: return locked value (preserves continuity)
        reward = torch.where(
            stage < 3,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 3, placement_reward, env._placement_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 3:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        print(
            f"   └─ Stage 3 (Placement, {mode_str}): in_zone={both_in_zone[0].item()} | "
            f"z1={pos1[0, 2]:.3f} | z2={pos2[0, 2]:.3f}"
        )

    return reward
