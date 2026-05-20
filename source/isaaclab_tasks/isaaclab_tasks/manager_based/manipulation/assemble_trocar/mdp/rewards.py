# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

__all__ = [
    "AssembleTrocarState",
    "update_task_stage",
    "lift_trocars_reward",
    "trocar_tip_alignment_reward",
    "trocar_insertion_reward",
    "trocar_placement_reward",
]


@dataclass
class AssembleTrocarState:
    """Namespaced task state for the assemble-trocar environment.

    Holds per-env stage tracking, reward caches, and debug bookkeeping.
    Attached to the env as ``env.assemble_trocar_state`` and initialised
    lazily on first access via :func:`get_assemble_trocar_state`.

    Stage semantics:
        0 - Initial (need to lift)
        1 - Lifted (need to find hole / tip alignment)
        2 - Hole found (need to insert / push in)
        3 - Inserted (need to place)
        4 - Placed (task complete)
    """

    task_stage: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Sparse-reward previous-stage trackers (one per reward term)
    prev_stage_lift: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_tip: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_insert: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_place: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Dense-reward locked caches
    lift_reward_locked: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tip_reward_locked: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    insertion_reward_locked: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    placement_reward_locked: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    # Cached tip offsets (populated on first call to get_trocar_tip_position)
    tip_offset_trocar_1: torch.Tensor | None = None
    tip_offset_trocar_2: torch.Tensor | None = None
    # Debug throttle
    last_debug_print_step: int = -1


def get_assemble_trocar_state(env: ManagerBasedRLEnv) -> AssembleTrocarState:
    """Get or lazily initialise the :class:`AssembleTrocarState` on *env*."""
    if not hasattr(env, "assemble_trocar_state"):
        s = AssembleTrocarState(
            task_stage=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            prev_stage_lift=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            prev_stage_tip=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            prev_stage_insert=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            prev_stage_place=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            lift_reward_locked=torch.zeros(env.num_envs, device=env.device),
            tip_reward_locked=torch.zeros(env.num_envs, device=env.device),
            insertion_reward_locked=torch.zeros(env.num_envs, device=env.device),
            placement_reward_locked=torch.zeros(env.num_envs, device=env.device),
        )
        env.assemble_trocar_state = s
    return env.assemble_trocar_state


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current per-env task stage tensor."""
    return get_assemble_trocar_state(env).task_stage


def should_print_debug(env: ManagerBasedRLEnv, print_interval: int = 50, print_log: bool = True) -> bool:
    """Check if debug info should be logged based on episode step counter."""
    if not print_log:
        return False
    if not hasattr(env, "episode_length_buf"):
        return False

    current_step = env.episode_length_buf[0].item()
    if current_step == 0 or current_step % print_interval != 0:
        return False

    state = get_assemble_trocar_state(env)
    if state.last_debug_print_step == current_step:
        return False

    state.last_debug_print_step = current_step
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
    Returns a zero-valued tensor (num_envs,) so it can be used as a
    weight=0 reward term to run before the actual reward terms.
    """
    state = get_assemble_trocar_state(env)
    stage = state.task_stage

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    pos1 = obj1.data.root_pos_w.torch
    pos2 = obj2.data.root_pos_w.torch
    quat1 = obj1.data.root_quat_w.torch
    quat2 = obj2.data.root_quat_w.torch
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
                logger.debug("Env %d: Stage %d → %d", env_id, old_stage[env_id].item(), stage[env_id].item())

    state.task_stage = stage
    return torch.zeros(env.num_envs, device=env.device)


def lift_trocars_reward(
    env: ManagerBasedRLEnv,
    table_height: float = 0.85483,
    lift_threshold: float = 0.05,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for lifting both trocars above the table.

    Only active in Stage 0. Once completed, this reward is locked at the achieved value.

    Args:
        use_sparse_reward: If True, only give reward (1.0) when stage transitions from 0->1.
                          If False, give continuous reward based on current state.
        print_log: If True, log debug information.
    """
    s = get_assemble_trocar_state(env)
    stage = s.task_stage

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    pos1 = obj1.data.root_pos_w.torch
    pos2 = obj2.data.root_pos_w.torch
    target_z = table_height + lift_threshold

    is_lifted_1 = pos1[:, 2] > target_z
    is_lifted_2 = pos2[:, 2] > target_z
    both_lifted = is_lifted_1 & is_lifted_2

    if use_sparse_reward:
        stage_just_completed = (s.prev_stage_lift == 0) & (stage >= 1)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )
        s.prev_stage_lift = stage.clone()
    else:
        current_reward = both_lifted.float()
        s.lift_reward_locked = torch.where(
            (stage >= 1) & (s.lift_reward_locked == 0),
            current_reward,
            s.lift_reward_locked,
        )
        reward = torch.where(stage == 0, current_reward, s.lift_reward_locked)

    if should_print_debug(env, print_log=print_log):
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        logger.debug(
            " Stage: %d | Lift (%s): %.2f | z1: %.3f | z2: %.3f",
            stage[0].item(),
            mode_str,
            reward[0].item(),
            pos1[0, 2],
            pos2[0, 2],
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

    # Cache the tip offset to avoid recalculating every step.
    # The local offset from root to tip is a static geometric property of the USD
    # asset and is identical across all replicated envs. We read it once from env_0's
    # USD prim, then apply it per-env at runtime using each env's dynamic root pose.
    s = get_assemble_trocar_state(env)
    cache_attr = f"tip_offset_{asset_cfg.name}"
    tip_offset_local = getattr(s, cache_attr, None)

    if tip_offset_local is None:
        usd_stage = env.scene.stage

        if asset_cfg.name == "trocar_1":
            tip_path = "/World/envs/env_0/trocar_1/Trocar002/White_pos"
            root_path = "/World/envs/env_0/trocar_1"
        elif asset_cfg.name == "trocar_2":
            tip_path = "/World/envs/env_0/trocar_2/DisposableLaparoscopicPunctureDevice001/Red_pos"
            root_path = "/World/envs/env_0/trocar_2"
        else:
            raise ValueError(f"Invalid asset configuration: {asset_cfg.name}")

        tip_prim = usd_stage.GetPrimAtPath(tip_path)
        root_prim = usd_stage.GetPrimAtPath(root_path)

        if not tip_prim.IsValid():
            logger.warning("Tip prim not found at %s, using zero offset", tip_path)
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

            root_quat_inv = root_rotation_quat.GetInverse()
            tip_offset_local_gf = root_quat_inv.Transform(tip_offset_world)

            tip_offset_local = torch.tensor(
                [tip_offset_local_gf[0], tip_offset_local_gf[1], tip_offset_local_gf[2]],
                dtype=torch.float32,
                device=env.device,
            )

            logger.debug("Cached tip offset for %s: %s", asset_cfg.name, tip_offset_local)

        setattr(s, cache_attr, tip_offset_local)

    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w = obj.data.root_pos_w.torch  # Shape: (num_envs, 3)
    root_quat_w = obj.data.root_quat_w.torch  # Shape: (num_envs, 4) XYZW

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
    s = get_assemble_trocar_state(env)
    stage = s.task_stage

    tip_pos1 = get_trocar_tip_position(env, asset_cfg1)
    tip_pos2 = get_trocar_tip_position(env, asset_cfg2)
    tip_dist = torch.norm(tip_pos1 - tip_pos2, dim=-1)

    if use_sparse_reward:
        stage_just_completed = (s.prev_stage_tip == 1) & (stage >= 2)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )
        s.prev_stage_tip = stage.clone()
    else:
        tip_reward = torch.exp(-torch.square(tip_dist) / (2 * tip_dist_std**2))
        s.tip_reward_locked = torch.where(
            (stage >= 2) & (s.tip_reward_locked == 0),
            tip_reward,
            s.tip_reward_locked,
        )
        reward = torch.where(
            stage < 1,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 1, tip_reward, s.tip_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 1:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        logger.debug(
            "   Stage 1 (Find Hole, %s): tip_pos_1=(%.3f, %.3f, %.3f)"
            " | tip_pos_2=(%.3f, %.3f, %.3f) | tip_d=%.4f | reward=%.3f",
            mode_str,
            tip_pos1[0, 0],
            tip_pos1[0, 1],
            tip_pos1[0, 2],
            tip_pos2[0, 0],
            tip_pos2[0, 1],
            tip_pos2[0, 2],
            tip_dist[0].item(),
            reward[0].item(),
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
    s = get_assemble_trocar_state(env)
    stage = s.task_stage

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    pos1 = obj1.data.root_pos_w.torch
    quat1 = obj1.data.root_quat_w.torch
    pos2 = obj2.data.root_pos_w.torch
    quat2 = obj2.data.root_quat_w.torch
    center_dist = torch.norm(pos1 - pos2, dim=-1)

    target_axis1 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    target_axis2 = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)

    axis1 = quat_apply(quat1, target_axis1)
    axis2 = quat_apply(quat2, target_axis2)

    dot_prod = torch.sum(axis1 * axis2, dim=-1)
    abs_dot = torch.clamp(torch.abs(dot_prod), max=1.0)
    angle = torch.acos(abs_dot)
    is_parallel = angle < angle_threshold

    if use_sparse_reward:
        stage_just_completed = (s.prev_stage_insert == 2) & (stage >= 3)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )
        s.prev_stage_insert = stage.clone()
    else:
        excess_angle = torch.clamp(angle - angle_threshold, min=0.0)
        align_reward = torch.exp(-torch.square(excess_angle) / (2 * angle_std**2))
        center_reward = torch.exp(-torch.square(center_dist) / (2 * center_dist_std**2))
        center_reward = torch.where(is_parallel, center_reward, torch.zeros_like(center_reward))
        insertion_reward = align_reward * center_reward

        s.insertion_reward_locked = torch.where(
            (stage >= 3) & (s.insertion_reward_locked == 0),
            insertion_reward,
            s.insertion_reward_locked,
        )
        reward = torch.where(
            stage < 2,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 2, insertion_reward, s.insertion_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 2:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        logger.debug(
            "   Stage 2 (Push In, %s): angle=%.3f | center_d=%.4f | is_parallel=%s | reward=%.3f",
            mode_str,
            angle[0].item(),
            center_dist[0].item(),
            is_parallel[0].item(),
            reward[0].item(),
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
    s = get_assemble_trocar_state(env)
    stage = s.task_stage

    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    pos1 = obj1.data.root_pos_w.torch
    pos2 = obj2.data.root_pos_w.torch
    env_origins = env.scene.env_origins

    curr_x_min = env_origins[:, 0] + min(x_min, x_max)
    curr_x_max = env_origins[:, 0] + max(x_min, x_max)
    curr_y_min = env_origins[:, 1] + min(y_min, y_max)
    curr_y_max = env_origins[:, 1] + max(y_min, y_max)

    in_zone_1 = (
        (pos1[:, 0] >= curr_x_min)
        & (pos1[:, 0] <= curr_x_max)
        & (pos1[:, 1] >= curr_y_min)
        & (pos1[:, 1] <= curr_y_max)
        & (pos1[:, 2] < z_min)
    )
    in_zone_2 = (
        (pos2[:, 0] >= curr_x_min)
        & (pos2[:, 0] <= curr_x_max)
        & (pos2[:, 1] >= curr_y_min)
        & (pos2[:, 1] <= curr_y_max)
        & (pos2[:, 2] < z_min)
    )
    both_in_zone = in_zone_1 & in_zone_2

    if use_sparse_reward:
        stage_just_completed = (s.prev_stage_place == 3) & (stage >= 4)
        reward = torch.where(
            stage_just_completed,
            torch.ones(env.num_envs, device=env.device) / env.step_dt,
            torch.zeros(env.num_envs, device=env.device),
        )
        s.prev_stage_place = stage.clone()
    else:
        placement_reward = both_in_zone.float()
        s.placement_reward_locked = torch.where(
            (stage >= 4) & (s.placement_reward_locked == 0),
            placement_reward,
            s.placement_reward_locked,
        )
        reward = torch.where(
            stage < 3,
            torch.zeros(env.num_envs, device=env.device),
            torch.where(stage == 3, placement_reward, s.placement_reward_locked),
        )

    # Debug info
    if should_print_debug(env, print_log=print_log) and stage[0].item() == 3:
        mode_str = "Sparse" if use_sparse_reward else "Dense"
        logger.debug(
            "   Stage 3 (Placement, %s): in_zone=%s | z1=%.3f | z2=%.3f",
            mode_str,
            both_in_zone[0].item(),
            pos1[0, 2],
            pos2[0, 2],
        )

    return reward
