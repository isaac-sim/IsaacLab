# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for pick place surgical environment."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

__all__ = [
    "reset_tray_with_random_rotation",
    "reset_robot_to_default_joint_positions",
    "reset_task_stage",
]


def reset_task_stage(
    env: ManagerBasedRLEnv,
    env_mask: torch.Tensor,
    print_log: bool = False,
) -> None:
    """Reset task stage to 0 for specified environments.

    This should be called during environment reset events.
    Also resets all locked reward caches to maintain continuity.

    Args:
        env: The environment instance
        env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        print_log: If True, log debug information. Counting the reset environments waits on the device.
    """
    from .rewards import get_assemble_trocar_state

    s = get_assemble_trocar_state(env)
    s.task_stage.masked_fill_(env_mask, 0)

    # Reset dense-reward locked caches
    s.lift_reward_locked.masked_fill_(env_mask, 0)
    s.tip_reward_locked.masked_fill_(env_mask, 0)
    s.insertion_reward_locked.masked_fill_(env_mask, 0)
    s.placement_reward_locked.masked_fill_(env_mask, 0)

    # Reset sparse-reward previous-stage trackers
    s.prev_stage_lift.masked_fill_(env_mask, 0)
    s.prev_stage_tip.masked_fill_(env_mask, 0)
    s.prev_stage_insert.masked_fill_(env_mask, 0)
    s.prev_stage_place.masked_fill_(env_mask, 0)

    # Reset debug throttle
    s.last_debug_print_step = -1

    if print_log:
        logger.debug("Reset task stage for %d environment(s)", int(env_mask.sum()))


def reset_tray_with_random_rotation(
    env: ManagerBasedRLEnv,
    env_mask: torch.Tensor,
    tray_cfg: SceneEntityCfg,
    trocar_1_cfg: SceneEntityCfg,
    trocar_2_cfg: SceneEntityCfg,
    rotation_range: tuple[float, float] | float = (-5.0, 5.0),  # (min, max) degrees or ±value
    deterministic_per_env: bool = False,
    deterministic_seed: int | None = None,
):
    """Reset tray with random rotation while keeping relative positions of trocars.

    This function:
    1. Applies a random yaw rotation within rotation_range to the tray
    2. Rotates trocar_1 and trocar_2 around the tray center to maintain relative positions
    3. Uses separate pose/velocity writes to ensure instant teleportation (no interpolation)

    Args:
        env: The environment instance.
        env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        tray_cfg: Scene entity config for the tray.
        trocar_1_cfg: Scene entity config for trocar_1.
        trocar_2_cfg: Scene entity config for trocar_2.
        rotation_range: Rotation angle range in degrees. Can be:
            - tuple (min, max): Random rotation between min and max degrees
            - float value: Random rotation between -value and +value degrees
            Examples: (0, 10), (-5, 15), 5.0 (equivalent to (-5, 5))
    """
    num_envs = env.num_envs

    # Parse rotation_range parameter
    if isinstance(rotation_range, (tuple, list)):
        # User provided (min, max) range
        min_angle_deg, max_angle_deg = rotation_range[0], rotation_range[1]
    else:
        # User provided single value (symmetric range ±value)
        min_angle_deg, max_angle_deg = -rotation_range, rotation_range

    # Get assets
    tray = env.scene[tray_cfg.name]
    trocar_1 = env.scene[trocar_1_cfg.name]
    trocar_2 = env.scene[trocar_2_cfg.name]

    # Get default poses and velocities (local coordinates relative to env origin)
    tray_default_pose = tray.data.default_root_pose.torch.clone()
    trocar_1_default_pose = trocar_1.data.default_root_pose.torch.clone()
    trocar_2_default_pose = trocar_2.data.default_root_pose.torch.clone()

    env_origins = env.scene.env_origins  # (num_envs, 3)

    # Convert local coordinate to world coordinate
    tray_default_pose[:, :3] += env_origins
    trocar_1_default_pose[:, :3] += env_origins
    trocar_2_default_pose[:, :3] += env_origins

    # Tray center position (pivot point for rotation) - now in world coordinates
    tray_center = tray_default_pose[:, :3]  # (num_envs, 3)

    # Generate yaw angles (in radians)
    # Convert degrees to radians
    min_angle_rad = min_angle_deg * math.pi / 180.0
    max_angle_rad = max_angle_deg * math.pi / 180.0

    # Generate angles uniformly distributed in [min_angle, max_angle]
    if deterministic_per_env:
        # Derive a stable "random" number per env id, so each env gets a distinct yaw,
        # but it is repeatable across resets/runs given the same seed + env_id.
        #
        # If deterministic_seed is not provided, we tie it to torch's global seed.
        # IsaacLab typically seeds torch during env reset with the provided seed.
        if deterministic_seed is None:
            deterministic_seed = int(torch.initial_seed())
        u = _deterministic_uniform_0_1(env, deterministic_seed)  # (num_envs,)
    else:
        u = torch.rand(num_envs, device=env.device)
    random_yaw = u * (max_angle_rad - min_angle_rad) + min_angle_rad  # (num_envs,)

    # Create rotation quaternion for yaw (rotation around Z-axis)
    # XYZW: quat = [x, y, z, w] = [0, 0, sin(θ/2), cos(θ/2)]
    half_angle = random_yaw / 2.0
    delta_quat = torch.zeros(num_envs, 4, device=env.device)
    delta_quat[:, 2] = torch.sin(half_angle)  # z
    delta_quat[:, 3] = torch.cos(half_angle)  # w

    # Apply rotation to tray quaternion
    tray_new_quat = quat_mul(delta_quat, tray_default_pose[:, 3:7])

    # Update tray pose
    tray_new_pose = tray_default_pose.clone()
    tray_new_pose[:, 3:7] = tray_new_quat

    # Rotate trocar positions around tray center
    trocar_1_relative_pos = trocar_1_default_pose[:, :3] - tray_center
    trocar_2_relative_pos = trocar_2_default_pose[:, :3] - tray_center

    # Rotate relative positions using the delta quaternion
    trocar_1_new_relative_pos = quat_apply(delta_quat, trocar_1_relative_pos)
    trocar_2_new_relative_pos = quat_apply(delta_quat, trocar_2_relative_pos)

    # New absolute poses
    trocar_1_new_pose = trocar_1_default_pose.clone()
    trocar_2_new_pose = trocar_2_default_pose.clone()

    trocar_1_new_pose[:, :3] = tray_center + trocar_1_new_relative_pos
    trocar_2_new_pose[:, :3] = tray_center + trocar_2_new_relative_pos

    # Also rotate trocar orientations
    trocar_1_new_pose[:, 3:7] = quat_mul(delta_quat, trocar_1_default_pose[:, 3:7])
    trocar_2_new_pose[:, 3:7] = quat_mul(delta_quat, trocar_2_default_pose[:, 3:7])

    zero_velocity = torch.zeros(num_envs, 6, device=env.device)  # [lin_vel(3), ang_vel(3)]

    tray.write_root_pose_to_sim_mask(root_pose=tray_new_pose, env_mask=env_mask)
    trocar_1.write_root_pose_to_sim_mask(root_pose=trocar_1_new_pose, env_mask=env_mask)
    trocar_2.write_root_pose_to_sim_mask(root_pose=trocar_2_new_pose, env_mask=env_mask)

    tray.write_root_velocity_to_sim_mask(root_velocity=zero_velocity, env_mask=env_mask)
    trocar_1.write_root_velocity_to_sim_mask(root_velocity=zero_velocity, env_mask=env_mask)
    trocar_2.write_root_velocity_to_sim_mask(root_velocity=zero_velocity, env_mask=env_mask)


def _deterministic_uniform_0_1(env: ManagerBasedRLEnv, seed: int) -> torch.Tensor:
    """Deterministically map each env id to a float in [0, 1) via a seeded lookup table.

    We generate a length-(env.num_envs) random table with a local torch.Generator
    seeded by `seed`, then return it, so entry ``i`` belongs to env ``i``. This is deterministic and avoids
    uint64 bitwise ops (which may not be supported on CPU).
    """
    device = env.device
    num_envs = int(env.num_envs)
    seed = int(seed)

    cache = getattr(env, "_deterministic_u_table_cache", None)
    cache_key = (seed, num_envs, str(device))
    if cache is None or cache.get("key") != cache_key:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed & 0xFFFFFFFFFFFFFFFF)
        u_table = torch.rand((num_envs,), generator=gen, device=device, dtype=torch.float32)
        cache = {"key": cache_key, "u_table": u_table}
        setattr(env, "_deterministic_u_table_cache", cache)

    return cache["u_table"]


def reset_robot_to_default_joint_positions(
    env: ManagerBasedRLEnv,
    env_mask: torch.Tensor,
    robot_cfg: SceneEntityCfg,
):
    """Reset robot joint positions directly to default values.

    This function directly writes joint positions and velocities to the simulation,
    bypassing the PD controller. This prevents the "drive to target" behavior
    that causes arms to swing from 0 position to the target position.

    Args:
        env: The environment instance.
        env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        robot_cfg: Scene entity config for the robot.
    """
    # Get robot asset
    robot = env.scene[robot_cfg.name]

    # Get default joint positions and velocities
    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    default_joint_vel = robot.data.default_joint_vel.torch.clone()

    # Directly write joint state to simulation (bypasses PD controller)
    robot.write_joint_position_to_sim_mask(position=default_joint_pos, env_mask=env_mask)
    robot.write_joint_velocity_to_sim_mask(velocity=default_joint_vel, env_mask=env_mask)

    # Also reset root pose and velocity
    default_root_pose = robot.data.default_root_pose.torch.clone()
    default_root_vel = robot.data.default_root_vel.torch.clone()
    robot.write_root_pose_to_sim_mask(root_pose=default_root_pose, env_mask=env_mask)
    robot.write_root_velocity_to_sim_mask(root_velocity=default_root_vel, env_mask=env_mask)
