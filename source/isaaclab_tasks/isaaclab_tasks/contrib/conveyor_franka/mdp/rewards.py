# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Progress utilities and sparse completion rewards for conveyor transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .kinematics import end_effector_pose
from .reset_events import CUBE_COUNT, CUBE_REST_Z, TRANSFER_X, side_inner_y

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ConveyorTransferCommand


def transfer_potential(
    cube_positions: torch.Tensor,
    tool_positions: torch.Tensor,
    finger_positions: torch.Tensor,
    source_side_ids: torch.Tensor,
) -> torch.Tensor:
    """Return a monotonic pickup-to-release shaping potential."""
    source_y = side_inner_y(source_side_ids)
    target_y = side_inner_y(1 - source_side_ids)
    target_position = torch.stack(
        (
            torch.full_like(source_y, TRANSFER_X),
            target_y,
            torch.full_like(source_y, CUBE_REST_Z),
        ),
        dim=1,
    )
    tool_distance = torch.linalg.vector_norm(tool_positions - cube_positions, dim=1)
    reach = torch.exp(-12.0 * tool_distance)
    gripper_closure = torch.clamp((0.04 - torch.amin(finger_positions, dim=1)) / 0.021, min=0.0, max=1.0)
    grasp = gripper_closure * torch.exp(-25.0 * tool_distance)
    lift = torch.clamp((cube_positions[:, 2] - CUBE_REST_Z) / 0.14, min=0.0, max=1.0)
    direction_denominator = (target_y - source_y).clamp(min=-1.0, max=1.0)
    crossing = (cube_positions[:, 1] - source_y) / direction_denominator
    crossing = torch.clamp(crossing, min=0.0, max=1.0)
    target_distance = torch.linalg.vector_norm(cube_positions - target_position, dim=1)
    target = torch.exp(-14.0 * target_distance)
    transport = crossing * torch.maximum(torch.clamp(2.0 * lift, max=1.0), target)
    released = (torch.amin(finger_positions, dim=1) > 0.027).float() * target
    return 0.5 * reach + 0.75 * grasp + 1.25 * lift + 2.0 * transport + 2.0 * target + released


def current_transfer_potential(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
    command: ConveyorTransferCommand | None = None,
) -> torch.Tensor:
    """Gather current task state and evaluate the shaping potential."""
    if command is None:
        command = env.command_manager.get_term(command_name)
    cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    index = command.target_cube_ids.view(env.num_envs, 1, 1).expand(-1, 1, 3)
    active_position = torch.gather(positions, 1, index).squeeze(1) - env.scene.env_origins
    tool_position, _ = end_effector_pose(env)
    tool_position = tool_position - env.scene.env_origins
    robot: Articulation = env.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
    finger_positions = robot.data.joint_pos.torch[:, finger_ids]
    return transfer_potential(active_position, tool_position, finger_positions, command.source_side_ids)


def transfer_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Return one on each stable transfer-completion transition."""
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    return command.new_success.float()


def terminal_failure(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Return one for non-timeout terminal failures."""
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    return (env.reset_terminated & ~command.pending_success).float()


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Penalize one named raw action term."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(action), dim=1)


def finite_action_rate_l2(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...] = ("arm_action", "gripper_action"),
) -> torch.Tensor:
    """Penalize changes between the finite policy commands accepted by action terms."""
    if not action_names:
        raise ValueError("At least one action term is required for the action-rate reward.")
    terms = tuple(env.action_manager.get_term(name) for name in action_names)
    action = torch.cat(tuple(term.raw_actions for term in terms), dim=1)
    previous_action = torch.cat(tuple(term.previous_actions for term in terms), dim=1)
    return torch.sum(torch.square(action - previous_action), dim=1)


def physical_cube_acquisition_mask(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
    command: ConveyorTransferCommand | None = None,
    minimum_lift: float = 0.025,
    maximum_tool_distance: float = 0.075,
    maximum_finger_position: float = 0.030,
) -> torch.Tensor:
    """Return physically closed, lifted, tool-local commanded-cube grasps."""
    if minimum_lift <= 0.0 or maximum_tool_distance <= 0.0 or maximum_finger_position <= 0.0:
        raise ValueError("Physical acquisition thresholds must be positive.")
    if command is None:
        command = env.command_manager.get_term(command_name)
    cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    index = command.target_cube_ids.view(env.num_envs, 1, 1).expand(-1, 1, 3)
    active_position = torch.gather(positions, 1, index).squeeze(1)
    tool_position, _ = end_effector_pose(env)
    robot: Articulation = env.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
    finger_positions = robot.data.joint_pos.torch[:, finger_ids]
    local_cube_z = active_position[:, 2] - env.scene.env_origins[:, 2]
    return (
        (local_cube_z >= CUBE_REST_Z + minimum_lift)
        & (torch.linalg.vector_norm(active_position - tool_position, dim=1) <= maximum_tool_distance)
        & (torch.amax(finger_positions, dim=1) <= maximum_finger_position)
    )


def finite_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"]),
    maximum_velocity: float = 3.0,
) -> torch.Tensor:
    """Penalize bounded arm velocity while sanitizing divergent states."""
    if maximum_velocity <= 0.0:
        raise ValueError("maximum_velocity must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = torch.nan_to_num(velocity, nan=0.0, posinf=maximum_velocity, neginf=-maximum_velocity)
    return torch.sum(torch.square(torch.clamp(velocity, -maximum_velocity, maximum_velocity)), dim=1)
