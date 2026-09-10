# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat(
        (cube_1.data.root_pos_w.torch, cube_2.data.root_pos_w.torch, cube_3.data.root_pos_w.torch),
        dim=1,
    )


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)

    return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
    """The orientation of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat(
        (
            cube_1.data.root_quat_w.torch,
            cube_2.data.root_quat_w.torch,
            cube_3.data.root_quat_w.torch,
        ),
        dim=1,
    )


def instance_randomize_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(cube_1.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = cube_1.data.root_pos_w.torch
    cube_1_quat_w = cube_1.data.root_quat_w.torch

    cube_2_pos_w = cube_2.data.root_pos_w.torch
    cube_2_quat_w = cube_2.data.root_quat_w.torch

    cube_3_pos_w = cube_3.data.root_pos_w.torch
    cube_3_quat_w = cube_3.data.root_quat_w.torch

    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def tool_axes(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """Return continuous configured-tool x/z axes."""
    from .robot_state import end_effector_pose

    _, orientation = end_effector_pose(
        env,
        robot_cfg=robot_cfg,
        body_name=tool_body_name,
        body_offset=tool_offset,
    )
    rotation = math_utils.matrix_from_quat(orientation)
    return torch.cat((rotation[:, :, 0], rotation[:, :, 2]), dim=1)


def tool_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """Return the configured tool-center linear/angular velocity."""
    from .robot_state import end_effector_velocity

    return end_effector_velocity(
        env,
        robot_cfg=robot_cfg,
        body_name=tool_body_name,
        body_offset=tool_offset,
    )


def body_positions_relative_to_tool(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """Return selected articulation-body positions relative to a configured tool center.

    The selected body origins are kept in the configured ``body_cfg`` order and
    expressed as world-aligned displacement vectors from the tool center. This
    provides compact fingertip geometry without requiring contact sensors.
    """
    from .robot_state import end_effector_pose

    robot: Articulation = env.scene[body_cfg.name]
    body_positions = robot.data.body_pos_w.torch[:, body_cfg.body_ids]
    tool_position, _ = end_effector_pose(
        env,
        robot_cfg=robot_cfg,
        body_name=tool_body_name,
        body_offset=tool_offset,
    )
    return (body_positions - tool_position.unsqueeze(1)).flatten(start_dim=1)


def franka_ee_axes(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return continuous Franka tool x/z axes."""
    return tool_axes(env, robot_cfg=robot_cfg)


def franka_ee_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the Franka tool-center position relative to its environment.

    This is a deployable kinematic quantity: on hardware it is computed from
    joint encoders and the calibrated Franka model, just like the tool axes.
    """
    from .robot_state import end_effector_pose

    tool_position, _ = end_effector_pose(env, robot_cfg=robot_cfg)
    return tool_position - env.scene.env_origins


def franka_ee_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the Franka tool-center linear/angular velocity."""
    return tool_velocity(env, robot_cfg=robot_cfg)


def role_conditioned_stack_obs(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_height: float = 0.04,
    tool_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    grasp_pair_tool_offsets: tuple[tuple[float, float, float], ...] | None = None,
    xy_threshold: float = 0.025,
    height_threshold: float = 0.012,
    role_order: tuple[int, int, int] = (0, 1, 2),
) -> torch.Tensor:
    """Return stable base/side-role tracks with randomized cube colors.

    The reset table stores one episode-long mapping from physical stack roles
    to cube assets. Gathering through that mapping makes the input exactly
    invariant to reset-time color permutations while keeping each role in a
    stable slot through grasp, lift, and placement. Role zero is the spatially
    central base; roles one and two are interchangeable side cubes.

    The 64-entry layout contains role-local positions, tool-relative
    positions, up axes, spatial velocities, every directed pair-placement
    error, and stack progress.
    """
    from .rewards import order_invariant_stack_progress
    from .robot_state import end_effector_pose, grasp_pair_end_effector_pose

    reset_state = get_stack_reset_runtime_state(env)

    cubes: tuple[RigidObject, ...] = tuple(env.scene[cfg.name] for cfg in cube_cfgs)
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    quaternions = torch.stack(tuple(cube.data.root_quat_w.torch for cube in cubes), dim=1)
    velocities = torch.stack(tuple(cube.data.root_vel_w.torch for cube in cubes), dim=1)
    if tuple(sorted(role_order)) != (0, 1, 2):
        raise ValueError("role_order must be a permutation of (0, 1, 2).")
    role_to_cube = reset_state.role_to_cube.long()
    role_to_cube = role_to_cube[:, role_to_cube.new_tensor(role_order)]

    def by_role(values: torch.Tensor) -> torch.Tensor:
        gather_index = role_to_cube.view(env.num_envs, 3, *([1] * (values.ndim - 2))).expand_as(values)
        return torch.gather(values, 1, gather_index)

    local_positions = by_role(positions) - env.scene.env_origins.unsqueeze(1)
    role_quaternions = by_role(quaternions)
    role_velocities = by_role(velocities)
    if grasp_pair_tool_offsets is None:
        tool_position, _ = end_effector_pose(
            env,
            robot_cfg=robot_cfg,
            body_name=tool_body_name,
            body_offset=tool_offset,
        )
    else:
        tool_position, _ = grasp_pair_end_effector_pose(
            env,
            robot_cfg=robot_cfg,
            body_name=tool_body_name,
            tool_offsets_by_pair=grasp_pair_tool_offsets,
        )
    rotation = math_utils.matrix_from_quat(role_quaternions.flatten(end_dim=1)).view(env.num_envs, 3, 3, 3)
    up_axes = rotation[..., 2]
    tool_relative = local_positions - (tool_position - env.scene.env_origins).unsqueeze(1)

    pair_errors = []
    vertical_offset = local_positions.new_tensor((0.0, 0.0, cube_height))
    for upper_id in range(3):
        for lower_id in range(3):
            if upper_id != lower_id:
                pair_errors.append(local_positions[:, lower_id] + vertical_offset - local_positions[:, upper_id])
    progress = (
        order_invariant_stack_progress(
            env,
            cube_cfgs=cube_cfgs,
            xy_threshold=xy_threshold,
            height_threshold=height_threshold,
            cube_height=cube_height,
        ).unsqueeze(1)
        / 2.0
    )
    return torch.cat(
        (
            local_positions.flatten(start_dim=1),
            tool_relative.flatten(start_dim=1),
            up_axes.flatten(start_dim=1),
            role_velocities.flatten(start_dim=1),
            torch.cat(pair_errors, dim=1),
            progress,
        ),
        dim=1,
    )


def stack_reset_recipe_one_hot(env: ManagerBasedRLEnv, recipe_count: int = 9) -> torch.Tensor:
    """Return privileged reset-recipe labels used only to balance cloning loss."""
    if recipe_count <= 0:
        raise ValueError("recipe_count must be positive.")
    reset_state = get_stack_reset_runtime_state(env)
    recipe_ids = reset_state.recipes.long().clamp(min=0, max=recipe_count - 1)
    return torch.nn.functional.one_hot(recipe_ids, num_classes=recipe_count).float()


def role_conditioned_cube_x_axes(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
) -> torch.Tensor:
    """Return each role-assigned cube's local x-axis in world coordinates.

    :func:`role_conditioned_stack_obs` already supplies each cube's local
    z-axis. Adding the x-axis makes yaw observable while preserving the same
    episode-long physical-role mapping and its permutation invariance.
    """
    reset_state = get_stack_reset_runtime_state(env)

    cubes: tuple[RigidObject, ...] = tuple(env.scene[cfg.name] for cfg in cube_cfgs)
    quaternions = torch.stack(tuple(cube.data.root_quat_w.torch for cube in cubes), dim=1)
    role_to_cube = reset_state.role_to_cube.long()
    gather_index = role_to_cube.unsqueeze(-1).expand_as(quaternions)
    role_quaternions = torch.gather(quaternions, 1, gather_index)
    rotation = math_utils.matrix_from_quat(role_quaternions.flatten(end_dim=1)).view(env.num_envs, 3, 3, 3)
    return rotation[..., 0].flatten(start_dim=1)


def grasp_pair_gripper_posture(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    joint_names_by_pair: tuple[tuple[str, ...], ...],
    open_joint_positions_by_pair: tuple[tuple[float, ...], ...],
    closed_joint_positions_by_pair: tuple[tuple[float, ...], ...],
    finger_joint_counts: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Return normalized closure of the episode's active two-finger pair."""
    from .robot_state import grasp_pair_posture_closure

    return grasp_pair_posture_closure(
        env,
        robot_cfg=robot_cfg,
        joint_names_by_pair=joint_names_by_pair,
        open_joint_positions_by_pair=open_joint_positions_by_pair,
        closed_joint_positions_by_pair=closed_joint_positions_by_pair,
        finger_joint_counts=finger_joint_counts,
    )


def grasp_pair_tool_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body_name: str = "palm_link",
    *,
    tool_offsets_by_pair: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """Return active grasp-pair tool-center linear/angular velocity."""
    from .robot_state import grasp_pair_end_effector_velocity

    return grasp_pair_end_effector_velocity(
        env,
        robot_cfg=robot_cfg,
        body_name=tool_body_name,
        tool_offsets_by_pair=tool_offsets_by_pair,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.body_link_pos_w.torch[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        cube_1_quat_w.append(cube_1.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.body_link_quat_w.torch[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w.torch[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w.torch[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(wp.to_torch(surface_gripper.state).view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            if len(gripper_joint_ids) == 1:
                # single-jaw gripper (e.g. SO-101)
                return robot.data.joint_pos.torch[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            assert len(gripper_joint_ids) == 2, (
                "Observation gripper_pos only supports single- or parallel-jaw (2-joint) grippers for now"
            )
            finger_joint_1 = robot.data.joint_pos.torch[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos.torch[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w.torch
    end_effector_pos = ee_frame.data.target_pos_w.torch[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = wp.to_torch(surface_gripper.state).view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) >= 1, "Observations require at least one gripper joint"

            # Grasped: the end-effector is close to the object and every gripper joint has moved
            # away from the open position (i.e. the jaws have closed on the object).
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            grasped = pose_diff < diff_threshold
            for joint_id in gripper_joint_ids:
                grasped = torch.logical_and(
                    grasped,
                    torch.abs(robot.data.joint_pos.torch[:, joint_id] - open_val) > env.cfg.gripper_threshold,
                )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w.torch - lower_object.data.root_pos_w.torch
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = wp.to_torch(surface_gripper.state).view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) >= 1, "Observations require at least one gripper joint"
            # Stacked also requires the gripper to be released (every jaw back at the open value).
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            for joint_id in gripper_joint_ids:
                stacked = torch.logical_and(
                    torch.isclose(
                        robot.data.joint_pos.torch[:, joint_id],
                        open_val,
                        atol=1e-4,
                        rtol=1e-4,
                    ),
                    stacked,
                )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def cube_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the cubes in the robot base frame."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_cube_1_world = cube_1.data.root_pos_w.torch
    pos_cube_2_world = cube_2.data.root_pos_w.torch
    pos_cube_3_world = cube_3.data.root_pos_w.torch

    quat_cube_1_world = cube_1.data.root_quat_w.torch
    quat_cube_2_world = cube_2.data.root_quat_w.torch
    quat_cube_3_world = cube_3.data.root_quat_w.torch

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_1_world, quat_cube_1_world
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_2_world, quat_cube_2_world
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_3_world, quat_cube_3_world
    )

    pos_cubes_base = torch.cat((pos_cube_1_base, pos_cube_2_base, pos_cube_3_base), dim=1)
    quat_cubes_base = torch.cat((quat_cube_1_base, quat_cube_2_base, quat_cube_3_base), dim=1)

    if return_key == "pos":
        return pos_cubes_base
    elif return_key == "quat":
        return quat_cubes_base
    else:
        return torch.cat((pos_cubes_base, quat_cubes_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations,
    and add abs gripper pos and quat in robot base frame
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper pos,
        gripper quat,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch

    cube_1_pos_w = cube_1.data.root_pos_w.torch
    cube_1_quat_w = cube_1.data.root_quat_w.torch

    cube_2_pos_w = cube_2.data.root_pos_w.torch
    cube_2_quat_w = cube_2.data.root_quat_w.torch

    cube_3_pos_w = cube_3.data.root_pos_w.torch
    cube_3_quat_w = cube_3.data.root_quat_w.torch

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_1_pos_w, cube_1_quat_w
    )
    pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_2_pos_w, cube_2_quat_w
    )
    pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_3_pos_w, cube_3_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_cube_1_base,
            quat_cube_1_base,
            pos_cube_2_base,
            quat_cube_2_base,
            pos_cube_3_base,
            quat_cube_3_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w.torch[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    else:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
