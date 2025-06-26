# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils import math as math_utils
from isaaclab.managers import ManagerTermBase
import numpy as np
import trimesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def lift_v0(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = torch.tensor([-0.5, 0., 0.75], device=env.device).repeat(env.num_envs, 1)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.abs(des_pos_w[:, 2] - object.data.root_pos_w[:, 2])
    # rewarded if the object is lifted above the threshold
    return torch.exp(-std * distance)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance_v0(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    points = env.reward_manager.get_term_cfg('lift').func.points
    num_points = points.shape[1]
    object_pos_w = object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
    object_rot_w = object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
    object_point_cloud_w = math_utils.quat_apply(object_rot_w, points) + object_pos_w

    lifted = ((env.episode_length_buf > 10) & torch.all(object_point_cloud_w[..., 2] > min_height, dim=1))

    return torch.where(lifted, torch.exp(-std * distance), 0)

class lifted(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        from pxr import UsdGeom
        from isaaclab.sim.utils import get_all_matching_child_prims
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.num_points: int = cfg.params.get("num_points", 10)
        self.visualize = cfg.params.get("visualize", True)
        self.object: RigidObject = env.scene[self.object_cfg.name]

        # uncomment to visualize
        if self.visualize:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            ray_cfg.markers["hit"].radius = 0.001
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points = torch.zeros((env.num_envs, self.num_points, 3), device=self.device)

        for i in range(env.num_envs):
            object_cfg = self.object.cfg
            prim_path = object_cfg.prim_path
            prim = get_all_matching_child_prims(prim_path.replace(".*", str(i)), predicate=lambda prim: prim.GetTypeName() == "Mesh")[0]
            mesh = UsdGeom.Mesh(prim)
            vertices = np.array(mesh.GetPointsAttr().Get())

            # load face‐counts and face‐indices
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()

            # triangulate "poly" faces into a (F,3) array
            faces = []
            it = iter(indices)
            for cnt in counts:
                poly = [next(it) for _ in range(cnt)]
                # fan‐triangulate
                for k in range(1, cnt-1):
                    faces.append([poly[0], poly[k], poly[k+1]])

            faces = np.array(faces, dtype=np.int64)

            # build trimesh and sample
            tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            samples, __ = tm.sample(self.num_points, return_index=True)
            self.points[i] = torch.from_numpy(samples).to(self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        visualize: bool = True
    ):
        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
        object_rot_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)

        # apply rotation + translation
        object_point_cloud_w = math_utils.quat_apply(object_rot_w, self.points) + object_pos_w

        if visualize:
            self.visualizer.visualize(translations=object_point_cloud_w.reshape(-1, 3))
        return ((env.episode_length_buf > 10) & torch.all(object_point_cloud_w[..., 2] > min_height, dim=1)).float()
