# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab.utils import math as math_utils
from isaaclab.managers import ManagerTermBase
import numpy as np
import trimesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))

def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


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

class lifted(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        from pxr import UsdGeom
        import hashlib
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
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RewardPointCloud")
            ray_cfg.markers["hit"].radius = 0.001
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points = torch.zeros((env.num_envs, self.num_points, 3), device=self.device)
        self.lifted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for i in range(env.num_envs):
            cache = getattr(env, "pointcloud_cache", None)
            if cache is None:
                cache = {}
                setattr(env, "pointcloud_cache", cache)
            object_cfg = self.object.cfg
            prim_path = object_cfg.prim_path
            prim = get_all_matching_child_prims(prim_path.replace(".*", str(i)), predicate=lambda prim: prim.GetTypeName() == "Mesh")[0]
            mesh = UsdGeom.Mesh(prim)
            vertices = np.array(mesh.GetPointsAttr().Get())
            
            key = hashlib.sha256()
            key.update(vertices.tobytes())
            geom_id = key.hexdigest()
            
            if geom_id in cache:
                samples = cache[geom_id]
            else:
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
                cache[geom_id] = samples
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
        self.lifted = (contacts(env, 1.0)) & (torch.all(object_point_cloud_w[..., 2] > min_height, dim=1))
        return self.lifted.float()


class Cubelifted(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        import isaacsim.core.utils.prims as prim_utils
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.visualize = cfg.params.get("visualize", True)
        self.object: RigidObject = env.scene[self.object_cfg.name]

        # uncomment to visualize
        if self.visualize:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            ray_cfg.markers["hit"].radius = 0.001
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points = torch.zeros((env.num_envs, 8, 3), device=self.device)
        self.lifted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for i in range(env.num_envs):
            object_cfg = self.object.cfg
            prim_path = object_cfg.prim_path

            prim = prim_utils.get_prim_at_path(prim_path.replace(".*", str(i)))
            # prim_spec = Sdf.CreatePrimInLayer(stage_utils.get_current_stage().GetRootLayer(), )
            scale = prim.GetAttribute("xformOp:scale").Get()
            self.points[i, 0] = torch.tensor([-scale[0] / 2, -scale[1] / 2, -scale[2] / 2], device=env.device)
            self.points[i, 1] = torch.tensor([-scale[0] / 2, -scale[1] / 2, scale[2] / 2], device=env.device)
            self.points[i, 2] = torch.tensor([-scale[0] / 2, scale[1] / 2, scale[2] / 2], device=env.device)
            self.points[i, 3] = torch.tensor([-scale[0] / 2, scale[1] / 2, -scale[2] / 2], device=env.device)
            self.points[i, 4] = torch.tensor([scale[0] / 2, -scale[1] / 2, -scale[2] / 2], device=env.device)
            self.points[i, 5] = torch.tensor([scale[0] / 2, -scale[1] / 2, scale[2] / 2], device=env.device)
            self.points[i, 6] = torch.tensor([scale[0] / 2, scale[1] / 2, -scale[2] / 2], device=env.device)
            self.points[i, 7] = torch.tensor([scale[0] / 2, scale[1] / 2, scale[2] / 2], device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        visualize: bool = True
    ):
        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, 8, 1)
        object_rot_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, 8, 1)

        # apply rotation + translation
        object_point_cloud_w = math_utils.quat_apply(object_rot_w, self.points) + object_pos_w

        if visualize:
            self.visualizer.visualize(translations=object_point_cloud_w.reshape(-1, 3))
        self.lifted = (contacts(env, 1.0)) & (torch.all(object_point_cloud_w[..., 2] > min_height, dim=1))
        return self.lifted.float()



def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_contact_sensor"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_contact_sensor"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_contact_sensor"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_contact_sensor"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    # print(f"force_matric: {contact_sensor.data.force_matrix_w}, net_force: {contact_sensor.data.net_forces_w}")
    good_contact_cond1 = (thumb_contact_mag > threshold) & ((index_contact_mag > threshold) |\
                        (middle_contact_mag > threshold) | (ring_contact_mag > threshold))

    return good_contact_cond1

def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7])
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        return (1 - torch.tanh(pos_dist / pos_std))
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    lifted = env.reward_manager.get_term_cfg('lift').func.lifted
    return (1 - torch.tanh(distance / std)) * lifted.float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the orientation using the tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)
    lifted = env.reward_manager.get_term_cfg('lift').func.lifted

    return (1 - torch.tanh(quat_distance / std)) * lifted.float()

