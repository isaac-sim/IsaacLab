# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import trimesh
import numpy as np

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position and quaternion of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_quat_w
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)

def projected_joint_force(env: ManagerBasedRLEnv, asset_cfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_dof_projected_joint_forces()[:, asset_cfg.joint_ids]

def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].clone().view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1) 
    return out.view(env.num_envs, -1)


class object_scale(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        import isaacsim.core.utils.prims as prim_utils
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.scale = torch.zeros((env.num_envs, 3), device=self.device)

        for i in range(env.num_envs):
            object_cfg = self.object.cfg
            prim_path = object_cfg.prim_path
            prim = prim_utils.get_prim_at_path(prim_path.replace(".*", str(i)))
            # prim_spec = Sdf.CreatePrimInLayer(stage_utils.get_current_stage().GetRootLayer(), )
            scale = prim.GetAttribute("xformOp:scale").Get()
            self.scale[i] = torch.tensor(scale, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        return self.scale


class object_point_cloud_b(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        from pxr import UsdGeom
        import hashlib
        from isaaclab.sim.utils import get_all_matching_child_prims
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 10)
        self.visualize = cfg.params.get("visualize", True)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        # uncomment to visualize
        if self.visualize:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.001
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points = torch.zeros((env.num_envs, self.num_points, 3), device=self.device)
        self.lifted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for i in range(env.num_envs):
            cache = getattr(env, "pointcloud_cache", None)
            if cache is None:
                cache = {}
                setattr(env, "pointcloud_cache", cache)
            prim_path = self.object.cfg.prim_path
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
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        visualize: bool = True
    ):
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
        # apply rotation + translation
        object_point_cloud_pos_w = quat_apply(object_quat_w, self.points) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=object_point_cloud_pos_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, object_point_cloud_pos_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1)


def fingers_contact_force_w(env: ManagerBasedRLEnv) -> torch.Tensor:
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

    return torch.cat((thumb_contact, index_contact, middle_contact, ring_contact), dim=1)