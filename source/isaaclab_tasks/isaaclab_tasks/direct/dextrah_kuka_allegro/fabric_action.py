# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .action_cfg import FabricActionCfg

class FabricAction(ActionTerm):
    cfg: FabricActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: FabricActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        palm_max_angle = [0., 0.7, 1., -2.4 + cfg.palm_rot_range, cfg.palm_rot_range, 3.14 + cfg.palm_rot_range]
        palm_min_angle = [-1.2, -0.7, 0., -2.4 - cfg.palm_rot_range, -cfg.palm_rot_range, 3.14 - cfg.palm_rot_range]
        self.palm_dim, self.hand_dim = len(palm_min_angle), len(cfg.pca_feat_min)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.palm_pose_upper_limits = torch.tensor(palm_max_angle, device=self.device)
        self.palm_pose_lower_limits = torch.tensor(palm_min_angle, device=self.device)
        self.hand_pca_upper_limits = torch.tensor(cfg.pca_feat_max, device=self.device)
        self.hand_pca_lower_limits = torch.tensor(cfg.pca_feat_min, device=self.device)
        self.fabric_robot_cfg = self.cfg.fabric_robot_scene_cfg
        self.fabric_robot_cfg.resolve(env.scene)

        self._setup_geometric_fabrics()
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path=get_robot_urdf_path("kuka_allegro", "kuka_allegro"),
            link_names=self.cfg.fabric_robot_scene_cfg.body_names,
            batch_size=self.num_envs,
            device=self.device
        )
        self.reset(env_ids=torch.arange(env.num_envs))
    
    @property
    def action_dim(self) -> int:
        return self.palm_dim + self.hand_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.cat([self.fabric_q, self.cfg.pd_vel_factor * self.fabric_qd], dim=1)


    def _setup_geometric_fabrics(self) -> None:
        # Set the warp cache directory based on device int
        initialize_warp(self.device[-1])
        self.world_model = WorldMeshesModel(self.num_envs, 20, self.device, world_filename='kuka_allegro_boxes')
        self.kuka_allegro_fabric = KukaAllegroPoseFabric(self.num_envs, self.device, self.cfg.fabrics_dt, True)
        self.kuka_allegro_integrator = DisplacementIntegrator(self.kuka_allegro_fabric)

        # Pre-allocate fabrics states
        self.fabric_q = self._asset.data.default_joint_pos.clone()
        self.fabric_qd = torch.zeros(self.num_envs, self.kuka_allegro_fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, self.kuka_allegro_fabric.num_joints, device=self.device)

        # Pre-allocate target tensors
        self.hand_pca_targets = torch.zeros(self.num_envs, self.hand_dim, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, self.palm_dim, device=self.device)  # (XYZ, Euler ZYX)

        # Fabric cspace damping gain
        self.fabric_damping_gain = self.cfg.fabric_damping_gain * torch.ones(self.num_envs, 1, device=self.device)
        # This reports back handles to the meshes which is consumed by the fabric for collision avoidance
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        # Graph capture
        self.inputs = [self.hand_pca_targets, self.palm_pose_targets, "euler_zyx", # actions in
                       self.fabric_q.detach(), self.fabric_qd.detach(), # fabric state
                       self.object_ids, self.object_indicator, # world model
                       self.fabric_damping_gain]

        # Capture the forward pass of evaluating the fabric given the inputs and integrating one step
        self.g, self.fabric_q_new, self.fabric_qd_new, self.fabric_qdd_new =\
            capture_fabric(self.kuka_allegro_fabric,
                           self.fabric_q,
                           self.fabric_qd,
                           self.fabric_qdd,
                           self.cfg.fabrics_dt,
                           self.kuka_allegro_integrator,
                           self.inputs,
                           self.device)

        # Preallocate tensors for fabrics state meant to go into obs buffer
        self.fabric_q_for_obs = torch.clone(self.fabric_q)
        self.fabric_qd_for_obs = torch.clone(self.fabric_qd)
        self.fabric_qdd_for_obs = torch.clone(self.fabric_qdd)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        # In-place update to targets
        palm_lim_range = self.palm_pose_upper_limits - self.palm_pose_lower_limits
        hand_lim_range = self.hand_pca_upper_limits - self.hand_pca_lower_limits
        palm_target = (self._raw_actions[:, :self.palm_dim] + 1) / 2 * palm_lim_range + self.palm_pose_lower_limits
        hand_target = (self._raw_actions[:, self.palm_dim:] + 1) / 2 * hand_lim_range + self.hand_pca_lower_limits
        self.palm_pose_targets.copy_(palm_target)
        self.hand_pca_targets.copy_(hand_target)

        # Update fabric cspace damping gain based on ADR
        self.fabric_damping_gain = self.cfg.fabric_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        # Replay through the fabric graph with the latest action inputs
        for i in range(self.cfg.fabric_decimation):
            self.g.replay()
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)

    def apply_actions(self) -> None:
        vel_target = self.cfg.pd_vel_factor * self.fabric_qd
        self._asset.set_joint_position_target(self.fabric_q, joint_ids=self.fabric_robot_cfg.joint_ids)
        self._asset.set_joint_velocity_target(vel_target, joint_ids=self.fabric_robot_cfg.joint_ids)
        
    def reset(self, env_ids: Sequence[int] | None):
        self.fabric_q[env_ids] = self._asset.data.joint_pos[env_ids[:, None], self.fabric_robot_cfg.joint_ids].clone()
        self.fabric_qd[env_ids] = self._asset.data.joint_vel[env_ids[:, None], self.fabric_robot_cfg.joint_ids].clone()
        
        self.fabric_damping_gain = self._env.dextrah_adr.get_custom_param_value("fabric_damping", "gain")
        self.cfg.pd_vel_factor = self._env.dextrah_adr.get_custom_param_value("pd_targets", "velocity_target_factor")
