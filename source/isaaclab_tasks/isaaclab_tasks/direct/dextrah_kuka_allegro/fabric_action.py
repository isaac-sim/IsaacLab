# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg, ActionTerm
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from . import dextrah_kuka_allegro_constants as constants

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .fabric_action_cfg import FabricActionCfg

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


def compute_absolute_action(
    raw_actions: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:

    # Apply actions to hand
    absolute_action = scale(
        x=raw_actions,
        lower=lower_limits,
        upper=upper_limits,
    )
    absolute_action = tensor_clamp(
        t=absolute_action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return absolute_action

class FabricAction(ActionTerm):
    cfg: FabricActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: FabricActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.palm_pose_upper_limits = to_torch(constants.PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_pose_lower_limits = to_torch(constants.PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.hand_pca_upper_limits = to_torch(constants.HAND_PCA_MAXS, device=self.device)
        self.hand_pca_lower_limits = to_torch(constants.HAND_PCA_MINS, device=self.device)
        self._setup_geometric_fabrics()
        robot_dir_name = "kuka_allegro"
        robot_name = "kuka_allegro"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(self.urdf_path, cfg.hand_body_names, self.num_envs, self.device)
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)


        self.robot_hand_bodies_cfg = SceneEntityCfg(
            "robot",
            body_names=cfg.hand_body_names,
            joint_names=cfg.actuated_joint_names,
            preserve_order=True,
        )
        self.robot_hand_bodies_cfg.resolve(env.scene)
        
        # Robot noise
        self.robot_joint_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_noise_width = torch.zeros(self.num_envs, 1, device=self.device)

        self._reset_idx(env_ids=torch.arange(env.num_envs))
    
    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


    def _setup_geometric_fabrics(self) -> None:
        # Set the warp cache directory based on device int
        initialize_warp(self.device[-1])
        # This creates a world model that book keeps all the meshes
        # in the world, their pose, name, etc.
        print('Creating fabrics world-------------------------------')
        self.world_model = WorldMeshesModel(batch_size=self.num_envs,
                                            max_objects_per_env=20,
                                            device=self.device,
                                            world_filename='kuka_allegro_boxes')
        self.kuka_allegro_fabric = KukaAllegroPoseFabric(self.num_envs, self.device, self.cfg.fabrics_dt, graph_capturable=True)
        self.kuka_allegro_integrator = DisplacementIntegrator(self.kuka_allegro_fabric)

        # Pre-allocate fabrics states
        self.fabric_q = self._asset.data.default_joint_pos.clone()
        self.fabric_qd = torch.zeros(self.num_envs, self.kuka_allegro_fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, self.kuka_allegro_fabric.num_joints, device=self.device)

        # Pre-allocate target tensors
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)  # pca dim = 5
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)  # (origin, Euler ZYX)

        # Fabric cspace damping gain
        self.fabric_damping_gain = self.cfg.fabric_damping_gain * torch.ones(self.num_envs, 1, device=self.device)
        # This reports back handles to the meshes which is consumed
        # by the fabric for collision avoidance
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()


        # Graph capture
        # NOTE: elements of inputs must be in the same order as expected in the set_features function
        # of the fabric
        # Establish inputs
        self.inputs = [self.hand_pca_targets, self.palm_pose_targets, "euler_zyx", # actions in
                       self.fabric_q.detach(), self.fabric_qd.detach(), # fabric state
                       self.object_ids, self.object_indicator, # world model
                       self.fabric_damping_gain]
        # Capture the forward pass of evaluating the fabric given the inputs and integrating one step
        # in time
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
        # Update the palm pose and pca targets based on agent actions
        palm_actions = actions[:, : (constants.NUM_XYZ + constants.NUM_RPY)]
        hand_actions = actions[:, (constants.NUM_XYZ + constants.NUM_RPY) : (constants.NUM_HAND_PCA + constants.NUM_XYZ + constants.NUM_RPY)]
        # In-place update to targets
        self.palm_pose_targets.copy_(compute_absolute_action(palm_actions, self.palm_pose_lower_limits, self.palm_pose_upper_limits))
        self.hand_pca_targets.copy_(compute_absolute_action(hand_actions, self.hand_pca_lower_limits, self.hand_pca_upper_limits))

        # Update fabric cspace damping gain based on ADR
        self.fabric_damping_gain = self.cfg.fabric_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        # Replay through the fabric graph with the latest action inputs
        for i in range(self.cfg.fabric_decimation):
            # Evaluate the fabric via graph replay
            self.g.replay()
            # Update the fabric states
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)
        
        vel_scale = self.cfg.pd_vel_factor
        dof_pos_targets = torch.clone(self.fabric_q)
        dof_vel_targets = torch.clone(self.fabric_qd)
        self._processed_actions = torch.cat([dof_pos_targets, vel_scale * dof_vel_targets], dim=1)
    
    def apply_actions(self) -> None:
        # Set fabric states to position and velocity targets
        self._asset.set_joint_position_target(self._processed_actions[:, :self._asset.num_joints], joint_ids=self.robot_hand_bodies_cfg.joint_ids)
        self._asset.set_joint_velocity_target(self._processed_actions[:, self._asset.num_joints:], joint_ids=self.robot_hand_bodies_cfg.joint_ids)


    def _reset_idx(self, env_ids: Sequence[int] | None):
        self.reset(env_ids=env_ids)
        robot_joint_pos_bias_width = self.cfg.robot_joint_pos_bias_width * torch.rand(len(env_ids), device=self.device)
        robot_joint_vel_bias_width = self.cfg.robot_joint_vel_bias_width * torch.rand(len(env_ids), device=self.device)
        self.robot_joint_pos_bias[env_ids, 0] = robot_joint_pos_bias_width * (torch.rand(len(env_ids), device=self.device) - 0.5)
        self.robot_joint_vel_bias[env_ids, 0] = robot_joint_vel_bias_width * (torch.rand(len(env_ids), device=self.device) - 0.5)
        self.robot_joint_pos_noise_width[env_ids, 0] = self.cfg.robot_joint_pos_noise * torch.rand(len(env_ids), device=self.device)
        self.robot_joint_vel_noise_width[env_ids, 0] = self.cfg.robot_joint_vel_noise * torch.rand(len(env_ids), device=self.device)
        
        self.fabric_start_pos = self.fabric_q.clone()
        self.fabric_start_vel = self.fabric_qd.clone()
        self.fabric_start_pos[env_ids, :] = self._asset.data.joint_pos[env_ids[:, None], self.robot_hand_bodies_cfg.joint_ids].clone()
        self.fabric_start_vel[env_ids, :] = self._asset.data.joint_vel[env_ids[:, None], self.robot_hand_bodies_cfg.joint_ids].clone()
        self.fabric_q.copy_(self.fabric_start_pos)
        self.fabric_qd.copy_(self.fabric_start_vel)

        self.robot_dof_pos = self._asset.data.joint_pos[:, self.robot_hand_bodies_cfg.joint_ids]
        noise = 2. * (torch.rand_like(self.robot_dof_pos) - 0.5)
        self.robot_dof_pos_noisy = self.robot_dof_pos + self.robot_joint_pos_noise_width * noise + self.robot_joint_pos_bias

        self.robot_dof_vel = self._asset.data.joint_vel[:, self.robot_hand_bodies_cfg.joint_ids]
        noise = 2. * (torch.rand_like(self.robot_dof_pos) - 0.5)
        self.robot_dof_vel_noisy = self.robot_dof_vel + self.robot_joint_vel_noise_width * noise + self.robot_joint_vel_bias
        self.robot_dof_vel_noisy *= self.cfg.observation_annealing_coefficient

        self._asset.data.joint_pos[:, self.robot_hand_bodies_cfg.joint_ids].clone()
        self.hand_pos_noisy, hand_points_jac = self.hand_points_taskmap(self.robot_dof_pos_noisy, None)
        self.hand_vel_noisy = torch.bmm(hand_points_jac, self.robot_dof_vel_noisy.unsqueeze(2)).squeeze(2)

        self.fabric_q_for_obs = self.fabric_q.clone()
        self.fabric_qd_for_obs = self.fabric_qd.clone() * self.cfg.observation_annealing_coefficient
        self.fabric_qdd_for_obs = self.fabric_qdd.clone() * self.cfg.observation_annealing_coefficient