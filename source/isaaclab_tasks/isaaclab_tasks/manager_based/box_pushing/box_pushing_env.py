# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import os
import torch

import pytorch_kinematics as pk

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.box_pushing.box_pushing_env_cfg import BoxPushingEnvCfg

# from timeit import default_timer as timer


class BoxPushingEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BoxPushingEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.step_dt  # TODO find better solution

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        print("Precomputing IK Solutions for the box sampling space")

        # TODO set cli to choose IK method

        # # computing the ik vector
        # init_box_pose = cfg.scene.object.init_state.pos
        # offset_range = cfg.events.reset_object_position.params["pose_range"]
        # self.x_sample_range = [init_box_pose[0] + offset_range["x"][0], init_box_pose[0] + offset_range["x"][1]]
        # self.y_sample_range = [init_box_pose[1] + offset_range["y"][0], init_box_pose[1] + offset_range["y"][1]]
        # self.z_sample_range = [init_box_pose[2] + offset_range["z"][0], init_box_pose[2] + offset_range["z"][1]]

        # x_limits = torch.linspace(
        #     self.x_sample_range[0], self.x_sample_range[1], cfg.ik_grid_precision + 1, device=self.device
        # )
        # y_limits = torch.linspace(
        #     self.y_sample_range[0], self.y_sample_range[1], cfg.ik_grid_precision + 1, device=self.device
        # )
        # z_limits = torch.linspace(
        #     self.z_sample_range[0], self.z_sample_range[1], cfg.ik_grid_precision + 1, device=self.device
        # )

        # x_centers = (x_limits[:-1] + x_limits[1:]) / 2
        # y_centers = (y_limits[:-1] + y_limits[1:]) / 2
        # z_centers = (z_limits[:-1] + z_limits[1:]) / 2
        # cx, cy, cz = torch.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        # centers = torch.stack((cx, cy, cz), dim=-1)

        # orientation = convert_to_torch([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
        #     cfg.ik_grid_precision, cfg.ik_grid_precision, cfg.ik_grid_precision, 1
        # )
        # centers = torch.cat((centers, orientation), dim=-1)

        # centers = centers.reshape(cfg.ik_grid_precision**3, 7)

        # ######
        # # IK #
        # ######

        # initializing kinematic chain from urdf for IK
        script_dir = os.path.dirname(__file__)
        urdf = os.path.abspath(os.path.join(script_dir, "assets", "franka_ik.urdf"))

        with open(urdf, "rb") as urdf_file:
            urdf_data = urdf_file.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_data, "panda_hand")
        self.chain = self.chain.to(device=self.device)

        # # processing target box poses
        # target_poses = centers + convert_to_torch([0.0, 0.0, 0.27, 0.0, 0.0, 0.0, 0.0], device=self.device)
        # target_transforms = Transform3d(pos=target_poses[:, :3], rot=target_poses[:, 3:7], device=self.device)

        # robot: Articulation = self.scene["robot"]

        # # solving IK
        # ik = pk.PseudoInverseIK(
        #     self.chain,
        #     retry_configs=robot.data.joint_pos[0, :7].unsqueeze(0),  # initial config
        #     joint_limits=robot.data.joint_limits[0, :7],
        #     lr=0.2,
        # )
        # sol = ik.solve(target_transforms)
        # joint_pos_des = sol.solutions[:, 0]
        # self.ik_grid_solutions = joint_pos_des.reshape(
        #     cfg.ik_grid_precision, cfg.ik_grid_precision, cfg.ik_grid_precision, 7
        # )

    def compute_index(self, value, min_val, max_val):
        """
        Compute the index for a given value along one dimension.
        """
        # TODO resolve value fetch
        precision = self.cfg.ik_grid_precision
        index = torch.floor((value - min_val) / (max_val - min_val) * precision).long()
        index = torch.clamp(index, 0, precision - 1)  # Ensure the index is within bounds
        return index

    def get_ik_solutions(self, sample_poses):
        """
        Find the indices in the grid for each sample pose.
        """
        x_indices = self.compute_index(sample_poses[:, 0], self.x_sample_range[0], self.x_sample_range[1])
        y_indices = self.compute_index(sample_poses[:, 1], self.y_sample_range[0], self.y_sample_range[1])
        z_indices = self.compute_index(sample_poses[:, 2], self.z_sample_range[0], self.z_sample_range[1])

        return self.ik_grid_solutions[x_indices, y_indices, z_indices]
