# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List

import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg


class CartpoleEnv(IsaacEnv):
    """Environment for 2-D cartpole.

    Reference:
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self, cfg: dict, **kwargs):
        """Initializes the environment.

        Args:
            cfg (dict): The configuration dictionary.
            kwargs (dict): Additional keyword arguments. See IsaacEnv for more details.
        """
        # copy configuration
        self.cfg_dict = cfg.copy()
        # configuration for the environment
        isaac_cfg = IsaacEnvCfg(
            env=EnvCfg(num_envs=self.cfg_dict["env"]["num_envs"], env_spacing=self.cfg_dict["env"]["env_spacing"])
        )
        isaac_cfg.sim.from_dict(self.cfg_dict["sim"])
        # initialize the base class to setup the scene.
        super().__init__(isaac_cfg, **kwargs)

        # define views over instances
        self.cartpoles = ArticulationView(prim_paths_expr=self.env_ns + "/.*/Cartpole", reset_xform_properties=False)

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()
        # initialize all the handles
        self.cartpoles.initialize(self.sim.physics_sim_view)
        # set the default state
        self.cartpoles.post_reset()

        # get quantities from scene we care about
        self._cart_dof_idx = self.cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self.cartpoles.get_dof_index("poleJoint")

        # compute the observation space
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(4,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # store maximum episode length
        self.max_episode_length = self.cfg_dict["env"]["episode_length"]

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # get nucleus assets path
        assets_root_path = nucleus_utils.get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError(
                "Unable to access the Nucleus server from Omniverse. For more information, please check: "
                "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
            )
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane")
        # robot
        robot_usd_path = assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"
        prim_utils.create_prim(
            prim_path=self.template_env_ns + "/Cartpole", usd_path=robot_usd_path, translation=(0.0, 0.0, 2.0)
        )
        # apply articulation settings
        kit_utils.set_articulation_properties(
            prim_path=self.template_env_ns + "/Cartpole",
            solver_position_iteration_count=self.cfg_dict["scene"]["cartpole"]["solver_position_iteration_count"],
            solver_velocity_iteration_count=self.cfg_dict["scene"]["cartpole"]["solver_velocity_iteration_count"],
            sleep_threshold=self.cfg_dict["scene"]["cartpole"]["sleep_threshold"],
            stabilization_threshold=self.cfg_dict["scene"]["cartpole"]["stabilization_threshold"],
            enable_self_collisions=self.cfg_dict["scene"]["cartpole"]["enable_self_collisions"],
        )
        # apply rigid body settings
        kit_utils.set_nested_rigid_body_properties(
            prim_path=self.template_env_ns + "/Cartpole",
            enable_gyroscopic_forces=self.cfg_dict["scene"]["cartpole"]["enable_gyroscopic_forces"],
            max_depenetration_velocity=self.cfg_dict["scene"]["cartpole"]["max_depenetration_velocity"],
        )
        # apply collider properties
        kit_utils.set_nested_collision_properties(
            prim_path=self.template_env_ns + "/Cartpole",
            contact_offset=self.cfg_dict["scene"]["cartpole"]["contact_offset"],
            rest_offset=self.cfg_dict["scene"]["cartpole"]["rest_offset"],
        )
        # return global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # get num envs to reset
        num_resets = len(env_ids)
        # randomize the MDP
        # -- DOF position
        dof_pos = torch.zeros((num_resets, self.cartpoles.num_dof), device=self.device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self.device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self.device))
        self.cartpoles.set_joint_positions(dof_pos, indices=env_ids)
        # -- DOF velocity
        dof_vel = torch.zeros((num_resets, self.cartpoles.num_dof), device=self.device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self.device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self.device))
        self.cartpoles.set_joint_velocities(dof_vel, indices=env_ids)
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        dof_forces = torch.zeros(
            (self.cartpoles.count, self.cartpoles.num_dof), dtype=torch.float32, device=self.device
        )
        dof_forces[:, self._cart_dof_idx] = self.cfg_dict["env"]["max_effort"] * self.actions[:, 0]
        indices = torch.arange(self.cartpoles.count, dtype=torch.int32, device=self.device)
        self.cartpoles.set_joint_efforts(dof_forces, indices=indices)
        # perform physics stepping
        for _ in range(self.cfg_dict["env"]["control_frequency_inv"]):
            # step simulation
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step: compute MDP
        self._compute_rewards()
        self._check_termination()
        # add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        # For more info: https://github.com/DLR-RM/stable-baselines3/issues/633
        self.extras["time_outs"] = self.episode_length_buf >= self.cfg_dict["env"]["episode_length"]

    def _get_observations(self) -> VecEnvObs:
        # access buffers from simulator
        dof_pos = self.cartpoles.get_joint_positions(clone=False)
        dof_vel = self.cartpoles.get_joint_velocities(clone=False)
        # concatenate and return
        obs_buf = torch.cat([dof_pos, dof_vel], dim=-1)

        return {"policy": obs_buf}

    """
    Helper functions.
    """

    def _compute_rewards(self) -> None:
        # access buffers from simulator
        dof_pos = self.cartpoles.get_joint_positions(clone=False)
        dof_vel = self.cartpoles.get_joint_velocities(clone=False)
        # extract values from buffer
        cart_pos = dof_pos[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]
        # compute reward
        reward = 1.0 - pole_pos * pole_pos - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        reward = torch.where(
            torch.abs(cart_pos) > self.cfg_dict["env"]["reset_dist"], torch.ones_like(reward) * -2.0, reward
        )
        reward = torch.where(torch.abs(pole_pos) > math.pi / 2, torch.ones_like(reward) * -2.0, reward)
        # set reward into buffer
        self.reward_buf[:] = reward

    def _check_termination(self) -> None:
        # access buffers from simulator
        dof_pos = self.cartpoles.get_joint_positions(clone=False)
        # extract values from buffer
        cart_pos = dof_pos[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        # compute resets
        # -- cart moved towards the edges
        resets = torch.where(torch.abs(cart_pos) > self.cfg_dict["env"]["reset_dist"], 1, 0)
        # -- pole fell down
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # -- episode length
        resets = torch.where(self.episode_length_buf >= self.max_episode_length, 1, resets)
        # set resets into buffer
        self.reset_buf[:] = resets
