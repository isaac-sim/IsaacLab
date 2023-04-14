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
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg


class HumanoidEnv(IsaacEnv):
    """Environment for an Humanoid on a flat terrain.

    Reference:
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v3.py
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
        self.humanoids = ArticulationView(
            prim_paths_expr=self.env_ns + "/.*/Humanoid/torso", reset_xform_properties=False
        )

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()
        # initialize all the handles
        self.humanoids.initialize(self.sim.physics_sim_view)
        # set the default state
        self.humanoids.post_reset()

        # get quantities from scene we care about
        self._dof_limits = self.humanoids.get_dof_limits()[0, :].to(self.device)
        self._initial_root_tf = self.humanoids.get_world_poses(clone=True)
        self._initial_dof_pos = self.humanoids.get_joint_positions(clone=True)

        # initialize buffers
        self.actions = torch.zeros((self.num_envs, 21), device=self.device)
        # create constants required later during simulation.
        self._define_environment_constants()
        # create other useful variables
        self.potentials = torch.full(
            (self.num_envs,), -1000.0 / self.physics_dt, dtype=torch.float32, device=self.device
        )
        self.prev_potentials = self.potentials.clone()

        # compute the observation space
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(87,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(21,))
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
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane", static_friction=0.5, dynamic_friction=0.5, restitution=0.8
        )
        # robot
        robot_usd_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid_instanceable.usd"
        prim_utils.create_prim(
            prim_path=self.template_env_ns + "/Humanoid", usd_path=robot_usd_path, translation=(0.0, 0.0, 1.34)
        )
        # apply articulation settings
        kit_utils.set_articulation_properties(
            prim_path=self.template_env_ns + "/Humanoid/torso",
            solver_position_iteration_count=self.cfg_dict["scene"]["humanoid"]["solver_position_iteration_count"],
            solver_velocity_iteration_count=self.cfg_dict["scene"]["humanoid"]["solver_velocity_iteration_count"],
            sleep_threshold=self.cfg_dict["scene"]["humanoid"]["sleep_threshold"],
            stabilization_threshold=self.cfg_dict["scene"]["humanoid"]["stabilization_threshold"],
            enable_self_collisions=self.cfg_dict["scene"]["humanoid"]["enable_self_collisions"],
        )
        # apply rigid body settings
        kit_utils.set_nested_rigid_body_properties(
            prim_path=self.template_env_ns + "/Humanoid",
            enable_gyroscopic_forces=self.cfg_dict["scene"]["humanoid"]["enable_gyroscopic_forces"],
            max_depenetration_velocity=self.cfg_dict["scene"]["humanoid"]["max_depenetration_velocity"],
        )
        # apply collider properties
        kit_utils.set_nested_collision_properties(
            prim_path=self.template_env_ns + "/Humanoid",
            contact_offset=self.cfg_dict["scene"]["humanoid"]["contact_offset"],
            rest_offset=self.cfg_dict["scene"]["humanoid"]["rest_offset"],
        )
        # return global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # get num envs to reset
        num_resets = len(env_ids)
        # randomize the MDP
        # -- DOF position
        dof_pos = torch_utils.torch_rand_float(-0.2, 0.2, (num_resets, self.humanoids.num_dof), device=self.device)
        dof_pos[:] = torch_utils.tensor_clamp(
            self._initial_dof_pos[env_ids] + dof_pos, self._dof_limits[:, 0], self._dof_limits[:, 1]
        )
        self.humanoids.set_joint_positions(dof_pos, indices=env_ids)
        # -- DOF velocity
        dof_vel = torch_utils.torch_rand_float(-0.1, 0.1, (num_resets, self.humanoids.num_dof), device=self.device)
        self.humanoids.set_joint_velocities(dof_vel, indices=env_ids)
        # -- Root pose
        root_pos, root_rot = self._initial_root_tf[0].clone()[env_ids], self._initial_root_tf[1].clone()[env_ids]
        self.humanoids.set_world_poses(root_pos, root_rot, indices=env_ids)
        # -- Root velocity
        root_vel = torch.zeros((num_resets, 6), device=self.device)
        self.humanoids.set_velocities(root_vel, indices=env_ids)
        # -- Reset potentials
        to_target = self._GOAL_POS[env_ids] - root_pos
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.physics_dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()
        # -- MDP reset
        self.actions[env_ids, :] = 0
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        dof_forces = self.actions * self._JOINT_GEARS * self.cfg_dict["env"]["power_scale"]
        indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.humanoids.set_joint_efforts(dof_forces, indices=indices)
        # perform physics stepping
        for _ in range(self.cfg_dict["env"]["control_frequency_inv"]):
            # step simulation
            self.sim.step(render=self.enable_render)
            # check if simulation is still running
            if self.sim.is_stopped():
                return
        # post-step: compute MDP
        self._cache_common_quantities()
        self._compute_rewards()
        self._check_termination()
        # add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        # For more info: https://github.com/DLR-RM/stable-baselines3/issues/633
        self.extras["time_outs"] = self.episode_length_buf >= self.cfg_dict["env"]["episode_length"]

    def _get_observations(self) -> VecEnvObs:
        # extract euler angles (in start frame)
        roll, _, yaw = torch_utils.get_euler_xyz(self._torso_quat_start)
        # compute heading direction
        # TODO: Check why is this using z direction instead of y.
        walk_target_angle = torch.atan2(
            self._GOAL_POS[:, 2] - self._torso_pos_start[:, 2], self._GOAL_POS[:, 0] - self._torso_pos_start[:, 0]
        )
        angle_to_target = walk_target_angle - yaw

        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(21), num_dofs(21), num_dofs(21)
        obs_buf = torch.cat(
            (
                self._torso_pos_start[:, 2].view(-1, 1),
                self._lin_vel_start,
                self._ang_vel_start * self.cfg_dict["env"]["angular_velocity_scale"],
                yaw.unsqueeze(-1),
                roll.unsqueeze(-1),
                angle_to_target.unsqueeze(-1),
                self._up_proj.unsqueeze(-1),
                self._heading_proj.unsqueeze(-1),
                self._dof_pos_scaled,
                self._dof_vel_scaled,
                self._feet_force_torques * self.cfg_dict["env"]["contact_force_scale"],
                self.actions,
            ),
            dim=-1,
        )

        return {"policy": obs_buf}

    """
    Helper functions - MDP.
    """

    def _cache_common_quantities(self) -> None:
        """Compute common quantities from simulator used for computing MDP signals."""
        # extract data from simulator
        torso_pos_world, torso_quat_world = self.humanoids.get_world_poses(clone=False)
        lin_vel_world = self.humanoids.get_linear_velocities(clone=False)
        ang_vel_world = self.humanoids.get_angular_velocities(clone=False)
        dof_pos = self.humanoids.get_joint_positions(clone=False)
        dof_vel = self.humanoids.get_joint_velocities(clone=False)
        # TODO: Remove direct usage of `_physics_view` when method exists in :class:`ArticulationView`
        feet_force_torques = self.humanoids._physics_view.get_force_sensor_forces()

        # scale the dof
        self._dof_pos_scaled = torch_utils.scale_transform(dof_pos, self._dof_limits[:, 0], self._dof_limits[:, 1])
        self._dof_vel_scaled = dof_vel * self.cfg_dict["env"]["dof_velocity_scale"]
        # feet contact forces
        self._feet_force_torques = feet_force_torques.reshape(self.num_envs, -1)

        # convert base pose w.r.t. start frame
        self._torso_pos_start = torso_pos_world
        self._torso_quat_start = torch_utils.quat_mul(torso_quat_world, self._INV_START_QUAT)
        # convert velocity (in base frame w.r.t. start frame)
        self._lin_vel_start = torch_utils.quat_rotate_inverse(self._torso_quat_start, lin_vel_world)
        self._ang_vel_start = torch_utils.quat_rotate_inverse(self._torso_quat_start, ang_vel_world)
        # convert basis vectors w.r.t. start frame
        up_vec = torch_utils.quat_rotate(self._torso_quat_start, self._UP_VEC)
        heading_vec = torch_utils.quat_rotate(self._torso_quat_start, self._HEADING_VEC)

        # compute relative movement to the target
        self._to_target = self._GOAL_POS - self._torso_pos_start
        self._to_target[:, 2] = 0.0
        to_target_dir = torch_utils.normalize(self._to_target)
        # compute projection of current heading to desired heading vector
        self._up_proj = up_vec[:, 2]
        self._heading_proj = torch.bmm(heading_vec.view(self.num_envs, 1, 3), to_target_dir.view(self.num_envs, 3, 1))
        self._heading_proj = self._heading_proj.view(self.num_envs)

    def _compute_rewards(self) -> None:
        # heading in the right direction
        heading_reward = torch.where(
            self._heading_proj > 0.8,
            self.cfg_dict["env"]["heading_weight"],
            self.cfg_dict["env"]["heading_weight"] * self._heading_proj.double() / 0.8,
        )
        # aligning up axis of robot and environment
        up_reward = torch.where(self._up_proj > 0.93, self.cfg_dict["env"]["up_weight"], 0.0)

        # penalty for high action commands
        actions_cost = torch.sum(self.actions**2, dim=-1)
        # energy penalty for movement (power = torque * vel)
        electricity_cost = torch.sum(torch.abs(self.actions * self._dof_vel_scaled), dim=-1)
        # reaching close to dof limits
        # TODO: Shouldn't this be absolute dof pos? Only checks for upper limit right now.
        motor_effort_ratio = self._JOINT_GEARS / self._MAX_MOTOR_EFFORT
        scaled_cost = (torch.abs(self._dof_pos_scaled) - 0.98) / 0.02
        dof_at_limit_cost = torch.sum(
            (torch.abs(self._dof_pos_scaled) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
        )
        # reward for duration of staying alive
        alive_reward = self.cfg_dict["env"]["alive_weight"]

        # compute x,y-potential score towards the goal
        self.prev_potentials = self.potentials.clone()
        self.potentials = -torch.norm(self._to_target, p=2, dim=-1) / self.physics_dt
        # reward for progressing towards the goal (through L2 potential)
        progress_reward = self.potentials - self.prev_potentials

        # compute reward
        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.cfg_dict["env"]["actions_cost"] * actions_cost
            - self.cfg_dict["env"]["energy_cost"] * electricity_cost
            - self.cfg_dict["env"]["joints_at_limit_cost"] * dof_at_limit_cost
        )
        # adjust reward for fallen agents
        total_reward = torch.where(
            self._torso_pos_start[:, 2] < self.cfg_dict["env"]["termination_height"],
            self.cfg_dict["env"]["death_cost"],
            total_reward,
        )
        # set reward into buffer
        self.reward_buf[:] = total_reward

    def _check_termination(self) -> None:
        # compute resets
        # -- base has collapsed
        resets = torch.where(
            self._torso_pos_start[:, 2] < self.cfg_dict["env"]["termination_height"], 1, self.reset_buf
        )
        # -- episode length
        resets = torch.where(self.episode_length_buf >= self.max_episode_length, 1, resets)
        # set reset into buffer
        self.reset_buf[:] = resets

    def _define_environment_constants(self):
        """Defines useful constants used by the implementation."""
        # desired goal position
        self._GOAL_POS = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        # gear ratio for joint control
        self._JOINT_GEARS = torch.tensor(
            [
                67.5000,  # lower_waist
                67.5000,  # lower_waist
                67.5000,  # right_upper_arm
                67.5000,  # right_upper_arm
                67.5000,  # left_upper_arm
                67.5000,  # left_upper_arm
                67.5000,  # pelvis
                45.0000,  # right_lower_arm
                45.0000,  # left_lower_arm
                45.0000,  # right_thigh: x
                135.0000,  # right_thigh: y
                45.0000,  # right_thigh: z
                45.0000,  # left_thigh: x
                135.0000,  # left_thigh: y
                45.0000,  # left_thigh: z
                90.0000,  # right_knee
                90.0000,  # left_knee
                22.5,  # right_foot
                22.5,  # right_foot
                22.5,  # left_foot
                22.5,  # left_foot
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # the maximum motor effort applicable
        self._MAX_MOTOR_EFFORT = torch.max(self._JOINT_GEARS)
        # initial spawn orientation
        self._START_QUAT = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32)
        self._INV_START_QUAT = torch_utils.quat_conjugate(self._START_QUAT).repeat((self.num_envs, 1))
        # heading direction for the robot
        self._HEADING_VEC = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        # up direction for the simulator
        self._UP_VEC = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
