# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .reach_cfg import RandomizationCfg, ReachEnvCfg


class ReachEnv(IsaacEnv):
    """Environment for reaching to desired pose for a single-arm manipulator."""

    def __init__(self, cfg: ReachEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = ReachObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = ReachRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)

    """
    Implementation specifics.
    """

    def _design_scene(self):
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
        # table
        prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.usd_path)
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")

        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = PointMarker("/Visuals/ee_goal", self.num_envs, radius=0.025)
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command", self.num_envs, usd_path=self.cfg.marker.usd_path, scale=self.cfg.marker.scale
                )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- desired end-effector pose
        self._randomize_ee_desired_pose(env_ids, cfg=self.cfg.randomization.ee_desired_pose)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            self._ik_controller.set_command(self.actions)
            # compute the joint commands
            self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            self.robot_actions[:, : self.robot.arm_num_dof] -= self.robot.data.actuator_pos_offset[
                :, : self.robot.arm_num_dof
            ]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:, : self.robot.arm_num_dof] = self.actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torch
        # randomization
        # -- desired pose
        config = self.cfg.randomization.ee_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            # note: we exclude gripper from actions in this env
            self.num_actions = self._ik_controller.num_actions
        elif self.cfg.control.control_type == "default":
            # note: we exclude gripper from actions in this env
            self.num_actions = self.robot.arm_num_dof

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.ee_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

    def _debug_vis(self):
        # compute error between end-effector and command
        error = torch.sum(torch.square(self.ee_des_pose_w[:, :3] - self.robot.data.ee_state_w[:, 0:3]), dim=1)
        # set indices of the prim based on error threshold
        goal_indices = torch.where(error < 0.002, 1, 0)
        # apply to instance manager
        # -- goal
        self._goal_markers.set_world_poses(self.ee_des_pose_w[:, :3], self.ee_des_pose_w[:, 3:7])
        self._goal_markers.set_status(goal_indices)
        # -- end-effector
        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
        # -- task-space commands
        if self.cfg.control.control_type == "inverse_kinematics":
            # convert to world frame
            ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
            ee_orientations = self._ik_controller.desired_ee_rot
            # set poses
            self._cmd_markers.set_world_poses(ee_positions, ee_orientations)

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # extract values from buffer
        # compute resets
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

    def _randomize_ee_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.EndEffectorDesiredPoseCfg):
        """Randomize the desired pose of the end-effector."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.ee_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.ee_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.ee_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.ee_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.ee_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]


class ReachObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos_normalized(self, env: ReachEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, :7, 0],
            env.robot.data.soft_dof_pos_limits[:, :7, 1],
        )

    def arm_dof_vel(self, env: ReachEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def ee_position(self, env: ReachEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def ee_position_command(self, env: ReachEnv):
        """Desired end-effector position of the arm."""
        return env.ee_des_pose_w[:, :3] - env.envs_positions

    def actions(self, env: ReachEnv):
        """Last actions provided to env."""
        return env.actions


class ReachRewardManager(RewardManager):
    """Reward manager for single-arm reaching environment."""

    def tracking_robot_position_l2(self, env: ReachEnv):
        """Penalize tracking position error using L2-kernel."""
        # compute error
        return torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)

    def tracking_robot_position_exp(self, env: ReachEnv, sigma: float):
        """Penalize tracking position error using exp-kernel."""
        # compute error
        error = torch.sum(torch.square(env.ee_des_pose_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        return torch.exp(-error / sigma)

    def penalizing_robot_dof_velocity_l2(self, env: ReachEnv):
        """Penalize large movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_robot_dof_acceleration_l2(self, env: ReachEnv):
        """Penalize fast movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def penalizing_action_rate_l2(self, env: ReachEnv):
        """Penalize large variations in action commands."""
        return torch.sum(torch.square(env.actions - env.previous_actions), dim=1)
