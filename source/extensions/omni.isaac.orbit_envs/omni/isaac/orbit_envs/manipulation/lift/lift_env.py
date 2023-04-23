# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .lift_cfg import LiftEnvCfg, RandomizationCfg


class LiftEnv(IsaacEnv):
    """Environment for lifting an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: LiftEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.object = RigidObject(cfg=self.cfg.object)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = LiftObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = LiftRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")

        # Take an initial step to initialize the scene.
        # This is required to compute quantities like Jacobians used in step().
        self.sim.step()
        # -- fill up buffers
        self.object.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
        # table
        prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.usd_path)
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")
        # object
        self.object.spawn(self.template_env_ns + "/Object")

        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = StaticMarker(
                "/Visuals/object_goal",
                self.num_envs,
                usd_path=self.cfg.goal_marker.usd_path,
                scale=self.cfg.goal_marker.scale,
            )
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current",
                self.num_envs,
                usd_path=self.cfg.frame_marker.usd_path,
                scale=self.cfg.frame_marker.scale,
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command",
                    self.num_envs,
                    usd_path=self.cfg.frame_marker.usd_path,
                    scale=self.cfg.frame_marker.scale,
                )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- object pose
        self._randomize_object_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_initial_pose)
        # -- goal pose
        self._randomize_object_desired_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_desired_pose)

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
            self._ik_controller.set_command(self.actions[:, :-1])
            # use IK to convert to joint-space commands
            self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            dof_pos_offset = self.robot.data.actuator_pos_offset
            self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
            # we assume last command is tool action so don't change that
            self.robot_actions[:, -1] = self.actions[:, -1]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:] = self.actions
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
        self.object.update_buffers(self.dt)
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
        # -- add information to extra if task completed
        object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
        self.extras["is_success"] = torch.where(object_position_error < 0.02, 1, self.reset_buf)
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
        """Pre-processing of configuration parameters."""
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

        # convert configuration parameters to torchee
        # randomization
        # -- initial pose
        config = self.cfg.randomization.object_initial_pose
        for attr in ["position_uniform_min", "position_uniform_max"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))
        # -- desired pose
        config = self.cfg.randomization.object_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.object.initialize(self.env_ns + "/.*/Object")

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.object_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        # buffers
        self.object_root_pose_ee = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        # apply to instance manager
        # -- goal
        self._goal_markers.set_world_poses(self.object_des_pose_w[:, 0:3], self.object_des_pose_w[:, 3:7])
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
        # access buffers from simulator
        object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when task is successful
        if self.cfg.terminations.is_success:
            object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
            self.reset_buf = torch.where(object_position_error < 0.02, 1, self.reset_buf)
        # -- object fell off the table (table at height: 0.0 m)
        if self.cfg.terminations.object_falling:
            self.reset_buf = torch.where(object_pos[:, 2] < -0.05, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

    def _randomize_object_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectInitialPoseCfg):
        """Randomize the initial pose of the object."""
        # get the default root state
        root_state = self.object.get_default_root_state(env_ids)
        # -- object root position
        if cfg.position_cat == "default":
            pass
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            root_state[:, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
        # -- object root orientation
        if cfg.orientation_cat == "default":
            pass
        elif cfg.orientation_cat == "uniform":
            # sample uniformly in SO(3)
            root_state[:, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(f"Invalid category for randomizing the object orientation '{cfg.orientation_cat}'.")
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # update object init pose
        self.object_init_pose_w[env_ids] = root_state[:, 0:7]
        # set the root state
        self.object.set_root_state(root_state, env_ids=env_ids)

    def _randomize_object_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectDesiredPoseCfg):
        """Randomize the desired pose of the object."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.object_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.object_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.object_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.object_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.object_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]


class LiftObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos(self, env: LiftEnv):
        """DOF positions for the arm."""
        return env.robot.data.arm_dof_pos

    def arm_dof_pos_scaled(self, env: LiftEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 1],
        )

    def arm_dof_vel(self, env: LiftEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def tool_dof_pos_scaled(self, env: LiftEnv):
        """DOF positions of the tool normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.tool_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 1],
        )

    def tool_positions(self, env: LiftEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def tool_orientations(self, env: LiftEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def object_positions(self, env: LiftEnv):
        """Current object position."""
        return env.object.data.root_pos_w - env.envs_positions

    def object_orientations(self, env: LiftEnv):
        """Current object orientation."""
        # make the first element positive
        quat_w = env.object.data.root_quat_w
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def object_relative_tool_positions(self, env: LiftEnv):
        """Current object position w.r.t. end-effector frame."""
        return env.object.data.root_pos_w - env.robot.data.ee_state_w[:, :3]

    def object_relative_tool_orientations(self, env: LiftEnv):
        """Current object orientation w.r.t. end-effector frame."""
        # compute the relative orientation
        quat_ee = quat_mul(quat_inv(env.robot.data.ee_state_w[:, 3:7]), env.object.data.root_quat_w)
        # make the first element positive
        quat_ee[quat_ee[:, 0] < 0] *= -1
        return quat_ee

    def object_desired_positions(self, env: LiftEnv):
        """Desired object position."""
        return env.object_des_pose_w[:, 0:3] - env.envs_positions

    def object_desired_orientations(self, env: LiftEnv):
        """Desired object orientation."""
        # make the first element positive
        quat_w = env.object_des_pose_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def arm_actions(self, env: LiftEnv):
        """Last arm actions provided to env."""
        return env.actions[:, :-1]

    def tool_actions(self, env: LiftEnv):
        """Last tool actions provided to env."""
        return env.actions[:, -1].unsqueeze(1)

    def tool_actions_bool(self, env: LiftEnv):
        """Last tool actions transformed to a boolean command."""
        return torch.sign(env.actions[:, -1]).unsqueeze(1)


class LiftRewardManager(RewardManager):
    """Reward manager for single-arm object lifting environment."""

    def reaching_object_position_l2(self, env: LiftEnv):
        """Penalize end-effector tracking position error using L2-kernel."""
        return torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w), dim=1)

    def reaching_object_position_exp(self, env: LiftEnv, sigma: float):
        """Penalize end-effector tracking position error using exp-kernel."""
        error = torch.sum(torch.square(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w), dim=1)
        return torch.exp(-error / sigma)

    def reaching_object_position_tanh(self, env: LiftEnv, sigma: float):
        """Penalize tool sites tracking position error using tanh-kernel."""
        # distance of end-effector to the object: (num_envs,)
        ee_distance = torch.norm(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
        # distance of the tool sites to the object: (num_envs, num_tool_sites)
        object_root_pos = env.object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        tool_sites_distance = torch.norm(env.robot.data.tool_sites_state_w[:, :, :3] - object_root_pos, dim=-1)
        # average distance of the tool sites to the object: (num_envs,)
        # note: we add the ee distance to the average to make sure that the ee is always closer to the object
        num_tool_sites = tool_sites_distance.shape[1]
        average_distance = (ee_distance + torch.sum(tool_sites_distance, dim=1)) / (num_tool_sites + 1)

        return 1 - torch.tanh(average_distance / sigma)

    def penalizing_arm_dof_velocity_l2(self, env: LiftEnv):
        """Penalize large movements of the robot arm."""
        return -torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_tool_dof_velocity_l2(self, env: LiftEnv):
        """Penalize large movements of the robot tool."""
        return -torch.sum(torch.square(env.robot.data.tool_dof_vel), dim=1)

    def penalizing_arm_action_rate_l2(self, env: LiftEnv):
        """Penalize large variations in action commands besides tool."""
        return -torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)

    def penalizing_tool_action_l2(self, env: LiftEnv):
        """Penalize large values in action commands for the tool."""
        return -torch.square(env.actions[:, -1])

    def tracking_object_position_exp(self, env: LiftEnv, sigma: float, threshold: float):
        """Penalize tracking object position error using exp-kernel."""
        # distance of the end-effector to the object: (num_envs,)
        error = torch.sum(torch.square(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w), dim=1)
        # rewarded if the object is lifted above the threshold
        return (env.object.data.root_pos_w[:, 2] > threshold) * torch.exp(-error / sigma)

    def tracking_object_position_tanh(self, env: LiftEnv, sigma: float, threshold: float):
        """Penalize tracking object position error using tanh-kernel."""
        # distance of the end-effector to the object: (num_envs,)
        distance = torch.norm(env.object_des_pose_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
        # rewarded if the object is lifted above the threshold
        return (env.object.data.root_pos_w[:, 2] > threshold) * (1 - torch.tanh(distance / sigma))

    def lifting_object_success(self, env: LiftEnv, threshold: float):
        """Sparse reward if object is lifted successfully."""
        return torch.where(env.object.data.root_pos_w[:, 2] > threshold, 1.0, 0.0)
