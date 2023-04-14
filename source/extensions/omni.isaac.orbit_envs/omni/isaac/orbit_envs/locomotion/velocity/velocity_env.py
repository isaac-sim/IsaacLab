# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import gym.spaces
import math
import torch
from typing import List, Sequence, Tuple

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import omni.replicator.isaac as rep_dr
from omni.isaac.core.utils.types import DynamicsViewState

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.robots.legged_robot import LeggedRobot
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform, wrap_to_pi
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .velocity_cfg import VelocityEnvCfg


class VelocityEnv(IsaacEnv):
    """Environment for tracking a base SE(2) velocity command for a legged robot."""

    def __init__(self, cfg: VelocityEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg

        # create classes
        self.robot = LeggedRobot(cfg=self.cfg.robot)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()
        # setup randomization in environment
        self._setup_randomization()

        # prepare the observation manager
        self._observation_manager = LocomotionVelocityObservationManager(
            class_to_dict(self.cfg.observations), self, self.device
        )
        # prepare the reward manager
        self._reward_manager = LocomotionVelocityRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space
        # TODO: Cleanup the `_group_obs_dim` variable.
        num_obs = self._observation_manager._group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # ground plane
        if self.cfg.terrain.use_default_ground_plane:
            # use the default ground plane
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                improve_patch_friction=True,
                combine_mode="max",
            )
        else:
            prim_utils.create_prim("/World/defaultGroundPlane", usd_path=self.cfg.terrain.usd_path)

        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")

        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer for foot contact
            self._feet_contact_marker = PointMarker(
                "/Visuals/feet_contact", self.num_envs * len(self.cfg.robot.feet_info), radius=0.035
            )
            # create point instancer to visualize the goal base velocity
            self._base_vel_goal_markers = StaticMarker(
                "/Visuals/base_vel_goal",
                self.num_envs,
                usd_path=self.cfg.marker.usd_path,
                scale=self.cfg.marker.scale,
                color=(1.0, 0.0, 0.0),
            )
            # create marker for viewing current base velocity
            self._base_vel_markers = StaticMarker(
                "/Visuals/base_vel",
                self.num_envs,
                usd_path=self.cfg.marker.usd_path,
                scale=self.cfg.marker.scale,
                color=(0.0, 0.0, 1.0),
            )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        """Reset environments based on specified indices.

        Calls the following functions on reset:
        - :func:`_reset_robot_state`: Reset the root state and DOF state of the robot.
        - :func:`_resample_commands`: Resample the goal/command for the task. E.x.: desired velocity command.
        - :func:`_sim_randomization`: Randomizes simulation properties using replicator. E.x.: friction, body mass.

        Addition to above, the function fills up episode information into extras and resets buffers.

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # randomize the MDP
        # -- robot state
        self._reset_robot_state(env_ids)
        # -- robot buffers
        self.robot.reset_buffers(env_ids)
        # -- resample commands
        self._resample_commands(env_ids)
        # -- randomize dynamics with replicator
        rep_dr.physics_view.step_randomization(env_ids)

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

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        # clip actions and move to env device
        self.actions = actions.clone().to(device=self.device)
        self.actions = self.actions.clip_(-self.cfg.control.action_clipping, self.cfg.control.action_clipping)
        # scaled actions
        scaled_actions = self.cfg.control.action_scale * self.actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(scaled_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.common_step_counter += 1
        # -- update robot buffers
        self.robot.update_buffers(dt=self.dt)

        # compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # in-between episode randomization
        # -- resample commands in between episodes
        env_ids = self.episode_length_buf % self._command_interval == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # -- update heading for new commands and on-going command
        self._update_command()
        # -- push robots
        if self.cfg.randomization.push_robot["enabled"]:
            if self.common_step_counter % self._push_interval == 0:
                self._push_robots()

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

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # store constants about the environment
        # randomization
        # -- random velocity command
        self._command_ranges = copy.deepcopy(self.cfg.commands.ranges)
        # -- command sampling
        self._command_interval = math.ceil(self.cfg.commands.resampling_time / self.dt)
        # -- push robots
        self._push_interval = math.ceil(self.cfg.randomization.push_robot["interval_s"] / self.dt)

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        # define action space
        self.num_actions = self.robot.num_actions

        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

    def _setup_randomization(self):
        """Randomize properties of scene at start."""
        # randomize body masses
        # TODO: Make configurable via class!
        if self.cfg.randomization.additive_body_mass["enabled"]:
            # read configuration for randomization
            body_name = self.cfg.randomization.additive_body_mass["body_name"]
            mass_range = self.cfg.randomization.additive_body_mass["range"]
            # set properties into robot
            body_index = self.robot.body_names.index(body_name)
            body_masses = self.robot.articulations.get_body_masses(body_indices=[body_index])
            body_masses += sample_uniform(mass_range[0], mass_range[1], size=(self.robot.count, 1), device=self.device)
            self.robot.articulations.set_body_masses(body_masses, body_indices=[body_index])

        # register views with replicator
        for body in self.robot.feet_bodies.values():
            # FIXME: Hacking default state because Isaac core doesn't handle non-root link well!
            body._dynamics_default_state = DynamicsViewState(
                positions=torch.zeros(body.count, 3, device=body._device),
                orientations=torch.zeros(body.count, 4, device=body._device),
                linear_velocities=torch.zeros(body.count, 3, device=body._device),
                angular_velocities=torch.zeros(body.count, 3, device=body._device),
            )
            # register with replicator
            rep_dr.physics_view.register_rigid_prim_view(rigid_prim_view=body)

        # create graph using replicator
        with rep_dr.trigger.on_rl_frame(num_envs=self.num_envs):
            with rep_dr.gate.on_env_reset():
                # -- feet
                if self.cfg.randomization.feet_material_properties["enabled"]:
                    # read configuration for randomization
                    sf = self.cfg.randomization.feet_material_properties["static_friction_range"]
                    df = self.cfg.randomization.feet_material_properties["dynamic_friction_range"]
                    res = self.cfg.randomization.feet_material_properties["restitution_range"]
                    # set properties into robot
                    for body in self.robot.feet_bodies.values():
                        rep_dr.physics_view.randomize_rigid_prim_view(
                            view_name=body.name,
                            operation="direct",
                            material_properties=rep.distribution.uniform(
                                [sf[0], df[0], res[0]] * body.num_shapes, [sf[1], df[1], res[1]] * body.num_shapes
                            ),
                            num_buckets=self.cfg.randomization.feet_material_properties["num_buckets"],
                        )
        # prepares/executes the action graph for randomization
        rep.orchestrator.run()

    def _debug_vis(self):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- command
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_w[:, :2])
        # -- feet state
        feet_pose_w = self.robot.data.feet_state_w[..., :7].clone().view(-1, 7)
        feet_status = torch.where(self.robot.data.feet_air_time.view(-1) > 0.0, 1, 2)
        # apply to instance manager
        # -- feet marker
        self._feet_contact_marker.set_world_poses(feet_pose_w[:, :3], feet_pose_w[:, 3:7])
        self._feet_contact_marker.set_status(feet_status)
        # -- goal
        self._base_vel_goal_markers.set_scales(vel_des_arrow_scale)
        self._base_vel_goal_markers.set_world_poses(base_pos_w, vel_des_arrow_quat)
        # -- base velocity
        self._base_vel_markers.set_scales(vel_arrow_scale)
        self._base_vel_markers.set_world_poses(base_pos_w, vel_arrow_quat)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # arrow-scale
        arrow_scale = torch.tensor(self.cfg.marker.scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1)
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat

    """
    Helper functions - MDP.
    """

    def _reset_robot_state(self, env_ids):
        """Resets root and dof states of robots in selected environments."""
        # handle trivial case
        if len(env_ids) == 0:
            return
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_random_dof_state(env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        root_state[:, :3] += self.envs_positions[env_ids]
        # xy position
        if self.cfg.randomization.initial_base_position["enabled"]:
            xy_range = self.cfg.randomization.initial_base_position["xy_range"]
            root_state[:, :2] += sample_uniform(xy_range[0], xy_range[1], (len(env_ids), 2), device=self.device)
        # base velocities: [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.randomization.initial_base_velocity["enabled"]:
            vel_range = self.cfg.randomization.initial_base_velocity["vel_range"]
            root_state[:, 7:13] = sample_uniform(vel_range[0], vel_range[1], (len(env_ids), 6), device=self.device)
        else:
            root_state[:, 7:13] = 0.0
        # set into robot
        self.robot.set_root_state(root_state, env_ids=env_ids)

    def _resample_commands(self, env_ids: Sequence[int]):
        """Randomly select commands of some environments."""
        # handle trivial case
        if len(env_ids) == 0:
            return
        # linear velocity - x direction
        _min, _max = self._command_ranges.lin_vel_x
        self.commands[env_ids, 0] = sample_uniform(_min, _max, len(env_ids), device=self.device)
        # linear velocity - y direction
        _min, _max = self._command_ranges.lin_vel_y
        self.commands[env_ids, 1] = sample_uniform(_min, _max, len(env_ids), device=self.device)
        # ang vel yaw or heading target
        if self.cfg.commands.heading_command:
            _min, _max = self._command_ranges.heading
            self.heading_target[env_ids] = sample_uniform(_min, _max, len(env_ids), device=self.device)
        else:
            _min, _max = self._command_ranges.ang_vel_yaw
            self.commands[env_ids, 2] = sample_uniform(_min, _max, len(env_ids), device=self.device)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1).float()

    def _update_command(self):
        """Recompute heading command based on current orientation of the robot."""
        if self.cfg.commands.heading_command:
            # convert current motion direction to world frame
            forward = quat_apply(self.robot.data.root_quat_w, self.robot._FORWARD_VEC_B)
            # convert direction to heading
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # adjust command to provide corrected heading based on target
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.heading_target - heading),
                self._command_ranges.ang_vel_yaw[0],
                self._command_ranges.ang_vel_yaw[1],
            )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        min_vel, max_vel = self.cfg.randomization.push_robot["vel_xy_range"]
        # get current root state
        root_state_w = self.robot.data.root_state_w
        # add random XY velocity to the base
        root_state_w[:, 7:9] += sample_uniform(min_vel, max_vel, (self.num_envs, 2), device=self.device)
        # set the root state
        self.robot.set_root_state(root_state_w)

    def _check_termination(self) -> None:
        # extract values from buffer
        # compute resets
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.episode_timeout["enabled"]:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
        # -- base height fall
        if self.cfg.terminations.base_height_fall["enabled"]:
            base_target_height = self.cfg.terminations.base_height_fall["min_height"]
            self.reset_buf = torch.where(self.robot.data.root_pos_w[:, 2] <= base_target_height, 1, self.reset_buf)


class LocomotionVelocityObservationManager(ObservationManager):
    """Observation manager for locomotion velocity-tracking environment."""

    def base_lin_vel(self, env: VelocityEnv):
        """Base linear velocity in base frame."""
        return env.robot.data.root_lin_vel_b

    def base_ang_vel(self, env: VelocityEnv):
        """Base angular velocity in base frame."""
        return env.robot.data.root_ang_vel_b

    def projected_gravity(self, env: VelocityEnv):
        """Gravity projection on base frame."""
        return env.robot.data.projected_gravity_b

    def velocity_commands(self, env: VelocityEnv):
        """Desired base velocity for the robot."""
        return env.commands

    def dof_pos(self, env: VelocityEnv):
        """DOF positions for legs offset by the drive default positions."""
        return env.robot.data.dof_pos - env.robot.data.actuator_pos_offset

    def dof_vel(self, env: VelocityEnv):
        """DOF velocity of the legs."""
        return env.robot.data.dof_vel - env.robot.data.actuator_vel_offset

    def actions(self, env: VelocityEnv):
        """Last actions provided to env."""
        return env.actions


class LocomotionVelocityRewardManager(RewardManager):
    """Reward manager for locomotion velocity-tracking environment."""

    def lin_vel_z_l2(self, env: VelocityEnv):
        """Penalize z-axis base linear velocity using L2-kernel."""
        return torch.square(env.robot.data.root_lin_vel_w[:, 2])

    def ang_vel_xy_l2(self, env: VelocityEnv):
        """Penalize xy-axii base angular velocity using L2-kernel."""
        return torch.sum(torch.square(env.robot.data.root_ang_vel_w[:, :2]), dim=1)

    def flat_orientation_l2(self, env: VelocityEnv):
        """Penalize non-float base orientation."""
        return torch.sum(torch.square(env.robot.data.projected_gravity_b[:, :2]), dim=1)

    def dof_torques_l2(self, env: VelocityEnv):
        """Penalize torques applied on the robot."""
        return torch.sum(torch.square(env.robot.data.applied_torques), dim=1)

    def dof_vel_l2(self, env: VelocityEnv):
        """Penalize dof velocities on the robot."""
        return torch.sum(torch.square(env.robot.data.dof_vel), dim=1)

    def dof_acc_l2(self, env: VelocityEnv):
        """Penalize dof accelerations on the robot."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def dof_pos_limits(self, env: VelocityEnv):
        """Penalize dof positions too close to the limit."""
        out_of_limits = -(env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[..., 0]).clip(max=0.0)
        out_of_limits += (env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[..., 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def dof_vel_limits(self, env: VelocityEnv, soft_ratio: float):
        """Penalize dof velocities too close to the limit.

        Args:
            soft_ratio (float): Defines the soft limit as a percentage of the hard limit.
        """
        out_of_limits = torch.abs(env.robot.data.dof_vel) - env.robot.data.soft_dof_vel_limits * soft_ratio
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
        return torch.sum(out_of_limits, dim=1)

    def action_rate_l2(self, env: VelocityEnv):
        """Penalize changes in actions."""
        return torch.sum(torch.square(env.previous_actions - env.actions), dim=1)

    def applied_torque_limits(self, env: VelocityEnv):
        """Penalize applied torques that are too close to the actuator limits."""
        out_of_limits = torch.abs(env.robot.data.applied_torques - env.robot.data.computed_torques)
        return torch.sum(out_of_limits, dim=1)

    def base_height_l2(self, env: VelocityEnv, target_height: float):
        """Penalize base height from its target."""
        # TODO: Fix this for rough-terrain.
        base_height = env.robot.data.root_pos_w[:, 2]
        return torch.square(base_height - target_height)

    def lin_vel_xy_exp(self, env: VelocityEnv, std: float):
        """Tracking of linear velocity commands (xy axes).

        Args:
            std (float): Defines the width of the bell-curve.
        """
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.robot.data.root_lin_vel_b[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / std)

    def ang_vel_z_exp(self, env: VelocityEnv, std):
        """Tracking of angular velocity commands (yaw).

        Args:
            std (float): Defines the width of the bell-curve.
        """
        ang_vel_error = torch.square(env.commands[:, 2] - env.robot.data.root_ang_vel_b[:, 2])
        return torch.exp(-ang_vel_error / std)

    def feet_air_time(self, env: VelocityEnv, time_threshold: float):
        """Reward long steps taken by the feet."""
        first_contact = env.robot.data.feet_air_time > 0.0
        reward = torch.sum((env.robot.data.feet_air_time - time_threshold) * first_contact, dim=1)
        # no reward for zero command
        reward *= torch.norm(env.commands[:, :2], dim=1) > 0.1
        return reward
