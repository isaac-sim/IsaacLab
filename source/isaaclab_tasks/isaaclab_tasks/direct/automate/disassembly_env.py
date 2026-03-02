# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from . import automate_algo_utils as automate_algo
from . import factory_control as fc
from .disassembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, DisassemblyEnvCfg


class DisassemblyEnv(DirectRLEnv):
    cfg: DisassemblyEnvCfg

    def __init__(self, cfg: DisassemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        self.cfg_task = cfg.tasks[cfg.task_name]

        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # Get the gripper open width based on plug object bounding box
        self.gripper_open_width = automate_algo.get_gripper_open_width(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
        )

        # initialized logging variables for disassembly paths
        self._init_log_data_per_assembly()

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = wp.to_torch(self._robot.root_view.get_inertias())
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_view.set_inertias(
            wp.from_torch(new_inertias), wp.from_torch(torch.arange(self.num_envs, dtype=torch.int32))
        )

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = wp.to_torch(asset.root_view.get_material_properties())
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu", dtype=torch.int32)
        asset.root_view.set_material_properties(wp.from_torch(materials), wp.from_torch(env_ids))

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        held_base_z_offset = 0.0

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()

        # Load grasp pose from json files given assembly ID
        # Grasp pose tensors
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.robot_to_gripper_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.plug_grasp_pos_local = self.plug_grasps[: self.num_envs, :3]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[: self.num_envs, 3:], -1, 1)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""

        retrieve_file_path(self.cfg_task.plug_grasp_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.plug_grasp_json)) as f:
            plug_grasp_dict = json.load(f)
        plug_grasps = [plug_grasp_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        retrieve_file_path(self.cfg_task.disassembly_dist_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.disassembly_dist_json)) as f:
            disassembly_dist_dict = json.load(f)
        disassembly_dists = [disassembly_dist_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.0, 0.0, 0.70711, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        # self._held_asset = Articulation(self.cfg_task.held_asset)
        # self._fixed_asset = RigidObject(self.cfg_task.fixed_asset)
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        # self.scene.articulations["held_asset"] = self._held_asset
        # self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = wp.to_torch(self._fixed_asset.data.root_pos_w) - self.scene.env_origins
        self.fixed_quat = wp.to_torch(self._fixed_asset.data.root_quat_w)

        self.held_pos = wp.to_torch(self._held_asset.data.root_pos_w) - self.scene.env_origins
        self.held_quat = wp.to_torch(self._held_asset.data.root_quat_w)

        self.fingertip_midpoint_pos = (
            wp.to_torch(self._robot.data.body_pos_w)[:, self.fingertip_body_idx] - self.scene.env_origins
        )
        self.fingertip_midpoint_quat = wp.to_torch(self._robot.data.body_quat_w)[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = wp.to_torch(self._robot.data.body_lin_vel_w)[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = wp.to_torch(self._robot.data.body_ang_vel_w)[:, self.fingertip_body_idx]

        jacobians = wp.to_torch(self._robot.root_view.get_jacobians())

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = wp.to_torch(self._robot.root_view.get_generalized_mass_matrices())[:, 0:7, 0:7]
        self.joint_pos = wp.to_torch(self._robot.data.joint_pos).clone()
        self.joint_vel = wp.to_torch(self._robot.data.joint_vel).clone()

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_pos, self.gripper_goal_quat = combine_frame_transforms(
            self.fixed_pos,
            self.fixed_quat,
            self.plug_grasp_pos_local,
            self.plug_grasp_quat_local,
        )

        self.gripper_goal_pos, self.gripper_goal_quat = combine_frame_transforms(
            self.gripper_goal_pos,
            self.gripper_goal_quat,
            self.palm_to_finger_center,
            self.robot_to_gripper_quat,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = quat_mul(self.fingertip_midpoint_quat, quat_conjugate(self.prev_fingertip_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 3]).unsqueeze(-1)  # W component is at index 3 in XYZW format
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_pos[:], self.held_base_quat[:] = combine_frame_transforms(
            self.held_pos, self.held_quat, self.held_base_pos_local, self.held_base_quat_local
        )
        self.target_held_base_pos[:], self.target_held_base_quat[:] = combine_frame_transforms(
            self.fixed_pos, self.fixed_quat, self.fixed_success_pos_local, self.identity_quat
        )

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""

        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
        }

        state_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "joint_vel": self.joint_vel[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
        }
        # obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        # state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

    def move_gripper_in_place(self, ctrl_target_gripper_dof_pos):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.linalg.norm(rot_actions, ord=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(euler_xyz_from_quat(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = euler_xyz_from_quat(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.linalg.norm(rot_actions, ord=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(euler_xyz_from_quat(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use GYM's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target_index(target=self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if time_out[0]:
            self.close_gripper(env_ids=np.array(range(self.num_envs)).reshape(-1))
            self._disassemble_plug_from_socket()

            if_intersect = (self.held_pos[:, 2] < self.fixed_pos[:, 2] + self.disassembly_dists).cpu().numpy()
            success_env_ids = np.argwhere(if_intersect == 0).reshape(-1)

            self._log_robot_state(success_env_ids)
            self._log_object_state(success_env_ids)
            self._save_log_traj()

        return time_out, time_out

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep

        rew_buf = self._update_rew_buf()
        return rew_buf

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        return torch.zeros((self.num_envs,), device=self.device)

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)
        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)
        self._init_log_data_per_episode()

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = torch.cat(
            [wp.to_torch(self._held_asset.data.default_root_pose), wp.to_torch(self._held_asset.data.default_root_vel)],
            dim=-1,
        )[env_ids].clone()
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim_index(root_pose=held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim_index(root_velocity=held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = torch.cat(
            [
                wp.to_torch(self._fixed_asset.data.default_root_pose),
                wp.to_torch(self._fixed_asset.data.default_root_vel),
            ],
            dim=-1,
        )[env_ids].clone()
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim_index(root_pose=fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim_index(root_velocity=fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""

        gripper_goal_pos, gripper_goal_quat = combine_frame_transforms(
            self.held_pos,
            self.held_quat,
            self.plug_grasp_pos_local,
            self.plug_grasp_quat_local,
        )

        gripper_goal_pos, gripper_goal_quat = combine_frame_transforms(
            gripper_goal_pos,
            gripper_goal_quat,
            self.palm_to_finger_center,
            self.robot_to_gripper_quat,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.50:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_position_to_sim_index(position=self.joint_pos)
            self._robot.write_joint_velocity_to_sim_index(velocity=self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _move_gripper_to_eef_pose(self, env_ids, goal_pos, goal_quat, sim_steps, if_log=False):
        for _ in range(sim_steps):
            if if_log:
                self._log_robot_state_per_timestep()

            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=goal_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=goal_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            self.actions *= 0.0
            self.actions[env_ids, :6] = delta_hand_pose

            is_rendering = self.sim.is_rendering
            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self._apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

            # Simulate and update tensors.
            self.step_sim_no_action()

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        # gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        # gripper_width = self.cfg_task.hand_width_max / 3.0
        gripper_width = self.gripper_open_width
        joint_pos = wp.to_torch(self._robot.data.default_joint_pos)[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target_index(target=joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        # (1.) Randomize fixed asset pose.
        fixed_state = torch.cat(
            [
                wp.to_torch(self._fixed_asset.data.default_root_pose),
                wp.to_torch(self._fixed_asset.data.default_root_vel),
            ],
            dim=-1,
        )[env_ids].clone()
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        fixed_state[:, 2] += self.cfg_task.fixed_asset_z_offset

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = quat_from_euler_xyz(fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2])
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim_index(root_pose=fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim_index(root_velocity=fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        # Set plug pos to assembled state
        held_state = torch.cat(
            [wp.to_torch(self._held_asset.data.default_root_pose), wp.to_torch(self._held_asset.data.default_root_vel)],
            dim=-1,
        ).clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        self._held_asset.write_root_pose_to_sim_index(root_pose=held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim_index(root_velocity=held_state[:, 7:])
        self._held_asset.reset()

        self.step_sim_no_action()

    def close_gripper(self, env_ids):
        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        import carb

        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        fixed_tip_pos, _ = combine_frame_transforms(
            self.fixed_pos, self.fixed_quat, fixed_tip_pos_local, self.identity_quat
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids, pre_grasp=True)

        self._move_gripper_to_grasp_pose(env_ids)

        self.randomize_held_initial_state(env_ids, pre_grasp=False)

        self.close_gripper(env_ids)

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        import carb

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _disassemble_plug_from_socket(self):
        """Lift plug from socket till disassembly and then randomize end-effector pose."""

        if_intersect = np.ones(self.num_envs, dtype=np.float32)

        env_ids = np.argwhere(if_intersect == 1).reshape(-1)
        self._lift_gripper(self.disassembly_dists * 3.0, self.cfg_task.disassemble_sim_steps, env_ids)

        self.step_sim_no_action()

        if_intersect = (self.held_pos[:, 2] < self.fixed_pos[:, 2] + self.disassembly_dists).cpu().numpy()
        env_ids = np.argwhere(if_intersect == 0).reshape(-1)
        self._randomize_gripper_pose(self.cfg_task.move_gripper_sim_steps, env_ids)

    def _lift_gripper(self, lift_distance, sim_steps, env_ids=None):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        ctrl_tgt_pos = torch.empty_like(self.fingertip_midpoint_pos).copy_(self.fingertip_midpoint_pos)
        ctrl_tgt_quat = torch.empty_like(self.fingertip_midpoint_quat).copy_(self.fingertip_midpoint_quat)
        ctrl_tgt_pos[:, 2] += lift_distance
        if len(env_ids) == 0:
            env_ids = np.array(range(self.num_envs)).reshape(-1)

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=True)

    def _randomize_gripper_pose(self, sim_steps, env_ids):
        """Move gripper to random pose."""

        ctrl_tgt_pos = torch.empty_like(self.gripper_goal_pos).copy_(self.gripper_goal_pos)
        ctrl_tgt_pos[:, 2] += self.cfg_task.gripper_rand_z_offset

        # ctrl_tgt_pos = torch.empty_like(self.fingertip_midpoint_pos).copy_(self.fingertip_midpoint_pos)

        fingertip_centered_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        fingertip_centered_pos_noise = fingertip_centered_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.gripper_rand_pos_noise, device=self.device)
        )
        ctrl_tgt_pos += fingertip_centered_pos_noise

        # Set target rot
        ctrl_target_fingertip_centered_euler = (
            torch.tensor(self.cfg_task.fingertip_centered_rot_initial, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        fingertip_centered_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        fingertip_centered_rot_noise = fingertip_centered_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.gripper_rand_rot_noise, device=self.device)
        )
        ctrl_target_fingertip_centered_euler += fingertip_centered_rot_noise
        ctrl_tgt_quat = quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        # ctrl_tgt_quat = torch.empty_like(self.fingertip_midpoint_quat).copy_(self.fingertip_midpoint_quat)

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=True)

    def _init_log_data_per_assembly(self):
        self.log_assembly_id = []
        self.log_plug_pos = []
        self.log_plug_quat = []
        self.log_init_plug_pos = []
        self.log_init_plug_quat = []
        self.log_plug_grasp_pos = []
        self.log_plug_grasp_quat = []
        self.log_fingertip_centered_pos = []
        self.log_fingertip_centered_quat = []
        self.log_arm_dof_pos = []

    def _init_log_data_per_episode(self):
        self.log_fingertip_centered_pos_traj = []
        self.log_fingertip_centered_quat_traj = []
        self.log_arm_dof_pos_traj = []
        self.log_plug_pos_traj = []
        self.log_plug_quat_traj = []

        self.init_plug_grasp_pos = self.gripper_goal_pos.clone().detach()
        self.init_plug_grasp_quat = self.gripper_goal_quat.clone().detach()
        self.init_plug_pos = self.held_pos.clone().detach()
        self.init_plug_quat = self.held_quat.clone().detach()

    def _log_robot_state(self, env_ids):
        self.log_plug_pos += torch.stack(self.log_plug_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_plug_quat += torch.stack(self.log_plug_quat_traj, dim=1)[env_ids].cpu().tolist()
        self.log_arm_dof_pos += torch.stack(self.log_arm_dof_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_fingertip_centered_pos += (
            torch.stack(self.log_fingertip_centered_pos_traj, dim=1)[env_ids].cpu().tolist()
        )
        self.log_fingertip_centered_quat += (
            torch.stack(self.log_fingertip_centered_quat_traj, dim=1)[env_ids].cpu().tolist()
        )

    def _log_robot_state_per_timestep(self):
        self.log_plug_pos_traj.append(self.held_pos.clone().detach())
        self.log_plug_quat_traj.append(self.held_quat.clone().detach())
        self.log_arm_dof_pos_traj.append(self.joint_pos[:, 0:7].clone().detach())
        self.log_fingertip_centered_pos_traj.append(self.fingertip_midpoint_pos.clone().detach())
        self.log_fingertip_centered_quat_traj.append(self.fingertip_midpoint_quat.clone().detach())

    def _log_object_state(self, env_ids):
        self.log_plug_grasp_pos += self.init_plug_grasp_pos[env_ids].cpu().tolist()
        self.log_plug_grasp_quat += self.init_plug_grasp_quat[env_ids].cpu().tolist()
        self.log_init_plug_pos += self.init_plug_pos[env_ids].cpu().tolist()
        self.log_init_plug_quat += self.init_plug_quat[env_ids].cpu().tolist()

    def _save_log_traj(self):
        if len(self.log_arm_dof_pos) > self.cfg_task.num_log_traj:
            log_item = []
            for i in range(self.cfg_task.num_log_traj):
                curr_dict = dict({})
                curr_dict["fingertip_centered_pos"] = self.log_fingertip_centered_pos[i]
                curr_dict["fingertip_centered_quat"] = self.log_fingertip_centered_quat[i]
                curr_dict["arm_dof_pos"] = self.log_arm_dof_pos[i]
                curr_dict["plug_grasp_pos"] = self.log_plug_grasp_pos[i]
                curr_dict["plug_grasp_quat"] = self.log_plug_grasp_quat[i]
                curr_dict["init_plug_pos"] = self.log_init_plug_pos[i]
                curr_dict["init_plug_quat"] = self.log_init_plug_quat[i]
                curr_dict["plug_pos"] = self.log_plug_pos[i]
                curr_dict["plug_quat"] = self.log_plug_quat[i]

                log_item.append(curr_dict)

            log_filename = os.path.join(
                os.getcwd(), self.cfg_task.disassembly_dir, self.cfg_task.assembly_id + "_disassemble_traj.json"
            )

            with open(log_filename, "w+") as out_file:
                json.dump(log_item, out_file, indent=6)

            print(f"Trajectory collection complete! Collected {len(self.log_arm_dof_pos)} trajectories!")
            exit(0)
        else:
            print(
                f"Collected {len(self.log_arm_dof_pos)} trajectories so far (target: > {self.cfg_task.num_log_traj})."
            )
