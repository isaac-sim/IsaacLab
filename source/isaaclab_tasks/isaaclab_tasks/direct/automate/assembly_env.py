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
from . import automate_log_utils as automate_log
from . import factory_control as fc
from . import industreal_algo_utils as industreal_algo
from .assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, AssemblyEnvCfg
from .soft_dtw_cuda import SoftDTW


class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        self.cfg_task = cfg.tasks[cfg.task_name]

        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # Load asset meshes in warp for SDF-based dense reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path,
            self.cfg_task.assembly_dir + self.cfg_task.fixed_asset_cfg.obj_path,
            self.cfg_task.num_mesh_sample_points,
            self.wp_device,
        )

        # Get the gripper open width based on plug object bounding box
        self.gripper_open_width = automate_algo.get_gripper_open_width(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
        )

        # Create criterion for dynamic time warping (later used for imitation reward)
        self.soft_dtw_criterion = SoftDTW(use_cuda=True, device=self.device, gamma=self.cfg_task.soft_dtw_gamma)

        # Evaluate
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

    def _init_eval_logging(self):
        self.held_asset_pose_log = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )  # (position, quaternion)
        self.fixed_asset_pose_log = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self.success_log = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        # Turn off SBC during evaluation so all plugs are initialized outside of the socket
        self.cfg_task.if_sbc = False

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
        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)
        self._load_disassembly_data()

        # Load grasp pose from json files given assembly ID
        # Grasp pose tensors
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.robot_to_gripper_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
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

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # SBC
        if self.cfg_task.if_sbc:
            self.curr_max_disp = self.curriculum_height_bound[:, 0]
        else:
            self.curr_max_disp = self.curriculum_height_bound[:, 1]

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

    def _get_curriculum_info(self, disassembly_dists):
        """Calculate the ranges and step sizes for Sampling-based Curriculum (SBC) in each environment."""

        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.num_curriculum_step
        curriculum_height_step[:, 1] = -curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """Load pre-collected disassembly trajectories (end-effector position only)."""

        retrieve_file_path(self.cfg_task.disassembly_path_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.disassembly_path_json)) as f:
            disassembly_traj = json.load(f)

        eef_pos_traj = []

        for i in range(len(disassembly_traj)):
            curr_ee_traj = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))
            curr_ee_goal = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))[0, :]

            # offset each trajectory to be relative to the goal
            eef_pos_traj.append(curr_ee_traj - curr_ee_goal)

        self.eef_pos_traj = torch.tensor(np.array(eef_pos_traj), dtype=torch.float32, device=self.device).squeeze()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

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
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
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

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = combine_frame_transforms(
                self.held_base_pos, self.held_base_quat, keypoint_offset.repeat(self.num_envs, 1), self.identity_quat
            )[0]
            self.keypoints_fixed[:, idx] = combine_frame_transforms(
                self.target_held_base_pos,
                self.target_held_base_quat,
                keypoint_offset.repeat(self.num_envs, 1),
                self.identity_quat,
            )[0]

        self.keypoint_dist = torch.linalg.norm(self.keypoints_held - self.keypoints_fixed, ord=2, dim=-1).mean(-1)
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

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

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

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep

        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )

        rew_buf = self._update_rew_buf(curr_successes)
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs

            sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                curr_max_disp=self.curr_max_disp,
                curriculum_height_bound=self.curriculum_height_bound,
            )

            rew_buf *= sbc_rwd_scale

            if self.cfg_task.if_sbc:
                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(self.ep_succeeded) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound,
                    curriculum_height_step=self.curriculum_height_step,
                    curr_max_disp=self.curr_max_disp,
                )

            self.extras["curr_max_disp"] = self.curr_max_disp

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat([self.success_log, self.ep_succeeded.reshape((self.num_envs, 1))], dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.cfg_task.eval_filename,
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = dict({})

        # SDF-based reward.
        rew_dict["sdf"] = industreal_algo.get_sdf_reward(
            self.plug_mesh,
            self.plug_sample_points,
            self.held_pos,
            self.held_quat,
            self.fixed_pos,
            self.fixed_quat,
            self.wp_device,
            self.device,
        )

        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Imitation Reward: Calculate reward
        curr_eef_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).reshape(
            -1, 3
        )  # relative position instead of absolute position
        rew_dict["imitation"] = automate_algo.get_imitation_reward_from_dtw(
            self.eef_pos_traj, curr_eef_pos, self.prev_fingertip_midpoint_pos, self.soft_dtw_criterion, self.device
        )

        self.prev_fingertip_midpoint_pos = torch.cat(
            (self.prev_fingertip_midpoint_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
        )

        rew_buf = (
            self.cfg_task.sdf_rwd_scale * rew_dict["sdf"]
            + self.cfg_task.imitation_rwd_scale * rew_dict["imitation"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                [self.held_asset_pose_log, torch.cat([self.held_pos, self.held_quat], dim=1)], dim=0
            )
            self.fixed_asset_pose_log = torch.cat(
                [self.fixed_asset_pose_log, torch.cat([self.fixed_pos, self.fixed_quat], dim=1)], dim=0
            )

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)
        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = wp.to_torch(self._held_asset.data.default_root_state).clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = wp.to_torch(self._fixed_asset.data.default_root_state).clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
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
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.gripper_open_width
        joint_pos = wp.to_torch(self._robot.data.default_joint_pos)[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        # (1.) Randomize fixed asset pose.
        fixed_state = wp.to_torch(self._fixed_asset.data.default_root_state).clone()[env_ids]
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
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
            held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            held_asset_init_pos_rand = torch.tensor(
                self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
            )
            self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = wp.to_torch(self._held_asset.data.default_root_state).clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(self.curriculum_disp > self.disassembly_dists)
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

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

        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 1.0:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

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
