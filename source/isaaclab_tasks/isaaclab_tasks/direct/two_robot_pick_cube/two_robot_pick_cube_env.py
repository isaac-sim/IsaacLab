# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import random
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, ContactSensor, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import normalize, quat_from_euler_xyz, sample_uniform, transform_points

from .two_robot_pick_cube_env_cfg import TwoRobotPickCubeCfg


class TwoRobotPickCubeEnv(DirectRLEnv):
    """
    TwoRobotPickCube environment inspired by ManiSkill3's TwoRobotPickCube-v1.

    Two Franka Panda robots work together to pick up a cube and place it at a randomized goal location.

    The environment provides both state and visual observations, and implements a dense, multi-stage reward.
    """

    cfg: TwoRobotPickCubeCfg

    def __init__(self, cfg: TwoRobotPickCubeCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the TwoRobotPickCube environment.

        Parameters:
            cfg (TwoRobotPickCubeCfg): Environment configuration.
            render_mode (str | None): Rendering mode, if any.
            **kwargs: Additional keyword arguments for DirectRLEnv.
        """
        super().__init__(cfg, render_mode, **kwargs)
        random.seed(self.cfg.seed)
        # find actuated joints for both robots (including fingers)
        self.joint_ids, _ = self.robot_left.find_joints("panda_joint.*|panda_finger_joint.*")

        # cache fingertip link indices for grasp detection
        self.left_finger_link_idx = self.robot_left.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot_left.find_bodies("panda_rightfinger")[0][0]

    def _setup_scene(self):
        """
        Set up the simulation scene:
        - Ground plane
        - Two Panda robots
        - Camera sensor
        - TCP and finger frame transformers
        - Target marker and cube object
        - Dome lighting
        """
        # ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0))

        # instantiate robots
        self.robot_left = Articulation(self.cfg.robot_left_cfg)
        self.robot_right = Articulation(self.cfg.robot_right_cfg)
        self.scene.articulations["robot_left"] = self.robot_left
        self.scene.articulations["robot_right"] = self.robot_right

        # camera
        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # Cube
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        # frame transformers for TCPs
        self.tcp_transformer_left = FrameTransformer(cfg=self.cfg.tcp_left_cfg)
        self.scene.sensors["tcp_left"] = self.tcp_transformer_left
        self.tcp_transformer_right = FrameTransformer(cfg=self.cfg.tcp_right_cfg)
        self.scene.sensors["tcp_right"] = self.tcp_transformer_right

        # frame transformers for fingertips
        self.left_finger_transformer = FrameTransformer(cfg=self.cfg.left_finger_cfg)
        self.scene.sensors["left_finger"] = self.left_finger_transformer
        self.right_finger_transformer = FrameTransformer(cfg=self.cfg.right_finger_cfg)
        self.scene.sensors["right_finger"] = self.right_finger_transformer

        # target marker
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)
        self.scene.extras["target"] = self.target_marker

        # contact sensors for grasp detection
        self.scene.sensors["right_robot_left_contact"] = ContactSensor(self.cfg.sensors[1])
        self.scene.sensors["right_robot_right_contact"] = ContactSensor(self.cfg.sensors[2])

        # replicate and add robots and cube
        self.scene.clone_environments(copy_from_source=True)

        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Cache actions before physics step.

        Parameters:
            actions (torch.Tensor): Action tensor of shape (N, 18).
        """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """
        Apply joint efforts to both robots from the cached actions.
        """
        left_effort = self.actions[:, 0:9]
        right_effort = self.actions[:, 9:18]
        if self.cfg.robot_controller == "task_space":
            self.robot_left.set_joint_position_target(left_effort, joint_ids=self.joint_ids)
            self.robot_right.set_joint_position_target(right_effort, joint_ids=self.joint_ids)
        else:
            self.robot_left.set_joint_effort_target(left_effort, joint_ids=self.joint_ids)
            self.robot_right.set_joint_effort_target(right_effort, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        """
        Compute observations for the RL policy.

        Returns:
            obs (dict): Contains:
                - state (torch.Tensor): State vector (N,30) if obs_mode is "state".
                - rgb (torch.Tensor): Image tensor (N,3,H,W) if obs_mode includes visual.
                - policy (torch.Tensor): The active modality for policy input.
        """
        tcp_poses = self.get_tcp_poses()
        cube_state = self.cube.data.root_state_w
        cube_state[:, :3] -= self.scene.env_origins
        cube_pos = cube_state[:, :3]

        pos_left = tcp_poses[:, :3]
        pos_right = tcp_poses[:, 7:10]
        left_to_cube = cube_pos - pos_left
        right_to_cube = cube_pos - pos_right
        cube_to_goal = self.target_pose[:, :3] - cube_pos

        state_obs = torch.cat(
            (
                tcp_poses,
                cube_state[:, :7],
                left_to_cube,
                right_to_cube,
                cube_to_goal,
            ),
            dim=-1,
        )
        rgb = self.scene.sensors["camera"].data.output["rgb"].to(torch.float32) / 255.0
        rgb_tensor = rgb.permute(0, 3, 1, 2).clone()

        if self.cfg.obs_mode == "state":
            return {"state": state_obs, "policy": state_obs}
        else:
            return {"rgb": rgb_tensor, "policy": rgb_tensor}

    def get_tcp_poses(self) -> torch.Tensor:
        """
        Concatenate left and right TCP poses.

        Returns:
            torch.Tensor: Shape (N,14) with [pos_L, quat_L, pos_R, quat_R].
        """
        quat_L = self.tcp_transformer_left.data.target_quat_w.squeeze(1)
        pos_L = self.tcp_transformer_left.data.target_pos_w.squeeze(1) - self.scene.env_origins
        quat_R = self.tcp_transformer_right.data.target_quat_w.squeeze(1)
        pos_R = self.tcp_transformer_right.data.target_pos_w.squeeze(1) - self.scene.env_origins
        return torch.cat((pos_L, quat_L, pos_R, quat_R), dim=1)

    def is_grasping(
        self,
        robot: Articulation,
        left_contact_name: str,
        right_contact_name: str,
        min_force: float = 0.2,
        max_angle: float = 130.0,
    ) -> torch.Tensor:
        """
        Check if the given robot is grasping an object by evaluating
        contact forces and approach angles on its fingertip sensors.

        Args:
            robot (Articulation): Robot to test.
            left_contact_name (str): Sensor name for left fingertip.
            right_contact_name (str): Sensor name for right fingertip.
            min_force (float): Minimum normal force to consider a contact.
            max_angle (float): Maximum angle (deg) between contact force and opening axis.
        Returns:
            torch.Tensor: Bool mask (N,) True if both fingers grasp.
        """
        left_sensor = self.scene.sensors[left_contact_name]
        right_sensor = self.scene.sensors[right_contact_name]
        l_f = left_sensor.data.net_forces_w.squeeze(1)
        r_f = right_sensor.data.net_forces_w.squeeze(1)
        l_mag = torch.linalg.norm(l_f, dim=1)
        r_mag = torch.linalg.norm(r_f, dim=1)

        N = l_f.shape[0]
        axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 1, 3)
        axes = axis_local.expand(N, 1, 3)
        pos = robot.data.body_pos_w
        quat = robot.data.body_quat_w
        l_dir = transform_points(axes, pos[:, self.left_finger_link_idx], quat[:, self.left_finger_link_idx]).squeeze(1)
        r_dir = transform_points(
            axes,
            pos[:, self.right_finger_link_idx],
            quat[:, self.right_finger_link_idx],
        ).squeeze(1)
        r_dir = -r_dir  # invert right axis

        l_dir_u = normalize(l_dir)
        r_dir_u = normalize(r_dir)
        l_f_u = normalize(l_f)
        r_f_u = normalize(r_f)
        l_ang = torch.acos((l_dir_u * l_f_u).sum(-1).clamp(-1, 1)) * (180.0 / torch.pi)
        r_ang = torch.acos((r_dir_u * r_f_u).sum(-1).clamp(-1, 1)) * (180.0 / torch.pi)

        l_ok = (l_mag >= min_force) & (l_ang <= max_angle)
        r_ok = (r_mag >= min_force) & (r_ang <= max_angle)
        return l_ok & r_ok

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute multi-stage dense reward.

        Returns:
            torch.Tensor: Reward tensor of shape (N,).
        """
        tcp = self.get_tcp_poses()
        tcp_L = tcp[:, :3]
        tcp_R = tcp[:, 7:10]
        cube_w = self.cube.data.root_state_w
        cube_pos = cube_w[:, :3] - self.scene.env_origins
        goal_pos = self.target_pose[:, :3]

        # Stage 1: reach & push
        dist_L = torch.linalg.norm(cube_pos - tcp_L, dim=-1)
        reach_reward_stage1 = 1 - torch.tanh(5 * dist_L)
        beyond = torch.relu(cube_pos[:, 1] + 0.05)
        push_reward_stage1 = 1 - torch.tanh(5 * beyond)
        reward = 0.5 * (reach_reward_stage1 + push_reward_stage1)
        push_condition_mask = cube_pos[:, 1] <= 0.0

        # Pre-grasp check
        left_finger_position = self.left_finger_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        right_finger_position = self.right_finger_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        left_finger_height, right_finger_height = left_finger_position[:, 2], right_finger_position[:, 2]
        height_difference_reward = 1 - torch.tanh(5 * torch.abs(left_finger_height - right_finger_height))
        distance_between_fingers = torch.linalg.norm(left_finger_position - right_finger_position, dim=-1)
        tip_distance_reward = 1 - torch.tanh(5 * torch.abs(distance_between_fingers - 0.07))
        tip_reward = 0.5 * (height_difference_reward + tip_distance_reward)

        # Stage 2: reach + tip + leave + pre-grasp
        dist_R = torch.linalg.norm(cube_pos - tcp_R, dim=-1)
        reach_reward_stage2 = 1 - torch.tanh(5 * dist_R)
        left_arm_leave_reward = 1 - torch.tanh(5 * torch.abs(tcp[:, 1] - 0.2))
        is_grasping = self.is_grasping(self.robot_right, "right_robot_left_contact", "right_robot_right_contact")
        reward[push_condition_mask] = (
            2.0
            + reach_reward_stage2[push_condition_mask]
            + tip_reward[push_condition_mask]
            + left_arm_leave_reward[push_condition_mask]
            + 2.0 * is_grasping[push_condition_mask]
        )

        # Stage 3: move toward goal + left return
        dist_goal = torch.linalg.norm(goal_pos - tcp_R, dim=-1)
        goal_reach_reward = 1 - torch.tanh(5 * dist_goal)
        left_qpos = self.robot_left.data.joint_pos
        left_init_qpos_reward = 1 - torch.tanh(torch.linalg.norm(left_qpos - self.left_init_qpos, dim=-1))
        reward[is_grasping] = 8.0 + (2.0 * goal_reach_reward + left_init_qpos_reward)[is_grasping]
        near_goal_grasp_mask = is_grasping & (dist_goal < 0.25)

        # Stage 4: intermediate near-goal bonus
        reward[near_goal_grasp_mask] = 12.0 + 2.0 * goal_reach_reward[near_goal_grasp_mask]

        # Stage 5: placed + static-arms
        is_placed = torch.linalg.norm(goal_pos - cube_pos, dim=-1) <= self.cfg.success_distance_threshold
        rv = self.robot_right.data.joint_vel[:, :-3]
        lv = self.robot_left.data.joint_vel[:, :-3]
        rs = 1 - torch.tanh(5 * torch.linalg.norm(rv, dim=-1))
        ls = 1 - torch.tanh(5 * torch.linalg.norm(lv, dim=-1))
        static = 0.5 * (rs + ls)
        reward[is_placed] = 19.0 + static[is_placed]

        # Final success overwrite
        success = self.is_success()
        reward[success] = 21.0
        return reward

    def is_robot_static(self, robot: Articulation, threshold: float = 0.2) -> torch.Tensor:
        """
        Check whether all actuated joints of a robot are below velocity threshold.

        Parameters:
            robot (Articulation): The robot articulation to check.
            threshold (float): Velocity threshold in rad/s.

        Returns:
            torch.Tensor: Boolean mask (N,) where True indicates static.
        """
        v = robot.data.joint_vel[:, self.joint_ids[:-2]]
        return torch.all(torch.abs(v) <= threshold, dim=-1)

    def is_success(self) -> torch.Tensor:
        """
        Check success condition: cube within threshold and right arm static.

        Returns:
            torch.Tensor: Boolean mask (N,) of success states.
        """
        cube_p = self.cube.data.root_state_w[:, :3] - self.scene.env_origins
        dist = torch.linalg.norm(cube_p - self.target_pose[:, :3], dim=-1)
        return (dist < self.cfg.success_distance_threshold) & self.is_robot_static(self.robot_right)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns done and timeout signals.

        Returns:
            done (torch.Tensor): Success mask (N,).
            timeout (torch.Tensor): Timeout mask (N,) based on episode length.
        """
        done = self.is_success()
        if self.cfg.robot_controller == "task_space":
            timeout = torch.zeros_like(done, dtype=torch.bool)
        else:
            timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        return done, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specified envs: robots, cube, target.

        Parameters:
            env_ids (Sequence[int] | None): Indices to reset. All if None.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        super()._reset_idx(env_ids)
        self.reset_robot(env_ids)
        self.sample_init_cube_pose(env_ids)
        self.sample_target_pose(env_ids)

    def reset_robot(self, env_ids: Sequence[int] | None = None):
        """
        Reset robot states to default and cache left-initial qpos.

        Parameters:
            env_ids (Sequence[int] | None): Indices to reset. All if None.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        # left
        jp = self.robot_left.data.default_joint_pos[env_ids]
        jv = self.robot_left.data.default_joint_vel[env_ids]
        rs = self.robot_left.data.default_root_state[env_ids]
        rs[:, :3] += self.scene.env_origins[env_ids]
        self.robot_left.write_root_pose_to_sim(rs[:, :7], env_ids)
        self.robot_left.write_root_velocity_to_sim(rs[:, 7:], env_ids)
        self.robot_left.write_joint_state_to_sim(jp, jv, None, env_ids)
        # right
        rsr = self.robot_right.data.default_root_state[env_ids]
        rsr[:, :3] += self.scene.env_origins[env_ids]
        self.robot_right.write_root_pose_to_sim(rsr[:, :7], env_ids)
        self.robot_right.write_root_velocity_to_sim(rsr[:, 7:], env_ids)
        self.robot_right.write_joint_state_to_sim(jp, jv, None, env_ids)
        # cache
        self.left_init_qpos = self.robot_left.data.joint_pos.clone()

    def sample_init_cube_pose(self, env_ids: Sequence[int] | None = None):
        """
        Randomize initial cube pose within cfg-defined range.

        Parameters:
            env_ids (Sequence[int] | None): Indices to sample. All if None.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        default = self.cube.data.default_root_state[env_ids]
        low = torch.tensor(self.cfg.cube_sample_range[0], device=self.device)
        high = torch.tensor(self.cfg.cube_sample_range[1], device=self.device)
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)
        pos = rand6[:, :3] + self.scene.env_origins[env_ids]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        state = torch.cat((pos, quat), dim=-1)
        self.cube.write_root_pose_to_sim(state[env_ids], env_ids)
        self.cube.write_root_velocity_to_sim(default[env_ids, 7:], env_ids)

    def sample_target_pose(self, env_ids: Sequence[int] | None = None):
        """
        Randomize target (goal) pose for the cube.

        Parameters:
            env_ids (Sequence[int] | None): Indices to sample. All if None.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        low = torch.tensor(self.cfg.target_sample_range[0], device=self.device)
        high = torch.tensor(self.cfg.target_sample_range[1], device=self.device)
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)
        pos = rand6[:, :3]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        self.target_pose = torch.cat((pos, quat), dim=-1)
        idxs = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        self.target_marker.visualize(pos + self.scene.env_origins, quat, marker_indices=idxs)
