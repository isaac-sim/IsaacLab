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
from isaaclab.sensors import Camera, FrameTransformer, ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    sample_uniform,
    transform_points,
    normalize,
)

from .two_robot_stack_cube_env_cfg import TwoRobotStackCubeCfg


class TwoRobotStackCubeEnv(DirectRLEnv):
    """
    TwoRobotStackCube environment for stacking a cube using two robots.
    """

    cfg: TwoRobotStackCubeCfg

    def __init__(
        self, cfg: TwoRobotStackCubeCfg, render_mode: str | None = None, **kwargs
    ):
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
        self.joint_ids, _ = self.robot_left.find_joints(
            "panda_joint.*|panda_finger_joint.*"
        )

        self.left_finger_link_idx = self.robot_left.find_bodies("panda_leftfinger")[0][
            0
        ]
        self.right_finger_link_idx = self.robot_left.find_bodies("panda_rightfinger")[
            0
        ][0]

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
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0)
        )

        # instantiate robots
        self.robot_left = Articulation(self.cfg.robot_left_cfg)
        self.robot_right = Articulation(self.cfg.robot_right_cfg)
        self.scene.articulations["robot_left"] = self.robot_left
        self.scene.articulations["robot_right"] = self.robot_right

        # camera
        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # cubes
        self.cube_green = RigidObject(self.cfg.cube_green_cfg)
        self.scene.rigid_objects["cube_green"] = self.cube_green
        self.cube_red = RigidObject(self.cfg.cube_red_cfg)
        self.scene.rigid_objects["cube_red"] = self.cube_red

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

        # contact sensors for both robots
        self.scene.sensors["left_robot_left_contact"] = ContactSensor(
            self.cfg.sensors[1]
        )
        self.scene.sensors["left_robot_right_contact"] = ContactSensor(
            self.cfg.sensors[2]
        )
        self.scene.sensors["right_robot_left_contact"] = ContactSensor(
            self.cfg.sensors[3]
        )
        self.scene.sensors["right_robot_right_contact"] = ContactSensor(
            self.cfg.sensors[4]
        )

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
        Apply joint effort targets to both robots based on stored actions.
        """
        self.robot_left.set_joint_effort_target(
            self.actions[:, 0:9], joint_ids=self.joint_ids
        )
        self.robot_right.set_joint_effort_target(
            self.actions[:, 9:18], joint_ids=self.joint_ids
        )

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

        target_pos = self.target_pose[:, :3]

        cube_green_state = self.cube_green.data.root_state_w
        cube_green_state[:, :3] -= self.scene.env_origins
        cube_green_pos = cube_green_state[:, :3]

        cube_red_state = self.cube_red.data.root_state_w
        cube_red_state[:, :3] -= self.scene.env_origins
        cube_red_pos = cube_red_state[:, :3]

        pos_left = tcp_poses[:, :3]
        pos_right = tcp_poses[:, 7:10]
        left_to_cube_green = cube_green_pos - pos_left
        right_to_cube_red = cube_red_pos - pos_right

        cube_green_to_cube_red = cube_red_pos - cube_green_pos

        state_obs = torch.cat(
            (
                tcp_poses,
                target_pos,
                cube_green_state[:, :7],
                cube_red_state[:, :7],
                left_to_cube_green,
                right_to_cube_red,
                cube_green_to_cube_red,
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
        pos_L = (
            self.tcp_transformer_left.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )
        quat_R = self.tcp_transformer_right.data.target_quat_w.squeeze(1)
        pos_R = (
            self.tcp_transformer_right.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )
        return torch.cat((pos_L, quat_L, pos_R, quat_R), dim=1)

    def is_grasping(
        self,
        robot: Articulation,
        left_contact_name: str,
        right_contact_name: str,
        min_force: float = 0.5,
        max_angle: float = 85.0,
    ) -> torch.Tensor:
        """
        Returns (N,) mask where the given robot is grasping the object
        that those two ContactSensors are filtered to see.
        """
        # grab the two sensors
        left_sensor = self.scene.sensors[left_contact_name]  # type: ContactSensor
        right_sensor = self.scene.sensors[right_contact_name]  # type: ContactSensor

        # 1) Net world‐frame forces: (N,1,3) → (N,3)
        l_f = left_sensor.data.net_forces_w.squeeze(1)
        r_f = right_sensor.data.net_forces_w.squeeze(1)

        # 2) Magnitudes
        l_mag = torch.linalg.norm(l_f, dim=1)
        r_mag = torch.linalg.norm(r_f, dim=1)

        # 3) “Opening” axis in fingertip‐local = +Y
        N = l_f.shape[0]
        axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 1, 3)
        axes = axis_local.expand(N, 1, 3)

        # 4) Rotate that +Y into world at each fingertip link
        body_pos = robot.data.body_pos_w  # (N, B, 3)
        body_quat = robot.data.body_quat_w  # (N, B, 4)
        l_dir = transform_points(
            points=axes,
            pos=body_pos[:, self.left_finger_link_idx],
            quat=body_quat[:, self.left_finger_link_idx],
        ).squeeze(1)
        r_dir = transform_points(
            points=axes,
            pos=body_pos[:, self.right_finger_link_idx],
            quat=body_quat[:, self.right_finger_link_idx],
        ).squeeze(1)
        # flip the right finger so +r_dir is “opening”
        r_dir = -r_dir

        # 5) Unit‐vectors
        l_dir_u = normalize(l_dir)
        r_dir_u = normalize(r_dir)
        l_f_u = normalize(l_f)
        r_f_u = normalize(r_f)

        # 6) Angle between force & opening‐axis, in degrees
        l_ang = torch.acos((l_dir_u * l_f_u).sum(-1).clamp(-1.0, 1.0)) * (
            180.0 / torch.pi
        )
        r_ang = torch.acos((r_dir_u * r_f_u).sum(-1).clamp(-1.0, 1.0)) * (
            180.0 / torch.pi
        )

        # 7) final check
        l_ok = (l_mag >= min_force) & (l_ang <= max_angle)
        r_ok = (r_mag >= min_force) & (r_ang <= max_angle)
        return l_ok & r_ok

    # TODO test this reward function
    def _get_rewards(self) -> torch.Tensor:
        """
        Dense, multi-stage reward adapted from ManiSkill’s compute_dense_reward.
        """
        # 1) common quantities
        N = self.num_envs
        tcp = self.get_tcp_poses()
        pos_L = tcp[:, :3]  # left TCP
        pos_R = tcp[:, 7:10]  # right TCP

        # cube positions in env frame
        green_w = self.cube_green.data.root_state_w[:, :3] - self.scene.env_origins
        red_w = self.cube_red.data.root_state_w[:, :3] - self.scene.env_origins

        # half edge of DexCube (0.033 m * 0.8 scale / 2)
        half_edge = 0.8 * 0.033 / 2.0

        # prepare reward tensor
        reward = torch.zeros(N, device=self.device)

        # Stage 1: reach & pre-grasp
        # left reach to bottom cube
        d_left = torch.linalg.norm(pos_L - green_w, dim=-1)
        # right reach toward a “push” point on top of red cube
        push_target = red_w + torch.tensor(
            [0.0, half_edge + 0.005, 0.0], device=self.device
        )
        d_right_push = torch.linalg.norm(pos_R - push_target, dim=-1)
        reach_reward = (
            1 - torch.tanh(5 * d_left) + 1 - torch.tanh(5 * d_right_push)
        ) / 2

        # grasp flags
        is_grasp_green = self.is_grasping(
            self.robot_left,
            "left_robot_left_contact",
            "left_robot_right_contact",
        ).to(self.device)

        # combine reach + grasp bonus
        reward[:] = (reach_reward + is_grasp_green) / 2

        # Stage 2: place bottom cube on target while holding
        on_target = (
            torch.linalg.norm(green_w[:, :2] - self.target_pose[:, :2], dim=-1)
            < self.cfg.goal_radius
        )
        place_bottom = 1 - torch.tanh(
            5 * torch.linalg.norm(green_w[:, :2] - self.target_pose[:, :2], dim=-1)
        )
        mask2 = is_grasp_green.bool()
        reward[mask2] = 2.0 + (place_bottom[mask2] + is_grasp_green[mask2]) / 2

        # 4) Stage 3: stack top cube on bottom & right arm leave
        placed_bottom = on_target & is_grasp_green.bool()
        # target for top cube = directly above green by 2×half_edge
        stack_target = torch.cat(
            [green_w[:, :2], (green_w[:, 2] + 2 * half_edge).unsqueeze(-1)], dim=1
        )
        place_top = 1 - torch.tanh(5 * torch.linalg.norm(stack_target - red_w, dim=-1))
        leave_right = 1 - torch.tanh(5 * torch.abs(pos_R[:, 1] + 0.2))
        mask3 = placed_bottom
        reward[mask3] = 4.0 + (place_top[mask3] * 2 + leave_right[mask3])

        # 5) Stage 4: release both grippers
        stacked = (
            torch.linalg.norm(red_w[:, :2] - green_w[:, :2], dim=-1)
            <= (2 * half_edge + 0.005)
        ) & (torch.abs(red_w[:, 2] - (green_w[:, 2] + 2 * half_edge)) <= 0.005)
        mask4 = placed_bottom & stacked
        released_green = (
            ~self.is_grasping(
                self.robot_left,
                "left_robot_left_contact",
                "left_robot_right_contact",
            )
        ).to(self.device)
        released_red = (
            ~self.is_grasping(
                self.robot_right,
                "right_robot_left_contact",
                "right_robot_right_contact",
            )
        ).to(self.device)
        reward[mask4] = 8.0 + (released_green[mask4] + released_red[mask4]) / 2

        # 6) Final success gets top score
        success = self.is_success()
        reward[success] = 10.0

        return reward

    def is_robot_static(
        self, robot: Articulation, threshold: float = 1e-3
    ) -> torch.Tensor:
        """
        Check whether all actuated joints of a robot are below velocity threshold.

        Parameters:
            robot (Articulation): The robot articulation to check.
            threshold (float): Velocity threshold in rad/s.

        Returns:
            torch.Tensor: Boolean mask (N,) where True indicates static.
        """
        v = robot.data.joint_vel[:, self.joint_ids]
        return torch.all(torch.abs(v) < threshold, dim=-1)

    def is_success(self) -> torch.Tensor:
        # get cube positions
        green_p = self.cube_green.data.root_state_w[:, :3] - self.scene.env_origins
        red_p = self.cube_red.data.root_state_w[:, :3] - self.scene.env_origins

        # bottom cube on target
        dist_to_goal = torch.linalg.norm(
            green_p[:, :2] - self.target_pose[:, :2], dim=-1
        )
        on_target = dist_to_goal <= self.cfg.goal_radius

        # top cube stacked on bottom
        # DexCube side ≈0.033? m scaled by 0.8 in USD TODO check dex cube size
        half_edge = 0.8 * 0.033 / 2.0
        offset = red_p - green_p
        xy_ok = torch.linalg.norm(offset[:, :2], dim=-1) <= (2 * half_edge + 0.005)
        z_ok = torch.abs(offset[:, 2] - 2 * half_edge) <= 0.005
        stacked = xy_ok & z_ok

        # 4) both robots have released their cubes
        released_bottom = ~self.is_grasping(
            self.robot_left,
            "left_robot_left_contact",
            "left_robot_right_contact",
        )
        released_top = ~self.is_grasping(
            self.robot_right,
            "right_robot_left_contact",
            "right_robot_right_contact",
        )

        # 5) success if all conditions hold
        return on_target & stacked & released_bottom & released_top

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns done and timeout signals.

        Returns:
            done (torch.Tensor): Success mask (N,).
            timeout (torch.Tensor): Timeout mask (N,) based on episode length.
        """
        done = self.is_success()
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
        self.sample_init_cube_pose(
            self.cube_green, self.cfg.cube_green_sample_range, env_ids
        )
        self.sample_init_cube_pose(
            self.cube_red, self.cfg.cube_red_sample_range, env_ids
        )
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

    def sample_init_cube_pose(
        self,
        cube: RigidObject,
        sample_range: list[list[float]],
        env_ids: Sequence[int] | None = None,
    ):
        """
        Randomize initial cube pose within cfg-defined range.

        Parameters:
            env_ids (Sequence[int] | None): Indices to sample. All if None.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        default = cube.data.default_root_state[env_ids]
        low = torch.tensor(sample_range[0], device=self.device)
        high = torch.tensor(sample_range[1], device=self.device)
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)
        pos = rand6[:, :3] + self.scene.env_origins[env_ids]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        state = torch.cat((pos, quat), dim=-1)
        cube.write_root_pose_to_sim(state, env_ids)
        cube.write_root_velocity_to_sim(default[:, 7:], env_ids)

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
        pos = rand6[:, :3] + self.scene.env_origins[env_ids]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        self.target_pose = torch.cat((pos, quat), dim=-1)
        idxs = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        self.target_marker.visualize(pos, quat, marker_indices=idxs)
