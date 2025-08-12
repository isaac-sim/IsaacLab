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

from .two_robot_stack_cube_env_cfg import TwoRobotStackCubeCfg


class TwoRobotStackCubeEnv(DirectRLEnv):
    """
    TwoRobotStackCube environment for stacking two cubes cooperatively
    using two Panda robot arms. The left arm picks and places the bottom cube,
    while the right arm picks and stacks the top cube onto it.
    """

    cfg: TwoRobotStackCubeCfg

    def __init__(self, cfg: TwoRobotStackCubeCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the TwoRobotStackCube environment.

        Args:
            cfg (TwoRobotStackCubeCfg): Configuration dataclass for the environment.
            render_mode (Optional[str]): Rendering mode for the environment.
            **kwargs: Additional keyword arguments forwarded to DirectRLEnv.
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
        Build and spawn scene components:
            - Ground plane
            - Two Panda robots (left and right)
            - RGB camera sensor
            - TCP and fingertip frame transformers
            - Target visualization marker
            - Cubes to be manipulated
            - Contact sensors for grasp detection
            - Dome lighting
        """
        # ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0))

        # instantiate and register robots
        self.robot_left = Articulation(self.cfg.robot_left_cfg)
        self.robot_right = Articulation(self.cfg.robot_right_cfg)
        self.scene.articulations["robot_left"] = self.robot_left
        self.scene.articulations["robot_right"] = self.robot_right

        # camera sensor
        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # cubes to stack
        self.cube_green = RigidObject(self.cfg.cube_green_cfg)
        self.scene.rigid_objects["cube_green"] = self.cube_green
        self.cube_red = RigidObject(self.cfg.cube_red_cfg)
        self.scene.rigid_objects["cube_red"] = self.cube_red

        # TCP frame transformers
        self.tcp_transformer_left = FrameTransformer(cfg=self.cfg.tcp_left_cfg)
        self.scene.sensors["tcp_left"] = self.tcp_transformer_left
        self.tcp_transformer_right = FrameTransformer(cfg=self.cfg.tcp_right_cfg)
        self.scene.sensors["tcp_right"] = self.tcp_transformer_right

        # fingertip frame transformers
        self.left_finger_transformer = FrameTransformer(cfg=self.cfg.left_finger_cfg)
        self.scene.sensors["left_finger"] = self.left_finger_transformer
        self.right_finger_transformer = FrameTransformer(cfg=self.cfg.right_finger_cfg)
        self.scene.sensors["right_finger"] = self.right_finger_transformer

        # goal region marker
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)
        self.scene.extras["target"] = self.target_marker

        # contact sensors for grasp detection
        self.scene.sensors["left_robot_left_contact"] = ContactSensor(self.cfg.sensors[1])
        self.scene.sensors["left_robot_right_contact"] = ContactSensor(self.cfg.sensors[2])
        self.scene.sensors["right_robot_left_contact"] = ContactSensor(self.cfg.sensors[3])
        self.scene.sensors["right_robot_right_contact"] = ContactSensor(self.cfg.sensors[4])

        # replicate environments and finalize scene
        self.scene.clone_environments(copy_from_source=True)

        # dome light for diffuse illumination
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Cache the incoming actions before stepping physics.

        Args:
            actions (torch.Tensor): Action tensor of shape (N,18),
                concatenated [left(0:9), right(9:18)] efforts.
        """
        self.actions = actions.clone()

    # TODO
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
        Build observations for the RL policy. Depending on `obs_mode`, returns:
            - State: concatenated vector of tcp, target, cube poses, and offsets
            - RGB: image tensor (N,3,H,W)

        Returns:
            dict: {
                "state": Tensor(N,40),  # if obs_mode="state"
                "rgb":   Tensor(N,3,128,128),  # if obs_mode includes camera
                "policy": Tensor,  # selected modality for policy
            }
        """
        tcp_poses = self.get_tcp_poses()
        target_pos = self.target_pose[:, :3]

        green_state = self.cube_green.data.root_state_w.clone()
        green_state[:, :3] -= self.scene.env_origins
        green_pos = green_state[:, :3]

        red_state = self.cube_red.data.root_state_w.clone()
        red_state[:, :3] -= self.scene.env_origins
        red_pos = red_state[:, :3]

        pos_L = tcp_poses[:, :3]
        pos_R = tcp_poses[:, 7:10]
        left_to_green = green_pos - pos_L
        right_to_red = red_pos - pos_R
        green_to_red = red_pos - green_pos

        state_obs = torch.cat(
            [
                tcp_poses,
                target_pos,
                green_state[:, :7],
                red_state[:, :7],
                left_to_green,
                right_to_red,
                green_to_red,
            ],
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
        Query and concatenate left/right TCP poses in world frame,
        shifted into the environment origins.

        Returns:
            torch.Tensor: (N,14) [pos_L(3), quat_L(4), pos_R(3), quat_R(4)]
        """
        quat_L = self.tcp_transformer_left.data.target_quat_w.squeeze(1)
        pos_L = self.tcp_transformer_left.data.target_pos_w.squeeze(1) - self.scene.env_origins
        quat_R = self.tcp_transformer_right.data.target_quat_w.squeeze(1)
        pos_R = self.tcp_transformer_right.data.target_pos_w.squeeze(1) - self.scene.env_origins
        return torch.cat([pos_L, quat_L, pos_R, quat_R], dim=1)

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
        Compute multi-stage dense reward following ManiSkill patterns:
          1) Reach & pre-grasp
          2) Place bottom cube on target
          3) Stack top cube and move right arm away
          4) Release both cubes
          5) Terminal success bonus

        Returns:
            torch.Tensor: Reward tensor of shape (N,).
        """
        N = self.num_envs
        tcp = self.get_tcp_poses()
        tcp_L, tcp_R = tcp[:, :3], tcp[:, 7:10]
        green_w = self.cube_green.data.root_state_w[:, :3] - self.scene.env_origins
        red_w = self.cube_red.data.root_state_w[:, :3] - self.scene.env_origins
        half_edge = 0.8 * self.cfg.DEX_CUBE_SIZE / 2.0
        reward = torch.zeros(N, device=self.device)

        # Stage 1: Reach and pre-grasp
        d_L = torch.linalg.norm(tcp_L - green_w, dim=-1)
        push_target = red_w + torch.tensor([0, half_edge + 0.005, 0], device=self.device)
        d_R = torch.linalg.norm(tcp_R - push_target, dim=-1)
        reach_reward = (1 - torch.tanh(5 * d_L) + 1 - torch.tanh(5 * d_R)) / 2
        graspL = self.is_grasping(self.robot_left, "left_robot_left_contact", "left_robot_right_contact").to(
            self.device
        )
        reward[:] = (reach_reward + graspL) / 2

        # Stage 2: Place bottom cube on target
        red_on_tgt = torch.linalg.norm(red_w[:, :2] - self.target_pose[:, :2], dim=-1) < self.cfg.GOAL_RADIUS
        place_reward = 1 - torch.tanh(5 * torch.linalg.norm(red_w[:, :2] - self.target_pose[:, :2], dim=-1))
        grasped_green_cube_mask = graspL.bool()
        reward[grasped_green_cube_mask] = (
            2.0 + (place_reward[grasped_green_cube_mask] + graspL[grasped_green_cube_mask]) / 2
        )

        # Stage 3: Stack top cube and move right arm away
        placed_bottom_cube_and_grasped_green = red_on_tgt & graspL.bool()

        stack_target = torch.cat([red_w[:, :2], (red_w[:, 2] + 2 * half_edge).unsqueeze(-1)], dim=1)
        place_top_reward = 1 - torch.tanh(5 * torch.linalg.norm(stack_target - green_w, dim=-1))
        move_right_robot_away_reward = 1 - torch.tanh(5 * torch.abs(tcp_R[:, 1] - 0.2))
        reward[placed_bottom_cube_and_grasped_green] = 4.0 + (
            2 * place_top_reward[placed_bottom_cube_and_grasped_green]
            + move_right_robot_away_reward[placed_bottom_cube_and_grasped_green]
        )

        # Stage 4: Release both cubes
        stacked = (torch.linalg.norm(green_w[:, :2] - red_w[:, :2], dim=-1) <= (2 * half_edge + 0.005)) & (
            torch.abs(green_w[:, 2] - (red_w[:, 2] + 2 * half_edge)) <= 0.005
        )
        both_cubes_stacked_mask = red_on_tgt & stacked
        rel_t = (~self.is_grasping(self.robot_left, "left_robot_left_contact", "left_robot_right_contact")).to(
            self.device
        )
        rel_b = (
            ~self.is_grasping(
                self.robot_right,
                "right_robot_left_contact",
                "right_robot_right_contact",
            )
        ).to(self.device)
        reward[both_cubes_stacked_mask] = 8.0 + (rel_b[both_cubes_stacked_mask] + rel_t[both_cubes_stacked_mask]) / 2

        # Stage 5: Terminal success bonus
        success = self.is_success()
        reward[success] = 10.0
        return reward

    def is_success(self) -> torch.Tensor:
        """
        Determine success when:
          1) Bottom cube is within goal radius of target
          2) Top cube is properly stacked above bottom
          3) Both robots have released their cubes

        Returns:
            torch.Tensor: Bool mask (N,) of success.
        """
        green_p = self.cube_green.data.root_state_w[:, :3] - self.scene.env_origins
        red_p = self.cube_red.data.root_state_w[:, :3] - self.scene.env_origins
        on_tgt = torch.linalg.norm(red_p[:, :2] - self.target_pose[:, :2], dim=-1) <= self.cfg.GOAL_RADIUS
        half_edge = 0.8 * self.cfg.DEX_CUBE_SIZE / 2.0
        offset = green_p - red_p
        xy_ok = torch.linalg.norm(offset[:, :2], dim=-1) <= (2 * half_edge + 0.005)
        z_ok = torch.abs(offset[:, 2] - (2 * half_edge)) <= 0.005
        stacked = xy_ok & z_ok
        rel_t = ~self.is_grasping(self.robot_left, "left_robot_left_contact", "left_robot_right_contact")
        rel_b = ~self.is_grasping(self.robot_right, "right_robot_left_contact", "right_robot_right_contact")
        return on_tgt & stacked & rel_b & rel_t

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute done and timeout signals for the environment.

        Returns:
            done (torch.Tensor): Success mask (N,).
            timeout (torch.Tensor): Timeout mask if max length reached (N,).
        """
        done = self.is_success()
        # timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        timeout = torch.zeros_like(done, dtype=torch.bool)  # TODO remove (only for testing)
        return done, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specified environments: robots, cubes, and target.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        super()._reset_idx(env_ids)
        self.reset_robot(env_ids)
        self.sample_init_cube_pose(self.cube_green, self.cfg.cube_green_sample_range, env_ids)
        self.sample_init_cube_pose(self.cube_red, self.cfg.cube_red_sample_range, env_ids)
        self.sample_target_pose(env_ids)

    def reset_robot(self, env_ids: Sequence[int] | None = None):
        """
        Reset both robots to their default joint and root states, and cache left init qpos.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
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
        self.left_init_qpos = self.robot_left.data.joint_pos.clone()

    def sample_init_cube_pose(
        self,
        cube: RigidObject,
        sample_range: list[list[float]],
        env_ids: Sequence[int] | None = None,
    ):
        """
        Randomize the initial pose of a cube within specified ranges.

        Args:
            cube (RigidObject): The cube object to randomize.
            sample_range (List[List[float]]): Min/max sampling ranges for pos+euler.
            env_ids (Sequence[int] | None): Indices to reset. If None, all.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        default = cube.data.default_root_state[env_ids]
        low = torch.tensor(sample_range[0], device=self.device)
        high = torch.tensor(sample_range[1], device=self.device)
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)
        pos = rand6[:, :3] + self.scene.env_origins[env_ids]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        state = torch.cat([pos, quat], dim=-1)
        cube.write_root_pose_to_sim(state[env_ids], env_ids)
        cube.write_root_velocity_to_sim(default[env_ids, 7:], env_ids)

    def sample_target_pose(self, env_ids: Sequence[int] | None = None):
        """
        Randomize the goal target's pose within configured ranges and visualize marker.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, all.
        """
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        low = torch.tensor(self.cfg.target_sample_range[0], device=self.device)
        high = torch.tensor(self.cfg.target_sample_range[1], device=self.device)
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)
        pos = rand6[:, :3]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])
        self.target_pose = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.target_pose[env_ids] = torch.cat([pos, quat], dim=-1)[env_ids]
        idxs = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        self.target_marker.visualize(pos + self.scene.env_origins, quat, marker_indices=idxs)
