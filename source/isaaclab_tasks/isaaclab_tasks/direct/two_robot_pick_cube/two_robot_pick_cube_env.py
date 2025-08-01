# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform
from isaaclab.sensors import FrameTransformer
from isaaclab.markers import VisualizationMarkers


from .two_robot_pick_cube_env_cfg import TwoRobotPickCubeCfg


class TwoRobotPickCubeEnv(DirectRLEnv):

    cfg: TwoRobotPickCubeCfg

    def __init__(
        self, cfg: TwoRobotPickCubeCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        random.seed(self.cfg.seed)

        self.joint_ids, _ = self.robot_left.find_joints(
            "panda_joint.*|panda_finger_joint.*"
        )

    def _setup_scene(self):
        # Creating the default scene
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0)
        )

        self.robot_left = Articulation(self.cfg.robot_left_cfg)
        self.robot_right = Articulation(self.cfg.robot_right_cfg)

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # Create the transformer and attach it to the scene
        self.tcp_transformer_left = FrameTransformer(cfg=self.cfg.tcp_left_cfg)
        self.scene.sensors["tcp_left"] = self.tcp_transformer_left
        self.tcp_transformer_right = FrameTransformer(cfg=self.cfg.tcp_right_cfg)
        self.scene.sensors["tcp_right"] = self.tcp_transformer_right

        self.left_finger_transformer = FrameTransformer(cfg=self.cfg.left_finger_cfg)
        self.scene.sensors["left_finger"] = self.left_finger_transformer
        self.right_finger_transformer = FrameTransformer(cfg=self.cfg.right_finger_cfg)
        self.scene.sensors["right_finger"] = self.right_finger_transformer

        # create the marker object
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)
        self.scene.extras["target"] = self.target_marker

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot_left"] = self.robot_left
        self.scene.articulations["robot_right"] = self.robot_right

        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot_left.set_joint_effort_target(
            self.actions[:, 0:9], joint_ids=self.joint_ids
        )
        self.robot_right.set_joint_effort_target(
            self.actions[:, 9:18], joint_ids=self.joint_ids
        )

    def _get_observations(self) -> dict:

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

        rgb = self.scene.sensors["camera"].data.output["rgb"]
        pixels = (rgb.to(torch.float32) / 255.0).clone()  # normalize to [0,1]

        # return according to obs_mode
        if self.cfg.obs_mode == "state":
            return {"policy": state_obs}
        else:
            return {"policy": pixels}

    def get_tcp_poses(self) -> torch.Tensor:
        """
        Returns an (N,14) tensor with the TCP poses of the left and right robot.
        The first 7 values are the left robot's TCP pose, and the last 7
        values are the right robot's TCP pose.
        Each pose is represented as (x, y, z, qx, qy, qz, qw).
        """
        quat_left = self.tcp_transformer_left.data.target_quat_w.squeeze(1)
        pos_left = (
            self.tcp_transformer_left.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )  # (N,3)
        quat_right = self.tcp_transformer_right.data.target_quat_w.squeeze(1)
        pos_right = (
            self.tcp_transformer_right.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )

        return torch.cat((pos_left, quat_left, pos_right, quat_right), dim=1)

    # TODO test in simulation
    def _get_rewards(self) -> torch.Tensor:
        # common quantities
        tcp = self.get_tcp_poses()
        pos_L = tcp[:, :3]
        pos_R = tcp[:, 7:10]

        cube_w = self.cube.data.root_state_w
        cube_pos = cube_w[:, :3] - self.scene.env_origins
        goal_pos = self.target_pose[:, :3]

        # Stage 1: reach & push to other side
        dist_L = torch.linalg.norm(cube_pos - pos_L, dim=-1)
        reach1 = 1 - torch.tanh(5 * dist_L)
        beyond = torch.clamp(0.05 - cube_pos[:, 1], min=0.0)
        push1 = 1 - torch.tanh(5 * beyond)
        reward = 0.5 * (reach1 + push1)
        mask1 = cube_pos[:, 1] >= 0.0

        # Pre-grasp check (exact ManiSkill3)
        finger1_pos = (
            self.left_finger_transformer.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )
        finger2_pos = (
            self.right_finger_transformer.data.target_pos_w.squeeze(1)
            - self.scene.env_origins
        )
        h1 = finger1_pos[:, 2]
        h2 = finger2_pos[:, 2]
        tip_height_reward = 1 - torch.tanh(5 * torch.abs(h1 - h2))
        d = torch.linalg.norm(finger1_pos - finger2_pos, dim=-1)
        tip_width_reward = 1 - torch.tanh(5 * torch.abs(d - 0.07))
        tip_reward = 0.5 * (tip_height_reward + tip_width_reward)
        is_pre_grasp = (tip_height_reward > 0.5) & (tip_width_reward > 0.5)

        # Stage 2: reach + tip + left-arm-leave + pre-grasp bonus
        dist_R = torch.linalg.norm(cube_pos - pos_R, dim=-1)
        reach2 = 1 - torch.tanh(5 * dist_R)
        left_y = tcp[:, 1]
        left_leave_reward = 1 - torch.tanh(5 * torch.abs(left_y + 0.2))
        reward[mask1] = (
            2.0
            + reach2[mask1]
            + tip_reward[mask1]
            + left_leave_reward[mask1]
            + 2.0 * is_pre_grasp[mask1]
        )

        # Stage 3: bring cube toward goal + left-arm-return
        mask2 = mask1 & is_pre_grasp
        dist_goal = torch.linalg.norm(goal_pos - pos_R, dim=-1)
        place3 = 1 - torch.tanh(5 * dist_goal)
        current_left_qpos = self.robot_left.data.joint_pos
        left_return_reward = 1 - torch.tanh(
            torch.linalg.norm(current_left_qpos - self.left_init_qpos, dim=-1)
        )
        reward[mask2] = 8.0 + (2.0 * place3 + left_return_reward)[mask2]

        # Stage 4: object near goal (<0.25m) intermediate bonus
        mask3 = mask2 & (dist_goal < 0.25)
        reward[mask3] = 12.0 + 2.0 * place3[mask3]

        # Stage 5: placed + static-arms bonus
        is_obj_placed = (
            torch.linalg.norm(goal_pos - cube_pos, dim=-1)
            <= self.cfg.success_distance_threshold
        )
        right_vel = self.robot_right.data.joint_vel[:, :-2]
        left_vel = self.robot_left.data.joint_vel[:, :-2]
        right_static = 1 - torch.tanh(5 * torch.linalg.norm(right_vel, dim=-1))
        left_static = 1 - torch.tanh(5 * torch.linalg.norm(left_vel, dim=-1))
        static_reward = 0.5 * (right_static + left_static)
        reward[is_obj_placed] = 19.0 + static_reward[is_obj_placed]

        # Final success overwrite
        success = self.is_success()
        reward[success] = 21.0

        return reward

    # TODO test and find out whi maniskill has a weired implementation
    def is_robot_static(self, robot: Articulation, threshold=1e-3) -> torch.Tensor:
        """
        Checks if the robots are static by comparing their joint velocities to a threshold.
        Returns a boolean tensor indicating whether each robot is static.
        """
        joint_vel = robot.data.joint_vel[:, self.joint_ids]

        return torch.all(torch.abs(joint_vel) < threshold, dim=-1)

    # TODO test in simulation
    def is_success(self) -> torch.Tensor:

        cube_pos = self.cube.data.root_state_w[:, :3] + self.scene.env_origins
        target_pos = self.target_pose[:, :3]

        distance = torch.norm(cube_pos - target_pos, dim=-1)

        return torch.logical_and(
            distance < self.cfg.success_distance_threshold,
            self.is_robot_static(self.robot_right),
        )

    # TODO implement the done condition
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # standard timeout
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        done = self.is_success()
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_left._ALL_INDICES
        super()._reset_idx(env_ids)

        self.reset_robot(env_ids)
        self.sample_init_cube_pose(env_ids)
        self.sample_target_pose(env_ids)

    def reset_robot(self, env_ids: Sequence[int] | None = None):
        joint_pos = self.robot_left.data.default_joint_pos[env_ids]
        joint_vel = self.robot_left.data.default_joint_vel[env_ids]

        default_root_state_left = self.robot_left.data.default_root_state[env_ids]
        default_root_state_left[:, :3] += self.scene.env_origins[env_ids]

        self.robot_left.write_root_pose_to_sim(default_root_state_left[:, :7], env_ids)
        self.robot_left.write_root_velocity_to_sim(
            default_root_state_left[:, 7:], env_ids
        )
        self.robot_left.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state_right = self.robot_right.data.default_root_state[env_ids]
        default_root_state_right[:, :3] += self.scene.env_origins[env_ids]

        self.robot_right.write_root_pose_to_sim(
            default_root_state_right[:, :7], env_ids
        )
        self.robot_right.write_root_velocity_to_sim(
            default_root_state_right[:, 7:], env_ids
        )
        self.robot_right.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.left_init_qpos = self.robot_left.data.joint_pos.clone()

    def sample_init_cube_pose(self, env_ids: Sequence[int] | None = None):
        defaul_cube_root_state = self.cube.data.default_root_state[env_ids]

        low = torch.tensor(
            self.cfg.cube_sample_range[0],
            device=self.device,
        )
        high = torch.tensor(
            self.cfg.cube_sample_range[1],
            device=self.device,
        )
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)

        pos = rand6[:, 0:3] + self.scene.env_origins[env_ids]
        euler = rand6[:, 3:]
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        cube_state = torch.cat((pos, quat), dim=-1)

        self.cube.write_root_pose_to_sim(cube_state, env_ids)
        self.cube.write_root_velocity_to_sim(defaul_cube_root_state[:, 7:], env_ids)

    def sample_target_pose(self, env_ids: Sequence[int] | None = None):
        """
        Samples a target pose for the cube within the specified range.
        """
        low = torch.tensor(
            self.cfg.target_sample_range[0],
            device=self.device,
        )
        high = torch.tensor(
            self.cfg.target_sample_range[1],
            device=self.device,
        )
        rand6 = sample_uniform(low, high, (self.num_envs, 6), device=self.device)

        pos = rand6[:, 0:3] + self.scene.env_origins[env_ids]
        euler = rand6[:, 3:]
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

        self.target_pose = torch.cat((pos, quat), dim=-1)

        idxs = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        self.target_marker.visualize(pos, quat, marker_indices=idxs)
