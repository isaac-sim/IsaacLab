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
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

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

        # camera
        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

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

        # replicate and add robots and cube
        self.scene.clone_environments(copy_from_source=True)
        self.scene.articulations["robot_left"] = self.robot_left
        self.scene.articulations["robot_right"] = self.robot_right
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

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

    # TODO test
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute multi-stage dense reward.

        Returns:
            torch.Tensor: Reward tensor of shape (N,).
        """
        tcp = self.get_tcp_poses()
        pos_L = tcp[:, :3]
        pos_R = tcp[:, 7:10]
        cube_w = self.cube.data.root_state_w
        cube_pos = cube_w[:, :3] - self.scene.env_origins
        goal_pos = self.target_pose[:, :3]

        # Stage 1: reach & push
        dist_L = torch.linalg.norm(cube_pos - pos_L, dim=-1)
        reach1 = 1 - torch.tanh(5 * dist_L)
        beyond = torch.clamp(0.05 - cube_pos[:, 1], min=0.0)
        push1 = 1 - torch.tanh(5 * beyond)
        reward = 0.5 * (reach1 + push1)
        mask1 = cube_pos[:, 1] >= 0.0

        # Pre-grasp check
        f1 = self.left_finger_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        f2 = self.right_finger_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        h1, h2 = f1[:, 2], f2[:, 2]
        th = 1 - torch.tanh(5 * torch.abs(h1 - h2))
        d = torch.linalg.norm(f1 - f2, dim=-1)
        tw = 1 - torch.tanh(5 * torch.abs(d - 0.07))
        tip_reward = 0.5 * (th + tw)
        is_pre = (th > 0.5) & (tw > 0.5)

        # Stage 2: reach + tip + leave + pre-grasp
        dist_R = torch.linalg.norm(cube_pos - pos_R, dim=-1)
        reach2 = 1 - torch.tanh(5 * dist_R)
        leave = 1 - torch.tanh(5 * torch.abs(tcp[:, 1] + 0.2))
        reward[mask1] = 2.0 + reach2[mask1] + tip_reward[mask1] + leave[mask1] + 2.0 * is_pre[mask1]

        # Stage 3: move toward goal + left return
        mask2 = mask1 & is_pre
        dist_goal = torch.linalg.norm(goal_pos - pos_R, dim=-1)
        place3 = 1 - torch.tanh(5 * dist_goal)
        left_q = self.robot_left.data.joint_pos
        left_return = 1 - torch.tanh(torch.linalg.norm(left_q - self.left_init_qpos, dim=-1))
        reward[mask2] = 8.0 + (2.0 * place3 + left_return)[mask2]

        # Stage 4: intermediate near-goal bonus
        mask3 = mask2 & (dist_goal < 0.25)
        reward[mask3] = 12.0 + 2.0 * place3[mask3]

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

    def is_robot_static(self, robot: Articulation, threshold: float = 1e-3) -> torch.Tensor:
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

    # TODO test
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
        # timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        timeout = torch.zeros_like(done, dtype=torch.bool)  # TODO reactivate timeout logic
        return done, timeout

    # TODO test
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
