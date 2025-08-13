# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import glob
import os
import random
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, ContactSensor, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import normalize, quat_from_euler_xyz, sample_uniform, transform_points

from .peg_insertion_side_env_cfg import PegInsertionSideEnvCfg


class PegInsertionSideEnv(DirectRLEnv):

    cfg: PegInsertionSideEnvCfg

    def compute_offsets(self):
        """
        From each peg USD filename (peg_L{L}_R{r}.usda), parse L and r,
        and build:
        - self.peg_head_offset:  (N,3) tensor = [L, 0, 0]
        - self.box_hole_offset:  (N,3) tensor = [0, c_y, c_z]
        - self.box_hole_radius:  (N,)   tensor = r + clearance
        Here c_y, c_z are your hole-center offsets if you sampled them;
        if not, set them to zero.
        """

        Ls, rs = [], []
        # if you sampled centers, use those; otherwise centers = 0
        Cs = [(0.0, 0.0) for _ in self.pegs_list]

        for peg_path, (c_y, c_z) in zip(self.pegs_list, Cs):
            name = os.path.splitext(os.path.basename(peg_path))[0]
            # name = "peg_L0.085_R0.015"
            key = name.replace("peg_", "")  # "L0.085_R0.015"
            L, r = key.split("_")
            L = float(L[1:])
            r = float(r[1:])
            Ls.append(L)
            rs.append(r)

        device = self.device
        Ls = torch.tensor(Ls, device=device)
        rs = torch.tensor(rs, device=device)
        Cs = torch.tensor(Cs, device=device)

        self.peg_head_offset = torch.stack([Ls, torch.zeros_like(Ls), torch.zeros_like(Ls)], dim=1)  # (N,3)
        self.box_hole_offset = torch.stack([torch.zeros_like(Ls), Cs[:, 0], Cs[:, 1]], dim=1)  # (N,3)
        self.box_hole_radius = rs + 0.003  # clearance

    def __init__(self, cfg: PegInsertionSideEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        random.seed(self.cfg.seed)

        self.joint_ids, _ = self.robot.find_joints("panda_joint.*|panda_finger_joint.*")

        # Find relevant link indices for runtime TCP computation
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

    def _sample_asset_pairs(self):
        """
        Scan cfg.asset_dir for peg_*.usda and box_*.usda, match on the L/R key,
        shuffle, and fill two parallel lists: self.pegs and self.boxes.
        """
        asset_dir = self.cfg.asset_dir
        peg_files = glob.glob(os.path.join(asset_dir, "peg_*.usda"))
        box_files = glob.glob(os.path.join(asset_dir, "box_*.usda"))

        def parse_key(path):
            # peg_L0.085_R0.015.usda → L0.085_R0.015
            name = os.path.splitext(os.path.basename(path))[0]
            return "_".join(name.split("_")[1:])  # drop the "peg"/"box" prefix

        peg_map = {parse_key(p): p for p in peg_files}
        box_map = {parse_key(b): b for b in box_files}

        # find the intersection of keys we have both peg & box for
        common_keys = list(set(peg_map.keys()) & set(box_map.keys()))
        random.shuffle(common_keys)

        # take the first num_envs keys
        selected = common_keys[: self.num_envs]

        # build two parallel lists
        self.pegs_list = [peg_map[k] for k in selected]
        self.boxes_list = [box_map[k] for k in selected]

        self.compute_offsets()

    def _load_default_scene(self):

        # Creating the default scene
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0))

        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        self.tcp_transformer = FrameTransformer(cfg=self.cfg.tcp_cfg)
        self.scene.sensors["tcp"] = self.tcp_transformer

        self.scene.sensors["left_contact"] = ContactSensor(cfg=self.cfg.sensors[1])
        self.scene.sensors["right_contact"] = ContactSensor(cfg=self.cfg.sensors[2])

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_scene(self):
        self._load_default_scene()

        self._sample_asset_pairs()

        self.peg_cfg: RigidObjectCfg = self.cfg.get_multi_cfg(self.pegs_list, "/World/envs/env_.*/Peg", False)
        self.peg: RigidObject = RigidObject(self.peg_cfg)
        self.scene.rigid_objects["Peg"] = self.peg

        self.box_cfg: RigidObjectCfg = self.cfg.get_multi_cfg(self.boxes_list, "/World/envs/env_.*/Box", False)
        self.box: RigidObject = RigidObject(self.box_cfg)
        self.scene.rigid_objects["Box"] = self.box

        # Filtering collisions for optimization of collisions between environment instances
        self.scene.filter_collisions([
            "/World/ground",
        ])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.cfg.robot_controller == "task_space":
            self.robot.set_joint_position_target(self.actions, joint_ids=self.joint_ids)
        else:
            self.robot.set_joint_effort_target(self.actions, joint_ids=self.joint_ids)

    # TODO implement the observation function
    def _get_observations(self) -> dict:

        peg_pose = self.peg.data.root_state_w  # (N,7) [x,y,z, qx,qy,qz,qw]
        peg_pose[:, :3] -= self.scene.env_origins  # subtract env origins

        clearance = 0.003
        L = self.peg_head_offset[:, 0]  # (N,)
        r = self.box_hole_radius - clearance  # (N,)
        peg_half_size = torch.stack((L, r, r), dim=1)

        box_p_w = self.box.data.root_pos_w  # (N,3)
        box_q = self.box.data.root_quat_w  # (N,4)
        # 2) box root in local frame
        box_p_local = box_p_w - self.scene.env_origins  # (N,3)
        hole_p_local = transform_points(
            self.box_hole_offset.unsqueeze(1),  # (N,1,3) local offset
            pos=box_p_local,  # treat as box origin in local frame
            quat=box_q,  # box orientation
        ).squeeze(
            1
        )  # (N,3)
        # 4) pack with box orientation (same quaternion)
        box_hole_pose = torch.cat((hole_p_local, box_q), dim=1)  # (N,7)

        state_obs = torch.cat(
            (
                self.get_tcp_poses(),
                peg_pose,
                peg_half_size,
                box_hole_pose,
                self.box_hole_radius.unsqueeze(1),
                self.robot.data.joint_pos[:, self.joint_ids],  # (N, J)
                self.robot.data.joint_vel[:, self.joint_ids],  # (N, J)
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
        Returns an (N,7) tensor [x, y, z, qx, qy, qz, qw]
        """
        pos = self.tcp_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        quat = self.tcp_transformer.data.target_quat_w.squeeze(1)
        return torch.cat((pos, quat), dim=1)

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

    # TODO test in simulation
    def _get_rewards(self) -> torch.Tensor:
        """
        Dense reward mirroring ManiSkill3’s compute_dense_reward:
         1) Reach (1 − tanh(4·dist_tcp→peg_tail))
         2) +1 for clean grasp
         3) + pre-insertion alignment term (× grasp mask)
         4) + insertion progress term (× pre-insertion mask)
         5) +10 on success
        """
        # 1) World-space poses of peg and box
        peg_q = self.peg.data.root_quat_w  # (N,4)
        peg_p = self.peg.data.root_pos_w - self.scene.env_origins  # (N,3)
        box_q = self.box.data.root_quat_w
        box_p = self.box.data.root_pos_w - self.scene.env_origins

        # 2) TCP world-position
        tcp = self.get_tcp_poses()  # (N,7) [x,y,z, qx,qy,qz,qw]
        tcp_p = tcp[:, :3]  # (N,3)

        # 3) Reach & grasp
        # peg tail = head_offset * (−1,0,0)
        tail_offset = torch.cat([-self.peg_head_offset[:, :1], self.peg_head_offset[:, 1:]], dim=1)  # (N,3)
        tail_world = transform_points(points=tail_offset.unsqueeze(1), pos=peg_p, quat=peg_q).squeeze(1)  # (N,3)

        d_tcp_peg = torch.linalg.norm(tcp_p - tail_world, dim=1)
        reach_rew = 1.0 - torch.tanh(4.0 * d_tcp_peg)

        is_grasped = self.is_grasping(self.robot, "left_contact", "right_contact")
        reward = reach_rew + is_grasped.to(dtype=reach_rew.dtype)

        # 4) Pre-insertion alignment
        head_world = transform_points(points=self.peg_head_offset.unsqueeze(1), pos=peg_p, quat=peg_q).squeeze(
            1
        )  # (N,3)
        hole_world = transform_points(points=self.box_hole_offset.unsqueeze(1), pos=box_p, quat=box_q).squeeze(1)

        # rotate into hole frame using inverse box quaternion
        inv_box_q = torch.cat([-box_q[:, :3], box_q[:, 3:4]], dim=1)  # (N,4)
        rel_head = head_world - hole_world
        rel_base = peg_p - hole_world

        head_local = transform_points(points=rel_head.unsqueeze(1), quat=inv_box_q).squeeze(1)  # (N,3)
        base_local = transform_points(points=rel_base.unsqueeze(1), quat=inv_box_q).squeeze(1)

        head_yz = torch.linalg.norm(head_local[:, 1:], dim=1)
        base_yz = torch.linalg.norm(base_local[:, 1:], dim=1)
        pre_ins = 3.0 * (1.0 - torch.tanh(0.5 * (head_yz + base_yz) + 4.5 * torch.maximum(head_yz, base_yz)))
        reward = reward + pre_ins * is_grasped.to(reward.dtype)
        pre_inserted = (head_yz < 0.01) & (base_yz < 0.01)

        # 5) Insertion depth
        rel_insert = transform_points(points=rel_head.unsqueeze(1), quat=inv_box_q).squeeze(1)
        insertion_rew = 5.0 * (1.0 - torch.tanh(5.0 * torch.linalg.norm(rel_insert, dim=1)))
        reward = (reward + insertion_rew) * (is_grasped & pre_inserted).to(reward.dtype)

        # 6) Success bonus
        success = self.is_success()
        reward[success] = 10.0

        return reward

    # TODO test in simulation
    def is_success(self) -> torch.Tensor:
        """
        Returns a (N,) boolean mask: True where the peg head is inserted into the box hole.
        Uses isaaclab.utils.math.transform_points under the hood.
        """

        peg_q = self.peg.data.root_quat_w
        peg_p = self.peg.data.root_pos_w - self.scene.env_origins
        box_q = self.box.data.root_quat_w
        box_p = self.box.data.root_pos_w - self.scene.env_origins

        # 3) your precomputed local offsets are (N,3).
        #    transform_points expects (N,P,3), so make P=1
        peg_head_local = self.peg_head_offset.unsqueeze(1)  # (N,1,3)
        box_hole_local = self.box_hole_offset.unsqueeze(1)  # (N,1,3)

        # 4) map each point into world frame: p_world = R_world * p_local + t_world
        head_world = transform_points(peg_head_local, pos=peg_p, quat=peg_q).squeeze(1)
        hole_world = transform_points(box_hole_local, pos=box_p, quat=box_q).squeeze(1)

        # 5) relative vector from hole to head
        rel = head_world - hole_world
        x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]

        # 6) ManiSkill’s insertion test
        x_ok = x >= -0.015
        y_ok = y.abs() <= self.box_hole_radius
        z_ok = z.abs() <= self.box_hole_radius

        return x_ok & y_ok & z_ok

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        done = self.is_success()
        if self.cfg.robot_controller == "task_space":
            timeout = torch.zeros_like(done, dtype=torch.bool)
        else:
            timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        return done, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specified environments: robot, peg, and box.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.reset_robot(env_ids)
        self.sample_init_peg_pose(env_ids)
        self.sample_init_box_pose(env_ids)

    def reset_robot(self, env_ids: Sequence[int] | None = None):
        """
        Reset the robot to its default joint and root states.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Defaults
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids].clone()

        # Place robot in its environment (world frame)
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def sample_init_peg_pose(self, env_ids: Sequence[int] | None = None):
        """
        Randomize the initial pose of the peg (position XY, yaw). Z is set to the peg half-length L.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        device = self.device
        b = len(env_ids)

        # 1) Sample local pose (x, y, z, r, p, yaw) from configured ranges
        low = torch.tensor(self.cfg.peg_sample_range[0], device=device, dtype=torch.float32)
        high = torch.tensor(self.cfg.peg_sample_range[1], device=device, dtype=torch.float32)
        rand6 = sample_uniform(low, high, (b, 6), device=device)

        # 2) Force Z to half-length L for each env (vectorized)
        #    L is per-env (taken from peg_head_offset)
        L = self.peg_head_offset[env_ids, 0]
        pos_local = rand6[:, :3]
        pos_local[:, 2] = L

        # 3) Orientation: roll/pitch from rand6 (are 0 by config), yaw from rand6[:, 5]
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])

        # 4) Convert to world coordinates
        pos_world = pos_local + self.scene.env_origins[env_ids]
        state = torch.cat([pos_world, quat], dim=-1)

        # 5) Velocities: use defaults (usually zero) for cleanliness
        default = self.peg.data.default_root_state[env_ids]
        vel = default[:, 7:]

        # 6) Write to sim
        self.peg.write_root_pose_to_sim(state, env_ids)
        self.peg.write_root_velocity_to_sim(vel, env_ids)

    def sample_init_box_pose(self, env_ids: Sequence[int] | None = None):
        """
        Randomize the initial pose of the box (position XY, yaw). Z is set to the peg half-length L.

        Args:
            env_ids (Sequence[int] | None): Indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        device = self.device
        b = len(env_ids)

        # 1) Sample local pose (x, y, z, r, p, yaw) from configured ranges
        low = torch.tensor(self.cfg.box_sample_range[0], device=device, dtype=torch.float32)
        high = torch.tensor(self.cfg.box_sample_range[1], device=device, dtype=torch.float32)
        rand6 = sample_uniform(low, high, (b, 6), device=device)

        # 2) Force Z to half-length L (same L vector as peg)
        L = self.peg_head_offset[env_ids, 0]
        pos_local = rand6[:, :3]
        pos_local[:, 2] = L

        # 3) Orientation
        quat = quat_from_euler_xyz(rand6[:, 3], rand6[:, 4], rand6[:, 5])

        # 4) Convert to world coordinates
        pos_world = pos_local + self.scene.env_origins[env_ids]
        state = torch.cat([pos_world, quat], dim=-1)

        # 5) Velocities: defaults
        default = self.box.data.default_root_state[env_ids]
        vel = default[:, 7:]

        # 6) Write to sim
        self.box.write_root_pose_to_sim(state, env_ids)
        self.box.write_root_velocity_to_sim(vel, env_ids)
