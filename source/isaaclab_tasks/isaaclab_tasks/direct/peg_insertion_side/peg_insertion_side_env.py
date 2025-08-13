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
import math
import os
import random
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import normalize, quat_from_euler_xyz, transform_points

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

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

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

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        tcp_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link7",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1),
                    ),
                ),
            ],
        )

        # Create the transformer and attach it to the scene
        self.tcp_transformer = FrameTransformer(cfg=tcp_cfg)
        self.scene.sensors["tcp"] = self.tcp_transformer

        # Contact sensor on the left fingertip, only reporting forces against the peg TODO move to cfg
        left_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Peg"],  # only peg contacts
        )
        self.scene.sensors["left_contact"] = ContactSensor(cfg=left_contact_cfg)

        # Contact sensor on the right fingertip
        right_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Peg"],
        )
        self.scene.sensors["right_contact"] = ContactSensor(cfg=right_contact_cfg)

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
        quat = self.tcp_transformer.data.target_quat_w.squeeze(1)  # (N,4)
        pos = self.tcp_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins  # (N,3)
        return torch.cat((pos, quat), dim=1)  # now (N,7)

    # TODO test in simulation and DEBUG See stack cube task
    def is_grasping(
        self,
        min_force: float = 0.5,
        max_angle: float = 85.0,
    ) -> torch.Tensor:
        """
        Returns an (N,) bool mask: True where both fingers are
        pressing on the peg with ≥ min_force N, and within max_angle°
        of their local opening axis.

        Requires you’ve set up two ContactSensors in _setup_scene:
          self.scene.sensors["left_contact"]
          self.scene.sensors["right_contact"]
        each filtered to only see the peg’s collisions.
        """

        # 1) read net world-frame contact forces: (N,1,3) → (N,3)
        l_forces = self.scene.sensors["left_contact"].data.net_forces_w.squeeze(1)
        r_forces = self.scene.sensors["right_contact"].data.net_forces_w.squeeze(1)

        # 2) force magnitudes
        l_mag = torch.norm(l_forces, dim=1)
        r_mag = torch.norm(r_forces, dim=1)

        # 3) define local “opening” axis = +Y in fingertip frame, broadcast to (N,1,3)
        N = l_forces.shape[0]
        axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 1, 3)
        axes = axis_local.expand(N, 1, 3)

        # 4) get each fingertip’s world-pose and rotate +Y into world
        pos = self.robot.data.body_pos_w  # (N, B, 3)
        quat = self.robot.data.body_quat_w  # (N, B, 4)

        l_dir = transform_points(
            points=axes,
            pos=pos[:, self.left_finger_link_idx],
            quat=quat[:, self.left_finger_link_idx],
        ).squeeze(1)

        r_dir = transform_points(
            points=axes,
            pos=pos[:, self.right_finger_link_idx],
            quat=quat[:, self.right_finger_link_idx],
        ).squeeze(1)
        # flip the right finger so +dir is the “opening” direction
        r_dir = -r_dir

        # 5) normalize vectors to compute angles
        l_dir_u = normalize(l_dir)
        r_dir_u = normalize(r_dir)
        l_f_u = normalize(l_forces)
        r_f_u = normalize(r_forces)

        # 6) compute angle = arccos(dot), in degrees
        l_cos = (l_dir_u * l_f_u).sum(dim=1).clamp(-1.0, 1.0)
        r_cos = (r_dir_u * r_f_u).sum(dim=1).clamp(-1.0, 1.0)
        l_ang = torch.acos(l_cos) * (180.0 / torch.pi)
        r_ang = torch.acos(r_cos) * (180.0 / torch.pi)

        # 7) check both magnitude & angle thresholds
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

        is_grasped = self.is_grasping(min_force=0.5, max_angle=20)
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
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Resetting the robot
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- vectorized sampling for peg & box ---
        b = len(env_ids)
        device = self.device

        # peg half-lengths L (b,)
        L_all = self.peg_head_offset[:, 0]  # (N,)
        L = L_all[env_ids]  # (b,)

        # zero velocity for both bodies
        zero_vel = torch.zeros((b, 6), device=device)

        # 1) peg positions in local frame: XY∼U([-0.1,-0.3],[0.1,0.0]), Z=L
        low_xy = torch.tensor([-0.1, -0.3], device=device)
        high_xy = torch.tensor([0.1, 0.0], device=device)
        peg_xy = torch.empty((b, 2), device=device).uniform_(0, 1)
        peg_xy = low_xy + (high_xy - low_xy) * peg_xy
        peg_pos_local = torch.cat([peg_xy, L.unsqueeze(1)], dim=1)
        peg_pos_local[:, 0] += 0.4  # (b,3)
        peg_pos = peg_pos_local + self.scene.env_origins[env_ids]  # world

        # 3) box positions: XY∼U([-0.05,0.2],[0.05,0.4]), Z=L
        low_bxy = torch.tensor([-0.05, 0.2], device=device)
        high_bxy = torch.tensor([0.05, 0.4], device=device)
        box_xy = torch.empty((b, 2), device=device).uniform_(0, 1)
        box_xy = low_bxy + (high_bxy - low_bxy) * box_xy
        box_pos_local = torch.cat([box_xy, L.unsqueeze(1)], dim=1)  # (b,3)
        box_pos_local[:, 0] += 0.4  # shift X to [0.2, 0.4]
        box_pos = box_pos_local + self.scene.env_origins[env_ids]  # world

        # sample yaw angles
        # peg yaw θ ∼ Uniform(min_a, max_a)
        min_a = math.pi / 2 - math.pi / 3  # = π/6
        max_a = math.pi / 2 + math.pi / 3  # = 5π/6

        # box yaw θ ∼ Uniform(min_b, max_b)
        min_b = math.pi / 2 - math.pi / 8  # = 3π/8
        max_b = math.pi / 2 + math.pi / 8  # = 5π/8

        angles = torch.empty(b, device=device).uniform_(min_a, max_a)  # peg yaw
        angles_b = torch.empty(b, device=device).uniform_(min_b, max_b)  # box yaw

        # convert to quaternion (x, y, z, w) using IsaacLab helper
        peg_quat = quat_from_euler_xyz(torch.zeros(b, device=device), torch.zeros(b, device=device), angles)  # -> (b,4)
        box_quat = quat_from_euler_xyz(
            torch.zeros(b, device=device), torch.zeros(b, device=device), angles_b
        )  # -> (b,4)

        # 5) write randomized states into sim
        self.peg.write_root_pose_to_sim(torch.cat([peg_pos, peg_quat], dim=1), env_ids)
        self.peg.write_root_velocity_to_sim(zero_vel, env_ids)

        self.box.write_root_pose_to_sim(torch.cat([box_pos, box_quat], dim=1), env_ids)
        self.box.write_root_velocity_to_sim(zero_vel, env_ids)
