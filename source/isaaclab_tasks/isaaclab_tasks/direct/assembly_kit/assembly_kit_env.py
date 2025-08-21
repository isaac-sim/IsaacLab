# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AssemblyKitEnv: direct‐RL environment for the Franka assembly‐kit benchmark."""

from __future__ import annotations

import math
import random
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from .assembly_kit_env_cfg import AssemblyKitEnvCfg, get_kit_cfg, get_model_cfg


class AssemblyKitEnv(DirectRLEnv):
    """Direct-RL environment for the Assembly-Kit task in Isaac Lab.

    This class handles scene setup observation & reward computation,
    and environment resets for the Franka assembly-kit benchmark.

    Attributes:
        cfg (AssemblyKitEnvCfg): Configuration for this environment.
    """

    cfg: AssemblyKitEnvCfg

    def __init__(self, cfg: AssemblyKitEnvCfg, render_mode: str | None = None, **kwargs):
        """Initializes the assembly-kit environment

        Sets up asset directories, loads episodes, spawns the scene

        Args:
            cfg:         AssemblyKitEnvCfg with environment parameters.
            render_mode: Optional render mode (e.g. "rgb" or "state").
            **kwargs:    Additional arguments for the base DirectRLEnv.
        """

        # parsing json episode file

        super().__init__(cfg, render_mode, **kwargs)

        random.seed(self.cfg.seed)

        self.joint_ids, _ = self.robot.find_joints("panda_joint.*|panda_finger_joint.*")

        self.target_model_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.target_model_goal_rot = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _load_table_scene(self):
        """Spawns the ground plane, table, robot, and camera into the simulation.

        Creates a dome light and sets up per-env cloning and debug visualization.
        """

        # Creating the default scene
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0))

        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # Create the transformer and attach it to the scene
        self.tcp_transformer = FrameTransformer(self.cfg.tcp_cfg)
        self.scene.sensors["tcp"] = self.tcp_transformer

        self.vis_marker = VisualizationMarkers(self.cfg.marker_cfg.replace(prim_path="/Visual/Markers"))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_scene(self):
        """Imports kits and models, reads episode JSON, and initializes sampling.

        Spawns the kit and object models, builds lookup tables, and filters inter-env collisions.
        """

        self._load_table_scene()

        # Importing the kits
        self.kit_cfg: RigidObjectCfg = get_kit_cfg(self.cfg.kit_usd_paths)
        self.kit: RigidObject = RigidObject(self.kit_cfg)
        self.scene.rigid_objects["Kit"] = self.kit

        # Stash the target starting positions and model positions per kit
        self.kit_target_starting_pos = torch.tensor(
            self.cfg.kit_target_starting_pos, dtype=torch.float32, device=self.device
        )
        self.model_target_pos = torch.tensor(self.cfg.kit_model_positions, dtype=torch.float32, device=self.device)
        self.model_target_rot = torch.tensor(self.cfg.kit_model_rots, dtype=torch.float32, device=self.device)
        # Stash which kit each env got
        self.kit_ids_per_env = torch.arange(self.num_envs, device=self.device) % len(self.cfg.kit_usd_paths)
        # Stash symmetry values for each model
        self.symmetry = torch.tensor(self.cfg.symmetry, dtype=torch.float32, device=self.device)

        # Importing the models
        self.models: list[RigidObject] = []
        for idx, kit_model_paths in enumerate(self.cfg.kit_models_paths):
            model_cfg = get_model_cfg(kit_model_paths, idx, self.cfg.color)
            self.models.append(RigidObject(model_cfg))
            self.scene.rigid_objects[f"Model_{idx}"] = self.models[-1]

        # Computing the Asset lookup table for the environment instances
        self.env_assets_info = self._get_assets_info_per_env(self.cfg.kit_usd_paths, self.cfg.kit_model_ids)

        self.init_model_sampling()

        # Filtering collisions for optimization of collisions between environment instances
        self.scene.filter_collisions([
            "/World/ground",
        ])

    def _get_assets_info_per_env(
        self,
        kit_usd_paths: list[str],
        kit_model_ids: list[list[int]],
    ) -> list[dict]:
        """Constructs per-env metadata about which kit and models to use.

        Args:
            kit_usd_paths:  List of all kit USD paths.
            kit_model_ids:  Nested list mapping kit to its object IDs.

        Returns:
            List of dicts per env with 'kit_id' and 'model_ids'.
        """
        num_envs = self.num_envs
        num_kits = len(kit_usd_paths)
        return [
            {
                "kit_id": env_id % num_kits,
                "model_ids": kit_model_ids[env_id % num_kits],
            }
            for env_id in range(num_envs)
        ]

    def init_model_sampling(self) -> None:
        """Preallocates tensors for target and other model sampling across envs."""
        # pulled straight from the list-of-dicts you already built
        model_ids_list = [info["model_ids"] for info in self.env_assets_info]
        # assume same-length lists → a dense matrix
        self.model_ids_matrix_per_env = torch.tensor(
            model_ids_list,
            dtype=torch.long,
            device=self.device,
        )

        self.num_shape_types = int(self.model_ids_matrix_per_env.max().item()) + 1

        num_envs, K = self.model_ids_matrix_per_env.shape
        # to write into each reset
        self.current_target = torch.empty(num_envs, 1, dtype=torch.long, device=self.device)
        # one fewer column
        self.current_others = torch.empty(num_envs, K - 1, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.cfg.robot_controller == "task_space":
            self.robot.set_joint_position_target(self.actions, joint_ids=self.joint_ids)
        else:
            self.robot.set_joint_effort_target(self.actions, joint_ids=self.joint_ids)

    def get_tcp_poses(self) -> torch.Tensor:
        """
        Returns an (N,7) tensor [x, y, z, qx, qy, qz, qw]
        """
        pos = self.tcp_transformer.data.target_pos_w.squeeze(1) - self.scene.env_origins
        quat = self.tcp_transformer.data.target_quat_w.squeeze(1)
        return torch.cat((pos, quat), dim=1)

    def _get_observations(self) -> dict:
        """Collects state or pixel observations for the policy.

        Reads TCP pose, target model pose, computes shape one-hot, and returns
        either concatenated state vectors or camera images.

        Returns:
            Dict with key 'policy' mapping to either a state tensor or image tensor.
        """
        tcp_pose = self.get_tcp_poses()

        target_model_pose = self.get_target_model_pose()

        shape_one_hot = (
            torch.nn.functional.one_hot(self.current_target_model_id, num_classes=self.num_shape_types)
            .to(self.device)
            .float()
        )

        state_obs = torch.cat(
            (
                tcp_pose,
                target_model_pose,
                target_model_pose[:, :3] - tcp_pose[:, :3],
                self.target_model_goal_pos,
                self.target_model_goal_rot.unsqueeze(1),
                self.target_model_goal_pos - target_model_pose[:, :3],
                shape_one_hot,
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

    def get_target_model_pose(self) -> torch.Tensor:
        """Returns the world pose of the current target object in env-local coords.

        Returns:
            Tensor of shape (num_envs, 7) with position and quaternion orientation.
        """

        # Get the index of the target object for each environment
        target_models: list[RigidObject] = [self.models[idx.item()] for idx in self.current_target[:, 0]]

        target_model_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)

        for idx, model in enumerate(target_models):
            target_model_pose[idx] = model.data.root_state_w[idx, :7]

        # convert to env-local
        target_model_pose[:, :3] -= self.scene.env_origins

        return target_model_pose

    def _get_rewards(self) -> torch.Tensor:
        success_mask = self.is_success()
        ones = torch.ones_like(success_mask, dtype=torch.int, device=self.device)
        return torch.where(success_mask, ones, -ones)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        done = self.is_success()
        if self.cfg.robot_controller == "task_space":
            timeout = torch.zeros_like(done, dtype=torch.bool)
        else:
            timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        return done, timeout

    def is_success(self, pos_eps=2e-2, rot_eps=math.radians(4), height_eps=3e-3) -> torch.Tensor:
        """Checks if target objects are correctly placed and oriented.

        Args:
            pos_eps:    Positional tolerance in XY plane.
            rot_eps:    Rotational tolerance around Z axis.
            height_eps: Height threshold for placement in slot.

        Returns:
            Boolean tensor per env indicating success.
        """

        target_model_pose = self.get_target_model_pose()

        # Compute positional difference (XY-plane)
        pos_diff = self.target_model_goal_pos[:, :2] - target_model_pose[:, :2]
        pos_diff_norm = torch.norm(pos_diff, dim=1)
        pos_correct = pos_diff_norm < pos_eps

        # Compute rotational difference considering symmetry
        target_model_quat = target_model_pose[:, 3:7]
        target_model_euler = euler_xyz_from_quat(target_model_quat)
        target_model_z_rot = target_model_euler[2]

        goal_z_rot = self.target_model_goal_rot
        rot_diff = torch.abs(target_model_z_rot - goal_z_rot) % self.symmetry[self.current_target_model_id]

        # Adjust symmetry difference
        symmetry_val = self.symmetry[self.current_target_model_id]
        half_symmetry = symmetry_val / 2
        rot_diff = torch.where(rot_diff > half_symmetry, symmetry_val - rot_diff, rot_diff)
        rot_correct = rot_diff < rot_eps

        # Check height to determine if the object is correctly placed in the slot

        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        env_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        obj_up_world = quat_apply(target_model_quat, local_up)

        upright_cos = (obj_up_world * env_normal).sum(dim=1).clamp(-1.0, 1.0)
        flip_mask = upright_cos < 0

        thr_upright = torch.full((self.num_envs,), height_eps, device=self.device)
        thr_flipped = torch.full((self.num_envs,), 0.023, device=self.device)
        height_thr = torch.where(flip_mask, thr_flipped, thr_upright)

        height_correct = target_model_pose[:, 2] < height_thr

        self.vis_marker.visualize(
            self.target_model_goal_pos[:, :3] + self.scene.env_origins,
            torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device),
        )

        success = pos_correct & rot_correct & height_correct

        return success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Resets robot, kit, and object states for the given env indices.

        Args:
            env_ids: List or tensor of environment indices to reset.
        """
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

        # resetting kit
        kit_default_state = self.kit.data.default_root_state.clone()[env_ids]
        kit_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.kit.write_root_pose_to_sim(kit_default_state[:, :7], env_ids)
        self.kit.write_root_velocity_to_sim(kit_default_state[:, 7:], env_ids)

        # sample objects for the environments
        self.sample_models_for_envs(torch.as_tensor(env_ids, dtype=torch.long, device=self.device))

    def sample_models_for_envs(self, env_ids: torch.Tensor) -> None:
        """Randomly selects a target model and places others at goal or side positions.

        Args:
            env_ids: Tensor of environment indices for sampling.
        """
        device = self.device
        models = self.models

        num_envs = env_ids.shape[0]
        num_models = self.model_target_pos.shape[1]

        # determine kit positions
        kit_ids = self.kit_ids_per_env[env_ids]
        kit_default_state = self.kit.data.default_root_state.clone()[env_ids]
        kit_pos = kit_default_state[:, :3]

        # pulling JSON target poses for the other model
        model_rel_pos = self.model_target_pos[kit_ids]
        model_rel_rot = self.model_target_rot[kit_ids]

        # pick target model and compute other model indices
        choice = torch.randint(0, num_models, (num_envs,), device=device)
        self.current_target[env_ids] = choice.unsqueeze(1)
        self.current_target_model_id = (
            torch.tensor([info["model_ids"] for info in self.env_assets_info], device=self.device)
            .gather(1, self.current_target)
            .squeeze(1)
        )
        all_cols = torch.arange(num_models, device=device).unsqueeze(0).expand(num_envs, num_models)
        others_mask = all_cols != choice.unsqueeze(1)
        others_idx = all_cols[others_mask].view(num_envs, num_models - 1)

        # compute and target starting poses
        sampled_idx = torch.randint(
            low=0,
            high=self.kit_target_starting_pos.shape[1],
            size=(num_envs,),
            device=device,
        )
        offsets = self.kit_target_starting_pos[kit_ids, sampled_idx]
        target_pos = kit_pos + offsets
        rot_t = torch.rand(num_envs, device=device) * 2 * math.pi
        target_quat = quat_from_euler_xyz(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            rot_t,
        )

        # Storing the target poses and orientations for observation and reward
        self.target_model_goal_pos[env_ids] = (
            model_rel_pos[torch.arange(model_rel_pos.size(0), device=choice.device), choice][env_ids] + kit_pos[env_ids]
        )
        self.target_model_goal_rot[env_ids] = model_rel_rot[
            torch.arange(model_rel_pos.size(0), device=choice.device), choice
        ]

        # compute world-space poses + rots for others models
        idx_b = torch.arange(num_envs, device=device).unsqueeze(1)
        others_pos = model_rel_pos[idx_b, others_idx]
        others_rot = model_rel_rot[idx_b, others_idx]

        # determine which others to place at the side
        mask_other_to_side = torch.zeros_like(others_rot, dtype=torch.bool, device=device)
        do_pick = torch.rand(num_envs, device=device) < 0.5
        if do_pick.any():
            cols = torch.randint(0, num_models - 1, (num_envs,), device=device)
            rows = torch.arange(num_envs, device=device)
            mask_other_to_side[rows[do_pick], cols[do_pick]] = True
        others_pos[mask_other_to_side] = torch.tensor(
            [[-self.cfg.TABLE_OFFSET, 0.2, 0.1]],
            device=device,
            dtype=others_pos.dtype,
        )
        others_pos_w = kit_pos.unsqueeze(1).expand(num_envs, num_models - 1, 3) + others_pos
        others_rot_w = others_rot
        # flatten for per-model grouping
        envs_o_flat = env_ids.unsqueeze(1).expand(num_envs, num_models - 1).reshape(-1)
        idx_o_flat = others_idx.reshape(-1)
        pos_o_flat = others_pos_w.reshape(-1, 3)
        rot_o_flat = others_rot_w.reshape(-1)

        # Setting poses and orientations for the target and other models
        for model_idx in range(num_models):
            default = models[model_idx].data.default_root_state[env_ids].clone()

            mask_t = choice == model_idx
            if mask_t.any():
                envs_t = env_ids[mask_t]
                poses_t = target_pos[mask_t]
                quats_t = target_quat[mask_t]

                default[envs_t, :3] = poses_t
                default[envs_t, 3:7] = quats_t

            mask_o = idx_o_flat == model_idx
            if mask_o.any():
                envs_o = envs_o_flat[mask_o]
                poses_o = pos_o_flat[mask_o]
                rots_o = rot_o_flat[mask_o]

                quats_o = quat_from_euler_xyz(
                    torch.zeros_like(mask_o.sum(), device=device),
                    torch.zeros_like(mask_o.sum(), device=device),
                    rots_o,
                )

                default[envs_o, :3] = poses_o
                default[envs_o, 3:7] = quats_o

            default[:, :3] += self.scene.env_origins[env_ids]
            models[model_idx].write_root_pose_to_sim(default[:, :7], env_ids)
