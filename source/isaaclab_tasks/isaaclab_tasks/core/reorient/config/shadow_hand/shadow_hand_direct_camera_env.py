# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.utils.math import scale_transform

from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor
from isaaclab_tasks.core.reorient.mdp.observations import compute_cube_keypoints
from isaaclab_tasks.core.reorient.reorient_common import CAMERA_GOAL_MARKER_POSITION
from isaaclab_tasks.core.reorient.reorient_direct_env import ReorientDirectEnv

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import ShadowHandCameraEnvCfg


class ShadowHandCameraEnv(ReorientDirectEnv):
    cfg: ShadowHandCameraEnvCfg

    def __init__(self, cfg: ShadowHandCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Derive CNN input data types from the resolved camera config so that any camera
        # preset (e.g. presets=rgb, presets=albedo) automatically configures the right
        # network input channels without requiring a separate env config class.
        self.feature_extractor = FeatureExtractor(
            self.cfg.feature_extractor,
            self.device,
            self.cfg.tiled_camera.data_types,
            self.cfg.log_dir,
            height=self.cfg.tiled_camera.height,
            width=self.cfg.tiled_camera.width,
        )
        # hide goal cubes
        self.goal_pos[:, :] = torch.tensor(CAMERA_GOAL_MARKER_POSITION, device=self.device)
        # keypoints buffer
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        self._tiled_camera = Camera(self.cfg.tiled_camera)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_image_observations(self):
        # generate ground truth keypoints for in-hand cube
        compute_cube_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), out=self.gt_keypoints)

        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            self._tiled_camera.data.output,
            object_pose,
        )

        self.embeddings = embeddings.clone().detach()
        # compute keypoints for goal cube
        compute_cube_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1), out=self.goal_keypoints
        )

        obs = torch.cat(
            (
                self.embeddings,
                self.goal_keypoints.view(-1, 24),
            ),
            dim=-1,
        )

        # log pose loss from CNN training (None when disabled or in inference mode)
        if pose_loss is not None:
            if "log" not in self.extras:
                self.extras["log"] = dict()
            self.extras["log"]["pose_loss"] = pose_loss

        return obs

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        obs = torch.cat(
            (
                # hand
                scale_transform(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        state = torch.cat((sim_states, self.embeddings), dim=-1)
        return state

    def _get_observations(self) -> dict:
        # proprioception observations
        state_obs = self._compute_proprio_observations()
        # vision observations from CMM
        image_obs = self._compute_image_observations()
        obs = torch.cat((state_obs, image_obs), dim=-1)
        self._update_fingertip_force_sensors()
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
        return observations
