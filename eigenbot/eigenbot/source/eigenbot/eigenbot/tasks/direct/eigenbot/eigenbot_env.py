# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .eigenbot_env_cfg import EigenbotEnvCfg


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
_USDZ_PATH = os.path.join(_ASSETS_DIR, "eigenbot_new.usdz")


class EigenbotEnv(DirectRLEnv):

    cfg: EigenbotEnvCfg

    def __init__(self, cfg: EigenbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.num_joints = self.joint_pos.shape[1]

    def _setup_scene(self):

        # Spawn URDF articulation (physics + joints)
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # Attach visual asset
        self._attach_usdz_visual_asset()

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register articulation
        self.scene.articulations["robot"] = self.robot

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _attach_usdz_visual_asset(self):
        """Attach the USDZ robot mesh as a visual asset under the articulation.

        This method references the USDZ once and allows Isaac Lab's environment
        cloning to instance it efficiently across thousands of environments.
        """

        import omni.usd

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            print("[WARN] No USD stage available")
            return

        if not os.path.isfile(_USDZ_PATH):
            print("[WARN] USDZ file not found:", _USDZ_PATH)
            return

        # Robot root prim
        robot_root = "/World/envs/env_0/Robot"

        robot_prim = stage.GetPrimAtPath(robot_root)

        if not robot_prim.IsValid():
            print("[WARN] Robot prim not found:", robot_root)
            return

        # Visual container
        visual_root = robot_root + "/visuals"

        stage.DefinePrim(visual_root, "Xform")

        # Add USDZ reference
        visuals_prim = stage.GetPrimAtPath(visual_root)
        visuals_prim.GetReferences().AddReference(_USDZ_PATH)

        # Transform the USDZ to match the physics robot:
        # - USDZ is in centimeters, URDF is in meters -> scale by 0.01
        # - USDZ is upside down -> rotate 180° around X axis
        from pxr import UsdGeom, Gf

        xformable = UsdGeom.Xformable(visuals_prim)
        xformable.ClearXformOpOrder()

        # Rotation: 180° around X to flip upside down
        rot_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
        rot_op.Set(Gf.Vec3f(180.0, 0.0, 0.0))

        # Scale: mm -> m
        scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale)
        scale_op.Set(Gf.Vec3f(0.001, 0.001, 0.001))

        # Disable collisions on visuals
        from pxr import Usd
        for prim in Usd.PrimRange(visuals_prim):
            if prim.HasAPI("UsdPhysicsCollisionAPI"):
                prim.RemoveAPI("UsdPhysicsCollisionAPI")

        print("[INFO] Attached USDZ visual asset with scale=0.01, rotX=180")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        targets = self.cfg.action_scale * self.actions + self.robot.data.default_joint_pos
        self.robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat((self.joint_pos, self.joint_vel), dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)