# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in source terrain sensing at the retained MicroDuck sole-site frames.

The legacy flat tasks keep their existing body-height terms. Rough variants can use
``configure_foot_terrain_sensing`` to measure the source site clearance and site velocity.
The source ring with two samples includes its center, hence three rays per foot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import trimesh
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporter


class MicroDuckTerrainImporter(TerrainImporter):
    """Author the source rough terrain's softer MuJoCo contact response before backend parsing."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        super().import_mesh(name, mesh)
        sim_utils.apply_collision_properties(
            f"{self.cfg.prim_path}/{name}/mesh",
            [MujocoCollisionCfg(solref=(0.04, 1.0), solimp=(0.85, 0.95, 0.001, 0.5, 2.0))],
        )


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg


HEIGHT_SENSOR_NAMES = ("left_foot_height", "right_foot_height")
FOOT_SITE_PATHS = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left/left_foot",
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right/right_foot",
)


def foot_height_safe(env: ManagerBasedEnv, height_sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Return minimum vertical site clearance, with upstream miss/backface handling."""
    heights = []
    for name in height_sensor_names:
        sensor = env.scene[name]
        data = sensor.data
        frame_z = data.pos_w.torch[:, 2, None]
        hit_z = data.ray_hits_w.torch[..., 2]
        valid = torch.isfinite(hit_z)
        clearance = frame_z - hit_z
        backface = valid & (data.ray_normals_w.torch[..., 2] < 0.0)
        clearance = torch.where(backface, 0.0, clearance)
        # This is source world-frame site z, even on an elevated terrain origin.
        fallback = frame_z.clamp(0.0, sensor.cfg.max_distance)
        missed = torch.where(valid.any(-1, keepdim=True), sensor.cfg.max_distance, fallback)
        clearance = torch.where(valid, clearance, missed)
        heights.append(clearance.amin(-1))
    return torch.nan_to_num(torch.stack(heights, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def _command_active(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, :2].norm(dim=1) + cmd[:, 2].abs() > threshold


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    height_sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """Price site clearance error with the sole site's planar velocity, including ankle rotation."""
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    body_vel = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    body_ang = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
    site_pos = torch.stack([env.scene[name].data.pos_w.torch for name in height_sensor_names], dim=1)
    site_vel = body_vel + torch.cross(body_ang, site_pos - body_pos, dim=-1)
    error = (foot_height_safe(env, height_sensor_names) - target_height).abs()
    cost = (error * site_vel[..., :2].norm(dim=-1)).sum(1)
    return cost * _command_active(env, command_name, command_threshold)


class foot_swing_height(ManagerTermBase):
    """Charge the source sole-site peak error on first contact, retaining source reset semantics."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._peak_heights = torch.zeros(env.num_envs, len(cfg.params["height_sensor_names"]), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        height_sensor_names: tuple[str, ...],
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        in_air = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        height = foot_height_safe(env, height_sensor_names)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, height), self._peak_heights)
        first = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self._peak_heights / target_height - 1.0
        cost = (error.square() * first).sum(1) * _command_active(env, command_name, command_threshold)
        env.extras["log"]["Metrics/peak_height_mean"] = (self._peak_heights * first).sum() / first.sum().clamp_min(1)
        self._peak_heights = torch.where(first, 0.0, self._peak_heights)
        return cost


def configure_foot_terrain_sensing(cfg: ManagerBasedRLEnvCfg) -> ManagerBasedRLEnvCfg:
    """Add terrain-only source-site rays and redirect the three consumers, without global MDP edits.

    Existing USD foot-site prims are attached via their rigid-body ancestor and fixed local pose
    by the native sensor. The site's full rotation/position precedes yaw-only scan alignment.
    The source target is 2cm at the site; the old ankle-frame target's 2.25cm offset is removed.
    """
    for name, path in zip(HEIGHT_SENSOR_NAMES, FOOT_SITE_PATHS, strict=True):
        setattr(
            cfg.scene,
            name,
            RayCasterCfg(
                prim_path=path,
                mesh_prim_paths=["/World/ground"],
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=(0.08, 0.0)),
                max_distance=1.0,
                global_world_only=True,
                update_period=cfg.sim.dt * cfg.decimation,
            ),
        )
    cfg.observations.critic.foot_height.func = foot_height_safe
    cfg.observations.critic.foot_height.params = {"height_sensor_names": HEIGHT_SENSOR_NAMES}
    cfg.rewards.foot_clearance.func = foot_clearance
    cfg.rewards.foot_swing_height.func = foot_swing_height
    for term in (cfg.rewards.foot_clearance, cfg.rewards.foot_swing_height):
        term.params.update(height_sensor_names=HEIGHT_SENSOR_NAMES, target_height=0.02)
    return cfg
