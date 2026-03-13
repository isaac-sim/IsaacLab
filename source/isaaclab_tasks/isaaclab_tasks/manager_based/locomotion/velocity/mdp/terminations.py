# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

def terminate_on_z_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate when the actor moves too close to the height of the terrain."""
    asset = env.scene[asset_cfg.name]
    robot_z_pos = asset.data.root_pos_w[:, 2]

    return robot_z_pos < height_threshold

def ground_contact_termination(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when any robot body in sensor list makes contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ground_contact_forces = contact_sensor.data.force_matrix_w

    return torch.any(torch.max(torch.norm(ground_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1)


def ground_contact_termination_history(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_hist = contact_sensor.data.net_forces_w_history

    if force_hist is not None:
        # select bodies -> (N, T, |Bids|, M, 3)
        f = force_hist[:, :, sensor_cfg.body_ids]
        # norm -> (N, T, |Bids|, M)
        f_norm = torch.norm(f, dim=-1)
        # per-step max over bodies & filtered prims -> (N, T)
        per_step = torch.amax(f_norm, dim=(2, 3))
    else:
        # Fallback: unfiltered net forces history (N, T, B, 3) :contentReference[oaicite:2]{index=2}
        net_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        per_step = torch.amax(torch.norm(net_hist, dim=-1), dim=2)  # (N, T)

    return torch.all(per_step > eps, dim=1)  # (N,)