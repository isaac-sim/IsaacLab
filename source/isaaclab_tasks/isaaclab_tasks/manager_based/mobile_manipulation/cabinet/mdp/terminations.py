from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def cube_in_cabinet(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),

    top_cabinet_height: float = 0.4,
    ztol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    cube_pos = cube_1.data.root_pos_w  
    
    cabinet_body_pos = cabinet.data.body_pos_w  # shape: (1, num_bodies, 3)

    # Compute min and max in the x, y, z dimensions
    cabinet_min = cabinet_body_pos.min(dim=1).values[0]  # shape: (3,)
    cabinet_max = cabinet_body_pos.max(dim=1).values[0]  # shape: (3,)

    # Unpack the results into variables directly
    cabinet_x_min, cabinet_y_min, cabinet_z_min = cabinet_min  # Shape (3,)
    cabinet_x_max, cabinet_y_max, cabinet_z_max = cabinet_max  # Shape (3,)

    print(f"cabinet_min: {cabinet_min}")
    print(f"cabinet_max: {cabinet_max}")
    print(f"cabinet_x_min: {cabinet_x_min}")
    print(f"cabinet_x_max: {cabinet_x_max}")
    print(f"cabinet_y_min: {cabinet_y_min}")
    print(f"cabinet_y_max: {cabinet_y_max}")
    print(f"cabinet_z_min: { cabinet_z_min + top_cabinet_height}")
    print(f"cabinet_z_max: {cabinet_z_max - ztol}")
    print(f"cube_pos: {cube_pos}")
    
    # x
    inside_cabinet =torch.logical_and(
        cube_pos[:, 0] > cabinet_x_min-0.1,
        cube_pos[:, 0] < cabinet_x_max,
    )
    print(f"inside_cabinet x : {inside_cabinet}")
    # y
    inside_cabinet =torch.logical_and(
        cube_pos[:, 1] > cabinet_y_min,
        inside_cabinet,
    )
    inside_cabinet =torch.logical_and(
        cube_pos[:, 1] < cabinet_y_max,
        inside_cabinet,
    )
    print(f"inside_cabinet y : {inside_cabinet}")
    # z
    inside_cabinet =torch.logical_and(
        cube_pos[:, 2] > cabinet_z_min + top_cabinet_height,
        inside_cabinet,
    )
    inside_cabinet =torch.logical_and(
        cube_pos[:, 2] < cabinet_z_max - ztol,
        inside_cabinet,
    )
    print(f"inside_cabinet z : {inside_cabinet}")

    return inside_cabinet