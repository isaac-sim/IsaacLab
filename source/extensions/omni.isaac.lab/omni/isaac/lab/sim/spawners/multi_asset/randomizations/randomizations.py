# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementations for randomizations that are applied at the time of spawning prims."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import carb
import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd

import omni.isaac.lab.sim as sim_utils

if TYPE_CHECKING:
    from .randomizations_cfg import RandomizeJointOffsetsCfg, RandomizeScaleCfg


def randomize_joint_offset(prim: Usd.Prim, cfg: RandomizeJointOffsetsCfg) -> Usd.Prim:
    """Randomize the joint offsets of a joint in an articulation by modifying the physics:localPos0 property.
    Args:
        prim: The root prim of the articulation to randomize
        cfg: The configuration for the randomization
    """

    # Find matching prims
    matching_prims = sim_utils.find_matching_prim_paths(str(prim.GetPath()) + cfg.joint_name)

    if len(matching_prims) == 0:
        return prim

    for prim_path in matching_prims:
        prim = prim_utils.get_prim_at_path(prim_path)
        prop = prim.GetProperty("physics:localPos0")
        value = prop.Get()
        value[0] += random.random() * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]
        value[1] += random.random() * (cfg.y_range[1] - cfg.y_range[0]) + cfg.y_range[0]
        value[2] += random.random() * (cfg.z_range[1] - cfg.z_range[0]) + cfg.z_range[0]

        # Set property
        if not prop.Set(value):
            carb.log_error("Failed to set property", prop, "for", prim_path)
    return prim


def randomize_scale(prim: Usd.Prim, cfg: RandomizeScaleCfg) -> Usd.Prim:
    """Randomize the scale of a prim by modifying the xformOp:scale property.
    Args:
        prim: The prim to randomize
        cfg: The configuration for the randomization
    """

    matching_prims = sim_utils.find_matching_prim_paths(str(prim.GetPath()))

    if len(matching_prims) == 0:
        return prim

    for prim_path in matching_prims:
        prim = prim_utils.get_prim_at_path(prim_path)
        prop = prim.GetProperty("xformOp:scale")
        value = prop.Get()
        if cfg.equal_scale:
            scale = cfg.x_range[0] + random.random() * (cfg.x_range[1] - cfg.x_range[0])
            value[0] = scale * value[0]
            value[1] = scale * value[1]
            value[2] = scale * value[2]
        else:
            value[0] *= random.random() * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]
            value[1] *= random.random() * (cfg.y_range[1] - cfg.y_range[0]) + cfg.y_range[0]
            value[2] *= random.random() * (cfg.z_range[1] - cfg.z_range[0]) + cfg.z_range[0]

        # Set property
        if not prop.Set(value):
            carb.log_error("Failed to set property", prop, "for", prim_path)
    return prim
