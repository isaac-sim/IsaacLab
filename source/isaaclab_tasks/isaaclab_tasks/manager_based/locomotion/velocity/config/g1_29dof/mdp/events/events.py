# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    contact_solver_name:str="physics_callback",
)->None:
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    friction_samples = math_utils.sample_uniform(
        friction_range[0], friction_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_friction_params(env_ids, friction_samples, friction_samples)

def randomize_terrain_stiffness(
    env: ManagerBasedEnv, 
    env_ids: Sequence[int], 
    stiffness_range: tuple[float, float],
    contact_solver_name:str="physics_callback",
)->None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    stiffness_samples = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, stiffness_samples)