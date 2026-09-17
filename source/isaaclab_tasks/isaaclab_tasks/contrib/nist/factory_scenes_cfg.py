# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene definitions for all 12 Factory task variants.

Each scene class inherits from :class:`FactorySceneBase` and specifies the
``fixed_asset`` / ``held_asset`` pair (plus extra scene entities for gear tasks).
:class:`FactorySceneCfg` is the :class:`PresetCfg` that selects among them.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import replace_config

from isaaclab_tasks.contrib.nist import factory_assets_cfg as assets
from isaaclab_tasks.utils import PresetCfg, preset

_FRANKA_PANDA_PHYSX_CFG = replace_config(assets.FRANKA_PANDA_PHYSX_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@dataclass
class FactorySceneBase(InteractiveSceneCfg):
    """Shared scene assets for all Factory tasks."""

    num_envs: int = 4096
    ground: Any = field(default_factory=lambda: deepcopy(assets.GROUND_CFG))
    table: Any = field(default_factory=lambda: deepcopy(assets.TABLE_CFG))
    nistboard: Any = field(default_factory=lambda: deepcopy(assets.NISTBOARD_CFG))
    robot: ArticulationCfg = field(
        default_factory=lambda: preset(  # type: ignore[assignment]
            default=_FRANKA_PANDA_PHYSX_CFG,
            isaacsim_physx=_FRANKA_PANDA_PHYSX_CFG,
            physx=_FRANKA_PANDA_PHYSX_CFG,
            newton_mjwarp=replace_config(assets.FRANKA_PANDA_NEWTON_CFG, prim_path="{ENV_REGEX_NS}/Robot"),
        )
    )
    dome_light: Any = field(default_factory=lambda: deepcopy(assets.DOMELIGHT_CFG))


@dataclass
class NutThreadM16SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.BOLT_M16_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.NUT_M16_CFG))


@dataclass
class GearMeshSmallSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.GEAR_BASE_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.SMALL_GEAR_CFG))
    medium_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.MEDIUM_GEAR_CFG))
    large_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.LARGE_GEAR_CFG))


@dataclass
class GearMeshMediumSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.GEAR_BASE_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.MEDIUM_GEAR_CFG))
    small_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.SMALL_GEAR_CFG))
    large_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.LARGE_GEAR_CFG))


@dataclass
class GearMeshLargeSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.GEAR_BASE_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.LARGE_GEAR_CFG))
    small_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.SMALL_GEAR_CFG))
    medium_gear: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.MEDIUM_GEAR_CFG))


@dataclass
class RodInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.HOLE_4MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.ROD_4MM_CFG))


@dataclass
class RodInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.HOLE_8MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.ROD_8MM_CFG))


@dataclass
class RodInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.HOLE_12MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.ROD_12MM_CFG))


@dataclass
class RodInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.HOLE_16MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.ROD_16MM_CFG))


@dataclass
class PegInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_HOLE_4MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_PEG_4MM_CFG))


@dataclass
class PegInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_HOLE_8MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_PEG_8MM_CFG))


@dataclass
class PegInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_HOLE_12MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_PEG_12MM_CFG))


@dataclass
class PegInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_HOLE_16MM_CFG))
    held_asset: RigidObjectCfg = field(default_factory=lambda: deepcopy(assets.RECTANGULAR_PEG_16MM_CFG))


@dataclass
class FactorySceneCfg(PresetCfg):
    """Task scene preset — resolves to the complete scene for the active task."""

    nut_thread_m16: NutThreadM16SceneCfg = field(default_factory=lambda: NutThreadM16SceneCfg(env_spacing=2.0))

    # Gear mesh
    gear_mesh_small: GearMeshSmallSceneCfg = field(default_factory=lambda: GearMeshSmallSceneCfg(env_spacing=2.0))
    gear_mesh_medium: GearMeshMediumSceneCfg = field(default_factory=lambda: GearMeshMediumSceneCfg(env_spacing=2.0))
    gear_mesh_large: GearMeshLargeSceneCfg = field(default_factory=lambda: GearMeshLargeSceneCfg(env_spacing=2.0))

    # Rod insert (round)
    rod_insert_4mm: RodInsert4MMSceneCfg = field(default_factory=lambda: RodInsert4MMSceneCfg(env_spacing=2.0))
    rod_insert_8mm: RodInsert8MMSceneCfg = field(default_factory=lambda: RodInsert8MMSceneCfg(env_spacing=2.0))
    rod_insert_12mm: RodInsert12MMSceneCfg = field(default_factory=lambda: RodInsert12MMSceneCfg(env_spacing=2.0))
    rod_insert_16mm: RodInsert16MMSceneCfg = field(default_factory=lambda: RodInsert16MMSceneCfg(env_spacing=2.0))

    # Peg insert (rectangular)
    peg_insert_4mm: PegInsert4MMSceneCfg = field(default_factory=lambda: PegInsert4MMSceneCfg(env_spacing=2.0))
    peg_insert_8mm: PegInsert8MMSceneCfg = field(default_factory=lambda: PegInsert8MMSceneCfg(env_spacing=2.0))
    peg_insert_12mm: PegInsert12MMSceneCfg = field(default_factory=lambda: PegInsert12MMSceneCfg(env_spacing=2.0))
    peg_insert_16mm: PegInsert16MMSceneCfg = field(default_factory=lambda: PegInsert16MMSceneCfg(env_spacing=2.0))

    default: NutThreadM16SceneCfg = field(default_factory=lambda: NutThreadM16SceneCfg(env_spacing=2.0))
