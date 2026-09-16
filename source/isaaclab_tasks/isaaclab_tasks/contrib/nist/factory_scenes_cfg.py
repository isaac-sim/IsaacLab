# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene definitions for all 12 Factory task variants.

Each scene class inherits from :class:`FactorySceneBase` and specifies the
``fixed_asset`` / ``held_asset`` pair (plus extra scene entities for gear tasks).
:class:`FactorySceneCfg` is the :class:`PresetCfg` that selects among them.
"""

from dataclasses import dataclass
from typing import Any

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import config_field, replace_config

from isaaclab_tasks.contrib.nist import factory_assets_cfg as assets
from isaaclab_tasks.utils import PresetCfg, preset

_FRANKA_PANDA_PHYSX_CFG = replace_config(assets.FRANKA_PANDA_PHYSX_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@dataclass
class FactorySceneBase(InteractiveSceneCfg):
    """Shared scene assets for all Factory tasks."""

    num_envs: int = config_field(4096)
    ground: Any = config_field(assets.GROUND_CFG)
    table: Any = config_field(assets.TABLE_CFG)
    nistboard: Any = config_field(assets.NISTBOARD_CFG)
    robot: ArticulationCfg = config_field(
        preset(  # type: ignore[assignment]
            default=_FRANKA_PANDA_PHYSX_CFG,
            isaacsim_physx=_FRANKA_PANDA_PHYSX_CFG,
            physx=_FRANKA_PANDA_PHYSX_CFG,
            newton_mjwarp=replace_config(assets.FRANKA_PANDA_NEWTON_CFG, prim_path="{ENV_REGEX_NS}/Robot"),
        )
    )
    dome_light: Any = config_field(assets.DOMELIGHT_CFG)


@dataclass
class NutThreadM16SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.BOLT_M16_CFG)
    held_asset: RigidObjectCfg = config_field(assets.NUT_M16_CFG)


@dataclass
class GearMeshSmallSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.GEAR_BASE_CFG)
    held_asset: RigidObjectCfg = config_field(assets.SMALL_GEAR_CFG)
    medium_gear: RigidObjectCfg = config_field(assets.MEDIUM_GEAR_CFG)
    large_gear: RigidObjectCfg = config_field(assets.LARGE_GEAR_CFG)


@dataclass
class GearMeshMediumSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.GEAR_BASE_CFG)
    held_asset: RigidObjectCfg = config_field(assets.MEDIUM_GEAR_CFG)
    small_gear: RigidObjectCfg = config_field(assets.SMALL_GEAR_CFG)
    large_gear: RigidObjectCfg = config_field(assets.LARGE_GEAR_CFG)


@dataclass
class GearMeshLargeSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.GEAR_BASE_CFG)
    held_asset: RigidObjectCfg = config_field(assets.LARGE_GEAR_CFG)
    small_gear: RigidObjectCfg = config_field(assets.SMALL_GEAR_CFG)
    medium_gear: RigidObjectCfg = config_field(assets.MEDIUM_GEAR_CFG)


@dataclass
class RodInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.HOLE_4MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.ROD_4MM_CFG)


@dataclass
class RodInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.HOLE_8MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.ROD_8MM_CFG)


@dataclass
class RodInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.HOLE_12MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.ROD_12MM_CFG)


@dataclass
class RodInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.HOLE_16MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.ROD_16MM_CFG)


@dataclass
class PegInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_HOLE_4MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_PEG_4MM_CFG)


@dataclass
class PegInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_HOLE_8MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_PEG_8MM_CFG)


@dataclass
class PegInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_HOLE_12MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_PEG_12MM_CFG)


@dataclass
class PegInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_HOLE_16MM_CFG)
    held_asset: RigidObjectCfg = config_field(assets.RECTANGULAR_PEG_16MM_CFG)


@dataclass
class FactorySceneCfg(PresetCfg):
    """Task scene preset — resolves to the complete scene for the active task."""

    nut_thread_m16: NutThreadM16SceneCfg = config_field(NutThreadM16SceneCfg(env_spacing=2.0))

    # Gear mesh
    gear_mesh_small: GearMeshSmallSceneCfg = config_field(GearMeshSmallSceneCfg(env_spacing=2.0))
    gear_mesh_medium: GearMeshMediumSceneCfg = config_field(GearMeshMediumSceneCfg(env_spacing=2.0))
    gear_mesh_large: GearMeshLargeSceneCfg = config_field(GearMeshLargeSceneCfg(env_spacing=2.0))

    # Rod insert (round)
    rod_insert_4mm: RodInsert4MMSceneCfg = config_field(RodInsert4MMSceneCfg(env_spacing=2.0))
    rod_insert_8mm: RodInsert8MMSceneCfg = config_field(RodInsert8MMSceneCfg(env_spacing=2.0))
    rod_insert_12mm: RodInsert12MMSceneCfg = config_field(RodInsert12MMSceneCfg(env_spacing=2.0))
    rod_insert_16mm: RodInsert16MMSceneCfg = config_field(RodInsert16MMSceneCfg(env_spacing=2.0))

    # Peg insert (rectangular)
    peg_insert_4mm: PegInsert4MMSceneCfg = config_field(PegInsert4MMSceneCfg(env_spacing=2.0))
    peg_insert_8mm: PegInsert8MMSceneCfg = config_field(PegInsert8MMSceneCfg(env_spacing=2.0))
    peg_insert_12mm: PegInsert12MMSceneCfg = config_field(PegInsert12MMSceneCfg(env_spacing=2.0))
    peg_insert_16mm: PegInsert16MMSceneCfg = config_field(PegInsert16MMSceneCfg(env_spacing=2.0))

    default: NutThreadM16SceneCfg = config_field(nut_thread_m16)
