# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene definitions for all 20 Factory task variants.

Each scene class inherits from :class:`FactorySceneBase` and specifies the
``fixed_asset`` / ``held_asset`` pair (plus extra scene entities for gear tasks).
:class:`FactorySceneCfg` is the :class:`PresetCfg` that selects among them.
"""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import factory_assets_cfg as assets
from .factory_presets import (
    RobotArticulationCfg,
)


@configclass
class FactorySceneBase(InteractiveSceneCfg):
    """Shared scene assets for all Factory tasks.

    The ``robot`` is resolved from the active robot preset, which the shipped
    task binds as the default; see :mod:`.factory_presets` for how to add a
    robot.
    """

    num_envs: int = 4096
    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG
    nistboard = assets.NISTBOARD_CFG
    robot: ArticulationCfg = RobotArticulationCfg()  # type: ignore
    dome_light = assets.DOMELIGHT_CFG


# ---------------------------------------------------------------------------
# Nut threading (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class NutThreadM16SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BOLT_M16_CFG
    held_asset: RigidObjectCfg = assets.NUT_M16_CFG


# ---------------------------------------------------------------------------
# Gear mesh (3 sizes — all share the same base with 3 gear shafts)
# ---------------------------------------------------------------------------


@configclass
class GearMeshSmallSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.SMALL_GEAR_CFG
    medium_gear: RigidObjectCfg = assets.MEDIUM_GEAR_CFG
    large_gear: RigidObjectCfg = assets.LARGE_GEAR_CFG


@configclass
class GearMeshMediumSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.MEDIUM_GEAR_CFG
    small_gear: RigidObjectCfg = assets.SMALL_GEAR_CFG
    large_gear: RigidObjectCfg = assets.LARGE_GEAR_CFG


@configclass
class GearMeshLargeSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.LARGE_GEAR_CFG
    small_gear: RigidObjectCfg = assets.SMALL_GEAR_CFG
    medium_gear: RigidObjectCfg = assets.MEDIUM_GEAR_CFG


# ---------------------------------------------------------------------------
# Rod insert — round (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class RodInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_4MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_4MM_CFG


@configclass
class RodInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_8MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_8MM_CFG


@configclass
class RodInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_12MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_12MM_CFG


@configclass
class RodInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_16MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_16MM_CFG


# ---------------------------------------------------------------------------
# Peg insert — rectangular (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class PegInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_4MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_4MM_CFG


@configclass
class PegInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_8MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_8MM_CFG


@configclass
class PegInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_12MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_12MM_CFG


@configclass
class PegInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_16MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_16MM_CFG


# ---------------------------------------------------------------------------
# Connector insert (5 types)
# ---------------------------------------------------------------------------


@configclass
class FactorySceneCfg(PresetCfg):
    """Task scene preset — resolves to the complete scene for the active task."""

    nut_thread_m16: NutThreadM16SceneCfg = NutThreadM16SceneCfg(env_spacing=2.0)

    # Gear mesh
    gear_mesh_small: GearMeshSmallSceneCfg = GearMeshSmallSceneCfg(env_spacing=2.0)
    gear_mesh_medium: GearMeshMediumSceneCfg = GearMeshMediumSceneCfg(env_spacing=2.0)
    gear_mesh_large: GearMeshLargeSceneCfg = GearMeshLargeSceneCfg(env_spacing=2.0)

    # Rod insert (round)
    rod_insert_4mm: RodInsert4MMSceneCfg = RodInsert4MMSceneCfg(env_spacing=2.0)
    rod_insert_8mm: RodInsert8MMSceneCfg = RodInsert8MMSceneCfg(env_spacing=2.0)
    rod_insert_12mm: RodInsert12MMSceneCfg = RodInsert12MMSceneCfg(env_spacing=2.0)
    rod_insert_16mm: RodInsert16MMSceneCfg = RodInsert16MMSceneCfg(env_spacing=2.0)

    # Peg insert (rectangular)
    peg_insert_4mm: PegInsert4MMSceneCfg = PegInsert4MMSceneCfg(env_spacing=2.0)
    peg_insert_8mm: PegInsert8MMSceneCfg = PegInsert8MMSceneCfg(env_spacing=2.0)
    peg_insert_12mm: PegInsert12MMSceneCfg = PegInsert12MMSceneCfg(env_spacing=2.0)
    peg_insert_16mm: PegInsert16MMSceneCfg = PegInsert16MMSceneCfg(env_spacing=2.0)

    default: NutThreadM16SceneCfg = nut_thread_m16
