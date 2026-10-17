# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Homogeneous Factory scene with reset-selectable assembly pairs."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import resolve_presets

from . import factory_assets_cfg as assets
from .assembly_variants import ASSEMBLY_VARIANTS
from .factory_presets import RobotArticulationCfg


def _variant_spawner(configs: tuple[RigidObjectCfg, ...]) -> sim_utils.MultiAssetSpawnerCfg:
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[resolve_presets(cfg.spawn.copy(), selected=("newton_mjwarp",)) for cfg in configs],
        activate_contact_sensors=True,
    )


@configclass
class FactoryVariantSceneCfg(InteractiveSceneCfg):
    """One scene layout shared by every assembly variant."""

    num_envs: int = 4096
    env_spacing: float = 2.0
    clone_cfg: cloner.CloneCfg = cloner.CloneCfg(valid_set=[[0, 0, 0, i, i] for i in range(len(ASSEMBLY_VARIANTS))])

    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG
    nistboard = resolve_presets(assets.NISTBOARD_CFG.copy(), selected=("newton_mjwarp",))
    robot: ArticulationCfg = resolve_presets(RobotArticulationCfg(), selected=("newton_mjwarp",))  # type: ignore
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=_variant_spawner(tuple(variant.fixed_asset for variant in ASSEMBLY_VARIANTS)),
        mesh_variants_enabled=True,
    )
    held_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=_variant_spawner(tuple(variant.held_asset for variant in ASSEMBLY_VARIANTS)),
        mesh_variants_enabled=True,
        mesh_variant_inertia_diagonal_offset=1.0e-5,
    )
    assembly_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/FixedAsset/.*"],
        history_length=1,
        update_period=0.0,
    )
    dome_light = assets.DOMELIGHT_CFG
