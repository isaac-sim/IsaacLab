# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Homogeneous Factory scene with reset-selectable assembly pairs."""

from __future__ import annotations

import numpy as np

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


def _paired_clone_strategy(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Start each world with matching fixed and held assembly variants."""
    count = len(ASSEMBLY_VARIANTS)
    if weights.size != count * count:
        raise ValueError("Factory variants expect one fixed/held prototype pair per assembly combination.")
    paired = np.arange(count, dtype=np.int32) * (count + 1)
    return paired[cloner.sequential(weights[paired], num_clones)]


def _variant_spawner(configs: tuple[RigidObjectCfg, ...]) -> sim_utils.MultiAssetSpawnerCfg:
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[resolve_presets(cfg.spawn.copy(), selected=("newton_mjwarp",)) for cfg in configs],
        activate_contact_sensors=True,
        static=False,
    )


@configclass
class FactoryVariantSceneCfg(InteractiveSceneCfg):
    """One scene layout shared by every assembly variant."""

    num_envs: int = 4096
    env_spacing: float = 2.0
    clone_cfg: cloner.CloneCfg = cloner.CloneCfg(clone_strategy=_paired_clone_strategy)

    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG
    nistboard = assets.NISTBOARD_CFG
    robot: ArticulationCfg = resolve_presets(RobotArticulationCfg(), selected=("newton_mjwarp",))  # type: ignore
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=_variant_spawner(tuple(variant.fixed_asset for variant in ASSEMBLY_VARIANTS)),
    )
    held_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=_variant_spawner(tuple(variant.held_asset for variant in ASSEMBLY_VARIANTS)),
        mesh_variant_inertia_diagonal_offset=1.0e-5,
    )
    assembly_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/FixedAsset/.*"],
        history_length=1,
        update_period=0.0,
    )
    dome_light = assets.DOMELIGHT_CFG
