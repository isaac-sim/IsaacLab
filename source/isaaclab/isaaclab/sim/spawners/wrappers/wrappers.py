# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pxr import Usd

import isaaclab.sim as sim_utils

from ..from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg

logger = logging.getLogger(__name__)

_OVERRIDABLE_PROPS = ("mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props")
"""Spawner settings a multi-asset wrapper applies to every asset it spawns."""


def spawn_multi_asset(
    prim_path: str,
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets into numbered prim paths derived from the provided configuration.

    Assets are created in the order they appear in ``cfg.assets_cfg`` using the base name in ``prim_path``,
    which must contain ``.*`` (for example, ``/World/Env_0/asset_.*`` spawns ``asset_0``, ``asset_1``, ...).

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    if cfg.spawn_paths is not None:
        if len(cfg.spawn_paths) != len(cfg.assets_cfg):
            raise ValueError(
                f"Expected spawn_paths to match assets_cfg length, got {len(cfg.spawn_paths)} and"
                f" {len(cfg.assets_cfg)}."
            )
        asset_prim_paths = list(cfg.spawn_paths)
    else:
        # split on separators only: a segment wildcard is written as a character class whose
        # text contains a '/' that is not a separator.
        split_path = sim_utils.split_path_expr(prim_path)
        prefix_path, base_name = "/".join(split_path[:-1]), split_path[-1]
        # the base name carries the index slot as a segment wildcard, in any of its spellings.
        # Normalizing to glob collapses them to the single '*' that the index replaces.
        base_glob = sim_utils.path_expr_to_glob(base_name)
        if "*" not in base_glob:
            raise ValueError(
                f" The base name '{base_name}' in the prim path '{prim_path}' must contain a segment wildcard"
                " (e.g. '.*' or '[^/]*') to indicate the path each individual multiple-asset to be spawned."
            )
        asset_prim_paths = [f"{prefix_path}/{base_glob.replace('*', str(i))}" for i in range(len(cfg.assets_cfg))]

    if cfg.random_choice:
        logger.warning(
            "`random_choice` parameter in `spawn_multi_asset` is deprecated, and nothing will happen. "
            "Use `isaaclab.scene.interactive_scene_cfg.InteractiveSceneCfg.random_heterogeneous_cloning` instead."
        )

    spawned_prim_paths: list[str] = []
    for asset_prim_path, asset_cfg in zip(asset_prim_paths, cfg.assets_cfg):
        if asset_prim_path is None:
            continue
        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags
        # wrapper-level physics settings override the per-asset ones
        for attr_name in _OVERRIDABLE_PROPS:
            attr_value = getattr(cfg, attr_name)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)

        asset_cfg.func(
            asset_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )
        spawned_prim_paths.append(asset_prim_path)
    if not spawned_prim_paths:
        raise ValueError("No assets were spawned. At least one spawn path must be active.")
    return sim_utils.find_first_matching_prim(spawned_prim_paths[0])


def spawn_multi_usd_file(
    prim_path: str,
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    # needed here to avoid circular imports
    from .wrappers_cfg import MultiAssetSpawnerCfg

    usd_paths = [cfg.usd_path] if isinstance(cfg.usd_path, str) else cfg.usd_path

    # every file shares the same settings, differing only in its usd path
    usd_template_cfg = UsdFileCfg()
    for attr_name, attr_value in cfg.__dict__.items():
        if attr_name not in ("func", "usd_path", "random_choice", "spawn_path", "spawn_paths"):
            setattr(usd_template_cfg, attr_name, attr_value)
    multi_asset_cfg = MultiAssetSpawnerCfg(
        assets_cfg=[usd_template_cfg.replace(usd_path=usd_path) for usd_path in usd_paths],
        spawn_paths=cfg.spawn_paths,
        # the wrapper's default (False) would otherwise override the per-file setting in spawn_multi_asset
        activate_contact_sensors=cfg.activate_contact_sensors,
    )
    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation, clone_in_fabric, replicate_physics)
