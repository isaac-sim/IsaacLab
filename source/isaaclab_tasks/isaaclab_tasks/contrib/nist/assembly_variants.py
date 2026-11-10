# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly pairs and their reset geometry."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.assets import RigidObjectCfg

from . import factory_assets_cfg as assets
from .assembly_keypoints import NIST_BOARD_CFG, Offset
from .assembly_profile_cfg import AssemblyProfileCfg
from .factory_presets import (
    GRASPED_POSE_RANGE,
    GRASPED_POSE_RANGE_CENTERED,
    FactoryAssemblyProfileCfg,
    FixedAssetTipCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetGraspDiameterCfg,
    HeldAssetGraspMiddleCfg,
    HeldAssetGraspPointCfg,
)


@dataclass(frozen=True, slots=True)
class AssemblyVariant:
    """One fixed/held asset pair and all geometry coupled to it."""

    name: str
    fixed_asset: RigidObjectCfg
    held_asset: RigidObjectCfg
    board_offset: Offset
    fixed_tip: Offset
    held_align: Offset
    held_grasp_point: Offset
    held_grasp_middle: Offset
    held_grasp_diameter: float
    profile: AssemblyProfileCfg
    grasped_pose_range: dict[str, tuple[float, float]]
    grasped_pose_range_centered: dict[str, tuple[float, float]]


_ASSET_PAIRS = (
    ("nut_thread_m4", assets.BOLT_M4_CFG, assets.NUT_M4_CFG, NIST_BOARD_CFG.bolt_m4),
    ("nut_thread_m8", assets.BOLT_M8_CFG, assets.NUT_M8_CFG, NIST_BOARD_CFG.bolt_m8),
    ("nut_thread_m12", assets.BOLT_M12_CFG, assets.NUT_M12_CFG, NIST_BOARD_CFG.bolt_m12),
    ("nut_thread_m16", assets.BOLT_M16_CFG, assets.NUT_M16_CFG, NIST_BOARD_CFG.bolt_m16),
    ("gear_mesh_small", assets.GEAR_BASE_CFG, assets.SMALL_GEAR_CFG, NIST_BOARD_CFG.gear_base),
    ("gear_mesh_medium", assets.GEAR_BASE_CFG, assets.MEDIUM_GEAR_CFG, NIST_BOARD_CFG.gear_base),
    ("gear_mesh_large", assets.GEAR_BASE_CFG, assets.LARGE_GEAR_CFG, NIST_BOARD_CFG.gear_base),
    ("rod_insert_4mm", assets.HOLE_4MM_CFG, assets.ROD_4MM_CFG, NIST_BOARD_CFG.hole_4mm),
    ("rod_insert_8mm", assets.HOLE_8MM_CFG, assets.ROD_8MM_CFG, NIST_BOARD_CFG.hole_8mm),
    ("rod_insert_12mm", assets.HOLE_12MM_CFG, assets.ROD_12MM_CFG, NIST_BOARD_CFG.hole_12mm),
    ("rod_insert_16mm", assets.HOLE_16MM_CFG, assets.ROD_16MM_CFG, NIST_BOARD_CFG.hole_16mm),
    (
        "peg_insert_4mm",
        assets.RECTANGULAR_HOLE_4MM_CFG,
        assets.RECTANGULAR_PEG_4MM_CFG,
        NIST_BOARD_CFG.rectangular_hole_4mm,
    ),
    (
        "peg_insert_8mm",
        assets.RECTANGULAR_HOLE_8MM_CFG,
        assets.RECTANGULAR_PEG_8MM_CFG,
        NIST_BOARD_CFG.rectangular_hole_8mm,
    ),
    (
        "peg_insert_12mm",
        assets.RECTANGULAR_HOLE_12MM_CFG,
        assets.RECTANGULAR_PEG_12MM_CFG,
        NIST_BOARD_CFG.rectangular_hole_12mm,
    ),
    (
        "peg_insert_16mm",
        assets.RECTANGULAR_HOLE_16MM_CFG,
        assets.RECTANGULAR_PEG_16MM_CFG,
        NIST_BOARD_CFG.rectangular_hole_16mm,
    ),
    ("usba", assets.USBA_SOCKET_CFG, assets.USBA_PLUG_CFG, NIST_BOARD_CFG.usba_socket),
    ("waterproof", assets.WATERPROOF_SOCKET_CFG, assets.WATERPROOF_PLUG_CFG, NIST_BOARD_CFG.waterproof_socket),
    ("bnc", assets.BNC_SOCKET_CFG, assets.BNC_PLUG_CFG, NIST_BOARD_CFG.bnc_socket),
    ("dsub", assets.DSUB_SOCKET_CFG, assets.DSUB_PLUG_CFG, NIST_BOARD_CFG.dsub_socket),
    ("rj45", assets.RJ45_SOCKET_CFG, assets.RJ45_PLUG_CFG, NIST_BOARD_CFG.rj45_socket),
)

_FIXED_TIPS = FixedAssetTipCfg()
_HELD_ALIGN_OFFSETS = HeldAssetAlignOffsetCfg()
_HELD_GRASP_POINTS = HeldAssetGraspPointCfg()
_HELD_GRASP_MIDDLES = HeldAssetGraspMiddleCfg()
_HELD_GRASP_DIAMETERS = HeldAssetGraspDiameterCfg()
_PROFILES = FactoryAssemblyProfileCfg()


ASSEMBLY_VARIANTS = tuple(
    AssemblyVariant(
        name=name,
        fixed_asset=fixed_asset,
        held_asset=held_asset,
        board_offset=board_offset,
        fixed_tip=getattr(_FIXED_TIPS, name),
        held_align=getattr(_HELD_ALIGN_OFFSETS, name),
        held_grasp_point=getattr(_HELD_GRASP_POINTS, name),
        held_grasp_middle=getattr(_HELD_GRASP_MIDDLES, name),
        held_grasp_diameter=getattr(_HELD_GRASP_DIAMETERS, name),
        profile=getattr(_PROFILES, name),
        grasped_pose_range=getattr(GRASPED_POSE_RANGE, name),
        grasped_pose_range_centered=getattr(GRASPED_POSE_RANGE_CENTERED, name),
    )
    for name, fixed_asset, held_asset, board_offset in _ASSET_PAIRS
)
"""Ordered assembly variants. This order is also the Newton mesh-variant index."""

ASSEMBLY_VARIANT_NAMES = tuple(variant.name for variant in ASSEMBLY_VARIANTS)
"""Assembly names in mesh-variant index order."""
