# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PresetCfg definitions for Factory task-specific parameters.

Each preset maps task variant names to the corresponding keypoint-derived values.
The ``default`` field selects which variant is active when no CLI override is given.

Task categories (20 total):
  - ``nut_thread_m{4,8,12,16}`` — nut threading
  - ``gear_mesh_{small,medium,large}`` — gear meshing
  - ``rod_insert_{4,8,12,16}mm`` — round rod insertion
  - ``peg_insert_{4,8,12,16}mm`` — rectangular peg insertion
  - ``{usba,waterproof,bnc,dsub,rj45}`` — connector insertion

Robot presets default to the Franka Panda and select backend-specific asset settings.
"""

import math
from dataclasses import fields

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from . import assembly_keypoints as kpts
from .assembly_profile_cfg import (
    AssemblyProfileCfg,
    DiscreteYawCfg,
    EndPointsSegmentCfg,
    IncrementalSegmentCfg,
    UniformYawCfg,
)
from .factory_assets_cfg import FRANKA_PANDA_NEWTON_CFG, FRANKA_PANDA_PHYSX_CFG

# ---------------------------------------------------------------------------
# Robot-specific presets
# ---------------------------------------------------------------------------


@configclass
class RobotArticulationCfg(PresetCfg):
    """Franka Panda asset settings for each physics backend."""

    default: ArticulationCfg = preset(
        default=FRANKA_PANDA_PHYSX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        physx=FRANKA_PANDA_PHYSX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        newton_mjwarp=FRANKA_PANDA_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )


@configclass
class EndEffectorBodyCfg(PresetCfg):
    """End-effector body name per robot variant."""

    default: str = "panda_hand"


@configclass
class GripperJointNamesCfg(PresetCfg):
    """Joint name regex for gripper/finger joints per robot variant."""

    default: list[str] | None = ["panda_finger.*"]


@configclass
class IKJointNamesCfg(PresetCfg):
    """Joint name regex used by the IK solver in reset strategies."""

    default: list[str] | None = ["panda_joint.*"]


@configclass
class GripperGraspOffsetCfg(PresetCfg):
    """Gripper grasp frame offset relative to the end-effector link per robot variant."""

    default: kpts.Offset = kpts.PANDA_HAND.gripper_center_grasp_point


@configclass
class JointEffortNamesCfg(PresetCfg):
    """Joint name regex for the effort penalty per robot variant."""

    default: str | None = "(?!panda_joint7$|panda_finger_.*$).*"


# ---------------------------------------------------------------------------
# Task-specific presets
# ---------------------------------------------------------------------------


@configclass
class FixedAssetMapCfg(PresetCfg):
    """Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name."""

    # Nut threading
    nut_thread_m4: dict = dict(fixed_asset="bolt_m4")
    nut_thread_m8: dict = dict(fixed_asset="bolt_m8")
    nut_thread_m12: dict = dict(fixed_asset="bolt_m12")
    nut_thread_m16: dict = dict(fixed_asset="bolt_m16")

    # Gear mesh — non-held gears are placed on the board as extra scene entities
    gear_mesh_small: dict = dict(fixed_asset="gear_base", medium_gear="medium_gear", large_gear="large_gear")
    gear_mesh_medium: dict = dict(fixed_asset="gear_base", small_gear="small_gear", large_gear="large_gear")
    gear_mesh_large: dict = dict(fixed_asset="gear_base", small_gear="small_gear", medium_gear="medium_gear")

    # Rod insert (round)
    rod_insert_4mm: dict = dict(fixed_asset="hole_4mm")
    rod_insert_8mm: dict = dict(fixed_asset="hole_8mm")
    rod_insert_12mm: dict = dict(fixed_asset="hole_12mm")
    rod_insert_16mm: dict = dict(fixed_asset="hole_16mm")

    # Peg insert (rectangular)
    peg_insert_4mm: dict = dict(fixed_asset="rectangular_hole_4mm")
    peg_insert_8mm: dict = dict(fixed_asset="rectangular_hole_8mm")
    peg_insert_12mm: dict = dict(fixed_asset="rectangular_hole_12mm")
    peg_insert_16mm: dict = dict(fixed_asset="rectangular_hole_16mm")

    # Connector insert
    usba: dict = dict(fixed_asset="usba_socket")
    waterproof: dict = dict(fixed_asset="waterproof_socket")
    bnc: dict = dict(fixed_asset="bnc_socket")
    dsub: dict = dict(fixed_asset="dsub_socket")
    rj45: dict = dict(fixed_asset="rj45_socket")

    default: dict = nut_thread_m16


@configclass
class AssemblyTipCfg(PresetCfg):
    # Nut threading — tip of the bolt shaft where the nut enters
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16.bolt_tip_offset

    # Gear mesh — tip of the gear shaft on the base
    gear_mesh_small: kpts.Offset = kpts.GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.BNC_SOCKET.entry
    dsub: kpts.Offset = kpts.D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


@configclass
class FixedAssetTipCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16.bolt_tip_offset

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


def _dist(a: kpts.Offset, b: kpts.Offset) -> tuple[float, float, float]:
    """Position delta from offset *a* to offset *b* [m]."""
    return (b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2])


@configclass
class FactoryAssemblyProfileCfg(PresetCfg):
    """Assembly profile per task variant.

    Each field is an :class:`AssemblyProfileCfg` describing the full assembly
    path geometry (segment endpoints, screw ratios, start noise).
    """

    # Nut threading — IncrementalSegment with screw pitch
    nut_thread_m4: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M4.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M4.fully_screwed_nut_offset, kpts.BOLT_M4.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M4.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m8: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M8.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M8.fully_screwed_nut_offset, kpts.BOLT_M8.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M8.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m12: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M12.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M12.fully_screwed_nut_offset, kpts.BOLT_M12.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M12.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m16: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M16.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M16.fully_screwed_nut_offset, kpts.BOLT_M16.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M16.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )

    # Gear mesh
    gear_mesh_small: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 1.0),
                start_sampler=UniformYawCfg(),
                start_pose=kpts.GEAR_BASE.small_gear_assembled_bottom_offset,
                distance=_dist(kpts.GEAR_BASE.small_gear_assembled_bottom_offset, kpts.GEAR_BASE.small_gear_tip_offset),
            )
        ]
    )
    gear_mesh_medium: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 1.0),
                start_sampler=UniformYawCfg(),
                start_pose=kpts.GEAR_BASE.medium_gear_assembled_bottom_offset,
                distance=_dist(
                    kpts.GEAR_BASE.medium_gear_assembled_bottom_offset, kpts.GEAR_BASE.medium_gear_tip_offset
                ),
            )
        ]
    )
    gear_mesh_large: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 1.0),
                start_sampler=UniformYawCfg(),
                start_pose=kpts.GEAR_BASE.large_gear_assembled_bottom_offset,
                distance=_dist(kpts.GEAR_BASE.large_gear_assembled_bottom_offset, kpts.GEAR_BASE.large_gear_tip_offset),
            )
        ]
    )

    # Rod insert (round) — any yaw is valid
    rod_insert_4mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=UniformYawCfg(),
                start_pose=kpts.HOLE_4MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_4MM.inserted_peg_base_offset, kpts.HOLE_4MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_8mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=UniformYawCfg(),
                start_pose=kpts.HOLE_8MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_8MM.inserted_peg_base_offset, kpts.HOLE_8MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_12mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=UniformYawCfg(),
                start_pose=kpts.HOLE_12MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_12MM.inserted_peg_base_offset, kpts.HOLE_12MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_16mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=UniformYawCfg(),
                start_pose=kpts.HOLE_16MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_16MM.inserted_peg_base_offset, kpts.HOLE_16MM.hole_tip_offset),
            )
        ]
    )

    # Peg insert (rectangular) — discrete yaw symmetry
    peg_insert_4mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]),
                start_pose=kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
                ),
            )
        ]
    )
    peg_insert_8mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
                ),
            )
        ]
    )
    peg_insert_12mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
                ),
            )
        ]
    )
    peg_insert_16mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset
                ),
            )
        ]
    )

    # Connector insert — pure linear, no yaw noise (keyed)
    usba: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.USB_A_SOCKET.plug_assembled,
                distance=_dist(kpts.USB_A_SOCKET.plug_assembled, kpts.USB_A_SOCKET.entry),
            )
        ]
    )
    waterproof: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.WATERPROOF_SOCKET.plug_assembled,
                distance=_dist(kpts.WATERPROOF_SOCKET.plug_assembled, kpts.WATERPROOF_SOCKET.entry),
            )
        ]
    )
    dsub: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.D_SUB_SOCKET.plug_assembled,
                distance=_dist(kpts.D_SUB_SOCKET.plug_assembled, kpts.D_SUB_SOCKET.entry),
            )
        ]
    )
    rj45: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.RJ45_SOCKET.plug_assembled,
                distance=_dist(kpts.RJ45_SOCKET.plug_assembled, kpts.RJ45_SOCKET.entry),
            )
        ]
    )

    # BNC — linear insertion then bayonet twist; endpoint rotations already encode the quarter-turn.
    bnc: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            EndPointsSegmentCfg(
                fraction=(0.0, 0.4),
                start_pose=kpts.BNC_SOCKET.plug_assembled,
                end_pose=kpts.BNC_SOCKET.insert_start,
            ),
            EndPointsSegmentCfg(
                fraction=(0.4, 1.0),
                start_pose=kpts.BNC_SOCKET.insert_start,
                end_pose=kpts.BNC_SOCKET.entry,
            ),
        ]
    )

    default: AssemblyProfileCfg = nut_thread_m16


@configclass
class HeldAssetAlignOffsetCfg(PresetCfg):
    # Nut threading — bottom of nut center axis for alignment
    nut_thread_m4: kpts.Offset = kpts.NUT_M4.center_axis_bottom
    nut_thread_m8: kpts.Offset = kpts.NUT_M8.center_axis_bottom
    nut_thread_m12: kpts.Offset = kpts.NUT_M12.center_axis_bottom
    nut_thread_m16: kpts.Offset = kpts.NUT_M16.center_axis_bottom

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.SMALL_GEAR.center_axis_bottom
    gear_mesh_medium: kpts.Offset = kpts.MEDIUM_GEAR.center_axis_bottom
    gear_mesh_large: kpts.Offset = kpts.LARGE_GEAR.center_axis_bottom

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.ROD_4MM.center_axis_bottom
    rod_insert_8mm: kpts.Offset = kpts.ROD_8MM.center_axis_bottom
    rod_insert_12mm: kpts.Offset = kpts.ROD_12MM.center_axis_bottom
    rod_insert_16mm: kpts.Offset = kpts.ROD_16MM.center_axis_bottom

    # Peg insert (rectangular) — peg tip is the alignment reference
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_PEG_4MM.peg_tip
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_PEG_8MM.peg_tip
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_PEG_12MM.peg_tip
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_PEG_16MM.peg_tip

    # Connector insert — insertion tip of the plug
    usba: kpts.Offset = kpts.USB_A_PLUG.insertion_tip
    waterproof: kpts.Offset = kpts.WATERPROOF_PLUG.insertion_tip
    bnc: kpts.Offset = kpts.BNC_PLUG.insertion_tip
    dsub: kpts.Offset = kpts.D_SUB_PLUG.insertion_tip
    rj45: kpts.Offset = kpts.RJ45_PLUG.insertion_tip

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspPointCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.NUT_M4.grasp_point
    nut_thread_m8: kpts.Offset = kpts.NUT_M8.grasp_point
    nut_thread_m12: kpts.Offset = kpts.NUT_M12.grasp_point
    nut_thread_m16: kpts.Offset = kpts.NUT_M16.grasp_point

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.SMALL_GEAR.grasp_point
    gear_mesh_medium: kpts.Offset = kpts.MEDIUM_GEAR.grasp_point
    gear_mesh_large: kpts.Offset = kpts.LARGE_GEAR.grasp_point

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.ROD_4MM.grasp_point
    rod_insert_8mm: kpts.Offset = kpts.ROD_8MM.grasp_point
    rod_insert_12mm: kpts.Offset = kpts.ROD_12MM.grasp_point
    rod_insert_16mm: kpts.Offset = kpts.ROD_16MM.grasp_point

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_PEG_4MM.grasp_point
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_PEG_8MM.grasp_point
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_PEG_12MM.grasp_point
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_PEG_16MM.grasp_point

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_PLUG.grasp_point
    waterproof: kpts.Offset = kpts.WATERPROOF_PLUG.grasp_point
    bnc: kpts.Offset = kpts.BNC_PLUG.grasp_point
    dsub: kpts.Offset = kpts.D_SUB_PLUG.grasp_point
    rj45: kpts.Offset = kpts.RJ45_PLUG.grasp_point

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspDiameterCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: float = kpts.NUT_M4.grasp_diameter
    nut_thread_m8: float = kpts.NUT_M8.grasp_diameter
    nut_thread_m12: float = kpts.NUT_M12.grasp_diameter
    nut_thread_m16: float = kpts.NUT_M16.grasp_diameter

    # Gear mesh
    gear_mesh_small: float = kpts.SMALL_GEAR.grasp_diameter
    gear_mesh_medium: float = kpts.MEDIUM_GEAR.grasp_diameter
    gear_mesh_large: float = kpts.LARGE_GEAR.grasp_diameter

    # Rod insert (round)
    rod_insert_4mm: float = kpts.ROD_4MM.grasp_diameter
    rod_insert_8mm: float = kpts.ROD_8MM.grasp_diameter
    rod_insert_12mm: float = kpts.ROD_12MM.grasp_diameter
    rod_insert_16mm: float = kpts.ROD_16MM.grasp_diameter

    # Peg insert (rectangular)
    peg_insert_4mm: float = kpts.RECTANGULAR_PEG_4MM.grasp_diameter
    peg_insert_8mm: float = kpts.RECTANGULAR_PEG_8MM.grasp_diameter
    peg_insert_12mm: float = kpts.RECTANGULAR_PEG_12MM.grasp_diameter
    peg_insert_16mm: float = kpts.RECTANGULAR_PEG_16MM.grasp_diameter

    # Connector insert
    usba: float = kpts.USB_A_PLUG.grasp_diameter
    waterproof: float = kpts.WATERPROOF_PLUG.grasp_diameter
    bnc: float = kpts.BNC_PLUG.grasp_diameter
    dsub: float = kpts.D_SUB_PLUG.grasp_diameter
    rj45: float = kpts.RJ45_PLUG.grasp_diameter

    default: float = nut_thread_m16


@configclass
class HeldAssetGraspMiddleCfg(PresetCfg):
    """Offset used for positioning the EE around the held asset.

    For nut variants this is the center_axis_middle (grasp from above the threading axis),
    while for all other variants it is the grasp_point.
    """

    # Nut threading — center axis middle for threading approach
    nut_thread_m4: kpts.Offset = kpts.NUT_M4.center_axis_middle
    nut_thread_m8: kpts.Offset = kpts.NUT_M8.center_axis_middle
    nut_thread_m12: kpts.Offset = kpts.NUT_M12.center_axis_middle
    nut_thread_m16: kpts.Offset = kpts.NUT_M16.center_axis_middle

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.SMALL_GEAR.grasp_point
    gear_mesh_medium: kpts.Offset = kpts.MEDIUM_GEAR.grasp_point
    gear_mesh_large: kpts.Offset = kpts.LARGE_GEAR.grasp_point

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.ROD_4MM.grasp_point
    rod_insert_8mm: kpts.Offset = kpts.ROD_8MM.grasp_point
    rod_insert_12mm: kpts.Offset = kpts.ROD_12MM.grasp_point
    rod_insert_16mm: kpts.Offset = kpts.ROD_16MM.grasp_point

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_PEG_4MM.grasp_point
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_PEG_8MM.grasp_point
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_PEG_12MM.grasp_point
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_PEG_16MM.grasp_point

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_PLUG.grasp_point
    waterproof: kpts.Offset = kpts.WATERPROOF_PLUG.grasp_point
    bnc: kpts.Offset = kpts.BNC_PLUG.grasp_point
    dsub: kpts.Offset = kpts.D_SUB_PLUG.grasp_point
    rj45: kpts.Offset = kpts.RJ45_PLUG.grasp_point

    default: kpts.Offset = nut_thread_m16


def _end_effector_pose_range(
    grasp_height: tuple[float, float],
    x: tuple[float, float],
    y: tuple[float, float],
    z: tuple[float, float],
    roll: tuple[float, float],
    pitch: tuple[float, float],
    yaw: tuple[float, float],
) -> dict:
    """How far the end effector may be randomized away from the target, for one variant.

    ``z`` is measured from the bottom and the top of ``grasp_height`` [m], the stretch of the
    held asset the gripper can close on. The rest are measured from the target grasp pose.
    """
    lowest_grasp, highest_grasp = grasp_height
    return dict(
        x=x,
        y=y,
        z=(lowest_grasp + z[0], highest_grasp + z[1]),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


_WIDE_RANGE = dict(
    x=(-0.15, 0.15), y=(-0.15, 0.15), z=(-0.05, 0.1), roll=(0.0, 0.0), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09)
)

_CENTERED_RANGE = dict(
    x=(-0.025, 0.025), y=(-0.025, 0.025), z=(-0.025, 0.025), roll=(0.0, 0.0), pitch=(-0.25, 0.25), yaw=(-1.045, 1.045)
)


_NUT_GRASP_HEIGHT = (0.000, 0.035)
_GEAR_GRASP_HEIGHT = (0.035, 0.045)
_INSERT_GRASP_HEIGHT = (0.047, 0.057)


@configclass
class _GraspHeightCfg(PresetCfg):
    """Lowest and highest graspable point on the held asset [m], measured above the fixed asset tip.

    The one per-variant input to :func:`_end_effector_pose_range`, so both pose-range tables
    below are generated from it and a new assembly variant only has to be added once.
    """

    # Nut threading
    nut_thread_m4: tuple = _NUT_GRASP_HEIGHT
    nut_thread_m8: tuple = _NUT_GRASP_HEIGHT
    nut_thread_m12: tuple = _NUT_GRASP_HEIGHT
    nut_thread_m16: tuple = _NUT_GRASP_HEIGHT

    # Gear mesh
    gear_mesh_small: tuple = _GEAR_GRASP_HEIGHT
    gear_mesh_medium: tuple = _GEAR_GRASP_HEIGHT
    gear_mesh_large: tuple = _GEAR_GRASP_HEIGHT

    # Rod insert (round)
    rod_insert_4mm: tuple = _INSERT_GRASP_HEIGHT
    rod_insert_8mm: tuple = _INSERT_GRASP_HEIGHT
    rod_insert_12mm: tuple = _INSERT_GRASP_HEIGHT
    rod_insert_16mm: tuple = _INSERT_GRASP_HEIGHT

    # Peg insert (rectangular)
    peg_insert_4mm: tuple = _INSERT_GRASP_HEIGHT
    peg_insert_8mm: tuple = _INSERT_GRASP_HEIGHT
    peg_insert_12mm: tuple = _INSERT_GRASP_HEIGHT
    peg_insert_16mm: tuple = _INSERT_GRASP_HEIGHT

    # Connector insert
    usba: tuple = _INSERT_GRASP_HEIGHT
    waterproof: tuple = _INSERT_GRASP_HEIGHT
    bnc: tuple = _INSERT_GRASP_HEIGHT
    dsub: tuple = _INSERT_GRASP_HEIGHT
    rj45: tuple = _INSERT_GRASP_HEIGHT

    default: tuple = _NUT_GRASP_HEIGHT


_GRASP_HEIGHTS = _GraspHeightCfg()
_VARIANTS = tuple(field.name for field in fields(_GRASP_HEIGHTS))

GRASPED_POSE_RANGE = preset(
    **{name: _end_effector_pose_range(getattr(_GRASP_HEIGHTS, name), **_WIDE_RANGE) for name in _VARIANTS}
)
"""Pose range for the ``start_near_grasped`` reset strategy."""

GRASPED_POSE_RANGE_CENTERED = preset(
    **{name: _end_effector_pose_range(getattr(_GRASP_HEIGHTS, name), **_CENTERED_RANGE) for name in _VARIANTS}
)
"""Pose range for the ``grasped_near_goal`` reset strategy."""


_NUT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005),
    y=(-0.005, 0.005),
    z=(0.00, 0.035),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)
_GEAR_GRASPED_RANGE = dict(
    x=(-0.02, 0.02),
    y=(-0.02, 0.02),
    z=(0.035, 0.045),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)
_INSERT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005),
    y=(-0.005, 0.005),
    z=(0.047, 0.057),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)


@configclass
class GraspedPoseRangeCfg(PresetCfg):
    """Pose range for the ``start_grasped_then_assembled`` reset strategy."""

    nut_thread_m16: dict = _NUT_GRASPED_RANGE

    # Gear mesh
    gear_mesh_small: dict = _GEAR_GRASPED_RANGE
    gear_mesh_medium: dict = _GEAR_GRASPED_RANGE
    gear_mesh_large: dict = _GEAR_GRASPED_RANGE

    # Rod insert (round)
    rod_insert_4mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_8mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_12mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_16mm: dict = _INSERT_GRASPED_RANGE

    # Peg insert (rectangular)
    peg_insert_4mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_8mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_12mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_16mm: dict = _INSERT_GRASPED_RANGE

    default: dict = nut_thread_m16


@configclass
class ResetAssetsCfg(PresetCfg):
    """Scene entities the accumulator banks and restores for each assembly.

    Anything omitted keeps whatever pose the previous episode left it in. The gear variants
    put two spare gears on the board, and the board itself is re-placed every reset, so they
    have to ride along or they end up seated against a board that has since moved.
    """

    default: list = ["nistboard", "fixed_asset", "held_asset", "robot"]
    gear_mesh_small: list = ["nistboard", "fixed_asset", "held_asset", "robot", "medium_gear", "large_gear"]
    gear_mesh_medium: list = ["nistboard", "fixed_asset", "held_asset", "robot", "small_gear", "large_gear"]
    gear_mesh_large: list = ["nistboard", "fixed_asset", "held_asset", "robot", "small_gear", "medium_gear"]


@configclass
class HeldAssetObstaclesCfg(PresetCfg):
    """Entities the held asset is checked against before a reset state is accepted.

    A state that seats the held asset inside something not listed here is banked as valid.
    PhysX pushes such a state apart; MuJoCo/Newton diverges to NaN instead, so the spare
    gears have to be checked as well.
    """

    default: list = [
        SceneEntityCfg("fixed_asset"),
        SceneEntityCfg("robot"),
        SceneEntityCfg("nistboard"),
        SceneEntityCfg("table"),
    ]
    gear_mesh_small: list = default + [SceneEntityCfg("medium_gear"), SceneEntityCfg("large_gear")]
    gear_mesh_medium: list = default + [SceneEntityCfg("small_gear"), SceneEntityCfg("large_gear")]
    gear_mesh_large: list = default + [SceneEntityCfg("small_gear"), SceneEntityCfg("medium_gear")]


@configclass
class RobotObstaclesCfg(PresetCfg):
    """Entities the robot is checked against before a reset state is accepted.

    Same omission as :class:`HeldAssetObstaclesCfg`, on the arm rather than the part: a pose
    that reaches the gripper into one of the spare gears is banked as valid because those
    gears are not screened. It bites hardest on ``gear_mesh_medium``, whose shaft has a gear
    on either side of it.
    """

    default: list = [
        SceneEntityCfg("table"),
        SceneEntityCfg("nistboard"),
        SceneEntityCfg("fixed_asset"),
    ]
    gear_mesh_small: list = default + [SceneEntityCfg("medium_gear"), SceneEntityCfg("large_gear")]
    gear_mesh_medium: list = default + [SceneEntityCfg("small_gear"), SceneEntityCfg("large_gear")]
    gear_mesh_large: list = default + [SceneEntityCfg("small_gear"), SceneEntityCfg("medium_gear")]
