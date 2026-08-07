# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PresetCfg definitions for Factory task-specific parameters.

Each preset maps task variant names to the corresponding keypoint-derived values.
The ``default`` field selects which variant is active when no CLI override is given.

Task categories (12 total):
  - ``nut_thread_m16`` — nut threading
  - ``gear_mesh_{small,medium,large}`` — gear meshing
  - ``rod_insert_{4,8,12,16}mm`` — round rod insertion
  - ``peg_insert_{4,8,12,16}mm`` — rectangular peg insertion

Robot-specific presets (``EndEffectorBodyCfg``, ``JointEffortNamesCfg``) map robot
variant names (``franka``) to body/joint name strings.
"""

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import assembly_keypoints as kpts
from .assembly_profile_cfg import (
    AssemblyProfileCfg,
    DiscreteYawCfg,
    IncrementalSegmentCfg,
    UniformPoseNoiseCfg,
    UniformYawCfg,
)

# ---------------------------------------------------------------------------
# Robot-specific presets
# ---------------------------------------------------------------------------


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full :class:`ArticulationCfg` for the robot asset at ``scene.robot``.

    Required -- no sensible default exists, so resolving without picking a
    robot preset leaves ``scene.robot`` as :data:`MISSING`.
    """

    default: ArticulationCfg = MISSING  # type: ignore[assignment]


@configclass
class RobotActionsCfg(PresetCfg):
    """Action term group bound to the active robot.

    Each robot preset assigns an :func:`isaaclab.utils.configclass`-decorated
    actions cfg (typically with ``arm_action`` and ``gripper_action`` fields)
    to a class attribute named after the robot.
    """

    default: object = MISSING  # type: ignore[assignment]


@configclass
class EndEffectorBodyCfg(PresetCfg):
    """End-effector body name per robot variant."""

    default: str = "end_effector"


@configclass
class GripperJointNamesCfg(PresetCfg):
    """Joint name regex for gripper/finger joints per robot variant."""

    default: list[str] | None = None


@configclass
class IKJointNamesCfg(PresetCfg):
    """Joint name regex used by the IK solver in reset strategies."""

    default: list[str] | None = None


@configclass
class GripperGraspOffsetCfg(PresetCfg):
    """Gripper grasp frame offset relative to the end-effector link per robot variant."""

    default: kpts.Offset = kpts.Offset()


@configclass
class JointEffortNamesCfg(PresetCfg):
    """Joint name regex for the effort penalty per robot variant."""

    default: str | None = None


# ---------------------------------------------------------------------------
# Task-specific presets
# ---------------------------------------------------------------------------


@configclass
class FixedAssetMapCfg(PresetCfg):
    """Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name."""

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

    default: dict = nut_thread_m16


@configclass
class HeldAssetTipCfg(PresetCfg):
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

    default: kpts.Offset = nut_thread_m16


@configclass
class FixedAssetTipCfg(PresetCfg):
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

    nut_thread_m16: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M16.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M16.fully_screwed_nut_offset, kpts.BOLT_M16.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M16.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )

    # Gear mesh — fraction starts at 0.1 so teeth are clear before yaw noise
    gear_mesh_small: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.1, 1.0),
                start_sampler=UniformYawCfg(),
                start_pose=kpts.GEAR_BASE.small_gear_assembled_bottom_offset,
                distance=_dist(kpts.GEAR_BASE.small_gear_assembled_bottom_offset, kpts.GEAR_BASE.small_gear_tip_offset),
            )
        ]
    )
    gear_mesh_medium: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.1, 1.0),
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
                fraction=(0.1, 1.0),
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
                fraction=(0.0, 0.7),
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]),
                start_pose=kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_4MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_8mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_8MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_8MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_8MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_12mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_12MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_12MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_12MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_16mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi]),
                start_pose=kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_16MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_16MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_16MM.above_hole_tip_offset,
                ),
            ),
        ]
    )

    default: AssemblyProfileCfg = nut_thread_m16


@configclass
class HeldAssetAlignOffsetCfg(PresetCfg):
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

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspPointCfg(PresetCfg):
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

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspDiameterCfg(PresetCfg):
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

    default: float = nut_thread_m16


@configclass
class HeldAssetGraspMiddleCfg(PresetCfg):
    """Offset used for positioning the EE around the held asset.

    For nut variants this is the center_axis_middle (grasp from above the threading axis),
    while for all other variants it is the grasp_point.
    """

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

    default: kpts.Offset = nut_thread_m16


# Pose ranges reused across size variants within each category
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
