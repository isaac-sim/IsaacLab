# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact parcel induction, suction sorting, and five physical collection lanes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec

from ..conveyor_franka.conveyor_franka_env_cfg import _ROBOT_INITIAL_STATE, _collision_properties, _conveyor_asset
from ..conveyor_franka.conveyor_geometry import ConveyorSectionSpec, CuboidSpec, MeshSpec, _prism_faces
from ..conveyor_franka.franka_robot_cfg import FRANKA_PANDA_CONVEYOR_CFG

if TYPE_CHECKING:
    from pxr import Usd

BELT_TOP_Z = 0.04
INTAKE_CENTER = (0.95, -0.47)
INTAKE_RADIUS = 0.28
INTAKE_HALF_STRAIGHT = 0.45
INTAKE_WIDTH = 0.24
FEEDER_POSITION = (2.40, -0.75, 0.42)
PARCEL_DIMENSIONS = (
    (0.04, 0.04, 0.04),
    (0.065, 0.045, 0.05),
    (0.085, 0.06, 0.06),
    (0.11, 0.075, 0.07),
    (0.14, 0.10, 0.08),
)
"""Nominal dimensions [m] of the five increasing parcel-volume classes."""
PARCEL_CLASS_IDS = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 3)
PARCEL_SIZES = (
    *PARCEL_DIMENSIONS,
    (0.045, 0.035, 0.038),
    (0.060, 0.050, 0.055),
    (0.090, 0.055, 0.065),
    (0.105, 0.080, 0.075),
    (0.150, 0.090, 0.085),
    (0.038, 0.045, 0.042),
    (0.115, 0.070, 0.065),
)
"""Per-asset cuboid dimensions [m]; shapes vary within a size class."""
CLASS_COLORS = ((0.20, 0.53, 0.79), (0.88, 0.48, 0.19), (0.26, 0.66, 0.46), (0.68, 0.39, 0.66), (0.74, 0.60, 0.35))
GOAL_ANGLES = tuple(math.radians(value) for value in (10, 35, 60, 85, 110))
GOAL_DIRECTIONS = tuple((math.cos(angle), math.sin(angle), 0.0) for angle in GOAL_ANGLES)
GOAL_POSITIONS = tuple((0.62 * x, 0.62 * y, 0.10) for x, y, _ in GOAL_DIRECTIONS)
"""Sloped receiving-deck positions in the robot's environment frame [m]."""
BIN_POSITIONS = tuple((2.17 * x, 2.17 * y, -0.48) for x, y, _ in GOAL_DIRECTIONS)
BIN_INTERIOR_SIZE = (0.60, 0.56, 0.56)
"""Bin interior dimensions [m], with local X pointing along its outfeed."""
_PARCEL_XY = (
    (0.60, -0.19),
    (0.90, -0.19),
    (1.20, -0.19),
    (1.598, -0.272),
    (1.68, -0.47),
    (1.598, -0.668),
    (1.20, -0.75),
    (0.90, -0.75),
    (0.60, -0.75),
    (0.302, -0.668),
    (0.22, -0.47),
)
PARCEL_POSITIONS = tuple(
    (x, y, BELT_TOP_Z + 0.5 * size[2] + 0.002) for (x, y), size in zip(_PARCEL_XY, PARCEL_SIZES[:-1], strict=True)
) + (FEEDER_POSITION,)
_BELT_COLOR = (0.085, 0.105, 0.12)
_FRAME_COLOR = (0.13, 0.28, 0.40)
_METAL_COLOR = (0.62, 0.66, 0.70)


@configclass
class SuctionSortingSceneCfg(InteractiveSceneCfg):
    """Table-mounted Franka between a parcel loop and five compact sorting lanes."""

    robot = FRANKA_PANDA_CONVEYOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_ROBOT_INITIAL_STATE,
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.80)),
        spawn=sim_utils.GroundPlaneCfg(color=(0.30, 0.32, 0.34)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.83, 0.89, 1.0), intensity=1100.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2798, -0.3647, -0.1159, 0.8805)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.93, 0.84), intensity=1800.0, angle=1.0),
    )

    def __post_init__(self) -> None:
        """Author small native primitives; no warehouse asset download is required."""
        super().__post_init__()
        for index, (size, position, class_id) in enumerate(
            zip(PARCEL_SIZES, PARCEL_POSITIONS, PARCEL_CLASS_IDS, strict=True)
        ):
            setattr(
                self,
                f"parcel_{index}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Cube{index}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                        mass_props=sim_utils.MassCfg(mass=max(0.03, 180.0 * math.prod(size))),
                        collision_props=_collision_properties(contact_margin=0.002),
                        physics_material=NewtonMaterialPropertiesCfg(static_friction=0.65, dynamic_friction=0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=CLASS_COLORS[class_id], roughness=0.75
                        ),
                    ),
                ),
            )
        for section in _conveyor_sections():
            name = section.geometry.name
            setattr(
                self,
                name,
                _conveyor_asset(
                    f"{{ENV_REGEX_NS}}/{name}",
                    section.geometry,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_BELT_COLOR, roughness=0.85),
                ),
            )
        _add_intake_frame(self)
        for index, angle in enumerate(GOAL_ANGLES):
            _add_destination(self, index, angle)
        _add_box(self, "RobotMount", (0.36, 0.36, 0.05), (0.0, 0.0, -0.025), _METAL_COLOR)
        _add_box(self, "RobotPedestal", (0.24, 0.24, 0.73), (0.0, 0.0, -0.415), _FRAME_COLOR)
        _add_box(self, "BackWall", (0.08, 5.9, 3.0), (3.05, 0.25, 0.70), (0.74, 0.77, 0.79), colliding=False)
        _add_box(self, "SideWall", (6.1, 0.08, 3.0), (0.0, 2.9, 0.70), (0.80, 0.82, 0.83), colliding=False)
        for index in range(7):
            _add_box(
                self,
                f"FloorMark{index}",
                (0.14, 0.04, 0.003),
                (-0.45 + index * 0.16, -1.10, -0.797),
                (0.95, 0.73, 0.15),
                colliding=False,
            )

    def build_conveyor_belt_specs(self, **kwargs: float) -> tuple[SurfaceVelocitySpec, ...]:
        """Return prescribed surface velocities in the authored collision frames."""
        return tuple(section.belt for section in _conveyor_sections(**kwargs))


def _conveyor_sections(
    velocity: float = 0.25,
    friction_coefficient: float = 0.6,
    contact_threshold: float = 0.995,
) -> tuple[ConveyorSectionSpec, ...]:
    """Pair each carrying surface with its local-frame motion."""
    sections: list[ConveyorSectionSpec] = []

    def add(
        geometry: MeshSpec | CuboidSpec,
        direction: tuple[float, float, float],
        **options: bool | float | tuple[float, float, float],
    ) -> None:
        sections.append(
            ConveyorSectionSpec(
                geometry,
                SurfaceVelocitySpec(
                    prim_path=f"{{ENV_REGEX_NS}}/{geometry.name}",
                    velocity=velocity,
                    direction=direction,
                    friction_coefficient=friction_coefficient,
                    contact_threshold=contact_threshold,
                    **options,
                ),
            )
        )

    center_x, center_y = INTAKE_CENTER
    left_x, right_x = center_x - INTAKE_HALF_STRAIGHT, center_x + INTAKE_HALF_STRAIGHT
    for name, y, direction in (
        ("IntakeUpper", center_y + INTAKE_RADIUS, 1.0),
        ("IntakeLower", center_y - INTAKE_RADIUS, -1.0),
    ):
        add(
            CuboidSpec(name, (2.0 * INTAKE_HALF_STRAIGHT + 0.004, INTAKE_WIDTH, 0.04), (center_x, y, 0.02)),
            (direction, 0.0, 0.0),
        )
    for name, x, start in (("IntakeRight", right_x, -0.5 * math.pi), ("IntakeLeft", left_x, 0.5 * math.pi)):
        add(
            _arc(name, x, start, INTAKE_WIDTH, 0.0, 0.04),
            (0.0, 0.0, -1.0),
            curved=True,
            pivot_point=(x, center_y, 0.0),
            radius=INTAKE_RADIUS,
        )
    # A short drop onto the return leg avoids two opposing contact fields at the merge.
    add(CuboidSpec("InductionBelt", (1.20, 0.24, 0.04), (2.0, -0.75, 0.24)), (-1.0, 0.0, 0.0))
    for index, (angle, direction) in enumerate(zip(GOAL_ANGLES, GOAL_DIRECTIONS, strict=True)):
        x, y, _ = direction
        slope = 0.25
        add(
            _deck(f"ReceivingDeck{index}", angle, 0.55, 0.79, 0.20, 0.24, 0.1175, 0.0575),
            (x, y, -slope),
            surface_normal=(slope * x, slope * y, 1.0),
        )
        add(_deck(f"Outfeed{index}", angle, 0.775, 1.95, 0.24, 0.24, 0.04, 0.04), direction)
    return tuple(sections)


def _arc(
    name: str, center_x: float, start: float, width: float, bottom: float, top: float, offset: float = 0.0
) -> MeshSpec:
    """Make a closed half-annular prism with outward triangle winding."""
    angles = tuple(start - 0.004 + (math.pi + 0.008) * index / 48 for index in range(49))
    vertices = tuple(
        (center_x + radius * math.cos(angle), INTAKE_CENTER[1] + radius * math.sin(angle), z)
        for z in (top, bottom)
        for radius in (INTAKE_RADIUS + offset - width / 2.0, INTAKE_RADIUS + offset + width / 2.0)
        for angle in angles
    )
    return MeshSpec(name, vertices, _prism_faces(len(angles), closed=False))


def _deck(
    name: str,
    angle: float,
    start: float,
    end: float,
    width_start: float,
    width_end: float,
    z_start: float,
    z_end: float,
) -> MeshSpec:
    """Build a tapered receiving deck or flat outfeed in environment coordinates."""
    corners = (
        (start, -width_start / 2.0, z_start),
        (end, -width_end / 2.0, z_end),
        (end, width_end / 2.0, z_end),
        (start, width_start / 2.0, z_start),
    )
    cosine, sine = math.cos(angle), math.sin(angle)
    vertices = tuple(
        (x * cosine - y * sine, x * sine + y * cosine, z - depth) for depth in (0.0, 0.025) for x, y, z in corners
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    )
    return MeshSpec(name, vertices, faces)


def _add_box(
    scene: SuctionSortingSceneCfg,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    yaw: float = 0.0,
    *,
    colliding: bool = True,
) -> None:
    """Add one fixed native cuboid with material appropriate to a facility frame."""
    asset = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=_collision_properties() if colliding else None,
            physics_material=NewtonMaterialPropertiesCfg(static_friction=0.4, dynamic_friction=0.3)
            if colliding
            else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45, metallic=0.25),
        ),
    )
    setattr(scene, name, asset)


def _add_intake_frame(scene: SuctionSortingSceneCfg) -> None:
    """Support the racetrack and induction belt without obstructing parcel travel."""
    for lane, y in (("Upper", -0.19), ("Lower", -0.75)):
        for side, sign in (("NegativeY", -1), ("PositiveY", 1)):
            _add_box(
                scene,
                f"IntakeRail{lane}{side}",
                (0.90, 0.018, 0.045),
                (0.95, y + sign * 0.13, 0.045),
                _METAL_COLOR,
            )
        _add_box(
            scene,
            f"IntakeBeam{lane}",
            (0.93, 0.27, 0.08),
            (0.95, y, -0.045),
            _FRAME_COLOR,
            colliding=False,
        )
    for turn, x, start in (("Right", 1.40, -math.pi / 2.0), ("Left", 0.50, math.pi / 2.0)):
        for side, offset in (("Inner", -0.13), ("Outer", 0.13)):
            spec = _arc(f"IntakeRail{turn}{side}", x, start, 0.018, 0.025, 0.065, offset)
            setattr(
                scene,
                spec.name,
                _conveyor_asset(
                    f"{{ENV_REGEX_NS}}/{spec.name}",
                    spec,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_METAL_COLOR, metallic=0.65, roughness=0.3
                    ),
                ),
            )
    for index, (x, y) in enumerate(((0.50, -0.19), (1.40, -0.19), (0.50, -0.75), (1.40, -0.75))):
        _add_box(scene, f"IntakeLeg{index}", (0.055, 0.055, 0.72), (x, y, -0.44), _FRAME_COLOR, colliding=False)
    _add_box(scene, "InductionFrame", (1.20, 0.28, 0.06), (2.0, -0.75, 0.19), _FRAME_COLOR, colliding=False)
    for index, x in enumerate((1.92, 2.45)):
        _add_box(scene, f"InductionLeg{index}", (0.055, 0.055, 1.0), (x, -0.75, -0.30), _FRAME_COLOR, colliding=False)
    for index, y in enumerate((-0.90, -0.60)):
        _add_box(scene, f"ScannerPost{index}", (0.05, 0.045, 0.30), (2.40, y, 0.40), _FRAME_COLOR, colliding=False)
    _add_box(scene, "ScannerBridge", (0.08, 0.345, 0.05), (2.40, -0.75, 0.575), _FRAME_COLOR, colliding=False)
    _add_box(scene, "ScannerLight", (0.008, 0.24, 0.008), (2.36, -0.75, 0.545), (0.30, 0.85, 0.75), colliding=False)


def _add_destination(scene: SuctionSortingSceneCfg, index: int, angle: float) -> None:
    """Build an open receiving mouth, guarded outfeed, and open collection bin."""
    cosine, sine = math.cos(angle), math.sin(angle)

    def box(
        part: str,
        size: tuple[float, float, float],
        radius: float,
        offset: float,
        z: float,
        color: tuple[float, float, float],
        *,
        colliding: bool = True,
    ) -> None:
        _add_box(
            scene,
            f"Lane{index}{part}",
            size,
            (radius * cosine - offset * sine, radius * sine + offset * cosine, z),
            color,
            angle,
            colliding=colliding,
        )

    for side, offset in (("Left", -0.119), ("Right", 0.119)):
        _add_box(
            scene,
            f"Lane{index}Mouth{side}",
            (0.245, 0.018, 0.13),
            (0.67 * cosine - offset * sine, 0.67 * sine + offset * cosine, 0.125),
            CLASS_COLORS[index],
            angle + math.copysign(math.atan2(0.02, 0.24), offset),
        )
        box(f"Outfeed{side}", (1.20, 0.020, 0.045), 1.35, math.copysign(0.13, offset), 0.05, _METAL_COLOR)
    box("Frame", (1.20, 0.28, 0.08), 1.35, 0.0, -0.045, _FRAME_COLOR, colliding=False)
    for leg, radius in enumerate((0.92, 1.78)):
        box(f"Leg{leg}", (0.065, 0.24, 0.72), radius, 0.0, -0.44, _FRAME_COLOR, colliding=False)
    # Extend below ground to prevent small parcels from crossing the floor during impact.
    box("BinFloor", (0.64, 0.60, 0.20), 2.17, 0.0, -0.86, _FRAME_COLOR)
    for part, radius in (("BinFront", 1.86), ("BinBack", 2.48)):
        box(part, (0.02, 0.60, 0.56), radius, 0.0, -0.48, CLASS_COLORS[index])
    for part, offset in (("BinLeft", -0.29), ("BinRight", 0.29)):
        box(part, (0.60, 0.02, 0.56), 2.17, offset, -0.48, CLASS_COLORS[index])


def spawn_suction_franka(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Keep the existing arm asset and author a compact, hand-mounted suction tool."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    sim_utils.make_uninstanceable(str(prim.GetPath()))
    hand = next(p for p in Usd.PrimRange(prim) if p.GetName() == "panda_hand")
    for child in Usd.PrimRange(prim):
        if "finger" in child.GetPath().pathString:
            if child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).MakeInvisible()
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
    tool = str(hand.GetPath()) + "/SuctionTool"
    for name, position, size, color in (
        ("Mount", (0.0, 0.0, 0.065), (0.035, 0.035, 0.055), (0.18, 0.20, 0.23)),
        ("Plate", (0.0, 0.0, 0.096), (0.028, 0.028, 0.005), (0.62, 0.67, 0.72)),
    ):
        geometry = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        geometry.func(f"{tool}/{name}", geometry, translation=position)
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        cup = UsdGeom.Cylinder.Define(prim.GetStage(), f"{tool}/Cup{index}")
        cup.CreateRadiusAttr(0.004)
        cup.CreateHeightAttr(0.006)
        cup.AddTranslateOp().Set(Gf.Vec3d(x * 0.007, y * 0.007, 0.102))
        cup.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.55, 0.65)])
        UsdPhysics.CollisionAPI.Apply(cup.GetPrim()).CreateCollisionEnabledAttr(True)
    return prim
