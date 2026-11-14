# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circulating parcel intake and coherent five- or six-lane sorting fixtures."""

from __future__ import annotations

import math
import random
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg
from isaaclab.utils import configclass

from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec

from ..conveyor_franka.conveyor_franka_env_cfg import _ROBOT_INITIAL_STATE, _collision_properties, _conveyor_asset
from ..conveyor_franka.conveyor_geometry import ConveyorSectionSpec, CuboidSpec, MeshSpec, _prism_faces
from ..conveyor_franka.franka_robot_cfg import FRANKA_PANDA_CONVEYOR_CFG
from .actions_cfg import TOOL_OFFSET

if TYPE_CHECKING:
    from pxr import Usd

BELT_TOP_Z = 0.04
INTAKE_CENTER = (0.95, -0.47)
INTAKE_RADIUS = 0.28
INTAKE_HALF_STRAIGHT = 0.45
INTAKE_WIDTH = 0.24
INTAKE_LENGTH = 4.0 * INTAKE_HALF_STRAIGHT + 2.0 * math.pi * INTAKE_RADIUS
SUCTION_PAD_RADIUS = 0.012
SUCTION_PAD_SPACING = 0.014
RECEIVING_PAD_RADIAL_BOUNDS = (0.58, 0.83)
RECEIVING_PAD_HALF_WIDTH = 0.12
PARCEL_DIMENSIONS = (
    (0.09, 0.075, 0.055),
    (0.11, 0.085, 0.065),
    (0.135, 0.095, 0.075),
    (0.16, 0.11, 0.085),
    (0.18, 0.125, 0.10),
)
"""Nominal dimensions [m] of the five increasing parcel-volume classes."""
PARCEL_CLASS_IDS = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 3)
PARCEL_SIZES = (
    *PARCEL_DIMENSIONS,
    (0.095, 0.075, 0.05),
    (0.105, 0.09, 0.065),
    (0.14, 0.09, 0.08),
    (0.155, 0.115, 0.09),
    (0.19, 0.12, 0.105),
    (0.085, 0.08, 0.055),
    (0.165, 0.105, 0.085),
)
"""Per-asset cuboid dimensions [m]; shapes vary within a size class."""
CLASS_COLORS = ((0.20, 0.53, 0.79), (0.88, 0.48, 0.19), (0.26, 0.66, 0.46), (0.68, 0.39, 0.66), (0.74, 0.60, 0.35))
BIN_INTERIOR_SIZE = (0.60, 0.56, 0.56)
"""Bin interior dimensions [m], with local X pointing along its outfeed."""


@configclass
class OutfeedLayout:
    """Lane angles [rad], radial translations [m], and height translations [m]."""

    angles: tuple[float, ...] = MISSING
    radial_offsets: tuple[float, ...] = MISSING
    height_offsets: tuple[float, ...] = MISSING


def make_logistics_layouts(seed: int = 73, count: int = 32, *, held_out: bool = False) -> tuple[OutfeedLayout, ...]:
    """Build separated five/six-outlet cells, reserving outer angular bands for evaluation.

    Training covers front angles [-145, -105] and back angles [20, 140] degrees.
    Held-out fixtures put two outlets in [-160, -153] and [151, 158] degrees;
    their inward radial offsets retain the downward robot's Cartesian workspace.
    Geometry is selected at spawn and remains fixed throughout an environment's lifetime.
    """
    if count < 1:
        raise ValueError("The logistics fixture catalog must contain at least one layout.")
    generator = random.Random(seed)
    layouts = []
    for index in range(count):
        lane_count = 5 + index % 2
        if held_out:
            sectors = ((-160.0, -153.0, 1), (20.0, 115.0, lane_count - 2), (151.0, 158.0, 1))
        else:
            front_count = 1 if index % 4 == 0 else 2
            sectors = ((-145.0, -105.0, front_count), (20.0, 140.0, lane_count - front_count))
        angles = []
        for lower, upper, lanes in sectors:
            slack = upper - lower - 30.0 * (lanes - 1)
            offsets = sorted(generator.uniform(0.0, slack) for _ in range(lanes))
            angles.extend(math.radians(lower + 30.0 * lane + offset) for lane, offset in enumerate(offsets))
        radial_bounds = (-0.025, -0.015) if held_out else (-0.015, 0.015)
        layouts.append(
            OutfeedLayout(
                angles=tuple(angles),
                radial_offsets=tuple(generator.uniform(*radial_bounds) for _ in angles),
                height_offsets=tuple(generator.uniform(-0.01, 0.01) for _ in angles),
            )
        )
    return tuple(layouts)


OUTFEED_LAYOUTS = make_logistics_layouts()
"""Static fixture variants shared by physics, observations, commands, and reset recipes."""


def intake_positions(distance: torch.Tensor) -> torch.Tensor:
    """Map clockwise arc lengths [m] to XY positions on the racetrack centerline."""
    distance = distance.remainder(INTAKE_LENGTH)
    straight = 2.0 * INTAKE_HALF_STRAIGHT
    bend = math.pi * INTAKE_RADIUS
    right = ((distance - straight) / INTAKE_RADIUS).clamp(0.0, math.pi)
    left = ((distance - 2.0 * straight - bend) / INTAKE_RADIUS).clamp(0.0, math.pi)
    x = INTAKE_CENTER[0] - INTAKE_HALF_STRAIGHT + distance.clamp(max=straight)
    x += INTAKE_RADIUS * (right.sin() - left.sin()) - (distance - straight - bend).clamp(0.0, straight)
    y = INTAKE_CENTER[1] + INTAKE_RADIUS * (right.cos() + 1.0 - left.cos())
    return torch.stack((x, y), dim=-1)


PARCEL_POSITIONS = tuple(
    (x, y, BELT_TOP_Z + 0.5 * size[2] + 0.002)
    for (x, y), size in zip(
        intake_positions(torch.arange(len(PARCEL_SIZES), device="cpu") * (INTAKE_LENGTH / len(PARCEL_SIZES))).tolist(),
        PARCEL_SIZES,
        strict=True,
    )
)
_ASSETS_DIR = Path(__file__).with_name("assets")
_BELT_COLOR = (0.025, 0.032, 0.038)
_FRAME_COLOR = (0.065, 0.095, 0.12)
_METAL_COLOR = (0.42, 0.46, 0.50)
_LANE_COLORS = (*CLASS_COLORS, (0.83, 0.72, 0.20))


@sim_utils.clone
def _spawn_floor(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Cover the shared ground with one collision-free concrete surface."""
    del cfg
    return _spawn_textured_surface(prim_path, "concrete.png", (100.0, 100.0), translation, orientation, tiles=32.0)


def _spawn_outfeeds(
    prim_path: str,
    cfg: OutfeedsCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Author one complete fixture before clone planning imports its collision geometry."""
    prim = sim_utils.create_prim(prim_path, translation=translation, orientation=orientation)
    layout = cfg.layout
    for index, (angle, radial, height) in enumerate(
        zip(layout.angles, layout.radial_offsets, layout.height_offsets, strict=True)
    ):
        parts = SimpleNamespace()
        _add_destination(parts, index, angle)
        for section in _outfeed_sections(index):
            asset = _conveyor_asset(
                f"{{ENV_REGEX_NS}}/{section.geometry.name}",
                section.geometry,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_BELT_COLOR, roughness=0.85),
            )
            asset.init_state.rot = (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))
            setattr(parts, section.geometry.name, asset)
        offset = (radial * math.cos(angle), radial * math.sin(angle), height)
        for name, asset in vars(parts).items():
            # MJWarp permits absent mesh slots, while primitive counts must match across worlds.
            if isinstance(asset.spawn, sim_utils.CuboidCfg):
                original = asset.spawn
                asset.spawn = sim_utils.MeshCuboidCfg(
                    size=original.size,
                    collision_props=original.collision_props,
                    mesh_collision_props=UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")
                    if original.collision_props is not None
                    else None,
                    physics_material=original.physics_material,
                    visual_material=original.visual_material,
                )
            position = tuple(value + shift for value, shift in zip(asset.init_state.pos, offset, strict=True))
            asset.spawn.func(f"{prim_path}/{name}", asset.spawn, translation=position, orientation=asset.init_state.rot)
    return prim


@configclass
class OutfeedsCfg(sim_utils.SpawnerCfg):
    """Select one coherent static sorting fixture from the task's layout catalog."""

    func = _spawn_outfeeds
    layout: OutfeedLayout = OUTFEED_LAYOUTS[0]


@configclass
class SuctionSortingSceneCfg(InteractiveSceneCfg):
    """Table-mounted Franka between a parcel loop and a catalog of sorting fixtures."""

    robot = FRANKA_PANDA_CONVEYOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_ROBOT_INITIAL_STATE,
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.80)),
        spawn=sim_utils.GroundPlaneCfg(color=(0.30, 0.32, 0.34)),
    )
    floor_finish = AssetBaseCfg(
        prim_path="/World/ConcreteFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.7998)),
        spawn=sim_utils.SpawnerCfg(func=_spawn_floor),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.95, 0.97, 1.0), intensity=450.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2798, -0.3647, -0.1159, 0.8805)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.89), intensity=1400.0, angle=8.0),
    )
    outfeeds = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Outfeeds",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[OutfeedsCfg(layout=layout) for layout in OUTFEED_LAYOUTS],
            random_choice=False,
        ),
    )

    def __post_init__(self) -> None:
        """Author small native primitives; no warehouse asset download is required."""
        super().__post_init__()
        for index, (size, position) in enumerate(zip(PARCEL_SIZES, PARCEL_POSITIONS, strict=True)):
            setattr(
                self,
                f"parcel_{index}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Cube{index}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                    spawn=sim_utils.CuboidCfg(
                        func=_spawn_carton,
                        size=size,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                        mass_props=sim_utils.MassCfg(mass=max(0.03, 180.0 * math.prod(size))),
                        collision_props=_collision_properties(contact_margin=0.002),
                        physics_material=NewtonMaterialPropertiesCfg(static_friction=0.65, dynamic_friction=0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=CLASS_COLORS[PARCEL_CLASS_IDS[index]], roughness=0.9
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
        _add_box(self, "RobotMount", (0.36, 0.36, 0.05), (0.0, 0.0, -0.025), _METAL_COLOR)
        _add_box(self, "RobotPedestal", (0.24, 0.24, 0.73), (0.0, 0.0, -0.415), _FRAME_COLOR)
        _add_box(self, "RobotFoot", (0.36, 0.36, 0.025), (0.0, 0.0, -0.7875), _METAL_COLOR, colliding=False)
        _add_box(self, "BackWall", (0.08, 7.0, 4.8), (3.25, 0.0, 1.6), (0.43, 0.46, 0.48), colliding=False)
        _add_box(self, "SideWall", (6.6, 0.08, 4.8), (0.0, 3.25, 1.6), (0.48, 0.50, 0.51), colliding=False)
        for index, y in enumerate((-2.1, 0.0, 2.4)):
            _add_box(self, f"HallColumn{index}", (0.14, 0.14, 4.5), (3.13, y, 1.45), _METAL_COLOR, colliding=False)
        for index, x in enumerate((-1.8, -0.4, 1.0)):
            _add_box(self, f"RoofBeam{index}", (0.12, 6.5, 0.18), (x, 0.0, 3.4), _METAL_COLOR, colliding=False)
            _add_box(self, f"RackPost{index}", (0.055, 0.45, 3.0), (x, 2.88, 0.7), _FRAME_COLOR, colliding=False)
        for level, z in enumerate((0.15, 1.15, 2.15)):
            _add_box(
                self, f"RackShelf{level}", (2.92, 0.5, 0.065), (-0.4, 2.88, z), (0.48, 0.16, 0.035), colliding=False
            )
            for index, x in enumerate((-1.4, -0.45, 0.45)):
                _add_box(
                    self,
                    f"StoredCarton{level}{index}",
                    (0.58, 0.38, 0.45),
                    (x, 2.88, z + 0.2575),
                    (0.34, 0.21, 0.11),
                    colliding=False,
                )
        _add_box(self, "LoadingDoor", (0.02, 1.45, 2.3), (3.19, -1.1, 0.35), (0.18, 0.20, 0.22), colliding=False)
        for index in range(7):
            _add_box(
                self,
                f"DoorSlat{index}",
                (0.025, 1.42, 0.012),
                (3.17, -1.1, -0.65 + 0.3 * index),
                _METAL_COLOR,
                colliding=False,
            )
        for index, (x, y) in enumerate(((-2.8, -2.6), (-2.8, 2.6), (2.8, -2.6), (2.8, 2.6))):
            for axis, size, position in (
                ("X", (0.55, 0.04, 0.001), (x - math.copysign(0.255, x), y, -0.799)),
                ("Y", (0.04, 0.55, 0.001), (x, y - math.copysign(0.255, y), -0.799)),
            ):
                _add_box(self, f"CellCorner{index}{axis}", size, position, (0.75, 0.49, 0.045), colliding=False)

    def build_conveyor_belt_specs(self, **kwargs: float) -> tuple[SurfaceVelocitySpec, ...]:
        """Return prescribed surface velocities in the authored collision frames."""
        sections = (
            *_conveyor_sections(**kwargs),
            *(section for index in range(6) for section in _outfeed_sections(index, **kwargs)),
        )
        return tuple(section.belt for section in sections)


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
    return tuple(sections)


def _outfeed_sections(
    index: int,
    velocity: float = 0.25,
    friction_coefficient: float = 0.6,
    contact_threshold: float = 0.995,
) -> tuple[ConveyorSectionSpec, ...]:
    """Keep carrying surfaces and their velocities in the same lane-local frame."""
    sections = []
    for name, bounds in ((f"ReceivingDeck{index}", RECEIVING_PAD_RADIAL_BOUNDS), (f"Outfeed{index}", (0.815, 1.95))):
        sections.append(
            ConveyorSectionSpec(
                _flat_deck(name, *bounds),
                SurfaceVelocitySpec(
                    prim_path=f"{{ENV_REGEX_NS}}/Outfeeds/{name}",
                    velocity=velocity,
                    direction=(1.0, 0.0, 0.0),
                    friction_coefficient=friction_coefficient,
                    contact_threshold=contact_threshold,
                    optional=index == 5,
                ),
            )
        )
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


def _flat_deck(name: str, start: float, end: float) -> MeshSpec:
    """Build a flat 24 cm-wide receiving deck or outfeed in lane coordinates."""
    corners = (
        (start, -0.12),
        (end, -0.12),
        (end, 0.12),
        (start, 0.12),
    )
    vertices = tuple((x, y, BELT_TOP_Z - depth) for depth in (0.0, 0.025) for x, y in corners)
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
    scene: SuctionSortingSceneCfg | SimpleNamespace,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    yaw: float = 0.0,
    *,
    colliding: bool = True,
    metallic: float = 0.0,
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6, metallic=metallic),
        ),
    )
    setattr(scene, name, asset)


def _add_intake_frame(scene: SuctionSortingSceneCfg) -> None:
    """Support the racetrack without obstructing parcel travel."""
    for lane, y in (("Upper", -0.19), ("Lower", -0.75)):
        for side, sign in (("NegativeY", -1), ("PositiveY", 1)):
            _add_box(
                scene,
                f"IntakeRail{lane}{side}",
                (0.90, 0.018, 0.045),
                (0.95, y + sign * 0.13, 0.045),
                _METAL_COLOR,
                metallic=0.7,
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


def _add_destination(scene: SuctionSortingSceneCfg | SimpleNamespace, index: int, angle: float) -> None:
    """Build an open receiving mouth, guarded outfeed, and open collection bin."""
    cosine, sine = math.cos(angle), math.sin(angle)
    color = _LANE_COLORS[index]

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

    for side, offset in (("Left", -0.129), ("Right", 0.129)):
        box(f"Mouth{side}", (0.25, 0.018, 0.13), 0.705, offset, 0.125, color)
        box(f"Outfeed{side}", (1.20, 0.020, 0.045), 1.35, math.copysign(0.13, offset), 0.05, _METAL_COLOR)
        box(f"RouteStripe{side}", (1.12, 0.015, 0.002), 1.35, offset, 0.074, color, colliding=False)
        box(f"BinRim{side}", (0.60, 0.025, 0.025), 2.17, math.copysign(0.29, offset), -0.19, color, colliding=False)
    box("Frame", (1.20, 0.28, 0.08), 1.35, 0.0, -0.045, _FRAME_COLOR, colliding=False)
    for leg, radius in enumerate((0.92, 1.78)):
        box(f"Leg{leg}", (0.065, 0.065, 0.72), radius, 0.0, -0.44, _FRAME_COLOR, colliding=False)
    # Extend below ground to prevent small parcels from crossing the floor during impact.
    box("BinFloor", (0.64, 0.60, 0.20), 2.17, 0.0, -0.86, _FRAME_COLOR)
    for part, radius in (("BinFront", 1.86), ("BinBack", 2.48)):
        box(part, (0.02, 0.60, 0.56), radius, 0.0, -0.48, _FRAME_COLOR)
    for part, offset in (("BinLeft", -0.29), ("BinRight", 0.29)):
        box(part, (0.60, 0.02, 0.56), 2.17, offset, -0.48, _FRAME_COLOR)


def _spawn_textured_surface(
    prim_path: str,
    texture: str,
    size: tuple[float, float],
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    tiles: float = 1.0,
) -> Usd.Prim:
    """Author a render-only paper or concrete quad with one repeating local texture."""
    from pxr import Sdf, UsdGeom, UsdShade

    prim = sim_utils.create_prim(prim_path, translation=translation, orientation=orientation, scale=(*size, 1.0))
    stage = prim.GetStage()
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    mesh.CreatePointsAttr([(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        [(0.0, 0.0), (tiles, 0.0), (tiles, tiles), (0.0, tiles)]
    )
    material_path = f"{prim_path}/Material"
    shader = UsdShade.Shader(sim_utils.spawn_preview_surface(material_path, sim_utils.PreviewSurfaceCfg(roughness=0.9)))
    sim_utils.bind_visual_material(str(mesh.GetPath()), material_path)
    uv = UsdShade.Shader.Define(stage, f"{material_path}/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    image = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    image.CreateIdAttr("UsdUVTexture")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(_ASSETS_DIR / texture))
    image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    for axis in ("S", "T"):
        image.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
    image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(image.ConnectableAPI(), "rgb")
    return prim


@sim_utils.clone
def _spawn_carton(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Keep the physical cuboid and attach one paper-textured, collision-free top face."""
    prim = sim_utils.spawn_cuboid(
        prim_path,
        cfg.replace(visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.21, 0.11), roughness=0.9)),
        translation=translation,
        orientation=orientation,
    )
    _spawn_textured_surface(
        f"{prim_path}/CartonTop", "carton_top.png", cfg.size[:2], translation=(0.0, 0.0, cfg.size[2] / 2.0 + 0.0002)
    )
    sim_utils.spawn_cuboid(
        f"{prim_path}/SortLabel",
        sim_utils.CuboidCfg(size=(0.15 * cfg.size[0], 0.5 * cfg.size[1], 0.0004), visual_material=cfg.visual_material),
        translation=(0.35 * cfg.size[0], 0.0, cfg.size[2] / 2.0 + 0.0005),
    )
    return prim


def spawn_suction_franka(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Keep the existing arm asset and author a compact, hand-mounted suction tool."""
    from pxr import Usd, UsdGeom, UsdPhysics

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
        ("Mount", (0.0, 0.0, 0.05), (0.045, 0.045, 0.05), (0.08, 0.28, 0.36)),
        ("Plate", (0.0, 0.0, 0.079), (0.075, 0.075, 0.008), (0.22, 0.27, 0.31)),
    ):
        geometry = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            collision_props=_collision_properties(contact_margin=0.001),
        )
        geometry.func(f"{tool}/{name}", geometry, translation=position)
    for part, radius, height, z, color in (
        ("Stem", 0.008, 0.016, 0.091, (0.025, 0.035, 0.04)),
        ("Lip", SUCTION_PAD_RADIUS, 0.006, TOOL_OFFSET[2] - 0.003, (0.04, 0.48, 0.56)),
    ):
        geometry = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
            collision_props=_collision_properties(contact_margin=0.001) if part == "Lip" else None,
        )
        for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            geometry.func(
                f"{tool}/{part}{index}",
                geometry,
                translation=(x * SUCTION_PAD_SPACING, y * SUCTION_PAD_SPACING, z),
            )
    return prim
