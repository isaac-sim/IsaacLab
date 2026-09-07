# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Digital Twin warehouse visuals for conveyor-Franka playback."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporter

from .conveyor_franka_env_cfg import (
    ConveyorFrankaEnvCfg,
    ConveyorFrankaSceneCfg,
    _spawn_shape_with_display_color,
)
from .conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_HALF_STRAIGHT,
    BELT_TOP_Z,
    BELT_TURN_RADIUS,
)

_CONVEYOR_ASSET_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors"
_A09_ASSET_PATH = f"{_CONVEYOR_ASSET_DIR}/ConveyorBelt_A09.usd"
_A12_ASSET_PATH = f"{_CONVEYOR_ASSET_DIR}/ConveyorBelt_A12.usd"
_THOR_TABLE_ASSET_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd"
_PACKING_TABLE_ASSET_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"
_PALLET_ASSET_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
_LOADED_PALLET_ASSET_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/o3dyn_pallet.usd"

# The A12 endpoints are 2.9922 m apart, its belt crown is 1.78053 m above the
# asset origin, and the lowest rendered point of both A09 and A12 is authored at
# z=0. Scale from those measured bounds so the asset feet sit on the global
# ground while the visual surface follows the existing task colliders.
_A12_ENDPOINT_SEPARATION = 2.9922
_ASSET_BELT_TOP_Z = 1.78053
_ASSET_LOWEST_Z = 0.0
_ASSET_XY_SCALE = 2.0 * BELT_TURN_RADIUS / _A12_ENDPOINT_SEPARATION
_GROUND_PLANE_Z = 0.0
# Preserve the assets' lateral/vertical proportions instead of stretching the
# supports to the original table height. The scaled belt crown then determines
# how far to elevate the policy workspace.
_ASSET_Z_SCALE = _ASSET_XY_SCALE
_ASSET_ROOT_Z = _GROUND_PLANE_Z - _ASSET_LOWEST_Z * _ASSET_Z_SCALE
_ASSET_BELT_WORLD_Z = _ASSET_ROOT_Z + _ASSET_BELT_TOP_Z * _ASSET_Z_SCALE
_WORKSPACE_ELEVATION = _ASSET_BELT_WORLD_Z - BELT_TOP_Z

# A09 is a 4 m straight. Its travel-axis scale is independent of the common
# lateral scale so one asset spans the task's complete 0.88 m straight run.
_A09_LENGTH = 4.0
_A09_X_SCALE = 2.0 * BELT_HALF_STRAIGHT / _A09_LENGTH

# The Thor table authors its mounting surface at local z=0 and its lowest foot
# at z=-0.795 m. Uniformly scaling that distance to the elevated robot base
# puts every foot on the global ground without distorting the table.
_THOR_TABLE_LOWEST_Z = -0.795
_THOR_TABLE_SCALE = _WORKSPACE_ELEVATION / -_THOR_TABLE_LOWEST_Z

_BACKDROP_COLOR = (0.075, 0.09, 0.12)
_BACKDROP_ACCENT_COLOR = (0.16, 0.20, 0.25)
_SAFETY_YELLOW = (0.95, 0.58, 0.055)

_PHYSICS_SCHEMA_PREFIXES = ("Physics", "Physx", "Newton", "Mujoco")
_PHYSICS_SCHEMA_NAMES = frozenset(("IsaacConveyorAPI",))


def _is_physics_schema(schema_name: str) -> bool:
    """Return whether an applied schema can add physics ownership to a visual asset."""
    return schema_name in _PHYSICS_SCHEMA_NAMES or schema_name.startswith(_PHYSICS_SCHEMA_PREFIXES)


def _make_usd_subtree_visual_only(root_prim) -> None:
    """Author a render-only override for a referenced USD subtree.

    The source asset remains untouched. All edits are stronger opinions in the
    task stage and fail closed if a physics schema cannot be removed.
    """
    # Import USD only when a stage exists. Besides respecting Kit startup
    # ordering, this keeps import-light task discovery kitless.
    from pxr import Sdf, Usd, UsdPhysics

    children = tuple(Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()))
    instance_proxies = tuple(str(child.GetPath()) for child in children if child.IsInstanceProxy())
    if instance_proxies:
        raise RuntimeError(
            "Visual-only USD overrides require editable descendants; set make_uninstanceable=True. "
            f"Found instance proxies below {root_prim.GetPath()}: {instance_proxies[:3]}"
        )

    with Sdf.ChangeBlock():
        for child in children:
            if child.GetTypeName().startswith("OmniGraph") or child.IsA(UsdPhysics.Scene):
                child.SetActive(False)

        for child in children:
            if not child.IsValid() or not child.IsActive():
                continue
            if child.IsA(UsdPhysics.Joint):
                child.SetActive(False)
                continue
            for schema_name in tuple(child.GetAppliedSchemas()):
                if _is_physics_schema(schema_name) and not child.RemoveAppliedSchema(schema_name):
                    raise RuntimeError(f"Failed to remove physics schema {schema_name!r} from {child.GetPath()}.")

    remaining = {
        str(child.GetPath()): tuple(schema for schema in child.GetAppliedSchemas() if _is_physics_schema(schema))
        for child in children
        if child.IsValid() and child.IsActive()
    }
    remaining = {path: schemas for path, schemas in remaining.items() if schemas}
    if remaining:
        raise RuntimeError(f"Visual-only USD subtree still contains physics schemas: {remaining}")


@sim_utils.clone
def _spawn_visual_only_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD asset and strip its authored physics metadata."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    # Presentation assets may contain nested dynamic props (the packing table,
    # for example, carries an authored container rigid body). Merely disabling
    # collision still exposes those bodies and joints to a backend parser.
    # Author local API deletions so the entire referenced hierarchy is a pure
    # render layer and cannot alter either Newton's model or PhysX's scene.
    _make_usd_subtree_visual_only(prim)
    return prim


@configclass
class _VisualOnlyUsdFileCfg(sim_utils.UsdFileCfg):
    """USD reference whose composed subtree is guaranteed to remain render-only."""

    func: Callable = _spawn_visual_only_usd
    # Recursive schema overrides cannot be authored on USD instance proxies.
    make_uninstanceable: bool = True


@configclass
class _ElevatedGroundPlaneCfg(TerrainImporterCfg):
    """Ground plane that reports the elevated workspace as its environment origin."""

    class_type: type[TerrainImporter] | str = (
        "{DIR}.conveyor_franka_asset_terrain:ConveyorFrankaGroundPlaneTerrainImporter"
    )
    workspace_origin_offset: tuple[float, float, float] = (0.0, 0.0, _WORKSPACE_ELEVATION)
    """Translation from clone-grid origins to policy workspaces [m]."""


def _visual_usd_asset(
    prim_path: str,
    usd_path: str,
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> AssetBaseCfg:
    """Build one visual asset whose authored colliders are explicitly disabled."""
    spawn = _VisualOnlyUsdFileCfg(
        usd_path=usd_path,
        scale=scale,
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=rotation),
        # Physics remains owned by the task's lightweight, hidden belt and rail
        # proxies. The spawn callback strips authored physics metadata to keep
        # the visual geometry entirely non-authoritative.
        spawn=spawn,
    )


def _visual_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    roughness: float = 0.72,
    metallic: float = 0.0,
) -> AssetBaseCfg:
    """Build one non-colliding scene-dressing cuboid."""
    spawn = sim_utils.CuboidCfg(
        func=_spawn_shape_with_display_color,
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        ),
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=spawn,
    )


@configclass
class ConveyorFrankaA09A12SceneCfg(ConveyorFrankaSceneCfg):
    """Checkpoint-compatible Digital Twin scene with visual warehouse dressing."""

    def __post_init__(self) -> None:
        """Replace procedural visuals while retaining the validated physics proxies."""
        super().__post_init__()

        # Raise every inherited env-scoped task component as one rigid
        # workspace. The environment reports the same offset as its origin, so
        # observations, resets, rewards, and the pretrained policy retain their
        # original local coordinates.
        for asset in vars(self).values():
            if isinstance(asset, AssetBaseCfg) and asset.prim_path.startswith("{ENV_REGEX_NS}/"):
                x, y, z = asset.init_state.pos
                asset.init_state.pos = (x, y, z + _WORKSPACE_ELEVATION)

        # The ground is global rather than env-scoped and remains at the USD
        # convention's default elevation.
        self.ground = _ElevatedGroundPlaneCfg(
            prim_path="/World/GroundPlane",
            terrain_type="plane",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.055, 0.065, 0.082), roughness=0.82),
        )
        self.dome_light.spawn.color = (0.70, 0.78, 0.92)
        self.dome_light.spawn.intensity = 1850.0

        left_x = BELT_CENTER_X - BELT_HALF_STRAIGHT
        right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
        straight_scale = (_A09_X_SCALE, _ASSET_XY_SCALE, _ASSET_Z_SCALE)
        turn_scale = (_ASSET_XY_SCALE, _ASSET_XY_SCALE, _ASSET_Z_SCALE)

        # The Franka is fixed at the elevated workspace origin and does not need
        # support collision. Replace the temporary plinth with the purpose-built
        # Thor table: its mount stays at the robot base and its feet land on z=0.
        self.tabletop = _visual_usd_asset(
            prim_path="{ENV_REGEX_NS}/RobotThorTableVisual",
            usd_path=_THOR_TABLE_ASSET_PATH,
            position=(0.0, 0.0, _WORKSPACE_ELEVATION),
            scale=(_THOR_TABLE_SCALE,) * 3,
        )
        self.table_pedestal = None

        for side in ("Left", "Right"):
            side_key = side.lower()
            center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y

            # The Digital Twin pieces already render their belt, frame, and
            # guides, so remove only the procedural render geometry. Hidden
            # belt and guide collision assets created above stay authoritative.
            setattr(self, f"conveyor_{side_key}_belt_visual", None)
            setattr(self, f"guard_{side_key}_inner_visual", None)
            setattr(self, f"guard_{side_key}_outer_visual", None)

            for run, y_position in (
                ("top", center_y + BELT_TURN_RADIUS),
                ("bottom", center_y - BELT_TURN_RADIUS),
            ):
                setattr(
                    self,
                    f"conveyor_{side_key}_{run}_a09_visual",
                    _visual_usd_asset(
                        prim_path=f"{{ENV_REGEX_NS}}/Conveyor{side}{run.title()}A09Visual",
                        usd_path=_A09_ASSET_PATH,
                        position=(right_x, y_position, _ASSET_ROOT_Z),
                        scale=straight_scale,
                    ),
                )

            # A12 starts at one end of its diameter and bends toward local +X.
            # The right piece uses its authored orientation. Rotating the left
            # piece by 180 degrees produces the opposite semicircle without a
            # negative scale or mirrored geometry.
            setattr(
                self,
                f"conveyor_{side_key}_right_a12_visual",
                _visual_usd_asset(
                    prim_path=f"{{ENV_REGEX_NS}}/Conveyor{side}RightA12Visual",
                    usd_path=_A12_ASSET_PATH,
                    position=(right_x, center_y + BELT_TURN_RADIUS, _ASSET_ROOT_Z),
                    scale=turn_scale,
                ),
            )
            setattr(
                self,
                f"conveyor_{side_key}_left_a12_visual",
                _visual_usd_asset(
                    prim_path=f"{{ENV_REGEX_NS}}/Conveyor{side}LeftA12Visual",
                    usd_path=_A12_ASSET_PATH,
                    position=(left_x, center_y - BELT_TURN_RADIUS, _ASSET_ROOT_Z),
                    scale=turn_scale,
                    rotation=(0.0, 0.0, 1.0, 0.0),
                ),
            )

        # Warehouse props provide scale and context but are intentionally
        # presentation-only. Their placement stays behind the robot and outside
        # the manipulation workspace.
        self.packing_station_visual = _visual_usd_asset(
            prim_path="{ENV_REGEX_NS}/PackingStationVisual",
            usd_path=_PACKING_TABLE_ASSET_PATH,
            position=(-1.05, -1.42, _GROUND_PLANE_Z),
            scale=(0.42, 0.42, 0.42),
            rotation=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        self.loaded_pallet_visual = _visual_usd_asset(
            prim_path="{ENV_REGEX_NS}/LoadedPalletVisual",
            usd_path=_LOADED_PALLET_ASSET_PATH,
            position=(-1.08, 1.22, _GROUND_PLANE_Z),
            scale=(0.56, 0.56, 0.56),
            rotation=(0.0, 0.0, -0.25881905, 0.96592583),
        )
        self.empty_pallet_visual = _visual_usd_asset(
            prim_path="{ENV_REGEX_NS}/EmptyPalletVisual",
            usd_path=_PALLET_ASSET_PATH,
            position=(0.00, 1.72, _GROUND_PLANE_Z),
            scale=(0.58, 0.58, 0.58),
            rotation=(0.0, 0.0, 0.13052619, 0.99144486),
        )

        # A low-detail wall and safety-zone markings frame the high-detail USD
        # assets without importing a full warehouse stage or adding collision.
        self.warehouse_back_wall_visual = _visual_cuboid(
            prim_path="{ENV_REGEX_NS}/WarehouseBackWallVisual",
            size=(0.06, 4.4, 2.0),
            position=(-1.72, 0.0, 1.0),
            color=_BACKDROP_COLOR,
            roughness=0.82,
        )
        self.warehouse_side_wall_visual = _visual_cuboid(
            prim_path="{ENV_REGEX_NS}/WarehouseSideWallVisual",
            size=(5.3, 0.06, 2.0),
            position=(0.93, 2.18, 1.0),
            color=_BACKDROP_COLOR,
            roughness=0.82,
        )
        for index, y_position in enumerate((-1.75, -0.58, 0.58, 1.75)):
            setattr(
                self,
                f"warehouse_wall_column_{index}_visual",
                _visual_cuboid(
                    prim_path=f"{{ENV_REGEX_NS}}/WarehouseWallColumn{index}Visual",
                    size=(0.09, 0.08, 2.08),
                    position=(-1.66, y_position, 1.04),
                    color=_BACKDROP_ACCENT_COLOR,
                    roughness=0.55,
                    metallic=0.35,
                ),
            )
        for index, x_position in enumerate((-1.62, -0.48, 0.66, 1.80, 2.94)):
            setattr(
                self,
                f"warehouse_side_column_{index}_visual",
                _visual_cuboid(
                    prim_path=f"{{ENV_REGEX_NS}}/WarehouseSideColumn{index}Visual",
                    size=(0.08, 0.09, 2.08),
                    position=(x_position, 2.12, 1.04),
                    color=_BACKDROP_ACCENT_COLOR,
                    roughness=0.55,
                    metallic=0.35,
                ),
            )
        for index, (size, position) in enumerate(
            (
                ((1.85, 0.025, 0.004), (0.48, 1.02, 0.002)),
                ((1.85, 0.025, 0.004), (0.48, -1.02, 0.002)),
                ((0.025, 2.065, 0.004), (-0.445, 0.0, 0.002)),
                ((0.025, 2.065, 0.004), (1.405, 0.0, 0.002)),
            )
        ):
            setattr(
                self,
                f"safety_zone_{index}_visual",
                _visual_cuboid(
                    prim_path=f"{{ENV_REGEX_NS}}/SafetyZone{index}Visual",
                    size=size,
                    position=position,
                    color=_SAFETY_YELLOW,
                    roughness=0.68,
                ),
            )


@configclass
class ConveyorFrankaA09A12EnvCfg(ConveyorFrankaEnvCfg):
    """Newton presentation variant with Digital Twin visuals and unchanged task physics."""

    scene: ConveyorFrankaA09A12SceneCfg = ConveyorFrankaA09A12SceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
    )

    def __post_init__(self) -> None:
        """Frame the complete presentation scene while retaining all task settings."""
        super().__post_init__()
        self.sim.default_visualizer_cfg.eye = (4.10, -3.65, 2.35)
        self.sim.default_visualizer_cfg.lookat = (0.80, 0.0, 0.38)
