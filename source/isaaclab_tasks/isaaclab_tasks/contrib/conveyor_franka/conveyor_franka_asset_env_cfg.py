# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Digital Twin warehouse visuals for conveyor-Franka playback."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import SurfaceVelocitySpec
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Sdf, Usd

    from isaaclab.terrains import TerrainImporter

from .conveyor_franka_env_cfg import (
    ConveyorFrankaEnvCfg,
    ConveyorFrankaSceneCfg,
    _hidden_collision_geometry,
    _hidden_collision_mesh,
    _spawn_shape_with_display_color,
)
from .conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_HALF_STRAIGHT,
    BELT_TOP_Z,
    BELT_TURN_RADIUS,
)
from .conveyor_warehouse_geometry import warehouse_belt_sections, warehouse_guard_meshes

_THOR_TABLE_ASSET_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd"

# The A12 endpoints are 2.9922 m apart, its belt crown is 1.78053 m above the
# asset origin. USD point overrides extend the legs to the floor while keeping
# the upper frames unchanged; the policy workspace follows the deck elevation.
_A12_ENDPOINT_SEPARATION = 2.9922
_ASSET_BELT_TOP_Z = 1.78053
_ASSET_XY_SCALE = 2.0 * BELT_TURN_RADIUS / _A12_ENDPOINT_SEPARATION
_CONVEYOR_SUPPORT_Z = 0.55
# Preserve the upper frames' proportions. The scaled belt crown determines
# how far to elevate the policy workspace.
_ASSET_Z_SCALE = _ASSET_XY_SCALE
_ASSET_ROOT_Z = _CONVEYOR_SUPPORT_Z
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


_PHYSICS_SCHEMA_PREFIXES = ("Physics", "Physx", "Newton", "Mujoco")
_PHYSICS_SCHEMA_NAMES = frozenset(("IsaacConveyorAPI",))
_PRESENTATION_ASSETS = Path(__file__).parent / "assets"


@cache
def _presentation_layer(usd_path: str) -> Sdf.Layer:
    """Resolve a local composition's remote assets without modifying its source layer."""
    from pxr import Sdf, UsdUtils

    source = Sdf.Layer.FindOrOpen(usd_path)
    layer = Sdf.Layer.CreateAnonymous(Path(usd_path).name)
    layer.TransferContent(source)
    resolved = {}
    for path in source.GetExternalReferences():
        if path.startswith("https://"):
            resolved[path] = retrieve_file_path(path)
        else:
            local_path = Sdf.ComputeAssetPathRelativeToLayer(source, path)
            resolved[path] = _presentation_layer(local_path).identifier
    UsdUtils.ModifyAssetPaths(
        layer,
        lambda path: resolved[path] if path in resolved else Sdf.ComputeAssetPathRelativeToLayer(source, path),
    )
    return layer


@sim_utils.clone
def _spawn_authored_visual(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Compose USD scenery with cached dependencies and no simulation ownership."""
    layer = _presentation_layer(cfg.usd_path)
    prim = sim_utils.create_prim(
        prim_path,
        translation=translation,
        orientation=orientation,
        scale=cfg.scale,
    )
    prim.GetReferences().AddReference(layer.identifier)
    sim_utils.make_uninstanceable(prim_path)
    _make_usd_subtree_visual_only(prim)
    return prim


@sim_utils.clone
def _spawn_carton_cube(
    prim_path: str,
    cfg: _ParcelCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Dress the original 40 mm collider with a centered, equally sized SimReady carton."""
    from pxr import UsdGeom

    prim = _spawn_shape_with_display_color(prim_path, cfg, translation, orientation, **kwargs)
    UsdGeom.Imageable(prim.GetStage().GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()
    visual_cfg = sim_utils.UsdFileCfg(usd_path=cfg.parcel_usd_path)
    _spawn_authored_visual(f"{prim_path}/CartonVisual", visual_cfg)
    return prim


@configclass
class _ParcelCuboidCfg(sim_utils.CuboidCfg):
    """Original task collider with a separately authored carton appearance."""

    parcel_usd_path: str = str(_PRESENTATION_ASSETS / "parcel.usda")
    """USD visual, normalized to the task's 40 mm cube."""


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


@configclass
class ConveyorFrankaA09A12SceneCfg(ConveyorFrankaSceneCfg):
    """Checkpoint-compatible Digital Twin scene with visual warehouse dressing."""

    def __post_init__(self) -> None:
        """Dress the workcell and extend its returns around the fixed manipulation sections."""
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
        self.dome_light.spawn.intensity = 700.0

        right_x = BELT_CENTER_X + BELT_HALF_STRAIGHT
        straight_scale = (_A09_X_SCALE, _ASSET_XY_SCALE, _ASSET_Z_SCALE)

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

            run = "bottom" if side == "Left" else "top"
            y_position = center_y + (-BELT_TURN_RADIUS if side == "Left" else BELT_TURN_RADIUS)
            setattr(
                self,
                f"conveyor_{side_key}_{run}_a09_visual",
                _visual_usd_asset(
                    prim_path=f"{{ENV_REGEX_NS}}/Conveyor{side}{run.title()}A09Visual",
                    usd_path=str(_PRESENTATION_ASSETS / "conveyor_straight_supported.usd"),
                    position=(right_x, y_position, _ASSET_ROOT_Z),
                    scale=straight_scale,
                ),
            )
            getattr(self, f"conveyor_{side_key}_{run}_a09_visual").spawn.func = _spawn_authored_visual

        self.warehouse_visual = _visual_usd_asset(
            prim_path="{ENV_REGEX_NS}/WarehouseVisual",
            usd_path=str(_PRESENTATION_ASSETS / "warehouse.usda"),
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
        self.warehouse_visual.spawn.func = _spawn_authored_visual
        for cube_id in range(4):
            cube = getattr(self, f"cube_{cube_id}")
            cube.spawn = _ParcelCuboidCfg(**vars(cube.spawn))
            cube.spawn.func = _spawn_carton_cube

    def _configure_route_assets(
        self, parcel_colors: tuple[str, ...] = ("blue", "orange", "green", "purple") * 6
    ) -> None:
        """Read USD route geometry after the application selects its USD runtime."""
        from .conveyor_warehouse_geometry import warehouse_parcel_positions

        positions = warehouse_parcel_positions()
        if len(parcel_colors) != len(positions) or not set(parcel_colors) <= {"blue", "orange", "green", "purple"}:
            raise ValueError("Each authored parcel requires a color: blue, orange, green, or purple.")
        for cube_id, (position, color) in enumerate(zip(positions, parcel_colors, strict=True)):
            cube = self.cube_0.copy()
            cube.spawn.parcel_usd_path = str(_PRESENTATION_ASSETS / f"parcel_{color}.usda")
            cube.prim_path = f"{{ENV_REGEX_NS}}/Cube{cube_id}"
            cube.init_state.pos = (position[0], position[1], position[2] + _WORKSPACE_ELEVATION)
            setattr(self, f"cube_{cube_id}", cube)
        for side in ("Left", "Right"):
            for key in ("top_straight", "bottom_straight", "right_turn", "left_turn"):
                setattr(self, f"conveyor_{side.lower()}_{key}_collision", None)
            for index, section in enumerate(warehouse_belt_sections(side)):
                asset = _hidden_collision_geometry(section.belt.prim_path, section.geometry, 1.1e-5, 1)
                x, y, z = asset.init_state.pos
                asset.init_state.pos = (x, y, z + _WORKSPACE_ELEVATION)
                setattr(self, f"warehouse_{side.lower()}_section_{index}", asset)
            for boundary in ("inner", "outer"):
                setattr(self, f"guard_{side.lower()}_{boundary}_collision", None)
            for guard in warehouse_guard_meshes(side):
                asset = _hidden_collision_mesh(f"{{ENV_REGEX_NS}}/{guard.name}Collision", guard, 1.1e-5, 1)
                asset.init_state.pos = (0.0, 0.0, _WORKSPACE_ELEVATION)
                setattr(self, f"warehouse_{guard.name}_collision", asset)

    def build_conveyor_belt_specs(self, **kwargs: float | bool) -> tuple[SurfaceVelocitySpec, ...]:
        """Describe the linked warehouse surfaces to the existing Newton conveyor driver."""
        return tuple(section.belt for side in ("Left", "Right") for section in warehouse_belt_sections(side, **kwargs))


@configclass
class ConveyorFrankaA09A12EnvCfg(ConveyorFrankaEnvCfg):
    """Newton presentation variant with Digital Twin visuals and extended return routes."""

    scene: ConveyorFrankaA09A12SceneCfg = ConveyorFrankaA09A12SceneCfg(
        num_envs=1,
        env_spacing=24.0,
        replicate_physics=True,
    )

    def __post_init__(self) -> None:
        """Frame the presentation scene and allow packages to travel along its extended returns."""
        super().__post_init__()
        from .mdp.sorting import ConveyorSortCommandCfg

        self.commands.transfer = ConveyorSortCommandCfg()
        # Kit renders the authored USD directly; Newton needs only the physical geometry.
        self.sim.physics.load_visual_shapes = False
        self.conveyor_force.transported_body_count_per_env = len(self.commands.transfer.parcel_destinations)
        self.sim.physics.solver_cfg.nconmax = 400
        self.sim.physics.solver_cfg.njmax = 600
        self.conveyor_force.transported_body_pattern = r"(?:^|/)Cube_?[0-9]+(?:/|$)"
        self.terminations.cube_out_of_workspace.params = {
            "minimum": (-0.4, -1.2, -0.05),
            "maximum": (2.85, 1.2, 0.8),
        }
        self.sim.default_visualizer_cfg.eye = (4.8, -5.2, 3.0)
        self.sim.default_visualizer_cfg.lookat = (0.9, 0.6, 0.95)
        self.sim.default_visualizer_cfg.focal_length = 24.0
