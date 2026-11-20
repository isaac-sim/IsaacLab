# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Digital Twin warehouse visuals for conveyor-Franka playback."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import ConveyorSurfaceCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Sdf, Usd

from .conveyor_franka_env_cfg import (
    ConveyorFrankaEnvCfg,
    ConveyorFrankaSceneCfg,
    _conveyor_asset,
)
from .conveyor_warehouse_geometry import warehouse_belt_sections, warehouse_guard_meshes

# The physical floor and shared camera align with the authored workcell.
_PRESENTATION_OFFSET_Z = -0.7956274313214358
_PHYSICS_SCHEMA_PREFIXES = ("Physics", "Physx", "Newton", "Mujoco")
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
    if cfg.variants is not None:
        sim_utils.select_usd_variants(prim_path, cfg.variants)
    sim_utils.make_uninstanceable(prim_path)
    _make_usd_subtree_visual_only(prim)
    _extend_conveyor_supports(prim)
    return prim


def _extend_conveyor_supports(root_prim: Usd.Prim) -> None:
    """Apply authored support parameters to referenced meshes without bundling geometry."""
    import numpy as np

    from pxr import Usd, UsdGeom, Vt

    for prim in Usd.PrimRange(root_prim):
        anchor_attr = prim.GetAttribute("conveyor:supportAnchor")
        if not anchor_attr or not anchor_attr.HasAuthoredValueOpinion():
            continue
        anchor = anchor_attr.Get()
        scale = prim.GetAttribute("conveyor:supportScale").Get()
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get()).copy()
        lower = points[:, 2] < anchor
        points[lower, 2] = anchor + scale * (points[lower, 2] - anchor)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack((points.min(axis=0), points.max(axis=0)))))
        # Face-varying normals follow the deformed surface, including faces crossing the anchor.
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        normal = np.cross(
            points[indices[starts + 1]] - points[indices[starts]],
            points[indices[starts + 2]] - points[indices[starts]],
        )
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(normal, counts, axis=0)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


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

    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    UsdGeom.Imageable(prim.GetStage().GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()
    visual_cfg = sim_utils.UsdFileCfg(
        usd_path=str(_PRESENTATION_ASSETS / "parcel.usda"), variants={"color": cfg.parcel_color}
    )
    _spawn_authored_visual(f"{prim_path}/CartonVisual", visual_cfg)
    return prim


@configclass
class _ParcelCuboidCfg(sim_utils.CuboidCfg):
    """Original task collider with a separately authored carton appearance."""

    parcel_color: str = "plain"
    """Authored color variant of the task's 40 mm carton visual."""


def _make_usd_subtree_visual_only(root_prim: Usd.Prim) -> None:
    """Remove physics ownership with stage overrides, preserving the source asset."""
    from pxr import Sdf, Usd, UsdPhysics

    children = tuple(Usd.PrimRange(root_prim))
    with Sdf.ChangeBlock():
        for child in children:
            if not child.IsValid() or not child.IsActive():
                continue
            if (
                child.GetTypeName().startswith("OmniGraph")
                or child.IsA(UsdPhysics.Scene)
                or child.IsA(UsdPhysics.Joint)
            ):
                child.SetActive(False)
                continue
            for schema_name in tuple(child.GetAppliedSchemas()):
                is_physics = schema_name == "IsaacConveyorAPI" or schema_name.startswith(_PHYSICS_SCHEMA_PREFIXES)
                if is_physics and not child.RemoveAppliedSchema(schema_name):
                    raise RuntimeError(f"Failed to remove physics schema {schema_name!r} from {child.GetPath()}.")


@configclass
class ConveyorFrankaA09A12SceneCfg(ConveyorFrankaSceneCfg):
    """Checkpoint-compatible warehouse; static appearance is authored in USD."""

    racetracks = None
    tabletop = None
    table_pedestal = None
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _PRESENTATION_OFFSET_Z)),
        spawn=sim_utils.GroundPlaneCfg(visible=False, color=(0.055, 0.065, 0.082)),
    )
    warehouse_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WarehouseVisual",
        spawn=sim_utils.UsdFileCfg(func=_spawn_authored_visual, usd_path=str(_PRESENTATION_ASSETS / "workcell.usda")),
    )

    def __post_init__(self) -> None:
        self.dome_light.spawn.color = (0.70, 0.78, 0.92)
        self.dome_light.spawn.intensity = 700.0
        for cube_id in range(4):
            cube = getattr(self, f"cube_{cube_id}")
            cube.spawn = _ParcelCuboidCfg(**vars(cube.spawn))
            cube.spawn.func = _spawn_carton_cube

    def _configure_route_assets(self, parcel_colors: tuple[str, ...]) -> None:
        """Read USD route geometry after the application selects its USD runtime."""
        from .conveyor_warehouse_geometry import warehouse_parcel_positions

        positions = warehouse_parcel_positions()
        if len(parcel_colors) != len(positions) or not set(parcel_colors) <= {"blue", "orange", "green", "purple"}:
            raise ValueError("Each authored parcel requires a color: blue, orange, green, or purple.")
        for cube_id, (position, color) in enumerate(zip(positions, parcel_colors, strict=True)):
            cube = self.cube_0.copy()
            cube.spawn.parcel_color = color
            cube.prim_path = f"{{ENV_REGEX_NS}}/Cube{cube_id}"
            cube.init_state.pos = position
            setattr(self, f"cube_{cube_id}", cube)
        for side in ("Left", "Right"):
            for index, section in enumerate(warehouse_belt_sections(side)):
                setattr(
                    self,
                    f"warehouse_{side.lower()}_section_{index}",
                    _conveyor_asset(section.belt.prim_path, section.geometry),
                )
            for guard in warehouse_guard_meshes(side):
                setattr(
                    self,
                    f"warehouse_{guard.name}_collision",
                    _conveyor_asset(f"{{ENV_REGEX_NS}}/{guard.name}Collision", guard),
                )

    def build_conveyor_belt_specs(self, **kwargs: float) -> tuple[ConveyorSurfaceCfg, ...]:
        """Describe the linked warehouse surfaces to the existing Newton conveyor driver."""
        return tuple(section.belt for side in ("Left", "Right") for section in warehouse_belt_sections(side, **kwargs))


@configclass
class ConveyorFrankaA09A12EnvCfg(ConveyorFrankaEnvCfg):
    """Newton presentation variant with Digital Twin visuals and extended return routes."""

    class_type: type | str = "{DIR}.conveyor_franka_warehouse_env:ConveyorFrankaWarehouseEnv"
    park_when_idle: bool = False
    """Replace idle actions with home-position control during pretrained-policy playback only."""
    scene: ConveyorFrankaA09A12SceneCfg = ConveyorFrankaA09A12SceneCfg(
        num_envs=1,
        env_spacing=24.0,
    )

    def __post_init__(self) -> None:
        """Frame the presentation scene and allow packages to travel along its extended returns."""
        super().__post_init__()
        from .mdp.sorting import ConveyorSortCommandCfg, reset_inventory

        self.commands.transfer = ConveyorSortCommandCfg()
        self.events.reset_from_state_table = None
        self.events.reset_inventory = EventTermCfg(func=reset_inventory, mode="reset")
        # Kit renders the authored USD directly; Newton needs only the physical geometry.
        self.sim.physics.load_visual_shapes = False
        # Adjacent static belt sections must not consume the parcel contact budget.
        self.sim.physics.collision_cfg.include_static_kinematic_pairs = False
        self.conveyor.body_count_per_env = len(self.commands.transfer.parcel_destinations)
        self.sim.physics.solver_cfg.nconmax = 400
        self.sim.physics.solver_cfg.njmax = 600
        self.conveyor.body_pattern = r"(?:^|/)Cube_?[0-9]+(?:/|$)"
        self.terminations.cube_out_of_workspace.params = {
            "minimum": (-0.4, -1.2, -0.05),
            "maximum": (2.85, 1.2, 0.8),
        }
        self.sim.visualizer_cfgs.eye = (4.8, -5.2, 3.0 + _PRESENTATION_OFFSET_Z)
        self.sim.visualizer_cfgs.lookat = (0.9, 0.6, 0.95 + _PRESENTATION_OFFSET_Z)
        self.sim.visualizer_cfgs.focal_length = 24.0

    def play_mode(self) -> None:
        """Enable the idle controller only for pretrained-policy playback."""
        super().play_mode()
        self.park_when_idle = True
