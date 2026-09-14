# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset and recording helpers for the Newton tablecloth demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.sim import SimulationContext

RIGID_GAP = 0.001

KITCHEN_ISLAND_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Counters/Island_A01/sm_fixture_island_a01_01.usd"
)
BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Dishware/Bowl_G01/sm_kitchenware_bowl_g01_01.usd"
WINE_GLASS_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Furnishing/Kitchen/Kitchenware/Dishware/Glass_Wine_A01/"
    "sm_dishware_glass_wine_a01_01.usd"
)
FORK_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Utensils/Fork_K01/sm_kitchenware_fork_k01_01.usd"

# Dimensions measured from the SimReady assets.
KITCHEN_ISLAND_SIZE = (0.631293, 1.240893, 0.900409)
BOWL_LOCAL_Z_MIN = 0.000001783
BOWL_LOCAL_RADIUS = 0.052765541
BOWL_LOCAL_HEIGHT = 0.043569469
WINE_GLASS_LOCAL_Z_MIN = 0.000358
FORK_LOCAL_Z_MIN = 0.000127
BOWL_SCALE = 1.35
FORK_SCALE = 0.85
FORK_ROTATION = (0.0, 0.0, -0.70710678, 0.70710678)
FORK_CENTER_OF_MASS = (0.0, 0.0092, 0.0089)
FORK_DIAGONAL_INERTIA = (1.86e-4, 6.2e-6, 1.90e-4)
# A base-only collision proxy would otherwise give the tall glass a flat-disk inertia.
WINE_GLASS_CENTER_OF_MASS = (0.0, 0.0, 0.04)
WINE_GLASS_DIAGONAL_INERTIA = (3.0e-3, 3.0e-3, 5.0e-4)


def rigid_material(
    *, density: float | None, friction: float, contact_stiffness: float, contact_damping: float
) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Return solver-common friction and Newton compliant-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
            density=density,
        ),
        NewtonMaterialCfg(contact_stiffness=contact_stiffness, contact_damping=contact_damping),
    ]


def collision_properties() -> list[UsdPhysicsCollisionCfg | NewtonCollisionCfg]:
    """Return collision properties shared by cloth-contacting rigid shapes."""
    return [
        UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=0.002, contact_gap=RIGID_GAP),
    ]


def tabletop_collider_cfg(
    size: tuple[float, float, float], *, friction: float, contact_stiffness: float, contact_damping: float
) -> sim_utils.CuboidCfg:
    """Create a low-cost collision proxy for a visual-only table."""
    return sim_utils.CuboidCfg(
        size=size,
        visible=False,
        collision_props=collision_properties(),
        physics_material=rigid_material(
            density=None,
            friction=friction,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
        ),
    )


def spawn_visual_table_from_usd(
    prim_path: str,
    cfg: VisualTableUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a visual-only island and remove material inputs unsupported by Newton GL."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    from pxr import UsdShade  # noqa: PLC0415

    for root in sim_utils.find_matching_prims(prim_path):
        for child in sim_utils.get_all_matching_child_prims(root.GetPath()):
            physics_binding = child.GetRelationship("material:binding:physics")
            if physics_binding:
                physics_binding.SetTargets([])
            if child.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(child)
                if shader.GetIdAttr().Get() == "UsdPreviewSurface":
                    for input_name in ("metallic", "roughness"):
                        shader.GetInput(input_name).DisconnectSource()
    return prim


@configclass
class VisualTableUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner config for a visual-only SimReady kitchen island."""

    func: Callable | str = spawn_visual_table_from_usd


def spawn_tableware_from_usd(
    prim_path: str,
    cfg: TablewareUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a tableware visual with measured mass and analytic collision proxies."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    from pxr import Gf, UsdPhysics, UsdShade  # noqa: PLC0415

    for root in sim_utils.find_matching_prims(prim_path):
        root_path = root.GetPath().pathString
        for child in sim_utils.get_all_matching_child_prims(root_path):
            physics_binding = child.GetRelationship("material:binding:physics")
            if physics_binding:
                physics_binding.SetTargets([])
            if child.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(child)
                shader_id = shader.GetIdAttr().Get()
                if shader_id == "UsdPreviewSurface":
                    for input_name in ("metallic", "roughness"):
                        shader.GetInput(input_name).DisconnectSource()
                    if cfg.visual_color is not None:
                        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*cfg.visual_color))
                    if cfg.visual_opacity is not None:
                        shader.GetInput("opacity").Set(cfg.visual_opacity)
                    if cfg.visual_roughness is not None:
                        shader.GetInput("roughness").Set(cfg.visual_roughness)
                elif shader_id == "mdl:OmniGlass":
                    if cfg.visual_color is not None:
                        shader.GetInput("glass_color").Set(Gf.Vec3f(*cfg.visual_color))
                    if cfg.visual_opacity is not None:
                        shader.GetInput("cutout_opacity").Set(cfg.visual_opacity)
                    if cfg.visual_roughness is not None:
                        shader.GetInput("frosting_roughness").Set(cfg.visual_roughness)

        mesh_colliders = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda child: child.HasAPI(UsdPhysics.CollisionAPI),
        )
        if len(mesh_colliders) != 1:
            raise RuntimeError(f"Expected one SimReady tableware mesh collider, found {len(mesh_colliders)}")
        UsdPhysics.CollisionAPI(mesh_colliders[0]).CreateCollisionEnabledAttr(False)

        bodies = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda child: child.HasAPI(UsdPhysics.RigidBodyAPI),
        )
        if len(bodies) != 1:
            raise RuntimeError(f"Expected one SimReady tableware rigid body, found {len(bodies)}")
        body = bodies[0]
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(cfg.mass)
        mass_api.CreateDensityAttr(0.0)
        if cfg.center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr(cfg.center_of_mass)
        if cfg.diagonal_inertia is not None:
            mass_api.CreateDiagonalInertiaAttr(cfg.diagonal_inertia)
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

        proxy_cfgs = (cfg.collision_proxy_cfg, *cfg.additional_collision_proxy_cfgs)
        proxy_positions = (cfg.collision_proxy_position, *cfg.additional_collision_proxy_positions)
        for index, (proxy_cfg, proxy_position) in enumerate(zip(proxy_cfgs, proxy_positions, strict=True)):
            suffix = "" if index == 0 else str(index)
            proxy_cfg.func(
                f"{body.GetPath()}/CollisionProxy{suffix}",
                proxy_cfg,
                translation=proxy_position,
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
    return prim


@configclass
class TablewareUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner config for SimReady visuals with analytic collision proxies."""

    func: Callable | str = spawn_tableware_from_usd
    mass: float = MISSING
    collision_proxy_cfg: sim_utils.ShapeCfg = MISSING
    collision_proxy_position: tuple[float, float, float] = MISSING
    additional_collision_proxy_cfgs: tuple[sim_utils.ShapeCfg, ...] = ()
    additional_collision_proxy_positions: tuple[tuple[float, float, float], ...] = ()
    center_of_mass: tuple[float, float, float] | None = None
    diagonal_inertia: tuple[float, float, float] | None = None
    visual_color: tuple[float, float, float] | None = None
    visual_opacity: float | None = None
    visual_roughness: float | None = None


def rigid_object_cfg(
    *,
    usd_path: str,
    position: tuple[float, float, float],
    mass: float,
    collision_proxy_cfg: sim_utils.ShapeCfg,
    collision_proxy_position: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    scale: tuple[float, float, float] | None = None,
    center_of_mass: tuple[float, float, float] | None = None,
    diagonal_inertia: tuple[float, float, float] | None = None,
    additional_collision_proxy_cfgs: tuple[sim_utils.ShapeCfg, ...] = (),
    additional_collision_proxy_positions: tuple[tuple[float, float, float], ...] = (),
    visual_color: tuple[float, float, float] | None = None,
    visual_opacity: float | None = None,
    visual_roughness: float | None = None,
) -> RigidObjectCfg:
    """Create a dynamic SimReady tableware object with analytic collision."""
    return RigidObjectCfg(
        prim_path="",
        spawn=TablewareUsdFileCfg(
            usd_path=usd_path,
            scale=scale,
            make_uninstanceable=True,
            rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False)],
            mass=mass,
            collision_proxy_cfg=collision_proxy_cfg,
            collision_proxy_position=collision_proxy_position,
            additional_collision_proxy_cfgs=additional_collision_proxy_cfgs,
            additional_collision_proxy_positions=additional_collision_proxy_positions,
            center_of_mass=center_of_mass,
            diagonal_inertia=diagonal_inertia,
            visual_color=visual_color,
            visual_opacity=visual_opacity,
            visual_roughness=visual_roughness,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation),
    )


class _StandaloneVideoTarget:
    """Expose the simulation fields expected by Isaac Lab's step-driven recorder."""

    def __init__(self, sim: SimulationContext, fps: int):
        self.sim = sim
        self.step_dt = sim.get_physics_dt()
        self.metadata = {"render_fps": fps}


def create_video_recorder(
    sim: SimulationContext,
    *,
    enabled: bool,
    output_dir: str,
    filename_prefix: str,
    video_length: int,
    fps: int,
) -> VideoRecorder | None:
    """Create a step-driven viewport recorder when requested."""
    if not enabled:
        return None

    from isaaclab.envs.utils.video_recorder import VideoRecorder  # noqa: PLC0415
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: PLC0415

    print(f"[INFO]: Recording {video_length / fps:.1f} s to {output_dir}/", flush=True)
    return VideoRecorder(
        VideoRecorderCfg(
            source="visualizer",
            output_dir=output_dir,
            output_filename_prefix=filename_prefix,
            fps=fps,
            video_length=video_length,
        ),
        _StandaloneVideoTarget(sim, fps),
    )
