# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable scene and IK control configuration for the coffee demonstration."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Literal

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonRTXBackendCfg

import isaaclab.envs.mdp as common_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg

from .assets import BOWL_POSITION, COFFEE_ASSET_DIR, CUP_HEIGHT, CUP_INNER_RADIUS, CUP_POSITION, coffee_cavity_radius
from .robot import (
    ARM_JOINT_NAMES,
    DEFAULT_SPOON_OFFSET_POS,
    DEFAULT_SPOON_OFFSET_ROT,
    HAND_JOINT_LIMITS,
    HAND_JOINT_NAMES,
    PALM_BODY_NAME,
)


@configclass
class CoffeeMPMSolverCfg(MPMSolverCfg):
    """Native MPM with optional passive rendering grains."""

    class_type: type | str = "{DIR}.physics:CoffeeMPMSolver"

    solver: str | tuple[str, ...] | None = None
    """Pressure solver; ``None`` selects the final wet or dry task profile."""
    max_iterations: int | None = None
    """Pressure budget; ``None`` selects the final wet or dry task profile."""
    collider_basis: str | None = None
    """Contact basis; ``None`` reserves particle-site capacity from the sampling density."""
    grains_per_particle: int = 0
    """Number of visual samples per physical particle; zero disables their allocation."""


def make_physics_cfg(voxel_size: float) -> NewtonCfg:
    """Configure coupling with material defaults resolved after scene overrides."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False, integrator="implicitfast", njmax=1024, nconmax=128
                    ),
                    bodies=[r"{ENV_REGEX_NS}/Robot"],
                ),
                CouplerEntryCfg(
                    name="media",
                    solver_cfg=CoffeeMPMSolverCfg(
                        voxel_size=voxel_size,
                        # configure_scene reserves a particle-count-based budget
                        # unless the caller supplies a positive capacity.
                        max_active_cell_count=-1,
                        max_lower_node_count=256,
                        max_upper_node_count=32,
                        # Resolve contacts at particle sites and pressure within
                        # each cell. Hard projection can permanently collapse
                        # powder onto the concave tool's surface.
                        strain_basis="P1d",
                        collider_velocity_mode="backward",
                        # Native background-air regularization bounds nearly empty grid nodes.
                        air_drag=1.0,
                        separate_worlds=True,
                    ),
                    bodies=[r"{ENV_REGEX_NS}/Spoon"],
                    all_particles=True,
                    include_static_shapes=True,
                    in_place=True,
                ),
            ],
            proxies=[],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        num_substeps=1,
        use_cuda_graph=True,
    )


@configclass
class CoffeeRTXVisualizerCfg(NewtonRTXBackendCfg):
    """Coffee presentation with an optional render-only public kitchen."""

    class_type: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering:CoffeeRTXVisualizer"
    backdrop: Literal["kitchen", "studio"] = "kitchen"
    """Presentation backdrop: ``kitchen`` or the original ``studio``."""


@configclass
class CoffeeKitBackendCfg(KitBackendCfg):
    """Coffee presentation with an optional render-only public kitchen."""

    class_type: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.kit_rendering:CoffeeKitVisualizer"
    backdrop: Literal["kitchen", "studio"] = "kitchen"
    """Presentation backdrop: ``kitchen`` or the original ``studio``."""
    render_settings: IsaacRtxRendererGlobalSettingsCfg = IsaacRtxRendererGlobalSettingsCfg()
    """Global RTX settings for the Kit viewport and its recorded frames."""


@configclass
class CoffeePresentationCfg(sim_utils.SpawnerCfg):
    """Author the coffee backdrop as ordinary scene content."""

    func: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering:spawn_presentation"
    backdrop: Literal["kitchen", "studio"] = "kitchen"


@configclass
class CoffeeSceneCfg(InteractiveSceneCfg):
    """Reusable robot, hosted props and optional MPM fill volumes."""

    robot: ArticulationCfg | None = None
    table: AssetBaseCfg | None = None
    bowl: AssetBaseCfg | None = None
    cup: AssetBaseCfg | None = None
    spoon: RigidObjectCfg | None = None
    coffee: MPMObjectCfg | None = None
    water: MPMObjectCfg | None = None
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.92, 0.95, 1.0)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.86, 0.69), angle=12.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.239, -0.370, -0.099, 0.892)),
    )

    presentation: AssetBaseCfg | None = None


@configclass
class CoffeeActionsCfg:
    """Absolute robot-base palm XYZ/XYZW followed by 22 named finger angles."""

    arm = common_mdp.DifferentialInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env:PalmPoseIKAction",
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=PALM_BODY_NAME,
        scale=1.0,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
    )
    hand = common_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(HAND_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=False,
        clip=dict(zip(HAND_JOINT_NAMES, HAND_JOINT_LIMITS, strict=True)),
    )


@configclass
class CoffeeEventsCfg:
    reset = EventTermCfg(func="isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env:reset_scene", mode="reset")


@configclass
class CoffeeDemoEnvCfg(ManagerBasedEnvCfg):
    """Fine MPM tool-use scene with independently simulated arm and material."""

    class_type: str = "{DIR}.demo_env:CoffeeDemoEnv"
    scene: CoffeeSceneCfg = CoffeeSceneCfg(num_envs=1, env_spacing=2.0)
    actions: CoffeeActionsCfg = CoffeeActionsCfg()
    observations: object = SimpleNamespace()
    events: CoffeeEventsCfg = CoffeeEventsCfg()

    voxel_size: float = 0.004
    spoon_margin_voxels: float = 0.48
    """Spoon collision expansion in grid cells, separate from the container margins."""
    spoon_rim_margin_scale: float = 1.0 / 6.0
    """Rim margin relative to the broad faces; zero keeps the lip at the actual mesh."""
    water_enabled: bool = False
    coffee_wetting_time: float = 0.02
    """Accumulated contact with water before soluble coffee becomes liquid [s]."""
    grains_per_particle: int = 0
    bowl_scale: float = 0.65
    """Horizontal scale of the bowl (at least 0.4); depth stays unchanged."""
    coffee_fill_scale: float = 0.65
    """Horizontal scale of the initial coffee pile, independent of the bowl size."""
    coffee_particle_spacing: float = 0.0015
    """Maximum coffee sampling distance [m], independent of passive rendering grains."""
    coffee_fill_height: float = 0.055
    water_fill_height: float = 0.0535
    """Initial water seed height [m]; gravity spreads the fill to a lower free surface."""
    min_delivered_fraction: float = 0.005
    """Minimum cup fraction required for the scripted demo to report completion."""
    min_delivered_mass: float = 0.001
    """Minimum coffee mass in the cup [kg], independent of the source fill size."""
    spoon_offset_pos: tuple[float, float, float] = DEFAULT_SPOON_OFFSET_POS
    spoon_offset_rot: tuple[float, float, float, float] = DEFAULT_SPOON_OFFSET_ROT

    def __post_init__(self):
        self.decimation = 8
        self.sim.dt = 1.0 / 800.0
        self.sim.render_interval = 16
        self.sim.use_newton_actuators = True
        self.sim.physics = make_physics_cfg(self.voxel_size)


def configure_scene(cfg: CoffeeDemoEnvCfg) -> None:
    """Resolve hosted assets and sample collision-free fill volumes."""
    from .robot import get_robot_cfg

    if not math.isfinite(cfg.voxel_size) or not 0.001 <= cfg.voxel_size <= 0.010:
        raise ValueError("voxel_size must be between 1 and 10 mm.")
    if not math.isfinite(cfg.spoon_margin_voxels) or cfg.spoon_margin_voxels < 0.0:
        raise ValueError("spoon_margin_voxels must be finite and nonnegative.")
    if not math.isfinite(cfg.spoon_rim_margin_scale) or not 0.0 <= cfg.spoon_rim_margin_scale <= 1.0:
        raise ValueError("spoon_rim_margin_scale must be between zero and one.")
    if not math.isfinite(cfg.bowl_scale) or cfg.bowl_scale < 0.4:
        raise ValueError("bowl_scale must be finite and at least 0.4.")
    if not math.isfinite(cfg.coffee_fill_scale) or not 0.0 < cfg.coffee_fill_scale <= cfg.bowl_scale:
        raise ValueError("coffee_fill_scale must be positive and no larger than bowl_scale.")
    if not math.isfinite(cfg.coffee_particle_spacing) or cfg.coffee_particle_spacing <= 0.0:
        raise ValueError("coffee_particle_spacing must be finite and positive.")
    if not 0.009 <= cfg.coffee_fill_height <= 0.075:
        raise ValueError("coffee_fill_height must be between 9 and 75 mm.")
    if not 0.012 <= cfg.water_fill_height <= CUP_HEIGHT - 0.002:
        raise ValueError("water_fill_height must be between 12 mm and 2 mm below the cup rim.")
    if not 0.0 < cfg.min_delivered_fraction <= 1.0:
        raise ValueError("min_delivered_fraction must be in (0, 1].")
    if not math.isfinite(cfg.min_delivered_mass) or cfg.min_delivered_mass <= 0.0:
        raise ValueError("min_delivered_mass must be finite and positive.")
    backdrops = {
        visualizer.backend.backdrop
        for visualizer in cfg.sim.visualizer_cfgs
        if isinstance(visualizer.backend, (CoffeeKitBackendCfg, CoffeeRTXVisualizerCfg))
    }
    if len(backdrops) > 1:
        raise ValueError("Coffee renderers sharing a scene must use the same backdrop.")
    for backdrop in backdrops:
        cfg.scene.presentation = AssetBaseCfg(
            prim_path="/World/CoffeePresentation", spawn=CoffeePresentationCfg(backdrop=backdrop), cloning_contexts=()
        )
    cfg.scene.robot = get_robot_cfg()
    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{COFFEE_ASSET_DIR}/table.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    cfg.scene.bowl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{COFFEE_ASSET_DIR}/bowl.usda", scale=(cfg.bowl_scale, cfg.bowl_scale, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOWL_POSITION),
    )
    cfg.scene.cup = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{COFFEE_ASSET_DIR}/cup_shallow.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CUP_POSITION),
    )
    cfg.scene.spoon = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Spoon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{COFFEE_ASSET_DIR}/spoon.usda",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.12, 0.15)),
    )
    spacing = cfg.voxel_size / 2.0
    cfg.scene.coffee = _particle_cfg(
        "Coffee",
        BOWL_POSITION,
        0.060 * cfg.coffee_fill_scale,
        cfg.coffee_fill_height,
        min(spacing, cfg.coffee_particle_spacing),
        False,
        cfg.coffee_fill_scale,
    )
    cfg.scene.water = (
        _particle_cfg("Water", CUP_POSITION, CUP_INNER_RADIUS - 0.002, cfg.water_fill_height, spacing, True)
        if cfg.water_enabled
        else None
    )
    for asset in (cfg.scene.coffee, cfg.scene.water):
        if asset is not None:
            asset.spawn.visible = cfg.grains_per_particle == 0
    solver = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    # Resolve defaults after Hydra/CLI overrides, preserving explicitly tuned
    # pressure settings and the independently configured integration rates.
    if solver.solver is None:
        solver.solver = "gauss-seidel" if cfg.water_enabled else "auto"
    if solver.max_iterations is None:
        # Tall liquid fills need converged pressure to retain their surface level.
        solver.max_iterations = 500 if cfg.water_enabled else 100
    solver.voxel_size = cfg.voxel_size
    solver.grains_per_particle = cfg.grains_per_particle
    if solver.collider_basis is None:
        sample_ratio = math.ceil(cfg.voxel_size / min(spacing, cfg.coffee_particle_spacing))
        contact_capacity = 2 * sample_ratio**3
        solver.collider_basis = f"pic{max(128, 1 << (contact_capacity - 1).bit_length())}"
    # The rebuildable grid allocates at most one active voxel per physical particle.
    # Its capacity also sizes solver launches, so avoid padding small fills excessively.
    particle_count = sum(
        len(asset.spawn.positions) for asset in (cfg.scene.coffee, cfg.scene.water) if asset is not None
    )
    if solver.max_active_cell_count < 0:
        solver.max_active_cell_count = max(1024, math.ceil(particle_count / 1024) * 1024) * cfg.scene.num_envs
    # Catch escaped grains on the studio floor, then reserve spatial-spread
    # headroom for repeated scoops. A lower node spans 128 voxels (256 mm at
    # the default resolution), independently of the active-cell count.
    solver.max_lower_node_count = 256 * cfg.scene.num_envs
    solver.max_upper_node_count = max(32, cfg.scene.num_envs)


def _particle_cfg(
    name: str,
    position: tuple[float, float, float],
    radius: float,
    height: float,
    spacing: float,
    water: bool,
    radial_scale: float = 1.0,
) -> MPMObjectCfg:
    """Seed a deterministic fill and bind its granular or liquid material."""
    axis = np.arange(-radius + spacing / 2.0, radius, spacing)
    z = np.arange(0.007 + spacing / 2.0, height, spacing)
    points = np.stack(np.meshgrid(axis, axis, z, indexing="ij"), axis=-1).reshape(-1, 3)
    # Keep the coffee seed volume fixed when changing its sampling density.
    edge_clearance = spacing if water else 0.002
    points = points[np.linalg.norm(points[:, :2], axis=1) < radius - edge_clearance]
    if not water:
        points = points[
            np.linalg.norm(points[:, :2], axis=1) < radial_scale * coffee_cavity_radius(points[:, 2]) - 0.004
        ]
    if not len(points):
        raise ValueError(f"{name} fill is empty at spacing={spacing:g} m and height={height:g} m.")
    rng = np.random.default_rng(7)
    jitter = (0.15 if water else 0.35) * spacing
    points += rng.uniform(-jitter, jitter, points.shape)
    density = 1000.0 if water else 450.0
    color = (0.65, 0.82, 0.88) if water else (0.075, 0.023, 0.009)
    material = (
        MPMParticleMaterialCfg(
            density=density,
            viscosity=0.001,
            friction=0.0,
            damping=0.001,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
        )
        if water
        else MPMParticleMaterialCfg(density=density, friction=0.55, damping=0.001)
    )
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=MPMPointsCfg(
            positions=points.astype(np.float32).tolist(),
            radius=spacing / 2.0,
            mass=density * spacing**3,
            material=material,
            visual_color=color,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85 if not water else 0.08),
        ),
        init_state=MPMObjectCfg.InitialStateCfg(pos=position),
    )
