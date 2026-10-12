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
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.envs.mdp as common_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import BOWL_POSITION, CUP_INNER_RADIUS, CUP_POSITION, coffee_cavity_radius
from .physics_cfg import configure_contact_grasp, make_physics_cfg
from .robot import (
    ARM_JOINT_NAMES,
    DEFAULT_SPOON_OFFSET_POS,
    DEFAULT_SPOON_OFFSET_ROT,
    HAND_JOINT_LIMITS,
    HAND_JOINT_NAMES,
    PALM_BODY_NAME,
)


@configclass
class CoffeeSceneCfg(InteractiveSceneCfg):
    """Reusable assets; local USD files are resolved after the simulation launcher."""

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


@configclass
class CoffeeActionsCfg:
    """Absolute robot-base palm XYZ/XYZW followed by 22 named finger angles."""

    arm = common_mdp.DifferentialInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.rizon_sharpa_coffee.control:PalmPoseIKAction",
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
    reset = EventTermCfg(func="isaaclab_tasks.contrib.rizon_sharpa_coffee.control:reset_scene", mode="reset")


@configclass
class CoffeeDemoEnvCfg(ManagerBasedEnvCfg):
    """Fine MPM tool-use scene with independently simulated arm and material."""

    class_type: str = "{DIR}.demo_env:CoffeeDemoEnv"
    scene: CoffeeSceneCfg = CoffeeSceneCfg(num_envs=1, env_spacing=2.0)
    actions: CoffeeActionsCfg = CoffeeActionsCfg()
    observations: object = SimpleNamespace()
    events: CoffeeEventsCfg = CoffeeEventsCfg()

    voxel_size: float = 0.004
    spoon_margin_voxels: float = 1.2
    """Spoon collision expansion in grid cells, separate from the container margins."""
    spoon_rim_margin_scale: float = 0.0
    """Rim margin relative to the broad faces; zero keeps the lip at the actual mesh."""
    spoon_proxy_mass_scale: float = 20.0
    """Finite inertia multiplier for the contact-held spoon's MPM proxy."""
    water_enabled: bool = False
    coffee_wetting_time: float = 0.12
    """Accumulated contact with water before soluble coffee becomes liquid [s]."""
    grains_per_particle: int = 0
    grasp_reset_path: str | None = None
    grasp_reset_joint_targets: dict[str, float] | None = None
    grasp_mode: Literal["mounted", "contact"] = "mounted"
    """Use a kinematic tool mount or a free spoon held by rigid finger contacts."""
    coffee_fill_height: float = 0.065
    water_fill_height: float = 0.075
    """Initial water seed height [m]; gravity spreads the fill to a lower free surface."""
    min_delivered_fraction: float = 0.005
    """Minimum cup fraction required for the scripted demo to report completion."""
    spoon_offset_pos: tuple[float, float, float] = DEFAULT_SPOON_OFFSET_POS
    spoon_offset_rot: tuple[float, float, float, float] = DEFAULT_SPOON_OFFSET_ROT

    def __post_init__(self):
        self.decimation = 8
        self.sim.dt = 1.0 / 800.0
        self.sim.render_interval = 16
        self.sim.use_newton_actuators = True
        self.sim.physics = make_physics_cfg(self.voxel_size, water=None)


def configure_scene(cfg: CoffeeDemoEnvCfg) -> None:
    """Resolve generated assets and sample collision-free fill volumes."""
    from .assets import build_coffee_assets
    from .robot import get_robot_cfg

    if not math.isfinite(cfg.voxel_size) or not 0.001 <= cfg.voxel_size <= 0.010:
        raise ValueError("voxel_size must be between 1 and 10 mm.")
    if not math.isfinite(cfg.spoon_margin_voxels) or cfg.spoon_margin_voxels < 0.0:
        raise ValueError("spoon_margin_voxels must be finite and nonnegative.")
    if not math.isfinite(cfg.spoon_rim_margin_scale) or not 0.0 <= cfg.spoon_rim_margin_scale <= 1.0:
        raise ValueError("spoon_rim_margin_scale must be between zero and one.")
    if not math.isfinite(cfg.spoon_proxy_mass_scale) or cfg.spoon_proxy_mass_scale <= 0.0:
        raise ValueError("spoon_proxy_mass_scale must be finite and positive.")
    if not 0.009 <= cfg.coffee_fill_height <= 0.075:
        raise ValueError("coffee_fill_height must be between 9 and 75 mm.")
    if not 0.012 <= cfg.water_fill_height <= 0.079:
        raise ValueError("water_fill_height must be between 12 and 79 mm.")
    if not 0.0 < cfg.min_delivered_fraction <= 1.0:
        raise ValueError("min_delivered_fraction must be in (0, 1].")
    paths = build_coffee_assets()
    cfg.scene.robot = get_robot_cfg(contact_grasp=cfg.grasp_mode == "contact")
    from .grasp_reset import configure_grasp_reset

    if cfg.grasp_mode not in ("mounted", "contact"):
        raise ValueError("grasp_mode must be 'mounted' or 'contact'.")
    if cfg.grasp_mode == "contact":
        configure_contact_grasp(
            cfg.sim.physics,
            rigid_substeps=max(1, round(1600.0 * cfg.sim.dt)),
            mass_scale=cfg.spoon_proxy_mass_scale,
        )
    configure_grasp_reset(cfg)
    for name, position in (("table", (0.0, 0.0, 0.0)), ("bowl", BOWL_POSITION), ("cup", CUP_POSITION)):
        setattr(
            cfg.scene,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=sim_utils.UsdFileCfg(usd_path=paths[name]),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            ),
        )
    cfg.scene.spoon = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Spoon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=paths["spoon"],
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=cfg.grasp_mode == "mounted"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.12, 0.15)),
    )
    spacing = cfg.voxel_size / 2.0
    cfg.scene.coffee = _particle_cfg("Coffee", BOWL_POSITION, 0.060, cfg.coffee_fill_height, spacing, False)
    cfg.scene.water = (
        _particle_cfg("Water", CUP_POSITION, CUP_INNER_RADIUS - 0.002, cfg.water_fill_height, spacing, True)
        if cfg.water_enabled
        else None
    )
    solver = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    # Resolve defaults after Hydra/CLI overrides, preserving explicitly tuned
    # pressure settings and the independently configured integration rates.
    if solver.solver is None:
        solver.solver = "gs" if cfg.water_enabled else "auto"
    if solver.max_iterations is None:
        solver.max_iterations = 500 if cfg.water_enabled else 100
    solver.voxel_size = cfg.voxel_size
    solver.grains_per_particle = cfg.grains_per_particle
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
) -> MPMObjectCfg:
    """Seed a deterministic fill and bind its granular or liquid material."""
    axis = np.arange(-radius + spacing / 2.0, radius, spacing)
    z = np.arange(0.007 + spacing / 2.0, height, spacing)
    points = np.stack(np.meshgrid(axis, axis, z, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[np.linalg.norm(points[:, :2], axis=1) < radius - spacing]
    if not water:
        points = points[np.linalg.norm(points[:, :2], axis=1) < coffee_cavity_radius(points[:, 2]) - 2.0 * spacing]
    if not len(points):
        raise ValueError(f"{name} fill is empty at spacing={spacing:g} m and height={height:g} m.")
    rng = np.random.default_rng(7)
    points += rng.uniform(-0.15 * spacing, 0.15 * spacing, points.shape)
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
