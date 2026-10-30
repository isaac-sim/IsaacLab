# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Material layers and whole-body-controller gains for the granular course."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MPMSolverCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.materials import spawn_physics_material
from isaaclab.terrains import TerrainImporter

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from .sonic_parameters import DAMPING, JOINT_NAMES, STIFFNESS

if TYPE_CHECKING:
    from .terrain_env_cfg import CourseCfg


class ParticleTerrainImporter(TerrainImporter):
    """Enable GPU particles with host state buffers for USD readback."""

    def import_usd(self, name: str, usd_path: str):
        from pxr import PhysxSchema  # noqa: PLC0415

        super().import_usd(name, usd_path)
        sim = SimulationContext.instance()
        sim.set_setting("/physics/cudaDevice", torch.cuda.current_device())
        sim.set_setting("/physics/suppressReadback", False)
        scene = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
        scene.CreateEnableGPUDynamicsAttr(True)
        scene.CreateBroadphaseTypeAttr("GPU")


def configure_granular(cfg, is_physx: bool):
    """Configure SONIC actuators and the selected particle backend."""
    for name, actuator in cfg.scene.robot.actuators.items():
        actuator.joint_effort_limit = {"feet": 20.0, "waist": 50.0}.get(name, 300.0)
        if name == "hands":
            continue
        selected = [
            i
            for i, joint in enumerate(JOINT_NAMES)
            if any(re.fullmatch(pattern, joint) for pattern in actuator.joint_names_expr)
        ]
        stiffness = {JOINT_NAMES[i]: float(STIFFNESS[i]) for i in selected}
        damping = {JOINT_NAMES[i]: float(DAMPING[i]) for i in selected}
        actuator.stiffness = stiffness
        actuator.damping = damping
    if is_physx:
        # Host buffers enable USD readback; the launcher has already selected the CUDA device.
        cfg.sim.device = "cpu"
        cfg.scene.terrain.class_type = ParticleTerrainImporter
        return
    robot_pattern = r"{ENV_REGEX_NS}/Robot"
    coupled = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="robot", solver_cfg=cfg.sim.physics.solver_cfg, bodies=[robot_pattern], include_static_shapes=True
            ),
            CouplerEntryCfg(
                name="media",
                solver_cfg=MPMSolverCfg(
                    voxel_size=0.01,
                    grid_type="sparse",
                    grid_padding=0,
                    max_active_cell_count=1 << 19,
                    max_leaf_node_count=1 << 15,
                    max_lower_node_count=1024,
                    max_upper_node_count=32,
                    max_iterations=50,
                    strain_basis="P0",
                    critical_fraction=0.0,
                    transfer_scheme="pic",
                ),
                all_particles=True,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="robot", destination="media", bodies=[robot_pattern], mode="lagged", collision_pipeline=None
            )
        ],
        iterations=1,
    )
    cfg.sim.physics.solver_cfg = coupled
    cfg.sim.physics.collision_cfg.soft_contact_max = 0
    top = cfg.course.granular_top
    cfg.scene.sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            lower=(8.25, -0.6, top - 0.03),
            upper=(11.5, 0.6, top),
            voxel_size=0.01,
            particles_per_cell=2,
            particle_placement="cell_center",
            jitter=0.002,
            material=MPMParticleMaterialCfg(density=1600.0, friction=0.6, yield_pressure=1e5),
            visual_color=(0.70, 0.55, 0.32),
        ),
    )
    cfg.scene.gravel = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Gravel",
        spawn=MPMGridCfg(
            lower=(5.5, -0.6, top - 0.03),
            upper=(8.25, 0.6, top),
            voxel_size=0.01,
            particles_per_cell=2,
            particle_placement="cell_center",
            jitter=0.002,
            material=MPMParticleMaterialCfg(density=1800.0, friction=0.85, yield_pressure=1e6),
            visual_color=(0.38, 0.37, 0.34),
        ),
    )


def create_pbd_beds(stage, cfg: CourseCfg):
    """Author rigid tray contacts and two native PhysX solid-particle layers."""
    from omni.physx.scripts import particleUtils, physicsUtils  # noqa: PLC0415
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    bottom = cfg.granular_top - 0.03
    material = "/Course/RigidMaterials/granular_container"
    spawn_physics_material(material, cfg.tray_material)
    boxes = {
        "floor": ((8.5, 0.0, bottom - 0.05), (6.2, 1.4, 0.1)),
        "left_wall": ((8.5, -0.65, bottom + 0.015), (6.2, 0.1, 0.03)),
        "right_wall": ((8.5, 0.65, bottom + 0.015), (6.2, 0.1, 0.03)),
        "entry_wall": ((5.45, 0.0, bottom + 0.015), (0.1, 1.2, 0.03)),
        "exit_wall": ((11.55, 0.0, bottom + 0.015), (0.1, 1.2, 0.03)),
        "gravel_floor_riser": ((6.825, 0.0, bottom + 0.005), (2.85, 1.4, 0.01)),
    }
    for name, (center, size) in boxes.items():
        cube_cfg = sim_utils.CuboidCfg(
            size=size,
            visible=name == "gravel_floor_riser",
            collision_props=cfg.tray_collision_props,
            visual_material=cfg.visual_materials[name] if name == "gravel_floor_riser" else None,
        )
        path = f"/Course/{name}"
        cube_cfg.func(path, cube_cfg, translation=center)
        sim_utils.bind_physics_material(path, material)

    particleUtils.add_pbd_particle_material(
        stage, "/Course/PBDMaterial", friction=0.8, particle_friction_scale=1.0, damping=0.1, cohesion=0.0
    )
    rng = np.random.default_rng(42)
    visual_rng = np.random.default_rng(17)
    for name, x0, x1, depth, diameter, density in (
        ("gravel", 5.5, 8.25, 0.02, 0.0075, 1800.0),
        ("sand", 8.25, 11.5, 0.03, 0.005, 1600.0),
    ):
        radius = diameter / 2
        system_path = f"/Course/{name}_ParticleSystem"
        system = particleUtils.add_physx_particle_system(
            stage,
            system_path,
            contact_offset=1.2 * radius,
            rest_offset=radius,
            particle_contact_offset=1.2 * radius,
            solid_rest_offset=radius,
            fluid_rest_offset=0.6 * radius,
            solver_position_iterations=16,
            max_velocity=5.0,
            enable_ccd=True,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
        )
        physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), "/Course/PBDMaterial")
        lower = np.array([x0, -0.6, cfg.granular_top - depth])
        upper = np.array([x1, 0.6, cfg.granular_top])
        count = np.floor((upper - lower) / diameter + 1e-6).astype(int)
        axes = [lower[i] + radius + diameter * np.arange(count[i]) for i in range(3)]
        positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        # Shift whole rows and trim boundary grains to avoid initial overlaps.
        positions[..., 0] += rng.uniform(-radius, radius, (1, count[1], count[2]))
        positions[..., 1] += rng.uniform(-radius, radius, (1, 1, count[2]))
        positions = positions.reshape(-1, 3)
        positions = positions[
            np.all((positions >= lower + radius - 1e-8) & (positions <= upper - radius + 1e-8), axis=1)
        ].astype(np.float32)
        points = UsdGeom.Points.Define(stage, f"/Course/{name}_particles")
        points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
        points.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(positions)))
        width_range = (1.0, 1.0) if name == "gravel" else (1.1, 1.3)
        widths = diameter * visual_rng.uniform(*width_range, len(positions)).astype(np.float32)
        points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(widths))
        material_cfg = cfg.visual_materials[f"{name}_particles"]
        colors = np.asarray(material_cfg.diffuse_color) * visual_rng.uniform(0.65, 1.15, (len(positions), 1))
        points.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        points.GetDisplayColorPrimvar().SetInterpolation("vertex")
        visual = f"/Course/Looks/{name}_particles"
        material_cfg.func(visual, material_cfg)
        reader = UsdShade.Shader.Define(stage, f"{visual}/GrainColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        surface = UsdShade.Shader(stage.GetPrimAtPath(f"{visual}/Shader"))
        surface.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
        sim_utils.bind_visual_material(points.GetPath(), visual)
        particleUtils.configure_particle_set(
            points.GetPrim(),
            system_path,
            self_collision=True,
            fluid=False,
            particle_group=0,
            mass=len(positions) * density * diameter**3,
            density=0.0,
        )
