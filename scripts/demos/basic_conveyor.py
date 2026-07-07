# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab port of Newton's basic conveyor example.

The scene is a baggage-claim style conveyor built from one rotating annular
belt, two static annular rails, and 18 dynamic bags. Its geometry, materials,
solver settings, initial state, and prescribed belt motion match Newton's
``example_basic_conveyor.py``.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/basic_conveyor.py --visualizer newton
    ./isaaclab.sh -p scripts/demos/basic_conveyor.py --visualizer newton --solver vbd
"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton baggage-claim conveyor demo for Isaac Lab.")
parser.add_argument("--solver", choices=["xpbd", "vbd"], default="xpbd", help="Newton solver backend to use.")
parser.add_argument(
    "--belt_speed",
    "--belt-speed",
    dest="belt_speed",
    type=float,
    default=0.75,
    help="Conveyor tangential speed [m/s].",
)
parser.add_argument(
    "--max_steps",
    "--max-steps",
    dest="max_steps",
    type=int,
    default=-1,
    help="Stop after this many rendered frames; negative runs forever.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

import math
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sim import schemas
from isaaclab.sim.utils import (
    bind_physics_material,
    bind_visual_material,
    clone,
    create_prim,
    get_current_stage,
    get_first_matching_child_prim,
)
from isaaclab.utils.configclass import configclass

FPS = 100
FRAME_DT = 1.0 / FPS
SIM_SUBSTEPS = 10
SIM_DT = FRAME_DT / SIM_SUBSTEPS

BELT_CENTER_Z = 0.55
BELT_RING_RADIUS = 1.8
BELT_HALF_WIDTH = 0.24
BELT_HALF_THICKNESS = 0.04
BELT_MESH_SEGMENTS = 96
RAIL_WALL_THICKNESS = 0.035
RAIL_HEIGHT = 0.16
RAIL_BASE_OVERLAP = 0.01
BAG_COUNT = 18
BAG_LANE_OFFSETS = (-0.12, 0.0, 0.12)
BAG_DROP_CLEARANCE = 0.035

CAMERA_EYE = (2.7, -1.3, 5.0)
CAMERA_TARGET = (0.350768448, -0.444949642, 0.669872981)

BELT_COLOR = (0.09, 0.09, 0.09)
RAIL_COLOR = (0.66, 0.69, 0.74)
GROUND_COLOR = (0.125, 0.125, 0.15)
SHAPE_COLOR_PALETTE = (
    (68 / 255, 119 / 255, 170 / 255),
    (102 / 255, 204 / 255, 238 / 255),
    (34 / 255, 136 / 255, 51 / 255),
    (204 / 255, 187 / 255, 68 / 255),
    (238 / 255, 102 / 255, 119 / 255),
    (170 / 255, 51 / 255, 119 / 255),
    (238 / 255, 153 / 255, 51 / 255),
    (0 / 255, 153 / 255, 136 / 255),
)


def create_annular_prism_mesh(
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the vertices and triangle faces of a closed ring prism."""
    if segments < 3:
        raise ValueError("segments must be >= 3")
    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("Expected 0 < inner_radius < outer_radius")
    if z_max <= z_min:
        raise ValueError("Expected z_max > z_min")

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    inner_top = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    outer_top = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    inner_bottom = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )
    outer_bottom = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )
    vertices = np.vstack((inner_top, outer_top, inner_bottom, outer_bottom)).astype(np.float32)

    outer_top_offset = segments
    inner_bottom_offset = 2 * segments
    outer_bottom_offset = 3 * segments
    indices: list[int] = []
    for i in range(segments):
        j = (i + 1) % segments
        inner_top_i, inner_top_j = i, j
        outer_top_i, outer_top_j = outer_top_offset + i, outer_top_offset + j
        inner_bottom_i, inner_bottom_j = inner_bottom_offset + i, inner_bottom_offset + j
        outer_bottom_i, outer_bottom_j = outer_bottom_offset + i, outer_bottom_offset + j

        indices.extend((inner_top_i, outer_top_i, outer_top_j, inner_top_i, outer_top_j, inner_top_j))
        indices.extend(
            (
                inner_bottom_i,
                inner_bottom_j,
                outer_bottom_j,
                inner_bottom_i,
                outer_bottom_j,
                outer_bottom_i,
            )
        )
        indices.extend(
            (
                outer_bottom_i,
                outer_bottom_j,
                outer_top_j,
                outer_bottom_i,
                outer_top_j,
                outer_top_i,
            )
        )
        indices.extend(
            (
                inner_bottom_i,
                inner_top_i,
                inner_top_j,
                inner_bottom_i,
                inner_top_j,
                inner_bottom_j,
            )
        )

    return vertices, np.asarray(indices, dtype=np.int32).reshape((-1, 3))


def author_preview_surface(
    prim_path: str,
    material_path: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> None:
    """Author and bind a portable USD preview material without requiring Kit."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot bind a material to missing prim: {prim_path}")

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


@clone
def spawn_demo_mesh(
    prim_path: str,
    cfg: DemoMeshCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_kwargs,
):
    """Spawn an arbitrary triangle mesh used by the belt and rails."""
    stage = get_current_stage()
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    geometry_path = f"{prim_path}/geometry"
    mesh_path = f"{geometry_path}/mesh"
    create_prim(geometry_path, prim_type="Xform", stage=stage)
    mesh_prim = create_prim(
        mesh_path,
        prim_type="Mesh",
        attributes={
            "points": np.asarray(cfg.vertices, dtype=np.float32),
            "faceVertexIndices": np.asarray(cfg.faces, dtype=np.int32).reshape(-1),
            "faceVertexCounts": np.full(len(cfg.faces), 3, dtype=np.int32),
            "subdivisionScheme": "none",
        },
        stage=stage,
    )
    author_preview_surface(
        mesh_path,
        f"{geometry_path}/material",
        cfg.display_color,
        cfg.roughness,
        cfg.metallic,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_path, cfg.collision_props, stage=stage)
    if cfg.mesh_collision_props is not None:
        schemas.define_mesh_collision_properties(mesh_path, cfg.mesh_collision_props, stage=stage)

    if cfg.visual_material is not None:
        visual_material_path = cfg.visual_material_path
        if not visual_material_path.startswith("/"):
            visual_material_path = f"{geometry_path}/{visual_material_path}"
        cfg.visual_material.func(visual_material_path, cfg.visual_material)
        bind_visual_material(mesh_path, visual_material_path, stage=stage)

    if cfg.physics_material is not None:
        physics_material_path = cfg.physics_material_path
        if not physics_material_path.startswith("/"):
            physics_material_path = f"{geometry_path}/{physics_material_path}"
        cfg.physics_material.func(physics_material_path, cfg.physics_material)
        bind_physics_material(mesh_path, physics_material_path, stage=stage)

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    if cfg.diagonal_inertia is not None:
        mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*cfg.diagonal_inertia))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    return mesh_prim.GetParent().GetParent()


@configclass
class DemoMeshCfg(sim_utils.MeshCfg):
    """Demo-local arbitrary triangle-mesh asset configuration."""

    func: Callable | str = spawn_demo_mesh
    vertices: list[list[float]] = MISSING
    faces: list[list[int]] = MISSING
    mesh_collision_props: sim_utils.NewtonMeshCollisionPropertiesCfg | None = None
    diagonal_inertia: tuple[float, float, float] | None = None
    display_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    roughness: float = 0.5
    metallic: float = 0.0


def rigid_material(
    friction: float,
    contact_stiffness: float,
    contact_damping: float,
    restitution: float = 0.0,
) -> sim_utils.NewtonMaterialPropertiesCfg:
    """Create a Newton rigid material matching a ``ShapeConfig``."""
    return sim_utils.NewtonMaterialPropertiesCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=restitution,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
    )


def create_visualizer_cfgs():
    """Create the Newton viewer configuration when it was requested."""
    if not args_cli.visualizer or "newton" not in args_cli.visualizer:
        return []

    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    return [NewtonVisualizerCfg(show_contacts=True)]


def create_sim_cfg() -> sim_utils.SimulationCfg:
    """Create simulation settings matching the Newton example."""
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg, XPBDSolverCfg

    if args_cli.solver == "vbd":
        from isaaclab_contrib.deformable import VBDSolverCfg

        solver_cfg = VBDSolverCfg(iterations=5, rigid_body_contact_buffer_size=512)
    else:
        solver_cfg = XPBDSolverCfg()

    return sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.1),
            simplify_meshes=False,
        ),
    )


def create_bag_collection_cfg() -> RigidObjectCollectionCfg:
    """Create the 18 dynamic bags from the Newton example."""
    belt_top_z = BELT_CENTER_Z + BELT_HALF_THICKNESS
    bag_angles = np.linspace(0.0, 2.0 * math.pi, BAG_COUNT, endpoint=False, dtype=np.float32)
    bags: dict[str, RigidObjectCfg] = {}

    for i, angle in enumerate(bag_angles):
        radius = BELT_RING_RADIUS + BAG_LANE_OFFSETS[i % len(BAG_LANE_OFFSETS)]
        bag_x = radius * math.cos(angle)
        bag_y = radius * math.sin(angle)
        bag_yaw = angle + 0.5 * math.pi
        shape_type = i % 3
        bag_vertical_extent = (0.08, 0.08, 0.11)[shape_type]
        bag_z = belt_top_z + bag_vertical_extent + BAG_DROP_CLEARANCE

        shape_kwargs = {
            "rigid_props": sim_utils.NewtonRigidBodyPropertiesCfg(rigid_body_enabled=True),
            "mass_props": sim_utils.MassPropertiesCfg(mass=2.8 + 0.1 * i),
            "collision_props": sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
            "physics_material": rigid_material(1.0, 1.0e5, 0.0),
        }
        if shape_type == 0:
            spawn = sim_utils.CuboidCfg(size=(0.36, 0.24, 0.16), **shape_kwargs)
        elif shape_type == 1:
            spawn = sim_utils.CapsuleCfg(radius=0.08, height=0.30, axis="X", **shape_kwargs)
        else:
            spawn = sim_utils.SphereCfg(radius=0.11, **shape_kwargs)

        bags[f"bag_{i}"] = RigidObjectCfg(
            prim_path=f"/World/Bags/Bag_{i}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bag_x, bag_y, bag_z),
                rot=(0.0, 0.0, math.sin(0.5 * bag_yaw), math.cos(0.5 * bag_yaw)),
            ),
        )

    return RigidObjectCollectionCfg(rigid_objects=bags)


def add_belt_collision_filters() -> None:
    """Disable the belt-ground and belt-rail collision pairs."""
    stage = get_current_stage()
    belt_mesh = stage.GetPrimAtPath("/World/ConveyorBelt/geometry/mesh")
    ground_plane = get_first_matching_child_prim(
        "/World/Ground", predicate=lambda prim: prim.GetTypeName() == "Plane", stage=stage
    )
    if not belt_mesh.IsValid() or ground_plane is None:
        raise RuntimeError("Failed to resolve conveyor belt or ground collision prim.")

    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(belt_mesh).CreateFilteredPairsRel()
    filtered_pairs.AddTarget(ground_plane.GetPath())
    filtered_pairs.AddTarget(Sdf.Path("/World/InnerRail/geometry/mesh"))
    filtered_pairs.AddTarget(Sdf.Path("/World/OuterRail/geometry/mesh"))


def design_scene() -> tuple[RigidObject, RigidObjectCollection]:
    """Spawn the conveyor scene and return its dynamic assets."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        color=GROUND_COLOR,
        physics_material=rigid_material(1.0, 2.5e3, 100.0),
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/DomeLight", light_cfg)

    island_cfg = sim_utils.CylinderCfg(
        radius=0.9,
        height=0.16,
    )
    island_cfg.func(
        "/World/CenterIsland",
        island_cfg,
        translation=(0.0, 0.0, BELT_CENTER_Z - BELT_HALF_THICKNESS - 0.08),
    )
    author_preview_surface(
        "/World/CenterIsland/geometry/mesh",
        "/World/CenterIsland/geometry/material",
        SHAPE_COLOR_PALETTE[1],
        roughness=0.5,
        metallic=0.0,
    )

    belt_inner_radius = BELT_RING_RADIUS - BELT_HALF_WIDTH
    belt_outer_radius = BELT_RING_RADIUS + BELT_HALF_WIDTH
    belt_vertices, belt_faces = create_annular_prism_mesh(
        belt_inner_radius,
        belt_outer_radius,
        -BELT_HALF_THICKNESS,
        BELT_HALF_THICKNESS,
        BELT_MESH_SEGMENTS,
    )
    rail_inner_vertices, rail_inner_faces = create_annular_prism_mesh(
        belt_inner_radius - RAIL_WALL_THICKNESS,
        belt_inner_radius,
        BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
        BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
        BELT_MESH_SEGMENTS,
    )
    rail_outer_vertices, rail_outer_faces = create_annular_prism_mesh(
        belt_outer_radius,
        belt_outer_radius + RAIL_WALL_THICKNESS,
        BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
        BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
        BELT_MESH_SEGMENTS,
    )

    rail_material = rigid_material(0.8, 1.0e5, 0.0)
    for prim_path, vertices, faces in (
        ("/World/InnerRail", rail_inner_vertices, rail_inner_faces),
        ("/World/OuterRail", rail_outer_vertices, rail_outer_faces),
    ):
        rail_cfg = DemoMeshCfg(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
            physics_material=rail_material,
            display_color=RAIL_COLOR,
            roughness=0.5,
            metallic=0.9,
        )
        rail_cfg.func(prim_path, rail_cfg, translation=(0.0, 0.0, BELT_CENTER_Z))

    belt_mass = 15.0
    belt_radii_sum_sq = belt_inner_radius**2 + belt_outer_radius**2
    belt_i_transverse = belt_mass / 12.0 * (3.0 * belt_radii_sum_sq + (2.0 * BELT_HALF_THICKNESS) ** 2)
    belt_i_axial = 0.5 * belt_mass * belt_radii_sum_sq
    belt_cfg = RigidObjectCfg(
        prim_path="/World/ConveyorBelt",
        spawn=DemoMeshCfg(
            vertices=belt_vertices.tolist(),
            faces=belt_faces.tolist(),
            rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=belt_mass),
            collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
            physics_material=rigid_material(1.2, 1.0e5, 0.0),
            display_color=BELT_COLOR,
            roughness=0.94,
            metallic=0.02,
            diagonal_inertia=(belt_i_transverse, belt_i_transverse, belt_i_axial),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, BELT_CENTER_Z),
            ang_vel=(0.0, 0.0, args_cli.belt_speed / BELT_RING_RADIUS),
        ),
    )
    belt = RigidObject(belt_cfg)
    bags = RigidObjectCollection(create_bag_collection_cfg())
    for i in range(BAG_COUNT):
        author_preview_surface(
            f"/World/Bags/Bag_{i}/geometry/mesh",
            f"/World/Bags/Bag_{i}/geometry/material",
            SHAPE_COLOR_PALETTE[(5 + i) % len(SHAPE_COLOR_PALETTE)],
            roughness=0.5,
            metallic=0.0,
        )
    add_belt_collision_filters()
    return belt, bags


def write_belt_state(belt: RigidObject, sim_time: float) -> None:
    """Write the belt's prescribed pose and angular velocity."""
    angular_speed = args_cli.belt_speed / BELT_RING_RADIUS
    angle = angular_speed * sim_time
    pose = torch.tensor(
        [[0.0, 0.0, BELT_CENTER_Z, 0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)]],
        dtype=torch.float32,
        device=belt.device,
    )
    velocity = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, angular_speed]], dtype=torch.float32, device=belt.device)
    belt.write_root_link_pose_to_sim_index(root_pose=pose)
    belt.write_root_link_velocity_to_sim_index(root_velocity=velocity)


def keep_running(sim: sim_utils.SimulationContext, frame_count: int) -> bool:
    """Return whether another displayed frame should be simulated."""
    if args_cli.max_steps >= 0 and frame_count >= args_cli.max_steps:
        return False
    return sim.is_headless_or_exist_active_visualizer()


def run_simulator(sim: sim_utils.SimulationContext, belt: RigidObject, bags: RigidObjectCollection) -> None:
    """Run ten solver steps per displayed frame, matching the Newton example."""
    sim_time = 0.0
    frame_count = 0
    while keep_running(sim, frame_count):
        for _ in range(SIM_SUBSTEPS):
            write_belt_state(belt, sim_time)
            belt.write_data_to_sim()
            bags.write_data_to_sim()
            sim.step(render=False)
            belt.update(SIM_DT)
            bags.update(SIM_DT)
            sim_time += SIM_DT

        if sim.is_rendering:
            sim.render()
        frame_count += 1


def validate_final_state(belt: RigidObject, bags: RigidObjectCollection) -> None:
    """Check the same final-state invariants as the Newton example."""
    belt_z = float(belt.data.root_link_pose_w.torch[0, 2])
    if abs(belt_z - BELT_CENTER_Z) >= 0.15:
        raise AssertionError(f"Belt body drifted off the conveyor plane: z={belt_z:.4f}")

    bag_poses = bags.data.body_link_pose_w.torch[0].tolist()
    for bag_index, pose in enumerate(bag_poses):
        x, y, z = pose[:3]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise AssertionError(f"Bag {bag_index} has non-finite pose values.")
        if z <= -0.5:
            raise AssertionError(f"Bag body {bag_index} fell through the floor: z={z:.4f}")
        if abs(x) >= 4.0 or abs(y) >= 4.0:
            raise AssertionError(f"Bag body {bag_index} left the scene bounds: ({x:.3f}, {y:.3f})")


def main() -> None:
    """Launch and run the Isaac Lab conveyor demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        belt, bags = design_scene()
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        print(
            f"[INFO]: Isaac Lab Newton conveyor ready with {BAG_COUNT} bags, "
            f"{args_cli.solver.upper()} solver, and belt speed {args_cli.belt_speed:g} m/s.",
            flush=True,
        )
        run_simulator(sim, belt, bags)
        validate_final_state(belt, bags)


if __name__ == "__main__":
    main()
