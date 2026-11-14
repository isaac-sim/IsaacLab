"""Export the fixed G1 PhysX particle scene without launching a simulator."""

import argparse
from pathlib import Path

import numpy as np
from export_g1_courses import GRANULAR_TOP, export_course
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from physx_usd_schemas import PhysxSchema

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material


def export_particles(output: Path) -> Path:
    """Save the initial particle state, systems, materials, and PhysX tray contacts."""
    source = output / "granular.usdc"
    if not source.exists():
        export_course(output, "granular")
    path = output / "granular_physx.usdc"
    Usd.Stage.Open(str(source)).Flatten().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    stage.GetPrimAtPath("/Course/granular_container/mesh").RemoveAPI(UsdPhysics.CollisionAPI)
    with sim_utils.use_stage(stage):
        author_particle_beds(stage)
    stage.GetRootLayer().Save()
    return path


def author_particle_beds(stage: Usd.Stage) -> None:
    """Author deterministic PBD beds; subsequent motion is computed by PhysX at runtime."""
    bottom = GRANULAR_TOP - 0.03
    material_path = "/Course/RigidMaterials/granular_container"
    spawn_physics_material(
        material_path, UsdPhysicsRigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6, restitution=0.0)
    )
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
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0),
            ],
            visual_material=(
                sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.19, 0.20), roughness=0.96)
                if name == "gravel_floor_riser"
                else None
            ),
        )
        path = f"/Course/{name}"
        cube_cfg.func(path, cube_cfg, translation=center)
        sim_utils.bind_physics_material(path, material_path)

    material = UsdShade.Material.Define(stage, "/Course/PBDMaterial")
    material_api = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
    material_api.CreateFrictionAttr(0.8)
    material_api.CreateParticleFrictionScaleAttr(1.0)
    material_api.CreateDampingAttr(0.1)
    material_api.CreateCohesionAttr(0.0)
    rng = np.random.default_rng(42)
    visual_rng = np.random.default_rng(17)
    for name, start, end, depth, diameter, density, color in (
        ("gravel", 5.5, 8.25, 0.02, 0.0075, 1800.0, (0.38, 0.37, 0.34)),
        ("sand", 8.25, 11.5, 0.03, 0.005, 1600.0, (0.70, 0.55, 0.32)),
    ):
        radius = diameter / 2
        system_path = f"/Course/{name}_ParticleSystem"
        system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
        system.CreateContactOffsetAttr(1.2 * radius)
        system.CreateRestOffsetAttr(radius)
        system.CreateParticleContactOffsetAttr(1.2 * radius)
        system.CreateSolidRestOffsetAttr(radius)
        system.CreateFluidRestOffsetAttr(0.6 * radius)
        system.CreateSolverPositionIterationCountAttr(16)
        system.CreateMaxVelocityAttr(5.0)
        system.CreateEnableCCDAttr(True)
        system.CreateGlobalSelfCollisionEnabledAttr(True)
        system.CreateNonParticleCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(material, materialPurpose="physics")
        lower = np.array([start, -0.6, GRANULAR_TOP - depth])
        upper = np.array([end, 0.6, GRANULAR_TOP])
        count = np.floor((upper - lower) / diameter + 1e-6).astype(int)
        axes = [lower[axis] + radius + diameter * np.arange(count[axis]) for axis in range(3)]
        positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
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
        colors = np.asarray(color) * visual_rng.uniform(0.65, 1.15, (len(positions), 1))
        points.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        points.GetDisplayColorPrimvar().SetInterpolation("vertex")
        visual_path = f"/Course/Looks/{name}_particles"
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.96)
        material_cfg.func(visual_path, material_cfg)
        reader = UsdShade.Shader.Define(stage, f"{visual_path}/GrainColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        surface = UsdShade.Shader(stage.GetPrimAtPath(f"{visual_path}/Shader"))
        surface.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
        sim_utils.bind_visual_material(points.GetPath(), visual_path)
        particle_set = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
        particle_api = PhysxSchema.PhysxParticleAPI(particle_set)
        particle_api.CreateSelfCollisionAttr(True)
        particle_set.CreateFluidAttr(False)
        particle_api.CreateParticleGroupAttr(0)
        particle_api.CreateParticleSystemRel().SetTargets([system_path])
        mass_api = UsdPhysics.MassAPI.Apply(points.GetPrim())
        mass_api.CreateMassAttr(len(positions) * density * diameter**3)
        mass_api.CreateDensityAttr(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_particles(args.output))


if __name__ == "__main__":
    main()
