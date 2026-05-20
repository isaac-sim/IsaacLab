# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn deformable prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/deformables.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="This script demonstrates how to spawn deformable prims into the scene.")
parser.add_argument("--backend", type=str, default="physx", choices=["physx", "newton"], help="Physics backend.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import numpy as np
import torch
import tqdm

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

if args_cli.backend == "newton":
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg as DeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import (
        NewtonDeformableBodyMaterialCfg as VolumeDeformableMaterialCfg,
    )
    from isaaclab_newton.sim.spawners.materials import (
        NewtonSurfaceDeformableBodyMaterialCfg as SurfaceDeformableMaterialCfg,
    )
else:
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg as DeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import (
        PhysxDeformableBodyMaterialCfg as VolumeDeformableMaterialCfg,
    )
    from isaaclab_physx.sim.spawners.materials import (
        PhysxSurfaceDeformableBodyMaterialCfg as SurfaceDeformableMaterialCfg,
    )


def define_origins(num_origins: int, radius: float = 2.0, center_height: float = 3.0) -> list[list[float]]:
    """Defines origins distributed on the surface of a sphere, sampled according to a Fibonacci lattice.

    Args:
        num_origins: Number of points to place.
        radius: Radius of the sphere [m].
        center_height: Height of the sphere center above ground [m].
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    env_origins = torch.zeros(num_origins, 3)
    for i in range(num_origins):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / num_origins)
        env_origins[i, 0] = radius * np.cos(theta) * np.sin(phi)
        env_origins[i, 1] = radius * np.sin(theta) * np.sin(phi)
        env_origins[i, 2] = radius * np.cos(phi) + center_height
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/light", cfg_light)

    # spawn a red cone
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=0.4,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
    )
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.6, 0.6, 0.6),
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.25,
        height=0.5,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
    )
    cfg_capsule = sim_utils.MeshCapsuleCfg(
        radius=0.35,
        height=0.5,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
    )
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.35,
        height=0.75,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
    )
    cfg_cloth = sim_utils.MeshRectangleCfg(
        size=(1.5, 1.0),
        resolution=(21, 21),
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=SurfaceDeformableMaterialCfg(),
    )
    cfg_usd = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=VolumeDeformableMaterialCfg(),
        scale=[0.05, 0.05, 0.05],
    )
    # create a dictionary of all the objects to be spawned
    objects_cfg = {
        "sphere": cfg_sphere,
        "cuboid": cfg_cuboid,
        "cylinder": cfg_cylinder,
        "capsule": cfg_capsule,
        "cone": cfg_cone,
        "cloth": cfg_cloth,
        "usd": cfg_usd,
    }

    # Create separate groups of deformable objects
    origins = define_origins(num_origins=12, radius=1.5, center_height=2.0)
    print("[INFO]: Spawning objects...")
    # Iterate over all the origins, spawn objects, and create a view for all the deformables
    # note: since we manually spawned random deformable meshes above, we don't need to
    #   specify the spawn configuration for the deformable object
    scene_entities = {}
    for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins)):
        # randomly select an object to spawn
        obj_name = random.choice(list(objects_cfg.keys()))
        obj_cfg = objects_cfg[obj_name]
        # randomize the deformable material stiffness
        if args_cli.backend == "newton" and obj_name == "cloth":
            obj_cfg.physics_material.tri_ke = random.uniform(5e3, 5e4)
            obj_cfg.physics_material.tri_ka = random.uniform(5e3, 5e4)
        else:
            youngs_modulus = random.uniform(5e5, 1e8)
            poissons_ratio = random.uniform(0.25, 0.45)
            if args_cli.backend == "newton":
                obj_cfg.physics_material.k_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
                obj_cfg.physics_material.k_lambda = (
                    youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
                )
            else:
                obj_cfg.physics_material.youngs_modulus = youngs_modulus
                obj_cfg.physics_material.poissons_ratio = poissons_ratio
        # randomize the color
        obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())
        # spawn the object, separate groups for surface and volume deformables
        if obj_name in ["cloth"]:
            prim_path = f"/World/Origin/Surface{idx:02d}"
            cfg = DeformableObjectCfg(
                prim_path=prim_path,
                spawn=obj_cfg,
                init_state=DeformableObjectCfg.InitialStateCfg(pos=origin),
            )
            scene_entities[f"Surface{idx:02d}"] = DeformableObject(cfg=cfg)
        else:
            prim_path = f"/World/Origin/Volume{idx:02d}"
            cfg = DeformableObjectCfg(
                prim_path=prim_path,
                spawn=obj_cfg,
                init_state=DeformableObjectCfg.InitialStateCfg(pos=origin),
            )
            scene_entities[f"Volume{idx:02d}"] = DeformableObject(cfg=cfg)

    # return the scene information
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % int(3.0 / sim_dt) == 0:
            # reset counters
            count = 0
            # reset deformable object state
            for _, deform_body in enumerate(entities.values()):
                # root state
                nodal_state = deform_body.data.default_nodal_state_w.torch.clone()
                deform_body.write_nodal_state_to_sim_index(nodal_state)
                # reset the internal state
                deform_body.reset()
            print("[INFO]: Resetting deformable object state...")
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for deform_body in entities.values():
            deform_body.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    if args_cli.backend == "newton":
        from isaaclab_newton.physics import NewtonCfg

        from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

        physics_cfg = NewtonCfg(solver_cfg=VBDSolverCfg(iterations=10), num_substeps=4)
    else:
        from isaaclab_physx.physics import PhysxCfg

        physics_cfg = PhysxCfg()
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, physics=physics_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 4.0, 3.0], [0.5, 0.5, 0.0])

    # Design scene by adding assets to it
    scene_entities, _ = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities)
    print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
