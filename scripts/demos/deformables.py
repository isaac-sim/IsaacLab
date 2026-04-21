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

# deformables supported in PhysX
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyMaterialCfg, DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


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
        physics_material=DeformableBodyMaterialCfg(),
    )
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.6, 0.6, 0.6),
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=DeformableBodyMaterialCfg(),
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.25,
        height=0.5,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=DeformableBodyMaterialCfg(),
    )
    cfg_capsule = sim_utils.MeshCapsuleCfg(
        radius=0.35,
        height=0.5,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=DeformableBodyMaterialCfg(),
    )
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.35,
        height=0.75,
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=DeformableBodyMaterialCfg(),
    )
    cfg_cloth = sim_utils.MeshSquareCfg(
        size=1.5,
        resolution=(21, 21),
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=SurfaceDeformableBodyMaterialCfg(),
    )
    cfg_usd = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=DeformableBodyMaterialCfg(),
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
    num_volumes = 0
    num_surfaces = 0
    # Iterate over all the origins and randomly spawn objects
    for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins)):
        # randomly select an object to spawn
        obj_name = random.choice(list(objects_cfg.keys()))
        obj_cfg = objects_cfg[obj_name]
        # randomize the young modulus
        obj_cfg.physics_material.youngs_modulus = random.uniform(5e5, 1e8)
        # higher mesh resolution causes instability at low stiffness
        if obj_name in ["sphere", "capsule", "cloth", "usd"]:
            obj_cfg.physics_material.youngs_modulus = random.uniform(1e8, 5e9)
        # randomize the poisson's ratio
        obj_cfg.physics_material.poissons_ratio = random.uniform(0.25, 0.45)
        # randomize the color
        obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())
        # spawn the object, separate groups for surface and volume deformables
        if obj_name in ["cloth"]:
            obj_cfg.func(f"/World/Origin/Surface{idx:02d}", obj_cfg, translation=origin)
            num_surfaces += 1
        else:
            obj_cfg.func(f"/World/Origin/Volume{idx:02d}", obj_cfg, translation=origin)
            num_volumes += 1

    # create a view for all the deformables, separate views for volume and surface deformables
    # note: since we manually spawned random deformable meshes above, we don't need to
    #   specify the spawn configuration for the deformable object
    scene_entities = {}
    if num_volumes > 0:
        cfg = DeformableObjectCfg(
            prim_path="/World/Origin/Volume.*",
            spawn=None,
            init_state=DeformableObjectCfg.InitialStateCfg(),
        )
        volume_deformable_object = DeformableObject(cfg=cfg)
        scene_entities["volume_deformable_object"] = volume_deformable_object
    if num_surfaces > 0:
        cfg = DeformableObjectCfg(
            prim_path="/World/Origin/Surface.*",
            spawn=None,
            init_state=DeformableObjectCfg.InitialStateCfg(),
        )
        surface_deformable_object = DeformableObject(cfg=cfg)
        scene_entities["surface_deformable_object"] = surface_deformable_object

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
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
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
