# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn deformable prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/deformables.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="This script demonstrates how to spawn deformable prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import random
import torch
import tqdm

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = torch.rand(num_origins) + 1.0
    # return the origins
    return env_origins.tolist()


def design_scene():
    """Designs the scene by spawning ground plane, light, and deformable meshes."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/light", cfg_light)

    # create new xform prims for all objects to be spawned under
    origins = define_origins(num_origins=4, spacing=5.5)
    for idx, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{idx:02d}", "Xform", translation=origin)

    # spawn a red cone
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=0.25,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_capsule = sim_utils.MeshCapsuleCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    # create a dictionary of all the objects to be spawned
    objects_cfg = {
        "sphere": cfg_sphere,
        "cuboid": cfg_cuboid,
        "cylinder": cfg_cylinder,
        "capsule": cfg_capsule,
        "cone": cfg_cone,
    }

    # Create separate groups of deformable objects
    origins = define_origins(num_origins=25, spacing=0.5)
    print("[INFO]: Spawning objects...")
    # Iterate over all the origins and randomly spawn objects
    for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins)):
        # randomly select an object to spawn
        obj_name = random.choice(list(objects_cfg.keys()))
        obj_cfg = objects_cfg[obj_name]
        # randomize the young modulus (somewhere between a Silicone 30 and Silicone 70)
        obj_cfg.physics_material.youngs_modulus = random.uniform(0.7e6, 3.3e6)
        # randomize the poisson's ratio
        obj_cfg.physics_material.poissons_ratio = random.uniform(0.25, 0.5)
        # randomize the color
        obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())
        # spawn the object
        obj_cfg.func(f"/World/Origin.*/Object{idx:02d}", obj_cfg, translation=origin)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([8.0, 8.0, 6.0], [0.0, 0.0, 0.0])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
