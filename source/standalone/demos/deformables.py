# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils


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

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 0.0, 0.6))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-0.5, 0.0, 0.3))

    # spawn a green cone with colliders and rigid body
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=0.25,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_sphere.func(
        "/World/Objects/Sphere",
        cfg_sphere,
        translation=(0.0, 0.0, 0.6),
    )

    # spawn a blue cuboid with deformable body
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg_cuboid.func("/World/Objects/Cuboid", cfg_cuboid, translation=(0.5, 0.0, 0.3))
    cfg_cuboid.func("/World/Objects/Cuboid2", cfg_cuboid, translation=(1.0, 0.0, 0.6))


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, 2.0, 1.5], [0.0, 0.0, 0.0])

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
