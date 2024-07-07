# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run IsaacSim via the AppLauncher

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/launch_app.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html?highlight=simulationapp#omni.isaac.kit.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn a cuboid
    # no rigid body properties defined, penetrating this one!!!
    # cfg_cuboid = sim_utils.CuboidCfg(
    #     size=[args_cli.size] * 3,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    # )
    # # Spawn cuboid, altering translation on the z-axis to scale to its size
    # cfg_cuboid.func("/World/Objects/cuboid", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))

    # not penetrating, but cannot adjust size
    # cube_cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    # )
    # cube_cfg.func("/World/Objects/cube", cube_cfg, translation=(0.0, 0.0, args_cli.size / 2))

    # -- multi-color cube
    # cfg_mc_cube = sim_utils.UsdFileCfg(usd_path=f"source/extensions/omni.isaac.lab_assets/data/Props/CubeMultiColor/cube_multicolor.usd")
    # cfg_mc_cube.func("/World/multicolor_cube", cfg_mc_cube)

    cfg_cracker_box = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd")
    cfg_cracker_box.func("/World/Objects/cracker_box", cfg_cracker_box, translation=(0, 0, 1.05))

    cfg_sugar_box = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd")
    cfg_sugar_box.func("/World/Objects/sugar_box", cfg_sugar_box, translation=(0, 0.1, 1.05))

    cfg_tomato_soup_can = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd")
    cfg_tomato_soup_can.func("/World/Objects/tomato_soup_can", cfg_tomato_soup_can, translation=(-0.2, -0.2, 1.05))

    cfg_mustard_bottle = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd")
    cfg_mustard_bottle.func("/World/Objects/mustard_bottle", cfg_mustard_bottle, translation=(-0.2, 0.2, 1.05))

    cube_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        # collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cube_cfg.func("/World/Objects/cube", cube_cfg, translation=(0.0, 0.0, 1.55))

    cfg_klt = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd")
    cfg_klt.func("/World/Objects/klt", cfg_klt, translation=(0, 0, 0.56))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

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
