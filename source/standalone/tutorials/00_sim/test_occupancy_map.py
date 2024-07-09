import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--size", type=float, default=0.5, help="Side-length of cuboid")
# parser.headless = True
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app headless=HEADLESS
# app_launcher = AppLauncher(headless=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni
# import sys
# sys.path.insert(0,'/home/frankie/git/isaac/IsaacLab/_isaac_sim/exts/omni.isaac.occupancy_map')
# print(sys.path)
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.occupancy_map")
from omni.isaac.occupancy_map.bindings import _occupancy_map

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

    # cfg_cuboid = sim_utils.CuboidCfg(
    #     size=[args_cli.size] * 3,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    # )
    # # Spawn cuboid, altering translation on the z-axis to scale to its size
    # cfg_cuboid.func("/World/Objects/cuboid", cfg_cuboid, translation=(0, 0, 1))

    cfg_cracker_box = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", 
                                           rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),)
    cfg_cracker_box.func("/World/Objects/cracker_box", cfg_cracker_box, translation=(0, 0, 1))


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

    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    generator = _occupancy_map.Generator(physx, stage_id)
    # 0.05m cell size, output buffer will have 4 for occupied cells, 5 for unoccupied, and 6 for cells that cannot be seen
    # this assumes your usd stage units are in m, and not cm
    generator.update_settings(.05, 4, 5, 6)
    # Set location to map from and the min and max bounds to map to
    generator.set_transform((0., 0., 1), (-0.5, -0.5, -0.05), (0.5, 0.5, 0.05))
    generator.generate2d()
    # Get locations of the occupied cells in the stage
    occupied_points = generator.get_occupied_positions()
    # Get locations of the unoccupied cells in the stage
    free_points = generator.get_free_positions()
    # Get computed 2d occupancy buffer
    buffer = generator.get_buffer()
    # Get dimensions for 2d buffer
    dims = generator.get_dimensions()

    print('occupied', occupied_points)
    print('free', free_points)
    # print('buffer', buffer)
    print('dims', dims)

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()