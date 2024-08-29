import argparse

from omni.isaac.lab.app import AppLauncher

import torch

# create argparser
parser = argparse.ArgumentParser(
    description="Tutorial on spawning prims into the scene."
)

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
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.assets import ArticulationCfg


def design_scene():
    # Spawning prims
    # First create a configuration for the primitive
    cfg_ground_plane = sim_utils.GroundPlaneCfg()
    # Call the spawner function
    cfg_ground_plane.func("/World/groundPlane", cfg_ground_plane)

    # Spawning some lights
    cfg_light = sim_utils.DistantLightCfg(color=(1.0, 0.75, 0.75), intensity=3000.0)
    cfg_light.func("/World/light", cfg_light, translation=(1, 0, 10))

    # Spawn a deformable
    #    cfg_deform = sim_utils.MeshCuboidCfg(
    #        size=(0.2, 0.5, 0.2),
    #        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
    #        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    #    )
    #    cfg_deform.func(
    #        "/World/deformable",
    #        cfg_deform,
    #        translation=(-0.2, 0.0, 2.0),
    #        orientation=(0.5, 0.0, 0.5, 0.0),
    #    )

    # Create groups with different origins one for each robot
    origins = [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    # Create a prim for each robot
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])
    prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])
    prim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])
    prim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])

    # Spawn a USD file
    haw_ur5_spawn = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/haw_ur5_arm_rg6/haw_ur5_gr6.usd",
    )
    prim_path = "/World/Origin.*/Robot"

    robots = haw_ur5_spawn.func(prim_path, haw_ur5_spawn)

    scene_entities = {"robots": robots}

    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, sim_utils.UsdFileCfg],
    origins: torch.Tensor,
):
    robot = entities["robots"]
    sim_dt = sim.get_physics_dt()
    # Loop over the simulation steps
    count = 0
    articulation_view = robot[0].get_articulation_view()

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            # Reset scene entities
            # root_state = robot.articulation_props.
        # Add your custom code here


def main():
    # Define the simulation context (Here all operations of the simulation can be handled like starting, stopping, stepping etc.)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)  # Physics timestep of 0.01 seconds
    sim = SimulationContext(sim_cfg)
    # Setting the main camera
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])

    # Design the scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()  # Reset / Start the simulation

    # Ready to run the simulator
    print("[INFO]: Setup complete. Running the simulator.")
    run_simulator(sim, scene_entities, scene_origins)


# python main function
if __name__ == "__main__":
    main()  # Call the main function
    simulation_app.close()  # Close the simulation
