import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the different camera sensor implementations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
import cv2
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass

from isaaclab.terrains import TerrainImporterCfg
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_test.utils.test_terrain_config import PARKOUR_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from parkour_tasks.extreme_parkour_task.config.go2 import agents 
from isaaclab.utils.math import quat_from_euler_xyz
import torchvision

CLIP_RANGE = (0.3, 3.)
RESIZE = (87, 58)
resize_transform = torchvision.transforms.Resize(
                                    (RESIZE[1], RESIZE[0]), 
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to('cuda')
    
def _process_depth_image( depth_image):
    depth_image = torch.from_numpy(depth_image[:,:,0]).to('cuda')
    # These operations are replicated on the hardware
    depth_image = _crop_depth_image(depth_image)
    depth_image = resize_transform(depth_image[None, :]).squeeze()
    depth_image = _normalize_depth_image(depth_image)
    return depth_image.detach().cpu().numpy()

def _crop_depth_image(depth_image):
    return depth_image[:-2, 4:-4]

def _normalize_depth_image( depth_image):
    depth_image = depth_image 
    depth_image = (depth_image - CLIP_RANGE[0]) / (CLIP_RANGE[1] - CLIP_RANGE[0])  - 0.5
    return depth_image

##
# Pre-defined configs
##
CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    height=60,
    width=106,
    history_length = 2,
    update_period = 0.005*5,
    data_types=["distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, 
        focus_distance=400.0, 
        horizontal_aperture=20.955,
        clipping_range=CLIP_RANGE
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08), 
        rot=quat_from_euler_xyz(*tuple(torch.deg2rad(torch.tensor([180,30,-90])))) * torch.tensor([1.,1.,1.,-1]), 
        convention="ros"
        ),
    colorize_semantic_segmentation = False , 
    colorize_instance_id_segmentation = False , 
    colorize_instance_segmentation = False , 
    depth_clipping_behavior = 'max'
)

CAMERA_USD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0],'d435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.33, 0.0, 0.08), 
            rot=quat_from_euler_xyz(*tuple(torch.deg2rad(torch.tensor([180,30,-90])))) * torch.tensor([1.,1.,1.,-1])
            )
)

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    terrain = TerrainImporterCfg(
        class_type= ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CAMERA_CFG

        # Add USD camera as a static asset
    camera_usd = CAMERA_USD_CFG
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            cfg = scene.terrain.cfg.terrain_generator
            origin = scene.env_origins
            origin[:,-1] = 0
            positions = root_state[:, 0:3] + origin - torch.tensor((cfg.size[1]/2, 0, 0)).to('cuda')
            scene["robot"].write_root_pose_to_sim(torch.cat([positions, root_state[:, 3:7]], dim=-1))
            camera: Camera = scene.sensors['camera']
            # random_pitch = (70 - 60) * torch.rand((scene.num_envs,1)) + 50
            # roll_tensor = torch.ones_like(random_pitch) * 180
            # yaw_tensor = torch.ones_like(random_pitch) * -90
            # data = quat_from_euler_xyz(roll_tensor, random_pitch, yaw_tensor) * torch.tensor([1.,1.,1.,-1]) 
            # camera.set_world_poses(orientations = data.squeeze(1), convention = 'ros')
            camera.reset()
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        
        ## visualize depth camera 
        depth_camera = scene["camera"].data.output["distance_to_image_plane"].detach().cpu().numpy()
        env_num, image_width, image_hight, _ = depth_camera.shape
        depth_camera = depth_camera.reshape(-1 ,image_hight,1)
        cv2.imshow('depth_camera',depth_camera)
        cv2.waitKey(1)
        process_image = _process_depth_image(depth_camera)
        cv2.imshow('process_image',process_image)
        cv2.waitKey(1)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(-0., 2., 1.), target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=5, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from parkour_tasks.extreme_parkour_task.config.go2 import agents 
    simulation_app.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # simulation_app.close()
