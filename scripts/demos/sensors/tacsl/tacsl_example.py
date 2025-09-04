# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating the TacSL tactile sensor implementation in IsaacLab.

This script shows how to use the TactileSensor for both camera-based and force field
tactile sensing with the gelsight finger setup.

.. code-block:: bash

    # Usage
    python tacsl_example.py --enable_cameras --num_envs 2 --indenter_type nut --save_viz --use_tactile_taxim --use_tactile_ff

"""

import argparse
import math
import numpy as np
import os
import torch

import cv2

from isaaclab.app import AppLauncher
from isaaclab.utils.timer import Timer

# Add argparse arguments
parser = argparse.ArgumentParser(description="TacSL tactile sensor example.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--tactile_kn", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tactile_kt", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--tactile_mu", type=float, default=2.0, help="Tactile friction coefficient.")
parser.add_argument("--tactile_damping", type=float, default=0.003, help="Tactile damping coefficient.")
parser.add_argument("--tactile_compliance_stiffness", type=float, default=100.0, help="Tactile compliance stiffness.")
parser.add_argument("--tactile_compliant_damping", type=float, default=1.0, help="Tactile compliant damping.")
parser.add_argument("--save_viz", action="store_true", help="Visualize tactile data.")
parser.add_argument("--save_viz_dir", type=str, default="tactile_record", help="Directory to save tactile data.")
parser.add_argument("--use_tactile_taxim", action="store_true", help="Use tactile taxim sensor data collection.")
parser.add_argument("--use_tactile_ff", action="store_true", help="Use tactile force field sensor data collection.")
parser.add_argument("--debug_sdf_closest_pts", action="store_true", help="Visualize closest SDF points.")
parser.add_argument("--debug_tactile_sensor_pts", action="store_true", help="Visualize tactile sensor points.")
parser.add_argument("--trimesh_vis_tactile_points", action="store_true", help="Visualize tactile points using trimesh.")
parser.add_argument(
    "--indenter_type", type=str, default="nut", choices=["none", "cube", "nut"], help="Type of indenter to use."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Import our TactileSensor
from isaaclab.sensors import TactileSensorCfg, TiledCameraCfg
from isaaclab.sensors.tacsl_sensor.tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


@configclass
class TactileSensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with tactile sensors on the robot."""

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot with tactile sensor
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # use local path for now
            usd_path=f"{ASSET_DIR}/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),  # 90° rotation
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Camera configuration for tactile sensing

    # TacSL Tactile Sensor
    tactile_sensor = TactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tactile_sensor",
        history_length=0,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        ## Sensor configuration
        sensor_type="gelsight_r15",
        enable_camera_tactile=args_cli.use_tactile_taxim,
        enable_force_field=args_cli.use_tactile_ff,
        ## Elastomer configuration
        elastomer_link_name="elastomer",
        elastomer_tip_link_name="elastomer_tip",
        # Force field configuration
        num_tactile_rows=20,
        num_tactile_cols=25,
        tactile_margin=0.003,
        ## Indenter configuration (will be set based on indenter type)
        indenter_actor_name=None,  # Will be updated based on indenter type
        indenter_link_name=None,  # Will be updated based on indenter type
        ## Force field physics parameters
        tactile_kn=args_cli.tactile_kn,
        tactile_damping=args_cli.tactile_damping,
        tactile_mu=args_cli.tactile_mu,
        tactile_kt=args_cli.tactile_kt,
        ## Compliant dynamics
        compliance_stiffness=args_cli.tactile_compliance_stiffness,
        compliant_damping=args_cli.tactile_compliant_damping,
        ## Camera configuration
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/tactile_cam",
            update_period=1 / 60,  # 60 Hz
            height=320,
            width=240,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.020342857142857145 * 100,
                focus_distance=400.0 / 1000,
                horizontal_aperture=0.0119885 * 2 * 100,
                clipping_range=(0.0001, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, -0.020342857142857145 + 0.00175), rot=(0.5, 0.5, -0.5, 0.5), convention="world"
            ),
        ),
        ## Debug Visualization
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/TactileSensorDebugPts",
            markers={
                "debug_pts": sim_utils.SphereCfg(
                    radius=0.0002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        ),
    )


@configclass
class CubeTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with cube indenter."""

    # Cube indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 + 0.06776, 0.51), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class NutTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with nut indenter."""

    # Nut indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=90.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 100.0),
        ),
    )


def mkdir_helper(dir_path):
    tactile_img_folder = dir_path
    os.makedirs(tactile_img_folder, exist_ok=True)
    tactile_force_field_dir = os.path.join(tactile_img_folder, "tactile_force_field")
    os.makedirs(tactile_force_field_dir, exist_ok=True)
    tactile_taxim_dir = os.path.join(tactile_img_folder, "tactile_taxim")
    os.makedirs(tactile_taxim_dir, exist_ok=True)
    return tactile_force_field_dir, tactile_taxim_dir


def save_viz_helper(dir_path_list, count, tactile_data, num_envs, nrows, ncols):
    tactile_force_field_dir, tactile_taxim_dir = dir_path_list

    if tactile_data.tactile_shear_force is not None and tactile_data.tactile_normal_force is not None:
        # visualize tactile forces
        tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
        tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

        tactile_image = visualize_tactile_shear_image(
            tactile_normal_force[0, :, :].detach().cpu().numpy(), tactile_shear_force[0, :, :].detach().cpu().numpy()
        )
        cv2.imwrite(os.path.join(tactile_force_field_dir, f"env0_{count}.png"), tactile_image * 255)

        if tactile_normal_force.shape[0] > 1:
            tactile_image_1 = visualize_tactile_shear_image(
                tactile_normal_force[1, :, :].detach().cpu().numpy(),
                tactile_shear_force[1, :, :].detach().cpu().numpy(),
            )
            cv2.imwrite(os.path.join(tactile_force_field_dir, f"env1_{count}.png"), tactile_image_1 * 255)

    if tactile_data.taxim_tactile is not None:
        taxim_data = tactile_data.taxim_tactile.cpu().numpy()
        # Only save the first 2 environments
        taxim_data_first_2 = taxim_data[:2] if len(taxim_data) >= 2 else taxim_data
        taxim_tiled = np.concatenate(taxim_data_first_2, axis=0)
        cv2.imwrite(os.path.join(tactile_taxim_dir, f"{count}.png"), taxim_tiled)


def run_simulator(sim, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Assign different masses to indenters in different environments
    num_envs = scene.num_envs

    if args_cli.save_viz:
        # Create output directories for tactile data
        dir_path_list = mkdir_helper(args_cli.save_viz_dir)

    # Create constant downward force
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    force_tensor[:, 0, 2] = -5.0

    nrows = scene["tactile_sensor"].cfg.num_tactile_rows
    ncols = scene["tactile_sensor"].cfg.num_tactile_cols

    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    scene.update(sim_dt)

    entity_list = ["robot"]
    if "indenter" in scene.keys():
        entity_list.append("indenter")

    while simulation_app.is_running():

        if count == 122:
            print(scene["tactile_sensor"].get_timing_summary())
            # Reset robot and indenter positions
            count = 0
            for entity in entity_list:
                root_state = scene[entity].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene[entity].write_root_state_to_sim(root_state)

            scene.reset()
            print("[INFO]: Resetting robot and indenter state...")

        if "indenter" in scene.keys():
            # rotation
            if count > 20:
                env_indices = torch.arange(scene.num_envs, device=sim.device)
                odd_mask = env_indices % 2 == 1
                even_mask = env_indices % 2 == 0
                # Juana: different direction of rotation for odd and even environments as a visual check of force field computation,
                # however, increasing the value here seems to not work as expected, I tested with 10, the object still rotates very slowly
                torque_tensor[odd_mask, 0, 2] = 2  # rotation for odd environments
                torque_tensor[even_mask, 0, 2] = -2  # rotation for even environments
            scene["indenter"].set_external_force_and_torque(force_tensor, torque_tensor)

        # Step simulation
        scene.write_data_to_sim()
        physics_timer.start()
        sim.step()
        physics_timer.stop()
        physics_total_time += physics_timer.total_run_time
        physics_total_count += 1
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Access tactile sensor data
        tactile_data = scene["tactile_sensor"].data

        if args_cli.save_viz:
            save_viz_helper(dir_path_list, count, tactile_data, num_envs, nrows, ncols)

    # Get timing summary from sensor and add physics timing
    timing_summary = scene["tactile_sensor"].get_timing_summary()

    # Add physics timing to the summary
    physics_avg = physics_total_time / (physics_total_count * scene.num_envs) if physics_total_count > 0 else 0.0
    timing_summary["physics_total"] = physics_total_time
    timing_summary["physics_average"] = physics_avg
    timing_summary["physics_fps"] = 1 / physics_avg if physics_avg > 0 else 0.0

    print(timing_summary)


def main():
    """Main function."""
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2
            ** 30,  # Important to prevent collisionStackSize buffer overflow in contact-rich environments.
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Create scene based on indenter type
    if args_cli.indenter_type == "cube":
        scene_cfg = CubeTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # Juana: disabled force field for cube indenter because a SDF collision mesh cannot be created for the Cube Shape, not sure why
        scene_cfg.tactile_sensor.enable_force_field = False
        # Update tactile sensor configuration for cube
        scene_cfg.tactile_sensor.indenter_actor_name = "indenter"
        scene_cfg.tactile_sensor.indenter_link_name = "geometry"
        scene_cfg.tactile_sensor.indenter_mesh_name = "geometry/mesh"
    elif args_cli.indenter_type == "nut":
        scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # Update tactile sensor configuration for nut
        scene_cfg.tactile_sensor.indenter_actor_name = "indenter"
        scene_cfg.tactile_sensor.indenter_link_name = "factory_nut_loose"
        scene_cfg.tactile_sensor.indenter_mesh_name = "factory_nut_loose/collisions"
    elif args_cli.indenter_type == "none":
        scene_cfg = TactileSensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # this flag is to visualize the tactile sensor points
        scene_cfg.tactile_sensor.debug_vis = True

    scene = InteractiveScene(scene_cfg)

    # Juana: this seems to only works after scene is initialized and before sim.reset() ?
    scene["tactile_sensor"].setup_compliant_materials()

    # Initialize simulation
    sim.reset()
    print("[INFO]: Setup complete...")

    # Juana: this should be manually called before running any simulation ?
    scene["tactile_sensor"].get_initial_render()

    # Run simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
