# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating the TacSL tactile sensor implementation in IsaacLab.

This script shows how to use the TactileSensor for both camera-based and force field
tactile sensing with the GelSight finger setup.

.. code-block:: bash

    # Usage
    python scripts/demos/sensors/tacsl_sensor.py \
        --use_tactile_rgb \
        --use_tactile_ff \
        --tactile_compliance_stiffness 100.0 \
        --num_envs 16 \
        --contact_object_type nut \
        --save_viz \
        --enable_cameras

"""

import argparse
import math
import os

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="TacSL tactile sensor example.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--normal_contact_stiffness", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tangential_stiffness", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--friction_coefficient", type=float, default=2.0, help="Tactile friction coefficient.")
parser.add_argument(
    "--tactile_compliance_stiffness",
    type=float,
    default=None,
    help="Optional: Override compliant contact stiffness (default: use USD asset values)",
)
parser.add_argument(
    "--tactile_compliant_damping",
    type=float,
    default=None,
    help="Optional: Override compliant contact damping (default: use USD asset values)",
)
parser.add_argument("--save_viz", action="store_true", help="Visualize tactile data.")
parser.add_argument("--save_viz_dir", type=str, default="tactile_record", help="Directory to save tactile data.")
parser.add_argument("--use_tactile_rgb", action="store_true", help="Use tactile RGB sensor data collection.")
parser.add_argument("--use_tactile_ff", action="store_true", help="Use tactile force field sensor data collection.")
parser.add_argument("--debug_sdf_closest_pts", action="store_true", help="Visualize closest SDF points.")
parser.add_argument("--debug_tactile_sensor_pts", action="store_true", help="Visualize tactile sensor points.")
parser.add_argument("--trimesh_vis_tactile_points", action="store_true", help="Visualize tactile points using trimesh.")
parser.add_argument(
    "--contact_object_type",
    type=str,
    default="nut",
    choices=["none", "cube", "nut"],
    help="Type of contact object to use.",
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.timer import Timer

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from isaaclab_assets.sensors import GELSIGHT_R15_CFG


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
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            compliant_contact_stiffness=args_cli.tactile_compliance_stiffness,
            compliant_contact_damping=args_cli.tactile_compliant_damping,
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
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
    tactile_sensor = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        # Sensor configuration
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=args_cli.use_tactile_rgb,
        enable_force_field=args_cli.use_tactile_ff,
        # Elastomer configuration
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        # Contact object configuration
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        # Force field physics parameters
        normal_contact_stiffness=args_cli.normal_contact_stiffness,
        friction_coefficient=args_cli.friction_coefficient,
        tangential_stiffness=args_cli.tangential_stiffness,
        # Camera configuration
        # Note: the camera is already spawned in the scene, properties are set in the
        # 'gelsight_r15_finger.usd' USD file
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        # Debug Visualization
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
    )


@configclass
class CubeTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with cube contact object."""

    # Cube contact object
    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
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
    """Scene with nut contact object."""

    # Nut contact object
    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def mkdir_helper(dir_path: str) -> tuple[str, str]:
    """Create directories for saving tactile sensor visualizations.

    Args:
        dir_path: The base directory path where visualizations will be saved.

    Returns:
        A tuple containing paths to the force field directory and RGB image directory.
    """
    tactile_img_folder = dir_path
    os.makedirs(tactile_img_folder, exist_ok=True)
    # create a subdirectory for the force field data
    tactile_force_field_dir = os.path.join(tactile_img_folder, "tactile_force_field")
    os.makedirs(tactile_force_field_dir, exist_ok=True)
    # create a subdirectory for the RGB image data
    tactile_rgb_image_dir = os.path.join(tactile_img_folder, "tactile_rgb_image")
    os.makedirs(tactile_rgb_image_dir, exist_ok=True)

    return tactile_force_field_dir, tactile_rgb_image_dir


def save_viz_helper(
    dir_path_list: tuple[str, str],
    count: int,
    tactile_data: VisuoTactileSensorData,
    num_envs: int,
    nrows: int,
    ncols: int,
):
    """Save visualization of tactile sensor data.

    Args:
        dir_path_list: A tuple containing paths to the force field directory and RGB image directory.
        count: The current simulation step count, used for naming saved files.
        tactile_data: The data object containing tactile sensor readings (forces, images).
        num_envs: Number of environments in the simulation.
        nrows: Number of rows in the tactile array.
        ncols: Number of columns in the tactile array.
    """
    # Only save the first 2 environments

    tactile_force_field_dir, tactile_rgb_image_dir = dir_path_list

    if tactile_data.tactile_shear_force is not None and tactile_data.tactile_normal_force is not None:
        # visualize tactile forces
        tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
        tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

        tactile_image = compute_tactile_shear_image(
            tactile_normal_force[0, :, :].detach().cpu().numpy(), tactile_shear_force[0, :, :].detach().cpu().numpy()
        )

        if tactile_normal_force.shape[0] > 1:
            tactile_image_1 = compute_tactile_shear_image(
                tactile_normal_force[1, :, :].detach().cpu().numpy(),
                tactile_shear_force[1, :, :].detach().cpu().numpy(),
            )
            combined_image = np.vstack([tactile_image, tactile_image_1])
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (combined_image * 255).astype(np.uint8)
            )
        else:
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (tactile_image * 255).astype(np.uint8)
            )

    if tactile_data.tactile_rgb_image is not None:
        tactile_rgb_data = tactile_data.tactile_rgb_image.cpu().numpy()
        tactile_rgb_data = np.transpose(tactile_rgb_data, axes=(0, 2, 1, 3))
        tactile_rgb_data_first_2 = tactile_rgb_data[:2] if len(tactile_rgb_data) >= 2 else tactile_rgb_data
        tactile_rgb_tiled = np.concatenate(tactile_rgb_data_first_2, axis=0)
        # Convert to uint8 if not already
        if tactile_rgb_tiled.dtype != np.uint8:
            tactile_rgb_tiled = (
                (tactile_rgb_tiled * 255).astype(np.uint8)
                if tactile_rgb_tiled.max() <= 1.0
                else tactile_rgb_tiled.astype(np.uint8)
            )
        cv2.imwrite(os.path.join(tactile_rgb_image_dir, f"{count:04d}.png"), tactile_rgb_tiled)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Assign different masses to contact objects in different environments
    num_envs = scene.num_envs

    if args_cli.save_viz:
        # Create output directories for tactile data
        print(f"[INFO]: Saving tactile data to: {args_cli.save_viz_dir}...")
        dir_path_list = mkdir_helper(args_cli.save_viz_dir)

    # Create constant downward force
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    force_tensor[:, 0, 2] = -1.0

    nrows = scene["tactile_sensor"].cfg.tactile_array_size[0]
    ncols = scene["tactile_sensor"].cfg.tactile_array_size[1]

    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    entity_list = ["robot"]
    if "contact_object" in scene.keys():
        entity_list.append("contact_object")

    while simulation_app.is_running():
        if count == 122:
            # Reset robot and contact object positions
            count = 0
            for entity in entity_list:
                root_state = scene[entity].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene[entity].write_root_state_to_sim(root_state)

            scene.reset()
            print("[INFO]: Resetting robot and contact object state...")

        if "contact_object" in scene.keys():
            # rotation
            if count > 20:
                env_indices = torch.arange(scene.num_envs, device=sim.device)
                odd_mask = env_indices % 2 == 1
                even_mask = env_indices % 2 == 0
                torque_tensor[odd_mask, 0, 2] = 10  # rotation for odd environments
                torque_tensor[even_mask, 0, 2] = -10  # rotation for even environments
                scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(force_tensor, torque_tensor)

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
    # Note: We set the gpu_collision_stack_size to prevent buffer overflow in contact-rich environments.
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])

    # Create scene based on contact object type
    if args_cli.contact_object_type == "cube":
        scene_cfg = CubeTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # disabled force field for cube contact object because a SDF collision mesh cannot
        # be created for the Shape Prims
        scene_cfg.tactile_sensor.enable_force_field = False
    elif args_cli.contact_object_type == "nut":
        scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
    elif args_cli.contact_object_type == "none":
        scene_cfg = TactileSensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        scene_cfg.tactile_sensor.contact_object_prim_path_expr = None
        # this flag is to visualize the tactile sensor points
        scene_cfg.tactile_sensor.debug_vis = True
    else:
        raise ValueError(
            f"Invalid contact object type: '{args_cli.contact_object_type}'. Must be 'none', 'cube', or 'nut'."
        )

    scene = InteractiveScene(scene_cfg)

    # Initialize simulation
    sim.reset()
    print("[INFO]: Setup complete...")

    # Get initial render
    scene["tactile_sensor"].get_initial_render()
    # Run simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
