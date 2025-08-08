# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows the issue with renderer in Isaac Sim that affects episodic resets.

The first few images of every new episode are not updated. They take multiple steps to update
and have the same image as the previous episode for the first few steps.

```
# run with cube
_isaac_sim/python.sh source/isaaclab/test/deps/isaacsim/check_camera.py --scenario cube
# run with anymal
_isaac_sim/python.sh source/isaaclab/test/deps/isaacsim/check_camera.py --scenario anymal
```
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# isaaclab
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows the issue with renderer in Isaac Sim that affects episodic resets."
)
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU device for camera rendering output.")
parser.add_argument("--scenario", type=str, default="anymal", help="Scenario to load.", choices=["anymal", "cube"])
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random

try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation, RigidPrim, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.carb import set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view
from PIL import Image, ImageChops
from pxr import Gf, UsdGeom

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    raise RuntimeError(msg)

ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""


def main():
    """Runs a camera sensor from isaaclab."""

    # Load kit helper
    world = World(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cpu")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable flatcache which avoids passing data over to USD structure
    # this speeds up the read-write operation of GPU buffers
    if world.get_physics_context().use_gpu_pipeline:
        world.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(world._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Populate scene
    # Ground
    world.scene.add_default_ground_plane()
    # Lights-1
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
    # Lights-2
    prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))
    # Xform to hold objects
    if args_cli.scenario == "cube":
        prim_utils.create_prim("/World/Objects", "Xform")
        # Random objects
        for i in range(8):
            # sample random position
            position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
            position *= np.asarray([1.5, 1.5, 0.5])
            # create prim
            prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
            _ = prim_utils.create_prim(
                f"/World/Objects/Obj_{i:02d}",
                prim_type,
                translation=position,
                scale=(0.25, 0.25, 0.25),
                semantic_label=prim_type,
            )
            # add rigid properties
            SingleGeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
            rigid_obj = SingleRigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
            # cast to geom prim
            geom_prim = getattr(UsdGeom, prim_type)(rigid_obj.prim)
            # set random color
            color = Gf.Vec3f(random.random(), random.random(), random.random())
            geom_prim.CreateDisplayColorAttr()
            geom_prim.GetDisplayColorAttr().Set([color])
        # Setup camera sensor on the world
        cam_prim_path = "/World/CameraSensor"
    else:
        # Robot
        prim_utils.create_prim(
            "/World/Robot",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd",
            translation=(0.0, 0.0, 0.6),
        )
        # Setup camera sensor on the robot
        cam_prim_path = "/World/CameraSensor"

    # Create camera
    cam_prim = prim_utils.create_prim(
        cam_prim_path,
        prim_type="Camera",
        translation=(5.0, 5.0, 5.0),
        orientation=(0.33985113, 0.17591988, 0.42470818, 0.82047324),
    )
    _ = UsdGeom.Camera(cam_prim)
    # Get render product
    render_prod_path = rep.create.render_product(cam_prim_path, resolution=(640, 480))
    # create annotator node
    rep_registry = {}
    for name in ["rgb", "distance_to_image_plane"]:
        # create annotator
        rep_annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
        rep_annotator.attach(render_prod_path)
        # add to registry
        rep_registry[name] = rep_annotator

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera", args_cli.scenario)
    os.makedirs(output_dir, exist_ok=True)

    # Create a view of the stuff we want to see
    if args_cli.scenario == "cube":
        view: RigidPrim = world.scene.add(RigidPrim("/World/Objects/.*", name="my_object"))
    else:
        view: Articulation = world.scene.add(Articulation("/World/Robot", name="my_object"))
    # Play simulator
    world.reset()
    # Get initial state
    if args_cli.scenario == "cube":
        initial_pos, initial_quat = view.get_world_poses()
        initial_joint_pos = None
        initial_joint_vel = None
    else:
        initial_pos, initial_quat = view.get_world_poses()
        initial_joint_pos = view.get_joint_positions()
        initial_joint_vel = view.get_joint_velocities()

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        world.step(render=True)

    # Counter
    count = 0
    prev_im = None
    # make episode directory
    episode_count = 0
    episode_dir = os.path.join(output_dir, f"episode_{episode_count:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if world.is_stopped():
            break
        # If simulation is paused, then skip.
        if not world.is_playing():
            world.step(render=False)
            continue
        # Reset on intervals
        if count % 25 == 0:
            # reset all the state
            view.set_world_poses(initial_pos, initial_quat)
            if initial_joint_pos is not None:
                view.set_joint_positions(initial_joint_pos)
            if initial_joint_vel is not None:
                view.set_joint_velocities(initial_joint_vel)
            # make a new episode directory
            episode_dir = os.path.join(output_dir, f"episode_{episode_count:06d}")
            os.makedirs(episode_dir, exist_ok=True)
            # reset counters
            count = 0
            episode_count += 1
        # Step simulation
        for _ in range(15):
            world.step(render=False)
        world.render()
        # Update camera data
        rgb_data = rep_registry["rgb"].get_data()
        depth_data = rep_registry["distance_to_image_plane"].get_data()

        # Show current image number
        print(f"[Epi {episode_count:03d}] Current image number: {count:06d}")
        # Save data
        curr_im = Image.fromarray(rgb_data)
        curr_im.save(os.path.join(episode_dir, f"{count:06d}_rgb.png"))
        # Save diff
        if prev_im is not None:
            diff_im = ImageChops.difference(curr_im, prev_im)
            # convert to grayscale and threshold
            diff_im = diff_im.convert("L")
            threshold = 30
            diff_im = diff_im.point(lambda p: p > threshold and 255)
            # Save all of them together
            dst_im = Image.new("RGB", (curr_im.width + prev_im.width + diff_im.width, diff_im.height))
            dst_im.paste(prev_im, (0, 0))
            dst_im.paste(curr_im, (prev_im.width, 0))
            dst_im.paste(diff_im, (2 * prev_im.width, 0))
            dst_im.save(os.path.join(episode_dir, f"{count:06d}_diff.png"))

        # Save to previous
        prev_im = curr_im.copy()
        # Update counter
        count += 1

        # Print camera info
        print("Received shape of rgb   image: ", rgb_data.shape)
        print("Received shape of depth image: ", depth_data.shape)
        print("-------------------------------")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
