# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use replicator to randomly change the textures of a USD scene.

Note:
    Currently this script fails since cloner does not support changing textures of cloned
    USD prims. This is because the prims are cloned using `Sdf.ChangeBlock` which does not
    allow individual texture changes.

Usage:

.. code-block:: bash

    ./isaaclab.sh -p source/isaaclab/test/deps/isaacsim/check_rep_texture_randomizer.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

# isaaclab
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows how to use replicator to randomly change the textures of a USD scene."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch

import omni.replicator.core as rep
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim.utils.prims as prim_utils


def main():
    """Spawn a bunch of balls and randomly change their textures."""

    # Load kit helper
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,  # deprecated from Isaac Sim 2023.1 onwards
        "use_fabric": True,  # used from Isaac Sim 2023.1 onwards
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )
    # Set main camera
    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Parameters
    num_balls = 128

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0, stage=sim.stage)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Define the scene
    # -- Ball
    DynamicSphere(prim_path="/World/envs/env_0/ball", translation=np.array([0.0, 0.0, 5.0]), mass=0.5, radius=0.25)

    # Clone the scene
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    env_positions = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True, copy_from_source=True
    )
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )

    # Use replicator to randomize color on the spheres
    with rep.new_layer():
        # Define a function to get all the shapes
        def get_shapes():
            shapes = rep.get.prims(path_pattern="/World/envs/env_.*/ball")
            with shapes:
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
            return shapes.node

        # Register the function
        rep.randomizer.register(get_shapes)
        # Specify the frequency of randomization
        with rep.trigger.on_frame():
            rep.randomizer.get_shapes()

    # Set ball positions over terrain origins
    # Create a view over all the balls
    ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)
    # cache initial state of the balls
    ball_initial_positions = torch.tensor(env_positions, dtype=torch.float, device=sim.device)
    ball_initial_positions[:, 2] += 5.0
    # set initial poses
    # note: setting here writes to USD :)
    ball_view.set_world_poses(positions=ball_initial_positions)

    # Play simulator
    sim.reset()
    # Step replicator to randomize colors
    rep.orchestrator.step(pause_timeline=False)
    # Stop replicator to prevent further randomization
    rep.orchestrator.stop()
    # Pause simulator at the beginning for inspection
    sim.pause()

    # Initialize the ball views for physics simulation
    ball_view.initialize()
    ball_initial_velocities = ball_view.get_velocities()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset the balls
            ball_view.set_world_poses(positions=ball_initial_positions)
            ball_view.set_velocities(ball_initial_velocities)
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
