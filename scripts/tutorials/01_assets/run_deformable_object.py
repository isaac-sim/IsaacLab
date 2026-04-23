# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
parser.add_argument("--backend", type=str, default="physx", choices=["physx", "newton"], help="Physics backend.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Create separate groups called "Origin0", "Origin1", ...
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=origin)

    # 3D Deformable Object
    cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5, density=500.0),
            # physics_material=sim_utils.SurfaceDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e4, surface_thickness=0.001, surface_bend_stiffness=1e0, surface_shear_stiffness=1e0, surface_stretch_stiffness=1e0),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )

    cube_object = DeformableObject(cfg=cfg)
    scene_entities["cube_object"] = cube_object

    # return the scene information
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor, output_dir: str):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube_object: DeformableObject = entities["cube_object"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = wp.to_torch(cube_object.data.nodal_kinematic_target).clone()

    # Simulate physics
    # for _ in range(200):
    while simulation_app.is_running():
        # reset at start and after 3 seconds
        if count % int(3.0 / sim_dt) == 0:
            # reset counters
            count = 0

            # reset buffers
            cube_object.reset()

            # reset the nodal state of the object
            nodal_state = wp.to_torch(cube_object.data.default_nodal_state_w).clone()
            # apply random pose to the object
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write nodal state to simulation
            cube_object.write_nodal_state_to_sim_index(nodal_state)

            # Write the nodal state to the kinematic target and free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target)

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # update the kinematic target for cubes at index 0 and 3
        kinematic_cubes = [0, 3]
        # we slightly move the cube in the z-direction by picking the vertex at index 0
        nodal_kinematic_target[kinematic_cubes, 0, 2] += 0.2 * sim_dt
        # set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        nodal_kinematic_target[kinematic_cubes, 0, 3] = 0.0
        # write kinematic target to simulation
        cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target)

        # write internal data to simulation
        cube_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube_object.update(sim_dt)

        # print the root positions every second
        if count % int(1.0 / sim_dt) == 0:
            print(
                f"Time {sim_time:.2f}s: \tRoot position (in world): {wp.to_torch(cube_object.data.root_pos_w)[:, :3]}"
            )


def main():
    """Main function."""
    # Load kit helper
    if args_cli.backend == "newton":
        from isaaclab_experimental.deformable import register_hooks as _register_deformable_hooks
        _register_deformable_hooks()
        from isaaclab_experimental.deformable.newton_manager_cfg import VBDSolverCfg
        from isaaclab_newton.physics import NewtonCfg
        physics_cfg = NewtonCfg(solver_cfg=VBDSolverCfg(iterations=10), num_substeps=4)
    else:
        from isaaclab_physx.physics import PhysxCfg
        physics_cfg = PhysxCfg()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.75])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    camera_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    run_simulator(sim, scene_entities, scene_origins, camera_output)
    print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
