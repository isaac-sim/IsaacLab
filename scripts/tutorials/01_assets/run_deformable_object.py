# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run --extra isaacsim --extra tetrahedralization python scripts/tutorials/01_assets/run_deformable_object.py

    # Usage with Newton VBD physics and default kit visualizer.
    uv run --extra isaacsim --extra tetrahedralization python scripts/tutorials/01_assets/run_deformable_object.py \
        --backend newton_vbd

    # Usage with OvPhysX physics without a visualizer.
    uv run --extra ovphysx --extra tetrahedralization python scripts/tutorials/01_assets/run_deformable_object.py \
        --backend ovphysx

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
parser.add_argument(
    "--backend", type=str, default="physx", choices=["physx", "newton_vbd", "ovphysx"], help="Physics backend."
)
# append simulation launcher CLI arguments
add_launcher_args(parser)
# Kit cannot be combined with OvPhysX, so use no visualizer by default for that backend
backend_args, _ = parser.parse_known_args()
parser.set_defaults(visualizer=None if backend_args.backend == "ovphysx" else ["kit"])
# parse the arguments
args_cli = parser.parse_args()
args_cli.physics = args_cli.backend

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.cloner import CloneCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.scene import InteractiveScene


youngs_modulus = 1e5
poissons_ratio = 0.4
density = 500.0
if args_cli.backend == "newton_vbd":
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

    deformable_props = NewtonDeformableBodyPropertiesCfg()
    # Newton's VBD path skips the simulation mesh collider, so collision offsets do not apply
    collision_props = None
    physics_material = NewtonDeformableBodyMaterialCfg(
        k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
        k_lambda=youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)),
        density=density,
    )
else:
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    deformable_props = PhysxDeformableBodyPropertiesCfg()
    collision_props = [PhysxCollisionCfg(rest_offset=0.0, contact_offset=0.001)]
    physics_material = PhysxDeformableBodyMaterialCfg(
        poissons_ratio=poissons_ratio, youngs_modulus=youngs_modulus, density=density
    )


@configclass
class DeformableSceneCfg(InteractiveSceneCfg):
    """Four soft cubes on a shared ground plane."""

    num_envs = 4
    env_spacing = 0.5
    filter_collisions = False
    clone_cfg = CloneCfg(clone_template="/World/env_{}")

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    # 3D Deformable Object
    cube_object = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=deformable_props,
            collision_props=collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=physics_material,
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: "InteractiveScene"):
    """Runs the simulation loop."""
    # Extract scene entities
    cube_object: DeformableObject = scene["cube_object"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.torch.clone()

    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        # reset at start and after 3 seconds
        if count % int(3.0 / sim_dt) == 0:
            # reset counters
            count = 0

            # reset the nodal state of the object
            nodal_state = cube_object.data.default_nodal_state_w.torch.clone()
            # apply random pose to the object
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + scene.env_origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write nodal state to simulation
            cube_object.write_nodal_state_to_sim_index(nodal_state)

            # Write the nodal state to the kinematic target and free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target)

            # reset buffers
            cube_object.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # update the kinematic target for cubes at the positive and negative diagonal corners
        kinematic_cubes = [1, 2]
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
            print(f"Time {sim_time:.2f}s: \tRoot position (in world): {cube_object.data.root_pos_w.torch[:, :3]}")


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if args_cli.backend == "newton_vbd":
            physics_cfg.solver_cfg.iterations = 10
            physics_cfg.num_substeps = 4
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.75])
        scene_cfg = DeformableSceneCfg()
        scene = scene_cfg.class_type(scene_cfg)
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene)
        print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    # run the main function
    main()
