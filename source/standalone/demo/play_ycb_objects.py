# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to import and use the YCB objects in Orbit
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Load YCB objects in Orbit and randomize their poses.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import quat_mul, random_yaw_orientation, sample_cylinder


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False))
    # Set main camera
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # add YCB objects
    ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }
    for key, usd_path in ycb_usd_paths.items():
        translation = torch.rand(3).tolist()
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func(f"/World/Objects/{key}", cfg, translation=translation)

    # Setup rigid object
    cfg = RigidObjectCfg(prim_path="/World/Objects/.*")
    # Create rigid object handler
    rigid_object = RigidObject(cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = rigid_object.data.default_root_state.clone()
            # -- position
            root_state[:, :3] = sample_cylinder(
                radius=0.5, h_range=(0.15, 0.25), size=rigid_object.num_instances, device=rigid_object.device
            )
            # -- orientation: apply yaw rotation
            root_state[:, 3:7] = quat_mul(
                random_yaw_orientation(rigid_object.num_instances, rigid_object.device), root_state[:, 3:7]
            )
            # -- set root state
            rigid_object.write_root_state_to_sim(root_state)
            # reset buffers
            rigid_object.reset()
            print(">>>>>>>> Reset!")
        # apply sim data
        rigid_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        rigid_object.update(sim_dt)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
