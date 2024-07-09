# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.
.. code-block:: bash
    # Usage
    ./isaaclab.sh -p source/standalone/demos/multi_object.py --num_envs 512
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on spawning different objects in multiple environments.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument("--randomize", action="store_true", help="Randomize the objects scale.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import carb

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sim.spawners.multi_asset.asset_randomizer_cfg import AssetRandomizerCfg
from omni.isaac.lab.sim.spawners.multi_asset.multi_asset_cfg import MultiAssetCfg
from omni.isaac.lab.sim.spawners.multi_asset.randomizations import RandomizeScaleCfg
from omni.isaac.lab.utils import configclass

check_size = True


def get_assets():
    kwargs = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
    }

    return [
        sim_utils.SphereCfg(
            radius=0.25,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            **kwargs,
        ),
        sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            **kwargs,
        ),
        sim_utils.CylinderCfg(
            radius=0.2,
            height=0.3,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            **kwargs,
        ),
    ]


def get_randomized_assets():
    assets = get_assets()

    return [
        AssetRandomizerCfg(
            child_spawner_cfg=assets[0],
            randomization_cfg=RandomizeScaleCfg(
                x_range=(0.5, 1.25),
                equal_scale=True,
            ),
            num_random_assets=args_cli.num_envs // 3,
        ),
        AssetRandomizerCfg(
            child_spawner_cfg=assets[1],
            randomization_cfg=RandomizeScaleCfg(
                x_range=(0.5, 1.25),
                equal_scale=True,
            ),
            num_random_assets=args_cli.num_envs // 3,
        ),
        AssetRandomizerCfg(
            child_spawner_cfg=assets[2],
            randomization_cfg=RandomizeScaleCfg(
                x_range=(0.5, 1.25),
                equal_scale=True,
            ),
            num_random_assets=args_cli.num_envs // 3,
        ),
    ]


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Objects",
        spawn=MultiAssetCfg(assets_cfg=get_randomized_assets() if args_cli.randomize else get_assets()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    rigid_object = scene["object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            rigid_object.write_root_state_to_sim(root_state)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim_cfg.use_fabric = False
    sim_cfg.physx.use_gpu = False
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


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
