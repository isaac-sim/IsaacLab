"""Basic sanity check script to load kbot in Isaacsim.

Usage:
# cd to IsaacLab root directory
./isaaclab.sh -p scripts/tutorials/01_assets/add_new_robot_kbot.py 
"""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from pathlib import Path


KBOT_USD = os.path.join(
    os.path.dirname(__file__),
    "../../../source/isaaclab_assets/isaaclab_assets/robots/temp_kbot_usd",
    "robot.usd",
)

KBOT_CONFIG = ArticulationCfg(
    # Spawn the USD you exported from the URDF
    spawn=sim_utils.UsdFileCfg(
        usd_path=KBOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,  # humanoids need more iters
            solver_velocity_iteration_count=0,
        ),
    ),
    # Start the robot upright ~1 m above the ground
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={},
    ),
    # For now just use the same actuator for all joints
    actuators={
        "all_dofs": ImplicitActuatorCfg(
            joint_names_expr=["dof_.*"],  # matches all 20 "dof_*" joints
            effort_limit_sim=150.0,
            velocity_limit_sim=20.0,
            stiffness=120.0,
            damping=10.0,
        )
    },
)


class KbotSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    #  Kbot
    Kbot = KBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Kbot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_state = scene["Kbot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the kbot's orientation and velocity
            scene["Kbot"].write_root_pose_to_sim(root_state[:, :7])
            scene["Kbot"].write_root_velocity_to_sim(root_state[:, 7:])
            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Kbot"].data.default_joint_pos.clone(),
                scene["Kbot"].data.default_joint_vel.clone(),
            )
            scene["Kbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Kbot reset")

        # tiny idle "wave" to see joints move
        wave = scene["Kbot"].data.default_joint_pos.clone()
        wave[:] += 0.15 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Kbot"].set_joint_position_target(wave)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = KbotSceneCfg(args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
