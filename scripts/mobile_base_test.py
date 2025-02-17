"""Launch Isaac Sim Simulator first."""

import argparse
import math
import numpy as np
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test mobile base control.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


# Define minimal scene config
@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    num_envs = 1  # Just one robot
    env_spacing = 2.0  # Space between environments

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # Mobile robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ridgeback_franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/apptronik/workspaces/isaac_ros-dev/src/cratos/usd/customized_panda.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Base joints
                "dummy_base_prismatic_y_joint": 0.0,
                "dummy_base_prismatic_x_joint": 0.0,
                "dummy_base_revolute_z_joint": 0.0,
                # Arm joints at home position
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.741,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            #"base": ImplicitActuatorCfg(
            #    joint_names_expr=["dummy_base_.*"],
            #    velocity_limit=100.0,
            #    effort_limit=1000.0,
            #    stiffness=1e7,
            #    damping=1e5,
            #),
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )


# Create a minimal environment wrapper
class MinimalEnv:
    def __init__(self, scene):
        self.scene = scene
        self.num_envs = 1
        self.device = "cuda:0"


def main():
    # Initialize simulation context first
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / 60.0, device=args.device if args.device else "cuda:0"
        )
    )

    # Create scene after simulation context is initialized
    scene = InteractiveScene(SimpleSceneCfg())
    env = MinimalEnv(scene)
    sim.reset()

    # World origin
    origin_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/origin",
        markers={
            "origin": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    origin_marker = VisualizationMarkers(origin_marker_cfg)
    origin_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0"),
        marker_indices=torch.tensor([0], device="cuda:0"),
    )

    # Print joint name to index mapping
    robot = scene._articulations["robot"]
    print("Joint Name to Index Mapping:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {name}: {i}")

    base_joint_indices = [8, 9, 6]
    arm_joint_indices = [2, 0, 1, 3, 4, 5, 7]

    NUM_JOINTS = 10
    ALL_ENV_INDICES = torch.arange(scene.num_envs, dtype=torch.long, device="cuda:0")

    root_physx_view = scene._articulations["robot"].root_physx_view
    joint_pos_target = torch.zeros((1, NUM_JOINTS), device="cuda:0")

    # Simulate
    step_count = 0
    while sim.app.is_running():
        # Set the base position via root transform
        target_base_x = 0.5 * math.cos(step_count / 60.0)
        target_base_y = 0.2 * math.sin(step_count / 60.0)
        base_pose = root_physx_view.get_root_transforms()
        base_pose[:, 0] = target_base_x
        base_pose[:, 1] = target_base_y
        indices = torch.arange(base_pose.shape[0], dtype=torch.int32, device="cuda")
        root_physx_view.set_root_transforms(base_pose, indices=indices)

        # Sinusoidal actuation of one arm joint
        joint_pos_target[0, arm_joint_indices] = torch.tensor(
            [[0.0, 0.0, math.cos(step_count / 60.0), 0.0, 0.0, 2.0, 0.74]], device="cuda:0"
        )
        root_physx_view.set_dof_position_targets(joint_pos_target, torch.arange(len(arm_joint_indices), dtype=torch.int32, device="cuda"))

        sim.step(render=True)

        joint_pos = root_physx_view.get_dof_positions()
        joint_vel = root_physx_view.get_dof_velocities()
        print(
            f"Base XYZ pos: {joint_pos[:, base_joint_indices]} \t vel: {joint_vel[:, base_joint_indices]}"
        )

        time.sleep(1 / 120.0)
        step_count += 1


if __name__ == "__main__":
    main()
