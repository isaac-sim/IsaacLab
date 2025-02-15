# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on using the mimic joints for Robotiq 140 gripper.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                "omniverse://ov-isaac-dev/Projects/IsaacARM/Assets/UR10/iakinola/ur10e_robotiq_140_variant.usd"
            ),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.0,
                "elbow_joint": 1.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            # rot=(0.0, 0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
                friction=None,
                armature=None,
                # effort_limit=330.0,
                # velocity_limit=2.175,
                # stiffness=400.0,
                # damping=40.0,
                # friction=0.0,
                # armature=0.0,  # 0.57
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
                friction=None,
                armature=None,
                # effort_limit=150.0,
                # velocity_limit=2.175,
                # stiffness=400.0,
                # damping=40.0,
                # friction=0.0,
                # armature=0.0,  # 0.57
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
                friction=None,
                armature=None,
                # effort_limit=56.0,
                # velocity_limit=2.175,
                # stiffness=400.0,
                # damping=40.0,
                # friction=0.0,
                # armature=0.0,  # 0.57
            ),
            "finger": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
                friction=None,
                armature=None,
                # effort_limit=10.0 * 10,
                # velocity_limit=130 / 180 * np.pi,
                # stiffness=0.1125,
                # damping=0.001,
                # # stiffness=400.0,
                # # damping=40.0,
                # # stiffness=400.0*2,
                # # damping=56.568542494923804,
                # friction=0.0,
                # armature=0.0,
            ),
            # "left_inner_finger_joint": ImplicitActuatorCfg(
            #     joint_names_expr=["left_inner_finger_joint"],
            #     effort_limit=1.0,
            #     velocity_limit=1e6,
            #     stiffness=0.002,
            #     damping=0.0001,
            #     friction=0.0,
            #     armature=0.0,
            # ),
            # "right_inner_finger_joint": ImplicitActuatorCfg(
            #     joint_names_expr=["right_inner_finger_joint"],
            #     effort_limit=1.0,
            #     velocity_limit=1e6,
            #     stiffness=0.002,
            #     damping=0.0001,
            #     friction=0.0,
            #     armature=0.0,
            # ),
        },
        actuated_joint_names=["shoulder_.*", "elbow_joint", "wrist_.*", "finger_joint"],
        mimic_joints_info={
            "right_outer_knuckle_joint": {
                "parent": "finger_joint",
                "multiplier": -1.0,
                "offset": 0.0,
            },
            "right_inner_finger_joint": {
                "parent": "finger_joint",
                "multiplier": 1.0,
                "offset": 0.0,
            },
            "right_inner_finger_pad_joint": {
                "parent": "finger_joint",
                "multiplier": 1.0,
                "offset": 0.0,
            },
            "left_outer_finger_joint": {
                "parent": "finger_joint",
                "multiplier": 1.0,
                "offset": 0.0,
            },
            "left_inner_finger_joint": {
                "parent": "finger_joint",
                "multiplier": 1.0,
                "offset": 0.0,
            },
            "left_inner_finger_pad_joint": {
                "parent": "finger_joint",
                "multiplier": 1.0,
                "offset": 0.0,
            },
        },
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot: Articulation = scene["robot"]

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["wrist_3_link"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

        # set gripper joint position
        gripper_joint_pos = joint_pos_des[:, -1:] * 0.0
        gripper_width = 0
        gripper_width = math.radians(45)  # open: 0, close=45 degrees
        gripper_joint_pos += gripper_width

        # apply actions
        robot.set_joint_position_target(joint_pos_des[:, :6], joint_ids=robot_entity_cfg.joint_ids[:6])
        robot.set_joint_position_target(gripper_joint_pos, joint_ids=[6])

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        # import ipdb; ipdb.set_trace()
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
