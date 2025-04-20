# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

# Define your robot-specific imports here (like `robot_cfg`)
from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class NaoEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 42  # Updated to match the number of joints
    observation_space = 138
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    # Robot Configuration
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Nao",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/extensions/omni.isaac.lab_assets/data/Robots/Nao/nao.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Corrected Joint Names
                "HeadYaw": 0.0,
                "HeadPitch": 0.0,
                "LHipYawPitch": 0.0,
                "LHipRoll": 0.0,
                "LHipPitch": -0.5,
                "LKneePitch": 0.5,
                "LAnklePitch": -0.5,
                "LAnkleRoll": 0.0,
                "RHipYawPitch": 0.0,
                "RHipRoll": 0.0,
                "RHipPitch": -0.5,
                "RKneePitch": 0.5,
                "RAnklePitch": -0.5,
                "RAnkleRoll": 0.0,
                "LShoulderPitch": 1.0,
                "LShoulderRoll": 0.5,
                "LElbowYaw": 0.5,
                "LElbowRoll": -1.0,
                "LWristYaw": 0.0,
                "RShoulderPitch": 1.0,
                "RShoulderRoll": -0.5,
                "RElbowYaw": 0.5,
                "RElbowRoll": 1.0,
                "RWristYaw": 0.0,
                "LHand": 0.0,
                "RHand": 0.0,
                "LFinger11": 0.0,
                "LFinger21": 0.0,
                "RFinger11": 0.0,
                "RFinger21": 0.0,
                "LThumb1": 0.0,
                "RThumb1": 0.0,
                "LFinger12": 0.0,
                "LFinger22": 0.0,
                "RFinger12": 0.0,
                "RFinger22": 0.0,
                "LFinger13": 0.0,
                "LFinger23": 0.0,
                "RFinger13": 0.0,
                "RFinger23": 0.0,
                "LThumb2": 0.0,
                "RThumb2": 0.0,
            },
            # pos=(1.0, 0.0, 0.00),  # Position of the robot in the environment
            pos=(1.0, 0.0, 0.30),  # Position of the robot in the environment
            rot=(0.0, 0.0, 0.0, 1.0),  # Orientation of the robot (quaternion)
        ),
        actuators={
            # Updated Actuator Configurations
            "Nao_arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
                    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"
                ],
                effort_limit=50.0,
                velocity_limit=1.5,
                stiffness=80.0,
                damping=4.0,
            ),
            "Nao_legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch",
                    "LAnklePitch", "LAnkleRoll", "RHipYawPitch", "RHipRoll",
                    "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"
                ],
                effort_limit=40.0,
                velocity_limit=1.5,
                stiffness=70.0,
                damping=3.0,
            ),
            "Nao_head": ImplicitActuatorCfg(
                joint_names_expr=["HeadYaw", "HeadPitch"],
                effort_limit=10.0,
                velocity_limit=1.0,
                stiffness=100.0,
                damping=5.0,
            ),
            "Nao_hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LHand", "RHand", "LFinger11", "RFinger11",
                    "LFinger12", "RFinger12", "LFinger21", "RFinger21",
                    "LFinger22", "RFinger22", "LThumb1", "RThumb1",
                    "LThumb2", "RThumb2", "LFinger13", "LFinger23",
                    "RFinger13", "RFinger23", "LWristYaw", "RWristYaw"
                ],
                effort_limit=5.0,
                velocity_limit=1.0,
                stiffness=50.0,
                damping=2.0,
            ),
        },

    )

    joint_gears: list = [
        10.0,  # HeadYaw
        10.0,  # HeadPitch
        50.0,  # LHipYawPitch
        50.0,  # LHipRoll
        50.0,  # LHipPitch
        40.0,  # LKneePitch
        40.0,  # LAnklePitch
        50.0,  # LAnkleRoll
        50.0,  # RHipYawPitch
        50.0,  # RHipRoll
        50.0,  # RHipPitch
        40.0,  # RKneePitch
        40.0,  # RAnklePitch
        50.0,  # RAnkleRoll
        80.0,  # LShoulderPitch
        80.0,  # LShoulderRoll
        50.0,  # LElbowYaw
        50.0,  # LElbowRoll
        30.0,  # LWristYaw
        80.0,  # RShoulderPitch
        80.0,  # RShoulderRoll
        50.0,  # RElbowYaw
        50.0,  # RElbowRoll
        30.0,  # RWristYaw
        20.0,  # LHand
        20.0,  # RHand
        10.0,  # LFinger11
        10.0,  # LFinger21
        10.0,  # RFinger11
        10.0,  # RFinger21
        40.0,  # LThumb1
        40.0,  # RThumb1
        10.0,  # LFinger12
        10.0,  # LFinger22
        10.0,  # RFinger12
        10.0,  # RFinger22
        10.0,  # LFinger13
        10.0,  # LFinger23
        10.0,  # RFinger13
        10.0,  # RFinger23
        30.0,  # LThumb2
        30.0,  # RThumb2
    ]



    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01



class NaoEnv(LocomotionEnv):
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
