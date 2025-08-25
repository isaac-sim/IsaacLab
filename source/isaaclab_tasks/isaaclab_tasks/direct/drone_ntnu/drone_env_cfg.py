# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_robot_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-2.0, 2.0),
                "yaw": (-1.0, 1.0),
            },  # yaw translated from xyzw (0, 0, 0.5236, 1) from aerial gym
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 1
    action_space = 4
    observation_space = 13
    state_space = 0
    debug_vis = True
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://isaac-dev.ov.nvidia.com/omni/web3/omniverse://isaac-dev.ov.nvidia.com/Users/zhengyuz@nvidia.com/Robots/NTNU/Quad/quad.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    events: EventCfg = EventCfg()

    # robot related variables -- need further clean
    thrust_to_torque_ratio = 0.01
    application_mask = [5, 6, 7, 8]  # TODO: use SceneEntityCfg
    force_application_level = "motor_link"
    motor_directions = [1, -1, 1, -1]
    allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-0.13, -0.13, 0.13, 0.13],
        [-0.13, 0.13, 0.13, -0.13],
        [-0.01, 0.01, -0.01, 0.01],
    ]

    # drag_variables
    body_vel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    body_vel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]

    # disturbance
    enable_disturbance = False
    max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]
    disturbance_probability = 0.02
