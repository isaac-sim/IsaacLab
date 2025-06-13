# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .assembly_tasks_cfg import ASSET_DIR, Insertion

OBS_DIM_CFG = {
    "joint_pos": 7,
    "fingertip_pos": 3,
    "fingertip_quat": 4,
    "fingertip_goal_pos": 3,
    "fingertip_goal_quat": 4,
    "delta_pos": 3,
}

STATE_DIM_CFG = {
    "joint_pos": 7,
    "joint_vel": 7,
    "fingertip_pos": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "fingertip_goal_pos": 3,
    "fingertip_goal_quat": 4,
    "held_pos": 3,
    "held_quat": 4,
    "delta_pos": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.1, 0.1, 0.1]
    rot_action_bounds = [0.01, 0.01, 0.01]

    pos_action_threshold = [0.1, 0.1, 0.1]
    rot_action_threshold = [0.01, 0.01, 0.01]

    reset_joints = [0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398]
    reset_task_prop_gains = [1000, 1000, 1000, 50, 50, 50]
    # reset_rot_deriv_scale = 1.0
    # default_task_prop_gains = [1000, 1000, 1000, 50, 50, 50]
    # reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class AssemblyEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 24
    state_space = 44
    obs_order: list = [
        "joint_pos",
        "fingertip_pos",
        "fingertip_quat",
        "fingertip_goal_pos",
        "fingertip_goal_quat",
        "delta_pos",
    ]
    state_order: list = [
        "joint_pos",
        "joint_vel",
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "fingertip_goal_pos",
        "fingertip_goal_quat",
        "held_pos",
        "held_quat",
        "delta_pos",
    ]

    task_name: str = "insertion"  # peg_insertion, gear_meshing, nut_threading
    tasks: dict = {"insertion": Insertion()}
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    # episode_length_s = 10.0  # Probably need to override.
    episode_length_s = 5.0
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            # usd_path=f'{ASSET_DIR}/automate_franka.usd',
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=1, debug_vis=True
    # )
