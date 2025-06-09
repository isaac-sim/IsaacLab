# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG  # isort: skip
from .action_cfg import LimitsScaledJointPositionActionCfg

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ISAACLAB_NUCLEUS_DIR = "source/isaaclab_assets/data"
objects_dir = f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/Objects"
sub_dirs = sorted(os.listdir(objects_dir))
sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(os.path.join(objects_dir, object_name))]
object_multi_asset_cfg = [sim_utils.UsdFileCfg(usd_path=os.path.join(objects_dir, obj_name, f"{obj_name}.usd"))for obj_name in sub_dirs]


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1., 1.),
            "damping_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0. , 0.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DextrahKukaAllegroEnvCfg(DirectRLEnvCfg):
    name = __name__

    # Custom Variables
    objects_dir = f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/Objects"

    # Class Expectant Variables
    decimation = 2  # 60 Hz
    episode_length_s = 10.  # 10.0
    state_space = -1  # set by DextrahKukaAllegroEnv Implementation code
    observation_space = -1  # set by DextrahKukaAllegroEnv Implementation code
    action_space = 23 * 2

    # viewer = ViewerCfg(eye=(-2.25, 0., 0.75), lookat=(0., 0., 0.3), origin_type='env')
    viewer = ViewerCfg(eye=(-5.0, 1., 0.75), lookat=(0., 1., 0.3), origin_type='env')
    joint_pos_action_cfg = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
    )
    # joint_pos_action_cfg = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=0.5,
    # )
    # joint_pos_action_cfg = LimitsScaledJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     ema_lambda=1.0
    # )
    joint_vel_action_cfg = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                'iiwa7_joint_1': -0.85,
                'iiwa7_joint_2': 0.0,
                'iiwa7_joint_3': 0.76,
                'iiwa7_joint_4': 1.25,
                'iiwa7_joint_5': -1.76,
                'iiwa7_joint_6': 0.90,
                'iiwa7_joint_7': 0.64,
                'index_joint_(0|1|2)': 0.0,
                'middle_joint_(0|1|2)': 0.3,
                'ring_joint_(0|1|2)': 0.3,
                'thumb_joint_(0|1|2)': 0.3,
                'index_joint_3': 1.5,
                'middle_joint_3': 0.60147215,
                'ring_joint_3': 0.33795027,
                'thumb_joint_3': 0.60845138
            },
        ),
        # Default dof friction coefficients
        # NOTE: these are set based on how far out they will scale multiplicatively
        # with the above joint_friction EventTerm above.
        actuators={
            "kuka_allegro_actuators": KUKA_ALLEGRO_CFG.actuators["kuka_allegro_actuators"].replace(
                friction={
                    "iiwa7_joint_(1|2|3|4|5|6|7)": 1.,
                    "index_joint_(0|1|2|3)": 0.01,
                    "middle_joint_(0|1|2|3)": 0.01,
                    "ring_joint_(0|1|2|3)": 0.01,
                    "thumb_joint_(0|1|2|3)": 0.01,
                }
            )
        },
    )
    
    objects_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=object_multi_asset_cfg,
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                kinematic_enabled=False,
                disable_gravity=False,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.32)),
    )

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.21 - 0.725 / 2, 0.668 - 1.16 / 2, 0.25 - 0.03 / 2),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2., replicate_physics=False)
    events: EventCfg = EventCfg()
    robot_scene_cfg = SceneEntityCfg(
        "robot",
        body_names=["palm_link", "index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip"],
        joint_names=".*"
    )

    curled_q = [0., 0., 0., 1.5, 0., 0., 0., 0.6015, 0., 0., 0., 0.3380, 0., 0., 0., 0.6085]  # isaac-sim order
    object_goal = [-0.5, 0., 0.75]

    # reward weights
    hand_to_object_weight = 1.
    hand_to_object_sharpness = 5.
    object_to_goal_weight = 5.
    lift_sharpness = 8.5

    # Goal reaching parameters
    object_goal_tol = 0.1  # m
    success_for_adr = 0.4
    min_steps_for_dr_change = 5 * int(episode_length_s / (decimation * sim.dt))

    # Object spawning params
    obj_spawn_width = (0.5, 0.8)

    # DR Controls
    enable_adr = True
    num_adr_increments = 50
    starting_adr_increments = 0 # 0 for no DR up to num_adr_increments for max DR

    # Object disturbance wrench fixed params
    wrench_trigger_every = int(1. / (decimation * sim.dt)) # 1 sec
    torsional_radius = 0.01  # m
    hand_to_object_dist_threshold = .3  # m

    # Object scaling
    object_scale = (0.5, 1.75)
    deactivate_object_scaling = True

    # These serve to set the maximum value ranges for the different physics parameters
    adr_cfg_dict = {
        "num_increments": num_adr_increments,  # number of times you can change the parameter ranges
        "robot_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.),
            "damping_distribution_params": (0.5, 2.),
        },
        "joint_friction": {
            "friction_distribution_params": (0., 5.),
        },
        "object_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.),
        },
    }

    # Dictionary of custom parameters for ADR
    # NOTE: first number in range is the starting value, second number is terminal value
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0., 10.)
        },
        "object_spawn": {
            "x_width_spawn": (0., obj_spawn_width[0]),
            "y_width_spawn": (0., obj_spawn_width[1]),
            "rotation": (0., 1.)
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03), # m
            "object_pos_bias": (0.0, 0.02), # m
            "object_rot_noise": (0.0, 0.1), # rad
            "object_rot_bias": (0.0, 0.08), # rad
        },
        "robot_spawn": {
            "joint_pos_noise": (0., 0.35),
            "joint_vel_noise": (0., 1.)
        },
        "robot_state_noise": {
            "joint_pos_noise": (0.0, 0.08), # rad
            "joint_pos_bias": (0.0, 0.08), # rad
            "joint_vel_noise": (0.0, 0.18), # rad
            "joint_vel_bias": (0.0, 0.08), # rad
        },
        "reward_weights": {
            "finger_curl_reg": (-0.01, -0.005),
            "object_to_goal_sharpness": (15., 20.),
            "lift_weight": (5., 0.)
        },
        "pd_targets": {
            "velocity_target_factor": (1., 0.)
        },
        "fabric_damping": {
            "gain": (10., 20.)
        },
        "observation_annealing": {
            "coefficient": (0., 0.)
        },
    }