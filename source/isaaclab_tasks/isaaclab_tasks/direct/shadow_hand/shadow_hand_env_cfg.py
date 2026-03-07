# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG


@configclass
class NewtonEventCfg:
    """Event randomization config for the Newton physics backend.

    Includes joint-parameter, mass, and gravity randomization.
    Material and tendon randomization are omitted: Newton does not expose
    per-body friction-material buckets or fixed-tendon APIs.
    """

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class PhysxEventCfg:
    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )


@configclass
class ShadowHandEventCfg(PresetCfg):
    physx = PhysxEventCfg()
    newton = NewtonEventCfg()
    default = physx


@configclass
class ShadowHandRobotCfg(PresetCfg):
    physx = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        )
    )
    newton = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # newton requires implicitactuators be specified in usd and there's a bug with physx tendons
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable_newton.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # WARNING(Octi): Newton's import_usd.py bakes the USD body xformOp rotation into
            # joint_X_p for the root fixed joint, which cancels with the matching localPose1
            # rotation in joint_X_c during FK (joint_X_p * inv(joint_X_c) ≈ identity). This
            # discards the root body's native USD orientation, so we must re-apply it here as a
            # spawn rotation. PhysX or USD does not have this issue. Remove once Newton fixes root joint
            # transform handling in import_usd.py.
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
                effort_limit_sim={
                    "robot0_WRJ1": 4.785,
                    "robot0_WRJ0": 2.175,
                    "robot0_(FF|MF|RF|LF)J1": 0.7245,
                    "robot0_FFJ(3|2)": 0.9,
                    "robot0_MFJ(3|2)": 0.9,
                    "robot0_RFJ(3|2)": 0.9,
                    "robot0_LFJ(4|3|2)": 0.9,
                    "robot0_THJ4": 2.3722,
                    "robot0_THJ3": 1.45,
                    "robot0_THJ(2|1)": 0.99,
                    "robot0_THJ0": 0.81,
                },
                stiffness={
                    "robot0_WRJ.*": 5.0,
                    "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                    "robot0_(LF|TH)J4": 1.0,
                    "robot0_THJ0": 1.0,
                },
                damping={
                    "robot0_WRJ.*": 0.5,
                    "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                    "robot0_(LF|TH)J4": 0.1,
                    "robot0_THJ0": 0.1,
                },
                friction=1e-2,
                armature=2e-3,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    default = physx


@configclass
class ObjectCfg(PresetCfg):
    physx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    newton = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.36, 0.535), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    default = physx


@configclass
class ShadowHandSceneCfg(PresetCfg):
    """Scene configuration presets for the shadow hand environment.

    PhysX supports ``clone_in_fabric=True`` for faster scene cloning via the Fabric layer.
    Newton does not support Fabric cloning, so ``clone_in_fabric`` must be ``False``.
    """

    physx: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    newton: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    default: InteractiveSceneCfg = physx


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = physx


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # robot
    robot_cfg: ShadowHandRobotCfg = ShadowHandRobotCfg()
    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    # in-hand object
    object_cfg: ObjectCfg = ObjectCfg()
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene — use ShadowHandSceneCfg so that presets=newton disables clone_in_fabric automatically
    scene: ShadowHandSceneCfg = ShadowHandSceneCfg()

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):
    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 20
    observation_space = 42
    state_space = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: ShadowHandEventCfg = ShadowHandEventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
