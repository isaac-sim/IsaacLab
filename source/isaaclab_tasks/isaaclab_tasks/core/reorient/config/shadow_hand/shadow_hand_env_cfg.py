# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.core.reorient.reorient_task_constants import (
    OPENAI_ACT_MOVING_AVERAGE,
    OPENAI_ACTION_PENALTY_SCALE,
    OPENAI_AV_FACTOR,
    OPENAI_DECIMATION,
    OPENAI_DIST_REWARD_SCALE,
    OPENAI_EPISODE_LENGTH_S,
    OPENAI_FALL_PENALTY,
    OPENAI_FORCE_TORQUE_OBS_SCALE,
    OPENAI_MAX_CONSECUTIVE_SUCCESS,
    OPENAI_REACH_GOAL_BONUS,
    OPENAI_RESET_DOF_POS_NOISE,
    OPENAI_RESET_DOF_VEL_NOISE,
    OPENAI_RESET_POSITION_NOISE,
    OPENAI_ROT_EPS,
    OPENAI_ROT_REWARD_SCALE,
    OPENAI_SIM_DT,
    OPENAI_SUCCESS_TOLERANCE,
    OPENAI_VEL_OBS_SCALE,
    SHADOW_ACT_MOVING_AVERAGE,
    SHADOW_ACTION_PENALTY_SCALE,
    SHADOW_ACTUATED_JOINT_NAMES,
    SHADOW_AV_FACTOR,
    SHADOW_DECIMATION,
    SHADOW_DIST_REWARD_SCALE,
    SHADOW_EPISODE_LENGTH_S,
    SHADOW_FALL_DIST,
    SHADOW_FALL_PENALTY,
    SHADOW_FINGERTIP_BODY_NAMES,
    SHADOW_FORCE_TORQUE_OBS_SCALE,
    SHADOW_REACH_GOAL_BONUS,
    SHADOW_RESET_DOF_POS_NOISE,
    SHADOW_RESET_DOF_VEL_NOISE,
    SHADOW_RESET_POSITION_NOISE,
    SHADOW_ROT_EPS,
    SHADOW_ROT_REWARD_SCALE,
    SHADOW_SIM_DT,
    SHADOW_SUCCESS_COUNT_THRESHOLD,
    SHADOW_SUCCESS_TOLERANCE,
    SHADOW_VEL_OBS_SCALE,
)
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
            "recompute_inertia": False,
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
    newton_mjwarp = NewtonEventCfg()
    default = physx
    newton_kamino = newton_mjwarp


@configclass
class ShadowHandRobotCfg(PresetCfg):
    physx = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        )
    )
    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # newton/mujoco have separate usd schema
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHandNewton/shadow_hand_instanceable.usda",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(damping=0.1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # WARNING(Octi): Newton's import_usd.py bakes the USD body xformOp rotation into
            # joint_X_p for the root fixed joint, which cancels with the matching localPose1
            # rotation in joint_X_c during FK (joint_X_p * inv(joint_X_c) ≈ identity). This
            # discards the root body's native USD orientation, so we must re-apply it here as a
            # spawn rotation. PhysX or USD does not have this issue. Remove once Newton fixes root joint
            # transform handling in import_usd.py.
            rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
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
    ovphysx = SHADOW_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # OVPhysX does not expose the fixed-tendon runtime API, so spawn without tendon overrides.
        spawn=SHADOW_HAND_CFG.spawn.replace(fixed_tendons_props=None),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
    )
    default = physx
    newton_kamino = newton_mjwarp


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

    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            semantic_tags=[("class", "cube")],
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.36, 0.535), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    default = physx
    newton_kamino = newton_mjwarp


@configclass
class ShadowHandSceneCfg(PresetCfg):
    """Scene configuration presets for the shadow hand environment.

    PhysX supports ``clone_in_fabric=True`` for faster scene cloning via the Fabric layer.
    Newton does not support Fabric cloning, so ``clone_in_fabric`` must be ``False``.
    """

    physx: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    newton_mjwarp: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    default: InteractiveSceneCfg = physx
    newton_kamino = newton_mjwarp


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
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
    ovphysx = OvPhysxCfg()
    default = physx
    newton_kamino = NewtonCfg(solver_cfg=KaminoSolverCfg(max_contacts_per_world=128))


# Scene pieces shared verbatim by the manager-based variants.
ROBOT_CFG = ShadowHandRobotCfg()
OBJECT_CFG = ObjectCfg()
GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
        )
    },
)


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = SHADOW_DECIMATION
    episode_length_s = SHADOW_EPISODE_LENGTH_S
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SHADOW_SIM_DT,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # robot
    robot_cfg: ShadowHandRobotCfg = ROBOT_CFG
    actuated_joint_names = SHADOW_ACTUATED_JOINT_NAMES
    fingertip_body_names = SHADOW_FINGERTIP_BODY_NAMES

    # in-hand object
    object_cfg: ObjectCfg = OBJECT_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_OBJECT_CFG
    # scene — use ShadowHandSceneCfg so that presets=newton_mjwarp disables clone_in_fabric automatically
    scene: ShadowHandSceneCfg = ShadowHandSceneCfg()

    # reset
    reset_position_noise = SHADOW_RESET_POSITION_NOISE  # range of position at reset
    reset_dof_pos_noise = SHADOW_RESET_DOF_POS_NOISE  # range of dof pos at reset
    reset_dof_vel_noise = SHADOW_RESET_DOF_VEL_NOISE  # range of dof vel at reset
    # reward scales
    dist_reward_scale = SHADOW_DIST_REWARD_SCALE
    rot_reward_scale = SHADOW_ROT_REWARD_SCALE
    rot_eps = SHADOW_ROT_EPS
    action_penalty_scale = SHADOW_ACTION_PENALTY_SCALE
    reach_goal_bonus = SHADOW_REACH_GOAL_BONUS
    fall_penalty = SHADOW_FALL_PENALTY
    fall_dist = SHADOW_FALL_DIST
    vel_obs_scale = SHADOW_VEL_OBS_SCALE
    success_tolerance = SHADOW_SUCCESS_TOLERANCE
    max_consecutive_success = 0
    success_count_threshold: int = SHADOW_SUCCESS_COUNT_THRESHOLD
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    av_factor = SHADOW_AV_FACTOR
    act_moving_average = SHADOW_ACT_MOVING_AVERAGE
    force_torque_obs_scale = SHADOW_FORCE_TORQUE_OBS_SCALE


# Per-step gaussian noise + reset-sampled bias, shared verbatim by the manager-based variant.
OPENAI_ACTION_NOISE_CFG = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
)
OPENAI_OBSERVATION_NOISE_CFG = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
)


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):
    # env
    decimation = OPENAI_DECIMATION
    episode_length_s = OPENAI_EPISODE_LENGTH_S
    action_space = 20
    observation_space = 42
    state_space = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=OPENAI_SIM_DT,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    # reset
    reset_position_noise = OPENAI_RESET_POSITION_NOISE  # range of position at reset
    reset_dof_pos_noise = OPENAI_RESET_DOF_POS_NOISE  # range of dof pos at reset
    reset_dof_vel_noise = OPENAI_RESET_DOF_VEL_NOISE  # range of dof vel at reset
    # reward scales
    dist_reward_scale = OPENAI_DIST_REWARD_SCALE
    rot_reward_scale = OPENAI_ROT_REWARD_SCALE
    rot_eps = OPENAI_ROT_EPS
    action_penalty_scale = OPENAI_ACTION_PENALTY_SCALE
    reach_goal_bonus = OPENAI_REACH_GOAL_BONUS
    fall_penalty = OPENAI_FALL_PENALTY
    vel_obs_scale = OPENAI_VEL_OBS_SCALE
    success_tolerance = OPENAI_SUCCESS_TOLERANCE
    max_consecutive_success = OPENAI_MAX_CONSECUTIVE_SUCCESS
    av_factor = OPENAI_AV_FACTOR
    act_moving_average = OPENAI_ACT_MOVING_AVERAGE
    force_torque_obs_scale = OPENAI_FORCE_TORQUE_OBS_SCALE
    # domain randomization config
    events: ShadowHandEventCfg = ShadowHandEventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = OPENAI_ACTION_NOISE_CFG
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = OPENAI_OBSERVATION_NOISE_CFG
