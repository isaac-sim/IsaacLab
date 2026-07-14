# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import FeatherPGSSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.locomotion.mdp as mdp
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.humanoid import HUMANOID_CFG  # isort:skip


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            njmax=80,
            nconmax=25,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    feather_pgs: NewtonCfg = NewtonCfg(
        solver_cfg=FeatherPGSSolverCfg(
            angular_damping=5.0,
            enable_joint_limits=True,
            pgs_iterations=24,
            pgs_beta=0.002,
            pgs_cfm=1.0e-4,
            pgs_omega=0.5,
            dense_max_constraints=64,
            mf_max_constraints=512,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )


##
# Scene definition
##


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["body"].armature = preset(default=robot.actuators["body"].armature, feather_pgs=0.2)

    # sensors
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"sensor_cfg": SceneEntityCfg("joint_wrench", body_names=["left_foot", "right_foot"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class HumanoidObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()
    feather_pgs: ObservationsCfg = newton_mjwarp


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class FeatherPGSEventCfg(EventCfg):
    robot_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (2.0, 2.0),
            "operation": "abs",
            "recompute_inertia": False,
        },
    )
    robot_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "inertia_distribution_params": (2.0, 2.0),
            "operation": "add",
            "diagonal_only": True,
        },
    )


@configclass
class HumanoidEventCfg(PresetCfg):
    default = EventCfg()
    feather_pgs = FeatherPGSEventCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # (3) Reward for upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.005,
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})


@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Humanoid walking environment."""

    # Scene settings
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    # Basic settings
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: HumanoidEventCfg = HumanoidEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = HumanoidPhysicsCfg()
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
