# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold-pose environment for the Disney DR Legs closed-loop biped.

The robot must keep its pelvis upright at a target height.
"""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg, NewtonShapeCfg
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
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.dr_legs.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.dr_legs import DR_LEGS_ACTUATED_JOINTS, DR_LEGS_IMPLICIT_PD_CFG

##
# Shared constants
##

_NUM_ENVS = 4096

_ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=DR_LEGS_ACTUATED_JOINTS, preserve_order=True)

_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
)


def _kamino_newton_cfg() -> NewtonCfg:
    """Kamino solver preset for DR Legs (closed-loop, implicit PD)."""
    return NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            sparse_jacobian=True,
            sparse_dynamics=False,
            use_collision_detector=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
            use_fk_solver=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1.0e-5,
            padmm_dual_tolerance=1.0e-5,
            padmm_compl_tolerance=1.0e-5,
            padmm_rho_0=0.02,
            padmm_eta=1.0e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            max_contacts_per_world=32,
        ),
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.001),
    )


@configclass
class DrLegsPhysicsCfg(PresetCfg):
    """Physics backend presets for DR Legs."""

    default: NewtonCfg = _kamino_newton_cfg()
    newton_kamino: NewtonCfg = _kamino_newton_cfg()
    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)


##
# Scene definition
##


@configclass
class HoldPoseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), physics_material=_PHYSICS_MATERIAL),
    )

    robot = DR_LEGS_IMPLICIT_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DR_LEGS_ACTUATED_JOINTS,
        preserve_order=True,
        scale=0.3,
        use_default_offset=True,
    )


def _physx_actions_cfg() -> ActionsCfg:
    """Return the reduced PhysX joint-target action profile."""
    cfg = ActionsCfg()
    cfg.joint_pos.scale = 0.1
    return cfg


@configclass
class DrLegsActionsCfg(PresetCfg):
    """Backend-specific DR Legs action presets."""

    default: ActionsCfg = ActionsCfg()
    newton_kamino: ActionsCfg = ActionsCfg()
    physx: ActionsCfg = _physx_actions_cfg()
    isaacsim_physx: ActionsCfg = physx


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ACTUATED_JOINT_CFG})
        setpoint_hist = ObsTerm(
            func=mdp.joint_position_setpoints,
            params={"action_term_name": "joint_pos"},
            history_length=2,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": _ACTUATED_JOINT_CFG,
            "operation": "scale",
            "distribution": "uniform",
            "armature_distribution_params": (0.8, 1.2),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": _ACTUATED_JOINT_CFG,
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.01, 0.015),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )


def _physx_event_cfg() -> EventCfg:
    """Return the PhysX reset profile with assembled joint coordinates."""
    cfg = EventCfg()
    cfg.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    return cfg


@configclass
class DrLegsEventCfg(PresetCfg):
    """Backend-specific DR Legs event presets."""

    default: EventCfg = EventCfg()
    newton_kamino: EventCfg = EventCfg()
    physx: EventCfg = _physx_event_cfg()
    isaacsim_physx: EventCfg = physx


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=5.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.265})
    lin_vel = RewTerm(func=mdp.base_lin_vel_l2, weight=-1.0)
    ang_vel = RewTerm(func=mdp.base_ang_vel_l2, weight=-0.5)
    joint_torque = RewTerm(
        func=mdp.joint_pd_command_l2,
        weight=-1.0e-5,
        params={"asset_cfg": _ACTUATED_JOINT_CFG, "stiffness": 5.0, "damping": 0.2},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_rate2 = RewTerm(func=mdp.ActionRate2L2, weight=-0.01)
    joint_pos_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": _ACTUATED_JOINT_CFG},
    )
    # Success metric (zero-weight, metric only): survived the full episode without falling/tilting.
    success_rate = RewTerm(func=mdp.survival_success_rate, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.12})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


##
# Environment configuration
##


@configclass
class DrLegsHoldPoseEnvCfg(ManagerBasedRLEnvCfg):
    """DR Legs hold-pose environment."""

    scene: HoldPoseSceneCfg = HoldPoseSceneCfg(num_envs=_NUM_ENVS, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: DrLegsActionsCfg = DrLegsActionsCfg()
    events: DrLegsEventCfg = DrLegsEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 150,
        render_interval=3,
        physics=DrLegsPhysicsCfg(),
        physics_material=_PHYSICS_MATERIAL,
    )

    def __post_init__(self) -> None:
        self.decimation = 3
        self.episode_length_s = 10.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.5, 0.5, 0.5), lookat=(0.0, 0.0, 0.265))
