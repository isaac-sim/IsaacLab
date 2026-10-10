# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the minimal dual-Franka Newton shoelace RL task."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerEntryCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftSceneCfg

from . import mdp
from . import shoelace_constants as physics
from .shoelace_assets import ShoelaceUsdCfg
from .shoelace_contacts_cfg import FingerTailContactSensorCfg

CABLE_CFGS = (SceneEntityCfg("shoelace_left"), SceneEntityCfg("shoelace_right"))
ROBOT_CFGS = (
    SceneEntityCfg("robot_left", joint_names=["panda_finger_joint1"], body_names=["panda_hand"]),
    SceneEntityCfg("robot_right", joint_names=["panda_finger_joint1"], body_names=["panda_hand"]),
)
_FINGER_FRICTION = {"/.*panda_(left|right)finger/.*": physics.FINGER_MU}
_GRIPPER_PARAMS = {
    "open_position": physics.GRIPPER_OPEN_POSITION,
    "closed_position": physics.GRIPPER_CLOSED_POSITION,
    "cable_cfgs": CABLE_CFGS,
    "robot_cfgs": ROBOT_CFGS,
}
_GRASP_PARAMS = {
    **_GRIPPER_PARAMS,
    "contact_std": 5.0e-4,
    "contact_penetration_tolerance": 1.0e-3,
    "relative_speed_std": 0.08,
    "grasp_filter_time_constant": 0.10,
}


def _franka_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    arm_joint_positions: dict[str, float],
) -> ArticulationCfg:
    """Build one fixed-base Franka in the nominal open-pregrasp pose."""
    robot = FrankaSoftSceneCfg().default.robot.replace(prim_path=prim_path)
    robot.spawn = ShoelaceUsdCfg(
        usd_path=robot.spawn.usd_path,
        rigid_props=robot.spawn.rigid_props,
        articulation_props=robot.spawn.articulation_props,
        variants=robot.spawn.variants,
        friction_overrides=_FINGER_FRICTION.copy(),
    )
    robot.init_state.pos = position
    robot.init_state.rot = rotation
    robot.init_state.joint_pos.update(arm_joint_positions)
    robot.init_state.joint_pos["panda_finger_joint.*"] = physics.GRIPPER_OPEN_POSITION
    # Newton ignores the inherited PhysX-only ``disable_gravity``; compensate gravity so zero actions hold pose.
    robot.spawn.rigid_props = {"/.*": [robot.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]}
    # Limit finger speed so closing fingers do not tunnel through the thin cable.
    robot.actuators["panda_hand"].actuator_velocity_limit = 0.04
    # Stiff finger drive for a stable grasp on the cable.
    robot.actuators["panda_hand"].stiffness = physics.GRIPPER_STIFFNESS
    return robot


def _arm_action(asset_name: str) -> env_mdp.DifferentialInverseKinematicsActionCfg:
    """Build one six-dimensional relative TCP action."""
    return env_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name=asset_name,
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(physics.ARM_ACTION_SCALE,) * 3 + (physics.ARM_ROTATION_ACTION_SCALE,) * 3,
        body_offset=env_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=physics.TCP_OFFSET),
    )


@configclass
class ShoelaceSceneCfg(InteractiveSceneCfg):
    """Replicated dual-Franka shoe and shoelace scene."""

    robot_left = _franka_cfg(
        "{ENV_REGEX_NS}/RobotLeft",
        physics.LEFT_ROBOT_POSITION,
        (0.0, 0.0, 0.0, 1.0),
        physics.LEFT_ARM_JOINT_POSITIONS,
    )
    robot_right = _franka_cfg(
        "{ENV_REGEX_NS}/RobotRight",
        physics.RIGHT_ROBOT_POSITION,
        (0.0, 0.0, 1.0, 0.0),
        physics.RIGHT_ARM_JOINT_POSITIONS,
    )
    shoelace_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShoelaceScene",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[ShoelaceUsdCfg(usd_path=str(usd_path)) for usd_path in physics.SHOELACE_ASSETS],
            random_choice=False,
        ),
    )
    shoe = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/ShoelaceScene/Shoe", spawn=None)
    shoelace_left = CableObjectCfg(prim_path="{ENV_REGEX_NS}/ShoelaceScene/ShoelaceLeft", spawn=None)
    shoelace_right = CableObjectCfg(prim_path="{ENV_REGEX_NS}/ShoelaceScene/ShoelaceRight", spawn=None)
    finger_tail_contacts = FingerTailContactSensorCfg(prim_path="{ENV_REGEX_NS}/ShoelaceScene")
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.08, 0.08, 0.08), size=(10.0, 10.0)),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0)),
    )


@configclass
class ActionsCfg:
    """Relative Cartesian arm and binary gripper actions."""

    left_arm = _arm_action("robot_left")
    left_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": physics.GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": physics.GRIPPER_CLOSED_POSITION},
    )
    right_arm = _arm_action("robot_right")
    right_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": physics.GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": physics.GRIPPER_CLOSED_POSITION},
    )


@configclass
class ObservationsCfg:
    """Basic dual-arm proprioception and free-tail state observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
        )
        right_joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
        )
        left_joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        right_joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        gripper_close_error = ObsTerm(
            func=mdp.gripper_close_error,
            params={"robot_cfgs": ROBOT_CFGS},
            clip=(0.0, physics.GRIPPER_OPEN_POSITION - physics.GRIPPER_CLOSED_POSITION),
            scale=1.0 / (physics.GRIPPER_OPEN_POSITION - physics.GRIPPER_CLOSED_POSITION),
        )
        tails_to_tcp = ObsTerm(
            func=mdp.tails_to_tcp,
            params={"cable_cfgs": CABLE_CFGS, "robot_cfgs": ROBOT_CFGS},
            scale=10.0,
        )
        finger_tail_signed_distance = ObsTerm(
            func=mdp.finger_tail_signed_distance,
            clip=(-physics.CONTACT_DISTANCE_CAP, physics.CONTACT_DISTANCE_CAP),
            scale=1.0 / physics.CONTACT_DISTANCE_CAP,
            history_length=physics.CONTACT_OBSERVATION_HISTORY_LENGTH,
        )
        tail_tcp_relative_speed = ObsTerm(
            func=mdp.tail_tcp_relative_speed,
            params={"cable_cfgs": CABLE_CFGS, "robot_cfgs": ROBOT_CFGS},
            scale=10.0,
        )
        last_action = ObsTerm(func=env_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Restore defaults, then perturb arm joints and translate the shoe with its laces."""

    settled_defaults = EventTerm(func=mdp.install_settled_default_state, mode="startup")

    reset_scene = EventTerm(
        func=env_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_left_arm = EventTerm(
        func=mdp.reset_arm_joints,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint[1-7]"]),
        },
    )
    reset_right_arm = EventTerm(
        func=mdp.reset_arm_joints,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint[1-7]"]),
        },
    )
    reset_shoe = EventTerm(
        func=mdp.reset_shoe_position,
        mode="reset",
        params={"position_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}},
    )


@configclass
class RewardsCfg:
    """Cooperative progress, sustained grasping, and completion with small arm motion penalties."""

    # Reward coarse approach and physical grasp acquisition changes, plus new outward pull records.
    dense_task = RewTerm(
        func=mdp.dense_task_reward,
        weight=10.0,
        params={
            **_GRASP_PARAMS,
            "reach_std": 0.08,
            "success_x_separation": physics.TAIL_SUCCESS_X_SEPARATION,
            "acquisition_weight": 0.6,
            "approach_fraction": 0.5,
            "bilateral_approach_fraction": 0.5,
            "bilateral_grasp_fraction": 0.7,
            "bilateral_pull_fraction": 0.8,
            "pull_grasp_threshold": 0.2,
        },
    )
    # Reward fine TCP-to-tail positional alignment and nearby finger-closure progress before contact.
    pregrasp = RewTerm(
        func=mdp.pregrasp_progress_reward,
        weight=1.0,
        params={
            **_GRIPPER_PARAMS,
            "alignment_std": 0.015,
            "closure_radius": 0.01,
            "alignment_weight": 0.75,
            "closure_weight": 0.25,
        },
    )
    # Reward retained contact-based, closed-finger, low-slip grasps even without further task progress.
    grasp_hold = RewTerm(
        func=mdp.grasp_hold_reward,
        weight=1.0,
        params={
            **_GRASP_PARAMS,
            "bilateral_grasp_fraction": 0.5,
            "full_reward_duration": 2.0,
            "sustained_reward_fraction": 0.2,
        },
    )

    success = RewTerm(func=mdp.shoelace_success_reward, weight=5.0)
    arm_action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-0.001)
    arm_action_magnitude = RewTerm(func=mdp.arm_action_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """Cooperative loaded pull followed by geometric completion, and episode timeout."""

    success = DoneTerm(
        func=mdp.shoelace_bilateral_pull_success,
        params={
            **_GRASP_PARAMS,
            "throat_radius": physics.THROAT_RADIUS,
            "maximum_throat_segments_per_arm": physics.MAXIMUM_THROAT_SEGMENTS_PER_ARM,
            "minimum_tail_outward_distance": physics.TAIL_SUCCESS_OUTWARD_DISTANCE,
            "minimum_pull_distance": 0.025,
            "grasp_threshold": 0.2,
        },
    )
    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)


@configclass
class ShoelaceEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal trainable Newton environment for dual-Franka shoelace control."""

    class_type: type | str = "{DIR}.shoelace_physics:create_shoelace_env"

    seed: int | None = 42
    decimation: int = 4
    episode_length_s: float = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=CouplerAdmmCfg(
                contact_max_triangle_pairs=physics.MIN_TRIANGLE_PAIRS,
                contact_reduction_hashtable_size_factor=0.25,
                entries=[
                    CouplerEntryCfg(
                        name="robots",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=40,
                            integrator="implicitfast",
                            njmax=2048,
                            nconmax=256,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/(Robot(Left|Right)|ShoelaceScene/Shoe)"],
                    ),
                    CouplerEntryCfg(
                        name="shoelace",
                        solver_cfg=VBDSolverCfg(
                            iterations=physics.VBD_ITERATIONS,
                            rigid_compliant_alm=True,
                            rigid_body_contact_buffer_size=physics.VBD_CONTACT_BUFFER,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/ShoelaceScene/Shoelace(Left|Right)"],
                        include_static_shapes=True,
                    ),
                ],
                contact_pairs=[("robots", "shoelace")],
                iterations=physics.ADMM_ITERATIONS,
                rho=physics.ADMM_RHO,
                gamma=0.0,
                baumgarte=physics.ADMM_BAUMGARTE,
                rigid_contact_matching=physics.ADMM_CONTACT_MATCHING,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=physics.CONTACTS_PER_ENV * 4,
                max_triangle_pairs=physics.MIN_TRIANGLE_PAIRS,
            ),
            collision_decimation=physics.NEWTON_COLLISION_DECIMATION,
            default_shape_cfg=NewtonShapeCfg(
                gap=physics.CONTACT_GAP,
                ke=2.5e4,
                kd=100.0,
                mu=10.0,
            ),
            num_substeps=physics.NEWTON_NUM_SUBSTEPS,
            use_cuda_graph=True,
        ),
    )
    scene: ShoelaceSceneCfg = ShoelaceSceneCfg(num_envs=4, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    ui_window_class_type = None

    cable_inertia_regularization: float = physics.CABLE_INERTIA_REGULARIZATION
    """Isotropic inertia added to each dynamic cable segment [kg*m^2], without changing its mass."""

    def validate_config(self) -> None:
        """Resolve environment-count-dependent contact capacity and validate the Newton backend."""
        if not isinstance(self.sim.physics, NewtonCfg):
            raise TypeError("The dual-Franka shoelace task requires Newton physics")
        collision = self.sim.physics.collision_cfg
        collision.rigid_contact_max = physics.CONTACTS_PER_ENV * self.scene.num_envs
        collision.max_triangle_pairs = max(
            collision.max_triangle_pairs,
            physics.MIN_TRIANGLE_PAIRS,
            physics.TRIANGLE_PAIRS_PER_ENV * self.scene.num_envs,
        )
        solver = self.sim.physics.solver_cfg
        solver.contact_max_triangle_pairs = max(solver.contact_max_triangle_pairs or 0, physics.MIN_TRIANGLE_PAIRS)
        # Newton 1.6 contact matching needs a triangle budget below 2**20; scale the table instead.
        solver.contact_reduction_hashtable_size_factor = max(
            solver.contact_reduction_hashtable_size_factor or 0.25,
            0.25 * physics.TRIANGLE_PAIRS_PER_ENV * self.scene.num_envs / solver.contact_max_triangle_pairs,
        )
