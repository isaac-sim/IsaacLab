# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Shadow Hand handover task."""

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import config_field

import isaaclab_tasks.core.handover.mdp as mdp
import isaaclab_tasks.core.reorient.mdp as reorient_mdp
from isaaclab_tasks.core.handover.handover_env_cfg import (
    BALL_CFG,
    LeftHandCfg,
    PhysicsCfg,
    RightHandCfg,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import (
    FINGERTIP_NAMES,
    JOINT_NAMES,
    TENDON_NAMES,
    TENDON_POSITION_LIMITS,
)


@dataclass
class HandoverManagerSceneCfg(InteractiveSceneCfg):
    """Two Shadow hands facing each other over a ground plane."""

    num_envs: Any = config_field(2048)
    env_spacing: Any = config_field(1.5)
    replicate_physics: Any = config_field(True)

    ground: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )
    )
    right_hand: RightHandCfg = config_field(RightHandCfg())
    left_hand: LeftHandCfg = config_field(LeftHandCfg())
    object: RigidObjectCfg = config_field(BALL_CFG)
    light: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )
    )


@dataclass
class CommandsCfg:
    """Handover goal command."""

    object_pose: Any = config_field(
        mdp.HandoverCommandCfg(asset_name="object", success_distance_threshold=0.1, debug_vis=True)
    )


@dataclass
class ActionsCfg:
    """Two-hand action terms, ordered right then left like the Direct adapter.

    Declaration order is the action layout: the manager concatenates terms as declared, and the
    Direct adapter reads each hand as its sixteen joints followed by its four tendons.
    """

    right_hand: Any = config_field(
        mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="right_hand",
            joint_names=JOINT_NAMES,
            alpha=1.0,
            rescale_to_limits=True,
        )
    )
    right_hand_tendons: Any = config_field(
        mdp.FixedTendonPositionActionCfg(
            asset_name="right_hand",
            tendon_names=TENDON_NAMES,
            # the other four motors pull a tendon across a finger's middle and distal joints;
            # tendons have their own index space, so no joint term can reach them. Map the
            # policy's [-1, 1] onto the tendon's commandable span.
            scale=0.5 * (TENDON_POSITION_LIMITS[1] - TENDON_POSITION_LIMITS[0]),
            offset=0.5 * (TENDON_POSITION_LIMITS[0] + TENDON_POSITION_LIMITS[1]),
            # the term maps [-1, 1] onto that span; clip states the bound the task guarantees
            clip={".*": TENDON_POSITION_LIMITS},
        )
    )
    left_hand: Any = config_field(
        mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="left_hand",
            joint_names=JOINT_NAMES,
            alpha=1.0,
            rescale_to_limits=True,
        )
    )
    left_hand_tendons: Any = config_field(
        mdp.FixedTendonPositionActionCfg(
            asset_name="left_hand",
            tendon_names=TENDON_NAMES,
            # the other four motors pull a tendon across a finger's middle and distal joints;
            # tendons have their own index space, so no joint term can reach them. Map the
            # policy's [-1, 1] onto the tendon's commandable span.
            scale=0.5 * (TENDON_POSITION_LIMITS[1] - TENDON_POSITION_LIMITS[0]),
            offset=0.5 * (TENDON_POSITION_LIMITS[0] + TENDON_POSITION_LIMITS[1]),
            # the term maps [-1, 1] onto that span; clip states the bound the task guarantees
            clip={".*": TENDON_POSITION_LIMITS},
        )
    )


@dataclass
class PolicyCfg(ObsGroup):
    # Right agent: 133 hand dimensions followed by 24 object/goal dimensions.
    # soft limits equal the hard limits here: soft_joint_pos_limits_factor defaults to 1.0
    right_joint_pos: Any = config_field(
        ObsTerm(
            func=mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("right_hand", joint_names=".*")}
        )
    )
    right_joint_vel: Any = config_field(
        ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": SceneEntityCfg("right_hand", joint_names=".*")})
    )
    right_fingertip_pose: Any = config_field(
        ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("right_hand", body_names=FINGERTIP_NAMES)})
    )
    right_fingertip_vel: Any = config_field(
        ObsTerm(
            func=reorient_mdp.fingertip_vel,
            params={"asset_cfg": SceneEntityCfg("right_hand", body_names=FINGERTIP_NAMES)},
        )
    )
    right_action: Any = config_field(ObsTerm(func=mdp.last_action, params={"action_name": "right_hand"}))
    # A hand's motors span two action terms, so its previous command does too: without the tendon
    # term the policy sees 16 of the 20 actions it took, and the group falls 4 short of the 133
    # hand dimensions the Direct task lays out.
    right_tendon_action: Any = config_field(ObsTerm(func=mdp.last_action, params={"action_name": "right_hand_tendons"}))
    object_pos: Any = config_field(ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")}))
    object_quat: Any = config_field(ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")}))
    object_lin_vel: Any = config_field(ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")}))
    object_ang_vel: Any = config_field(
        ObsTerm(func=mdp.root_ang_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("object")})
    )
    goal_pose: Any = config_field(ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"}))
    goal_quat_diff: Any = config_field(
        ObsTerm(
            func=reorient_mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
    )

    # Left agent: the same 157-dimensional layout.
    # soft limits equal the hard limits here: soft_joint_pos_limits_factor defaults to 1.0
    left_joint_pos: Any = config_field(
        ObsTerm(
            func=mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("left_hand", joint_names=".*")}
        )
    )
    left_joint_vel: Any = config_field(
        ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": SceneEntityCfg("left_hand", joint_names=".*")})
    )
    left_fingertip_pose: Any = config_field(
        ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("left_hand", body_names=FINGERTIP_NAMES)})
    )
    left_fingertip_vel: Any = config_field(
        ObsTerm(
            func=reorient_mdp.fingertip_vel,
            params={"asset_cfg": SceneEntityCfg("left_hand", body_names=FINGERTIP_NAMES)},
        )
    )
    left_action: Any = config_field(ObsTerm(func=mdp.last_action, params={"action_name": "left_hand"}))
    left_tendon_action: Any = config_field(ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_tendons"}))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@dataclass
class ObservationsCfg:
    """Single-agent observations matching the Direct MARL adapter."""

    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class RandomizationEventCfg:
    """Randomization of both hands and the object, applied on every physics backend."""

    right_hand_joint_stiffness_and_damping: Any = config_field(
        EventTerm(
            func=mdp.randomize_actuator_gains,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("right_hand"),
                "stiffness_distribution_params": (0.75, 1.5),
                "damping_distribution_params": (0.3, 3.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
    )
    left_hand_joint_stiffness_and_damping: Any = config_field(
        EventTerm(
            func=mdp.randomize_actuator_gains,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("left_hand"),
                "stiffness_distribution_params": (0.75, 1.5),
                "damping_distribution_params": (0.3, 3.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
    )
    object_scale_mass: Any = config_field(
        EventTerm(
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
    )

    # -- scene
    reset_gravity: Any = config_field(
        EventTerm(
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
    )


@dataclass
class ResetEventCfg:
    """Reset distributions matching the Direct handover environment."""

    reset_object: Any = config_field(
        EventTerm(
            func=mdp.reset_root_state_with_random_orientation,
            mode="reset",
            params={
                # the Direct task jitters the drop position and samples a random orientation
                "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},  # [m]
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
    )
    reset_right_hand: Any = config_field(
        EventTerm(
            func=reorient_mdp.reset_reorient_hand,
            mode="reset",
            params={
                "joint_position_noise": 0.2,  # [rad]
                "joint_velocity_noise": 0.0,  # [rad/s]
                "robot_cfg": SceneEntityCfg("right_hand"),
            },
        )
    )
    reset_left_hand: Any = config_field(
        EventTerm(
            func=reorient_mdp.reset_reorient_hand,
            mode="reset",
            params={
                "joint_position_noise": 0.2,  # [rad]
                "joint_velocity_noise": 0.0,  # [rad/s]
                "robot_cfg": SceneEntityCfg("left_hand"),
            },
        )
    )


@dataclass
class HandoverEventCfg(RandomizationEventCfg, ResetEventCfg):
    """Randomization plus the state reset the manager task applies on every episode."""


@dataclass
class HandoverEventPresetCfg(PresetCfg):
    """``presets=randomized`` adds the domain-randomization terms to the reset."""

    randomized: Any = config_field(HandoverEventCfg())
    default: Any = config_field(ResetEventCfg())


@dataclass
class RewardsCfg:
    """Summed two-agent reward exposed by the Direct single-agent adapter."""

    goal_distance: Any = config_field(
        RewTerm(
            func=mdp.handover_goal_distance_reward,
            weight=1.0,
            params={
                "command_name": "object_pose",
                "distance_scale": 20.0,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
    )


@dataclass
class TerminationsCfg:
    """Termination conditions for the handover task.

    The generic ``time_out`` term ends an episode one control step later than the Direct
    environment, which stops at ``max_episode_length - 1``.
    """

    object_out_of_reach: Any = config_field(
        DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.24, "asset_cfg": SceneEntityCfg("object")},
        )
    )
    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))


@dataclass
class HandoverManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based handover environment matching the Direct RSL-RL view."""

    scene: HandoverManagerSceneCfg = config_field(HandoverManagerSceneCfg())
    observations: ObservationsCfg = config_field(ObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    commands: CommandsCfg = config_field(CommandsCfg())
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: HandoverEventPresetCfg = config_field(HandoverEventPresetCfg())

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.5
        # simulation — mirrors the Direct cfg
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        self.sim.physics = PhysicsCfg()
        self.viewer.eye = (2.0, 2.0, 2.0)
