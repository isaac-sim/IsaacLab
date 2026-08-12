# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Shadow Hand handover task."""

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
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.handover.mdp as mdp
from isaaclab_tasks.core.handover.handover_common import (
    ACTUATED_JOINT_NAMES_PRESET,
    FINGERTIP_BODY_NAMES,
)
from isaaclab_tasks.core.handover.handover_env_cfg import (
    LEFT_HAND_CFG,
    RIGHT_HAND_CFG,
    ObjectCfg,
    PhysicsCfg,
)
from isaaclab_tasks.utils import PresetCfg


@configclass
class HandoverManagerSceneCfg(InteractiveSceneCfg):
    """Two Shadow hands facing each other over a ground plane."""

    num_envs = 2048
    env_spacing = 1.5
    replicate_physics = True

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    right_hand: PresetCfg = RIGHT_HAND_CFG
    left_hand: PresetCfg = LEFT_HAND_CFG
    object: ObjectCfg = ObjectCfg()
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    """Handover goal command."""

    object_pose = mdp.HandoverCommandCfg(asset_name="object", success_distance_threshold=0.1, debug_vis=True)


@configclass
class ActionsCfg:
    """Two-hand action terms, ordered right then left like the Direct adapter."""

    right_hand = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="right_hand",
        joint_names=ACTUATED_JOINT_NAMES_PRESET,
        alpha=1.0,
        rescale_to_limits=True,
    )
    left_hand = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="left_hand",
        joint_names=ACTUATED_JOINT_NAMES_PRESET,
        alpha=1.0,
        rescale_to_limits=True,
    )


def _hand_entity(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name, joint_names=".*")


def _fingertip_entity(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name, body_names=FINGERTIP_BODY_NAMES)


@configclass
class ObservationsCfg:
    """Single-agent observations matching the Direct MARL adapter."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Right agent: 133 hand dimensions followed by 24 object/goal dimensions.
        # soft limits equal the hard limits here: soft_joint_pos_limits_factor defaults to 1.0
        right_joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": _hand_entity("right_hand")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": _hand_entity("right_hand")})
        right_fingertip_pos = ObsTerm(func=mdp.fingertip_pos, params={"asset_cfg": _fingertip_entity("right_hand")})
        right_fingertip_quat = ObsTerm(func=mdp.fingertip_quat, params={"asset_cfg": _fingertip_entity("right_hand")})
        right_fingertip_vel = ObsTerm(func=mdp.fingertip_vel, params={"asset_cfg": _fingertip_entity("right_hand")})
        right_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_hand"})
        right_object_goal = ObsTerm(
            func=mdp.object_goal,
            params={"command_name": "object_pose", "object_cfg": SceneEntityCfg("object"), "vel_obs_scale": 0.2},
        )

        # Left agent: the same 157-dimensional layout.
        # soft limits equal the hard limits here: soft_joint_pos_limits_factor defaults to 1.0
        left_joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": _hand_entity("left_hand")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": _hand_entity("left_hand")})
        left_fingertip_pos = ObsTerm(func=mdp.fingertip_pos, params={"asset_cfg": _fingertip_entity("left_hand")})
        left_fingertip_quat = ObsTerm(func=mdp.fingertip_quat, params={"asset_cfg": _fingertip_entity("left_hand")})
        left_fingertip_vel = ObsTerm(func=mdp.fingertip_vel, params={"asset_cfg": _fingertip_entity("left_hand")})
        left_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand"})
        left_object_goal = ObsTerm(
            func=mdp.object_goal,
            params={"command_name": "object_pose", "object_cfg": SceneEntityCfg("object"), "vel_obs_scale": 0.2},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset distributions matching the Direct handover environment."""

    reset_object = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            # the Direct task jitters the drop position and samples a random orientation
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},  # [m]
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_hands = EventTerm(
        func=mdp.reset_handover_hands,
        mode="reset",
        params={
            "joint_position_noise": 0.2,  # [rad]
            "joint_velocity_noise": 0.0,  # [rad/s]
        },
    )


@configclass
class RewardsCfg:
    """Summed two-agent reward exposed by the Direct single-agent adapter."""

    goal_distance = RewTerm(
        func=mdp.handover_goal_distance_reward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": 20.0,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the handover task.

    The generic ``time_out`` term ends an episode one control step later than the Direct
    environment, which stops at ``max_episode_length - 1``.
    """

    object_out_of_reach = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.24, "asset_cfg": SceneEntityCfg("object")},
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class HandoverManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based handover environment matching the Direct RSL-RL view."""

    scene: HandoverManagerSceneCfg = HandoverManagerSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.5
        # simulation — mirrors the Direct cfg
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        self.sim.physics = PhysicsCfg()
        self.viewer.eye = (2.0, 2.0, 2.0)
