# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the state-based Shadow Hand reorientation task."""

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

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    ROBOT_CFG,
    SHADOW_ACTUATED_JOINT_NAMES,
    SHADOW_FINGERTIP_BODY_NAMES,
    ObjectCfg,
    PhysicsCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.utils import PresetCfg

# ---------------------------------- state task ----------------------------------


@configclass
class _ShadowHandManagerSceneCfg(InteractiveSceneCfg):
    """Scene shared by the Shadow Hand Manager backend alternatives."""

    num_envs = 8192
    env_spacing = 0.75
    replicate_physics = True

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: PresetCfg = ROBOT_CFG
    object: ObjectCfg = OBJECT_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ShadowHandManagerSceneCfg(PresetCfg):
    """Backend-specific scene cloning settings matching the Direct task."""

    physx = _ShadowHandManagerSceneCfg(clone_in_fabric=True)
    newton_mjwarp = _ShadowHandManagerSceneCfg(clone_in_fabric=False)
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Twenty actuated Shadow Hand joints."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=1.0,
        rescale_to_limits=True,
    )


@configclass
class FullStateWithoutActionCfg(ObsGroup):
    """Shared first 137 dimensions of the full Shadow state."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
    )
    object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )
    object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_ang_vel = ObsTerm(
        func=mdp.root_ang_vel_w,
        scale=0.2,
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    goal_quat_diff = ObsTerm(
        func=mdp.goal_quat_diff,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
    )
    fingertip_pos = ObsTerm(
        func=mdp.fingertip_pos,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_quat = ObsTerm(
        func=mdp.fingertip_quat,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_vel = ObsTerm(
        func=mdp.fingertip_vel,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Full 157-dimensional state observation in Direct order."""

    @configclass
    class PolicyCfg(FullStateWithoutActionCfg):
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset distributions matching the Direct task."""

    reset_state = EventTerm(
        func=mdp.reset_reorient_state,
        mode="reset",
        params={
            "position_noise": 0.01,
            "joint_position_noise": 0.2,
            "joint_velocity_noise": 0.0,
            "action_name": "joint_pos",
        },
    )


@configclass
class RewardsCfg:
    """Direct-compatible reward and success accounting."""

    reorient = RewTerm(
        func=mdp.ReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": -10.0,
            "rotation_scale": 1.0,
            "rotation_epsilon": 0.1,
            "action_penalty_scale": -0.0002,
            "success_tolerance": 0.1,
            "success_bonus": 250.0,
            "fall_distance": 0.24,
            "fall_penalty": 0.0,
            "averaging_factor": 0.1,
            "success_count_threshold": 1,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions matching the Direct task."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ShadowHandManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based state Shadow Hand task with Direct-compatible semantics."""

    scene: ShadowHandManagerSceneCfg = ShadowHandManagerSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation — mirrors the Direct cfg (guarded by the value-parity test)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        self.sim.physics = PhysicsCfg()
        self.viewer.eye = (2.0, 2.0, 2.0)
